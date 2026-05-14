from __future__ import annotations

import logging
import math
import time
from typing import Any

from redisvl.index import SearchIndex
from redisvl.query import AggregateHybridQuery, TextQuery, VectorQuery
from redisvl.query.filter import Num, Tag
from redisvl.schema import IndexSchema
from redisvl.utils.vectorize import HFTextVectorizer

from app.config import settings
from app.dataset import fetch_movies_json, normalize_movies
from app.schemas import DataOverviewResponse, IndexFieldDescriptor, RetrievedRow, SearchResponse, SearchResultItem, SimilarityResponse

logger = logging.getLogger(__name__)

RETURN_FIELDS = [
    "id",
    "title",
    "year",
    "plot",
    "rating",
    "genres",
    "actors",
    "release_date",
    "rank",
    "image_url",
    "running_time_secs",
]


def fuse_rankings_rrf(
    ranked_lists: list[list[str]],
    weights: list[float] | None = None,
    k: int = 60,
) -> dict[str, float]:
    if not ranked_lists:
        return {}
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights length must match ranked_lists length")
    if k <= 0:
        raise ValueError("k must be > 0")

    scores: dict[str, float] = {}
    for list_weight, ranked in zip(weights, ranked_lists, strict=False):
        for idx, doc_id in enumerate(ranked):
            rank = idx + 1
            scores[doc_id] = scores.get(doc_id, 0.0) + (list_weight / (k + rank))
    return scores


class SearchService:
    def bootstrap(self) -> None:
        raise NotImplementedError

    def search_text(self, query: str, limit: int, genres: list[str], min_rating: float | None) -> SearchResponse:
        raise NotImplementedError

    def search_vector(self, query: str, limit: int, genres: list[str], min_rating: float | None) -> SearchResponse:
        raise NotImplementedError

    def search_hybrid(
        self,
        query: str,
        limit: int,
        genres: list[str],
        min_rating: float | None,
        alpha: float,
    ) -> SearchResponse:
        raise NotImplementedError

    def search_rrf(
        self,
        query: str,
        limit: int,
        genres: list[str],
        min_rating: float | None,
        rrf_k: int,
        rrf_weights: list[float],
    ) -> SearchResponse:
        raise NotImplementedError

    def search_rerank(
        self,
        query: str,
        limit: int,
        genres: list[str],
        min_rating: float | None,
        rerank_top_n: int,
    ) -> SearchResponse:
        raise NotImplementedError

    def get_data_overview(self) -> DataOverviewResponse:
        raise NotImplementedError

    def compare_sentence_similarity(self, sentence_a: str, sentence_b: str) -> SimilarityResponse:
        raise NotImplementedError


def _build_schema() -> IndexSchema:
    return IndexSchema.from_dict(
        {
            "index": {
                "name": settings.redis_index_name,
                "prefix": settings.redis_key_prefix,
                "storage_type": "json",
            },
            "fields": [
                {"name": "id", "type": "tag"},
                {"name": "title", "type": "text"},
                {"name": "year", "type": "numeric"},
                {"name": "plot", "type": "text"},
                {"name": "rating", "type": "numeric"},
                {"name": "genres", "type": "tag"},
                {"name": "actors", "type": "text"},
                {"name": "release_date", "type": "text"},
                {"name": "rank", "type": "numeric"},
                {"name": "image_url", "type": "text"},
                {"name": "running_time_secs", "type": "numeric"},
                {
                    "name": "plot_embedding",
                    "type": "vector",
                    "attrs": {
                        "algorithm": "flat",
                        "dims": 384,
                        "distance_metric": "cosine",
                        "datatype": "float32",
                    },
                },
            ],
        }
    )


class RedisVLSearchService(SearchService):
    def __init__(self) -> None:
        self.vectorizer: HFTextVectorizer | None = None
        self.reranker: Any | None = None
        self.index = SearchIndex(schema=_build_schema(), redis_url=settings.redis_url)
        self._bootstrapped = False

    def _ensure_bootstrapped(self) -> None:
        if not self._bootstrapped:
            logger.info("service.ensure_bootstrapped.triggered")
            self.bootstrap()

    def _get_vectorizer(self) -> HFTextVectorizer:
        if self.vectorizer is None:
            logger.info("service.vectorizer.init model=%s", settings.embedding_model)
            self.vectorizer = HFTextVectorizer(model=settings.embedding_model)
        return self.vectorizer

    def _get_reranker(self):
        if self.reranker is None:
            logger.info("service.reranker.init model=%s", settings.reranker_model)
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(settings.reranker_model)
        return self.reranker

    def bootstrap(self) -> None:
        if self._bootstrapped:
            logger.info("service.bootstrap.skip reason=already_bootstrapped")
            return

        logger.info("service.bootstrap.start dataset_source=%s index_name=%s", settings.dataset_source, settings.redis_index_name)
        start = time.perf_counter()

        vectorizer = self._get_vectorizer()
        raw = fetch_movies_json(settings.dataset_source)
        records = normalize_movies(raw)
        logger.info("service.bootstrap.dataset_loaded raw_count=%s normalized_count=%s", len(raw), len(records))

        plots = [row["plot"] for row in records]
        embeddings = vectorizer.embed_many(plots, as_buffer=False)
        for row, emb in zip(records, embeddings, strict=False):
            row["plot_embedding"] = emb

        try:
            if not self.index.exists():
                logger.info("service.bootstrap.index.create index_name=%s", settings.redis_index_name)
                self.index.create(overwrite=False, drop=False)
                self.index.load(records, id_field="id")
                logger.info("service.bootstrap.index.load count=%s", len(records))
            else:
                current_count = self._index_doc_count()
                logger.info("service.bootstrap.index.exists index_name=%s num_docs=%s", settings.redis_index_name, current_count)
                if current_count == 0:
                    logger.info("service.bootstrap.index.reload reason=empty_index count=%s", len(records))
                    self.index.load(records, id_field="id")
                    logger.info("service.bootstrap.index.reload.done")
        except Exception:
            logger.exception("service.bootstrap.failed")
            raise

        self._bootstrapped = True
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info("service.bootstrap.done elapsed_ms=%s", elapsed_ms)

    def _schema_fields(self) -> list[IndexFieldDescriptor]:
        schema_dict = self.index.schema.to_dict()
        fields = schema_dict.get("fields", [])
        return [
            IndexFieldDescriptor(
                name=str(field.get("name", "")),
                type=str(field.get("type", "")),
                attrs=field.get("attrs", {}) or {},
            )
            for field in fields
        ]

    def _parse_redis_index_info(self) -> dict[str, Any]:
        try:
            info_raw = self.index.client.execute_command("FT.INFO", settings.redis_index_name)
            if not isinstance(info_raw, list):
                return {}
            parsed: dict[str, Any] = {}
            for i in range(0, len(info_raw) - 1, 2):
                key = info_raw[i]
                value = info_raw[i + 1]
                key_str = key.decode("utf-8") if isinstance(key, (bytes, bytearray)) else str(key)
                if isinstance(value, (bytes, bytearray)):
                    parsed[key_str] = value.decode("utf-8", errors="ignore")
                else:
                    parsed[key_str] = value
            compact: dict[str, Any] = {}
            for k in ("num_docs", "num_terms", "num_records", "inverted_sz_mb", "vector_index_sz_mb"):
                if k in parsed:
                    compact[k] = parsed[k]
            return compact
        except Exception:
            logger.exception("service.overview.index_info_failed")
            return {}

    def _index_doc_count(self) -> int:
        info = self._parse_redis_index_info()
        raw = info.get("num_docs")
        try:
            return int(float(raw)) if raw is not None else 0
        except Exception:
            return 0

    def _similarity_band(self, score: float) -> str:
        if score >= 0.8:
            return "Very similar meaning"
        if score >= 0.6:
            return "Related topics with moderate semantic overlap"
        if score >= 0.35:
            return "Loosely related concepts"
        return "Different meanings"

    def _build_filter(self, genres: list[str], min_rating: float | None):
        expression = None
        if genres:
            genre_expr = None
            for genre in genres:
                current = Tag("genres") == genre
                genre_expr = current if genre_expr is None else (genre_expr | current)
            expression = genre_expr
        if min_rating is not None:
            rating_expr = Num("rating") >= float(min_rating)
            expression = rating_expr if expression is None else (expression & rating_expr)
        return expression

    def _to_item(self, row: RetrievedRow, score: float | None, explanation: str | None) -> SearchResultItem:
        return SearchResultItem(
            id=row.id,
            title=row.title,
            year=row.year,
            plot=row.plot,
            rating=row.rating,
            genres=row.genres,
            actors=row.actors,
            release_date=row.release_date,
            rank=row.rank,
            image_url=row.image_url,
            running_time_secs=row.running_time_secs,
            score=score,
            explanation=explanation,
        )

    def _coerce_rows(self, rows: list[dict[str, Any]], source: str) -> list[RetrievedRow]:
        parsed: list[RetrievedRow] = []
        for row in rows:
            try:
                parsed.append(RetrievedRow.model_validate(row))
            except Exception:
                logger.exception("service.query.row_parse_failed source=%s row_keys=%s", source, list(row.keys()))
        return parsed

    def _query_text_rows(self, query: str, limit: int, genres: list[str], min_rating: float | None) -> tuple[list[RetrievedRow], int]:
        start = time.perf_counter()
        q = TextQuery(
            text=query,
            text_field_name={"title": 1.25, "plot": 1.0},
            num_results=limit,
            filter_expression=self._build_filter(genres, min_rating),
            return_fields=RETURN_FIELDS,
            return_score=True
        )
        raw_rows = self.index.query(q)
        rows = self._coerce_rows(raw_rows, source="text")
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info("service.query.text.done query=%r limit=%s result_count=%s search_ms=%s", query, limit, len(rows), elapsed_ms)
        return rows, elapsed_ms

    def _query_vector_rows(
        self,
        query: str,
        limit: int,
        genres: list[str],
        min_rating: float | None,
    ) -> tuple[list[RetrievedRow], int, int]:
        vectorizer = self._get_vectorizer()
        embed_start = time.perf_counter()
        vector = vectorizer.embed(query)
        embed_ms = int((time.perf_counter() - embed_start) * 1000)
        search_start = time.perf_counter()
        q = VectorQuery(
            vector=vector,
            vector_field_name="plot_embedding",
            num_results=limit,
            filter_expression=self._build_filter(genres, min_rating),
            return_fields=RETURN_FIELDS,
            return_score=True,
        )
        raw_rows = self.index.query(q)
        rows = self._coerce_rows(raw_rows, source="vector")
        search_ms = int((time.perf_counter() - search_start) * 1000)
        logger.info(
            "service.query.vector.done query=%r limit=%s result_count=%s embed_ms=%s search_ms=%s",
            query,
            limit,
            len(rows),
            embed_ms,
            search_ms,
        )
        return rows, embed_ms, search_ms

    def search_text(self, query: str, limit: int, genres: list[str], min_rating: float | None) -> SearchResponse:
        self._ensure_bootstrapped()
        rows, elapsed_ms = self._query_text_rows(query, limit, genres, min_rating)
        results = [self._to_item(row, row.score, "Lexical match over title/plot terms") for row in rows]
        logger.info("service.search.text.done query=%r returned=%s", query, len(results))
        return SearchResponse(
            mode="text",
            query=query,
            results=results,
            timings={"search_ms": elapsed_ms},
            lesson_takeaway="Keyword precision is strong, but paraphrase tolerance is limited.",
        )

    def search_vector(self, query: str, limit: int, genres: list[str], min_rating: float | None) -> SearchResponse:
        self._ensure_bootstrapped()
        rows, embed_ms, search_ms = self._query_vector_rows(query, limit, genres, min_rating)
        results = [self._to_item(row, row.vector_distance, "Semantic similarity over plot meaning") for row in rows]
        logger.info("service.search.vector.done query=%r returned=%s", query, len(results))
        return SearchResponse(
            mode="vector",
            query=query,
            results=results,
            timings={"embed_ms": embed_ms, "search_ms": search_ms},
            lesson_takeaway="Intent matching improves significantly compared with strict keyword search.",
        )

    def search_hybrid(
        self,
        query: str,
        limit: int,
        genres: list[str],
        min_rating: float | None,
        alpha: float,
    ) -> SearchResponse:
        self._ensure_bootstrapped()
        vectorizer = self._get_vectorizer()
        embed_start = time.perf_counter()
        vector = vectorizer.embed(query)
        embed_ms = int((time.perf_counter() - embed_start) * 1000)

        search_start = time.perf_counter()
        q = AggregateHybridQuery(
            text=query,
            text_field_name="plot",
            vector=vector,
            vector_field_name="plot_embedding",
            filter_expression=self._build_filter(genres, min_rating),
            alpha=alpha,
            num_results=limit,
            return_fields=RETURN_FIELDS,
            stopwords=None,
        )
        raw_rows = self.index.query(q)
        rows = self._coerce_rows(raw_rows, source="hybrid")
        search_ms = int((time.perf_counter() - search_start) * 1000)

        results = [self._to_item(row, row.hybrid_score, "Combined lexical and semantic relevance") for row in rows]
        logger.info("service.search.hybrid.done query=%r alpha=%s returned=%s", query, alpha, len(results))
        return SearchResponse(
            mode="hybrid",
            query=query,
            results=results,
            timings={"embed_ms": embed_ms, "search_ms": search_ms},
            lesson_takeaway="Hybrid retrieval balances exact terms and semantic intent.",
        )

    def search_rrf(
        self,
        query: str,
        limit: int,
        genres: list[str],
        min_rating: float | None,
        rrf_k: int,
        rrf_weights: list[float],
    ) -> SearchResponse:
        self._ensure_bootstrapped()
        retrieval_limit = max(limit * 3, 20)
        text_rows, text_ms = self._query_text_rows(query, retrieval_limit, genres, min_rating)
        vector_rows, embed_ms, vector_ms = self._query_vector_rows(query, retrieval_limit, genres, min_rating)

        text_ids = [r.id for r in text_rows]
        vector_ids = [r.id for r in vector_rows]
        rrf_scores = fuse_rankings_rrf([text_ids, vector_ids], weights=rrf_weights, k=rrf_k)

        lookup: dict[str, RetrievedRow] = {}
        for row in text_rows + vector_rows:
            lookup[row.id] = row

        ranked_ids = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        results = [
            self._to_item(lookup[doc_id], score, "Reciprocal Rank Fusion across text and vector lists")
            for doc_id, score in ranked_ids
            if doc_id in lookup
        ]
        logger.info(
            "service.search.rrf.done query=%r rrf_k=%s weights=%s returned=%s",
            query,
            rrf_k,
            rrf_weights,
            len(results),
        )
        return SearchResponse(
            mode="rrf",
            query=query,
            results=results,
            timings={
                "embed_ms": embed_ms,
                "text_search_ms": text_ms,
                "vector_search_ms": vector_ms,
                "search_ms": text_ms + vector_ms,
            },
            lesson_takeaway="RRF can stabilize ranking quality when score scales differ between methods.",
        )

    def search_rerank(
        self,
        query: str,
        limit: int,
        genres: list[str],
        min_rating: float | None,
        rerank_top_n: int,
    ) -> SearchResponse:
        self._ensure_bootstrapped()
        text_rows, text_ms = self._query_text_rows(query, rerank_top_n, genres, min_rating)
        vector_rows, embed_ms, vector_ms = self._query_vector_rows(query, rerank_top_n, genres, min_rating)
        merged: dict[str, RetrievedRow] = {}
        for row in text_rows + vector_rows:
            merged[row.id] = row
        candidates = list(merged.values())[:rerank_top_n]
        if not candidates:
            logger.info("service.search.rerank.empty query=%r", query)
            return SearchResponse(mode="rerank", query=query, results=[], timings={"search_ms": text_ms + vector_ms})

        rerank_start = time.perf_counter()
        model = self._get_reranker()
        pairs = [(query, f"{row.title}. {row.plot}") for row in candidates]
        scores = model.predict(pairs)
        rerank_ms = int((time.perf_counter() - rerank_start) * 1000)

        scored_rows = sorted(zip(candidates, scores, strict=False), key=lambda it: float(it[1]), reverse=True)[:limit]
        results = [
            self._to_item(row, float(score), "Cross-encoder reranked candidate set from text+vector retrieval")
            for row, score in scored_rows
        ]
        logger.info("service.search.rerank.done query=%r rerank_top_n=%s returned=%s", query, rerank_top_n, len(results))
        return SearchResponse(
            mode="rerank",
            query=query,
            results=results,
            timings={
                "embed_ms": embed_ms,
                "text_search_ms": text_ms,
                "vector_search_ms": vector_ms,
                "rerank_ms": rerank_ms,
                "search_ms": text_ms + vector_ms + rerank_ms,
            },
            lesson_takeaway="Cross-encoder reranking improves ordering quality but adds model latency.",
        )

    def get_data_overview(self) -> DataOverviewResponse:
        self._ensure_bootstrapped()
        raw = fetch_movies_json(settings.dataset_source)
        normalized = normalize_movies(raw)
        response = DataOverviewResponse(
            dataset_source=settings.dataset_source,
            total_documents=len(normalized),
            raw_shape_example=raw[0] if raw else {},
            normalized_shape_example=normalized[0] if normalized else {},
            index_name=settings.redis_index_name,
            index_fields=self._schema_fields(),
            redis_index_info=self._parse_redis_index_info(),
        )
        logger.info("service.overview.done total_documents=%s index_fields=%s", response.total_documents, len(response.index_fields))
        return response

    def compare_sentence_similarity(self, sentence_a: str, sentence_b: str) -> SimilarityResponse:
        vectorizer = self._get_vectorizer()
        vec_a = vectorizer.embed(sentence_a)
        vec_b = vectorizer.embed(sentence_b)
        dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        similarity = 0.0 if norm_a == 0 or norm_b == 0 else dot / (norm_a * norm_b)
        response = SimilarityResponse(
            sentence_a=sentence_a,
            sentence_b=sentence_b,
            cosine_similarity=float(similarity),
            interpretation=self._similarity_band(float(similarity)),
            embedding_preview={
                "sentence_a_first8": [float(v) for v in vec_a[:8]],
                "sentence_b_first8": [float(v) for v in vec_b[:8]],
            },
        )
        logger.info("service.similarity.done cosine=%.4f interpretation=%r", response.cosine_similarity, response.interpretation)
        return response
