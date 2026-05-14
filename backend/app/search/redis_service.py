from __future__ import annotations

import logging
from typing import Any

from redisvl.index import SearchIndex
from redisvl.utils.vectorize import HFTextVectorizer

from app.config import settings
from app.dataset import fetch_movies_json, normalize_movies
from app.schemas import DataOverviewResponse, IndexFieldDescriptor, SearchResponse, SimilarityResponse
from app.search.modes.advanced import (
    build_row_lookup,
    collect_rerank_candidates,
    collect_rrf_candidates,
    fuse_rankings_rrf,
    rank_fused_results,
    rerank_candidates,
)
from app.search.bootstrap import bootstrap_index, parse_redis_index_info
from app.search.contracts import SearchService
from app.search.mappers import to_search_result_item
from app.search.modes.full_text import query_text_rows
from app.search.modes.hybrid import query_hybrid_rows
from app.search.modes.semantic import compare_sentence_similarity, query_vector_rows
from app.search.schema import build_schema

logger = logging.getLogger(__name__)


class RedisVLSearchService(SearchService):
    def __init__(self) -> None:
        self.vectorizer: HFTextVectorizer | None = None
        self.reranker: Any | None = None
        self.index = SearchIndex(schema=build_schema(), redis_url=settings.redis_url)
        self._bootstrapped = False

    def _ensure_bootstrapped(self) -> None:
        if not self._bootstrapped:
            logger.info("service.ensure_bootstrapped.triggered")
            self.bootstrap()

    def _get_vectorizer(self) -> HFTextVectorizer:
        if self.vectorizer is None:
            # Delay model startup until the first endpoint actually needs embeddings.
            logger.info("service.vectorizer.init model=%s", settings.embedding_model)
            self.vectorizer = HFTextVectorizer(model=settings.embedding_model)

        return self.vectorizer

    def _get_reranker(self):
        if self.reranker is None:
            # Import and load the cross-encoder lazily because it is the heaviest search path.
            logger.info("service.reranker.init model=%s", settings.reranker_model)
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(settings.reranker_model)

        return self.reranker

    def bootstrap(self) -> None:
        if self._bootstrapped:
            logger.info("service.bootstrap.skip reason=already_bootstrapped")
            return

        bootstrap_index(self.index, self._get_vectorizer())
        self._bootstrapped = True

    def _schema_fields(self) -> list[IndexFieldDescriptor]:
        schema_dict = self.index.schema.to_dict()
        fields = schema_dict.get("fields", [])

        # Normalize raw schema metadata into the API response model.
        return [
            IndexFieldDescriptor(
                name=str(field.get("name", "")),
                type=str(field.get("type", "")),
                attrs=field.get("attrs", {}) or {},
            )
            for field in fields
        ]

    def search_text(self, query: str, limit: int, genres: list[str], min_rating: float | None) -> SearchResponse:
        self._ensure_bootstrapped()

        rows, elapsed_ms = query_text_rows(self.index, query, limit, genres, min_rating)
        results = [to_search_result_item(row, row.score, "Lexical match over title/plot terms") for row in rows]

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

        rows, embed_ms, search_ms = query_vector_rows(self.index, self._get_vectorizer(), query, limit, genres, min_rating)
        results = [to_search_result_item(row, row.vector_distance, "Semantic similarity over plot meaning") for row in rows]

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

        rows, embed_ms, search_ms = query_hybrid_rows(
            self.index,
            self._get_vectorizer(),
            query,
            limit,
            genres,
            min_rating,
            alpha,
        )
        results = [to_search_result_item(row, row.hybrid_score, "Combined lexical and semantic relevance") for row in rows]

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

        # Pull a larger candidate pool from each retrieval mode before fusing their ranks.
        retrieval_limit = max(limit * 3, 20)
        text_rows, vector_rows, embed_ms, text_ms, vector_ms = collect_rrf_candidates(
            index=self.index,
            vectorizer=self._get_vectorizer(),
            query=query,
            retrieval_limit=retrieval_limit,
            genres=genres,
            min_rating=min_rating,
        )
        rrf_scores = fuse_rankings_rrf(
            [[row.id for row in text_rows], [row.id for row in vector_rows]],
            weights=rrf_weights,
            k=rrf_k,
        )
        lookup = build_row_lookup(text_rows, vector_rows)

        results = [
            to_search_result_item(lookup[doc_id], score, "Reciprocal Rank Fusion across text and vector lists")
            for doc_id, score in rank_fused_results(rrf_scores, limit)
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

        candidates, embed_ms, text_ms, vector_ms = collect_rerank_candidates(
            index=self.index,
            vectorizer=self._get_vectorizer(),
            query=query,
            rerank_top_n=rerank_top_n,
            genres=genres,
            min_rating=min_rating,
        )
        if not candidates:
            logger.info("service.search.rerank.empty query=%r", query)
            return SearchResponse(mode="rerank", query=query, results=[], timings={"search_ms": text_ms + vector_ms})

        # Rerank only the shared candidate pool instead of re-running retrieval.
        scored_rows, rerank_ms = rerank_candidates(self._get_reranker(), query, candidates)
        results = [
            to_search_result_item(row, score, "Cross-encoder reranked candidate set from text+vector retrieval")
            for row, score in scored_rows[:limit]
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
            redis_index_info=parse_redis_index_info(self.index),
        )

        logger.info("service.overview.done total_documents=%s index_fields=%s", response.total_documents, len(response.index_fields))

        return response

    def compare_sentence_similarity(self, sentence_a: str, sentence_b: str) -> SimilarityResponse:
        return compare_sentence_similarity(self._get_vectorizer(), sentence_a, sentence_b)


__all__ = ["RedisVLSearchService"]
