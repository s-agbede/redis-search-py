"""Advanced ranking helpers for fused and reranked retrieval."""

from __future__ import annotations

import logging
import time
from typing import Any

from redisvl.index import SearchIndex
from redisvl.utils.vectorize import HFTextVectorizer

from app.search.modes.full_text import query_text_rows
from app.search.modes.semantic import query_vector_rows
from app.schemas import RetrievedRow

logger = logging.getLogger(__name__)


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

    # Accumulate one fused score per document across all ranked source lists.
    scores: dict[str, float] = {}
    for list_weight, ranked in zip(weights, ranked_lists, strict=False):
        for idx, doc_id in enumerate(ranked):
            rank = idx + 1
            scores[doc_id] = scores.get(doc_id, 0.0) + (list_weight / (k + rank))

    return scores


def collect_rrf_candidates(
    index: SearchIndex,
    vectorizer: HFTextVectorizer,
    query: str,
    retrieval_limit: int,
    genres: list[str],
    min_rating: float | None,
) -> tuple[list[RetrievedRow], list[RetrievedRow], int, int, int]:
    # Keep lexical and vector result sets separate so fusion can use each ranking directly.
    text_rows, text_ms = query_text_rows(index, query, retrieval_limit, genres, min_rating)
    vector_rows, embed_ms, vector_ms = query_vector_rows(index, vectorizer, query, retrieval_limit, genres, min_rating)

    return text_rows, vector_rows, embed_ms, text_ms, vector_ms


def build_row_lookup(*row_groups: list[RetrievedRow]) -> dict[str, RetrievedRow]:
    lookup: dict[str, RetrievedRow] = {}

    for rows in row_groups:
        for row in rows:
            lookup[row.id] = row

    return lookup


def rank_fused_results(rrf_scores: dict[str, float], limit: int) -> list[tuple[str, float]]:
    return sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)[:limit]


def collect_rerank_candidates(
    index: SearchIndex,
    vectorizer: HFTextVectorizer,
    query: str,
    rerank_top_n: int,
    genres: list[str],
    min_rating: float | None,
) -> tuple[list[RetrievedRow], int, int, int]:
    text_rows, vector_rows, embed_ms, text_ms, vector_ms = collect_rrf_candidates(
        index=index,
        vectorizer=vectorizer,
        query=query,
        retrieval_limit=rerank_top_n,
        genres=genres,
        min_rating=min_rating,
    )

    # Deduplicate overlapping documents before handing the candidate pool to the reranker.
    merged = build_row_lookup(text_rows, vector_rows)

    return list(merged.values())[:rerank_top_n], embed_ms, text_ms, vector_ms


def rerank_candidates(
    reranker: Any,
    query: str,
    candidates: list[RetrievedRow],
) -> tuple[list[tuple[RetrievedRow, float]], int]:
    rerank_start = time.perf_counter()

    # Cross-encoders score query/document pairs rather than raw embeddings.
    pairs = [(query, f"{row.title}. {row.plot}") for row in candidates]
    scores = reranker.predict(pairs)
    rerank_ms = int((time.perf_counter() - rerank_start) * 1000)

    ranked = sorted(zip(candidates, scores, strict=False), key=lambda item: float(item[1]), reverse=True)

    return [(row, float(score)) for row, score in ranked], rerank_ms


__all__ = [
    "build_row_lookup",
    "collect_rerank_candidates",
    "collect_rrf_candidates",
    "fuse_rankings_rrf",
    "rank_fused_results",
    "rerank_candidates",
]
