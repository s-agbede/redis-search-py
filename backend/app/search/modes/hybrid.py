"""Hybrid retrieval helpers."""

from __future__ import annotations

import logging
import time

from redisvl.index import SearchIndex
from redisvl.query import AggregateHybridQuery
from redisvl.utils.vectorize import HFTextVectorizer

from app.search.constants import RETURN_FIELDS
from app.search.filters import build_filter_expression
from app.search.mappers import coerce_rows
from app.search.modes.semantic import embed_query
from app.schemas import RetrievedRow

logger = logging.getLogger(__name__)


def build_hybrid_query(
    query: str,
    vector,
    limit: int,
    genres: list[str],
    min_rating: float | None,
    alpha: float,
) -> AggregateHybridQuery:
    # Hybrid queries blend text relevance with vector similarity in one Redis call.
    return AggregateHybridQuery(
        text=query,
        text_field_name="plot",
        vector=vector,
        vector_field_name="plot_embedding",
        filter_expression=build_filter_expression(genres, min_rating),
        alpha=alpha,
        num_results=limit,
        return_fields=RETURN_FIELDS,
        stopwords=None,
    )


def query_hybrid_rows(
    index: SearchIndex,
    vectorizer: HFTextVectorizer,
    query: str,
    limit: int,
    genres: list[str],
    min_rating: float | None,
    alpha: float,
) -> tuple[list[RetrievedRow], int, int]:
    # Embed once up front, then spend the remaining time in Redis.
    vector, embed_ms = embed_query(vectorizer, query)

    search_start = time.perf_counter()
    raw_rows = index.query(build_hybrid_query(query, vector, limit, genres, min_rating, alpha))
    rows = coerce_rows(raw_rows, source="hybrid", logger=logger)

    search_ms = int((time.perf_counter() - search_start) * 1000)
    logger.info(
        "service.query.hybrid.done query=%r alpha=%s result_count=%s embed_ms=%s search_ms=%s",
        query,
        alpha,
        len(rows),
        embed_ms,
        search_ms,
    )

    return rows, embed_ms, search_ms


__all__ = ["build_hybrid_query", "query_hybrid_rows"]
