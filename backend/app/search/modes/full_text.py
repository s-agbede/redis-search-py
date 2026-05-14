"""Full-text retrieval helpers."""

from __future__ import annotations

import logging
import time

from redisvl.index import SearchIndex
from redisvl.query import TextQuery

from app.search.constants import RETURN_FIELDS
from app.search.filters import build_filter_expression
from app.search.mappers import coerce_rows
from app.schemas import RetrievedRow

logger = logging.getLogger(__name__)


def build_text_query(
    query: str,
    limit: int,
    genres: list[str],
    min_rating: float | None,
) -> TextQuery:
    # Boost title matches slightly above plot matches for keyword-heavy queries.
    return TextQuery(
        text=query,
        text_field_name={"title": 1.25, "plot": 1.0},
        num_results=limit,
        filter_expression=build_filter_expression(genres, min_rating),
        return_fields=RETURN_FIELDS,
        return_score=True,
    )


def query_text_rows(
    index: SearchIndex,
    query: str,
    limit: int,
    genres: list[str],
    min_rating: float | None,
) -> tuple[list[RetrievedRow], int]:
    start = time.perf_counter()

    raw_rows = index.query(build_text_query(query, limit, genres, min_rating))
    rows = coerce_rows(raw_rows, source="text", logger=logger)

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    logger.info("service.query.text.done query=%r limit=%s result_count=%s search_ms=%s", query, limit, len(rows), elapsed_ms)

    return rows, elapsed_ms


__all__ = ["build_text_query", "query_text_rows"]
