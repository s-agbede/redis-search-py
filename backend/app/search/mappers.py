"""Helpers for coercing Redis rows into API-facing models."""

from __future__ import annotations

import logging
from typing import Any

from app.schemas import RetrievedRow, SearchResultItem


def coerce_rows(
    rows: list[dict[str, Any]],
    source: str,
    *,
    logger: logging.Logger | None = None,
) -> list[RetrievedRow]:
    bound_logger = logger or logging.getLogger(__name__)
    parsed: list[RetrievedRow] = []

    for row in rows:
        try:
            parsed.append(RetrievedRow.model_validate(row))
        except Exception:
            # Drop malformed rows so one bad payload does not fail the whole response.
            bound_logger.exception("service.query.row_parse_failed source=%s row_keys=%s", source, list(row.keys()))

    return parsed


def to_search_result_item(
    row: RetrievedRow,
    score: float | None,
    explanation: str | None,
) -> SearchResultItem:
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


__all__ = ["coerce_rows", "to_search_result_item"]
