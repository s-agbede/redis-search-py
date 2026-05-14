"""Bootstrap helpers for the Redis-backed search service."""

from __future__ import annotations

import logging
import time
from typing import Any

from redisvl.index import SearchIndex
from redisvl.utils.vectorize import HFTextVectorizer

from app.config import settings
from app.dataset import fetch_movies_json, normalize_movies

logger = logging.getLogger(__name__)


def parse_redis_index_info(index: SearchIndex) -> dict[str, Any]:
    try:
        info_raw = index.client.execute_command("FT.INFO", settings.redis_index_name)
        if not isinstance(info_raw, list):
            return {}

        # Redis returns FT.INFO as alternating key/value entries.
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
        for key in ("num_docs", "num_terms", "num_records", "inverted_sz_mb", "vector_index_sz_mb"):
            if key in parsed:
                compact[key] = parsed[key]

        return compact
    except Exception:
        logger.exception("service.overview.index_info_failed")
        return {}


def get_index_doc_count(index: SearchIndex) -> int:
    info = parse_redis_index_info(index)
    raw = info.get("num_docs")
    try:
        return int(float(raw)) if raw is not None else 0
    except Exception:
        return 0


def bootstrap_index(index: SearchIndex, vectorizer: HFTextVectorizer) -> None:
    logger.info(
        "service.bootstrap.start dataset_source=%s index_name=%s",
        settings.dataset_source,
        settings.redis_index_name,
    )
    start = time.perf_counter()

    raw = fetch_movies_json(settings.dataset_source)
    records = normalize_movies(raw)
    logger.info("service.bootstrap.dataset_loaded raw_count=%s normalized_count=%s", len(raw), len(records))

    # Precompute embeddings once so every record can be loaded with its vector field.
    plots = [row["plot"] for row in records]
    embeddings = vectorizer.embed_many(plots, as_buffer=False)
    for row, emb in zip(records, embeddings, strict=False):
        row["plot_embedding"] = emb

    try:
        # Create and populate a brand-new index, or backfill an existing empty one.
        if not index.exists():
            logger.info("service.bootstrap.index.create index_name=%s", settings.redis_index_name)
            index.create(overwrite=False, drop=False)

            index.load(records, id_field="id")
            logger.info("service.bootstrap.index.load count=%s", len(records))
        else:
            current_count = get_index_doc_count(index)
            logger.info("service.bootstrap.index.exists index_name=%s num_docs=%s", settings.redis_index_name, current_count)

            if current_count == 0:
                logger.info("service.bootstrap.index.reload reason=empty_index count=%s", len(records))
                index.load(records, id_field="id")
                logger.info("service.bootstrap.index.reload.done")
    except Exception:
        logger.exception("service.bootstrap.failed")
        raise

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    logger.info("service.bootstrap.done elapsed_ms=%s", elapsed_ms)


__all__ = ["bootstrap_index", "get_index_doc_count", "parse_redis_index_info"]
