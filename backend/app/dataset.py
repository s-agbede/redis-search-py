import hashlib
import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)


def stable_movie_id(title: str, year: int | None, rank: int | None, release_date: str | None) -> str:
    token = f"{title}|{year}|{rank}|{release_date}"
    return hashlib.sha1(token.encode("utf-8")).hexdigest()[:16]


def normalize_movies(raw_movies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    logger.info("dataset.normalize.start raw_count=%s", len(raw_movies))
    normalized: list[dict[str, Any]] = []
    for item in raw_movies:
        info = item.get("info", {}) or {}
        title = str(item.get("title", "")).strip()
        year = item.get("year")
        plot = str(info.get("plot", "")).strip()
        rating = info.get("rating")
        genres = info.get("genres") or []
        actors = info.get("actors") or []
        release_date = info.get("release_date")
        rank = info.get("rank")
        image_url = info.get("image_url")
        running_time_secs = info.get("running_time_secs")

        if not title or not plot:
            continue

        normalized.append(
            {
                "id": stable_movie_id(title, year, rank, release_date),
                "title": title,
                "year": int(year) if year is not None else 0,
                "plot": plot,
                "rating": float(rating) if rating is not None else None,
                "genres": [str(g) for g in genres],
                "actors": [str(a) for a in actors],
                "release_date": release_date,
                "rank": int(rank) if rank is not None else None,
                "image_url": image_url,
                "running_time_secs": int(running_time_secs) if running_time_secs is not None else None,
            }
        )
    logger.info("dataset.normalize.done normalized_count=%s", len(normalized))
    return normalized


def _is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"}


def fetch_movies_json(dataset_source: str, timeout: float = 30.0) -> list[dict[str, Any]]:
    logger.info("dataset.fetch.start source=%s", dataset_source)
    if _is_url(dataset_source):
        logger.info("dataset.fetch.mode=http")
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(dataset_source)
            response.raise_for_status()
        payload = response.json()
    else:
        logger.info("dataset.fetch.mode=file")
        path = Path(dataset_source)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / dataset_source
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, list):
        raise ValueError("Dataset payload must be a list of movie objects")
    logger.info("dataset.fetch.done records=%s", len(payload))
    return payload
