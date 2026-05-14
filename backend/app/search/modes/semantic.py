"""Semantic retrieval and sentence-similarity helpers."""

from __future__ import annotations

import logging
import math
import time
from typing import Any

from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.utils.vectorize import HFTextVectorizer

from app.search.constants import RETURN_FIELDS
from app.search.filters import build_filter_expression
from app.search.mappers import coerce_rows
from app.schemas import RetrievedRow, SimilarityResponse

logger = logging.getLogger(__name__)


def embed_query(vectorizer: HFTextVectorizer, query: str) -> tuple[Any, int]:
    embed_start = time.perf_counter()
    vector = vectorizer.embed(query)
    embed_ms = int((time.perf_counter() - embed_start) * 1000)

    return vector, embed_ms


def build_vector_query(
    vector: Any,
    limit: int,
    genres: list[str],
    min_rating: float | None,
) -> VectorQuery:
    # The vector field search still respects the same filters and return payload.
    return VectorQuery(
        vector=vector,
        vector_field_name="plot_embedding",
        num_results=limit,
        filter_expression=build_filter_expression(genres, min_rating),
        return_fields=RETURN_FIELDS,
        return_score=True,
    )


def query_vector_rows(
    index: SearchIndex,
    vectorizer: HFTextVectorizer,
    query: str,
    limit: int,
    genres: list[str],
    min_rating: float | None,
) -> tuple[list[RetrievedRow], int, int]:
    vector, embed_ms = embed_query(vectorizer, query)

    search_start = time.perf_counter()
    raw_rows = index.query(build_vector_query(vector, limit, genres, min_rating))
    rows = coerce_rows(raw_rows, source="vector", logger=logger)

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


def similarity_band(score: float) -> str:
    if score >= 0.8:
        return "Very similar meaning"

    if score >= 0.6:
        return "Related topics with moderate semantic overlap"

    if score >= 0.35:
        return "Loosely related concepts"

    return "Different meanings"


def compare_sentence_similarity(
    vectorizer: HFTextVectorizer,
    sentence_a: str,
    sentence_b: str,
) -> SimilarityResponse:
    vec_a = vectorizer.embed(sentence_a)
    vec_b = vectorizer.embed(sentence_b)

    # Compute cosine similarity directly so the response can expose the math inputs.
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    similarity = 0.0 if norm_a == 0 or norm_b == 0 else dot / (norm_a * norm_b)

    response = SimilarityResponse(
        sentence_a=sentence_a,
        sentence_b=sentence_b,
        cosine_similarity=float(similarity),
        interpretation=similarity_band(float(similarity)),
        embedding_preview={
            "sentence_a_first8": [float(v) for v in vec_a[:8]],
            "sentence_b_first8": [float(v) for v in vec_b[:8]],
        },
    )

    logger.info("service.similarity.done cosine=%.4f interpretation=%r", response.cosine_similarity, response.interpretation)

    return response


__all__ = [
    "build_vector_query",
    "compare_sentence_similarity",
    "embed_query",
    "query_vector_rows",
    "similarity_band",
]
