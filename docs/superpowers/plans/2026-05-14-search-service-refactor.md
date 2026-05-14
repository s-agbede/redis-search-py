# Search Service Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `backend/app/service.py` into focused internal modules while preserving the `app.service` public import path and current API behavior.

**Architecture:** Introduce a new `backend/app/search/` package for contracts, shared helpers, and mode-specific search logic. Keep `backend/app/service.py` as a thin compatibility facade that re-exports the same public symbols used by the rest of the backend and tests.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic, RedisVL, pytest

---

## File Map

### New files

- `backend/app/search/__init__.py`: package-level exports used by `app.service`
- `backend/app/search/contracts.py`: abstract `SearchService` contract
- `backend/app/search/constants.py`: shared return field list
- `backend/app/search/schema.py`: RedisVL index schema builder
- `backend/app/search/filters.py`: reusable Redis filter-expression builder
- `backend/app/search/mappers.py`: Redis row parsing and API item conversion
- `backend/app/search/bootstrap.py`: dataset loading, embeddings, and index population
- `backend/app/search/full_text.py`: `TextQuery` execution helper
- `backend/app/search/semantic.py`: vector retrieval and sentence similarity helpers
- `backend/app/search/hybrid.py`: weighted hybrid retrieval helper
- `backend/app/search/advanced.py`: RRF fusion and reranking helpers
- `backend/app/search/redis_service.py`: concrete `RedisVLSearchService`
- `backend/tests/test_service_public_imports.py`: compatibility tests for `app.service`

### Modified files

- `backend/app/service.py`: compatibility facade only
- `docs/full-text-search-basics.md`: point “Read the Code” links at the new full-text modules
- `docs/semantic-search-basics.md`: point “Read the Code” links at the new semantic modules
- `docs/hybrid-search-basics.md`: point “Read the Code” links at the new hybrid/advanced modules

### Existing tests to run

- `backend/tests/test_service_public_imports.py`
- `backend/tests/test_rrf.py`
- `backend/tests/test_api.py`
- `backend/tests/test_dataset.py`

### Task 1: Add Compatibility Test First

**Files:**
- Create: `backend/tests/test_service_public_imports.py`
- Test: `backend/tests/test_service_public_imports.py`

- [ ] **Step 1: Write the failing test**

```python
from app.search.advanced import fuse_rankings_rrf as internal_rrf
from app.search.contracts import SearchService as InternalSearchService
from app.search.redis_service import RedisVLSearchService as InternalRedisVLSearchService
from app.service import RedisVLSearchService, SearchService, fuse_rankings_rrf


def test_service_module_reexports_public_symbols():
    assert SearchService is InternalSearchService
    assert RedisVLSearchService is InternalRedisVLSearchService
    assert fuse_rankings_rrf is internal_rrf
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest -q backend/tests/test_service_public_imports.py`

Expected: FAIL with an import error because `app.search` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create placeholder package files so the test can import the new module paths during the next iteration:

```python
# backend/app/search/__init__.py

from app.search.advanced import fuse_rankings_rrf
from app.search.contracts import SearchService
from app.search.redis_service import RedisVLSearchService

__all__ = ["SearchService", "RedisVLSearchService", "fuse_rankings_rrf"]
```

```python
# backend/app/search/contracts.py

from __future__ import annotations

from app.schemas import DataOverviewResponse, SearchResponse, SimilarityResponse


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
```

```python
# backend/app/search/advanced.py

from __future__ import annotations


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
```

```python
# backend/app/search/redis_service.py

from __future__ import annotations

from app.search.contracts import SearchService


class RedisVLSearchService(SearchService):
    pass
```

- [ ] **Step 4: Run test to verify it passes or fails for the next useful reason**

Run: `uv run pytest -q backend/tests/test_service_public_imports.py`

Expected: either PASS immediately or FAIL because `app.service` still contains the old implementation instead of the re-export facade. Either result is useful before Task 2.

- [ ] **Step 5: Checkpoint**

Run: `uv run pytest -q backend/tests/test_service_public_imports.py backend/tests/test_rrf.py`

Expected: the new compatibility test should be active and `test_rrf.py` should still pass once the RRF helper is preserved.

### Task 2: Extract Shared Contracts and Helper Modules

**Files:**
- Create: `backend/app/search/constants.py`
- Create: `backend/app/search/schema.py`
- Create: `backend/app/search/filters.py`
- Create: `backend/app/search/mappers.py`
- Modify: `backend/app/search/contracts.py`
- Test: `backend/tests/test_service_public_imports.py`

- [ ] **Step 1: Write a focused failing assertion for the shared helper layer**

Extend `backend/tests/test_service_public_imports.py` with:

```python
from app.search.constants import RETURN_FIELDS


def test_return_fields_keep_expected_api_shape():
    assert RETURN_FIELDS == [
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest -q backend/tests/test_service_public_imports.py::test_return_fields_keep_expected_api_shape`

Expected: FAIL with `ModuleNotFoundError` for `app.search.constants`.

- [ ] **Step 3: Write minimal implementation**

Add the shared modules with comments at responsibility boundaries:

```python
# backend/app/search/constants.py

"""Shared Redis return field definitions used across search modes."""

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
```

```python
# backend/app/search/schema.py

from __future__ import annotations

from redisvl.schema import IndexSchema

from app.config import settings


def build_schema() -> IndexSchema:
    """Build the single Redis index schema used by the learning lab."""
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
```

```python
# backend/app/search/filters.py

from __future__ import annotations

from redisvl.query.filter import Num, Tag


def build_filter_expression(genres: list[str], min_rating: float | None):
    """Compose RedisVL filter expressions shared by every search mode."""
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
```

```python
# backend/app/search/mappers.py

from __future__ import annotations

import logging
from typing import Any

from app.schemas import RetrievedRow, SearchResultItem

logger = logging.getLogger(__name__)


def coerce_rows(rows: list[dict[str, Any]], source: str) -> list[RetrievedRow]:
    """Validate Redis rows before they are transformed into API results."""
    parsed: list[RetrievedRow] = []
    for row in rows:
        try:
            parsed.append(RetrievedRow.model_validate(row))
        except Exception:
            logger.exception("service.query.row_parse_failed source=%s row_keys=%s", source, list(row.keys()))
    return parsed


def to_item(row: RetrievedRow, score: float | None, explanation: str | None) -> SearchResultItem:
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
```

- [ ] **Step 4: Run tests to verify the shared layer is wired correctly**

Run: `uv run pytest -q backend/tests/test_service_public_imports.py backend/tests/test_rrf.py`

Expected: PASS for the `RETURN_FIELDS` assertion and existing RRF validation behavior.

- [ ] **Step 5: Checkpoint**

If git metadata becomes available, commit:

```bash
git add backend/app/search/constants.py backend/app/search/schema.py backend/app/search/filters.py backend/app/search/mappers.py backend/app/search/contracts.py backend/tests/test_service_public_imports.py
git commit -m "refactor: extract shared search helpers"
```

If git is still unavailable in this workspace, record that the checkpoint was verified without a commit.

### Task 3: Extract Full-Text, Semantic, Hybrid, and Advanced Modules

**Files:**
- Create: `backend/app/search/full_text.py`
- Create: `backend/app/search/semantic.py`
- Create: `backend/app/search/hybrid.py`
- Modify: `backend/app/search/advanced.py`
- Test: `backend/tests/test_rrf.py`

- [ ] **Step 1: Write the failing behavioral test for the advanced helper**

Extend `backend/tests/test_rrf.py` with:

```python
from app.search.advanced import fuse_rankings_rrf


def test_fuse_rankings_rrf_returns_empty_scores_for_no_lists():
    assert fuse_rankings_rrf([]) == {}
```

- [ ] **Step 2: Run test to verify it fails or proves existing coverage is already green**

Run: `uv run pytest -q backend/tests/test_rrf.py::test_fuse_rankings_rrf_returns_empty_scores_for_no_lists`

Expected: PASS if the helper already preserves this behavior, which confirms the extracted advanced module is safe to keep using. If it fails, fix the helper before proceeding.

- [ ] **Step 3: Write the minimal implementation**

Create the focused retrieval modules by moving the existing query logic out of `service.py` with only naming cleanups and comments:

```python
# backend/app/search/full_text.py

from __future__ import annotations

import logging
import time

from redisvl.query import TextQuery
from redisvl.index import SearchIndex

from app.search.constants import RETURN_FIELDS
from app.search.filters import build_filter_expression
from app.search.mappers import coerce_rows
from app.schemas import RetrievedRow

logger = logging.getLogger(__name__)


def query_text_rows(
    index: SearchIndex,
    query: str,
    limit: int,
    genres: list[str],
    min_rating: float | None,
) -> tuple[list[RetrievedRow], int]:
    """Run lexical retrieval with the same field weighting used before the refactor."""
    start = time.perf_counter()
    q = TextQuery(
        text=query,
        text_field_name={"title": 1.25, "plot": 1.0},
        num_results=limit,
        filter_expression=build_filter_expression(genres, min_rating),
        return_fields=RETURN_FIELDS,
        return_score=True,
    )
    rows = coerce_rows(index.query(q), source="text")
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    logger.info("service.query.text.done query=%r limit=%s result_count=%s search_ms=%s", query, limit, len(rows), elapsed_ms)
    return rows, elapsed_ms
```

```python
# backend/app/search/semantic.py

from __future__ import annotations

import logging
import math
import time

from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.utils.vectorize import HFTextVectorizer

from app.search.constants import RETURN_FIELDS
from app.search.filters import build_filter_expression
from app.search.mappers import coerce_rows
from app.schemas import RetrievedRow, SimilarityResponse

logger = logging.getLogger(__name__)


def query_vector_rows(
    index: SearchIndex,
    vectorizer: HFTextVectorizer,
    query: str,
    limit: int,
    genres: list[str],
    min_rating: float | None,
) -> tuple[list[RetrievedRow], int, int]:
    embed_start = time.perf_counter()
    vector = vectorizer.embed(query)
    embed_ms = int((time.perf_counter() - embed_start) * 1000)
    search_start = time.perf_counter()
    q = VectorQuery(
        vector=vector,
        vector_field_name="plot_embedding",
        num_results=limit,
        filter_expression=build_filter_expression(genres, min_rating),
        return_fields=RETURN_FIELDS,
        return_score=True,
    )
    rows = coerce_rows(index.query(q), source="vector")
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
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    similarity = 0.0 if norm_a == 0 or norm_b == 0 else dot / (norm_a * norm_b)
    return SimilarityResponse(
        sentence_a=sentence_a,
        sentence_b=sentence_b,
        cosine_similarity=float(similarity),
        interpretation=similarity_band(float(similarity)),
        embedding_preview={
            "sentence_a_first8": [float(v) for v in vec_a[:8]],
            "sentence_b_first8": [float(v) for v in vec_b[:8]],
        },
    )
```

```python
# backend/app/search/hybrid.py

from __future__ import annotations

import time

from redisvl.index import SearchIndex
from redisvl.query import AggregateHybridQuery
from redisvl.utils.vectorize import HFTextVectorizer

from app.search.constants import RETURN_FIELDS
from app.search.filters import build_filter_expression
from app.search.mappers import coerce_rows
from app.schemas import RetrievedRow


def query_hybrid_rows(
    index: SearchIndex,
    vectorizer: HFTextVectorizer,
    query: str,
    limit: int,
    genres: list[str],
    min_rating: float | None,
    alpha: float,
) -> tuple[list[RetrievedRow], int, int]:
    """Blend lexical and semantic retrieval using Redis' weighted hybrid query."""
    embed_start = time.perf_counter()
    vector = vectorizer.embed(query)
    embed_ms = int((time.perf_counter() - embed_start) * 1000)
    search_start = time.perf_counter()
    q = AggregateHybridQuery(
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
    rows = coerce_rows(index.query(q), source="hybrid")
    search_ms = int((time.perf_counter() - search_start) * 1000)
    return rows, embed_ms, search_ms
```

```python
# backend/app/search/advanced.py

from __future__ import annotations

import time
from typing import Any

from redisvl.index import SearchIndex
from redisvl.utils.vectorize import HFTextVectorizer

from app.search.full_text import query_text_rows
from app.search.semantic import query_vector_rows
from app.schemas import RetrievedRow


def fuse_rankings_rrf(
    ranked_lists: list[list[str]],
    weights: list[float] | None = None,
    k: int = 60,
) -> dict[str, float]:
    """Fuse independently ranked result lists without assuming score scales match."""
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


def collect_rrf_candidates(
    index: SearchIndex,
    vectorizer: HFTextVectorizer,
    query: str,
    retrieval_limit: int,
    genres: list[str],
    min_rating: float | None,
) -> tuple[list[RetrievedRow], list[RetrievedRow], int, int, int]:
    text_rows, text_ms = query_text_rows(index, query, retrieval_limit, genres, min_rating)
    vector_rows, embed_ms, vector_ms = query_vector_rows(index, vectorizer, query, retrieval_limit, genres, min_rating)
    return text_rows, vector_rows, embed_ms, text_ms, vector_ms


def rerank_candidates(
    reranker: Any,
    query: str,
    candidates: list[RetrievedRow],
) -> tuple[list[tuple[RetrievedRow, float]], int]:
    rerank_start = time.perf_counter()
    pairs = [(query, f"{row.title}. {row.plot}") for row in candidates]
    scores = reranker.predict(pairs)
    rerank_ms = int((time.perf_counter() - rerank_start) * 1000)
    ranked = sorted(zip(candidates, scores, strict=False), key=lambda item: float(item[1]), reverse=True)
    return [(row, float(score)) for row, score in ranked], rerank_ms
```

- [ ] **Step 4: Run focused tests**

Run: `uv run pytest -q backend/tests/test_rrf.py`

Expected: PASS. The extracted `advanced.py` must preserve the current RRF behavior exactly.

- [ ] **Step 5: Checkpoint**

If git metadata becomes available, commit:

```bash
git add backend/app/search/full_text.py backend/app/search/semantic.py backend/app/search/hybrid.py backend/app/search/advanced.py backend/tests/test_rrf.py
git commit -m "refactor: split search modes into focused modules"
```

If git is still unavailable, record the verification results and continue.

### Task 4: Rebuild the Concrete Service and Replace `app.service` with a Facade

**Files:**
- Create: `backend/app/search/bootstrap.py`
- Create: `backend/app/search/redis_service.py`
- Modify: `backend/app/search/__init__.py`
- Modify: `backend/app/service.py`
- Test: `backend/tests/test_api.py`
- Test: `backend/tests/test_service_public_imports.py`

- [ ] **Step 1: Write the failing compatibility and API test run**

Run: `uv run pytest -q backend/tests/test_service_public_imports.py backend/tests/test_api.py`

Expected: FAIL until `app.service` is reduced to the re-export facade and `RedisVLSearchService` regains the full implementation.

- [ ] **Step 2: Verify the failure reason**

Confirm that the failure is caused by missing concrete behavior in `backend/app/search/redis_service.py` or by `backend/app/service.py` still owning the old implementation, not by unrelated schema/test breakage.

- [ ] **Step 3: Write minimal implementation**

Move the orchestration logic into `backend/app/search/redis_service.py`, keep comments at non-obvious boundaries, and replace `backend/app/service.py` with a compatibility module:

```python
# backend/app/search/bootstrap.py

from __future__ import annotations

import logging

from redisvl.index import SearchIndex
from redisvl.utils.vectorize import HFTextVectorizer

from app.config import settings
from app.dataset import fetch_movies_json, normalize_movies

logger = logging.getLogger(__name__)


def bootstrap_index(index: SearchIndex, vectorizer: HFTextVectorizer) -> None:
    """Load the movie dataset and backfill embeddings if the index is missing or empty."""
    raw = fetch_movies_json(settings.dataset_source)
    records = normalize_movies(raw)
    logger.info("service.bootstrap.dataset_loaded raw_count=%s normalized_count=%s", len(raw), len(records))
    plots = [row["plot"] for row in records]
    embeddings = vectorizer.embed_many(plots, as_buffer=False)
    for row, emb in zip(records, embeddings, strict=False):
        row["plot_embedding"] = emb

    if not index.exists():
        logger.info("service.bootstrap.index.create index_name=%s", settings.redis_index_name)
        index.create(overwrite=False, drop=False)
        index.load(records, id_field="id")
        logger.info("service.bootstrap.index.load count=%s", len(records))
        return

    info_raw = index.client.execute_command("FT.INFO", settings.redis_index_name)
    info_map = {}
    for i in range(0, len(info_raw) - 1, 2):
        key = info_raw[i]
        value = info_raw[i + 1]
        key_str = key.decode("utf-8") if isinstance(key, (bytes, bytearray)) else str(key)
        info_map[key_str] = value.decode("utf-8", errors="ignore") if isinstance(value, (bytes, bytearray)) else value
    current_count = int(float(info_map.get("num_docs", 0)))
    logger.info("service.bootstrap.index.exists index_name=%s num_docs=%s", settings.redis_index_name, current_count)
    if current_count == 0:
        logger.info("service.bootstrap.index.reload reason=empty_index count=%s", len(records))
        index.load(records, id_field="id")
        logger.info("service.bootstrap.index.reload.done")
```

```python
# backend/app/search/redis_service.py

from __future__ import annotations

import logging
import time
from typing import Any

from redisvl.index import SearchIndex
from redisvl.utils.vectorize import HFTextVectorizer

from app.config import settings
from app.dataset import fetch_movies_json, normalize_movies
from app.schemas import DataOverviewResponse, IndexFieldDescriptor, SearchResponse
from app.search.advanced import collect_rrf_candidates, fuse_rankings_rrf, rerank_candidates
from app.search.bootstrap import bootstrap_index
from app.search.contracts import SearchService
from app.search.hybrid import query_hybrid_rows
from app.search.mappers import to_item
from app.search.schema import build_schema
from app.search.semantic import compare_sentence_similarity, query_vector_rows
from app.search.full_text import query_text_rows

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
        bootstrap_index(self.index, self._get_vectorizer())
        self._bootstrapped = True
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info("service.bootstrap.done elapsed_ms=%s", elapsed_ms)

    def _schema_fields(self) -> list[IndexFieldDescriptor]:
        schema_dict = self.index.schema.to_dict()
        fields = schema_dict.get("fields", [])
        return [
            IndexFieldDescriptor(name=str(field.get("name", "")), type=str(field.get("type", "")), attrs=field.get("attrs", {}) or {})
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
                parsed[key_str] = value.decode("utf-8", errors="ignore") if isinstance(value, (bytes, bytearray)) else value
            return {k: parsed[k] for k in ("num_docs", "num_terms", "num_records", "inverted_sz_mb", "vector_index_sz_mb") if k in parsed}
        except Exception:
            logger.exception("service.overview.index_info_failed")
            return {}

    def search_text(self, query: str, limit: int, genres: list[str], min_rating: float | None) -> SearchResponse:
        self._ensure_bootstrapped()
        rows, elapsed_ms = query_text_rows(self.index, query, limit, genres, min_rating)
        return SearchResponse(
            mode="text",
            query=query,
            results=[to_item(row, row.score, "Lexical match over title/plot terms") for row in rows],
            timings={"search_ms": elapsed_ms},
            lesson_takeaway="Keyword precision is strong, but paraphrase tolerance is limited.",
        )

    def search_vector(self, query: str, limit: int, genres: list[str], min_rating: float | None) -> SearchResponse:
        self._ensure_bootstrapped()
        rows, embed_ms, search_ms = query_vector_rows(self.index, self._get_vectorizer(), query, limit, genres, min_rating)
        return SearchResponse(
            mode="vector",
            query=query,
            results=[to_item(row, row.vector_distance, "Semantic similarity over plot meaning") for row in rows],
            timings={"embed_ms": embed_ms, "search_ms": search_ms},
            lesson_takeaway="Intent matching improves significantly compared with strict keyword search.",
        )
```

```python
# backend/app/search/redis_service.py (continue the same file)

    def search_hybrid(
        self,
        query: str,
        limit: int,
        genres: list[str],
        min_rating: float | None,
        alpha: float,
    ) -> SearchResponse:
        self._ensure_bootstrapped()
        rows, embed_ms, search_ms = query_hybrid_rows(self.index, self._get_vectorizer(), query, limit, genres, min_rating, alpha)
        return SearchResponse(
            mode="hybrid",
            query=query,
            results=[to_item(row, row.hybrid_score, "Combined lexical and semantic relevance") for row in rows],
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
        text_rows, vector_rows, embed_ms, text_ms, vector_ms = collect_rrf_candidates(
            self.index,
            self._get_vectorizer(),
            query,
            retrieval_limit,
            genres,
            min_rating,
        )
        lookup = {row.id: row for row in text_rows + vector_rows}
        ranked_ids = sorted(
            fuse_rankings_rrf(
                [[row.id for row in text_rows], [row.id for row in vector_rows]],
                weights=rrf_weights,
                k=rrf_k,
            ).items(),
            key=lambda item: item[1],
            reverse=True,
        )[:limit]
        return SearchResponse(
            mode="rrf",
            query=query,
            results=[
                to_item(lookup[doc_id], score, "Reciprocal Rank Fusion across text and vector lists")
                for doc_id, score in ranked_ids
                if doc_id in lookup
            ],
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
        text_rows, vector_rows, embed_ms, text_ms, vector_ms = collect_rrf_candidates(
            self.index,
            self._get_vectorizer(),
            query,
            rerank_top_n,
            genres,
            min_rating,
        )
        merged = {row.id: row for row in text_rows + vector_rows}
        candidates = list(merged.values())[:rerank_top_n]
        if not candidates:
            return SearchResponse(mode="rerank", query=query, results=[], timings={"search_ms": text_ms + vector_ms})
        scored_rows, rerank_ms = rerank_candidates(self._get_reranker(), query, candidates)
        return SearchResponse(
            mode="rerank",
            query=query,
            results=[
                to_item(row, score, "Cross-encoder reranked candidate set from text+vector retrieval")
                for row, score in scored_rows[:limit]
            ],
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
        return DataOverviewResponse(
            dataset_source=settings.dataset_source,
            total_documents=len(normalized),
            raw_shape_example=raw[0] if raw else {},
            normalized_shape_example=normalized[0] if normalized else {},
            index_name=settings.redis_index_name,
            index_fields=self._schema_fields(),
            redis_index_info=self._parse_redis_index_info(),
        )

    def compare_sentence_similarity(self, sentence_a: str, sentence_b: str):
        return compare_sentence_similarity(self._get_vectorizer(), sentence_a, sentence_b)
```

```python
# backend/app/service.py

"""Backward-compatible public search service imports."""

from app.search import RedisVLSearchService, SearchService, fuse_rankings_rrf

__all__ = ["SearchService", "RedisVLSearchService", "fuse_rankings_rrf"]
```

- [ ] **Step 4: Run tests to verify the facade preserves behavior**

Run: `uv run pytest -q backend/tests/test_service_public_imports.py backend/tests/test_api.py`

Expected: PASS. The FastAPI tests should keep using `SearchService` from `app.service` without modification.

- [ ] **Step 5: Checkpoint**

If git metadata becomes available, commit:

```bash
git add backend/app/search/bootstrap.py backend/app/search/redis_service.py backend/app/search/__init__.py backend/app/service.py backend/tests/test_service_public_imports.py backend/tests/test_api.py
git commit -m "refactor: move search service behind facade"
```

If git is still unavailable, record that the service and API compatibility tests passed without a commit.

### Task 5: Update Teaching Docs and Run Full Verification

**Files:**
- Modify: `docs/full-text-search-basics.md`
- Modify: `docs/semantic-search-basics.md`
- Modify: `docs/hybrid-search-basics.md`
- Test: `backend/tests/test_service_public_imports.py`
- Test: `backend/tests/test_rrf.py`
- Test: `backend/tests/test_api.py`
- Test: `backend/tests/test_dataset.py`

- [ ] **Step 1: Write the failing verification target**

Run the full backend test suite before updating the docs:

Run: `uv run pytest -q backend/tests/test_service_public_imports.py backend/tests/test_rrf.py backend/tests/test_api.py backend/tests/test_dataset.py`

Expected: PASS. If anything fails here, fix code before touching docs.

- [ ] **Step 2: Verify the codebase now points at stale service file references**

Use:

```bash
rg -n "backend/app/service.py|_query_text_rows|search_vector|search_hybrid|fuse_rankings_rrf" docs/full-text-search-basics.md docs/semantic-search-basics.md docs/hybrid-search-basics.md
```

Expected: matches that still reference the old monolithic file.

- [ ] **Step 3: Write the minimal documentation updates**

Adjust the “Read the Code” sections so they point at the new modules:

```markdown
<!-- docs/full-text-search-basics.md -->
- Backend text query builder and execution:
  - [`query_text_rows`](../backend/app/search/full_text.py)
  - [`RedisVLSearchService.search_text`](../backend/app/search/redis_service.py)
```

```markdown
<!-- docs/semantic-search-basics.md -->
- Backend vectorization and vector query:
  - [`query_vector_rows`](../backend/app/search/semantic.py)
  - [`RedisVLSearchService.search_vector`](../backend/app/search/redis_service.py)
  - [`compare_sentence_similarity`](../backend/app/search/semantic.py)
```

```markdown
<!-- docs/hybrid-search-basics.md -->
- Weighted hybrid implementation:
  - [`query_hybrid_rows`](../backend/app/search/hybrid.py)
  - [`RedisVLSearchService.search_hybrid`](../backend/app/search/redis_service.py)
- RRF implementation:
  - [`fuse_rankings_rrf`](../backend/app/search/advanced.py)
  - [`RedisVLSearchService.search_rrf`](../backend/app/search/redis_service.py)
```

- [ ] **Step 4: Run full verification again**

Run: `uv run pytest -q backend/tests/test_service_public_imports.py backend/tests/test_rrf.py backend/tests/test_api.py backend/tests/test_dataset.py`

Expected: PASS with no API regressions.

- [ ] **Step 5: Final checkpoint**

If git metadata becomes available, commit:

```bash
git add docs/full-text-search-basics.md docs/semantic-search-basics.md docs/hybrid-search-basics.md
git commit -m "docs: update search module references"
```

If git is still unavailable, capture the successful verification commands in the final handoff notes.

## Self-Review

- Spec coverage: the plan covers the compatibility facade, helper extraction, mode-specific modules, docs updates, and verification.
- Placeholder scan: all tasks use exact file paths, test targets, and concrete module content.
- Type consistency: the plan preserves the existing public names `SearchService`, `RedisVLSearchService`, and `fuse_rankings_rrf`, and the new helper names are used consistently across tasks.
