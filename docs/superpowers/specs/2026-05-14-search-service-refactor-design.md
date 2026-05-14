# Search Service Refactor Design

## Goal

Refactor `backend/app/service.py` into a small public facade plus focused internal modules so the code is easier to teach, maintain, and extend without changing the current API behavior or the current import path (`from app.service import ...`).

## Current Problem

`backend/app/service.py` currently mixes several responsibilities in one file:

- public service contract
- Redis index schema construction
- bootstrap and indexing logic
- reusable filter and row-mapping helpers
- full-text retrieval
- semantic retrieval
- weighted hybrid retrieval
- advanced ranking strategies such as RRF and reranking
- sentence similarity logic

That makes the file harder to scan and increases the chance that small changes in one area create confusion in another.

## Non-Goals

- Changing API routes or response schemas
- Changing request validation behavior
- Changing Redis schema fields or index settings
- Changing retrieval scoring behavior on purpose
- Reworking the frontend or docs beyond reference-path updates needed by the refactor

## Design Summary

Keep `backend/app/service.py` as the stable import surface and move the implementation into a new internal package, `backend/app/search/`.

`service.py` will become a compatibility facade that re-exports:

- `SearchService`
- `RedisVLSearchService`
- `fuse_rankings_rrf`

The new package will separate reusable infrastructure from mode-specific behavior so each file has one clear purpose.

## Proposed Package Layout

```text
backend/app/
  service.py                  # public compatibility facade
  search/
    __init__.py               # package exports used by service.py
    advanced.py               # RRF fusion and reranking workflows
    bootstrap.py              # dataset loading, embedding generation, index loading
    constants.py              # shared return fields and similar constants
    contracts.py              # abstract SearchService interface
    filters.py                # reusable Redis filter-expression builder
    full_text.py              # TextQuery execution helpers
    hybrid.py                 # AggregateHybridQuery execution helpers
    mappers.py                # row parsing and SearchResultItem mapping helpers
    redis_service.py          # RedisVLSearchService orchestration class
    schema.py                 # Redis index schema builder
    semantic.py               # vector query + sentence similarity helpers
```

## Responsibility Boundaries

### `service.py`

Purpose:
Provide a stable module path for the rest of the app and any teaching material that still imports from `app.service`.

Rules:
- no business logic beyond re-exporting public symbols
- add a short module comment explaining that the file exists for backward-compatible imports

### `contracts.py`

Purpose:
Define the abstract service interface used by the API layer.

Contents:
- `SearchService`

### `constants.py`

Purpose:
Hold shared values used across multiple search modes.

Contents:
- `RETURN_FIELDS`

### `schema.py`

Purpose:
Construct the RedisVL `IndexSchema` in one place.

Contents:
- `_build_schema()` or a renamed public helper such as `build_index_schema()`

### `filters.py`

Purpose:
Build reusable Redis filter expressions from request filters.

Contents:
- helper for combining genre tags with optional minimum rating

### `mappers.py`

Purpose:
Convert raw Redis rows into validated internal rows and then into API result items.

Contents:
- row coercion helper
- `SearchResultItem` mapping helper

### `bootstrap.py`

Purpose:
Own dataset fetch, normalization, embedding generation, and initial index population.

Contents:
- helper that loads movie data
- helper that attaches `plot_embedding`
- helper that creates or reloads the Redis index when needed

### `full_text.py`

Purpose:
Own lexical retrieval logic only.

Contents:
- `TextQuery` creation
- raw row execution and timing capture

### `semantic.py`

Purpose:
Own vector-based retrieval and sentence similarity helpers.

Contents:
- query embedding helper
- `VectorQuery` execution and timing capture
- cosine similarity helper and interpretation helper

### `hybrid.py`

Purpose:
Own the weighted hybrid retrieval path only.

Contents:
- `AggregateHybridQuery` construction
- execution and timing capture

### `advanced.py`

Purpose:
Own advanced ranking workflows that combine lower-level retrieval steps.

Contents:
- `fuse_rankings_rrf`
- RRF search workflow
- reranking workflow using the cross-encoder

### `redis_service.py`

Purpose:
Provide the concrete orchestration class that wires the helpers together while keeping the public method surface unchanged.

Contents:
- `RedisVLSearchService`
- lazy vectorizer and reranker initialization
- overview helpers
- public methods used by the FastAPI layer

## Why This Split

This structure follows the way the project is taught:

- full-text search is isolated in one file
- semantic search is isolated in one file
- hybrid and advanced ranking logic are isolated in their own files
- reusable pieces are pulled into helpers instead of being hidden in a large class

It also keeps the API layer simple because `main.py` can continue depending on the same `SearchService` contract and `RedisVLSearchService` implementation.

## Data Flow After Refactor

### Bootstrap Flow

1. `RedisVLSearchService.bootstrap()` delegates to `bootstrap.py`.
2. The bootstrap helper fetches and normalizes movie data.
3. The vectorizer generates plot embeddings.
4. The schema helper builds the Redis index schema.
5. The index is created or reloaded if needed.

### Text Search Flow

1. API route calls `svc.search_text(...)`.
2. `RedisVLSearchService` ensures bootstrap has happened.
3. `full_text.py` builds and runs `TextQuery`.
4. `mappers.py` validates raw rows into `RetrievedRow`.
5. Results are converted into `SearchResultItem` objects.

### Semantic Search Flow

1. API route calls `svc.search_vector(...)`.
2. Query text is embedded in `semantic.py`.
3. `VectorQuery` runs with shared filters.
4. Rows are validated and mapped to response items.

### Hybrid / Advanced Flows

1. Weighted hybrid delegates to `hybrid.py`.
2. RRF delegates to `advanced.py`, which reuses full-text and semantic retrieval helpers.
3. Rerank delegates to `advanced.py`, which reuses retrieval helpers and then applies the cross-encoder.

## Comments Strategy

Add concise comments only where they improve comprehension, especially:

- in `service.py`, explaining the compatibility facade
- in helper modules where RedisVL setup is not self-evident
- around RRF and reranking, where the ranking strategy may not be obvious to readers
- around bootstrap/index-reload behavior, where the control flow has hidden operational context

Avoid low-value comments that restate straightforward code.

## Import and Dependency Rules

To keep the package easy to reason about:

- mode modules should not import each other unless one truly composes another
- `advanced.py` may depend on shared retrieval helpers from `full_text.py` and `semantic.py`
- `redis_service.py` may depend on all helper modules
- `service.py` should only import from `app.search`

## Backward Compatibility

The refactor must preserve:

- `from app.service import SearchService`
- `from app.service import RedisVLSearchService`
- current FastAPI endpoint behavior
- current request and response payload shapes
- current logging behavior, except for small wording changes if needed

## Documentation Impact

The teaching docs currently link directly into `backend/app/service.py`.

After the refactor:

- update “Read the Code” links in the search-basics docs to point at the new focused modules
- keep the examples accurate to the new module boundaries
- avoid rewriting the conceptual teaching content unless a path reference becomes stale

## Testing Strategy

Run the existing backend tests that cover:

- API behavior
- dataset normalization
- RRF scoring

If the refactor introduces any import or packaging risk, add a small compatibility test that confirms public imports still work from `app.service`.

## Risks

### Import Cycles

Splitting helpers too aggressively can create circular imports. Keep contracts, constants, and pure helpers light, and keep orchestration in `redis_service.py`.

### Behavior Drift

Moving query code across files can accidentally change timing capture, score fields, or explanations. Reuse current logic as directly as possible.

### Doc Staleness

The docs currently point to line numbers in `service.py`, so they will become outdated unless updated alongside the refactor.

## Implementation Order

1. Create `app/search/` and move the abstract contract and shared helpers first.
2. Move mode-specific query logic into focused modules.
3. Rebuild `RedisVLSearchService` in `redis_service.py` using those helpers.
4. Replace `app/service.py` with a thin compatibility facade.
5. Update docs and any tests affected by import-path or code-reference changes.
6. Run backend tests.

## Success Criteria

- `backend/app/service.py` becomes small and easy to understand
- search behavior remains unchanged from the API consumer’s perspective
- code for full-text, semantic, hybrid, and advanced search lives in separate focused files
- shared helpers are reusable and clearly named
- comments are present where they improve readability
- existing imports from `app.service` continue to work
