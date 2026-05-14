# Search Architecture Diagram

This page gives you a presentation-ready architecture visual for the Redis Search Learning Lab.

## Diagram (Image)

![Redis Search Architecture](./assets/search-architecture.svg)

The SVG above is the presentation-accurate version. Use the Mermaid block below as the conceptual/editable fallback.

## Mermaid Version (Editable)

```mermaid
flowchart LR
    U["1. React UI<br/>User query + filters"] --> A["2. FastAPI API Layer<br/>/api/search/text | /vector | /hybrid | /advanced/rrf | /advanced/rerank"]
    A --> Q{"3. Mode Router"}

    Q --> FT["Full-Text Path<br/>TextQuery<br/>title: 1.25, plot: 1.0"]
    Q --> VS["Semantic Path<br/>Embed Query (MiniLM)<br/>VectorQuery on plot_embedding"]
    Q --> HY["Hybrid/RRF Path<br/>AggregateHybridQuery (alpha)<br/>or RRF fusion"]

    FT --> R["4. Redis Index<br/>title, plot, genres, rating, plot_embedding"]
    VS --> R
    HY --> R

    R --> M["5. Result Mapper<br/>score, explanation, timings"]
    M --> U
```

## Code Touchpoints

- Full-text mode: [backend/app/search/modes/full_text.py](../backend/app/search/modes/full_text.py)
- Semantic mode: [backend/app/search/modes/semantic.py](../backend/app/search/modes/semantic.py)
- Hybrid mode: [backend/app/search/modes/hybrid.py](../backend/app/search/modes/hybrid.py)
- RRF/service orchestration: [backend/app/search/redis_service.py](../backend/app/search/redis_service.py)
- API routes: [backend/app/main.py](../backend/app/main.py)
- Frontend request wiring: [frontend/src/api.ts](../frontend/src/api.ts)
