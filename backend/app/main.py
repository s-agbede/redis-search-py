import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.schemas import DataOverviewResponse, SearchRequest, SearchResponse, SimilarityRequest, SimilarityResponse
from app.service import RedisVLSearchService, SearchService

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    level_name = settings.log_level.upper()
    level = getattr(logging, level_name, logging.INFO)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
    else:
        logging.getLogger().setLevel(level)


def create_app(service: SearchService | None = None) -> FastAPI:
    configure_logging()
    search_service = service or RedisVLSearchService()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        logger.info("api.lifespan.start bootstrap_on_startup=%s", settings.bootstrap_on_startup)
        if settings.bootstrap_on_startup:
            try:
                search_service.bootstrap()
            except Exception:
                logger.exception("api.bootstrap.failed")
                raise
        logger.info("api.lifespan.ready")
        yield
        logger.info("api.lifespan.shutdown")

    app = FastAPI(title="Redis Search Learning Lab API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_service() -> SearchService:
        return search_service

    @app.get("/api/health")
    def health() -> dict[str, str]:
        logger.info("api.health")
        return {"status": "ok"}

    @app.get("/api/overview", response_model=DataOverviewResponse)
    def overview(svc: SearchService = Depends(get_service)):
        logger.info("api.overview")
        response = svc.get_data_overview()
        logger.info("api.overview.done total_documents=%s", response.total_documents)
        return response

    @app.post("/api/search/text", response_model=SearchResponse)
    def text_search(payload: SearchRequest, svc: SearchService = Depends(get_service)):
        logger.info("api.search.text query=%r limit=%s genres=%s min_rating=%s", payload.query, payload.limit, payload.filters.genres, payload.filters.min_rating)
        response = svc.search_text(
            payload.query,
            payload.limit,
            payload.filters.genres,
            payload.filters.min_rating,
        )
        logger.info("api.search.text.done result_count=%s timings=%s", len(response.results), response.timings)
        return response

    @app.post("/api/search/vector", response_model=SearchResponse)
    def vector_search(payload: SearchRequest, svc: SearchService = Depends(get_service)):
        logger.info("api.search.vector query=%r limit=%s genres=%s min_rating=%s", payload.query, payload.limit, payload.filters.genres, payload.filters.min_rating)
        response = svc.search_vector(
            payload.query,
            payload.limit,
            payload.filters.genres,
            payload.filters.min_rating,
        )
        logger.info("api.search.vector.done result_count=%s timings=%s", len(response.results), response.timings)
        return response

    @app.post("/api/search/hybrid", response_model=SearchResponse)
    def hybrid_search(payload: SearchRequest, svc: SearchService = Depends(get_service)):
        logger.info(
            "api.search.hybrid query=%r limit=%s genres=%s min_rating=%s alpha=%s",
            payload.query,
            payload.limit,
            payload.filters.genres,
            payload.filters.min_rating,
            payload.hybrid.alpha,
        )
        response = svc.search_hybrid(
            payload.query,
            payload.limit,
            payload.filters.genres,
            payload.filters.min_rating,
            payload.hybrid.alpha,
        )
        logger.info("api.search.hybrid.done result_count=%s timings=%s", len(response.results), response.timings)
        return response

    @app.post("/api/search/advanced/rrf", response_model=SearchResponse)
    def advanced_rrf_search(payload: SearchRequest, svc: SearchService = Depends(get_service)):
        logger.info(
            "api.search.rrf query=%r limit=%s genres=%s min_rating=%s rrf_k=%s rrf_weights=%s",
            payload.query,
            payload.limit,
            payload.filters.genres,
            payload.filters.min_rating,
            payload.advanced.rrf_k,
            payload.advanced.rrf_weights,
        )
        response = svc.search_rrf(
            payload.query,
            payload.limit,
            payload.filters.genres,
            payload.filters.min_rating,
            payload.advanced.rrf_k,
            payload.advanced.rrf_weights,
        )
        logger.info("api.search.rrf.done result_count=%s timings=%s", len(response.results), response.timings)
        return response

    @app.post("/api/search/advanced/rerank", response_model=SearchResponse)
    def advanced_rerank_search(payload: SearchRequest, svc: SearchService = Depends(get_service)):
        logger.info(
            "api.search.rerank query=%r limit=%s genres=%s min_rating=%s rerank_top_n=%s",
            payload.query,
            payload.limit,
            payload.filters.genres,
            payload.filters.min_rating,
            payload.advanced.rerank_top_n,
        )
        response = svc.search_rerank(
            payload.query,
            payload.limit,
            payload.filters.genres,
            payload.filters.min_rating,
            payload.advanced.rerank_top_n,
        )
        logger.info("api.search.rerank.done result_count=%s timings=%s", len(response.results), response.timings)
        return response

    @app.post("/api/similarity", response_model=SimilarityResponse)
    def sentence_similarity(payload: SimilarityRequest, svc: SearchService = Depends(get_service)):
        logger.info("api.similarity sentence_a_len=%s sentence_b_len=%s", len(payload.sentence_a), len(payload.sentence_b))
        response = svc.compare_sentence_similarity(payload.sentence_a, payload.sentence_b)
        logger.info("api.similarity.done cosine=%.4f", response.cosine_similarity)
        return response

    return app


app = create_app()
