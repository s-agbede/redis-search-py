from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas import DataOverviewResponse, SearchResponse, SimilarityResponse
from app.service import SearchService


class FakeService(SearchService):
    def bootstrap(self) -> None:
        return None

    def search_text(self, query: str, limit: int, genres: list[str], min_rating: float | None) -> SearchResponse:
        return SearchResponse(
            mode="text",
            query=query,
            results=[],
            timings={"search_ms": 1},
            lesson_takeaway="text",
        )

    def search_vector(self, query: str, limit: int, genres: list[str], min_rating: float | None) -> SearchResponse:
        return SearchResponse(
            mode="vector",
            query=query,
            results=[],
            timings={"embed_ms": 1, "search_ms": 2},
        )

    def search_hybrid(
        self,
        query: str,
        limit: int,
        genres: list[str],
        min_rating: float | None,
        alpha: float,
    ) -> SearchResponse:
        return SearchResponse(
            mode="hybrid",
            query=query,
            results=[],
            timings={"embed_ms": 1, "search_ms": 3},
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
        return SearchResponse(mode="rrf", query=query, results=[], timings={"search_ms": 4})

    def search_rerank(
        self,
        query: str,
        limit: int,
        genres: list[str],
        min_rating: float | None,
        rerank_top_n: int,
    ) -> SearchResponse:
        return SearchResponse(mode="rerank", query=query, results=[], timings={"search_ms": 5})

    def get_data_overview(self) -> DataOverviewResponse:
        return DataOverviewResponse(
            dataset_source="data/movies.json",
            total_documents=1,
            raw_shape_example={"title": "Example", "year": 1999, "info": {"plot": "plot"}},
            normalized_shape_example={"id": "abc", "title": "Example", "year": 1999, "plot": "plot"},
            index_name="idx:movies",
            index_fields=[],
            redis_index_info={"num_docs": "1"},
        )

    def compare_sentence_similarity(self, sentence_a: str, sentence_b: str) -> SimilarityResponse:
        return SimilarityResponse(
            sentence_a=sentence_a,
            sentence_b=sentence_b,
            cosine_similarity=0.77,
            interpretation="Related topics with moderate semantic overlap",
            embedding_preview={"sentence_a_first8": [0.1], "sentence_b_first8": [0.2]},
        )


def test_api_contract_for_text_search():
    app = create_app(FakeService())
    client = TestClient(app)

    res = client.post(
        "/api/search/text",
        json={"query": "star", "limit": 5, "filters": {"genres": ["Adventure"], "min_rating": 7.0}},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["mode"] == "text"
    assert body["query"] == "star"
    assert "results" in body
    assert "timings" in body


def test_api_rejects_empty_query():
    app = create_app(FakeService())
    client = TestClient(app)

    res = client.post("/api/search/vector", json={"query": " ", "limit": 5})
    assert res.status_code == 422


def test_overview_endpoint():
    app = create_app(FakeService())
    client = TestClient(app)

    res = client.get("/api/overview")
    assert res.status_code == 200
    body = res.json()
    assert body["dataset_source"] == "data/movies.json"
    assert body["total_documents"] == 1


def test_advanced_rrf_endpoint():
    app = create_app(FakeService())
    client = TestClient(app)

    res = client.post(
        "/api/search/advanced/rrf",
        json={"query": "spy thriller", "limit": 5, "advanced": {"rrf_k": 80, "rrf_weights": [0.6, 0.4]}},
    )
    assert res.status_code == 200
    assert res.json()["mode"] == "rrf"


def test_advanced_rerank_endpoint():
    app = create_app(FakeService())
    client = TestClient(app)

    res = client.post(
        "/api/search/advanced/rerank",
        json={"query": "spy thriller", "limit": 5, "advanced": {"rerank_top_n": 25}},
    )
    assert res.status_code == 200
    assert res.json()["mode"] == "rerank"


def test_similarity_endpoint():
    app = create_app(FakeService())
    client = TestClient(app)

    res = client.post(
        "/api/similarity",
        json={"sentence_a": "astronaut lands on moon", "sentence_b": "space mission reaches lunar surface"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["cosine_similarity"] == 0.77
    assert "embedding_preview" in body
