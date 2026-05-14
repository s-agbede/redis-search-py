from app.schemas import SimilarityResponse
from app.search.redis_service import RedisVLSearchService


class StubVectorizer:
    def embed(self, text: str) -> list[float]:
        return {
            "astronaut lands on moon": [1.0, 0.0],
            "space mission reaches lunar surface": [0.5, 0.5],
        }[text]


def test_compare_sentence_similarity_does_not_require_bootstrap(monkeypatch):
    service = RedisVLSearchService()

    def fail_if_bootstrap_runs() -> None:
        raise AssertionError("bootstrap should not run for sentence similarity")

    monkeypatch.setattr(service, "_ensure_bootstrapped", fail_if_bootstrap_runs)
    monkeypatch.setattr(service, "_get_vectorizer", lambda: StubVectorizer())

    result = service.compare_sentence_similarity(
        "astronaut lands on moon",
        "space mission reaches lunar surface",
    )

    assert isinstance(result, SimilarityResponse)
    assert result.sentence_a == "astronaut lands on moon"
    assert result.sentence_b == "space mission reaches lunar surface"
