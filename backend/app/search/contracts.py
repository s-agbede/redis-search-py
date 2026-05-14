"""Shared search-service contract."""

from __future__ import annotations

from app.schemas import DataOverviewResponse, SearchResponse, SimilarityResponse


class SearchService:
    def bootstrap(self) -> None:
        """Prepare the backing search system for use."""
        raise NotImplementedError

    def search_text(self, query: str, limit: int, genres: list[str], min_rating: float | None) -> SearchResponse:
        """Run lexical search."""
        raise NotImplementedError

    def search_vector(self, query: str, limit: int, genres: list[str], min_rating: float | None) -> SearchResponse:
        """Run vector similarity search."""
        raise NotImplementedError

    def search_hybrid(
        self,
        query: str,
        limit: int,
        genres: list[str],
        min_rating: float | None,
        alpha: float,
    ) -> SearchResponse:
        """Run hybrid lexical and vector search."""
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
        """Run reciprocal-rank-fusion search."""
        raise NotImplementedError

    def search_rerank(
        self,
        query: str,
        limit: int,
        genres: list[str],
        min_rating: float | None,
        rerank_top_n: int,
    ) -> SearchResponse:
        """Run candidate retrieval followed by reranking."""
        raise NotImplementedError

    def get_data_overview(self) -> DataOverviewResponse:
        """Describe the indexed dataset and schema."""
        raise NotImplementedError

    def compare_sentence_similarity(self, sentence_a: str, sentence_b: str) -> SimilarityResponse:
        """Compare two sentences in embedding space."""
        raise NotImplementedError


__all__ = ["SearchService"]
