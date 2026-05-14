import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


SearchMode = Literal["text", "vector", "hybrid", "rrf", "rerank"]


class SearchFilters(BaseModel):
    genres: list[str] = Field(default_factory=list)
    min_rating: float | None = None


class HybridOptions(BaseModel):
    alpha: float = 0.7

    @field_validator("alpha")
    @classmethod
    def validate_alpha(cls, value: float) -> float:
        if value < 0 or value > 1:
            raise ValueError("alpha must be between 0 and 1")
        return value


class AdvancedOptions(BaseModel):
    rrf_k: int = Field(default=60, ge=1, le=500)
    rrf_weights: list[float] = Field(default_factory=lambda: [0.5, 0.5])
    rerank_top_n: int = Field(default=20, ge=2, le=100)

    @field_validator("rrf_weights")
    @classmethod
    def validate_weights(cls, value: list[float]) -> list[float]:
        if len(value) != 2:
            raise ValueError("rrf_weights must contain exactly two weights: [text_weight, vector_weight]")
        if value[0] < 0 or value[1] < 0:
            raise ValueError("rrf_weights values must be >= 0")
        total = value[0] + value[1]
        if total == 0:
            raise ValueError("rrf_weights must not sum to 0")
        return [value[0] / total, value[1] / total]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=8, ge=1, le=30)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    hybrid: HybridOptions = Field(default_factory=HybridOptions)
    advanced: AdvancedOptions = Field(default_factory=AdvancedOptions)

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        clean = value.strip()
        if not clean:
            raise ValueError("query must not be empty")
        return clean


class SearchResultItem(BaseModel):
    id: str
    title: str
    year: int
    plot: str
    rating: float | None = None
    genres: list[str] = Field(default_factory=list)
    actors: list[str] = Field(default_factory=list)
    release_date: str | None = None
    rank: int | None = None
    image_url: str | None = None
    running_time_secs: int | None = None
    score: float | None = None
    explanation: str | None = None


class RetrievedRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = ""
    title: str = ""
    year: int = 0
    plot: str = ""
    rating: float | None = None
    genres: list[str] = Field(default_factory=list)
    actors: list[str] = Field(default_factory=list)
    release_date: str | None = None
    rank: int | None = None
    image_url: str | None = None
    running_time_secs: int | None = None
    score: float | None = None
    vector_distance: float | None = None
    hybrid_score: float | None = None

    @field_validator("id", "title", "plot", mode="before")
    @classmethod
    def coerce_required_text(cls, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (bytes, bytearray)):
            return value.decode("utf-8", errors="ignore")
        return str(value)

    @field_validator("release_date", "image_url", mode="before")
    @classmethod
    def coerce_optional_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, (bytes, bytearray)):
            return value.decode("utf-8", errors="ignore")
        text = str(value)
        return text if text else None

    @field_validator("year", mode="before")
    @classmethod
    def coerce_year(cls, value: Any) -> int:
        if value is None or value == "":
            return 0
        return int(value)

    @field_validator("rating", mode="before")
    @classmethod
    def coerce_rating(cls, value: Any) -> float | None:
        if value is None or value == "":
            return None
        return float(value)

    @field_validator("rank", "running_time_secs", mode="before")
    @classmethod
    def coerce_optional_int(cls, value: Any) -> int | None:
        if value is None or value == "":
            return None
        return int(value)

    @field_validator("score", "vector_distance", "hybrid_score", mode="before")
    @classmethod
    def coerce_optional_float(cls, value: Any) -> float | None:
        if value is None or value == "":
            return None
        return float(value)

    @field_validator("genres", "actors", mode="before")
    @classmethod
    def coerce_string_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v) for v in value]
        if isinstance(value, (bytes, bytearray)):
            value = value.decode("utf-8", errors="ignore")
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed]
            except Exception:
                pass
            return [part.strip() for part in value.split(",") if part.strip()]
        return [str(value)]


class SearchResponse(BaseModel):
    mode: SearchMode
    query: str
    results: list[SearchResultItem]
    timings: dict[str, Any]
    lesson_takeaway: str | None = None


class IndexFieldDescriptor(BaseModel):
    name: str
    type: str
    attrs: dict[str, Any] = Field(default_factory=dict)


class DataOverviewResponse(BaseModel):
    dataset_source: str
    total_documents: int
    raw_shape_example: dict[str, Any]
    normalized_shape_example: dict[str, Any]
    index_name: str
    index_fields: list[IndexFieldDescriptor]
    redis_index_info: dict[str, Any] = Field(default_factory=dict)


class SimilarityRequest(BaseModel):
    sentence_a: str = Field(min_length=1)
    sentence_b: str = Field(min_length=1)

    @field_validator("sentence_a", "sentence_b")
    @classmethod
    def strip_sentence(cls, value: str) -> str:
        clean = value.strip()
        if not clean:
            raise ValueError("sentence must not be empty")
        return clean


class SimilarityResponse(BaseModel):
    sentence_a: str
    sentence_b: str
    cosine_similarity: float
    interpretation: str
    embedding_preview: dict[str, list[float]]
