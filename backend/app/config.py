from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    redis_url: str = "redis://localhost:6379"
    redis_index_name: str = "movies_learning_index"
    redis_key_prefix: str = "movie"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    bootstrap_on_startup: bool = True
    dataset_source: str = "data/movies.json"
    request_limit_default: int = 8
    request_limit_max: int = 30
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_prefix="LAB_", extra="ignore")


settings = Settings()
