"""RedisVL schema builder shared by search implementations."""

from redisvl.schema import IndexSchema

from app.config import settings


def build_schema() -> IndexSchema:
    # Keep the Redis index definition centralized so service setup and queries stay aligned.
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


__all__ = ["build_schema"]
