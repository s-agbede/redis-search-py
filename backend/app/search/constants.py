"""Field set returned from Redis queries."""

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

__all__ = ["RETURN_FIELDS"]
