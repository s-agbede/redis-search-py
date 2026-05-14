"""Stable compatibility facade for callers that still import from app.service."""

from app.search import RedisVLSearchService, SearchService, fuse_rankings_rrf
from app.search.constants import RETURN_FIELDS

__all__ = ["RETURN_FIELDS", "SearchService", "RedisVLSearchService", "fuse_rankings_rrf"]
