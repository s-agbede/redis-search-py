import importlib


def _optional_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None


def test_service_public_symbols_alias_search_facade():
    service = importlib.import_module("app.service")
    search = importlib.import_module("app.search")
    contracts = importlib.import_module("app.search.contracts")
    constants = importlib.import_module("app.search.constants")

    assert service.RETURN_FIELDS is constants.RETURN_FIELDS
    assert service.SearchService is contracts.SearchService
    assert service.SearchService is search.SearchService
    assert service.RedisVLSearchService is search.RedisVLSearchService
    assert service.fuse_rankings_rrf is search.fuse_rankings_rrf

    redis_service = _optional_import("app.search.redis_service")
    if redis_service is not None:
        assert search.RedisVLSearchService is redis_service.RedisVLSearchService

    advanced = _optional_import("app.search.advanced")
    if advanced is not None:
        assert search.fuse_rankings_rrf is advanced.fuse_rankings_rrf
