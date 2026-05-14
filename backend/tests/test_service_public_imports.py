import importlib
import inspect

import pytest


def test_service_public_symbols_alias_search_facade():
    service = importlib.import_module("app.service")
    search = importlib.import_module("app.search")
    advanced = importlib.import_module("app.search.modes.advanced")
    contracts = importlib.import_module("app.search.contracts")
    constants = importlib.import_module("app.search.constants")

    assert service.RETURN_FIELDS is constants.RETURN_FIELDS
    assert service.SearchService is contracts.SearchService
    assert contracts.SearchService.__module__ == "app.search.contracts"
    assert search.SearchService is contracts.SearchService
    assert service.SearchService is search.SearchService
    assert service.RedisVLSearchService is search.RedisVLSearchService
    assert service.fuse_rankings_rrf is search.fuse_rankings_rrf
    assert not hasattr(advanced, "install_service_rrf_alias")
    assert search.RedisVLSearchService.__module__ == "app.search.redis_service"
    assert search.fuse_rankings_rrf.__module__ == "app.search.modes.advanced"
    assert service.RedisVLSearchService.__module__ == "app.search.redis_service"
    assert service.fuse_rankings_rrf.__module__ == "app.search.modes.advanced"

    expected_methods = [
        "bootstrap",
        "search_text",
        "search_vector",
        "search_hybrid",
        "search_rrf",
        "search_rerank",
        "get_data_overview",
        "compare_sentence_similarity",
    ]
    contract_methods = [name for name, member in inspect.getmembers(contracts.SearchService, inspect.isfunction) if not name.startswith("_")]
    assert contract_methods == sorted(expected_methods)
    for method_name in expected_methods:
        assert inspect.signature(getattr(contracts.SearchService, method_name)) == inspect.signature(
            getattr(service.SearchService, method_name)
        )


def test_search_service_is_constructible_and_base_methods_raise_not_implemented():
    service = importlib.import_module("app.service")
    search_service = service.SearchService()

    with pytest.raises(NotImplementedError):
        search_service.bootstrap()
