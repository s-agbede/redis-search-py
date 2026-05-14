from app.search.modes.advanced import fuse_rankings_rrf


def test_fuse_rankings_rrf_prefers_items_ranked_high_in_both_lists():
    ranked_lists = [
        ["a", "b", "c", "d"],
        ["b", "a", "e", "f"],
    ]
    scores = fuse_rankings_rrf(ranked_lists, weights=[0.5, 0.5], k=60)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    assert ranked[0][0] in {"a", "b"}
    assert scores["a"] > scores["c"]
    assert scores["b"] > scores["e"]


def test_fuse_rankings_rrf_validates_weight_length():
    try:
        fuse_rankings_rrf([["a"], ["b"]], weights=[1.0], k=60)
    except ValueError as exc:
        assert "weights length" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid weight length")
