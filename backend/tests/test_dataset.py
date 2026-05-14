import json

from app.dataset import fetch_movies_json, normalize_movies


def test_normalize_movies_extracts_nested_fields():
    payload = [
        {
            "title": "A Test Movie",
            "year": 2012,
            "info": {
                "plot": "A brave tester explores a new codebase.",
                "rating": 7.4,
                "genres": ["Adventure", "Drama"],
                "actors": ["Alex", "Sam"],
                "release_date": "2012-01-01T00:00:00Z",
                "rank": 42,
                "image_url": "https://example.com/test.jpg",
                "running_time_secs": 5400,
            },
        }
    ]

    rows = normalize_movies(payload)

    assert len(rows) == 1
    row = rows[0]
    assert row["title"] == "A Test Movie"
    assert row["year"] == 2012
    assert row["plot"] == "A brave tester explores a new codebase."
    assert row["rating"] == 7.4
    assert row["genres"] == ["Adventure", "Drama"]
    assert row["actors"] == ["Alex", "Sam"]
    assert row["release_date"] == "2012-01-01T00:00:00Z"
    assert row["rank"] == 42
    assert row["image_url"] == "https://example.com/test.jpg"
    assert row["running_time_secs"] == 5400


def test_normalize_movies_uses_stable_id():
    payload = [
        {
            "title": "Stable Movie",
            "year": 2001,
            "info": {"plot": "test", "rank": 1},
        }
    ]

    row_a = normalize_movies(payload)[0]
    row_b = normalize_movies(payload)[0]
    assert row_a["id"] == row_b["id"]


def test_fetch_movies_json_from_local_file(tmp_path):
    path = tmp_path / "movies.json"
    payload = [{"title": "t", "year": 2024, "info": {"plot": "p"}}]
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = fetch_movies_json(str(path))
    assert loaded == payload
