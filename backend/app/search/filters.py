"""Reusable RedisVL filter-expression helpers."""

from redisvl.query.filter import Num, Tag


def build_filter_expression(genres: list[str], min_rating: float | None):
    expression = None

    if genres:
        # Multiple genres are treated as OR clauses on the same tag field.
        genre_expr = None
        for genre in genres:
            current = Tag("genres") == genre
            genre_expr = current if genre_expr is None else (genre_expr | current)

        expression = genre_expr

    if min_rating is not None:
        rating_expr = Num("rating") >= float(min_rating)
        expression = rating_expr if expression is None else (expression & rating_expr)

    return expression


__all__ = ["build_filter_expression"]
