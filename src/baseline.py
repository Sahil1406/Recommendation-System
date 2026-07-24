"""Baseline recommender (brief component 1).

Popularity baseline: rank items by number of ratings, tie-broken by mean
rating (with a Bayesian shrinkage so a movie with two 5-star ratings does
not beat one with 3000 ratings averaging 4.2).

This is the non-personalized reference every personalized model must beat.
"""

import pandas as pd


def popularity_scores(train: pd.DataFrame, m: int = 50) -> pd.Series:
    """Return a score per movieId. IMDb-style weighted rating:
        WR = (v / (v + m)) * R + (m / (v + m)) * C
    v = #ratings for the movie, R = its mean rating,
    C = global mean rating, m = shrinkage prior.
    """
    stats = train.groupby("movieId")["rating"].agg(["count", "mean"])
    C = train["rating"].mean()
    v, R = stats["count"], stats["mean"]
    wr = (v / (v + m)) * R + (m / (v + m)) * C
    # Popularity = weighted rating, but count dominates ranking for classic
    # "most popular" behaviour; we blend by sorting on (wr) computed above,
    # which already balances volume and quality.
    return wr.sort_values(ascending=False)


def recommend_popular(train: pd.DataFrame, user_id: int, n: int = 10) -> list:
    """Top-N most popular movies the user has NOT rated in train."""
    seen = set(train.loc[train.userId == user_id, "movieId"])
    scores = popularity_scores(train)
    recs = [mid for mid in scores.index if mid not in seen]
    return recs[:n]
