"""Data loading, preprocessing and train/test splitting for MovieLens ml-latest-small.

Covers brief requirement 1: preprocessing and user selection.
- Loads ratings, movies, tags
- Splits each user's ratings into train (seen) and test (held-out, treated as unseen)
- Uses a *temporal* split per user: the most recent 20% of each user's ratings
  are held out. This mimics the real task: predict what the user watches next.
"""

import pandas as pd
import numpy as np

DATA_DIR = "data"


def load_data(data_dir: str = DATA_DIR):
    ratings = pd.read_csv(f"{data_dir}/ratings.csv")
    movies = pd.read_csv(f"{data_dir}/movies.csv")
    tags = pd.read_csv(f"{data_dir}/tags.csv")
    return ratings, movies, tags


def temporal_split(ratings: pd.DataFrame, test_frac: float = 0.2, min_ratings: int = 20):
    """Per-user temporal split.

    Users with fewer than `min_ratings` ratings are kept only in train
    (not enough history to evaluate them fairly).
    Returns (train_df, test_df).
    """
    ratings = ratings.sort_values(["userId", "timestamp"])
    train_parts, test_parts = [], []
    for _, grp in ratings.groupby("userId"):
        if len(grp) < min_ratings:
            train_parts.append(grp)
            continue
        n_test = max(1, int(len(grp) * test_frac))
        train_parts.append(grp.iloc[:-n_test])
        test_parts.append(grp.iloc[-n_test:])
    return pd.concat(train_parts), pd.concat(test_parts)


def build_user_item_matrix(train: pd.DataFrame):
    """Pivot train ratings into a dense user x item matrix (NaN = unrated)."""
    return train.pivot_table(index="userId", columns="movieId", values="rating")


def pick_demo_user(train: pd.DataFrame, test: pd.DataFrame,
                   min_train: int = 100, min_test: int = 20) -> int:
    """Pick a user suitable for demonstration: rich history, enough held-out
    items, and diverse ratings (uses both high and low ratings, so the
    profile is informative)."""
    tr_counts = train.groupby("userId").size()
    te_counts = test.groupby("userId").size()
    rating_std = train.groupby("userId")["rating"].std()
    candidates = tr_counts[(tr_counts >= min_train)].index
    candidates = [u for u in candidates if te_counts.get(u, 0) >= min_test]
    # Prefer users whose ratings vary (std close to 1 -> expressive profile)
    candidates = sorted(candidates, key=lambda u: -rating_std.get(u, 0))
    return candidates[0]
