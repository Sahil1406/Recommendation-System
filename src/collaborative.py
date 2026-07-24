"""Collaborative filtering (brief component 3). Two methods:

1. Item-Item CF: cosine similarity between mean-centered item rating vectors.
   Learns "people who liked X also liked Y" purely from interactions.
   Score for an unseen item = similarity-weighted average of the user's own
   ratings on its k most similar neighbours.

2. SVD (matrix factorization): factorizes the mean-centered user-item matrix
   into latent user and item factors. Learns hidden taste dimensions
   (e.g. "dark vs feel-good") without any metadata.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity


class ItemItemCF:
    def __init__(self, train: pd.DataFrame, k: int = 30, min_ratings: int = 5):
        self.k = k
        # keep items with enough ratings so similarities are stable
        counts = train.groupby("movieId").size()
        keep = counts[counts >= min_ratings].index
        train = train[train.movieId.isin(keep)]

        self.user_ids = np.sort(train.userId.unique())
        self.movie_ids = np.sort(train.movieId.unique())
        self.uidx = {u: i for i, u in enumerate(self.user_ids)}
        self.midx = {m: i for i, m in enumerate(self.movie_ids)}

        # mean-center per user (removes "generous vs harsh rater" bias)
        self.user_means = train.groupby("userId")["rating"].mean()
        centered = train.rating.values - train.userId.map(self.user_means).values
        rows = train.userId.map(self.uidx).values
        cols = train.movieId.map(self.midx).values
        self.R = csr_matrix((centered, (rows, cols)),
                            shape=(len(self.user_ids), len(self.movie_ids)))
        # item-item cosine similarity on centered ratings (= adjusted cosine)
        self.sim = cosine_similarity(self.R.T, dense_output=False)
        self.train = train

    def scores(self, user_id: int) -> pd.Series:
        ur = self.train[self.train.userId == user_id]
        rated_idx = np.array([self.midx[m] for m in ur.movieId])
        centered = ur.rating.values - self.user_means[user_id]

        sim_block = self.sim[:, rated_idx].toarray()  # (n_items, n_rated)
        # keep only top-k neighbours per target item
        if sim_block.shape[1] > self.k:
            thresh = np.sort(sim_block, axis=1)[:, -self.k][:, None]
            sim_block[sim_block < thresh] = 0.0
        num = sim_block @ centered
        den = np.abs(sim_block).sum(axis=1) + 1e-9
        preds = self.user_means[user_id] + num / den
        return pd.Series(preds, index=self.movie_ids)

    def recommend(self, user_id: int, n: int = 10) -> list:
        seen = set(self.train.loc[self.train.userId == user_id, "movieId"])
        s = self.scores(user_id).sort_values(ascending=False)
        return [m for m in s.index if m not in seen][:n]


class SVDRecommender:
    def __init__(self, train: pd.DataFrame, n_factors: int = 50):
        self.user_ids = np.sort(train.userId.unique())
        self.movie_ids = np.sort(train.movieId.unique())
        self.uidx = {u: i for i, u in enumerate(self.user_ids)}
        self.midx = {m: i for i, m in enumerate(self.movie_ids)}
        self.user_means = train.groupby("userId")["rating"].mean()

        centered = train.rating.values - train.userId.map(self.user_means).values
        rows = train.userId.map(self.uidx).values
        cols = train.movieId.map(self.midx).values
        R = csr_matrix((centered, (rows, cols)),
                       shape=(len(self.user_ids), len(self.movie_ids)))

        U, s, Vt = svds(R, k=n_factors)
        self.user_factors = U * s          # (n_users, k)
        self.item_factors = Vt.T           # (n_items, k)
        self.train = train

    def scores(self, user_id: int) -> pd.Series:
        u = self.user_factors[self.uidx[user_id]]
        preds = self.user_means[user_id] + self.item_factors @ u
        return pd.Series(preds, index=self.movie_ids)

    def recommend(self, user_id: int, n: int = 10) -> list:
        seen = set(self.train.loc[self.train.userId == user_id, "movieId"])
        s = self.scores(user_id).sort_values(ascending=False)
        return [m for m in s.index if m not in seen][:n]
