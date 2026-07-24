"""Content-based recommender (brief component 2).

Item representation: TF-IDF over a text "document" per movie built from its
genres and user-assigned tags. Similarity: cosine.

User profile: rating-weighted average of the TF-IDF vectors of the movies the
user rated in train. Weights are (rating - user_mean), so movies the user
LIKED pull the profile toward them and movies the user DISLIKED push it away.
This is the "rating-weighted user profile" improvement from the brief.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_item_documents(movies: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    """One text document per movie: genres (pipe-split) + all tags."""
    tag_docs = (tags.assign(tag=tags.tag.astype(str).str.lower())
                    .groupby("movieId")["tag"].apply(" ".join))
    docs = movies[["movieId", "title"]].copy()
    genre_text = movies["genres"].str.replace("|", " ", regex=False).str.lower()
    docs["text"] = genre_text + " " + docs["movieId"].map(tag_docs).fillna("")
    return docs


class ContentRecommender:
    def __init__(self, movies: pd.DataFrame, tags: pd.DataFrame):
        self.docs = build_item_documents(movies, tags)
        self.vectorizer = TfidfVectorizer(min_df=2, stop_words="english")
        self.item_matrix = self.vectorizer.fit_transform(self.docs["text"])
        self.movie_ids = self.docs["movieId"].values
        self.id_to_row = {m: i for i, m in enumerate(self.movie_ids)}

    def user_profile(self, train: pd.DataFrame, user_id: int):
        """Rating-weighted centroid of the user's rated movies."""
        ur = train[train.userId == user_id]
        rows = [self.id_to_row[m] for m in ur.movieId if m in self.id_to_row]
        weights = (ur.rating - ur.rating.mean()).values[:len(rows)]
        if len(rows) == 0:
            return None
        vecs = self.item_matrix[rows]
        profile = weights @ vecs.toarray()
        norm = np.linalg.norm(profile)
        return profile / norm if norm > 0 else profile

    def scores(self, train: pd.DataFrame, user_id: int) -> pd.Series:
        """Cosine similarity between user profile and every movie."""
        profile = self.user_profile(train, user_id)
        sims = cosine_similarity(profile.reshape(1, -1), self.item_matrix).ravel()
        return pd.Series(sims, index=self.movie_ids)

    def recommend(self, train: pd.DataFrame, user_id: int, n: int = 10) -> list:
        seen = set(train.loc[train.userId == user_id, "movieId"])
        s = self.scores(train, user_id).sort_values(ascending=False)
        return [m for m in s.index if m not in seen][:n]
