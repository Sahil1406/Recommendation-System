"""End-to-end experiment: builds all recommenders, prints Top-10 lists for a
demo user, and evaluates all models against the popularity baseline.

Run:  python run_experiment.py
"""

import sys
import numpy as np
import pandas as pd

sys.path.append("src")
from data import load_data, temporal_split, pick_demo_user
from baseline import popularity_scores, recommend_popular
from content_based import ContentRecommender
from collaborative import ItemItemCF, SVDRecommender
from hybrid import hybrid_recommend, hybrid_scores
from evaluate import evaluate_model

pd.set_option("display.width", 120)
np.random.seed(42)


def titles(movie_ids, movies):
    t = movies.set_index("movieId")["title"]
    return [t.get(m, str(m)) for m in movie_ids]


def main():
    # ---------- 1. Data ----------
    ratings, movies, tags = load_data()
    train, test = temporal_split(ratings, test_frac=0.2, min_ratings=20)
    print(f"Ratings: {len(ratings):,} | train {len(train):,} / test {len(test):,}")
    print(f"Users: {ratings.userId.nunique()} | Movies: {ratings.movieId.nunique()}\n")

    # ---------- 2. Demo user ----------
    demo_user = pick_demo_user(train, test)
    ur = train[train.userId == demo_user].merge(movies, on="movieId")
    top_liked = ur.sort_values("rating", ascending=False).head(8)
    print(f"=== Demo user: {demo_user} "
          f"({len(ur)} train ratings, mean {ur.rating.mean():.2f}) ===")
    print("Their highest-rated movies (taste profile):")
    for _, r in top_liked.iterrows():
        print(f"  {r.rating:.1f}  {r.title}  [{r.genres}]")
    print()

    # ---------- 3. Build models ----------
    print("Building models...")
    content = ContentRecommender(movies, tags)
    itemcf = ItemItemCF(train, k=30)
    svd = SVDRecommender(train, n_factors=50)

    seen = set(train.loc[train.userId == demo_user, "movieId"])

    lists = {
        "Popularity baseline": recommend_popular(train, demo_user),
        "Content-based (TF-IDF)": content.recommend(train, demo_user),
        "Item-Item CF": itemcf.recommend(demo_user),
        "SVD": svd.recommend(demo_user),
        "Hybrid (0.6*SVD + 0.4*content)": hybrid_recommend(
            content.scores(train, demo_user), svd.scores(demo_user),
            seen, alpha=0.6),
    }

    # ---------- 4. Top-10 lists ----------
    for name, recs in lists.items():
        print(f"\n--- Top-10: {name} ---")
        for i, t in enumerate(titles(recs, movies), 1):
            print(f"  {i:2d}. {t}")

    # ---------- 5. Evaluation over all eligible users ----------
    eval_users = sorted(set(test.userId) & set(train.userId))
    # restrict to users all models can score
    eval_users = [u for u in eval_users
                  if u in itemcf.uidx and u in svd.uidx]
    print(f"\nEvaluating on {len(eval_users)} users...")

    pop_ranked = popularity_scores(train)

    def pop_fn(u):
        s = set(train.loc[train.userId == u, "movieId"])
        return [m for m in pop_ranked.index if m not in s][:10]

    def content_fn(u):
        return content.recommend(train, u)

    def itemcf_fn(u):
        return itemcf.recommend(u)

    def svd_fn(u):
        return svd.recommend(u)

    def hybrid_fn(u):
        s = set(train.loc[train.userId == u, "movieId"])
        return hybrid_recommend(content.scores(train, u), svd.scores(u), s, 0.6)

    results = {}
    for name, fn in [("Popularity", pop_fn), ("Content", content_fn),
                     ("Item-Item CF", itemcf_fn), ("SVD", svd_fn),
                     ("Hybrid", hybrid_fn)]:
        results[name] = evaluate_model(fn, eval_users, test)
        print(f"  done: {name}")

    table = pd.DataFrame(results).T.round(4)
    print("\n=== Evaluation (avg over users, relevant = held-out ratings >= 4) ===")
    print(table)
    table.to_csv("results.csv")
    print("\nSaved metrics to results.csv")


if __name__ == "__main__":
    main()
