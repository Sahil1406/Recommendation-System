"""Evaluation (brief component 5).

Protocol:
- For each evaluation user, "relevant" = held-out (test) items the user rated
  >= 4.0 (they demonstrably liked them).
- Each model ranks the user's unseen items; we take Top-K and compare
  against the relevant set.

Metrics: Precision@K, Recall@K, HitRate@K, NDCG@K
All averaged over evaluation users.
"""

import numpy as np


def precision_at_k(recs, relevant, k=10):
    return len(set(recs[:k]) & relevant) / k


def recall_at_k(recs, relevant, k=10):
    return len(set(recs[:k]) & relevant) / len(relevant) if relevant else 0.0


def hitrate_at_k(recs, relevant, k=10):
    return 1.0 if set(recs[:k]) & relevant else 0.0


def ndcg_at_k(recs, relevant, k=10):
    dcg = sum(1.0 / np.log2(i + 2) for i, m in enumerate(recs[:k]) if m in relevant)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def evaluate_model(recommend_fn, users, test, k=10, like_threshold=4.0):
    """recommend_fn(user_id) -> ranked list of movieIds (unseen only).

    Returns dict of averaged metrics over users that have at least one
    relevant held-out item.
    """
    metrics = {"Precision@10": [], "Recall@10": [], "HitRate@10": [], "NDCG@10": []}
    for u in users:
        relevant = set(test.loc[(test.userId == u) &
                                (test.rating >= like_threshold), "movieId"])
        if not relevant:
            continue
        recs = recommend_fn(u)
        metrics["Precision@10"].append(precision_at_k(recs, relevant, k))
        metrics["Recall@10"].append(recall_at_k(recs, relevant, k))
        metrics["HitRate@10"].append(hitrate_at_k(recs, relevant, k))
        metrics["NDCG@10"].append(ndcg_at_k(recs, relevant, k))
    return {m: float(np.mean(v)) for m, v in metrics.items()}
