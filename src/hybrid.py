"""Hybrid recommender (brief component 4).

Weighted hybrid: min-max normalize the score vectors of the content-based
model and the CF model to [0, 1], then blend:

    hybrid = alpha * CF + (1 - alpha) * content

Why useful:
- CF is strong for users with rich history but ignores WHY items fit.
- Content-based handles niche/new items (no cold-start on item side)
  and is interpretable ("recommended because you liked dark sci-fi").
- Blending recovers items each model alone under-ranks and makes the
  final list both accurate and explainable.
"""

import pandas as pd


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else s * 0


def hybrid_recommend(content_scores: pd.Series, cf_scores: pd.Series,
                     seen: set, alpha: float = 0.6, n: int = 10) -> list:
    c = _minmax(content_scores)
    f = _minmax(cf_scores)
    common = c.index.intersection(f.index)
    blend = alpha * f[common] + (1 - alpha) * c[common]
    blend = blend.sort_values(ascending=False)
    return [m for m in blend.index if m not in seen][:n]


def hybrid_scores(content_scores: pd.Series, cf_scores: pd.Series,
                  alpha: float = 0.6) -> pd.Series:
    c, f = _minmax(content_scores), _minmax(cf_scores)
    common = c.index.intersection(f.index)
    return alpha * f[common] + (1 - alpha) * c[common]
