# Personalized Movie Recommender — MovieLens ml-latest-small

Exam project for *Recommendation Systems and Media Personalization* (k_ADSA_006, Summer 2026).

Builds, evaluates and explains a personalized recommender that beats a popularity baseline, using content-based filtering, collaborative filtering (Item-Item CF + SVD), and a weighted hybrid.

## How to run

```bash
pip install pandas numpy scikit-learn scipy streamlit
python run_experiment.py     # full evaluation -> results.csv
streamlit run app.py         # interactive demo UI
```

`run_experiment.py` prints the demo user's taste profile, Top-10 lists from all five models, and an evaluation table (saved to `results.csv`).

## Demo UI (`app.py`)

A Streamlit app ("Projector") built for the exam-day live demo:

- **Pick any user** from the sidebar and switch between all five models live — the audience sees the list *change per user and per model*, which is the whole argument for personalization.
- **Explanation cards** — every recommendation states *why* in plain language: *"Because you rated Children of Men ★4.5 — shares your favourite genres: drama, sci-fi."* Popularity-baseline cards deliberately say "same list for everyone — not personalized" for contrast.
- **✓ markers** flag recommended movies the user *actually watched and liked* in the held-out test period — live proof the model predicts real future behaviour.
- **α slider** for the hybrid lets you show, live, how the list morphs from pure content-based (α=0) to pure SVD (α=1).
- The models train once on first launch (~1 min) and are cached; everything after that is instant.

## Data

MovieLens **ml-latest-small** (`data/`): ~100,000 ratings (0.5–5.0 stars) by 610 users on ~9,700 movies, plus genres and free-text tags.
- `ratings.csv` — userId, movieId, rating, timestamp (explicit feedback)
- `movies.csv` — title, pipe-separated genres (metadata)
- `tags.csv` — free-text user tags (text metadata)

**Preprocessing:** per-user *temporal* split — the most recent 20% of each user's ratings are held out as test ("unseen"). Users with <20 ratings stay train-only. Temporal splitting mimics the real task: predict what the user watches *next*, and prevents leakage from future interactions.

## Project structure

| File | Component (maps to brief) |
|---|---|
| `src/data.py` | Preprocessing, train/test split, demo-user selection |
| `src/baseline.py` | Popularity baseline (Bayesian-weighted rating) |
| `src/content_based.py` | Content-based: TF-IDF over genres+tags, rating-weighted user profile, cosine similarity |
| `src/collaborative.py` | Item-Item CF (adjusted cosine, top-k neighbours) and SVD matrix factorization |
| `src/hybrid.py` | Improvement: weighted hybrid `α·SVD + (1−α)·content` |
| `src/evaluate.py` | Precision@10, Recall@10, HitRate@10, NDCG@10 |
| `run_experiment.py` | End-to-end pipeline |

## Methods in one paragraph each

**Baseline.** Movies ranked by an IMDb-style weighted rating that shrinks small-sample averages toward the global mean — the same list for every user. This is what "recommend what's popular" looks like.

**Content-based.** Each movie becomes a TF-IDF document from its genres and tags. The user profile is the *rating-weighted* centroid of their rated movies: ratings above the user's own mean pull the profile toward those movies, ratings below push away. Unseen movies are ranked by cosine similarity to the profile. Fully interpretable ("because you liked animated fantasy films").

**Item-Item CF.** Adjusted-cosine similarity between items computed from mean-centered ratings (removes generous/harsh rater bias). An unseen item's score is the similarity-weighted average of the user's own ratings on its 30 nearest neighbours. Learns "people who liked X also liked Y" from behaviour alone — no metadata.

**SVD.** Factorizes the mean-centered user–item matrix into 50 latent factors. Each user and movie gets a position in a hidden "taste space"; the predicted rating is the dot product plus the user's mean. Captures patterns no single genre tag describes.

**Hybrid (improvement).** Min-max normalize the content and SVD score vectors, blend with α=0.6. CF contributes accuracy, content contributes interpretability and coverage of niche items with few ratings.

## Results (610 users, relevant = held-out ratings ≥ 4)

| Model | Precision@10 | Recall@10 | HitRate@10 | NDCG@10 |
|---|---|---|---|---|
| Popularity | 0.0582 | 0.0502 | 0.3249 | 0.0748 |
| Content | 0.0054 | 0.0043 | 0.0491 | 0.0063 |
| Item-Item CF | 0.0242 | 0.0075 | 0.1320 | 0.0229 |
| **SVD** | **0.0651** | **0.0645** | **0.3892** | **0.0835** |
| Hybrid (α=0.6) | 0.0457 | 0.0468 | 0.3080 | 0.0659 |

**Why is this personalized beyond popularity?** SVD outperforms the popularity baseline on every metric while producing a *different list per user*, driven by that user's own rating history. The demo user's Top-10s (see `run_experiment.py` output) contain animation and character-driven films matching their profile, not the global blockbusters the baseline returns.

**Honest limitations (great for the exam Q&A):**
- Pure content-based scores low on accuracy metrics: it recommends *similar* movies, but held-out test items are often outside the user's dominant genre — a known weakness (over-specialization / filter bubble). Its value is interpretability and item cold-start, which is why we hybridize.
- Popularity is a surprisingly strong baseline on MovieLens (popular movies really are widely liked) — beating it at all is meaningful.
- The hybrid at α=0.6 sits between its parents; tuning α on a validation split (or a learned blend) is the natural next step.
- Offline metrics only measure "did we predict what they happened to watch next", not true recommendation quality; an A/B test would be needed in production.

## Suggested member split (3–5 people)

1. Data + preprocessing + demo-user story (`data.py`)
2. Content-based method + interpretation of recommendations (`content_based.py`)
3. Collaborative filtering (`collaborative.py`)
4. Hybrid + evaluation + baseline comparison (`hybrid.py`, `evaluate.py`, `baseline.py`)
5. Slides, demo walkthrough, report
