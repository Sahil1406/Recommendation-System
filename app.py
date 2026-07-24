"""Interactive demo UI for the exam project.

Run:  streamlit run app.py

Pick a user, pick a model, get Top-N recommendations — each with a plain-
language explanation of WHY it was recommended (exam requirement #9).
"""

import sys
import numpy as np
import pandas as pd
import streamlit as st

sys.path.append("src")
from data import load_data, temporal_split
from baseline import popularity_scores
from content_based import ContentRecommender
from collaborative import ItemItemCF, SVDRecommender
from hybrid import hybrid_scores

# ---------------------------------------------------------------- page setup
st.set_page_config(page_title="Projector — personalized movie recommender",
                   page_icon="🎞️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Inter:wght@400;500;600&display=swap');

:root {
  --ink: #14171F;        /* room dark */
  --panel: #1D2230;      /* card surface */
  --beam: #F2B33D;       /* projector amber */
  --beam-soft: rgba(242,179,61,.14);
  --text: #EDEAE2;       /* screen white */
  --muted: #9AA0AE;
}
.stApp { background: var(--ink); color: var(--text); }
h1, h2, h3 { font-family: 'Fraunces', Georgia, serif !important; color: var(--text) !important; }
html, body, p, div, span, label { font-family: 'Inter', system-ui, sans-serif; }

.title-block { border-left: 4px solid var(--beam); padding-left: 18px; margin-bottom: 4px; }
.title-block .eyebrow { color: var(--beam); font-size: .78rem; letter-spacing: .14em;
  text-transform: uppercase; font-weight: 600; }

.rec-card { background: var(--panel); border-radius: 10px; padding: 14px 18px;
  margin-bottom: 10px; border: 1px solid rgba(255,255,255,.05); }
.rec-rank { color: var(--beam); font-family: 'Fraunces', serif; font-size: 1.35rem;
  font-weight: 700; width: 2rem; display: inline-block; }
.rec-title { font-weight: 600; font-size: 1.02rem; }
.rec-genres { color: var(--muted); font-size: .8rem; margin: 2px 0 6px 2rem; }
.rec-why { background: var(--beam-soft); border-radius: 6px; padding: 7px 12px;
  font-size: .85rem; margin-left: 2rem; color: var(--text); }
.rec-why b { color: var(--beam); }

.profile-chip { display: inline-block; background: var(--panel); border-radius: 999px;
  padding: 4px 12px; margin: 3px 4px 3px 0; font-size: .82rem;
  border: 1px solid rgba(255,255,255,.08); }
.profile-chip .star { color: var(--beam); font-weight: 600; }

[data-testid="stSidebar"] { background: #101319; }
[data-testid="stMetricValue"] { color: var(--beam); font-family: 'Fraunces', serif; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- cached data
@st.cache_resource(show_spinner="Loading data and training models (first run only)...")
def load_everything():
    ratings, movies, tags = load_data()
    train, test = temporal_split(ratings)
    content = ContentRecommender(movies, tags)
    itemcf = ItemItemCF(train, k=30)
    svd = SVDRecommender(train, n_factors=50)
    pop = popularity_scores(train)
    title_map = movies.set_index("movieId")["title"]
    genre_map = movies.set_index("movieId")["genres"]
    return ratings, movies, tags, train, test, content, itemcf, svd, pop, title_map, genre_map

(ratings, movies, tags, train, test, content, itemcf, svd, pop,
 title_map, genre_map) = load_everything()

# ---------------------------------------------------------------- helpers
def explain(movie_id, user_train, model_name):
    """Plain-language reason for a recommendation."""
    liked = user_train[user_train.rating >= 4.0]
    if model_name.startswith("Popularity"):
        n = int(train[train.movieId == movie_id].shape[0])
        return f"Rated by <b>{n} users</b> — same list for everyone. Not personalized."
    # shared genres with the user's liked movies
    target_genres = set(str(genre_map.get(movie_id, "")).split("|"))
    liked_genres = pd.Series(
        [g for gs in liked.movieId.map(genre_map).dropna() for g in gs.split("|")]
    ).value_counts()
    shared = [g for g in liked_genres.index if g in target_genres][:3]
    # most similar liked movie via item-item similarity (works for CF & hybrid)
    anchor = None
    if movie_id in itemcf.midx:
        col = itemcf.sim[:, itemcf.midx[movie_id]].toarray().ravel()
        best_sim, best_m = 0, None
        for m, r in zip(liked.movieId, liked.rating):
            if m in itemcf.midx and col[itemcf.midx[m]] > best_sim:
                best_sim, best_m = col[itemcf.midx[m]], m
        if best_m is not None and best_sim > 0.05:
            anchor = (title_map.get(best_m, "?"),
                      float(liked.loc[liked.movieId == best_m, "rating"].iloc[0]))
    parts = []
    if anchor:
        parts.append(f"Because you rated <b>{anchor[0]}</b> ★{anchor[1]:.1f}")
    if shared:
        parts.append(f"shares your favourite genres: <b>{', '.join(g.lower() for g in shared)}</b>")
    return " — ".join(parts) if parts else "Close to your overall taste profile in the model's learned space."


def get_scores(model_name, user_id):
    if model_name == "Popularity baseline":
        return pop
    if model_name == "Content-based (TF-IDF)":
        return content.scores(train, user_id)
    if model_name == "Item-Item CF":
        return itemcf.scores(user_id)
    if model_name == "SVD (matrix factorization)":
        return svd.scores(user_id)
    return hybrid_scores(content.scores(train, user_id), svd.scores(user_id),
                         alpha=st.session_state.get("alpha", 0.6))

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Controls")
    mode = st.radio("Start from...", ["A user", "A movie"], horizontal=True)

    eligible = sorted(set(train.userId) & set(itemcf.uidx) & set(svd.uidx))
    counts = train.groupby("userId").size()

    if mode == "A user":
        default_user = 599 if 599 in eligible else eligible[0]
        user_id = st.selectbox("User", eligible, index=eligible.index(default_user),
                               format_func=lambda u: f"User {u} · {counts[u]} ratings")
        model_name = st.radio("Model", ["Popularity baseline", "Content-based (TF-IDF)",
                                        "Item-Item CF", "SVD (matrix factorization)",
                                        "Hybrid (SVD + content)"])
        if model_name.startswith("Hybrid"):
            st.slider("α — weight on SVD vs content", 0.0, 1.0, 0.6, 0.05, key="alpha")
    else:
        # movies with enough ratings so CF similarities are meaningful
        m_counts = train.groupby("movieId").size()
        pickable = [m for m in m_counts[m_counts >= 20].index
                    if m in itemcf.midx and m in content.id_to_row]
        pickable = sorted(pickable, key=lambda m: -m_counts[m])
        seed_movie = st.selectbox(
            "Movie (type to search)", pickable,
            format_func=lambda m: f"{title_map.get(m)} · {m_counts[m]} ratings")
        item_model = st.radio("Similarity from", [
            "Content (what the movie IS)",
            "Item-Item CF (who LIKED it)",
            "Blend of both"])
        if item_model.startswith("Blend"):
            st.slider("α — weight on CF vs content", 0.0, 1.0, 0.5, 0.05, key="item_alpha")

    top_n = st.slider("Recommendations", 5, 20, 10)
    st.caption("Models train once and are cached. Held-out test items are "
               "never used for recommending — only for evaluation.")

# ---------------------------------------------------------------- header
st.markdown("""
<div class="title-block">
  <div class="eyebrow">Recommendation systems · exam project</div>
  <h1 style="margin:0">Projector</h1>
  <p style="color:var(--muted); margin-top:2px">A personalized movie recommender
  that can say <i>why</i> — not just <i>what</i>.</p>
</div>
""", unsafe_allow_html=True)

if mode == "A user":
    user_train = train[train.userId == user_id].merge(movies, on="movieId")

    # ------------------------------------------------------------ taste profile
    st.markdown("### This user's taste")
    c1, c2, c3 = st.columns(3)
    c1.metric("Movies rated", len(user_train))
    c2.metric("Mean rating", f"{user_train.rating.mean():.2f}")
    fav_genre = pd.Series([g for gs in user_train.genres for g in gs.split("|")]).value_counts()
    c3.metric("Top genre", fav_genre.index[0])

    chips = user_train.sort_values("rating", ascending=False).head(8)
    st.markdown("".join(
        f'<span class="profile-chip"><span class="star">★{r.rating:.1f}</span> {r.title}</span>'
        for _, r in chips.iterrows()), unsafe_allow_html=True)

    # ------------------------------------------------------------ recommendations
    st.markdown(f"### Top-{top_n} · {model_name}")
    scores = get_scores(model_name, user_id)
    seen = set(train.loc[train.userId == user_id, "movieId"])
    ranked = [m for m in scores.sort_values(ascending=False).index if m not in seen][:top_n]

    held_out_liked = set(test.loc[(test.userId == user_id) & (test.rating >= 4.0), "movieId"])

    for i, m in enumerate(ranked, 1):
        hit = " ✓ user actually watched & liked this later" if m in held_out_liked else ""
        st.markdown(f"""
        <div class="rec-card">
          <span class="rec-rank">{i:02d}</span>
          <span class="rec-title">{title_map.get(m, m)}</span>
          <span style="color:var(--beam); font-size:.8rem">{hit}</span>
          <div class="rec-genres">{str(genre_map.get(m, "")).replace("|", " · ")}</div>
          <div class="rec-why">{explain(m, user_train, model_name)}</div>
        </div>""", unsafe_allow_html=True)

else:
    # ------------------------------------------------------------ movie mode
    seed_title = title_map.get(seed_movie)
    st.markdown(f"### Movies like *{seed_title}*")
    st.markdown(f'<span class="profile-chip">'
                f'{str(genre_map.get(seed_movie, "")).replace("|", " · ")}</span>',
                unsafe_allow_html=True)

    content_sim = content.similar_to(seed_movie)
    cf_col = itemcf.sim[:, itemcf.midx[seed_movie]].toarray().ravel()
    cf_sim = pd.Series(cf_col, index=itemcf.movie_ids).drop(seed_movie)

    # ignore rarely-rated movies in CF ranking (similarities too noisy)
    m_counts = train.groupby("movieId").size()
    stable = m_counts[m_counts >= 10].index

    if item_model.startswith("Content"):
        ranked_s = content_sim.sort_values(ascending=False)
    elif item_model.startswith("Item-Item"):
        ranked_s = cf_sim[cf_sim.index.isin(stable)].sort_values(ascending=False)
    else:
        a = st.session_state.get("item_alpha", 0.5)
        common = content_sim.index.intersection(cf_sim.index).intersection(stable)
        c = content_sim[common]; f = cf_sim[common]
        c = (c - c.min()) / (c.max() - c.min())
        f = (f - f.min()) / (f.max() - f.min())
        ranked_s = (a * f + (1 - a) * c).sort_values(ascending=False)

    ranked = list(ranked_s.index[:top_n])

    n_seed = int(m_counts.get(seed_movie, 0))
    for i, m in enumerate(ranked, 1):
        # explanation: shared content terms + co-liking evidence
        why = []
        terms = content.shared_terms(seed_movie, m) if m in content.id_to_row else []
        if terms:
            why.append(f"Shares: <b>{', '.join(terms)}</b>")
        if m in itemcf.midx:
            both = train[train.movieId.isin([seed_movie, m]) & (train.rating >= 4.0)]
            co = both.groupby("userId").movieId.nunique()
            n_co = int((co == 2).sum())
            if n_co > 0:
                why.append(f"<b>{n_co} users</b> loved both movies")
        why = " — ".join(why) if why else "Similar position in the model's learned space."
        st.markdown(f"""
        <div class="rec-card">
          <span class="rec-rank">{i:02d}</span>
          <span class="rec-title">{title_map.get(m, m)}</span>
          <div class="rec-genres">{str(genre_map.get(m, "")).replace("|", " · ")}</div>
          <div class="rec-why">{why}</div>
        </div>""", unsafe_allow_html=True)

    st.caption(f"Content similarity compares what *{seed_title}* IS (genres, tags). "
               f"Item-Item CF compares who LIKED it (co-rating patterns across "
               f"{n_seed} raters). The lists differ — that difference is the "
               "content-vs-collaborative story of this project.")

# ---------------------------------------------------------------- evaluation
st.markdown("### How the models compare (610 users)")
try:
    res = pd.read_csv("results.csv", index_col=0)
    st.dataframe(res.style.highlight_max(axis=0, color="#3d3216"),
                 use_container_width=True)
    st.caption("Relevant = held-out ratings ≥ 4.0. SVD beats the popularity "
               "baseline on every metric — the recommender is personalized "
               "beyond popularity.")
except FileNotFoundError:
    st.info("Run `python run_experiment.py` once to generate results.csv.")
