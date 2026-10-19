from __future__ import annotations

import argparse
import json
from time import perf_counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

from inf8239_u03.config import MOVIELENS_DIR, ROOT
from inf8239_u03.data import load_movielens
from inf8239_u03.metrics import catalog_coverage, hit_rate_at_k
from inf8239_u03.recommenders import MatrixFactorization, temporal_leave_one_out

parser = argparse.ArgumentParser()
parser.add_argument("--factors", type=int, default=20)
parser.add_argument("--epochs", type=int, default=12)
parser.add_argument("--alpha", type=float, default=0.75)
args = parser.parse_args()

ratings, movies = load_movielens(MOVIELENS_DIR)
train, test = temporal_leave_one_out(ratings)
start = perf_counter()
model = MatrixFactorization(factors=args.factors).fit(train, epochs=args.epochs)
train_seconds = perf_counter() - start
evaluable = test[test["userId"].isin(model.user_index) & test["movieId"].isin(model.item_index)]
predictions = [model.predict(row.userId, row.movieId) for row in evaluable.itertuples()]
rmse = root_mean_squared_error(evaluable["rating"], predictions)
popular = train.groupby("movieId")["rating"].agg(["mean", "count"]).sort_values(["count", "mean"], ascending=False)
popular_items = popular.index.tolist()
movie_table = movies.reset_index(drop=True).copy()
movie_table["genres_text"] = movie_table["genres"].str.replace("|", " ", regex=False)
genre_matrix = TfidfVectorizer().fit_transform(movie_table["genres_text"])
movie_row = {int(movie_id): index for index, movie_id in enumerate(movie_table["movieId"])}
recommendations = {}
for user_id in evaluable["userId"].unique():
    seen = set(train.loc[train["userId"].eq(user_id), "movieId"])
    collaborative = model.top_n(user_id, seen, 100)
    collaborative["normalized"] = (collaborative["collaborative_score"] - collaborative["collaborative_score"].min()) / max(collaborative["collaborative_score"].max() - collaborative["collaborative_score"].min(), 1e-9)
    history = train.loc[train["userId"].eq(user_id) & train["movieId"].isin(movie_row), ["movieId", "rating"]]
    history_rows = [movie_row[int(item)] for item in history["movieId"]]
    weights = np.clip(history["rating"].to_numpy() - 2.5, 0.1, None)
    profile = genre_matrix[history_rows].multiply(weights[:, None]).sum(axis=0) / weights.sum()
    candidate_rows = [movie_row[int(item)] for item in collaborative["movieId"]]
    content_scores = cosine_similarity(profile, genre_matrix[candidate_rows]).ravel()
    content_min, content_max = content_scores.min(), content_scores.max()
    collaborative["content_score"] = (content_scores - content_min) / max(content_max - content_min, 1e-9)
    collaborative["hybrid_score"] = args.alpha * collaborative["normalized"] + (1 - args.alpha) * collaborative["content_score"]
    recommendations[int(user_id)] = collaborative.nlargest(10, "hybrid_score")["movieId"].astype(int).tolist()
metrics = {
    "rmse": float(rmse),
    "hit_rate_at_10": hit_rate_at_k(recommendations, evaluable, 10),
    "catalog_coverage": catalog_coverage(recommendations, len(model.items)),
    "train_seconds": train_seconds,
    "factors": args.factors,
    "epochs": args.epochs,
    "alpha": args.alpha,
    "cold_start_policy": "popularidad por cantidad y media; no personalizada",
}
(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports/hybrid_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
cold_start_items = popular_items[:10]
movies[movies["movieId"].isin(cold_start_items)].set_index("movieId").loc[cold_start_items].reset_index().to_csv(
    ROOT / "reports/cold_start_fallback.csv", index=False
)
print(json.dumps(metrics, indent=2))
