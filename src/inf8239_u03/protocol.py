"""Evaluación temporal, candidatos comunes y selección independiente de prueba."""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from .recommenders import ContentRecommender


def global_time_split(ratings):
    if len(ratings) < 10:
        raise ValueError("Se necesitan al menos diez interacciones")
    c1 = int(ratings.timestamp.quantile(0.8))
    c2 = int(ratings.timestamp.quantile(0.9))
    train = ratings.loc[ratings.timestamp <= c1].copy()
    valid = ratings.loc[(ratings.timestamp > c1) & (ratings.timestamp <= c2)].copy()
    test = ratings.loc[ratings.timestamp > c2].copy()
    if any(f.empty for f in (train, valid, test)):
        raise ValueError("Cortes temporales vacíos")
    return train, valid, test, {"train_end": c1, "validation_end": c2}


def pareto_flags(frame, benefit="ndcg", costs=("fit_s",)):
    a = frame[[benefit, *costs]].to_numpy(float)
    if not np.isfinite(a).all():
        raise ValueError("Pareto necesita valores finitos")
    a[:, 0] *= -1
    return np.array(
        [not np.any(np.all(a <= row, axis=1) & np.any(a < row, axis=1)) for row in a]
    )


def select_config(validation, tolerance=0.005):
    if validation.empty:
        raise ValueError("Validación vacía")
    # Regla previa: retener NDCG dentro de .005 del mejor; mayor cobertura,
    # luego menor costo. Ninguna métrica de prueba se usa en esta función.
    table = validation.groupby("model", as_index=False).agg(
        ndcg=("ndcg", "mean"), coverage=("coverage", "mean"), fit_s=("fit_s", "median")
    )
    eligible = table.loc[table.ndcg >= table.ndcg.max() - tolerance]
    return str(
        eligible.sort_values(
            ["coverage", "fit_s", "model"], ascending=[False, True, True]
        ).iloc[0]["model"]
    )


class Scorer:
    """Mismo catálogo observado antes del corte para todos los modelos."""

    def __init__(self, train, movies):
        self.train = train
        self.ids = np.sort(train.movieId.unique()).astype(int)
        self.id_to_col = {int(v): i for i, v in enumerate(self.ids)}
        self.movies = movies.set_index("movieId").loc[self.ids].reset_index()
        self.content = ContentRecommender().fit(self.movies)
        stats = train.groupby("movieId").rating.agg(["mean", "count"]).reindex(self.ids)
        m = float(stats["count"].quantile(0.8))
        mu = float(train.rating.mean())
        self.pop = (
            (stats["count"] * stats["mean"] + m * mu) / (stats["count"] + m)
        ).to_numpy()
        self.count = stats["count"].to_numpy(float)
        self.item_means = stats["mean"].to_dict()
        self.mu = mu
        self.seen = {
            int(u): set(g.movieId.astype(int)) for u, g in train.groupby("userId")
        }
        self.histories = {int(u): g for u, g in train.groupby("userId")}
        self.genres = [
            set(g.split("|")) - {"(no genres listed)"} for g in self.movies.genres
        ]

    def scores(self, user, model, mf=None):
        u = int(user)
        if model in ("popularity", "count") or u not in self.histories:
            return self.count.copy() if model == "count" else self.pop.copy()
        h = self.histories[u]
        rows = [self.id_to_col[int(x)] for x in h.movieId]
        weights = np.clip(h.rating.to_numpy() - 2.5, 0.1, None)
        profile = (
            np.asarray(self.content.matrix[rows].multiply(weights[:, None]).sum(axis=0))
            / weights.sum()
        )
        content = cosine_similarity(profile, self.content.matrix).ravel()
        if model == "content":
            return content
        if mf is None or u not in mf.user_index:
            return self.pop.copy()
        cols = [mf.item_index[int(i)] for i in self.ids]
        cf = mf.global_mean + mf.item_factors[cols] @ mf.user_factors[mf.user_index[u]]
        if model.startswith("mf"):
            return cf
        alpha = 0.25 if model == "hybrid25" else 0.75

        def norm(x):
            return (x - x.min()) / max(float(np.ptp(x)), 1e-12)

        return alpha * norm(cf) + (1 - alpha) * norm(content)

    def recommend(self, user, model, mf=None, k=10):
        if k < 1:
            raise ValueError("k debe ser positivo")
        scores = self.scores(user, model, mf)
        seen = self.seen.get(int(user), set())
        mask = np.array([int(i) not in seen for i in self.ids])
        ids = self.ids[mask]
        values = scores[mask]
        order = np.lexsort((ids, -values))[:k]
        return ids[order].astype(int).tolist()


def user_metrics(ranked, relevant, k=10):
    if k < 1:
        raise ValueError("k debe ser positivo")
    if len(ranked) != len(set(ranked)):
        raise ValueError("Ranking duplicado")
    if not relevant:
        return {"precision": np.nan, "recall": np.nan, "ndcg": np.nan, "hit": np.nan}
    hits = np.array([x in relevant for x in ranked[:k]], float)
    dcg = np.sum(hits / np.log2(np.arange(len(hits)) + 2))
    ideal = np.sum(1 / np.log2(np.arange(min(k, len(relevant))) + 2))
    return {
        "precision": float(hits.sum() / k),
        "recall": float(hits.sum() / len(relevant)),
        "ndcg": float(dcg / ideal),
        "hit": float(hits.any()),
    }


def evaluate_rankings(scorer, truth, rankings, k=10):
    # Relevante: rating >=4. Se conserva como fallo el relevante fuera del
    # catálogo de entrenamiento. Usuarios sin positivos se reportan aparte.
    rows = []
    novel = []
    diversity = []
    counts = scorer.train.movieId.value_counts()
    total = len(scorer.train)
    for user, g in truth.groupby("userId"):
        relevant = set(g.loc[g.rating >= 4, "movieId"].astype(int)) - scorer.seen.get(
            int(user), set()
        )
        ranked = rankings[int(user)]
        m = user_metrics(ranked, relevant, k)
        m.update(
            user=int(user),
            relevant=len(relevant),
            reachable=len(relevant & set(scorer.ids)),
            cold=int(user) not in scorer.seen,
        )
        rows.append(m)
        if ranked:
            novel.append(
                float(np.mean([-np.log2(counts[int(i)] / total) for i in ranked]))
            )
            gs = [scorer.genres[scorer.id_to_col[int(i)]] for i in ranked]
            pairs = [
                1 - len(a & b) / max(len(a | b), 1)
                for j, a in enumerate(gs)
                for b in gs[j + 1 :]
            ]
            diversity.append(float(np.mean(pairs)) if pairs else 0.0)
    per_user = pd.DataFrame(rows)
    eligible = per_user.loc[per_user.relevant > 0]
    result = {
        x: float(eligible[x].mean()) if len(eligible) else 0.0
        for x in ("precision", "recall", "ndcg", "hit")
    }
    result.update(
        users=len(per_user),
        users_positive=len(eligible),
        users_no_positive=int((per_user.relevant == 0).sum()),
        cold_users=int(per_user.cold.sum()),
        unreachable_positives=int((per_user.relevant - per_user.reachable).sum()),
        coverage=len(set(i for r in rankings.values() for i in r)) / len(scorer.ids),
        diversity=float(np.mean(diversity)),
        novelty=float(np.mean(novel)),
    )
    return result, per_user
