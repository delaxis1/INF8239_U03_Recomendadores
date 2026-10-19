"""Experimento completo del Ejercicio 05: no elegir modelos con la prueba."""

from __future__ import annotations
import os

os.environ.setdefault("MPLBACKEND", "Agg")
import hashlib, io, json, platform, sys, time, zipfile
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import requests
import sklearn
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.metrics import root_mean_squared_error
from threadpoolctl import threadpool_limits
from inf8239_u03.config import ROOT, MOVIELENS_DIR, MOVIELENS_URL
from inf8239_u03.data import load_movielens
from inf8239_u03.recommenders import MatrixFactorization, weighted_popularity
from inf8239_u03.protocol import (
    Scorer,
    global_time_split,
    evaluate_rankings,
    pareto_flags,
    select_config,
)

SEEDS = (42, 7, 2026)
METHODS = ("count", "popularity", "content", "mf10", "mf20", "hybrid25", "hybrid75")


def download():
    manifest = ROOT / "docs/data_source.json"
    zip_path = ROOT / "data/raw/ml-latest-small.zip"
    if not zip_path.exists():
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        sess = requests.Session()
        sess.mount(
            "https://",
            HTTPAdapter(
                max_retries=Retry(
                    total=3,
                    backoff_factor=1,
                    status_forcelist=[429, 500, 502, 503, 504],
                )
            ),
        )
        response = sess.get(MOVIELENS_URL, timeout=90)
        response.raise_for_status()
        zip_path.write_bytes(response.content)
    data = zip_path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if manifest.exists() and json.loads(manifest.read_text())["sha256"] != sha:
        raise ValueError(
            "MovieLens cambió: revisar fuente y actualizar huella explícitamente"
        )
    if not manifest.exists():
        manifest.write_text(
            json.dumps(
                {
                    "url": MOVIELENS_URL,
                    "sha256": sha,
                    "bytes": len(data),
                    "downloaded_utc": datetime.now(timezone.utc).isoformat(),
                    "license": "Términos README de GroupLens; no redistribución",
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    # Restaurar siempre desde el ZIP verificado: no confiar en CSV locales editados.
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        dest = (ROOT / "data/raw").resolve()
        for name in archive.namelist():
            if not (dest / name).resolve().is_relative_to(dest):
                raise ValueError("ZIP inseguro")
        archive.extractall(dest)
    return sha


def audit(r, m, sha):
    if r.empty or m.empty:
        raise ValueError("Dataset vacío")
    if r.isna().any().any() or m.isna().any().any():
        raise ValueError("Valores ausentes")
    if m.movieId.duplicated().any():
        raise ValueError("Clave de película duplicada")
    if r[["userId", "movieId"]].duplicated().any():
        raise ValueError("Interacción duplicada")
    if not (r.timestamp > 0).all():
        raise ValueError("Timestamp inválido")
    result = {
        "ratings": len(r),
        "users": r.userId.nunique(),
        "movies": len(m),
        "rated_movies": r.movieId.nunique(),
        "density": len(r) / (r.userId.nunique() * len(m)),
        "duplicates": int(r.duplicated().sum()),
        "date_min": str(pd.to_datetime(r.timestamp.min(), unit="s", utc=True)),
        "date_max": str(pd.to_datetime(r.timestamp.max(), unit="s", utc=True)),
        "sha256": sha,
    }
    return {k: int(v) if isinstance(v, np.integer) else v for k, v in result.items()}


def evaluate_phase(train, truth, movies, phase, model_dir):
    t = time.perf_counter()
    scorer = Scorer(train, movies)
    prep_s = time.perf_counter() - t
    users = sorted(truth.userId.unique())
    records = []
    user_tables = []
    all_rankings = {}
    models = {}
    for seed in SEEDS:
        mf_models = {}
        fit = {}
        for factors in (10, 20):
            t = time.perf_counter()
            mf_models[factors] = MatrixFactorization(factors=factors, seed=seed).fit(
                train, epochs=12
            )
            fit[factors] = time.perf_counter() - t
            print(
                phase,
                "seed",
                seed,
                "factors",
                factors,
                "fit",
                round(fit[factors], 2),
                flush=True,
            )
        for method in METHODS:
            mf = (
                None
                if method in ("count", "popularity", "content")
                else mf_models[10 if method == "mf10" else 20]
            )
            t = time.perf_counter()
            rankings = {int(u): scorer.recommend(u, method, mf, 10) for u in users}
            predict_s = time.perf_counter() - t
            metrics, per_user = evaluate_rankings(scorer, truth, rankings)
            baseline_pred = []
            for row in truth.itertuples():
                if (
                    mf is not None
                    and row.userId in mf.user_index
                    and row.movieId in mf.item_index
                ):
                    baseline_pred.append(
                        np.clip(mf.predict(row.userId, row.movieId), 0.5, 5.0)
                    )
                else:
                    baseline_pred.append(scorer.item_means.get(row.movieId, scorer.mu))
            rmse = float(root_mean_squared_error(truth.rating, baseline_pred))
            artifact = model_dir / f"{phase}_{method}_seed{seed}.joblib"
            joblib.dump(
                {
                    "method": method,
                    "mf": mf,
                    "content": scorer.content
                    if method in ("content", "hybrid25", "hybrid75")
                    else None,
                    "popularity": scorer.pop,
                    "item_ids": scorer.ids,
                },
                artifact,
                compress=3,
            )
            metrics.update(
                model=method,
                seed=seed,
                phase=phase,
                fit_s=prep_s
                + (fit[10 if method == "mf10" else 20] if mf is not None else 0),
                predict_ms_per_user=1000 * predict_s / len(users),
                size_kib=artifact.stat().st_size / 1024,
                rmse=rmse,
            )
            records.append(metrics)
            per_user["model"] = method
            per_user["seed"] = seed
            per_user["phase"] = phase
            user_tables.append(per_user)
            if seed == 42:
                all_rankings[method] = rankings
        if phase == "test" and seed == 42:
            models = mf_models
    return (
        pd.DataFrame(records),
        pd.concat(user_tables, ignore_index=True),
        scorer,
        models,
        all_rankings,
    )


def figures(summary, valid, scorer, rankings):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
        }
    )
    out = ROOT / "reports/figures"
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    ss = summary.sort_values("ndcg")
    for ax, col, title in [
        (axes[0], "ndcg", "NDCG@10 en prueba"),
        (axes[1], "coverage", "Cobertura del catálogo"),
    ]:
        ax.barh(ss.index, ss[col], color="#2a78d6")
        ax.set_title(title)
        ax.set_xlim(left=0)
        ax.grid(axis="x", color="#ededed", zorder=0)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        if col == "coverage":
            ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    fig.savefig(out / "ranking_coverage.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for key, row in summary.iterrows():
        ax.scatter(
            row.fit_s, row.ndcg, color="#2a78d6" if row.pareto else "#999999", s=65
        )
        ax.annotate(
            key,
            (row.fit_s, row.ndcg),
            textcoords="offset points",
            xytext=(6, 8 if key not in ("hybrid25", "mf20") else -13),
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Ajuste mediano en segundos (escala log)")
    ax.set_ylabel("NDCG@10")
    ax.set_title("Pareto en prueba: utilidad y costo; elección fijada en validación")
    fig.savefig(out / "pareto.png")
    plt.close(fig)
    val = valid.groupby("model").agg(
        ndcg=("ndcg", "mean"), coverage=("coverage", "mean")
    )
    fig, ax = plt.subplots(figsize=(7, 3.8), layout="constrained")
    keys = list(summary.index)
    x = np.arange(len(keys))
    ax.bar(
        x - 0.17, val.loc[keys, "ndcg"], width=0.34, label="Validación", color="#2a78d6"
    )
    ax.bar(x + 0.17, summary.ndcg, width=0.34, label="Prueba", color="#eb6834")
    ax.set_xticks(x, keys, rotation=35, ha="right")
    ax.set_ylabel("NDCG@10")
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.savefig(out / "validation_test.png")
    plt.close(fig)


def main():
    start = time.perf_counter()
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    sha = download()
    ratings, movies = load_movielens(MOVIELENS_DIR)
    dataset = audit(ratings, movies, sha)
    train, valid, test, cuts = global_time_split(ratings)
    assignments = np.where(
        ratings.timestamp <= cuts["train_end"],
        "train",
        np.where(ratings.timestamp <= cuts["validation_end"], "validation", "test"),
    )
    index_frame = pd.DataFrame({"row": ratings.index, "partition": assignments})
    # Índices originales y datos quedan locales, no se redistribuyen en GitHub.
    index_frame.to_csv(ROOT / "data/raw/partitions.csv", index=False)
    split_hash = hashlib.sha256(index_frame.to_csv(index=False).encode()).hexdigest()
    (reports / "audit.json").write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with threadpool_limits(limits=1):
        val, vu, _, _, _ = evaluate_phase(
            train, valid, movies, "validation", models_dir
        )
        val.to_csv(reports / "validation_runs.csv", index=False)
        choice = select_config(val)
        (reports / "decision_locked.json").write_text(
            json.dumps(
                {
                    "chosen": choice,
                    "rule": "NDCG a .005 del máximo; mayor cobertura, menor costo",
                    "source": "validation_runs.csv",
                    "locked_before_test": True,
                    "locked_utc": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print("ELECCIÓN FIJADA ANTES DE PRUEBA:", choice, flush=True)
        final_train = pd.concat([train, valid], ignore_index=True)
        results, users, scorer, mfs, recs = evaluate_phase(
            final_train, test, movies, "test", models_dir
        )
    results.to_csv(reports / "test_runs.csv", index=False)
    users.to_csv(reports / "user_metrics.csv", index=False)
    summary = results.groupby("model").agg(
        ndcg=("ndcg", "mean"),
        ndcg_std=("ndcg", "std"),
        hit=("hit", "mean"),
        precision=("precision", "mean"),
        recall=("recall", "mean"),
        coverage=("coverage", "mean"),
        diversity=("diversity", "mean"),
        novelty=("novelty", "mean"),
        rmse=("rmse", "mean"),
        fit_s=("fit_s", "median"),
        predict_ms_per_user=("predict_ms_per_user", "median"),
        size_kib=("size_kib", "median"),
        users=("users", "first"),
        users_positive=("users_positive", "first"),
        users_no_positive=("users_no_positive", "first"),
        cold_users=("cold_users", "first"),
        unreachable_positives=("unreachable_positives", "first"),
    )
    summary["pareto"] = pareto_flags(summary)
    summary["selected"] = summary.index == choice
    summary.to_csv(reports / "main_results.csv")
    sizes = final_train.groupby("userId").size()
    profiles = {
        "amplio": int(sizes.idxmax()),
        "pequeño": int(sizes.idxmin()),
        "nuevo": int(ratings.userId.max() + 1),
    }
    rows = []
    mf = mfs[10 if choice == "mf10" else 20]
    for label, uid in profiles.items():
        for rank, item in enumerate(scorer.recommend(uid, choice, mf, 10), 1):
            movie = movies.loc[movies.movieId == item].iloc[0]
            rows.append(
                {
                    "profile": label,
                    "user": uid,
                    "rank": rank,
                    "movieId": item,
                    "title": movie.title,
                    "genres": movie.genres,
                    "history": int(sizes.get(uid, 0)),
                    "fallback": uid not in scorer.seen,
                }
            )
    pd.DataFrame(rows).to_csv(reports / "profiles.csv", index=False)
    weighted_popularity(final_train, movies).head(10).to_csv(
        reports / "popularity_top10.csv"
    )
    query_rows = []
    for title in (
        "Toy Story (1995)",
        "Shawshank Redemption, The (1994)",
        "Alien (1979)",
    ):
        if title in scorer.movies.title.values:
            r = scorer.content.recommend(title, 10)
            r["query"] = title
            query_rows.append(r)
    pd.concat(query_rows).to_csv(reports / "content_queries.csv", index=False)
    figures(summary, val, scorer, recs)
    # Segmentos separados: no ocultar usuarios completamente nuevos.
    users["segment"] = np.where(users.cold, "frio", "con_historial")
    users.groupby(["model", "segment"])[
        ["ndcg", "hit", "precision", "recall"]
    ].mean().to_csv(reports / "segments.csv")
    joblib.dump(
        {"scorer": scorer, "mf": mf, "choice": choice},
        models_dir / "serving.joblib",
        compress=3,
    )
    meta = {
        "dataset": dataset,
        "split": {
            "train": len(train),
            "validation": len(valid),
            "test": len(test),
            "cuts": cuts,
            "assignment_sha256": split_hash,
        },
        "seeds": list(SEEDS),
        "epochs": 12,
        "factors": [10, 20],
        "relevance": ">=4 estrellas",
        "k": 10,
        "chosen": choice,
        "candidate_policy": "películas valoradas antes del corte; excluir vistas; positivos fuera de catálogo cuentan como fallos",
        "python": sys.version.split()[0],
        "sklearn": sklearn.__version__,
        "system": platform.platform(),
        "seconds": time.perf_counter() - start,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (reports / "metrics.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary.round(4).to_string(), flush=True)
    print("DURACIÓN:", round(meta["seconds"], 1), "s", flush=True)


if __name__ == "__main__":
    main()
