import numpy as np
import pandas as pd
import pytest
from inf8239_u03.protocol import (
    Scorer,
    global_time_split,
    user_metrics,
    evaluate_rankings,
    pareto_flags,
    select_config,
)


def test_global_split_has_no_future():
    r = pd.DataFrame({"timestamp": range(100), "userId": [1] * 100})
    tr, va, te, _ = global_time_split(r)
    assert (
        tr.timestamp.max()
        < va.timestamp.min()
        <= va.timestamp.max()
        < te.timestamp.min()
    )
    assert len(tr) + len(va) + len(te) == len(r)


def test_tied_timestamps_stay_together():
    r = pd.DataFrame({"timestamp": np.repeat(range(20), 5)})
    tr, va, te, _ = global_time_split(r)
    assert not set(tr.timestamp) & set(va.timestamp)
    assert not set(va.timestamp) & set(te.timestamp)


def test_metrics_known_example():
    m = user_metrics([1, 2, 3, 4, 5], {1, 3, 6}, 5)
    assert m["precision"] == 0.4 and m["recall"] == 2 / 3
    assert m["ndcg"] == pytest.approx(1.5 / (1 + 1 / np.log2(3) + 0.5))


def test_no_positive_is_not_a_failure():
    assert np.isnan(user_metrics([1], set(), 10)["ndcg"])


def test_duplicate_rank_is_rejected():
    with pytest.raises(ValueError):
        user_metrics([1, 1], {1})


def test_cold_user_fallback_and_seen_exclusion(sample_ratings, sample_movies):
    s = Scorer(sample_ratings, sample_movies)
    assert s.recommend(999, "hybrid75") == s.recommend(999, "popularity")
    assert not set(s.recommend(1, "content")) & s.seen[1]
    assert len(s.recommend(1, "content")) == len(s.ids) - len(s.seen[1])


def test_unknown_positive_is_kept_as_miss(sample_ratings, sample_movies):
    s = Scorer(sample_ratings, sample_movies)
    truth = pd.DataFrame(
        {"userId": [99, 100], "movieId": [999, 2], "rating": [5.0, 2.0]}
    )
    result, users = evaluate_rankings(s, truth, {99: [1, 2], 100: [2, 3]}, 10)
    assert result["users_no_positive"] == 1
    assert result["unreachable_positives"] == 1
    assert result["ndcg"] == 0
    assert result["cold_users"] == 2


def test_pareto_tradeoff_and_duplicate_points():
    f = pd.DataFrame({"ndcg": [0.1, 0.2, 0.15, 0.2], "fit_s": [1, 2, 3, 2]})
    assert pareto_flags(f).tolist() == [True, True, False, True]


def test_selector_uses_coverage_within_fixed_tolerance():
    f = pd.DataFrame(
        {
            "model": ["a", "b", "c"],
            "ndcg": [0.1, 0.097, 0.09],
            "coverage": [0.1, 0.8, 0.9],
            "fit_s": [1, 3, 1],
        }
    )
    assert select_config(f) == "b"
