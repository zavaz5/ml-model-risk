from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from mrisk.data import add_squares, base_name, make_split
from mrisk.uncertainty import auc_ci, auc_difference, benjamini_hochberg, holm, permutation_test


def test_delong_auc_equals_sklearn_and_float32_is_safe():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    p = (rng.random(500) + 0.4 * y).astype(np.float32)
    assert auc_ci(y, p)["auc"] == pytest.approx(roc_auc_score(y, p), abs=1e-12)


def test_delong_paired_difference_of_identical_scores_is_zero():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    p = rng.random(300) + 0.3 * y
    d = auc_difference(y, p, p)
    assert d["diff"] == 0 and d["se"] == pytest.approx(0, abs=1e-12)


def test_delong_se_matches_bootstrap_roughly():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 2000)
    p = rng.random(2000) + 0.5 * y
    se = auc_ci(y, p)["se"]
    boot = [roc_auc_score(y[k], p[k]) for k in (rng.integers(0, 2000, 2000) for _ in range(400))]
    assert se == pytest.approx(np.std(boot), rel=0.15)


def test_multiple_testing_corrections():
    p = [0.01, 0.04, 0.03, 0.20]
    assert holm(p) == pytest.approx([0.04, 0.09, 0.09, 0.20])
    assert benjamini_hochberg(p) == pytest.approx([0.04, 0.0533333, 0.0533333, 0.20], rel=1e-5)


def test_permutation_test_detects_a_clear_difference():
    assert permutation_test(np.arange(10) + 10.0, np.arange(10) * 1.0, reps=2000) < 0.01


def test_squares_and_base_names():
    x = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, -1.0]})
    sq = add_squares(x)
    assert list(sq.columns) == ["a", "b", "a_sq", "b_sq"]
    assert sq["b_sq"].tolist() == [9.0, 1.0]
    assert base_name("a_sq") == "a" and base_name("a") == "a"


def test_split_is_stratified_and_disjoint():
    y = np.array([0] * 900 + [1] * 100)
    s = make_split(y)
    assert len(s.test) == 200 and len(np.intersect1d(s.train, s.test)) == 0
    assert y[s.test].mean() == pytest.approx(0.1, abs=0.01)
    assert set(s.fit) | set(s.valid) == set(s.train)


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "data/raw/gmsc/cs-training.csv").exists(),
                    reason="Kaggle data not downloaded")
def test_gmsc_complete_cases_match_the_papers_count():
    from mrisk.data import GMSC_FEATURES, load_gmsc

    frame = load_gmsc()
    assert len(frame) == 120_269  # the paper: "120,000 granted loans"
    assert set(GMSC_FEATURES) <= set(frame.columns)
