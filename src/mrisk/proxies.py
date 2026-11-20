"""Measure the paper's five model-risk proxies (Table 5) for one model.

* stability: spread of test AUC (and of individual PDs) across retrains on
  random 60-100% subsamples of the training rows (p. 12, p. 18);
* training latency: seconds to fit on the full training rows (p. 18);
* SHAP latency: seconds to compute SHAP values for a fixed set of rows (p. 19).

The hyperparameter count and the transparency class are definitions; they
live in ``models.py``.
"""
from __future__ import annotations

import os
import time
import warnings

import numpy as np

from .models import make_model, predict_pd


def single_thread() -> None:
    """Pin BLAS/OpenMP to one thread so timings do not depend on idle cores."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"


def time_fit(name: str, x, y, lasso_c: float, repeats: int = 5, seed: int = 0) -> dict:
    """Median and spread of fit time over ``repeats`` fits, after one warm-up fit."""
    make_model(name, seed, lasso_c).fit(x, y)  # warm-up: imports, caches, page faults
    seconds = []
    for r in range(repeats):
        model = make_model(name, seed + r, lasso_c)
        start = time.perf_counter()
        model.fit(x, y)
        seconds.append(time.perf_counter() - start)
    return {"median": float(np.median(seconds)), "q25": float(np.percentile(seconds, 25)),
            "q75": float(np.percentile(seconds, 75)), "runs": seconds}


def subsample_fraction(rng: np.random.Generator, low: float = 0.60, high: float = 1.00) -> float:
    return float(rng.uniform(low, high))


def stability_run(name: str, x_train, y_train, x_test, lasso_c: float, run: int) -> tuple[float, np.ndarray]:
    """One retrain on a random 60-100% subsample; returns the subsample fraction and test PDs.

    Seeds depend only on ``run``, so the 100 subsamples are identical across
    models: differences in spread come from the models, not from the draws.
    """
    single_thread()
    rng = np.random.default_rng(10_000 + run)
    frac = subsample_fraction(rng)
    rows = rng.choice(len(y_train), size=int(round(frac * len(y_train))), replace=False)
    model = make_model(name, seed=run, lasso_c=lasso_c)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_train[rows], y_train[rows])
    return frac, predict_pd(model, x_test).astype(np.float32)


def shap_values(name: str, model, x_background: np.ndarray, x_explain: np.ndarray,
                common: bool = False) -> tuple[np.ndarray, float]:
    """SHAP values and the seconds they took.

    ``common=False`` (reproduction) uses each library's default explainer, as
    the paper's figures suggest: path-dependent TreeSHAP for trees, whose
    output is the probability for RF and CART but the log-odds for XGBoost
    (the paper's Fig. 7 axis runs from -1 to 3), and the model-agnostic
    permutation explainer on the probability for the MLP and linear models.

    ``common=True`` (improvement) gives every model the same permutation
    explainer, background rows and probability scale, so that differences in
    time and in rankings come from the models, not from the explainers.
    """
    import shap

    def f(x):
        return predict_pd(model, x)

    start = time.perf_counter()
    if common or name not in ("CART", "RF", "XGBoost"):
        explainer = shap.PermutationExplainer(f, shap.maskers.Independent(x_background))
        values = explainer(x_explain, silent=True).values
    else:
        values = shap.TreeExplainer(model).shap_values(x_explain, check_additivity=False)
        if isinstance(values, list):  # one array per class
            values = values[1]
        if np.ndim(values) == 3:  # (rows, features, classes)
            values = values[:, :, 1]
    return np.asarray(values), time.perf_counter() - start


def importance_by_feature(values: np.ndarray, columns: list[str], base_name) -> dict:
    """Mean |SHAP| per borrower characteristic.

    SHAP values are additive, so the contributions of x and x² are summed per
    row before taking the absolute value: the result is the importance of the
    characteristic, as in the paper's Figs. 6-8, which show ten features.
    """
    out: dict[str, np.ndarray] = {}
    for j, col in enumerate(columns):
        out.setdefault(base_name(col), np.zeros(values.shape[0]))
        out[base_name(col)] += values[:, j]
    return {k: float(np.abs(v).mean()) for k, v in out.items()}
