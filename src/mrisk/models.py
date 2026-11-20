"""The six classifiers of the paper's empirical example, with its settings.

Hyperparameters are the ones the paper reports after its own 5-fold
cross-validation (p. 17). Where the paper is silent we use library defaults
and say so in DEVIATIONS.md. Every model runs single-threaded so that
training times are comparable across models (CLAUDE.md, timing rule).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .paper_numbers import HYPERPARAMETERS

MODEL_NAMES = ["Logit", "Lasso", "CART", "RF", "XGBoost", "MLP"]
LASSO_GRID = np.logspace(-4, 1, 11)  # inverse penalty strength C; the paper does not give its grid

# The paper's own count of hyperparameters (Table 5, p. 18). It is a
# definition, not a measurement, so it is reproduced as stated.
PAPER_HYPERPARAMETER_COUNT = {"Lasso": 0, "CART": 1, "RF": 2, "XGBoost": 2, "MLP": 5}

# Transparency class (p. 18-19): parametric < tree < tree ensemble < deep learning.
TRANSPARENCY = {"Logit": 1, "Lasso": 1, "CART": 2, "RF": 3, "XGBoost": 3, "MLP": 5}


def make_model(name: str, seed: int = 0, lasso_c: float = 1.0):
    """Return an unfitted estimator with the paper's settings.

    Linear models and the MLP see standardised inputs: the squared features
    span many orders of magnitude, and neither the solver nor Adam converges
    on raw values. Trees are scale-invariant and get the raw design.
    """
    if name == "Logit":  # unpenalised logistic regression, the paper's benchmark
        return make_pipeline(StandardScaler(), LogisticRegression(C=np.inf, max_iter=5000))
    if name == "Lasso":
        # saga, not liblinear: with the heavy-tailed squared features liblinear's L1
        # solver needs over 90 s per fit at C = 1; saga converges in about 18 s.
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=lasso_c, l1_ratio=1.0, solver="saga", max_iter=2000, random_state=seed))
    if name == "CART":
        return DecisionTreeClassifier(**HYPERPARAMETERS["CART"], random_state=seed)
    if name == "RF":
        return RandomForestClassifier(**HYPERPARAMETERS["RF"], n_jobs=1, random_state=seed)
    if name == "XGBoost":
        return XGBClassifier(**HYPERPARAMETERS["XGBoost"], n_jobs=1, random_state=seed,
                             tree_method="hist", eval_metric="logloss")
    if name == "MLP":
        # The paper does not report epochs; early stopping on 10% of the
        # training rows keeps the fit near the paper's 31 s (DEVIATIONS.md).
        return make_pipeline(StandardScaler(), MLPClassifier(
            **HYPERPARAMETERS["MLP"], early_stopping=True, max_iter=200, random_state=seed))
    raise ValueError(name)


def _fold_auc(x, y, train, test, c, seed) -> float:
    from sklearn.metrics import roc_auc_score

    model = make_model("Lasso", seed, c).fit(x[train], y[train])
    return roc_auc_score(y[test], predict_pd(model, x[test]))


def choose_lasso_c(x: np.ndarray, y: np.ndarray, seed: int = 0, n_jobs: int = 14) -> float:
    """Pick the L1 penalty by 5-fold cross-validated AUC on the training rows, as the paper does.

    The 55 (penalty, fold) fits run in parallel: this step is not timed.
    """
    from joblib import Parallel, delayed

    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(x, y))
    jobs = [(c, tr, te) for c in LASSO_GRID for tr, te in folds]
    aucs = Parallel(n_jobs=n_jobs)(delayed(_fold_auc)(x, y, tr, te, c, seed) for c, tr, te in jobs)
    scores = np.array(aucs).reshape(len(LASSO_GRID), len(folds)).mean(axis=1)
    return float(LASSO_GRID[int(np.argmax(scores))])


def predict_pd(model, x) -> np.ndarray:
    """Probability of default (class 1)."""
    return model.predict_proba(x)[:, 1]


def fitted_parameter_count(name: str, model) -> int:
    """Number of quantities the fit estimates: an objective size measure (improvement I-8).

    Linear models: non-zero coefficients plus intercept. Trees: one value per
    leaf plus one threshold per split. MLP: all weights and biases.
    """
    est = model[-1] if hasattr(model, "steps") else model
    if name in ("Logit", "Lasso"):
        return int(np.count_nonzero(est.coef_) + 1)
    if name == "CART":
        return int(est.tree_.node_count)
    if name == "RF":
        return int(sum(t.tree_.node_count for t in est.estimators_))
    if name == "XGBoost":
        dump = est.get_booster().get_dump()
        return int(sum(line.count("leaf=") + line.count("[f") for line in dump))
    if name == "MLP":
        return int(sum(w.size for w in est.coefs_) + sum(b.size for b in est.intercepts_))
    raise ValueError(name)
