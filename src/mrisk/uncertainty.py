"""Standard errors, intervals and tests used for every "±" in the project.

AUC: DeLong et al. (1988) variance, in the fast form of Sun and Xu (2014).
It is exact for the empirical AUC and gives a paired test for two models
scored on the same rows, which a bootstrap of each model separately cannot.
Everything else: a nonparametric bootstrap with a fixed seed.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

BOOT_REPS = 1000
BOOT_SEED = 0


def _midranks(x: np.ndarray) -> np.ndarray:
    return stats.rankdata(x, method="average")


def _structural_components(y: np.ndarray, scores: np.ndarray):
    """DeLong's placement values V10 (per positive) and V01 (per negative) for each score row."""
    pos, neg = scores[:, y == 1], scores[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    v10, v01, aucs = [], [], []
    for p, q in zip(pos, neg):
        r_all = _midranks(np.concatenate([p, q]))
        r_pos, r_neg = _midranks(p), _midranks(q)
        auc = (r_all[:m].sum() - m * (m + 1) / 2) / (m * n)
        v10.append((r_all[:m] - r_pos) / n)
        v01.append(1 - (r_all[m:] - r_neg) / m)
        aucs.append(auc)
    return np.array(aucs), np.array(v10), np.array(v01)


def delong(y: np.ndarray, *score_sets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """AUCs and their DeLong covariance matrix for models scored on the same rows."""
    scores = np.vstack(score_sets).astype(np.float64)  # float32 PDs would round the ranks
    aucs, v10, v01 = _structural_components(np.asarray(y), scores)
    m, n = v10.shape[1], v01.shape[1]
    cov = np.atleast_2d(np.cov(v10)) / m + np.atleast_2d(np.cov(v01)) / n
    return aucs, cov


def auc_ci(y: np.ndarray, score: np.ndarray, level: float = 0.95) -> dict:
    """AUC with DeLong standard error and a normal interval."""
    auc, cov = delong(y, score)
    se = float(np.sqrt(cov[0, 0]))
    z = stats.norm.ppf(0.5 + level / 2)
    return {"auc": float(auc[0]), "se": se, "lo": float(auc[0] - z * se), "hi": float(auc[0] + z * se)}


def auc_difference(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray) -> dict:
    """Paired DeLong test of AUC(a) - AUC(b) on the same rows."""
    aucs, cov = delong(y, score_a, score_b)
    diff = float(aucs[0] - aucs[1])
    se = float(np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]))
    z = diff / se if se > 0 else np.inf
    return {"diff": diff, "se": se, "lo": diff - 1.96 * se, "hi": diff + 1.96 * se,
            "p": float(2 * stats.norm.sf(abs(z)))}


def bootstrap(values: np.ndarray, statistic, reps: int = BOOT_REPS, seed: int = BOOT_SEED) -> dict:
    """Bootstrap standard error and 95% percentile interval of ``statistic(values)``."""
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    draws = np.array([statistic(values[rng.integers(0, len(values), len(values))]) for _ in range(reps)])
    return {"est": float(statistic(values)), "se": float(draws.std(ddof=1)),
            "lo": float(np.percentile(draws, 2.5)), "hi": float(np.percentile(draws, 97.5))}


def welch(a: np.ndarray, b: np.ndarray) -> dict:
    """Welch t-test (unequal variances, Welch-Satterthwaite degrees of freedom)."""
    res = stats.ttest_ind(a, b, equal_var=False)
    return {"t": float(res.statistic), "df": float(res.df), "p": float(res.pvalue)}


def permutation_test(a: np.ndarray, b: np.ndarray, reps: int = 10_000, seed: int = BOOT_SEED) -> float:
    """Two-sided permutation p-value for a difference in means; exact in spirit for tiny samples."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled, k = np.concatenate([a, b]), len(a)
    observed = abs(a.mean() - b.mean())
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(reps):
        perm = rng.permutation(pooled)
        hits += abs(perm[:k].mean() - perm[k:].mean()) >= observed - 1e-15
    return (hits + 1) / (reps + 1)


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (family-wise error control)."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (false discovery rate)."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)[::-1]
    adjusted = np.empty_like(p)
    running = 1.0
    for rank, idx in enumerate(order):
        running = min(running, p[idx] * len(p) / (len(p) - rank))
        adjusted[idx] = running
    return adjusted.tolist()
