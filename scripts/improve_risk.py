"""I13: a rule instead of expert scores, and the uncertainty of the whole scorecard.

The paper turns its measured proxies into darkness scores Z by expert
judgement (p. 18) and draws one risk-reward picture without uncertainty
(Fig. 5). Here every proxy is mapped to [1, 5] by min-max across the five
models (log scale for times, linear for the AUC spread and the ordinal
counts), a category's Z is the mean of its proxies, and a Monte Carlo over
every input that has sampling noise asks how often each model is
risk-efficient (not dominated in gain and risk). Decision rule:
docs/preregistration.md, I13.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from mrisk.corpus import CATEGORIES, USES
from mrisk.models import PAPER_HYPERPARAMETER_COUNT, TRANSPARENCY
from mrisk.uncertainty import BOOT_REPS, BOOT_SEED

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "results" / "reproduction"
OUT = ROOT / "results" / "improvements" / "risk"
MODELS = ["Lasso", "CART", "RF", "XGBoost", "MLP"]


def minmax(values: np.ndarray, log: bool) -> np.ndarray:
    """Map the five models' values to [1, 5]: 1 for the least risky, 5 for the most."""
    v = np.log10(values) if log else np.asarray(values, float)
    span = v.max() - v.min()
    return np.ones_like(v) * 3 if span == 0 else 1 + 4 * (v - v.min()) / span


def rule_scores(auc_sd, n_hyper, fit_seconds, transparency, shap_seconds) -> dict[str, np.ndarray]:
    """Darkness scores by rule; arrays are ordered as MODELS."""
    return {
        "statistics": (minmax(auc_sd, log=False) + minmax(n_hyper, log=False)) / 2,
        "technology": (minmax(fit_seconds, log=True) + minmax(transparency, log=False)) / 2,
        "conduct": minmax(shap_seconds, log=True),
    }


def normalised_risk(z: dict, weights: np.ndarray) -> np.ndarray:
    """Σ_i H̄_ij Z_im / (5 Σ_i H̄_ij): one scale for every use, in [0.2, 1] (uses × models)."""
    zz = np.vstack([z[c] for c in CATEGORIES])  # categories × models
    return (weights @ zz) / (5 * weights.sum(axis=1, keepdims=True))


def frontier(gain: np.ndarray, risk: np.ndarray) -> np.ndarray:
    """True for models that no other model beats on both gain (≥) and risk (≤), one strictly."""
    on = np.ones(len(gain), bool)
    for m in range(len(gain)):
        for k in range(len(gain)):
            if k != m and gain[k] >= gain[m] and risk[k] <= risk[m] and (gain[k] > gain[m] or risk[k] < risk[m]):
                on[m] = False
    return on


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prox = pd.read_csv(REPRO / "models" / "proxies.csv").set_index("model").loc[MODELS]
    shap_default = pd.read_csv(REPRO / "models" / "shap_default.csv").set_index("model").loc[MODELS, "seconds"]
    weights_point = pd.read_csv(REPRO / "nlp" / "table2.csv", index_col=0).loc[USES, CATEGORIES].to_numpy()
    weight_draws = np.load(ROOT / "results" / "improvements" / "nlp" / "i3_weight_draws.npy")[:, :, :3]
    aucs = {m: np.load(REPRO / "models" / f"stability_aucs_{m}.npy") for m in MODELS}
    test_pd = np.load(ROOT / "results" / "improvements" / "models" / "test_pd.npy")  # rows: Logit + MODELS
    y_test = np.load(ROOT / "results" / "improvements" / "models" / "y_test.npy")

    n_hyper = np.array([PAPER_HYPERPARAMETER_COUNT[m] for m in MODELS], float)
    transp = np.array([TRANSPARENCY[m] for m in MODELS], float)
    z_point = rule_scores(prox["auc_sd"].to_numpy(), n_hyper, prox["fit_seconds"].to_numpy(), transp,
                          shap_default.to_numpy())
    auc_point = np.array([roc_auc_score(y_test, p) for p in test_pd])
    gain_point = auc_point[1:] - auc_point[0]
    risk_point = normalised_risk(z_point, weights_point)

    rng = np.random.default_rng(BOOT_SEED)
    on_frontier = np.zeros((len(USES), len(MODELS)))
    best_gain = np.zeros(len(MODELS))
    risk_draws = []
    for _ in range(BOOT_REPS):
        r = rng.integers(0, 100, 100)
        sd = np.array([aucs[m][r].std(ddof=1) for m in MODELS])
        fit = rng.uniform(prox["fit_seconds_q25"].to_numpy(), prox["fit_seconds_q75"].to_numpy())
        z = rule_scores(sd, n_hyper, fit, transp, shap_default.to_numpy())
        w = weight_draws[rng.integers(0, len(weight_draws))]
        rows = rng.integers(0, len(y_test), len(y_test))
        a = np.array([roc_auc_score(y_test[rows], p[rows]) for p in test_pd])
        gain = a[1:] - a[0]
        risk = normalised_risk(z, w)
        risk_draws.append(risk)
        best_gain[np.argmax(gain)] += 1
        for j in range(len(USES)):
            on_frontier[j] += frontier(gain, risk[j])
    on_frontier /= BOOT_REPS
    risk_se = np.std(np.stack(risk_draws), axis=0, ddof=1)

    table = []
    for j, use in enumerate(USES):
        for i, m in enumerate(MODELS):
            table.append({"use": use, "model": m, "risk": risk_point[j, i], "risk_se": risk_se[j, i],
                          "gain": gain_point[i], "p_frontier": on_frontier[j, i],
                          **{f"z_{c}": z_point[c][i] for c in CATEGORIES}})
    frame = pd.DataFrame(table)
    frame.to_csv(OUT / "i13_scorecard.csv", index=False)
    summary = {"p_best_gain": dict(zip(MODELS, (best_gain / BOOT_REPS).tolist())),
               "xgb_on_frontier_all_uses": float(on_frontier[:, MODELS.index("XGBoost")].min()),
               "lowest_risk_use_by_model": {m: USES[int(np.argmin(risk_point[:, i]))] for i, m in enumerate(MODELS)},
               "draws": BOOT_REPS}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(frame.round(3).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
