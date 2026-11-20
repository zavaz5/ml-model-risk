"""I14: the I13 rule, with the appendix factors (A1-A5) added to their categories.

Statistics: AUC spread, the paper's hyperparameter count, overfitting gap,
features used, feature-engineering sensitivity. Technology: training time (log)
and transparency, as in I13. Conduct: SHAP time (log), surrogate infidelity
(1 - R²) and the age flip rate. Each proxy is mapped to [1, 5] by min-max
across the five models; risk is normalised by its maximum (as in I13). Gains
are measured on the validation rows. Pre-registered (addendum, I14).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mrisk.models import PAPER_HYPERPARAMETER_COUNT, TRANSPARENCY

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "improvements" / "appendix"
MODELS5 = ["Lasso", "CART", "RF", "XGBoost", "MLP"]
CATS, USES = ["statistics", "technology", "conduct"], ["capital", "scoring", "provisions"]


def minmax(values, log: bool = False) -> np.ndarray:
    v = np.log10(np.asarray(values, float)) if log else np.asarray(values, float)
    return np.full_like(v, 3.0) if np.ptp(v) == 0 else 1 + 4 * (v - v.min()) / np.ptp(v)


def frontier(gain: np.ndarray, risk: np.ndarray) -> list[str]:
    return [m for i, m in enumerate(MODELS5)
            if not any(gain[k] >= gain[i] and risk[k] <= risk[i] and (gain[k] > gain[i] or risk[k] < risk[i])
                       for k in range(len(MODELS5)) if k != i)]


def main() -> None:
    a = pd.read_csv(OUT / "appendix_factors.csv").set_index("model")
    t = a.loc[MODELS5]
    rep = ROOT / "results" / "reproduction"
    prox = pd.read_csv(rep / "models" / "proxies.csv").set_index("model").loc[MODELS5]
    shap = pd.read_csv(rep / "models" / "shap_default.csv").set_index("model").loc[MODELS5, "seconds"]
    z = {"statistics": np.mean([minmax(prox["auc_sd"]), minmax([PAPER_HYPERPARAMETER_COUNT[m] for m in MODELS5]),
                                minmax(t["overfit_gap"]), minmax(t["features_used"]), minmax(t["fe_sensitivity"])], axis=0),
         "technology": np.mean([minmax(prox["fit_seconds"], log=True), minmax([TRANSPARENCY[m] for m in MODELS5])], axis=0),
         "conduct": np.mean([minmax(shap, log=True), minmax(1 - t["surrogate_r2"]), minmax(t["flip_rate"])], axis=0)}
    w = pd.read_csv(rep / "nlp" / "table2.csv", index_col=0).loc[USES, CATS].to_numpy()
    zz = np.vstack([z[c] for c in CATS])
    risk = (w @ zz) / (5 * w.sum(axis=1, keepdims=True))
    gain = (t["auc_val"] - a.loc["Logit", "auc_val"]).to_numpy()
    table = pd.DataFrame({"model": MODELS5, **{f"z_{c}": z[c] for c in CATS}, "gain_val": gain,
                          **{f"risk_{u}": risk[j] for j, u in enumerate(USES)}})
    table.to_csv(OUT / "i14_extended_scorecard.csv", index=False)
    summary = {"frontier": {u: frontier(gain, risk[j]) for j, u in enumerate(USES)}}
    (OUT / "i14_summary.json").write_text(json.dumps(summary, indent=2))
    print(table.round(3).to_string(index=False))
    print(summary)


if __name__ == "__main__":
    main()
