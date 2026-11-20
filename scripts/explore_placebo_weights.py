"""Exploratory (not pre-registered): does the text-mining phase change which model wins?

Phase 2 of the paper turns 36 regulatory documents into category weights H̄.
A placebo test replaces those weights by random ones: if the conclusion stays
the same under weights that know nothing about regulation, the conclusion
comes from phase 1 (the darkness scores Z) and the AUC gains, not from the
texts. The idea follows a falsification check in a team's earlier seminar
project (noise predictors in place of real ones).

Inputs are the paper's own numbers only (Table 6 scores, Table 7 gains), plus
our gains as a second line; no data rows are read.

A model is on the risk-gain frontier if no other model has a risk at most as
high and a gain at least as high, one of them strictly. Random weights are
drawn uniformly from the simplex (Dirichlet(1, 1, 1)); the scale of the
weights does not matter for the frontier, only their proportions.

Output: results/improvements/appendix/placebo_weights.json and
results/report/figures/placebo_weights.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mrisk import paper_numbers as P

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "improvements" / "appendix" / "placebo_weights.json"
FIGURE = ROOT / "results" / "report" / "figures" / "placebo_weights.png"
KEYS = ROOT / "results" / "report" / "key_numbers.json"
DRAWS, SEED = 10_000, 0
CATS, MODELS = P.CATEGORIES, P.MODELS


def on_frontier(risk: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """Boolean per model: no other model is at least as good on both axes and better on one."""
    n = len(risk)
    keep = np.ones(n, dtype=bool)
    for m in range(n):
        for o in range(n):
            if o != m and risk[o] <= risk[m] and gain[o] >= gain[m] and (risk[o] < risk[m] or gain[o] > gain[m]):
                keep[m] = False
    return keep


def frontier_shares(z: np.ndarray, gain: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Share of weight vectors (rows of `weights`) under which each model is on the frontier."""
    hits = np.zeros(len(gain))
    for w in weights:
        hits += on_frontier(w @ z, gain)
    return hits / len(weights)


def dominates_everywhere(z: np.ndarray, gain: np.ndarray, a: int, b: int) -> bool:
    """True if model a beats model b for every non-negative weight vector: Z no higher in any category, gain higher."""
    return bool((z[:, a] <= z[:, b]).all() and gain[a] > gain[b])


def chart(res: dict, path: Path) -> Path:
    """Share of random weight vectors with each model on the frontier; how many uses it wins with the paper's weights."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mrisk.plots import GREY, NAVY, _save

    share = res["random_weights_share_on_frontier"]
    paper_uses = {m: sum(m in v for v in res["paper_weights_on_frontier"].values()) for m in MODELS}
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ys = np.arange(len(MODELS))[::-1]
    for y, m in zip(ys, MODELS):
        ax.barh(y, 100 * share[m], height=0.55, color=NAVY if share[m] > 0 else GREY)
        ax.text(100 * share[m] + 2, y, f"{100 * share[m]:.0f}%", va="center", fontsize=9)
        ax.text(132, y, f"{paper_uses[m]} of 3", va="center", ha="center", fontsize=9, color=NAVY)
    ax.text(132, len(MODELS) - 0.35, "paper's\nweights", ha="center", va="bottom", fontsize=8, color=GREY)
    ax.set_yticks(ys, MODELS)
    ax.set_xlim(0, 145)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel(f"Random weight vectors with the model on the frontier (%, {DRAWS:,} draws)")
    texts = len(pd.read_csv(ROOT / "data" / "corpus_manifest.csv"))
    ax.set_title(f"Placebo: random weights instead of the {texts} regulatory texts", fontsize=10.5, color=NAVY, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)


def main() -> None:
    z = np.array([[P.TABLE6_Z[c][m] for m in MODELS] for c in CATS], dtype=float)  # categories × models
    gains = {"paper": np.array([P.TABLE7_AUC[m] - P.TABLE7_AUC["Logit"] for m in MODELS])}
    if KEYS.exists():
        auc = json.loads(KEYS.read_text())["models"]["auc"]
        gains["ours"] = np.array([100 * auc[m]["gain"] for m in MODELS])
    weights = np.random.default_rng(SEED).dirichlet(np.ones(len(CATS)), size=DRAWS)
    paper_w = {u: np.array([P.TABLE2[u][c] for c in CATS]) for u in P.USES}
    out = {"draws": DRAWS, "seed": SEED, "models": MODELS, "z_source": "paper Table 6", "results": {}}
    for name, gain in gains.items():
        xgb = MODELS.index("XGBoost")
        out["results"][name] = {
            "random_weights_share_on_frontier": dict(zip(MODELS, frontier_shares(z, gain, weights).round(4).tolist())),
            "paper_weights_on_frontier": {u: [m for m, k in zip(MODELS, on_frontier(w @ z, gain)) if k] for u, w in paper_w.items()},
            "xgboost_dominates_for_any_weights": [MODELS[b] for b in range(len(MODELS)) if b != xgb and dominates_everywhere(z, gain, xgb, b)],
            "xgboost_highest_gain": bool(gain.argmax() == xgb),
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    chart(out["results"]["paper"], FIGURE)
    print(json.dumps(out["results"], indent=2))


if __name__ == "__main__":
    main()
