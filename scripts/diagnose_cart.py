"""Why is our CART AUC far above the paper's (84.4 vs 80.4)? A diagnostic, not a tuning step.

The paper reports a depth-7 tree with AUC 80.4 [79.7, 81.2] and the largest
spread across refits of all models (SD 0.005, Table 5). Ours, with the same
depth, scores about four points higher. We try every reading of the paper's
set-up we can think of, on the validation rows only (fit on the fit rows), so
that the test rows play no part, and record for each its AUC and its spread
over 30 refits on 60-100% subsamples. A hypothesis fits if it lands near both
of the paper's numbers.

H0  ours: scikit-learn, depth 7, defaults
H1  a different depth (2-12)
H2  a small, pruned tree, as R's rpart builds by default (cp = 0.01):
    emulated by capping the number of leaves
H3  class weights balanced for the 7% default rate
H4  entropy instead of Gini
H5  the file's unnamed row index used as an 11th characteristic (22 variables, as the paper counts)
H6  the tree scored by hard class labels instead of probabilities
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from mrisk.data import GMSC_FEATURES, GMSC_TARGET, RAW, add_squares, make_split
from mrisk.uncertainty import auc_ci

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "reproduction" / "models" / "cart_diagnostic.json"
PAPER_AUC, PAPER_CI, PAPER_SD = 0.804, (0.797, 0.812), 0.005
REFITS = 30


def complete_cases_with_index() -> pd.DataFrame:
    """Like data.load_gmsc, but keeps the file's unnamed row index (for H5)."""
    frame = pd.read_csv(RAW / "gmsc" / "cs-training.csv").rename(columns={"Unnamed: 0": "row_index"})
    return frame.dropna(subset=GMSC_FEATURES + [GMSC_TARGET]).reset_index(drop=True)


def evaluate(params: dict, x_fit, y_fit, x_val, y_val, hard: bool = False) -> dict:
    """Validation AUC (DeLong ±) and the spread of validation AUC over subsample refits."""
    tree = DecisionTreeClassifier(random_state=0, **params).fit(x_fit, y_fit)
    score = tree.predict(x_val) if hard else tree.predict_proba(x_val)[:, 1]
    auc = auc_ci(y_val, score)
    aucs = []
    for r in range(REFITS):
        rng = np.random.default_rng(10_000 + r)
        rows = rng.choice(len(y_fit), int(rng.uniform(0.6, 1.0) * len(y_fit)), replace=False)
        t = DecisionTreeClassifier(random_state=r, **params).fit(x_fit[rows], y_fit[rows])
        aucs.append(roc_auc_score(y_val, t.predict(x_val) if hard else t.predict_proba(x_val)[:, 1]))
    return {"auc": auc["auc"], "se": auc["se"], "sd_refits": float(np.std(aucs, ddof=1)),
            "leaves": int(tree.get_n_leaves()), "depth": int(tree.get_depth()),
            "inside_paper_ci": bool(PAPER_CI[0] <= auc["auc"] <= PAPER_CI[1])}


def main() -> None:
    frame = complete_cases_with_index()
    y = frame[GMSC_TARGET].to_numpy(dtype=int)
    s = make_split(y)
    x = add_squares(frame[GMSC_FEATURES].astype(float)).to_numpy()
    x_id = add_squares(frame[["row_index"] + GMSC_FEATURES].astype(float)).to_numpy()
    args = (x[s.fit], y[s.fit], x[s.valid], y[s.valid])
    out = {"H0 depth 7 (ours)": evaluate({"max_depth": 7}, *args)}
    for d in [2, 3, 4, 8, 9, 10, 11, 12]:
        out[f"H1 depth {d}"] = evaluate({"max_depth": d}, *args)
    for leaves in [3, 4, 6, 8, 12, 16]:
        out[f"H2 at most {leaves} leaves"] = evaluate({"max_depth": 7, "max_leaf_nodes": leaves}, *args)
    out["H3 depth 7, balanced class weights"] = evaluate({"max_depth": 7, "class_weight": "balanced"}, *args)
    out["H4 depth 7, entropy"] = evaluate({"max_depth": 7, "criterion": "entropy"}, *args)
    out["H5 depth 7, row index as a feature"] = evaluate({"max_depth": 7}, x_id[s.fit], y[s.fit], x_id[s.valid], y[s.valid])
    out["H6 depth 7, hard labels"] = evaluate({"max_depth": 7}, *args, hard=True)
    fits = [k for k, v in out.items() if v["inside_paper_ci"]]
    result = {"paper": {"auc": PAPER_AUC, "ci": PAPER_CI, "sd": PAPER_SD}, "hypotheses": out,
              "inside_paper_ci": fits}
    OUT.write_text(json.dumps(result, indent=2))
    print(pd.DataFrame(out).T.round(4).to_string())
    print("inside the paper's interval:", fits)


if __name__ == "__main__":
    main()
