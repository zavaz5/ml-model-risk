"""Why is our CART AUC far above the paper's (84.4 vs 80.4)? A diagnostic, not a tuning step.

The paper reports a depth-7 tree with AUC 80.4. Ours, with the same depth,
is about four points higher. We look for settings that would give the
paper's number, on the validation rows only (fit on the fit rows), so that
the test rows play no part: tree depth, leaf size, and scoring the tree by
hard class labels instead of probabilities.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from mrisk.data import GMSC_FEATURES, GMSC_TARGET, design_matrix, load_gmsc, make_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "reproduction" / "models" / "cart_diagnostic.json"


def main() -> None:
    x, y = design_matrix(load_gmsc(), GMSC_FEATURES, GMSC_TARGET, squares=True)
    s = make_split(y)
    xv = x.to_numpy()
    xf, yf, xv_, yv = xv[s.fit], y[s.fit], xv[s.valid], y[s.valid]
    out = {"depth": {}, "min_samples_leaf_depth7": {}}
    for depth in [3, 5, 7, 9, 11, 13, 15, 20, 25, None]:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(xf, yf)
        out["depth"][str(depth)] = roc_auc_score(yv, tree.predict_proba(xv_)[:, 1])
    tree7 = DecisionTreeClassifier(max_depth=7, random_state=0).fit(xf, yf)
    out["depth7_hard_labels"] = roc_auc_score(yv, tree7.predict(xv_))
    for leaf in [1, 50, 200, 1000]:
        tree = DecisionTreeClassifier(max_depth=7, min_samples_leaf=leaf, random_state=0).fit(xf, yf)
        out["min_samples_leaf_depth7"][str(leaf)] = roc_auc_score(yv, tree.predict_proba(xv_)[:, 1])
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
