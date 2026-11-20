"""SHAP values behind our versions of the paper's Figs. 6-8 (RF, XGBoost, MLP).

Same set-up as reproduce_models.py: the paper's settings fitted on the 80%
training rows, 1,000 random test rows explained, each library's default
explainer (XGBoost in log-odds, as the paper's Fig. 7). The values of x and
x² are summed per row, so each plot shows the paper's ten characteristics;
points are coloured by the raw characteristic's value, as the paper's are.

Output: results/reproduction/models/shap_values_<model>.npz
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from mrisk.data import GMSC_FEATURES, GMSC_TARGET, base_name, design_matrix, load_gmsc, make_split
from mrisk.models import make_model
from mrisk.proxies import shap_values, single_thread

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "reproduction" / "models"
ROWS, BACKGROUND = 1000, 100


def main() -> None:
    single_thread()
    frame = load_gmsc()
    x, y = design_matrix(frame, GMSC_FEATURES, GMSC_TARGET, squares=True)
    s = make_split(y)
    xv, columns = x.to_numpy(), list(x.columns)
    rng = np.random.default_rng(0)
    background = xv[s.train][rng.choice(len(s.train), BACKGROUND, replace=False)]
    rows = s.test[rng.choice(len(s.test), ROWS, replace=False)]
    for name in ("RF", "XGBoost", "MLP"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = make_model(name, 0).fit(xv[s.train], y[s.train])
        values, seconds = shap_values(name, model, background, xv[rows])
        summed = np.column_stack([values[:, [j for j, c in enumerate(columns) if base_name(c) == f]].sum(axis=1)
                                  for f in GMSC_FEATURES])
        np.savez(OUT / f"shap_values_{name}.npz", values=summed, features=frame.loc[rows, GMSC_FEATURES].to_numpy(),
                 names=np.array(GMSC_FEATURES), scale="log-odds" if name == "XGBoost" else "probability")
        print(f"{name}: {seconds:.0f} s", flush=True)


if __name__ == "__main__":
    main()
