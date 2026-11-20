"""Exploratory (not pre-registered): is the paper's logit a fair benchmark?

A3 showed that log(1 + x) of the inputs lifts the logit's validation AUC by
about four points while leaving the tree ensembles unchanged. Here we test
that gap formally: the logit on the paper's design and on log-transformed
inputs, against XGBoost on the paper's design, with paired DeLong tests on
the validation rows (fit rows for fitting; the test rows are not used).
Credit scorers routinely transform skewed ratios (logs, binning, weight of
evidence) before a logistic regression; the paper's logit gets raw values
and their squares.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from mrisk.data import GMSC_FEATURES, GMSC_TARGET, add_squares, load_gmsc, make_split
from mrisk.models import make_model, predict_pd
from mrisk.uncertainty import auc_ci, auc_difference

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "improvements" / "appendix" / "logit_benchmark.json"


def main() -> None:
    frame = load_gmsc()
    y = frame[GMSC_TARGET].to_numpy(dtype=int)
    s = make_split(y)
    raw = frame[GMSC_FEATURES].astype(float)
    designs = {"paper": add_squares(raw).to_numpy(), "log1p": add_squares(np.log1p(raw.clip(lower=0))).to_numpy()}
    pds = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, x in designs.items():
            pds[f"Logit, {name} inputs"] = predict_pd(make_model("Logit").fit(x[s.fit], y[s.fit]), x[s.valid])
        pds["XGBoost, paper inputs"] = predict_pd(make_model("XGBoost").fit(designs["paper"][s.fit], y[s.fit]), designs["paper"][s.valid])
    yv = y[s.valid]
    out = {"auc": {k: auc_ci(yv, p) for k, p in pds.items()},
           "xgb_minus_logit_paper": auc_difference(yv, pds["XGBoost, paper inputs"], pds["Logit, paper inputs"]),
           "xgb_minus_logit_log1p": auc_difference(yv, pds["XGBoost, paper inputs"], pds["Logit, log1p inputs"]),
           "rows": int(len(yv)), "preregistered": False}
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
