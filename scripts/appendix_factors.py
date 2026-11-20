"""Measure the risk factors the paper names but does not measure (its appendix, pp. 23-25).

Pre-registered in docs/preregistration.md (addendum, A1-A5 and I14). Every
model is fitted on the fit rows with the paper's settings and evaluated on
the validation rows; the test rows are not used.

A1  Overfitting: AUC on the fit rows minus AUC on the validation rows; learning curve.
A2  Dynamic calibration (features in production): inputs the fitted model uses.
A3  Feature engineering: largest change in validation AUC under four preprocessing variants.
A4  Auditability: R² of a depth-5 surrogate tree that mimics the model's PD.
A5  Bias: share of decisions that flip when a borrower's age moves by ±10 years.
(I14, the scorecard with these proxies added, is scripts/extended_scorecard.py.)

Third-party providers, cyber risk and privacy need a bank's IT set-up and
data inventory; we do not invent a proxy for them.
"""
from __future__ import annotations

import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.tree import DecisionTreeRegressor

from mrisk.data import GMSC_FEATURES, GMSC_TARGET, RAW, add_squares, load_gmsc, make_split
from mrisk.models import MODEL_NAMES, choose_lasso_c, make_model, predict_pd
from mrisk.proxies import single_thread

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "improvements" / "appendix"
MODELS5 = ["Lasso", "CART", "RF", "XGBoost", "MLP"]
CURVE = [0.10, 0.25, 0.50, 1.00]
VARIANTS = ["no_squares", "winsorised", "log1p", "imputed_rows_added"]
AGE_SHIFT, REJECT_SHARE = 10, 0.10


def design(raw: pd.DataFrame, variant: str | None, reference: pd.DataFrame) -> pd.DataFrame:
    """The paper's design (characteristics and squares) or one preprocessing variant of it.

    Winsorising cut-offs come from ``reference`` (the fit rows), so the
    validation rows never inform the preprocessing.
    """
    x = raw[GMSC_FEATURES].astype(float)
    if variant == "no_squares":
        return x
    if variant == "winsorised":
        lo, hi = reference[GMSC_FEATURES].quantile(0.01), reference[GMSC_FEATURES].quantile(0.99)
        x = x.clip(lo, hi, axis=1)
    if variant == "log1p":
        x = np.log1p(x.clip(lower=0))
    return add_squares(x)


def fit(name: str, x: np.ndarray, y: np.ndarray, lasso_c: float):
    single_thread()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return make_model(name, 0, lasso_c).fit(x, y)


def _job(args):
    """One fit, scored on the validation rows; returns what the caller asked for."""
    name, x_fit, y_fit, x_val, y_val, lasso_c, keep = args
    model = fit(name, x_fit, y_fit, lasso_c)
    p_val = predict_pd(model, x_val)
    out = {"auc_val": roc_auc_score(y_val, p_val)}
    if keep:
        out["model"] = model
        out["auc_fit"] = roc_auc_score(y_fit, predict_pd(model, x_fit))
    return name, out


def features_used(name: str, model, n: int) -> int:
    est = model[-1] if hasattr(model, "steps") else model
    if name in ("Logit", "Lasso"):
        return int(np.count_nonzero(est.coef_))
    if name == "CART":
        return int(len(set(est.tree_.feature[est.tree_.feature >= 0])))
    if name == "RF":
        return int(len(set(np.concatenate([t.tree_.feature[t.tree_.feature >= 0] for t in est.estimators_]))))
    if name == "XGBoost":
        return int(len(est.get_booster().get_score(importance_type="weight")))
    if name == "MLP":
        return int(np.count_nonzero(np.abs(est.coefs_[0]).sum(axis=1)))
    raise ValueError(name)


def surrogate_fidelity(model, x_fit, x_val) -> float:
    """R² between a depth-5 regression tree's imitation of the model's PD and the PD itself."""
    tree = DecisionTreeRegressor(max_depth=5, random_state=0).fit(x_fit, predict_pd(model, x_fit))
    return float(r2_score(predict_pd(model, x_val), tree.predict(x_val)))


def age_flips(model, raw_val: pd.DataFrame, age_range: tuple[float, float]) -> dict:
    """Decisions reject the 10% highest baseline PDs; count flips when only age moves by ±10 years."""
    base = predict_pd(model, design(raw_val, None, raw_val).to_numpy())
    cutoff = np.quantile(base, 1 - REJECT_SHARE)
    reject = base > cutoff
    out = {}
    for shift in (AGE_SHIFT, -AGE_SHIFT):
        moved = raw_val.copy()
        moved["age"] = (moved["age"] + shift).clip(*age_range)
        p = predict_pd(model, design(moved, None, raw_val).to_numpy())
        out[f"flip_{shift:+d}"] = float(np.mean((p > cutoff) != reject))
        out[f"mean_abs_dpd_{shift:+d}"] = float(np.mean(np.abs(p - base)))
    out["flip_rate"] = (out[f"flip_{AGE_SHIFT:+d}"] + out[f"flip_{-AGE_SHIFT:+d}"]) / 2
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = load_gmsc()
    y = frame[GMSC_TARGET].to_numpy(dtype=int)
    s = make_split(y)
    raw_fit, raw_val = frame.iloc[s.fit], frame.iloc[s.valid]
    y_fit, y_val = y[s.fit], y[s.valid]
    x_fit = design(raw_fit, None, raw_fit).to_numpy()
    x_val = design(raw_val, None, raw_fit).to_numpy()
    lasso_c = choose_lasso_c(x_fit, y_fit)
    print(f"Lasso C on fit rows: {lasso_c:g}", flush=True)

    jobs = [(m, x_fit, y_fit, x_val, y_val, lasso_c, True) for m in MODEL_NAMES]
    rng = np.random.default_rng(0)
    for frac in CURVE[:-1]:
        rows = rng.choice(len(y_fit), int(frac * len(y_fit)), replace=False)
        jobs += [(m, x_fit[rows], y_fit[rows], x_val, y_val, lasso_c, False) for m in MODEL_NAMES]
    variant_jobs = []
    for v in VARIANTS[:-1]:
        xf, xv = design(raw_fit, v, raw_fit).to_numpy(), design(raw_val, v, raw_fit).to_numpy()
        variant_jobs += [(m, xf, y_fit, xv, y_val, lasso_c, False) for m in MODEL_NAMES]
    raw_all = pd.read_csv(RAW / "gmsc" / "cs-training.csv")
    incomplete = raw_all[raw_all[GMSC_FEATURES + [GMSC_TARGET]].isna().any(axis=1)].copy()
    incomplete[GMSC_FEATURES] = incomplete[GMSC_FEATURES].fillna(raw_fit[GMSC_FEATURES].median())
    raw_imp = pd.concat([raw_fit, incomplete[raw_fit.columns.intersection(incomplete.columns)]])
    x_imp = design(raw_imp, None, raw_fit).to_numpy()
    y_imp = raw_imp[GMSC_TARGET].to_numpy(dtype=int)
    variant_jobs += [(m, x_imp, y_imp, x_val, y_val, lasso_c, False) for m in MODEL_NAMES]

    with ProcessPoolExecutor(14, initializer=single_thread) as pool:
        base = list(pool.map(_job, jobs[: len(MODEL_NAMES)]))
        curve = list(pool.map(_job, jobs[len(MODEL_NAMES):]))
        var = list(pool.map(_job, variant_jobs))
    fitted = {name: res for name, res in base}

    age_range = (raw_fit["age"].min(), raw_fit["age"].max())
    rows = []
    for k, m in enumerate(MODEL_NAMES):
        model = fitted[m]["model"]
        lc = {f"curve_{int(100 * f)}": curve[i * len(MODEL_NAMES) + k][1]["auc_val"] for i, f in enumerate(CURVE[:-1])}
        lc["curve_100"] = fitted[m]["auc_val"]
        deltas = {v: var[i * len(MODEL_NAMES) + k][1]["auc_val"] - fitted[m]["auc_val"] for i, v in enumerate(VARIANTS)}
        rows.append({"model": m, "auc_fit": fitted[m]["auc_fit"], "auc_val": fitted[m]["auc_val"],
                     "overfit_gap": fitted[m]["auc_fit"] - fitted[m]["auc_val"], **lc,
                     "features_used": features_used(m, model, x_fit.shape[1]),
                     **{f"d_auc_{v}": d for v, d in deltas.items()},
                     "fe_sensitivity": max(abs(d) for d in deltas.values()),
                     "surrogate_r2": surrogate_fidelity(model, x_fit, x_val),
                     **age_flips(model, raw_val, age_range)})
        print(f"  {m}: gap {rows[-1]['overfit_gap']:.4f}, features {rows[-1]['features_used']}, "
              f"FE {rows[-1]['fe_sensitivity']:.4f}, surrogate R² {rows[-1]['surrogate_r2']:.3f}, "
              f"age flips {rows[-1]['flip_rate']:.4f}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "appendix_factors.csv", index=False)

    summary = {"lasso_c_fit": lasso_c, "not_measurable": ["third-party providers", "cyber risk", "privacy"]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
