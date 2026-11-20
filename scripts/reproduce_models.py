"""Reproduce the paper's empirical example: Tables 5 and 7 and Figs. 6-8.

What is fitted: each of the six classifiers on the paper's 80% training rows,
by its own loss (log-loss for the logistic models, the MLP and XGBoost; Gini
impurity for the trees), with the paper's hyperparameters.
What is chosen: only the Lasso penalty, by 5-fold cross-validated AUC on the
training rows (the paper's procedure; it does not report the value).
What is evaluated: AUC-ROC on the 20% test rows, the five model-risk proxies,
and the SHAP feature ranking.

Run order matters for timing hygiene (CLAUDE.md rule 7): single-threaded
timings first, on an otherwise idle machine, then the parallel retrains.

Usage: python scripts/reproduce_models.py [--rows N] [--workers K]
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from mrisk.data import GMSC_FEATURES, GMSC_TARGET, base_name, design_matrix, load_gmsc, make_split
from mrisk.models import MODEL_NAMES, PAPER_HYPERPARAMETER_COUNT, TRANSPARENCY, choose_lasso_c, \
    fitted_parameter_count, make_model, predict_pd
from mrisk.proxies import importance_by_feature, shap_values, single_thread, stability_run, time_fit
from mrisk.uncertainty import auc_ci, auc_difference, bootstrap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "reproduction" / "models"
N_RETRAINS = 100
SHAP_ROWS = 1000  # test rows explained; the paper does not say how many (DEVIATIONS.md)
SHAP_BACKGROUND = 100


def environment() -> dict:
    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    return {"python": platform.python_version(), "os": platform.platform(), "cpu": cpu}


def load(rows: int | None):
    frame = load_gmsc()
    if rows:
        frame = frame.sample(rows, random_state=0).reset_index(drop=True)
    x, y = design_matrix(frame, GMSC_FEATURES, GMSC_TARGET, squares=True)
    return frame, x, y


def fit_and_score(x_train, y_train, x_test, y_test, lasso_c):
    """Fit every model once; AUC with DeLong interval and the paired gain over the logit."""
    models, pds = {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in MODEL_NAMES:
            models[name] = make_model(name, 0, lasso_c).fit(x_train, y_train)
            pds[name] = predict_pd(models[name], x_test)
    rows = []
    for name in MODEL_NAMES:
        auc = auc_ci(y_test, pds[name])
        gain = auc_difference(y_test, pds[name], pds["Logit"])
        rows.append({"model": name, **{f"auc_{k}": v for k, v in auc.items()},
                     **{f"gain_{k}": v for k, v in gain.items()},
                     "fitted_parameters": fitted_parameter_count(name, models[name])})
    return models, pds, pd.DataFrame(rows)


def stability(x_train, y_train, x_test, y_test, lasso_c, workers):
    """100 retrains per model on 60-100% subsamples; spread of test AUC and of PDs."""
    from sklearn.metrics import roc_auc_score

    rows, pd_spread = [], {}
    with ProcessPoolExecutor(workers, initializer=single_thread) as pool:
        for name in MODEL_NAMES:
            jobs = [pool.submit(stability_run, name, x_train, y_train, x_test, lasso_c, r) for r in range(N_RETRAINS)]
            results = [j.result() for j in jobs]
            fracs = np.array([f for f, _ in results])
            preds = np.vstack([p for _, p in results])
            aucs = np.array([roc_auc_score(y_test, p) for p in preds])
            sd = bootstrap(aucs, lambda a: a.std(ddof=1))
            pd_sd = preds.std(axis=0, ddof=1)
            pd_spread[name] = pd_sd
            rows.append({"model": name, "auc_mean": aucs.mean(), "auc_sd": sd["est"], "auc_sd_se": sd["se"],
                         "auc_sd_lo": sd["lo"], "auc_sd_hi": sd["hi"], "pd_sd_mean": float(pd_sd.mean()),
                         "frac_min": fracs.min(), "frac_max": fracs.max()})
            np.save(OUT / f"stability_aucs_{name}.npy", aucs)
            print(f"  stability {name}: SD(AUC) = {sd['est']:.4f} ± {sd['se']:.4f}", flush=True)
    return pd.DataFrame(rows)


def shap_block(models, x_train, x_test, columns, common: bool):
    """SHAP time and per-characteristic importance for every model on the same rows."""
    rng = np.random.default_rng(0)
    background = x_train[rng.choice(len(x_train), SHAP_BACKGROUND, replace=False)]
    explain = x_test[rng.choice(len(x_test), min(SHAP_ROWS, len(x_test)), replace=False)]
    rows, ranks = [], {}
    for name in MODEL_NAMES:
        values, seconds = shap_values(name, models[name], background, explain, common=common)
        imp = importance_by_feature(values, columns, base_name)
        order = sorted(imp, key=imp.get, reverse=True)
        ranks[name] = order
        rows.append({"model": name, "seconds": seconds, "rows": len(explain),
                     **{f"imp_{k}": v for k, v in imp.items()}})
        print(f"  SHAP{' (common)' if common else ''} {name}: {seconds:.1f} s, top-4 {order[:4]}", flush=True)
    return pd.DataFrame(rows), ranks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=None, help="smoke test on a random subset")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    single_thread()
    OUT.mkdir(parents=True, exist_ok=True)

    frame, x, y = load(args.rows)
    split = make_split(y)
    columns = list(x.columns)
    xv = x.to_numpy()
    x_train, y_train, x_test, y_test = xv[split.train], y[split.train], xv[split.test], y[split.test]
    np.savez(OUT / "split_indices.npz", train=split.train, test=split.test, fit=split.fit, valid=split.valid)
    print(f"rows {len(y):,} (default rate {y.mean():.4f}), features {len(columns)}, train {len(y_train):,}, test {len(y_test):,}")

    lasso_c = choose_lasso_c(x_train, y_train)
    print(f"Lasso C chosen by 5-fold CV: {lasso_c:g}")

    models, pds, auc_table = fit_and_score(x_train, y_train, x_test, y_test, lasso_c)
    print(auc_table[["model", "auc_auc", "auc_lo", "auc_hi", "gain_diff", "gain_p"]].round(4).to_string(index=False))

    timings = {}
    for name in MODEL_NAMES:
        timings[name] = time_fit(name, x_train, y_train, lasso_c)
        print(f"  fit time {name}: {timings[name]['median']:.2f} s", flush=True)

    shap_paper, ranks_paper = shap_block(models, x_train, x_test, columns, common=False)
    shap_common, ranks_common = shap_block(models, x_train, x_test, columns, common=True)

    stab = stability(x_train, y_train, x_test, y_test, lasso_c, args.workers)

    proxies = pd.DataFrame({
        "model": MODEL_NAMES,
        "n_hyperparameters_paper": [PAPER_HYPERPARAMETER_COUNT.get(m) for m in MODEL_NAMES],
        "transparency": [TRANSPARENCY[m] for m in MODEL_NAMES],
        "fit_seconds": [timings[m]["median"] for m in MODEL_NAMES],
        "fit_seconds_q25": [timings[m]["q25"] for m in MODEL_NAMES],
        "fit_seconds_q75": [timings[m]["q75"] for m in MODEL_NAMES],
    }).merge(stab, on="model")

    auc_table.to_csv(OUT / "auc.csv", index=False)
    proxies.to_csv(OUT / "proxies.csv", index=False)
    shap_paper.to_csv(OUT / "shap_default.csv", index=False)
    shap_common.to_csv(OUT / "shap_common.csv", index=False)
    raw = pd.read_csv(ROOT / "data" / "raw" / "gmsc" / "cs-training.csv")
    meta = {"rows": int(len(y)), "rows_raw": int(len(raw)), "raw_columns": list(raw.columns),
            "defaults_raw": int(raw[GMSC_TARGET].sum()),
            "default_rate": float(y.mean()), "features": columns, "lasso_c": lasso_c,
            "train": int(len(y_train)), "test": int(len(y_test)), "smoke_rows": args.rows,
            "shap_rows": SHAP_ROWS, "shap_background": SHAP_BACKGROUND, "retrains": N_RETRAINS,
            "ranks_default": ranks_paper, "ranks_common": ranks_common, "environment": environment(),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(proxies.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
