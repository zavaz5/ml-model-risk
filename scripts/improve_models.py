"""Pre-registered improvements on the model side: I1, I7, I11, I12.

I1  Same-protocol benchmark on three datasets (GMSC, Taiwan default, German credit).
I7  Both stability definitions (AUC spread, PD spread) with paired bootstrap comparisons.
I11 SHAP rankings on the probability scale over five seeds.
I12 Calibration metrics (Brier, ECE, calibration slope) next to AUC.

What is fitted: the six models, with the paper's hyperparameters, on each
dataset's 80% training rows. What is chosen: only the Lasso penalty (5-fold
CV on training rows). What is evaluated: test rows, after everything is fixed.
See docs/preregistration.md for every decision rule.

Usage: python scripts/improve_models.py [--rows N] [--workers K] [--only I1,I7,...]
"""
from __future__ import annotations

import argparse
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from mrisk.data import GMSC_FEATURES, GMSC_TARGET, add_squares, base_name, design_matrix, load_benchmark, \
    load_gmsc, make_split
from mrisk.models import MODEL_NAMES, choose_lasso_c, make_model, predict_pd
from mrisk.proxies import importance_by_feature, shap_values, single_thread, stability_run
from mrisk.uncertainty import BOOT_REPS, BOOT_SEED, auc_ci, auc_difference

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "improvements" / "models"
SEEDS = range(5)
SHAP_ROWS, SHAP_BACKGROUND = 1000, 100


def gmsc(rows=None):
    frame = load_gmsc()
    if rows:
        frame = frame.sample(rows, random_state=0).reset_index(drop=True)
    x, y = design_matrix(frame, GMSC_FEATURES, GMSC_TARGET, squares=True)
    return x, y


def benchmark_design(name: str, rows=None):
    """Numeric design with squares of the numeric columns (one-hot dummies are not squared)."""
    if name == "gmsc":
        return gmsc(rows)
    x, y = load_benchmark(name)
    numeric = [c for c in x.columns if x[c].nunique() > 2]
    x = pd.concat([x, x[numeric].pow(2).add_suffix("_sq")], axis=1)
    if rows and rows < len(y):
        keep = np.random.default_rng(0).choice(len(y), rows, replace=False)
        x, y = x.iloc[keep].reset_index(drop=True), y[keep]
    return x, y


def fit_all(x_train, y_train, seed=0, lasso_c=None):
    lasso_c = lasso_c if lasso_c is not None else choose_lasso_c(x_train, y_train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {m: make_model(m, seed, lasso_c).fit(x_train, y_train) for m in MODEL_NAMES}, lasso_c


# ---------------------------------------------------------------- I1
def improvement_1(rows):
    """AUC gain over the logit, paired DeLong, on three datasets; Spearman with transparency."""
    from mrisk.models import TRANSPARENCY

    out = []
    for dataset in ("gmsc", "taiwan_default", "german_credit"):
        x, y = benchmark_design(dataset, rows)
        s = make_split(y)
        xv = x.to_numpy(dtype=float)
        models, _ = fit_all(xv[s.train], y[s.train])
        pds = {m: predict_pd(models[m], xv[s.test]) for m in MODEL_NAMES}
        for m in MODEL_NAMES:
            a = auc_ci(y[s.test], pds[m])
            g = auc_difference(y[s.test], pds[m], pds["Logit"])
            out.append({"dataset": dataset, "model": m, "n": len(y), "default_rate": y.mean(),
                        "auc": a["auc"], "auc_se": a["se"], "gain": g["diff"], "gain_se": g["se"],
                        "gain_lo": g["lo"], "gain_hi": g["hi"], "gain_p": g["p"]})
        frame = pd.DataFrame([r for r in out if r["dataset"] == dataset and r["model"] != "Logit"])
        best, runner = frame.sort_values("gain", ascending=False)["model"].iloc[:2]
        top = auc_difference(y[s.test], pds[best], pds[runner])
        rho = stats.spearmanr([TRANSPARENCY[m] for m in frame["model"]], frame["gain"]).statistic
        print(f"  I1 {dataset}: best {best} (vs {runner}: diff {top['diff']:+.4f}, p={top['p']:.3f}), "
              f"Spearman(transparency, gain) = {rho:+.2f}", flush=True)
        out.append({"dataset": dataset, "model": "_summary", "best": best, "runner_up": runner,
                    "best_vs_runner_diff": top["diff"], "best_vs_runner_p": top["p"], "spearman": rho})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- I7
def improvement_7(x_train, y_train, x_test, y_test, lasso_c, workers):
    """Same 100 subsamples for every model; AUC spread and PD spread with paired bootstrap."""
    from sklearn.metrics import roc_auc_score

    aucs, pd_sd = {}, {}
    with ProcessPoolExecutor(workers, initializer=single_thread) as pool:
        for m in MODEL_NAMES:
            jobs = [pool.submit(stability_run, m, x_train, y_train, x_test, lasso_c, r) for r in range(100)]
            preds = np.vstack([j.result()[1] for j in jobs])
            aucs[m] = np.array([roc_auc_score(y_test, p) for p in preds])
            pd_sd[m] = preds.std(axis=0, ddof=1)
            print(f"  I7 {m}: SD(AUC) {aucs[m].std(ddof=1):.4f}, mean SD(PD) {pd_sd[m].mean():.4f}", flush=True)
    rng = np.random.default_rng(BOOT_SEED)
    runs, rows_ = len(aucs["Logit"]), len(y_test)
    draws = {"auc_sd": {m: [] for m in MODEL_NAMES}, "pd_sd": {m: [] for m in MODEL_NAMES}}
    for _ in range(BOOT_REPS):
        r = rng.integers(0, runs, runs)  # resample retrains (paired across models)
        k = rng.integers(0, rows_, rows_)  # resample test rows for the PD definition
        for m in MODEL_NAMES:
            draws["auc_sd"][m].append(aucs[m][r].std(ddof=1))
            draws["pd_sd"][m].append(pd_sd[m][k].mean())
    table = []
    for m in MODEL_NAMES:
        table.append({"model": m, "auc_sd": aucs[m].std(ddof=1), "auc_sd_se": np.std(draws["auc_sd"][m], ddof=1),
                      "pd_sd": pd_sd[m].mean(), "pd_sd_se": np.std(draws["pd_sd"][m], ddof=1)})
    claim = {}
    for key in ("auc_sd", "pd_sd"):
        d = {m: np.array(v) for m, v in draws[key].items()}
        stable = np.maximum(d["Lasso"], d["XGBoost"])
        unstable = np.minimum(d["CART"], d["MLP"])
        claim[key] = float(np.mean(stable < unstable))
    return pd.DataFrame(table), claim


# ---------------------------------------------------------------- I11
def _shap_seed(seed, x_train, y_train, x_test, columns, lasso_c):
    single_thread()
    rng = np.random.default_rng(seed)
    background = x_train[rng.choice(len(x_train), SHAP_BACKGROUND, replace=False)]
    explain = x_test[rng.choice(len(x_test), min(SHAP_ROWS, len(x_test)), replace=False)]
    models, _ = fit_all(x_train, y_train, seed, lasso_c)
    out = {}
    for m in ("RF", "XGBoost", "MLP"):
        for common in (False, True):
            values, _ = shap_values(m, models[m], background, explain, common=common)
            imp = importance_by_feature(values, columns, base_name)
            out[(m, "probability" if common else "default")] = sorted(imp, key=imp.get, reverse=True)
    return seed, out


def improvement_11(x_train, y_train, x_test, columns, lasso_c, workers):
    with ProcessPoolExecutor(min(workers, len(SEEDS))) as pool:
        results = list(pool.map(_shap_seed, SEEDS, *([x_train] * 5, [y_train] * 5, [x_test] * 5,
                                                     [columns] * 5, [lasso_c] * 5)))
    rows = []
    for seed, ranks in results:
        for (m, scale), order in ranks.items():
            rows.append({"seed": seed, "model": m, "scale": scale, **{f"rank_{i + 1}": f for i, f in enumerate(order)}})
    frame = pd.DataFrame(rows)

    def holds(scale):
        f = frame[frame["scale"] == scale]
        top3 = lambda r: [r["rank_1"], r["rank_2"], r["rank_3"]]
        c1 = f.apply(lambda r: {"NumberOfTimes90DaysLate", "NumberOfTime30-59DaysPastDueNotWorse"} <= set(top3(r)), axis=1)
        c2_x = f[f["model"] == "XGBoost"]["rank_1"] == "RevolvingUtilizationOfUnsecuredLines"
        c2_o = f[f["model"] != "XGBoost"]["rank_1"] != "RevolvingUtilizationOfUnsecuredLines"
        c3 = f["rank_4"] == "age"
        per_seed = lambda s: s.groupby(f.loc[s.index, "seed"]).all()
        return {"top3_late_payments": int(per_seed(c1).sum()),
                "revolving_first_only_xgb": int((per_seed(c2_x) & per_seed(c2_o)).sum()),
                "age_fourth": int(per_seed(c3).sum()), "seeds": len(SEEDS)}

    return frame, {"default": holds("default"), "probability": holds("probability")}


# ---------------------------------------------------------------- I12
def calibration(y, p, bins=10):
    """Brier score, expected calibration error (equal-count bins) and calibration slope."""
    from sklearn.linear_model import LogisticRegression

    brier = float(np.mean((p - y) ** 2))
    order = np.argsort(p)
    ece = sum(len(b) / len(p) * abs(p[b].mean() - y[b].mean()) for b in np.array_split(order, bins))
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    slope = float(LogisticRegression(C=np.inf).fit(logit, y).coef_[0, 0])
    return brier, float(ece), slope


def improvement_12(models, x_test, y_test):
    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    pds = {m: predict_pd(models[m], x_test) for m in MODEL_NAMES}
    boot = [rng.integers(0, len(y_test), len(y_test)) for _ in range(200)]  # 200 draws: ECE and slope are slow
    for m in MODEL_NAMES:
        b, e, s = calibration(y_test, pds[m])
        draws = np.array([calibration(y_test[k], pds[m][k]) for k in boot])
        rows.append({"model": m, "auc": auc_ci(y_test, pds[m])["auc"], "brier": b, "brier_se": draws[:, 0].std(ddof=1),
                     "ece": e, "ece_se": draws[:, 1].std(ddof=1), "slope": s, "slope_se": draws[:, 2].std(ddof=1)})
    frame = pd.DataFrame(rows)
    agree = stats.spearmanr(frame["auc"], -frame["brier"]).statistic
    return frame, float(agree)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--only", default="I1,I7,I11,I12")
    args = parser.parse_args()
    only = set(args.only.split(","))
    single_thread()
    OUT.mkdir(parents=True, exist_ok=True)
    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    x, y = gmsc(args.rows)
    s = make_split(y)
    xv, columns = x.to_numpy(), list(x.columns)
    x_train, y_train, x_test, y_test = xv[s.train], y[s.train], xv[s.test], y[s.test]
    lasso_c = choose_lasso_c(x_train, y_train)

    if "I12" in only:
        models, _ = fit_all(x_train, y_train, 0, lasso_c)
        # Test-row PDs in MODEL_NAMES order, for the scorecard Monte Carlo (improve_risk.py).
        np.save(OUT / "test_pd.npy", np.vstack([predict_pd(models[m], x_test) for m in MODEL_NAMES]))
        np.save(OUT / "y_test.npy", y_test)
        i12, agree = improvement_12(models, x_test, y_test)
        i12.to_csv(OUT / "i12_calibration.csv", index=False)
        summary["I12_spearman_auc_vs_brier"] = agree
        print(i12.round(4).to_string(index=False), flush=True)
    if "I7" in only:
        i7, claim = improvement_7(x_train, y_train, x_test, y_test, lasso_c, args.workers)
        i7.to_csv(OUT / "i7_stability.csv", index=False)
        summary["I7_claim_probability"] = claim
    if "I11" in only:
        i11, holds = improvement_11(x_train, y_train, x_test, columns, lasso_c, args.workers)
        i11.to_csv(OUT / "i11_shap_ranks.csv", index=False)
        summary["I11_claims_seeds"] = holds
        print(holds, flush=True)
    if "I1" in only:
        i1 = improvement_1(args.rows)
        i1.to_csv(OUT / "i1_benchmark.csv", index=False)
    summary["smoke_rows"] = args.rows
    summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
