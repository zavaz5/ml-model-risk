"""Collect every number of ours from results/ into one dictionary, and decide the verdicts.

The README, the slides and the speaker script all read ``key_numbers.json``
written from this module, so a number appears in one place only. Verdict
rules are those of docs/preregistration.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import paper_numbers as P

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
MODELS5 = ["Lasso", "CART", "RF", "XGBoost", "MLP"]
CATS, USES = P.CATEGORIES, P.USES


def _json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _order(values: dict, models=MODELS5) -> list[str]:
    return sorted(models, key=lambda m: values[m])


def models_block() -> dict:
    d = RES / "reproduction" / "models"
    auc = pd.read_csv(d / "auc.csv").set_index("model")
    prox = pd.read_csv(d / "proxies.csv").set_index("model")
    shap_default = pd.read_csv(d / "shap_default.csv").set_index("model")
    shap_common = pd.read_csv(d / "shap_common.csv").set_index("model")
    meta = _json(d / "meta.json")
    out = {"meta": {k: meta[k] for k in ("rows", "rows_raw", "defaults_raw", "default_rate", "train", "test",
                                         "lasso_c", "shap_rows", "shap_background", "retrains", "environment")},
           "n_features": len(meta["features"]), "auc": {}, "proxies": {}, "ranks_default": meta["ranks_default"],
           "ranks_common": meta["ranks_common"]}
    for m in auc.index:
        a = auc.loc[m]
        out["auc"][m] = {"auc": a["auc_auc"], "se": a["auc_se"], "lo": a["auc_lo"], "hi": a["auc_hi"],
                         "gain": a["gain_diff"], "gain_se": a["gain_se"], "gain_p": a["gain_p"],
                         "params": int(a["fitted_parameters"])}
        p = prox.loc[m]
        out["proxies"][m] = {"fit_s": p["fit_seconds"], "fit_q25": p["fit_seconds_q25"], "fit_q75": p["fit_seconds_q75"],
                             "auc_sd": p["auc_sd"], "auc_sd_se": p["auc_sd_se"], "pd_sd": p["pd_sd_mean"],
                             "shap_s": shap_default.loc[m, "seconds"], "shap_common_s": shap_common.loc[m, "seconds"]}
    return out


def nlp_block() -> dict:
    d = RES / "reproduction" / "nlp"
    if not (d / "table2.csv").exists():
        return {}
    t2 = pd.read_csv(d / "table2.csv", index_col=0)
    t3 = pd.read_csv(d / "table3.csv")
    neg10 = pd.read_csv(d / "negatives_d10.csv", index_col=0)
    neg20 = pd.read_csv(d / "negatives_d20.csv", index_col=0)
    share = pd.read_csv(d / "negative_share.csv", index_col=0)["share"]
    meta = _json(d / "meta.json")
    return {"table2": (100 * t2).to_dict(orient="index"),
            "table3": {f"{r.pair}|{r.category}": {"p": r.p, "df": r.df} for r in t3.itertuples()},
            "neg_d10": neg10.to_dict(orient="index"), "neg_d20": neg20.to_dict(orient="index"),
            "neg_share": (100 * share).to_dict(), "meta": meta}


def verdicts(k: dict) -> list[dict]:
    """One row per paper result R1-R16: what the paper says, what we find, and the verdict."""
    rows = []
    m, n, imp = k["models"], k.get("nlp", {}), k.get("improvements", {})
    lit = pd.read_csv(RES / "paper_extracted" / "fig1_literature.csv")
    med = lit[lit["complexity_rank"] > 0].groupby("complexity_rank")["gain_pct"].median()
    monotone = bool(np.all(np.diff(med.to_numpy()) >= 0))
    rows.append(literature_verdict(lit, monotone, med))
    if n:
        mpd = n["meta"]["mentions_per_document"]
        others = max(v for t, v in mpd.items() if t != "stability")
        rows.append({"id": "R2", "result": "\"Stability\" mentioned 6.4 times per document, others ≤ 0.33 (p. 11)",
                     "ours": f"{mpd['stability']:.1f} vs at most {others:.2f}",
                     "verdict": _range_verdict(P.MENTIONS_PER_DOC["stability"], imp.get("I2", {}).get("stability"),
                                               mpd["stability"] > others)})
        t2 = n["table2"]
        cl = imp.get("I3", {}).get("claims_point", {})
        held = sum(bool(v) for v in cl.values())
        rows.append({"id": "R3", "result": "Weights: statistics → capital; technology and conduct → scoring; provisions lowest (Table 2)",
                     "ours": f"{held} of {len(cl)} hold; levels about half (totals "
                         + " / ".join(f"{t2[u]['total']:.1f}" for u in USES) + "%)",
                     "verdict": "reproduced" if cl and held == len(cl) else ("close" if held >= 2 else "not reproduced")})
        sig_paper = {f"{a} vs {b}|{c}" for (a, b), cs in P.TABLE3_BOLD.items() for c in cs}
        sig_ours = {key for key, v in n["table3"].items() if v["p"] < 0.10}
        agree = len(sig_paper & sig_ours) + len(set(n["table3"]) - sig_paper - sig_ours)
        rows.append({"id": "R4", "result": "Six of nine use differences significant at 10% (Table 3)",
                     "ours": f"{len(sig_ours)} of 9 significant; agree with the paper on {agree} of 9",
                     "verdict": "reproduced" if agree >= 8 else ("close" if agree >= 6 else "not reproduced")})
        neg = n["neg_d10"]
        tech = neg["scoring"]["technology"] > max(neg[u]["technology"] for u in USES if u != "scoring")
        cond = neg["scoring"]["conduct"] > max(neg[u]["conduct"] for u in USES if u != "scoring")
        rows.append({"id": "R5", "result": "Sentiment: technology and conduct terms most negative in scoring texts (Tables 8-10)",
                     "ours": f"technology {'yes' if tech else 'no'}, conduct {'yes' if cond else 'no'} (d = 10)",
                     "verdict": "reproduced" if tech and cond else ("close" if tech or cond else "not reproduced")})
    sc = _json(RES / "reproduction" / "scorecard" / "checks.json")
    ex = sc.get("table4_example", {})
    rows.append({"id": "R6", "result": "Worked example: riskiness 0.523 / 0.554 / 0.536 (Table 4)",
                 "ours": "with the paper's weights: " + " / ".join(f"{ex[u]:.4f}" for u in USES),
                 "verdict": "reproduced" if all(abs(ex[u] - P.TABLE4_RISKINESS[u]) < 1e-3 for u in USES) else "not reproduced"})
    sd = {mm: m["proxies"][mm]["auc_sd"] for mm in MODELS5}
    stable_ok = max(sd["Lasso"], sd["XGBoost"]) < min(sd["CART"], sd["MLP"])
    rows.append({"id": "R7", "result": "Stability: Lasso, XGBoost most stable; CART, MLP least (Table 5)",
                 "ours": "order of SD(AUC): " + " < ".join(_order(sd)),
                 "verdict": "reproduced" if stable_ok else "not reproduced"})
    rows.append({"id": "R8", "result": "Hyperparameters: Lasso 0, CART 1, RF 2, XGBoost 2, MLP 5+ (Table 5)",
                 "ours": "a definition, taken as stated", "verdict": "not testable"})
    fit = {mm: m["proxies"][mm]["fit_s"] for mm in MODELS5}
    paper_fit = _order(P.TABLE5["training_seconds"])
    rows.append({"id": "R9", "result": "Training time: " + " < ".join(paper_fit) + " (Table 5)",
                 "ours": " < ".join(_order(fit)),
                 "verdict": "reproduced" if _order(fit) == paper_fit else
                 ("close" if _order(fit)[-1] == paper_fit[-1] else "not reproduced")})
    shp = {mm: m["proxies"][mm]["shap_s"] for mm in ("RF", "XGBoost", "MLP")}
    paper_shap = _order({mm: P.TABLE5["shap_seconds"][mm] for mm in ("RF", "XGBoost", "MLP")}, ["RF", "XGBoost", "MLP"])
    rows.append({"id": "R10", "result": "SHAP time: XGBoost 8 s < RF 60 s < MLP 2,000 s (Table 5)",
                 "ours": " < ".join(f"{mm} {shp[mm]:.1f} s" for mm in _order(shp, ["RF", "XGBoost", "MLP"])),
                 "verdict": "reproduced" if _order(shp, ["RF", "XGBoost", "MLP"]) == paper_shap else "not reproduced"})
    r = m["ranks_default"]
    top3 = all({"NumberOfTimes90DaysLate", "NumberOfTime30-59DaysPastDueNotWorse"} <= set(r[mm][:3]) for mm in ("RF", "XGBoost", "MLP"))
    rev = r["XGBoost"][0] == "RevolvingUtilizationOfUnsecuredLines" and all(r[mm][0] != "RevolvingUtilizationOfUnsecuredLines" for mm in ("RF", "MLP"))
    age = all(r[mm][3] == "age" for mm in ("RF", "XGBoost", "MLP"))
    held = top3 + rev + age
    rows.append({"id": "R11", "result": "SHAP: late payments top 3; revolving first only for XGBoost; age 4th (Figs. 6-8)",
                 "ours": f"{held} of 3 statements hold", "verdict": "reproduced" if held == 3 else ("close" if held == 2 else "not reproduced")})
    inside = [mm for mm in P.TABLE7_AUC if m["auc"][mm]["lo"] <= P.TABLE7_AUC[mm] / 100 <= m["auc"][mm]["hi"]]
    order_ok = max(MODELS5, key=lambda mm: m["auc"][mm]["auc"]) == "XGBoost"
    rows.append({"id": "R12", "result": "AUC: logit 80.0 to XGBoost 85.3; XGBoost best (Table 7)",
                 "ours": f"{len(inside)} of 6 paper AUCs inside our 95% intervals; best: "
                         f"{max(MODELS5, key=lambda mm: m['auc'][mm]['auc'])}",
                 "verdict": "reproduced" if len(inside) >= 5 and order_ok else ("close" if order_ok else "not reproduced")})
    i13 = imp.get("I13", {})
    rows.append({"id": "R13", "result": "Risk-reward: XGBoost most risk-efficient; provisions least risky use (Table 6, Fig. 5)",
                 "ours": k.get("r13_text", ""), "verdict": k.get("r13_verdict", "")})
    extra = _json(RES / "reproduction" / "extra.json")
    if extra:
        sp = extra["R14"]["spearman"]
        rows.append({"id": "R14", "result": "Per-document term shares (Fig. 9)",
                     "ours": f"rank correlation over {extra['R14']['documents']} documents: "
                             + " / ".join(f"{sp[c]['rho']:.2f}" for c in CATS),
                     "verdict": extra["R14"]["verdict"]})
        r15 = extra["R15"]
        rows.append({"id": "R15", "result": "Sentiment at d = 20 (Table 9)",
                     "ours": f"technology {'yes' if r15['scoring_highest_technology'] else 'no'}, "
                             f"conduct {'yes' if r15['scoring_highest_conduct'] else 'no'}",
                     "verdict": r15["verdict"]})
        diffs = extra["R16"]["differences_from_table5"]
        rows.append({"id": "R16", "result": "Darkness scores plotted in Fig. 4",
                     "ours": "equal to Table 5" if not diffs else
                     "differs from Table 5: " + "; ".join(f"{d['model']} {d['category']} {d['fig4']:.0f} vs {d['table5']}" for d in diffs),
                     "verdict": extra["R16"]["verdict"]})
    return rows


def literature_verdict(lit: pd.DataFrame, monotone: bool, med: pd.Series) -> dict:
    """R1, revised (pre-registration addendum): each labelled point checked against its source."""
    shape = (f"figure: max {lit['gain_pct'].max():.1f}%, class medians " + " / ".join(f"{v:.1f}" for v in med) + "%")
    src = ROOT / "data" / "fig1_sources.csv"
    row = {"id": "R1", "result": "Literature gains of 20 papers (Fig. 1)"}
    if not src.exists():
        return {**row, "ours": shape + "; sources not checked", "verdict": "not testable"}
    f = pd.read_csv(src)
    checked = f[f["status"].isin(["matches", "differs"])]
    n_match = int((checked["status"] == "matches").sum())
    share = n_match / len(checked) if len(checked) else 0.0
    counts = f["status"].value_counts().to_dict()
    verdict = ("reproduced" if len(checked) and share >= 0.75 else "close" if len(checked) and share >= 0.5
               else "not reproduced" if len(checked) else "not testable")
    detail = ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in counts.items())
    return {**row, "ours": f"{n_match} of {len(checked)} checkable points match their source ({detail})", "verdict": verdict,
            "shape": shape}


def _range_verdict(paper_value: float, interval: dict | None, direction: bool) -> str:
    if interval and interval["lo"] <= paper_value <= interval["hi"] and direction:
        return "reproduced"
    return "close" if direction else "not reproduced"
