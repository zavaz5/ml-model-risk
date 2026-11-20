"""Write results/report/key_numbers.json and the results section of README.md.

Every number of ours in the README, the slides and the speaker script comes
from key_numbers.json, which this script builds from results/. The paper's
numbers come from src/mrisk/paper_numbers.py. Paper and ours never share a
table (CLAUDE.md, rule 2).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mrisk import paper_numbers as P
from mrisk import plots
from mrisk.scorecard import MODELS5, models_block, nlp_block, verdicts

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
REPORT = RES / "report"
FIG = REPORT / "figures"
README = ROOT / "README.md"
START, END = "<!-- results:start -->", "<!-- results:end -->"
CATS, USES = P.CATEGORIES, P.USES
USE_NAME = {"capital": "Capital", "scoring": "Credit scoring", "provisions": "Provisions"}
CAT_NAME = {"statistics": "Statistics", "technology": "Technology", "conduct": "Market conduct"}


def _json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}%"


def md_table(frame: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, frame.columns)) + " |\n|" + "---|" * len(frame.columns) + "\n"
    return head + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in frame.itertuples(index=False))


# ------------------------------------------------------------------ numbers
def frontier(gain: dict, risk: dict) -> list[str]:
    out = []
    for m in gain:
        dominated = any(gain[k] >= gain[m] and risk[k] <= risk[m] and (gain[k] > gain[m] or risk[k] < risk[m])
                        for k in gain if k != m)
        if not dominated:
            out.append(m)
    return out


def r13(k: dict) -> tuple[str, str, dict]:
    """Paper's Z, our weights and our gains: who is risk-efficient, which use is least risky."""
    path = RES / "reproduction" / "scorecard" / "table6_our_weights.csv"
    if not path.exists():
        return "", "", {}
    t6 = pd.read_csv(path, index_col=0)
    gain = {m: k["models"]["auc"][m]["gain"] for m in MODELS5}
    fr = {u: frontier(gain, t6[u].to_dict()) for u in USES}
    lowest = {m: t6.loc[m].idxmin() for m in MODELS5}
    xgb_all = all("XGBoost" in fr[u] for u in USES)
    prov_all = all(v == "provisions" for v in lowest.values())
    text = (f"risk-efficient (paper's Z, our weights and gains): {', '.join(fr['capital'])}; "
            f"provisions least risky for {sum(v == 'provisions' for v in lowest.values())} of 5 models")
    verdict = "reproduced" if xgb_all and prov_all else ("close" if xgb_all or prov_all else "not reproduced")
    return text, verdict, {"frontier": fr, "lowest_use": lowest, "table6_ours": t6.to_dict()}


def improvements() -> dict:
    out = {}
    nlp = _json(RES / "improvements" / "nlp" / "summary.json")
    out.update({k: v for k, v in nlp.items()})
    mod = _json(RES / "improvements" / "models" / "summary.json")
    d = RES / "improvements" / "models"
    if (d / "i1_benchmark.csv").exists():
        i1 = pd.read_csv(d / "i1_benchmark.csv")
        out["I1"] = {"summary": i1[i1["model"] == "_summary"].set_index("dataset")[
            ["best", "runner_up", "best_vs_runner_diff", "best_vs_runner_p", "spearman"]].to_dict(orient="index"),
            "gains": {s: g[g["model"] != "_summary"].set_index("model")[["auc", "gain", "gain_lo", "gain_hi", "gain_p", "n", "default_rate"]]
                      .to_dict(orient="index") for s, g in i1.groupby("dataset")}}
    if (d / "i7_stability.csv").exists():
        out["I7"] = {"table": pd.read_csv(d / "i7_stability.csv").set_index("model").to_dict(orient="index"),
                     "claim_probability": mod.get("I7_claim_probability")}
    if (d / "i11_shap_ranks.csv").exists():
        out["I11"] = {"claims_seeds": mod.get("I11_claims_seeds")}
    if (d / "i12_calibration.csv").exists():
        out["I12"] = {"table": pd.read_csv(d / "i12_calibration.csv").set_index("model").to_dict(orient="index"),
                      "spearman_auc_vs_brier": mod.get("I12_spearman_auc_vs_brier")}
    risk = RES / "improvements" / "risk"
    if (risk / "i13_scorecard.csv").exists():
        out["I13"] = {"table": pd.read_csv(risk / "i13_scorecard.csv").to_dict(orient="records"),
                      **_json(risk / "summary.json")}
    return out


def build_numbers() -> dict:
    k = {"models": models_block(), "nlp": nlp_block(), "improvements": improvements(),
         "errata": _json(RES / "reproduction" / "errata.json") or [],
         "scorecard_checks": _json(RES / "reproduction" / "scorecard" / "checks.json")}
    text, verdict, extra = r13(k)
    k.update({"r13_text": text, "r13_verdict": verdict, "r13": extra})
    k["verdicts"] = verdicts(k)
    return k


# ------------------------------------------------------------------ figures
def figures(k: dict) -> dict:
    out = {}
    gain_paper = pd.Series({m: P.TABLE7_AUC[m] - P.TABLE7_AUC["Logit"] for m in MODELS5})
    out["rr_paper"] = plots.risk_reward(pd.DataFrame(P.TABLE6_RISK), gain_paper, FIG / "risk_reward_paper.png",
                                        "Paper: Table 6 risk and Table 7 gains")
    if k["r13"]:
        t6 = pd.DataFrame(k["r13"]["table6_ours"])
        a = k["models"]["auc"]
        gain = pd.Series({m: 100 * a[m]["gain"] for m in MODELS5})
        gain_se = pd.Series({m: 196 * a[m]["gain_se"] for m in MODELS5})
        out["rr_ours"] = plots.risk_reward(t6, gain, FIG / "risk_reward_ours.png",
                                           "Ours: paper's Z × our weights; our gains (95% CI)", gain_se=gain_se)
    if k["nlp"]:
        t2 = pd.DataFrame(k["nlp"]["table2"]).T / 100
        se = None
        i3 = RES / "improvements" / "nlp" / "i3_weights.csv"
        if i3.exists():
            frame = pd.read_csv(i3, index_col=[0, 1])
            se = frame.loc["se"]
        out["weights_ours"] = plots.weights_bars(t2, FIG / "weights_ours.png", "Ours: our corpus (± bootstrap s.e.)", se)
        out["weights_paper"] = plots.weights_bars(pd.DataFrame(P.TABLE2).T[CATS] / 100, FIG / "weights_paper.png", "Paper: Table 2")
    i1 = RES / "improvements" / "models" / "i1_benchmark.csv"
    if i1.exists():
        out["benchmark"] = plots.benchmark(pd.read_csv(i1), FIG / "benchmark.png")
    r = k["models"]["ranks_default"]
    out["shap_default"] = plots.shap_ranks({m: r[m] for m in ("RF", "XGBoost", "MLP")}, FIG / "shap_default.png",
                                           "Ours, each library's default explainer (XGBoost in log-odds)")
    rc = k["models"]["ranks_common"]
    out["shap_common"] = plots.shap_ranks({m: rc[m] for m in ("RF", "XGBoost", "MLP")}, FIG / "shap_common.png",
                                          "Ours, one explainer, probability scale")
    return {key: str(p.relative_to(ROOT)) for key, p in out.items()}


# ------------------------------------------------------------------ README
def section_models(k: dict) -> str:
    m = k["models"]
    a, px = m["auc"], m["proxies"]
    paper7 = pd.DataFrame([(mm, f"{P.TABLE7_AUC[mm]:.1f}", f"[{P.TABLE7_CI[mm][0]:.1f}, {P.TABLE7_CI[mm][1]:.1f}]",
                            "–" if mm == "Logit" else f"{P.TABLE7_AUC[mm] - P.TABLE7_AUC['Logit']:+.1f}") for mm in P.TABLE7_AUC],
                          columns=["Model", "AUC (%)", "95% interval", "Gain over logit (points)"])
    ours7 = pd.DataFrame([(mm, f"{100 * a[mm]['auc']:.1f} ± {100 * a[mm]['se']:.1f}", f"[{100 * a[mm]['lo']:.1f}, {100 * a[mm]['hi']:.1f}]",
                           "–" if mm == "Logit" else f"{100 * a[mm]['gain']:+.1f} ± {100 * a[mm]['gain_se']:.1f} (p = {a[mm]['gain_p']:.3f})")
                          for mm in a], columns=["Model", "AUC (%, ± s.e.)", "95% interval", "Gain over logit (points, paired DeLong)"])
    paper5 = pd.DataFrame([(mm, P.TABLE5["stability_sd"][mm], P.TABLE5["n_hyperparameters"][mm], P.TABLE5["training_seconds"][mm],
                            P.TABLE5["transparency"][mm], P.TABLE5["shap_seconds"][mm],
                            f"{P.TABLE5['z_statistics'][mm]} / {P.TABLE5['z_technology'][mm]} / {P.TABLE5['z_conduct'][mm]}")
                           for mm in MODELS5],
                          columns=["Model", "SD of AUC", "Hyperparameters", "Training (s)", "Transparency", "SHAP (s)", "Z: stat. / tech. / conduct"])
    ours5 = pd.DataFrame([(mm, f"{px[mm]['auc_sd']:.4f} ± {px[mm]['auc_sd_se']:.4f}", f"{a[mm]['params']:,}",
                           f"{px[mm]['fit_s']:.2f} [{px[mm]['fit_q25']:.2f}, {px[mm]['fit_q75']:.2f}]", f"{px[mm]['shap_s']:.1f}",
                           f"{px[mm]['shap_common_s']:.1f}") for mm in MODELS5],
                         columns=["Model", "SD of AUC (± s.e.)", "Fitted parameters", "Training (s), median [IQR]",
                                  "SHAP (s), default explainer", "SHAP (s), one explainer"])
    meta = m["meta"]
    env = meta["environment"]
    return f"""### Data and setup (ours)

Give Me Some Credit: {meta['rows_raw']:,} rows in `cs-training.csv`, of which {meta['rows']:,} complete cases (the paper: 120,000); default rate {pct(meta['default_rate'], 2)}. {k['models']['n_features']} features (10 characteristics and their squares). Stratified 80/20 split: {meta['train']:,} training and {meta['test']:,} test rows. Lasso penalty chosen by 5-fold CV on the training rows: C = {meta['lasso_c']:g}. Timings on {env['cpu']}, {env['os']}, Python {env['python']}.

**What is fitted, chosen and evaluated.** Each model is fitted to the training rows by its own loss (log-loss for the logit, Lasso, MLP and XGBoost; Gini impurity for the trees). Only the Lasso penalty is chosen (5-fold cross-validated AUC on the training rows); every other hyperparameter is the paper's. Everything is evaluated on the test rows. The darkness scores Z are the paper's expert judgement; nothing in phase 2 is fitted.

### Prediction (Table 7)

**Paper** (its data and split):

{md_table(paper7)}

**Ours** (± DeLong standard error; the gain test is paired, on the same test rows):

{md_table(ours7)}

### Model-risk proxies (Table 5)

**Paper:**

{md_table(paper5)}

**Ours** (SD over {meta['retrains']} retrains on 60–100% subsamples, bootstrap s.e.; times single-threaded, median of 5 fits after a warm-up; SHAP for {meta['shap_rows']:,} test rows):

{md_table(ours5)}
"""


def section_shap(k: dict) -> str:
    r = k["models"]["ranks_default"]
    paper = pd.DataFrame({"Rank": [1, 2, 3, 4],
                          "RF (Fig. 6)": ["90+ days late", "30–59 days", "60–89 days", "age"],
                          "XGBoost (Fig. 7)": ["Revolving utilisation", "30–59 days", "90+ days late", "age"],
                          "MLP (Fig. 8)": ["30–59 days", "90+ days late", "60–89 days", "age"]})
    ours = pd.DataFrame({"Rank": [1, 2, 3, 4], **{f"{m}": r[m][:4] for m in ("RF", "XGBoost", "MLP")}})
    return f"""### SHAP feature rankings (Figs. 6–8)

**Paper** (top four, read from the figures):

{md_table(paper)}

**Ours** (mean |SHAP| per characteristic, x and x² summed; each library's default explainer):

{md_table(ours)}
"""


def section_nlp(k: dict) -> str:
    n = k["nlp"]
    if not n:
        return ""
    paper2 = pd.DataFrame([(USE_NAME[u], *[f"{P.TABLE2[u][c]:.2f}" for c in CATS], f"{P.TABLE2[u]['total']:.2f}") for u in USES],
                          columns=["Use", *[CAT_NAME[c] for c in CATS], "Total"])
    i3 = RES / "improvements" / "nlp" / "i3_weights.csv"
    se = pd.read_csv(i3, index_col=[0, 1]).loc["se"] if i3.exists() else None
    fmt = lambda u, c: f"{n['table2'][u][c]:.2f}" + ("" if se is None else f" ± {100 * se.loc[u, c]:.2f}")
    ours2 = pd.DataFrame([(USE_NAME[u], *[fmt(u, c) for c in CATS], fmt(u, "total")) for u in USES],
                         columns=["Use", *[CAT_NAME[c] for c in CATS], "Total"])
    paper3 = pd.DataFrame([(f"{a} vs {b}", *[f"{P.TABLE3[(a, b)][c]:.4f}" + (" *" if c in P.TABLE3_BOLD[(a, b)] else "") for c in CATS])
                           for (a, b) in P.TABLE3], columns=["Uses", *[CAT_NAME[c] for c in CATS]])
    ours3 = pd.DataFrame([(f"{a} vs {b}", *[f"{n['table3'][f'{a} vs {b}|{c}']['p']:.4f}" + (" *" if n['table3'][f'{a} vs {b}|{c}']['p'] < 0.10 else "")
                                            for c in CATS]) for (a, b) in P.TABLE3], columns=["Uses", *[CAT_NAME[c] for c in CATS]])
    paper8 = pd.DataFrame([(USE_NAME[u], *[f"{P.TABLE8_D10[u][c]:.2f}" for c in CATS], f"{P.TABLE10_NEG_SHARE[u]:.2f}%") for u in USES],
                          columns=["Use", *[CAT_NAME[c] for c in CATS], "Negative words, all text"])
    ours8 = pd.DataFrame([(USE_NAME[u], *[f"{n['neg_d10'][u][c]:.2f}" for c in CATS], f"{n['neg_share'][u]:.2f}%") for u in USES],
                         columns=["Use", *[CAT_NAME[c] for c in CATS], "Negative words, all text"])
    meta = n["meta"]
    docs = meta["documents"]
    mpd = meta["mentions_per_document"]
    return f"""### Weights from regulatory texts (Table 2, Fig. 3)

Our corpus: {docs['capital']} capital, {docs['scoring']} credit-scoring, {docs['provisions']} provisioning and {docs['common']} common-area documents ({meta['tokens_total']:,} words); missing: {', '.join(meta['missing']) or 'none'}. Sources, versions and hashes: `data/corpus_manifest.csv`.

**Paper** (percent of all words):

{md_table(paper2)}

**Ours** (percent of all words, ± document-bootstrap s.e.):

{md_table(ours2)}

Mentions per document (p. 11). Paper: stability {P.MENTIONS_PER_DOC['stability']}, feature engineering {P.MENTIONS_PER_DOC['feature engineering']}, hyperparameters {P.MENTIONS_PER_DOC['hyperparameters']}, overfitting {P.MENTIONS_PER_DOC['overfitting']}, dynamic calibration {P.MENTIONS_PER_DOC['dynamic calibration']}. Ours: stability {mpd['stability']:.2f}, feature engineering {mpd['feature engineering']:.2f}, hyperparameters {mpd['hyperparameters']:.2f}, overfitting {mpd['overfitting']:.2f}, dynamic calibration {mpd['dynamic calibration']:.2f}.

### Are the differences significant? (Table 3)

**Paper** (p-values; * marked significant at 10%):

{md_table(paper3)}

**Ours** (Welch t-test p-values; * below 0.10):

{md_table(ours3)}

### Sentiment robustness (Tables 8 and 10)

**Paper** (negative words within 10 words of a mention, per mention):

{md_table(paper8)}

**Ours:**

{md_table(ours8)}
"""


def section_verdicts(k: dict) -> str:
    v = pd.DataFrame(k["verdicts"])[["id", "result", "ours", "verdict"]]
    v.columns = ["#", "Paper's result", "What we find", "Verdict"]
    errata = pd.DataFrame(k["errata"])[["id", "where", "what"]] if k["errata"] else pd.DataFrame()
    if len(errata):
        errata.columns = ["#", "Where", "Inconsistency"]
    return f"""### Verdicts

Rules from `docs/preregistration.md`: *reproduced* if the paper's value is inside our 95% interval (or, for an ordering, our point estimates give the same order); *close* if it is outside but the conclusion drawn from it holds; *not reproduced* if the conclusion does not hold; *not testable* if the paper gives too little to recompute it.

{md_table(v)}

### Errata: the paper against itself

Found by `scripts/reproduce_scorecard.py` from the paper's own numbers (no data of ours involved).

{md_table(errata) if len(errata) else 'None found.'}
"""


def section_improvements(k: dict) -> str:
    imp = k["improvements"]
    rows = []
    for row in improvement_rows(k):
        rows.append((row["id"], row["what"], row["finding"]))
    t = pd.DataFrame(rows, columns=["#", "Improvement", "Finding"])
    return f"""### Improvements, one per result

Pre-registered in `docs/preregistration.md` before they were run.

{md_table(t)}
"""


def improvement_rows(k: dict) -> list[dict]:
    """One line per improvement: what we did and what came out. Shared with the slides."""
    imp, rows = k["improvements"], []
    if "I1" in imp:
        s = imp["I1"]["summary"]
        txt = "; ".join(f"{n}: best {v['best']} (vs {v['runner_up']} p = {v['best_vs_runner_p']:.3f}), "
                        f"Spearman(complexity, gain) {v['spearman']:+.2f}" for n, v in s.items())
        rows.append({"id": "I1", "what": "Same-protocol benchmark on three datasets with paired DeLong intervals", "finding": txt})
    if "I2" in imp:
        st = imp["I2"]["stability"]
        rows.append({"id": "I2", "what": "Bootstrap intervals for the mentions per document",
                     "finding": f"stability {st['mean']:.2f} [{st['lo']:.2f}, {st['hi']:.2f}] per document; "
                                f"{pct(st['top3_share'], 0)} of all mentions in the top three documents"})
    if "I3" in imp:
        p3 = imp["I3"]["claim_probability"]
        tr = imp["I3"]["claims_trimmed"]
        rows.append({"id": "I3", "what": "Document bootstrap of the weights; dictionary audit",
                     "finding": "; ".join(f"{c.replace('_', ' ')}: P = {p:.2f}{'' if tr[c] else ' (fails with trimmed dictionary)'}"
                                          for c, p in p3.items())
                                + f". Dominant terms: {', '.join(t.split('|')[1] for t in imp['I3']['dominant_terms'])}"})
    if "I4" in imp:
        i4 = imp["I4"]
        rows.append({"id": "I4", "what": "Permutation tests, Holm and Benjamini–Hochberg over nine tests",
                     "finding": f"{i4['welch_10pct']} of 9 below 10% (Welch); after Holm: {i4['holm_10pct']} at 10%, {i4['holm_5pct']} at 5%; "
                                f"after BH: {i4['bh_10pct']} at 10%"})
    if "I5" in imp:
        i5 = imp["I5"]
        rows.append({"id": "I5", "what": "Negative words relative to the base rate (lift), bootstrap",
                     "finding": f"P(scoring has the highest lift): technology {i5['p_scoring_highest_technology']:.2f}, "
                                f"conduct {i5['p_scoring_highest_conduct']:.2f}"})
    if "I6" in imp:
        i6 = imp["I6"]
        rows.append({"id": "I6", "what": "Worked example with weight uncertainty",
                     "finding": " / ".join(f"{i6[u]['est']:.3f} ± {i6[u]['se']:.3f}" for u in USES)
                                + f"; P(scoring riskiest) = {i6['p_scoring_riskiest']:.2f}"})
    if "I7" in imp:
        cp = imp["I7"]["claim_probability"]
        rows.append({"id": "I7", "what": "Both stability definitions, paired bootstrap",
                     "finding": f"P(Lasso and XGBoost more stable than CART and MLP): AUC spread {cp['auc_sd']:.2f}, PD spread {cp['pd_sd']:.2f}"})
    a = k["models"]["auc"]
    from scipy.stats import spearmanr
    rho = spearmanr([P.TABLE5["n_hyperparameters"][m] for m in MODELS5], [a[m]["params"] for m in MODELS5]).statistic
    rows.append({"id": "I8", "what": "Fitted parameters instead of a hyperparameter count",
                 "finding": "; ".join(f"{m} {a[m]['params']:,}" for m in MODELS5) + f" (Spearman with the paper's count {rho:+.2f})"})
    px = k["models"]["proxies"]
    rel = {m: px[m]["fit_s"] / px["Logit"]["fit_s"] if px["Logit"]["fit_s"] > 0 else np.nan for m in MODELS5}
    rows.append({"id": "I9", "what": "Timing hygiene: single thread, warm-up, median and IQR; relative to the logit",
                 "finding": "; ".join(f"{m} ×{rel[m]:,.0f}" for m in sorted(MODELS5, key=rel.get))})
    rows.append({"id": "I10", "what": "One SHAP explainer and scale for every model",
                 "finding": "; ".join(f"{m} {px[m]['shap_common_s']:.0f} s" for m in sorted(MODELS5, key=lambda m: px[m]["shap_common_s"]))})
    if "I11" in imp:
        c = imp["I11"]["claims_seeds"]
        rows.append({"id": "I11", "what": "SHAP rankings on the probability scale, five seeds",
                     "finding": "seeds where the claim holds, default explainer → probability scale: "
                                + "; ".join(f"{name.replace('_', ' ')} {c['default'][name]}→{c['probability'][name]} of {c['default']['seeds']}"
                                            for name in ("top3_late_payments", "revolving_first_only_xgb", "age_fourth"))})
    if "I12" in imp:
        t = imp["I12"]["table"]
        best_b = min(t, key=lambda m: t[m]["brier"])
        rows.append({"id": "I12", "what": "Calibration next to AUC (Brier, ECE, slope)",
                     "finding": f"best Brier: {best_b}; Spearman(AUC, −Brier) = {imp['I12']['spearman_auc_vs_brier']:+.2f}; "
                                + "; ".join(f"{m} slope {t[m]['slope']:.2f}" for m in MODELS5)})
    if "I13" in imp:
        i13 = imp["I13"]
        tab = pd.DataFrame(i13["table"])
        pf = tab.groupby("model")["p_frontier"].min()
        rows.append({"id": "I13", "what": "Rule-based Z and Monte Carlo over all inputs",
                     "finding": "P(on the frontier, worst use): " + "; ".join(f"{m} {pf[m]:.2f}" for m in MODELS5)
                                + f". P(highest gain): XGBoost {i13['p_best_gain']['XGBoost']:.2f}"})
    return rows


def write_readme(section: str) -> None:
    text = README.read_text() if README.exists() else ""
    block = f"{START}\n<!-- Generated by scripts/make_report.py from results/. Do not edit by hand. -->\n\n{section}\n{END}"
    if START in text:
        text = text[: text.index(START)] + block + text[text.index(END) + len(END):]
    else:
        text += "\n## Results\n\n" + block + "\n"
    README.write_text(text)


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    k = build_numbers()
    k["figures"] = figures(k)
    k["improvement_rows"] = improvement_rows(k)
    (REPORT / "key_numbers.json").write_text(json.dumps(k, indent=2, default=float))
    figs = k["figures"]
    pics = ""
    if "rr_ours" in figs:
        pics = f"\n![Risk-reward, paper]({figs['rr_paper']}) ![Risk-reward, ours]({figs['rr_ours']})\n"
    section = "\n".join([section_models(k), section_shap(k), section_nlp(k), pics, section_verdicts(k), section_improvements(k)])
    write_readme(section)
    print(pd.DataFrame(k["verdicts"])[["id", "verdict", "ours"]].to_string(index=False))


if __name__ == "__main__":
    main()
