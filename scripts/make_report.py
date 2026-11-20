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
SHORT = {"NumberOfTimes90DaysLate": "90+ days late", "NumberOfTime30-59DaysPastDueNotWorse": "30–59 days",
         "NumberOfTime60-89DaysPastDueNotWorse": "60–89 days", "RevolvingUtilizationOfUnsecuredLines": "Revolving utilisation",
         "age": "age", "DebtRatio": "Debt ratio", "MonthlyIncome": "Monthly income",
         "NumberOfOpenCreditLinesAndLoans": "Open credit lines", "NumberRealEstateLoansOrLines": "Real-estate loans",
         "NumberOfDependents": "Dependents"}
CAT_NAME = {"statistics": "Statistics", "technology": "Technology", "conduct": "Market conduct"}


def _json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}%"


def pval(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


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
    text = (f"frontier: {', '.join(fr['capital'])}; provisions least risky for "
            f"{sum(v == 'provisions' for v in lowest.values())} of 5 models")
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


def appendix_block() -> dict:
    d = RES / "improvements" / "appendix"
    if not (d / "appendix_factors.csv").exists():
        return {}
    out = {"factors": pd.read_csv(d / "appendix_factors.csv").set_index("model").to_dict(orient="index"),
           **_json(d / "summary.json")}
    if (d / "i14_extended_scorecard.csv").exists():
        out["i14"] = pd.read_csv(d / "i14_extended_scorecard.csv").set_index("model").to_dict(orient="index")
        out.update(_json(d / "i14_summary.json"))
    return out


def build_numbers() -> dict:
    k = {"models": models_block(), "nlp": nlp_block(), "improvements": improvements(),
         "errata": _json(RES / "reproduction" / "errata.json") or [],
         "scorecard_checks": _json(RES / "reproduction" / "scorecard" / "checks.json"),
         "extra": _json(RES / "reproduction" / "extra.json"), "appendix": appendix_block()}
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
    f9 = RES / "reproduction" / "nlp" / "fig9_comparison.csv"
    if f9.exists():
        out["fig9"] = plots.fig9_scatter(pd.read_csv(f9, header=[0, 1], index_col=0), FIG / "fig9_ranks.png")
    r = k["models"]["ranks_default"]
    out["shap_default"] = plots.shap_ranks({m: r[m] for m in ("RF", "XGBoost", "MLP")}, FIG / "shap_default.png",
                                           "Ours, each library's default explainer (XGBoost in log-odds)")
    rc = k["models"]["ranks_common"]
    out["shap_common"] = plots.shap_ranks({m: rc[m] for m in ("RF", "XGBoost", "MLP")}, FIG / "shap_common.png",
                                          "Ours, one explainer, probability scale")
    return {key: str(p.relative_to(ROOT)) for key, p in out.items()}


# ------------------------------------------------------------------ README
def cart_explanation(cart: dict) -> str:
    """The CART diagnostic in one paragraph (shared with the slides)."""
    h = cart["hypotheses"]
    small = [v for key, v in h.items() if key.startswith("H2") and 8 <= v["leaves"] <= 12]
    d10, ours = h["H1 depth 10"], h["H0 depth 7 (ours)"]
    others = [h["H3 depth 7, balanced class weights"], h["H4 depth 7, entropy"], h["H5 depth 7, row index as a feature"]]
    return (f"**Why our CART is far above the paper's.** We tried every reading of the paper's set-up we could think of, on the "
            f"validation rows only. A depth-7 tree scores {100 * ours['auc']:.1f}; balanced class weights, entropy or the file's row "
            f"index as a feature keep it at {100 * min(o['auc'] for o in others):.1f}–{100 * max(o['auc'] for o in others):.1f}, and "
            f"scoring by hard labels drops it to {100 * h['H6 depth 7, hard labels']['auc']:.1f}. Only two kinds of tree reach the "
            f"paper's 80.4: a depth-10 tree ({100 * d10['auc']:.1f}, but its spread over refits, {d10['sd_refits']:.3f}, is twice the "
            f"paper's 0.005) and a small pruned tree of 8–12 leaves, as R's rpart builds by default "
            f"({100 * min(v['auc'] for v in small):.1f}–{100 * max(v['auc'] for v in small):.1f}). Since the other four models match the "
            "paper, the data and split are comparable: the paper's tree was most likely not the depth-7 tree it reports "
            "(`scripts/diagnose_cart.py`).")


def section_models(k: dict) -> str:
    m = k["models"]
    a, px = m["auc"], m["proxies"]
    paper7 = pd.DataFrame([(mm, f"{P.TABLE7_AUC[mm]:.1f}", f"[{P.TABLE7_CI[mm][0]:.1f}, {P.TABLE7_CI[mm][1]:.1f}]",
                            "–" if mm == "Logit" else f"{P.TABLE7_AUC[mm] - P.TABLE7_AUC['Logit']:+.1f}") for mm in P.TABLE7_AUC],
                          columns=["Model", "AUC (%)", "95% interval", "Gain over logit (points)"])
    ours7 = pd.DataFrame([(mm, f"{100 * a[mm]['auc']:.1f} ± {100 * a[mm]['se']:.1f}", f"[{100 * a[mm]['lo']:.1f}, {100 * a[mm]['hi']:.1f}]",
                           "–" if mm == "Logit" else f"{100 * a[mm]['gain']:+.2f} ± {100 * a[mm]['gain_se']:.2f} ({pval(a[mm]['gain_p'])})")
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
    cart = _json(RES / "reproduction" / "models" / "cart_diagnostic.json")
    cart_txt = ""
    if cart:
        cart_txt = "\n" + cart_explanation(cart) + "\n"
    return f"""### Data and setup (ours)

Give Me Some Credit: {meta['rows_raw']:,} rows in `cs-training.csv`, of which {meta['rows']:,} complete cases (the paper: 120,000); default rate {pct(meta['default_rate'], 2)}. {k['models']['n_features']} features (10 characteristics and their squares). Stratified 80/20 split: {meta['train']:,} training and {meta['test']:,} test rows. Lasso penalty chosen by 5-fold CV on the training rows: C = {meta['lasso_c']:g}. Timings on {env['cpu']}, {env['os']}, Python {env['python']}.

**What is fitted, chosen and evaluated.** Each model is fitted to the training rows by its own loss (log-loss for the logit, Lasso, MLP and XGBoost; Gini impurity for the trees). Only the Lasso penalty is chosen (5-fold cross-validated AUC on the training rows); every other hyperparameter is the paper's. Everything is evaluated on the test rows. The darkness scores Z are the paper's expert judgement; nothing in phase 2 is fitted.

### Prediction (Table 7)

**Paper** (its data and split):

{md_table(paper7)}

**Ours** (± DeLong standard error; the gain test is paired, on the same test rows):

{md_table(ours7)}
{cart_txt}
### Model-risk proxies (Table 5)

**Paper:**

{md_table(paper5)}

**Ours** (SD over {meta['retrains']} retrains on 60–100% subsamples, bootstrap s.e.; times single-threaded, median of 5 fits after a warm-up; SHAP for {meta['shap_rows']:,} test rows):

{md_table(ours5)}

**Times depend on the machine and the software, not on the method.** The paper reports no hardware, library, solver or explainer. Where the software is comparable, times agree (random-forest SHAP {px['RF']['shap_s']:.0f} s vs the paper's {P.TABLE5['shap_seconds']['RF']} s); where it differs, they do not (the Lasso's saga solver, the MLP's early stopping and its permutation explainer). The verdicts for R9 and R10 below record the different orderings; their cause is the computing set-up.
"""


def section_shap(k: dict) -> str:
    r = k["models"]["ranks_default"]
    paper = pd.DataFrame({"Rank": [1, 2, 3, 4],
                          "RF (Fig. 6)": ["90+ days late", "30–59 days", "60–89 days", "age"],
                          "XGBoost (Fig. 7)": ["Revolving utilisation", "30–59 days", "90+ days late", "age"],
                          "MLP (Fig. 8)": ["30–59 days", "90+ days late", "60–89 days", "age"]})
    ours = pd.DataFrame({"Rank": [1, 2, 3, 4], **{m: [SHORT.get(f, f) for f in r[m][:4]] for m in ("RF", "XGBoost", "MLP")}})
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
    sw = _json(RES / "reproduction" / "nlp" / "stopword_diagnostic.json")
    diag = ""
    if sw:
        diag = (f"\n**Why our levels are about half the paper's.** Every ordering matches, but our shares are roughly half. "
                f"If the denominator excludes English stop words (a step the paper does not mention; {pct(1 - sw['kept_share_mean'], 0)} of our tokens), "
                f"the totals become " + " / ".join(f"{sw['table2_pct'][u]['total']:.1f}%" for u in USES)
                + " (paper " + " / ".join(f"{P.TABLE2[u]['total']:.1f}%" for u in USES) + ") and the share of negative words "
                + " / ".join(f"{sw['neg_share_pct'][u]:.2f}%" for u in USES) + " (paper "
                + " / ".join(f"{P.TABLE10_NEG_SHARE[u]:.2f}%" for u in USES) + "). "
                "So the paper most likely divided by non-stop-words. Our main results keep all words, as the paper's text describes "
                "(`scripts/diagnose_nlp_levels.py`).\n")
    return f"""### Weights from regulatory texts (Table 2, Fig. 3)

Our corpus: {docs['capital']} capital, {docs['scoring']} credit-scoring, {docs['provisions']} provisioning and {docs['common']} common-area documents ({meta['tokens_total']:,} words); missing: {', '.join(meta['missing']) or 'none'}. Sources, versions and hashes: `data/corpus_manifest.csv`.

**Paper** (percent of all words):

{md_table(paper2)}

**Ours** (percent of all words, ± document-bootstrap s.e.):

{md_table(ours2)}

{diag}
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


def section_extra(k: dict) -> str:
    x = k.get("extra")
    if not x:
        return ""
    r14 = x["R14"]
    f9 = pd.DataFrame([(CAT_NAME[c], f"{r14['spearman'][c]['rho']:.2f} [{r14['spearman'][c]['lo']:.2f}, {r14['spearman'][c]['hi']:.2f}]",
                        f"{r14['level_ratio'][c]:.2f}") for c in CATS],
                      columns=["Category", "Spearman ρ, paper vs ours [95% bootstrap]", "Median ratio, paper / ours"])
    paper9 = pd.DataFrame([(USE_NAME[u], *[f"{P.TABLE9_D20[u][c]:.2f}" for c in CATS]) for u in USES],
                          columns=["Use", *[CAT_NAME[c] for c in CATS]])
    ours9 = pd.DataFrame([(USE_NAME[u], *[f"{x['R15']['ours'][u][c]:.2f}" for c in CATS]) for u in USES],
                         columns=["Use", *[CAT_NAME[c] for c in CATS]])
    f4 = pd.DataFrame([(m, *[f"{x['R16']['scores'][m][c]} / {P.TABLE5['z_' + c][m]}" for c in CATS]) for m in MODELS5],
                      columns=["Model", *[f"{CAT_NAME[c]}: Fig. 4 / Table 5" for c in CATS]])
    unl = x["R14"].get("unlabelled_series")
    return f"""### Document by document (Fig. 9)

Values read off the paper's vector figure for the {r14['documents']} documents we also have, against our shares. Levels differ (we divide by all words; see above), so the test is the rank order across documents:

{md_table(f9)}
{'' if not unl else f"One series in the paper's common-area panel has no label; its values (statistics {unl['statistics']:.4f}) match IIF (2018), the document with the highest statistics share in our counts too (erratum E11)."}

### Sentiment at d = 20 (Table 9)

**Paper:**

{md_table(paper9)}

**Ours:**

{md_table(ours9)}

### What Fig. 4 plots (the paper's numbers only)

The radar chart read off the PDF, against Table 5 (Fig. 4 / Table 5):

{md_table(f4)}
"""


def section_appendix(k: dict) -> str:
    ap = k.get("appendix")
    if not ap:
        return ""
    f = ap["factors"]
    t = pd.DataFrame([(m, f"{f[m]['auc_fit']:.3f} − {f[m]['auc_val']:.3f} = {f[m]['overfit_gap']:+.3f}",
                       " / ".join(f"{f[m][f'curve_{p}']:.3f}" for p in (10, 25, 50, 100)), f"{f[m]['features_used']} of 20",
                       f"{f[m]['fe_sensitivity']:.4f}", f"{f[m]['surrogate_r2']:.3f}", f"{100 * f[m]['flip_rate']:.2f}%")
                      for m in f],
                     columns=["Model", "A1 AUC fit − validation", "A1 learning curve (10/25/50/100%)", "A2 inputs used",
                              "A3 largest ΔAUC", "A4 surrogate R²", "A5 decisions flipped by ±10 years of age"])
    fr = ap.get("frontier", {})
    return f"""### The paper's remaining risk factors (appendix, A1–A5) and the extended scorecard (I14)

The paper names 13 risk factors and measures 5. Its appendix proposes measures for the rest; we computed every one public data allow, on the fit and validation rows only. Third-party providers, cyber risk and privacy need a bank's IT set-up and data inventory, so we do not invent proxies for them.

{md_table(t)}

A3's variants: no squared features, winsorising at the 1st/99th percentiles, log(1 + x), and median imputation that adds the {k['models']['meta']['rows_raw'] - k['models']['meta']['rows']:,} incomplete rows. A5 rejects the 10% highest predicted PDs and counts decisions that change when only age moves by ±10 years.

**Exploratory, not pre-registered: is the paper's logit a fair benchmark?** A3 shows that log(1 + x) lifts the logit by about four points while leaving the tree ensembles unchanged. On the validation rows, {next((r['finding'] for r in improvement_rows(k) if r['id'] == 'X1'), '')}. Most of the ML "gain" in the paper's setting comes from giving the logit raw, heavy-tailed ratios (`scripts/explore_logit_benchmark.py`).

**I14, the rule-based scorecard with these factors:** risk-gain frontier per use: {'; '.join(f"{USE_NAME[u]}: {', '.join(v)}" for u, v in fr.items())}.
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
        names = {"gmsc": "GMSC", "taiwan_default": "Taiwan", "german_credit": "German (1,000 loans)"}
        txt = "; ".join(f"{names[n]}: best {v['best']} (vs {v['runner_up']}, {pval(v['best_vs_runner_p'])})"
                        for n, v in s.items()) + ". Spearman(complexity, gain): " + ", ".join(
            f"{names[n].split(' ')[0]} {v['spearman']:+.2f}" for n, v in s.items())
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
                                + ". Terms above 10% of their category: " + ", ".join(
                                    f"{t.split('|')[1]} ({pct(imp['I3']['dominant_shares'][t], 0)} of {t.split('|')[0]})"
                                    for t in sorted(imp['I3']['dominant_terms'], key=lambda t: -imp['I3']['dominant_shares'][t]))})
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
                                + "; ".join(f"{label} {c['default'][name]}→{c['probability'][name]} of {c['default']['seeds']}"
                                            for name, label in (("top3_late_payments", "late payments in the top three"),
                                                                ("revolving_first_only_xgb", "revolving utilisation first only for XGBoost"),
                                                                ("age_fourth", "age fourth")))})
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
        lowest = i13["lowest_risk_use_by_model"]
        n_prov = sum(v == "provisions" for v in lowest.values())
        rows.append({"id": "I13", "what": "Rule-based Z and Monte Carlo over all inputs",
                     "finding": "P(on the frontier, worst use): " + "; ".join(f"{m} {pf[m]:.2f}" for m in MODELS5)
                                + f". P(highest gain): XGBoost {i13['p_best_gain']['XGBoost']:.2f}. On the normalised scale, "
                                f"provisions is the least risky use for {n_prov} of 5 models: the paper's ordering of uses "
                                "comes from provisioning texts using fewer risk terms overall"})
    ap = k.get("appendix")
    if ap:
        f = ap["factors"]
        worst = lambda key: max(MODELS5, key=lambda m: f[m][key])
        rows.append({"id": "A1–A5", "what": "The paper's remaining risk factors, measured as its appendix proposes",
                     "finding": f"largest overfitting gap {worst('overfit_gap')} ({f[worst('overfit_gap')]['overfit_gap']:+.3f}); "
                                f"most sensitive to preprocessing {worst('fe_sensitivity')}; hardest to imitate with a surrogate "
                                f"{min(MODELS5, key=lambda m: f[m]['surrogate_r2'])} (R² {min(f[m]['surrogate_r2'] for m in MODELS5):.2f}); "
                                f"most age-sensitive {worst('flip_rate')} ({100 * f[worst('flip_rate')]['flip_rate']:.1f}% of decisions flip)"})
        lb = _json(RES / "improvements" / "appendix" / "logit_benchmark.json")
        if lb:
            a1, a2 = lb["xgb_minus_logit_paper"], lb["xgb_minus_logit_log1p"]
            rows.append({"id": "X1", "what": "Exploratory (not pre-registered): a logit with log-transformed inputs as the benchmark",
                         "finding": f"logit AUC {100 * lb['auc']['Logit, paper inputs']['auc']:.1f} → {100 * lb['auc']['Logit, log1p inputs']['auc']:.1f} "
                                    f"on validation rows; XGBoost's lead shrinks from {100 * a1['diff']:.2f} ± {100 * a1['se']:.2f} "
                                    f"({pval(a1['p'])}) to {100 * a2['diff']:.2f} ± {100 * a2['se']:.2f} points ({pval(a2['p'])})"})
        if "frontier" in ap:
            rows.append({"id": "I14", "what": "Rule-based scorecard with the appendix factors",
                         "finding": "frontier: " + "; ".join(f"{u} {', '.join(v)}" for u, v in ap["frontier"].items())})
    return rows


def write_readme(section: str) -> None:
    text = README.read_text() if README.exists() else ""
    block = f"{START}\n<!-- Generated by scripts/make_report.py from results/. Do not edit by hand. -->\n\n{section}\n{END}"
    if START in text:
        text = text[: text.index(START)] + block + text[text.index(END) + len(END):]
    else:
        text += ("" if "## Results" in text else "\n## Results\n") + "\n" + block + "\n"
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
    section = "\n".join([section_models(k), section_shap(k), section_nlp(k), section_extra(k), pics, section_verdicts(k),
                         section_improvements(k), section_appendix(k)])
    write_readme(section)
    print(pd.DataFrame(k["verdicts"])[["id", "verdict", "ours"]].to_string(index=False))


if __name__ == "__main__":
    main()
