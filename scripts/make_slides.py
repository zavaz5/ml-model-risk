"""Build docs/presentation.pptx on the seminar's HSE template.

Numbers come from results/report/key_numbers.json (ours, written by
make_report.py) and src/mrisk/paper_numbers.py (the paper's). Paper and ours
are always in separate panels. Each slide carries speaker notes; the speaker
script (make_speaker_script.py) is built from them.

The talk is at most 15 minutes: 10 main slides (title to final), backups after.
At the team's request there is no verdict scorecard slide (the README keeps
the verdicts); a result that differs from the paper only because of the
computer and software is marked on the slide that shows it.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.shapes import MSO_SHAPE  # noqa: E402
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN  # noqa: E402
from pptx.util import Inches, Pt  # noqa: E402

from mrisk import paper_numbers as P  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
BUILD = DOCS / "build"
TEMPLATE = DOCS / "hse_template.pptx"
OUT = DOCS / "presentation.pptx"
MODELS5 = ["Lasso", "CART", "RF", "XGBoost", "MLP"]

NAVY = RGBColor(0x0F, 0x2C, 0x68)
BLUE = RGBColor(0x16, 0x56, 0xA6)
ORANGE = RGBColor(0xEB, 0x8C, 0x3C)
GREEN = RGBColor(0x00, 0x9B, 0x64)
RED = RGBColor(0xC0, 0x39, 0x2B)
TEXT = RGBColor(0x22, 0x22, 0x22)
MUTED = RGBColor(0x59, 0x59, 0x59)
LIGHT = RGBColor(0xEE, 0xF2, 0xF8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

MAIN_ORDER = [
    None,
    "Pricing model risk into machine-learning performance",
    "Data, and what is fitted, chosen and evaluated",
    "Reproduction 1: prediction and the model-risk proxies",
    "Reproduction 2: weights from regulatory texts",
    "Reproduction 3: the risk–reward picture",
    "Improvements 1: how certain are the conclusions?",
    "Improvements 2: benchmarks, explanations, missing factors",
    "Conclusions and critique",
    "FINAL",
]
BACKUP_ORDER = [
    "Backup: errata, the paper against itself",
    "Backup: Fig. 1 against its sources",
    "Backup: Fig. 9 document by document",
    "Backup: the paper's remaining risk factors",
    "Backup: every improvement",
    "Backup: SHAP rankings under two explainers",
    "Backup: calibration next to AUC",
    "Backup: deviations from the paper",
    "Backup: repository and how to run it",
]


# ---------------------------------------------------------------- helpers (as in the team's VPIN deck)
def layout(prs, name):
    return next(lay for lay in prs.slide_masters[0].slide_layouts if lay.name == name)


def drop(placeholder):
    placeholder.element.getparent().remove(placeholder.element)


def slide_with_title(prs, title: str, notes: str):
    s = prs.slides.add_slide(layout(prs, "Text_2"))
    s.shapes.title.text = title
    for ph in list(s.placeholders):
        idx = ph.placeholder_format.idx
        if idx == 2:
            ph.text = "Research Seminar"
        elif idx == 1:
            drop(ph)
    s.notes_slide.notes_text_frame.text = notes
    return s


def _runs(paragraph, text: str, size: float, color, bold=False):
    """**double stars** mark bold segments."""
    for i, part in enumerate(text.split("**")):
        if not part:
            continue
        r = paragraph.add_run()
        r.text = part
        r.font.size = Pt(size)
        r.font.bold = bold or i % 2 == 1
        r.font.color.rgb = color


def text(slide, box, items, size=16, color=TEXT, space=6):
    x, y, w, h = box
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.05)
    for i, item in enumerate(items):
        s, opt = (item, {}) if isinstance(item, str) else item
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(opt.get("space", space))
        p.alignment = opt.get("align", PP_ALIGN.LEFT)
        prefix = "• " if opt.get("bullet", True) and s else ""
        _runs(p, prefix + s, opt.get("size", size), opt.get("color", color), opt.get("bold", False))
    return tb


def heading(slide, box, label, size=13):
    return text(slide, box, [(label, {"bullet": False, "bold": True, "color": NAVY, "size": size})])


def picture(slide, path: Path, box):
    x, y, w, h = box
    iw, ih = Image.open(path).size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    return slide.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))


def table(slide, df: pd.DataFrame, box, widths=None, size=11, fills=None):
    x, y, w, h = box
    rows, cols = df.shape[0] + 1, df.shape[1]
    t = slide.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w), Inches(h)).table
    for j, cw in enumerate(widths or [w / cols] * cols):
        t.columns[j].width = Inches(cw)
    for i in range(rows):
        for j in range(cols):
            cell = t.cell(i, j)
            cell.text = ""
            value = df.columns[j] if i == 0 else df.iat[i - 1, j]
            _runs(cell.text_frame.paragraphs[0], str(value), size, WHITE if i == 0 else TEXT, bold=i == 0)
            cell.margin_left = cell.margin_right = Inches(0.05)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY if i == 0 else (LIGHT if i % 2 == 0 else WHITE)
            if fills and (i - 1, j) in fills:
                cell.fill.fore_color.rgb = fills[(i - 1, j)]
                for r in cell.text_frame.paragraphs[0].runs:
                    r.font.color.rgb = WHITE
                    r.font.bold = True
    return t


def box_shape(slide, box, label, fill, size=13):
    x, y, w, h = box
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.fill.background()
    tf = shp.text_frame
    tf.word_wrap = True
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    _runs(tf.paragraphs[0], label, size, WHITE, bold=True)
    return shp


def note_box(slide, box, items, fill=RGBColor(0xFD, 0xEB, 0xD9), edge=ORANGE, size=11):
    """A shaded call-out: marks what differs from the paper only because of the machine and software."""
    x, y, w, h = box
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.adjustments[0] = 0.08
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.color.rgb = edge
    shp.line.width = Pt(1.25)
    tf = shp.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.12)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(3)
        _runs(p, item, size, TEXT)
    return shp


def arrow_right(slide, x, y, w=0.4, h=0.32):
    a = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    a.fill.solid()
    a.fill.fore_color.rgb = MUTED
    a.line.fill.background()


def equation(tex: str, path: Path, size=24):
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${tex}$", fontsize=size, color="#0F2C68")
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05, transparent=True)
    plt.close(fig)
    return path


def slide_title(slide) -> str:
    return slide.shapes.title.text if slide.shapes.title is not None and slide.shapes.title.has_text_frame else ""


def reorder(prs):
    ids = prs.slides._sldIdLst
    entries = list(zip(list(ids), list(prs.slides)))
    first, final = entries[0], next(e for e in entries if e[1].slide_layout.name == "Final_1")
    by_title = {slide_title(sl): (el, sl) for el, sl in entries}
    main = [first] + [by_title[t] if t != "FINAL" else final for t in MAIN_ORDER[1:]]
    used = {id(e[0]) for e in main}
    backup = sorted((e for e in entries if id(e[0]) not in used), key=lambda e: BACKUP_ORDER.index(slide_title(e[1])))
    for el, _ in entries:
        ids.remove(el)
    for el, _ in main + backup:
        ids.append(el)


def frontier_chart(i13: dict, path: Path) -> Path:
    """P(model is risk-efficient), per use, from the I13 Monte Carlo."""
    tab = pd.DataFrame(i13["table"])
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    colors = {"capital": "#2A78D6", "scoring": "#EB6834", "provisions": "#1BAF7A"}
    for k, use in enumerate(["capital", "scoring", "provisions"]):
        d = tab[tab["use"] == use].set_index("model").loc[MODELS5]
        ax.bar([i + (k - 1) * 0.26 for i in range(5)], d["p_frontier"], 0.24, color=colors[use], label=use.capitalize())
    ax.axhline(0.95, color="#555555", lw=0.8, ls="--")
    ax.text(4.45, 0.97, "95%", fontsize=8, color="#555555", ha="right")
    ax.set_xticks(range(5), MODELS5)
    ax.set_ylabel("P(on the risk–gain frontier)")
    ax.set_ylim(0, 1.08)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def fig1_points() -> int:
    """Points plotted in Fig. 1: rows of the source table except the note on the wrong reference."""
    lit = pd.read_csv(ROOT / "data" / "fig1_sources.csv")
    return int((lit["status"] != "wrong_reference").sum())


def manifest_found() -> int:
    m = pd.read_csv(ROOT / "data" / "corpus_manifest.csv")
    return int(m["status"].isin(["found", "substitute"]).sum())


def cart_line() -> str:
    """One bullet on why our CART differs, read from the diagnostic (validation rows only)."""
    path = ROOT / "results" / "reproduction" / "models" / "cart_diagnostic.json"
    if not path.exists():
        return ""
    h = json.loads(path.read_text())["hypotheses"]
    small = [v["auc"] for key, v in h.items() if key.startswith("H2") and 8 <= v["leaves"] <= 12]
    return (f"**CART 84.4 vs 80.4:** no depth-7 tree gets there (class weights, entropy, row index, hard labels all fail); a pruned "
            f"tree of 8–12 leaves ({100 * min(small):.1f}–{100 * max(small):.1f}) or a depth-10 tree ({100 * h['H1 depth 10']['auc']:.1f}) "
            "does. The paper's tree was most likely not the depth-7 tree it reports.")


# ---------------------------------------------------------------- slides
def build() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    k = json.loads((ROOT / "results" / "report" / "key_numbers.json").read_text())
    m, n, imp = k["models"], k["nlp"], k["improvements"]
    a, px, meta = m["auc"], m["proxies"], m["meta"]
    fig = {key: ROOT / p for key, p in k["figures"].items()}
    eq = equation(r"\mathrm{Risk}_{m,j}=\sum_{i=1}^{3}\bar H_{i,j}\,Z_{i,m}", BUILD / "eq_risk.png")
    prs = Presentation(TEMPLATE)

    # 1 Title
    s = prs.slides.add_slide(layout(prs, "Title_1"))
    for ph in list(s.placeholders):
        idx = ph.placeholder_format.idx
        if idx == 0:
            ph.text = "MEASURING THE MODEL RISK-ADJUSTED PERFORMANCE OF ML IN CREDIT DEFAULT PREDICTION"
            for run in ph.text_frame.paragraphs[0].runs:
                run.font.size = Pt(30)
        elif idx == 1:
            ph.text = "Research Seminar"
        elif idx == 2:
            ph.text = "Group 8"
        elif idx == 4:
            ph.text = "ANDRÉS ALONSO ROBISCO AND JOSÉ MANUEL CARBÓ MARTÍNEZ"
        else:
            drop(ph)
    text(s, (1.12, 6.0, 8, 1.0), [("Financial Innovation 8:70, 2022 (Banco de España)", {"bullet": False, "size": 14, "color": NAVY}),
                                  ("Nazarov · Doroshenko · Burlakov · Daunov · Sergeev", {"bullet": False, "size": 14, "color": MUTED})])
    s.notes_slide.notes_text_frame.text = (
        "Timing, 14 minutes in all: the paper 2 · objectives 1 · reproduction 5.5 · improvements 3.5 · conclusions 2. "
        "Then 10 minutes of questions. Our task for this paper: reproduce every result it lists and propose an improvement for each.")

    # 2 The paper
    s = slide_with_title(prs, "Pricing model risk into machine-learning performance", (
        "The paper, ~2 min. Banks see ML predict defaults better, but supervisors see new risks, and banks do not know how a "
        "supervisor would weigh them. The authors propose a scorecard. Phase 1: each model gets a darkness score Z from 1 to 5 in "
        "three categories, from five measured proxies and the authors' judgement. Phase 2: each category's weight H-bar for a use "
        "(capital, credit scoring, provisions) is the share of the category's terms among all words of the regulatory texts for that "
        "use. Risk is the weighted sum, plotted against the AUC gain over a logit. The answer on Kaggle data: XGBoost is the most "
        "risk-efficient model. Note it gives a ranking, not an approval threshold: the supervisor's loss function is unknown."))
    text(s, (0.64, 2.35, 6.3, 4.6), [
        "**Problem:** ML predicts defaults better than a logit, but supervisors see new **model risks**: instability, opaque, slow and cloud-dependent models, hard-to-explain decisions.",
        "**Question:** how much model risk does each algorithm carry, and is its extra accuracy worth it?",
        "**13 risk factors** from the Basel IRB validation process, in 3 categories: statistics, technology, market conduct.",
        "**Answer (Kaggle data):** XGBoost gives the best gain for its risk; the MLP is the riskiest for a small gain.",
    ], size=15, space=10)
    box_shape(s, (7.3, 2.45, 2.65, 1.25), "Phase 1: darkness Z (1–5) per model and category, from 5 proxies", BLUE, 12)
    arrow_right(s, 10.0, 2.9)
    box_shape(s, (10.45, 2.45, 2.55, 1.25), "Phase 2: weight H̄ per use, from term counts in regulatory texts", BLUE, 12)
    picture(s, eq, (7.6, 4.05, 5.1, 0.9))
    box_shape(s, (7.3, 5.2, 5.7, 0.95), "Risk vs AUC gain over the logit → rank models per use", NAVY, 13)

    # 3 Objectives
    s = slide_with_title(prs, "Data, and what is fitted, chosen and evaluated", (
        "Objectives, ~1 min (the instructor asked us to say this explicitly). Fitted: six classifiers, each by its own loss. Chosen: "
        "only the Lasso penalty, by cross-validated AUC on training rows; everything else is the paper's setting. Evaluated: test AUC "
        "and the five proxies. Phase 2 fits nothing: weights are word shares. Z is expert judgement in the paper; our improvement I13 "
        "replaces it by a rule. Data: complete cases of Give Me Some Credit match the paper's 120,000 loans. Improvements were "
        "pre-registered before running and make their choices on validation rows only."))
    obj = pd.DataFrame([
        ("Six classifiers (Logit, Lasso, CART, RF, XGBoost, MLP)", "Own loss: log-loss or Gini, on 80% training rows", "Paper's hyperparameters", "Test AUC (20% rows), DeLong ±"),
        ("Lasso penalty", "–", "5-fold CV AUC on training rows", "–"),
        ("Five proxies (stability, size, time, transparency, SHAP time)", "100 refits on 60–100% subsamples", "–", "Spread of test AUC; seconds"),
        ("Darkness scores Z", "–", "Paper: expert judgement. Ours (I13): a rule", "–"),
        ("Weights H̄ (phase 2)", "Nothing: share of category terms in all words", "Dictionary of the paper (Table 11)",
         f"{manifest_found()} of the paper's {len(pd.read_csv(ROOT / 'data' / 'corpus_manifest.csv'))} regulatory documents"),
    ], columns=["Step", "Fitted", "Chosen", "Evaluated"])
    table(s, obj, (0.64, 2.35, 12.0, 2.9), widths=[3.4, 3.2, 2.9, 2.5], size=11)
    text(s, (0.64, 5.45, 12.0, 1.5), [
        f"**Data:** Give Me Some Credit (Kaggle). {meta['rows_raw']:,} rows; the {meta['rows']:,} complete cases match the paper's 120,000. "
        f"Default rate {100 * meta['default_rate']:.2f}%. {m['n_features']} features: 10 borrower characteristics and their squares. "
        f"Stratified 80/20 split: {meta['train']:,} / {meta['test']:,} rows.",
        f"**Coverage:** every numbered result of the paper, {len(k['verdicts'])} in all (Tables 2–10, Figs. 1 and 3–9), plus Fig. 1's "
        f"{fig1_points()} literature points checked in their sources. Improvements were "
        "pre-registered and make choices on validation rows only; the test split is locked.",
    ], size=13, space=6)

    # 4 Reproduction 1
    s = slide_with_title(prs, "Reproduction 1: prediction and the model-risk proxies", (
        "Reproduction, ~2.5 min. Left the paper's Table 7 and Table 5, right ours, never mixed. Read across: AUC levels and the "
        "ranking of models; say which paper AUCs fall inside our intervals. Stability: our spread of test AUC over 100 refits, same "
        "order as the paper. Then the orange columns: times are a property of the computer and the software, not of the method. "
        "The paper reports no hardware, library, solver or explainer. Where the software is comparable the times agree (random "
        "forest SHAP); where it is not, they differ (Lasso solver, MLP training and explainer). We do not count that as a failure "
        "of the idea. CART is the one model whose AUC differs: we tried every reading of the paper's set-up; only a small pruned "
        "tree (as R's rpart builds) or a depth-10 tree reaches 80.4, so the paper's tree was probably not the depth-7 tree it reports."))
    heading(s, (0.64, 2.25, 6.0, 0.4), "Paper: Tables 7 and 5")
    paper = pd.DataFrame([(mm, f"{P.TABLE7_AUC[mm]:.1f}", f"{P.TABLE5['stability_sd'][mm]:.3f}", f"{P.TABLE5['training_seconds'][mm]:.2f}",
                           f"{P.TABLE5['shap_seconds'][mm]:,}") for mm in MODELS5],
                         columns=["Model", "AUC %", "SD of AUC", "Train s", "SHAP s"])
    paper.loc[-1] = ["Logit", f"{P.TABLE7_AUC['Logit']:.1f}", "–", "–", "–"]
    paper = paper.sort_index().reset_index(drop=True)
    machine = {(-1, 3): ORANGE, (-1, 4): ORANGE}  # time columns: depend on the computer and the software
    table(s, paper, (0.64, 2.65, 5.9, 2.4), widths=[1.2, 1.0, 1.25, 1.15, 1.3], size=10, fills=machine)
    heading(s, (6.85, 2.25, 6.2, 0.4), "Ours: same protocol, our split (± s.e.)")
    ours = pd.DataFrame([(mm, f"{100 * a[mm]['auc']:.1f} ± {100 * a[mm]['se']:.1f}",
                          "–" if mm == "Logit" else f"{px[mm]['auc_sd']:.4f}", f"{px[mm]['fit_s']:.2f}",
                          "–" if mm == "Logit" else f"{px[mm]['shap_s']:,.1f}") for mm in ["Logit"] + MODELS5],
                        columns=["Model", "AUC %", "SD of AUC", "Train s", "SHAP s"])
    table(s, ours, (6.85, 2.65, 6.2, 2.4), widths=[1.15, 1.45, 1.2, 1.1, 1.3], size=10, fills=machine)
    best = max(MODELS5, key=lambda mm: a[mm]["auc"])
    v = {r["id"]: r for r in k["verdicts"]}
    text(s, (0.64, 5.15, 6.0, 2.0), [
        f"**AUC:** {v['R12']['ours']}. Gain of {best} over the logit: {100 * a[best]['gain']:+.1f} ± {100 * a[best]['gain_se']:.1f} points (paired DeLong {'p < 0.001' if a[best]['gain_p'] < 0.001 else f"p = {a[best]['gain_p']:.3f}"}).",
        f"**Stability:** {v['R7']['ours'].split(': ')[1]}: Lasso and XGBoost most stable, CART and the MLP least, as in the paper.",
        cart_line(),
    ], size=10, space=4)
    cpu = meta["environment"]["cpu"]
    note_box(s, (6.85, 5.2, 6.2, 1.5), [
        "**Orange columns: machine and software, not method.** The paper reports no hardware, library, solver or explainer; "
        f"we time on an {cpu}, one thread.",
        f"Comparable software, same time: random forest SHAP {px['RF']['shap_s']:.0f} s vs {P.TABLE5['shap_seconds']['RF']} s.",
        f"Different software, different time: Lasso solver {px['Lasso']['fit_s']:.1f} s vs {P.TABLE5['training_seconds']['Lasso']} s; "
        f"MLP explainer {px['MLP']['shap_s']:.0f} s vs {P.TABLE5['shap_seconds']['MLP']:,} s.",
    ], size=10)

    # 5 Reproduction 2
    s = slide_with_title(prs, "Reproduction 2: weights from regulatory texts", (
        "Reproduction, ~1.5 min. We found 35 of the 36 documents of Table 12 (one PwC note is not public) and counted the 134 terms "
        "of Table 11. Left the paper's Table 2, right ours with bootstrap standard errors over documents. Then significance "
        "(Table 3) and the sentiment check (Tables 8-10). Say which of the paper's statements hold."))
    picture(s, fig["weights_paper"], (0.5, 2.25, 6.1, 3.3))
    picture(s, fig["weights_ours"], (6.8, 2.25, 6.1, 3.3))
    held = sum(imp.get("I3", {}).get("claims_point", {}).values())
    r14 = k.get("extra", {}).get("R14", {}).get("spearman", {})
    text(s, (0.64, 5.65, 12.4, 1.4), [
        f"**Table 2:** {held} of 4 of the paper's statements hold; our levels are about half (we divide by all words, the paper most "
        f"likely by words without stop words). **Mentions per document:** \"stability\" {v['R2']['ours']}."
        + (f" **Fig. 9, document by document:** rank correlation " + " / ".join(f"{r14[c]['rho']:.2f}" for c in r14) + "." if r14 else ""),
        f"**Table 3:** {v['R4']['ours']}. **Sentiment (Tables 8–9):** in our corpus, credit-scoring texts are not the most negative "
        "around technology or conduct terms, at 10 or 20 words.",
    ], size=12, space=5)

    # 6 Reproduction 3
    s = slide_with_title(prs, "Reproduction 3: the risk–reward picture", (
        "Reproduction, ~1.5 min. Left: the paper's Table 6 risk against its Table 7 gains. Right: the paper's Z with our weights "
        "and our gains with 95% intervals. Then the errata: recomputing the paper's own numbers shows the MLP's statistics score "
        "is 5 in Table 5 but 4 in Table 6, and the random forest in Figs. 4 and 5 is drawn with a statistics score of 2, not 3. "
        "Neither changes the ranking. Last line: we looked up every point of the paper's literature figure in its source; "
        "12 of 20 match, three of those are a different metric (KS) on an AUC axis, and the biggest gain is mis-cited."))
    picture(s, fig["rr_paper"], (0.5, 2.25, 6.1, 3.5))
    if "rr_ours" in fig:
        picture(s, fig["rr_ours"], (6.8, 2.25, 6.1, 3.5))
    err = {e["id"]: e for e in k["errata"]}
    lit = pd.read_csv(ROOT / "data" / "fig1_sources.csv")
    lit = lit[lit["status"] != "wrong_reference"]  # an extra row on the cited reference, not a plotted point
    n_match = int((lit["status"] == "matches").sum())
    n_ks = int(lit["metric"].fillna("").str.startswith("KS").sum())
    text(s, (0.64, 5.8, 12.4, 1.4), [
        f"**With our weights and gains:** {v['R13']['ours']}.",
        "**The paper against itself:** three versions of the statistics scores (Table 5: RF 3, MLP 5; Table 6: RF 3, MLP 4; "
        "Figs. 4 and 5: RF 2, MLP 4). None changes the ranking.",
        f"**Fig. 1 against its {len(lit)} sources:** {n_match} points match; {n_ks} are KS differences drawn on an AUC axis; the largest "
        "gain (20 points) is credited to a paper on price formation, not credit.",
    ], size=11, space=4)

    # 8 Improvements 1
    s = slide_with_title(prs, "Improvements 1: how certain are the conclusions?", (
        "Improvements, ~2 min. The paper reports no uncertainty anywhere. We bootstrap the documents behind the weights, "
        "correct the nine tests for multiple testing, compare negative words with their base rate, and replace the expert Z by "
        "a rule inside a Monte Carlo over every noisy input. The chart: how often each model is on the risk–gain frontier."))
    rows = {r["id"]: r for r in k["improvement_rows"]}
    text(s, (0.64, 2.3, 6.4, 4.8), [f"**{i}. {rows[i]['what']}:** {rows[i]['finding']}" for i in ("I3", "I4", "I5", "I7") if i in rows],
         size=11, space=7)
    if "I13" in imp:
        picture(s, frontier_chart(imp["I13"], BUILD / "frontier.png"), (7.2, 2.3, 5.8, 3.4))
        text(s, (7.2, 5.8, 5.8, 1.2), [f"**I13. {rows['I13']['what']}:** {rows['I13']['finding']}"], size=11)

    # 9 Improvements 2
    s = slide_with_title(prs, "Improvements 2: benchmarks, explanations, missing factors", (
        "Improvements, ~1.5 min. Benchmark: same protocol on two more public datasets, does XGBoost stay on top? SHAP: the paper "
        "explains XGBoost in log-odds and the others in probabilities, so its rankings are not comparable; with one explainer and "
        "one scale we check which ranking claims survive over five seeds. Appendix: the paper names 13 risk factors but measures 5; "
        "we measure five more as its appendix proposes (overfitting, inputs used, preprocessing sensitivity, surrogate fidelity, "
        "age counterfactual) and redo the frontier with them."))
    if "benchmark" in fig:
        picture(s, fig["benchmark"], (0.5, 2.25, 6.6, 3.4))
    text(s, (7.3, 2.3, 5.8, 4.8), [f"**{i}. {rows[i]['what']}:** {rows[i]['finding']}" for i in ("I1", "X1", "I11", "A1–A5", "I14") if i in rows],
         size=10, space=6)

    # 10 Conclusions
    s = slide_with_title(prs, "Conclusions and critique", (
        "Conclusions, ~2 min. Summarise what holds, what does not, and what the paper cannot claim: one dataset, expert scores, "
        "word counts as preferences, no uncertainty. Then next steps."))
    i1 = imp.get("I1", {}).get("summary", {})
    german = i1.get("german_credit", {})
    text(s, (0.64, 2.3, 12.2, 4.8), [
        f"**Matched:** {v['R12']['ours'].split(';')[0]}, XGBoost best; the stability order; every ordering of the text weights, document by document too; the paper's own arithmetic.",
        "**Differs only because of the machine and software:** training and SHAP times (hardware, libraries, solver and explainer are not reported).",
        "**Differs in substance:** the sentiment check, SHAP mid-ranks and CART's AUC; and the paper contradicts itself on the statistics scores (Tables 5, 6 and Figs. 4, 5).",
        "**The biggest caveat (exploratory):** a logit with log-transformed inputs comes within a fraction of a point of XGBoost; most of the ML gain is a weak benchmark.",
        "**What is fragile:** \"provisions is the least risky use\" only reflects fewer risk words in provisioning texts; with few loans the ML gain disappears"
        + (f" (German credit: best is {german['best']})." if german else "."),
        "**Critique and next steps:** one dataset, no uncertainty, expert Z, word counts as supervisory preference, no approval threshold. Next: use-specific metrics (calibration for capital and provisions), a supervisor's loss function, time-split validation.",
    ], size=13, space=8)

    prs.slides.add_slide(layout(prs, "Final_1"))

    # Backups
    s = slide_with_title(prs, "Backup: errata, the paper against itself", "Every inconsistency found by recomputing the paper's own numbers.")
    if k["errata"]:
        e = pd.DataFrame(k["errata"])[["id", "where", "what"]]
        e.columns = ["#", "Where", "What"]
        table(s, e, (0.64, 2.3, 12.2, 4.6), widths=[0.5, 2.6, 9.1], size=10)
    lit_path = ROOT / "data" / "fig1_sources.csv"
    s = slide_with_title(prs, "Backup: Fig. 1 against its sources", "Each labelled point of Fig. 1 looked up in its cited source (open-access copies).")
    if lit_path.exists():
        lit = pd.read_csv(lit_path)
        show = pd.DataFrame({"Label in Fig. 1": lit["label"].str.slice(0, 34), "Metric": lit["metric"].str.slice(0, 26),
                             "Logit": lit["logit_value"].fillna("–").astype(str).str.slice(0, 16),
                             "ML model": lit["ml_model"].fillna("–").astype(str).str.slice(0, 28),
                             "ML": lit["ml_value"].fillna("–").astype(str).str.slice(0, 16),
                             "Gain, points": lit["abs_gain_points"].map(lambda v: "–" if pd.isna(v) else f"{v:+.2f}"),
                             "Plotted": lit["plotted_value_matched"].fillna("–").astype(str).str.slice(0, 22),
                             "Status": lit["status"].str.replace("_", " ")})
        table(s, show, (0.4, 2.2, 12.5, 4.9), widths=[2.1, 1.7, 1.25, 2.0, 1.25, 0.85, 1.75, 1.6], size=7)
    s = slide_with_title(prs, "Backup: Fig. 9 document by document", "R14: values read off the paper's Fig. 9 against ours; read the ranks, not the levels.")
    if "fig9" in fig:
        picture(s, fig["fig9"], (0.5, 2.3, 12.3, 3.8))
        r14 = k["extra"]["R14"]["spearman"]
        text(s, (0.64, 6.2, 12.2, 0.8), ["Spearman ρ, paper vs ours: " + "; ".join(f"{c} {r14[c]['rho']:.2f} [{r14[c]['lo']:.2f}, {r14[c]['hi']:.2f}]" for c in r14)
                                         + ". Levels differ: the paper most likely divided by words without stop words."], size=12)
    s = slide_with_title(prs, "Backup: the paper's remaining risk factors", "A1-A5 on validation rows; third-party, cyber and privacy need bank data.")
    ap = k.get("appendix", {})
    if ap:
        f = ap["factors"]
        at = pd.DataFrame([(m, f"{f[m]['overfit_gap']:+.3f}", f"{f[m]['features_used']} / 20", f"{f[m]['fe_sensitivity']:.4f}",
                            f"{f[m]['surrogate_r2']:.3f}", f"{100 * f[m]['flip_rate']:.2f}%") for m in f],
                          columns=["Model", "A1 overfitting (AUC gap)", "A2 inputs used", "A3 largest ΔAUC", "A4 surrogate R²", "A5 age flips"])
        table(s, at, (0.64, 2.3, 12.0, 3.0), size=11)
        if "frontier" in ap:
            text(s, (0.64, 5.6, 12.0, 1.2), ["**I14, scorecard with these factors, risk-gain frontier:** "
                                             + "; ".join(f"{u} {', '.join(v)}" for u, v in ap["frontier"].items())], size=13)
    s = slide_with_title(prs, "Backup: every improvement", "All improvements and their findings.")
    table(s, pd.DataFrame([(r["id"], r["what"], r["finding"]) for r in k["improvement_rows"]], columns=["#", "Improvement", "Finding"]),
          (0.5, 2.3, 12.4, 4.9), widths=[0.45, 3.2, 8.75], size=7)
    s = slide_with_title(prs, "Backup: SHAP rankings under two explainers", "Default explainers (XGBoost in log-odds) vs one explainer on the probability scale.")
    picture(s, fig["shap_default"], (0.5, 2.3, 6.2, 3.2))
    picture(s, fig["shap_common"], (6.8, 2.3, 6.2, 3.2))
    s = slide_with_title(prs, "Backup: calibration next to AUC", "I12: Brier score, expected calibration error, calibration slope on test rows.")
    if "I12" in imp:
        t = imp["I12"]["table"]
        cal = pd.DataFrame([(mm, f"{100 * t[mm]['auc']:.1f}", f"{t[mm]['brier']:.4f} ± {t[mm]['brier_se']:.4f}",
                             f"{t[mm]['ece']:.4f} ± {t[mm]['ece_se']:.4f}", f"{t[mm]['slope']:.2f} ± {t[mm]['slope_se']:.2f}") for mm in t],
                           columns=["Model", "AUC %", "Brier", "ECE", "Calibration slope"])
        table(s, cal, (0.64, 2.4, 10.0, 3.0), size=12)
    s = slide_with_title(prs, "Backup: deviations from the paper", "Where we depart from the paper or it is silent: DEVIATIONS.md.")
    text(s, (0.64, 2.3, 12.2, 4.8), [
        "Complete cases (120,269) match the paper's 120,000; 10 characteristics + squares = 20 features (paper: 22; the 11th column is a row index).",
        "Stratified split, seed 0. Lasso penalty by 5-fold CV over 11 values; saga solver (liblinear needed > 90 s per fit).",
        "MLP: scikit-learn, early stopping (epochs not reported). XGBoost, RF: library defaults beyond depth and trees.",
        "Stability: 100 refits on 60–100% subsamples, the same for every model; the paper describes two procedures.",
        "SHAP: 1,000 test rows; TreeSHAP for trees (XGBoost in log-odds, as in the paper's Fig. 7), permutation explainer otherwise.",
        "Corpus: official copies; Fig. 9's assignment of documents; PwC (2017) missing; plural-only lemmatisation.",
    ], size=13, space=8)
    s = slide_with_title(prs, "Backup: repository and how to run it", "github.com/zavaz5/ml-model-risk")
    text(s, (0.64, 2.3, 12.2, 4.8), [
        "**github.com/zavaz5/ml-model-risk** · code in src/mrisk, scripts in scripts/, tests in tests/.",
        "python scripts/download_data.py → extract_paper.py → reproduce_models.py → reproduce_nlp.py → reproduce_scorecard.py",
        "→ improve_nlp.py → improve_models.py → improve_risk.py → make_report.py → make_slides.py",
        "Every number on these slides is read from results/report/key_numbers.json; the paper's numbers from src/mrisk/paper_numbers.py.",
    ], size=14, space=10)

    reorder(prs)
    prs.save(OUT)
    print(f"saved {OUT} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    build()
