"""Build docs/presentation.pptx on the seminar's HSE template.

Numbers come from results/report/key_numbers.json (ours, written by
make_report.py) and src/mrisk/paper_numbers.py (the paper's). Paper and ours
are always in separate panels. Each slide carries speaker notes; the speaker
script (make_speaker_script.py) is built from them.

At the team's request the main slides show every result of the paper next
to ours, one result per slide, with a green box when they agree, an orange
box that says why when they do not, and a blue box with our improvement of
that result. Extra checks beyond the paper's results are in the backups.
The 15-minute limit is set aside for now (about 17.5 minutes by the notes).
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
    "Fig. 1: what the literature finds",
    "Table 2 and Fig. 3: weights from regulatory texts",
    "Fig. 9: the weights document by document",
    "Table 3: are the differences between uses significant?",
    "Tables 8–10: the sentiment check",
    "Table 4: the worked example",
    "Table 7: predictive performance",
    "Table 5: the model-risk proxies",
    "Fig. 4: the darkness scores",
    "Table 6 and Fig. 5: risk against reward",
    "Figs. 6–8: what drives the predictions (SHAP)",
    "Conclusions",
    "FINAL",
]
BACKUP_ORDER = [
    "Backup: errata, the paper against itself",
    "Backup: Fig. 1 against its sources",
    "Backup: the same protocol on three datasets (I1)",
    "Backup: Fig. 9 document by document",
    "Backup: the scorecard Monte Carlo (I13)",
    "Backup: beyond the paper: appendix factors and a fair logit",
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
    ours = json.loads((ROOT / "results" / "report" / "key_numbers.json").read_text())["models"]["auc"]["CART"]["auc"]
    return (f"**CART:** we fit the tree the paper describes (depth 7) and get {100 * ours:.1f}. Its {P.TABLE7_AUC['CART']:.1f} comes "
            f"only from a pruned tree of 8–12 leaves, as R's rpart builds by default ({100 * min(small):.1f}–{100 * max(small):.1f}), "
            f"or a depth-10 tree ({100 * h['H1 depth 10']['auc']:.1f}). The difference is in the tree's settings, not in the method.")


# ---------------------------------------------------------------- one slide per result
PAPER_PDF = ROOT / "paper" / "alonso_carbo_2022.pdf"
FIGS = ROOT / "results" / "report" / "figures"
GREEN_FILL, ORANGE_FILL, BLUE_FILL = RGBColor(0xE3, 0xF4, 0xEC), RGBColor(0xFD, 0xEB, 0xD9), RGBColor(0xE6, 0xEE, 0xF8)
# The paper's figures, cropped from its PDF (CC BY 4.0): (1-based page, clip in PDF points).
PAPER_FIGS = {"fig1": (4, (118, 89, 478, 268)), "fig3": (15, (118, 202, 478, 382)), "fig4": (19, (118, 89, 478, 241)),
              "fig5": (22, (118, 89, 478, 210)), "fig6": (26, (128, 89, 468, 245)), "fig7": (26, (128, 286, 468, 444)),
              "fig8": (27, (128, 89, 468, 246))}


def crop_paper(name: str) -> Path:
    import pymupdf

    page, clip = PAPER_FIGS[name]
    path = BUILD / f"paper_{name}.png"
    with pymupdf.open(PAPER_PDF) as doc:
        doc[page - 1].get_pixmap(dpi=220, clip=pymupdf.Rect(*clip)).save(path)
    return path


def place(slide, content, box, size=10):
    """A picture, a row of pictures, or a table (DataFrame, or (DataFrame, fills, widths))."""
    x, y, w, h = box
    if isinstance(content, Path):
        return picture(slide, content, box)
    if isinstance(content, list):
        step = w / len(content)
        for i, p in enumerate(content):
            picture(slide, p, (x + i * step, y, step - 0.05, h))
        return None
    frame, fills, widths = content if isinstance(content, tuple) else (content, None, None)
    return table(slide, frame, (x, y, w, min(h, 0.36 * (len(frame) + 1))), widths=widths, size=size, fills=fills)


def result_slide(prs, title, notes, paper, ours, same, why=None, improvement=None,
                 paper_label="Paper", ours_label="Ours", stacked=False, table_size=12):
    """Paper left, ours right (or stacked rows), then a green 'same' or orange 'why it differs' box, then the improvement."""
    s = slide_with_title(prs, title, notes)
    if stacked:  # two rows of three pictures (the SHAP figures)
        heading(s, (0.4, 2.04, 4.0, 0.3), paper_label, 11)
        place(s, paper, (0.4, 2.32, 12.5, 1.45))
        heading(s, (0.4, 3.78, 4.0, 0.3), ours_label, 11)
        place(s, ours, (0.4, 4.06, 12.5, 1.5))
        top = 5.62
    else:
        heading(s, (0.64, 2.0, 6.0, 0.35), paper_label, 12)
        heading(s, (6.85, 2.0, 6.2, 0.35), ours_label, 12)
        place(s, paper, (0.64, 2.38, 6.0, 2.92), size=table_size)
        place(s, ours, (6.85, 2.38, 6.2, 2.92), size=table_size)
        top = 5.42
    if why:
        note_box(s, (0.64, top, 12.4, 0.95), [f"**Compared with the paper:** {why[0].upper() + why[1:]}"], fill=ORANGE_FILL, edge=ORANGE, size=11)
    else:
        note_box(s, (0.64, top, 12.4, 0.95), [f"**Same as the paper:** {same[0].upper() + same[1:]}"], fill=GREEN_FILL, edge=GREEN, size=11)
    if improvement:
        note_box(s, (0.64, top + 1.02, 12.4, 0.78), [f"**Our improvement:** {improvement[0].upper() + improvement[1:]}"], fill=BLUE_FILL, edge=BLUE, size=11)
    return s


def pfmt(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def result_slides(prs, k: dict, fig: dict) -> None:
    m, n, imp = k["models"], k["nlp"], k["improvements"]
    a, px = m["auc"], m["proxies"]
    v = {r["id"]: r for r in k["verdicts"]}
    rows = {r["id"]: r for r in k["improvement_rows"]}
    cats = [("statistics", "Statistics"), ("technology", "Technology"), ("conduct", "Conduct")]
    uses = [("capital", "Capital"), ("scoring", "Scoring"), ("provisions", "Provisions")]

    # Fig. 1
    lit = pd.read_csv(ROOT / "data" / "fig1_sources.csv")
    lit = lit[lit["status"] != "wrong_reference"]
    n_match = int((lit["status"] == "matches").sum())
    n_ks = int(lit["metric"].fillna("").str.startswith("KS").sum())
    i1 = imp.get("I1", {}).get("summary", {})
    result_slide(prs, "Fig. 1: what the literature finds", (
        "~1 min. The paper's first figure collects ML gains over a logit from 20 earlier studies. We looked up every point in "
        "its source (two papers the team downloaded through subscriptions). Green matches, red does not, orange is another "
        "metric; hollow circles are the source's own value."),
        crop_paper("fig1"), FIGS / "paper_fig1_checked.png", "",
        why=(f"{n_match} of {len(lit)} points match their source. {n_ks} of those are KS differences, not AUC; two sources report "
             "recall or accuracy; the 20-point gain is cited to a paper on price formation (the numbers come from a mortgage paper "
             "by the same author); a negative Lasso gain is drawn at 0. Gains are AUC points, not percent."),
        improvement=("one protocol on three public datasets instead of 20 incomparable studies: XGBoost is best on Give Me Some "
                     f"Credit ({pfmt(i1['gmsc']['best_vs_runner_p'])} against {i1['gmsc']['runner_up']}) and Taiwan (tied with "
                     f"{i1['taiwan_default']['runner_up']}, {pfmt(i1['taiwan_default']['best_vs_runner_p'])}); on 1,000 German loans "
                     "the trees lose to the logit. The gain from ML depends on the data." if i1 else None),
        paper_label="Paper, Fig. 1", ours_label="Ours: each point looked up in its source")

    # Table 2 / Fig. 3
    sw = json.loads((ROOT / "results" / "reproduction" / "nlp" / "stopword_diagnostic.json").read_text())
    t2 = n["table2"]
    p3 = imp["I3"]["claim_probability"]
    mpd = n["meta"]["mentions_per_document"]
    result_slide(prs, "Table 2 and Fig. 3: weights from regulatory texts", (
        "~1.5 min. Each weight is the share of a category's terms among all words of the documents for a use. Left the paper's "
        "Fig. 3, right ours with bootstrap errors. The four statements the paper draws hold; the level is about half because "
        "the paper does not say what it divides by."),
        crop_paper("fig3"), fig["weights_ours"], "",
        why=("all four of the paper's orderings hold (statistics → capital, technology and conduct → scoring, provisions lowest), "
             f"and \"stability\" leads the mentions ({mpd['stability']:.1f} per document vs the paper's 6.4). Only the level is about "
             "half: the paper does not state its denominator. Without stop words our totals are "
             + " / ".join(f"{sw['table2_pct'][u]['total']:.1f}%" for u, _ in uses) + " vs its "
             + " / ".join(f"{P.TABLE2[u]['total']:.1f}%" for u, _ in uses)
             + ". A common scale changes no ranking of models."),
        improvement=("1,000 document bootstraps: statistics → capital holds in "
                     f"{100 * p3['statistics_highest_for_capital']:.0f}%, conduct → scoring {100 * p3['conduct_highest_for_scoring']:.0f}%, "
                     f"technology → scoring {100 * p3['technology_highest_for_scoring']:.0f}%, provisions lowest "
                     f"{100 * p3['provisions_lowest_total']:.0f}%: the last two depend on which documents are in. One word, "
                     f"\"model\", makes {100 * imp['I3']['dominant_shares']['statistics|Model']:.0f}% of all statistics mentions."),
        paper_label="Paper, Fig. 3 (= Table 2)", ours_label="Ours (± bootstrap s.e. over documents)")

    # Fig. 9
    r14 = k["extra"]["R14"]["spearman"]
    result_slide(prs, "Fig. 9: the weights document by document", (
        "~1 min. The paper's appendix shows each document's shares. We read its values off the vector PDF and draw both in "
        "the paper's layout. Shapes and the order of documents agree; levels differ for the reason on the previous slide."),
        FIGS / "paper_fig9_layout.png", FIGS / "ours_fig9_layout.png", "",
        why=("the same shapes and the same order of documents (rank correlation " + " / ".join(f"{r14[c]['rho']:.2f}" for c, _ in cats)
             + " for statistics / technology / conduct); only the level is lower, for the reason on the previous slide. One series "
             "in the paper's common-area panel has no label; its values match IIF (2018)."),
        improvement=("each rank correlation with a bootstrap interval over documents: " + "; ".join(
            f"{lab.lower()} {r14[c]['rho']:.2f} [{r14[c]['lo']:.2f}, {r14[c]['hi']:.2f}]" for c, lab in cats) + "."),
        paper_label="Paper, Fig. 9 (values read off its PDF)", ours_label="Ours")

    # Table 3
    paper3 = pd.DataFrame([(f"{a_} vs {b_}", *[f"{P.TABLE3[(a_, b_)][c]:.4f}" + (" *" if c in P.TABLE3_BOLD[(a_, b_)] else "")
                                               for c, _ in cats]) for (a_, b_) in P.TABLE3], columns=["Uses", *[l for _, l in cats]])
    ours3 = pd.DataFrame([(f"{a_} vs {b_}", *[f"{n['table3'][f'{a_} vs {b_}|{c}']['p']:.4f}" + (" *" if n['table3'][f'{a_} vs {b_}|{c}']['p'] < 0.10 else "")
                                              for c, _ in cats]) for (a_, b_) in P.TABLE3], columns=["Uses", *[l for _, l in cats]])
    i4 = imp["I4"]
    result_slide(prs, "Table 3: are the differences between uses significant?", (
        "~1 min. t-tests of the per-document shares between each pair of uses; * marks p < 0.10. We agree on 8 of 9 cells; the "
        "ninth is the paper's own marking error. Then the multiple-testing correction."),
        (paper3, None, [2.4, 1.2, 1.2, 1.2]), (ours3, None, [2.5, 1.23, 1.23, 1.24]), "",
        why=(f"we agree on 8 of 9 cells. The ninth is the paper's own error: it marks p = 0.1097 (scoring vs provisions, "
             f"technology) as significant at 10%; ours is {n['table3']['scoring vs provisions|technology']['p']:.3f}, not significant either."),
        improvement=(f"nine tests at once inflate false positives: after Holm's correction only {i4['holm_10pct']} differences "
                     f"survive at 10% ({i4['holm_5pct']} at 5%); with Benjamini–Hochberg on permutation p-values, {i4['bh_10pct']}."),
        paper_label="Paper, Table 3 (p-values; * marked significant)", ours_label="Ours (Welch t-test p-values; * below 0.10)")

    # Tables 8-10
    def sentiment(t10, t20, share):
        return pd.DataFrame([(lab, *[f"{t10[u][c]:.2f}" for c in ("technology", "conduct")],
                              *[f"{t20[u][c]:.2f}" for c in ("technology", "conduct")], f"{share[u]:.2f}%") for u, lab in uses],
                            columns=["Use", "Tech, 10", "Conduct, 10", "Tech, 20", "Conduct, 20", "Negative, all"])
    i5 = imp["I5"]
    result_slide(prs, "Tables 8–10: the sentiment check", (
        "~1 min. Negative words within 10 or 20 words of each risk term, per mention. The paper finds scoring texts most "
        "negative around technology and conduct terms. We do not; the bootstrap shows why the claim cannot be trusted."),
        sentiment(P.TABLE8_D10, P.TABLE9_D20, P.TABLE10_NEG_SHARE),
        sentiment(n["neg_d10"], n["neg_d20"], n["neg_share"]), "",
        why=("in our corpus credit-scoring texts are not the most negative around technology or conduct terms, at 10 or 20 "
             "words; the overall share of negative words has the paper's order (scoring lowest, provisions highest). With 5–12 "
             "documents per use these averages are noisy, and the paper's window and tokenisation are not stated."),
        improvement=(f"against each text's base rate of negative words, scoring comes out most negative in only "
                     f"{100 * i5['p_scoring_highest_technology']:.0f}% (technology) and {100 * i5['p_scoring_highest_conduct']:.0f}% "
                     "(conduct) of document bootstraps: the paper's claim is close to a coin toss."),
        paper_label="Paper, Tables 8–10", ours_label="Ours")

    # Table 4
    ex = k["scorecard_checks"]["table4_example"]
    i6 = imp["I6"]
    paper4 = pd.DataFrame([(lab, f"{P.TABLE4_RISKINESS[u]:.3f}") for u, lab in uses], columns=["Use", "Riskiness, Z = (2, 4, 3)"])
    ours4 = pd.DataFrame([(lab, f"{ex[u]:.4f}", f"{i6[u]['est']:.3f} ± {i6[u]['se']:.3f}") for u, lab in uses],
                         columns=["Use", "Paper's weights, recomputed", "Our weights (± bootstrap)"])
    result_slide(prs, "Table 4: the worked example", (
        "~0.5 min. The paper's example model with darkness scores (2, 4, 3). Its formula with its own weights gives its numbers "
        "exactly (it truncates rather than rounds). With our weights the example keeps its order."),
        paper4, ours4,
        "recomputing the paper's formula with its own weights gives " + " / ".join(f"{ex[u]:.4f}" for u, _ in uses)
        + ", its numbers (it truncates rather than rounds).",
        improvement=("with our weights and their uncertainty: " + " / ".join(f"{i6[u]['est']:.3f} ± {i6[u]['se']:.3f}" for u, _ in uses)
                     + f"; credit scoring is the riskiest use in {100 * i6['p_scoring_riskiest']:.0f}% of draws."),
        paper_label="Paper, Table 4 and p. 16", ours_label="Ours")

    # Table 7
    paper7 = pd.DataFrame([(mm, f"{P.TABLE7_AUC[mm]:.1f}", f"[{P.TABLE7_CI[mm][0]:.1f}, {P.TABLE7_CI[mm][1]:.1f}]",
                            "–" if mm == "Logit" else f"{P.TABLE7_AUC[mm] - P.TABLE7_AUC['Logit']:+.1f}") for mm in P.TABLE7_AUC],
                          columns=["Model", "AUC %", "95% interval", "Gain"])
    ours7 = pd.DataFrame([(mm, f"{100 * a[mm]['auc']:.1f} ± {100 * a[mm]['se']:.1f}", f"[{100 * a[mm]['lo']:.1f}, {100 * a[mm]['hi']:.1f}]",
                           "–" if mm == "Logit" else f"{100 * a[mm]['gain']:+.1f}") for mm in ["Logit"] + MODELS5],
                         columns=["Model", "AUC %", "95% interval", "Gain"])
    cart_row = ["Logit"] + MODELS5
    fills7 = {(cart_row.index("CART"), 1): ORANGE}
    h = json.loads((ROOT / "results" / "reproduction" / "models" / "cart_diagnostic.json").read_text())["hypotheses"]
    small = [x["auc"] for key, x in h.items() if key.startswith("H2") and 8 <= x["leaves"] <= 12]
    lb = json.loads((ROOT / "results" / "improvements" / "appendix" / "logit_benchmark.json").read_text())
    cal = imp["I12"]["table"]
    result_slide(prs, "Table 7: predictive performance", (
        "~1.5 min. AUC on the test rows. Five of six paper AUCs are inside our intervals and XGBoost is best. CART is the "
        "exception (orange): we fit the depth-7 tree the paper describes; only a different tree gives its number."),
        paper7, (ours7, fills7, None),
        "",
        why=(f"5 of 6 models match (the paper's AUC inside our 95% interval) and XGBoost is best. CART does not: we fit the "
             f"depth-7 tree the paper describes and get {100 * a['CART']['auc']:.1f}. Its 80.4 comes only from a pruned tree of 8–12 "
             f"leaves, as R's rpart builds by default ({100 * min(small):.1f}–{100 * max(small):.1f}), or a depth-10 tree: the tree's "
             "settings, not the method."),
        improvement=(f"paired DeLong tests for every gain (XGBoost +{100 * a['XGBoost']['gain']:.1f} ± {100 * a['XGBoost']['gain_se']:.1f}); "
                     f"calibration added (XGBoost also best Brier, slope {cal['XGBoost']['slope']:.2f}). Exploratory: a logit with log "
                     f"inputs reaches {100 * lb['auc']['Logit, log1p inputs']['auc']:.1f} on validation rows, "
                     f"{100 * lb['xgb_minus_logit_log1p']['diff']:.2f} ± {100 * lb['xgb_minus_logit_log1p']['se']:.2f} behind XGBoost "
                     f"({pfmt(lb['xgb_minus_logit_log1p']['p'])}): most of the ML gain is a weak benchmark."),
        paper_label="Paper, Table 7", ours_label="Ours (± DeLong s.e.; same protocol, our split)")

    # Table 5
    paper5 = pd.DataFrame([(mm, f"{P.TABLE5['stability_sd'][mm]:.3f}", P.TABLE5["n_hyperparameters"][mm], P.TABLE5["transparency"][mm],
                            f"{P.TABLE5['training_seconds'][mm]:.2f}", f"{P.TABLE5['shap_seconds'][mm]:,}") for mm in MODELS5],
                          columns=["Model", "SD of AUC", "Hyper.", "Transp.", "Train s", "SHAP s"])
    ours5 = pd.DataFrame([(mm, f"{px[mm]['auc_sd']:.4f}", P.TABLE5["n_hyperparameters"][mm], P.TABLE5["transparency"][mm],
                           f"{px[mm]['fit_s']:.2f}", f"{px[mm]['shap_s']:.1f}") for mm in MODELS5],
                         columns=["Model", "SD of AUC", "Hyper.", "Transp.", "Train s", "SHAP s"])
    machine = {(-1, 4): ORANGE, (-1, 5): ORANGE}
    cpu = m["meta"]["environment"]["cpu"]
    ranks_c = {mm: px[mm]["shap_common_s"] for mm in MODELS5}
    result_slide(prs, "Table 5: the model-risk proxies", (
        "~1.5 min. Stability, hyperparameters, transparency, training time and SHAP time. Stability has the paper's order; "
        "the two counts are definitions. The orange columns depend on the computer and the software, which the paper "
        "does not report."),
        (paper5, machine, [1.1, 1.1, 0.8, 0.85, 1.0, 1.15]), (ours5, machine, [1.1, 1.15, 0.8, 0.85, 1.1, 1.2]), "",
        why=("stability has the paper's order (Lasso and XGBoost most stable, CART and the MLP least); hyperparameter counts and "
             "transparency are definitions. The orange columns differ only because of the computer and the software, which the "
             f"paper does not report (ours: {cpu}, one thread). Comparable software, comparable time: RF SHAP "
             f"{px['RF']['shap_s']:.0f} s vs 60 s; different software, different time: the Lasso's solver "
             f"({px['Lasso']['fit_s']:.1f} s vs 3.92 s), the MLP's explainer ({px['MLP']['shap_s']:.0f} s vs 2,000 s)."),
        improvement=(f"both of the paper's stability definitions agree (probability {imp['I7']['claim_probability']['auc_sd']:.2f}); "
                     f"an objective size, fitted parameters (Lasso {a['Lasso']['params']:,}, MLP {a['MLP']['params']:,}, RF {a['RF']['params']:,}); "
                     f"with one explainer for every model RF, not the MLP, is slowest to explain ({ranks_c['RF']:.0f} s vs {ranks_c['MLP']:.0f} s)."),
        paper_label="Paper, Table 5", ours_label="Ours (orange: machine- and software-dependent)")

    # Fig. 4
    result_slide(prs, "Fig. 4: the darkness scores", (
        "~1 min. The paper's scores from 1 to 5 are expert judgement, so they cannot be recomputed; we read Fig. 4 off the PDF "
        "and find it disagrees with Table 5. Our improvement replaces judgement by a rule on the measured proxies."),
        crop_paper("fig4"), FIGS / "ours_fig4_radar.png", "",
        why=("the paper's scores are expert judgement, so there is nothing to recompute, and the paper contradicts itself: Fig. 4 "
             "draws RF statistics 2 and MLP 4, Table 5 says 3 and 5, Table 6 uses 3 and 4. No ranking changes."),
        improvement=("scores from a fixed rule on the measured proxies (right): each proxy mapped to 1–5 across models, a "
                     "category's score is their mean. The MLP comes out darkest on statistics and technology, RF on conduct; no "
                     "judgement, and anyone can recompute it."),
        paper_label="Paper, Fig. 4", ours_label="Ours: scores by a rule (I13)")

    # Table 6 / Fig. 5
    i13 = pd.DataFrame(imp["I13"]["table"])
    pf = i13.pivot(index="model", columns="use", values="p_frontier")
    result_slide(prs, "Table 6 and Fig. 5: risk against reward", (
        "~1.5 min. Supervisory risk (weights times scores) against the AUC gain over the logit. XGBoost gives the most gain for "
        "its risk in both. Our CART sits higher because of Table 7. The Monte Carlo below puts a probability on it."),
        crop_paper("fig5"), fig["rr_ours"], "",
        why=("XGBoost gives the most gain for its risk for every use, as in the paper, and Table 6's arithmetic is exact. Our CART "
             "also sits on the frontier because its AUC is 4 points higher (Table 7). The paper's Fig. 5 draws RF with a statistics "
             "score of 2, not Table 6's 3."),
        improvement=("rule-based scores and 1,000 Monte Carlo draws over every noisy input: XGBoost is on the risk–gain frontier in "
                     + f"{100 * pf.loc['XGBoost'].min():.0f}% of draws for every use, CART in {100 * pf.loc['CART', 'scoring']:.0f}% for "
                     "scoring, the others almost never. On a normalised scale \"provisions is least risky\" disappears."),
        paper_label="Paper, Fig. 5", ours_label="Ours: the paper's scores × our weights; our gains (95% CI)")

    # Figs. 6-8
    c = imp["I11"]["claims_seeds"]
    result_slide(prs, "Figs. 6–8: what drives the predictions (SHAP)", (
        "~1 min. SHAP summary plots for RF, XGBoost (in log-odds, as the paper's Fig. 7) and the MLP. Late payments dominate "
        "everywhere; XGBoost's top four match exactly; the other ranking claims do not survive a second fit."),
        [crop_paper("fig6"), crop_paper("fig7"), crop_paper("fig8")],
        [FIGS / "ours_shap_RF.png", FIGS / "ours_shap_XGBoost.png", FIGS / "ours_shap_MLP.png"], "",
        why=("late payments (90+ and 30–59 days) are in the top three for every model, and XGBoost's top four match Fig. 7 exactly. "
             "But revolving utilisation is first for our RF too, and age is not fourth everywhere: the paper ranks from one fit, "
             "and mid-ranked features swap places between fits."),
        improvement=(f"five seeds on one probability scale: \"late payments in the top three\" holds in {c['probability']['top3_late_payments']} "
                     f"to {c['default']['top3_late_payments']} of 5; \"revolving first only for XGBoost\" and \"age fourth\" in "
                     f"{c['default']['revolving_first_only_xgb']} and {c['default']['age_fourth']} of 5."),
        paper_label="Paper, Figs. 6–8 (RF, XGBoost, MLP)", ours_label="Ours (same order)", stacked=True)


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
        "Timing, about 17.5 minutes: the paper 2 · objectives 1 · eleven results with their improvements 12.5 · conclusions 2. "
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

    # 4-14: one slide per result of the paper: the paper's version left, ours right, then what matches or why it
    # differs, then our improvement of that result.
    result_slides(prs, k, fig)

    # 15 Conclusions
    s = slide_with_title(prs, "Conclusions", (
        "Conclusions, ~2 min. Sort every result into: same as the paper; different because of the computer and software; "
        "different because of how the paper reports its own set-up; different in substance. Then what the improvements change."))
    v12 = next(r for r in k["verdicts"] if r["id"] == "R12")["ours"].split(";")[0]
    text(s, (0.64, 2.25, 12.2, 4.9), [
        f"**Same as the paper:** {v12}, XGBoost best; the stability order; every ordering of the text weights, "
        "document by document too; the significance tests (one cell is mismarked in the paper); the paper's own arithmetic.",
        "**Different only because of the computer and software:** training and SHAP times (hardware, libraries, solver and "
        "explainer are not reported; where the software is comparable, the times agree).",
        "**Different because of how the paper reports its set-up:** CART's settings; three versions of the statistics scores "
        "(Tables 5, 6 and Figs. 4, 5); Fig. 1's mixed metrics and mis-cited source; word shares divided by an unstated base.",
        "**Different in substance:** the sentiment check and the SHAP mid-ranks: both rest on too little data to hold.",
        "**What the improvements change:** with uncertainty XGBoost still wins everywhere, but two of the four weight orderings "
        "and most of the significance claims are fragile; a logit with log inputs nearly closes the ML gain (exploratory).",
    ], size=14, space=10)

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
    s = slide_with_title(prs, "Backup: the same protocol on three datasets (I1)", "The improvement of Fig. 1: one protocol, three public datasets, paired DeLong intervals.")
    if "benchmark" in fig:
        picture(s, fig["benchmark"], (0.6, 2.2, 8.2, 4.4))
        text(s, (9.0, 2.4, 4.0, 4.4), [f"**{r['id']}.** {r['finding']}" for r in k["improvement_rows"] if r["id"] == "I1"], size=12)
    s = slide_with_title(prs, "Backup: the scorecard Monte Carlo (I13)", "How often each model is on the risk–gain frontier when every noisy input is redrawn.")
    if "I13" in imp:
        picture(s, frontier_chart(imp["I13"], BUILD / "frontier.png"), (0.6, 2.2, 7.6, 4.4))
        text(s, (8.4, 2.4, 4.6, 4.4), [f"**{r['id']}.** {r['finding']}" for r in k["improvement_rows"] if r["id"] == "I13"], size=12)
    s = slide_with_title(prs, "Backup: Fig. 9 document by document", "R14: values read off the paper's Fig. 9 against ours; read the ranks, not the levels.")
    if "fig9" in fig:
        picture(s, fig["fig9"], (0.5, 2.3, 12.3, 3.8))
        r14 = k["extra"]["R14"]["spearman"]
        text(s, (0.64, 6.2, 12.2, 0.8), ["Spearman ρ, paper vs ours: " + "; ".join(f"{c} {r14[c]['rho']:.2f} [{r14[c]['lo']:.2f}, {r14[c]['hi']:.2f}]" for c in r14)
                                         + ". Levels differ: the paper most likely divided by words without stop words."], size=12)
    s = slide_with_title(prs, "Backup: beyond the paper: appendix factors and a fair logit", "A1-A5 on validation rows; third-party, cyber and privacy need bank data. X1 is exploratory.")
    ap = k.get("appendix", {})
    if ap:
        f = ap["factors"]
        at = pd.DataFrame([(m, f"{f[m]['overfit_gap']:+.3f}", f"{f[m]['features_used']} / 20", f"{f[m]['fe_sensitivity']:.4f}",
                            f"{f[m]['surrogate_r2']:.3f}", f"{100 * f[m]['flip_rate']:.2f}%") for m in f],
                          columns=["Model", "A1 overfitting (AUC gap)", "A2 inputs used", "A3 largest ΔAUC", "A4 surrogate R²", "A5 age flips"])
        table(s, at, (0.64, 2.2, 12.0, 2.6), size=11)
        lines = []
        if "frontier" in ap:
            lines.append("**I14, scorecard with these factors, risk-gain frontier:** "
                         + "; ".join(f"{u} {', '.join(v)}" for u, v in ap["frontier"].items()))
        lines += [f"**X1, {r['what']}:** {r['finding']}" for r in k["improvement_rows"] if r["id"] == "X1"]
        text(s, (0.64, 5.1, 12.0, 2.0), lines, size=12, space=6)
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
