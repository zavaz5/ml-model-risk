"""Our versions of the paper's figures that the slides did not yet show in the paper's own form.

Fig. 1   the paper's plotted literature gains, coloured by what their sources say (data/fig1_sources.csv)
Fig. 4   the darkness-score radar: the paper's (read off its PDF) and ours (the I13 rule), separate panels
Figs 6-8 SHAP summary plots for RF, XGBoost and the MLP (scripts/reproduce_shap_figures.py)
Fig. 9   per-document term shares in the paper's layout: the paper's values and ours, separate figures

The paper's numbers and ours are never drawn in the same panel (CLAUDE.md, rule 2), except in Fig. 1,
where both series are the literature's numbers (plotted by the paper vs printed in the cited source).
Paired figures share their axis ranges so they can be read side by side, as the seminar asks.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mrisk.plots import BLUE, GREEN, GREY, NAVY, ORANGE, _save  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "report" / "figures"
MODELS5 = ["Lasso", "CART", "RF", "XGBoost", "MLP"]
CATS = [("statistics", "Statistics"), ("technology", "Technology"), ("conduct", "Market conduct")]
RED = "#C0392B"

# Which plotted point each label of Fig. 1 belongs to: (complexity class, plotted gain). The pairing
# follows the labels' positions on the page; every value is checked against the extracted figure below.
CLASSES = ["Logit", "Regularisation", "Trees, RF, XGBoost", "Deep networks", "RL, CNN, ensembles"]
LABEL_POINT = {
    "Cheng & Xiang (2017a)": (1, 5.98), "Cheng & Xiang (2017b)": (1, 6.34), "Cheng & Xiang (2017c)": (1, 7.56),
    "Jones et al (2015d)": (1, -0.06), "Guegan Hassani (2018b)": (1, -0.06),
    "Sigrist & Hirnschall (2019)": (2, 16.96), "Petropoulos et al (2018a)": (2, 11.97), "Jones et al (2015a)": (2, 7.97),
    "Jones et al (2015c)": (2, 7.97), "Butaru et al (2016)": (2, 7.56), "Moscatelli et al (2019)": (2, 2.94),
    "Guegan Hassani (2018a)": (2, 1.94),
    "Sirigniano et al (2019a)": (3, 19.95), "Sirigniano et al (2019b)": (3, 5.98), "Petropoulos et al (2018b)": (3, 5.98),
    "Babaev et al (2019)": (3, 4.93), "Jones et al (2015b)": (3, 4.34), "Albanesi y Vamossy (2019)": (3, 2.94),
    "Turiel (2019)": (3, -0.06), "Kvamme et al (2018)": (4, 4.84),
}
# x-axis ranges of the paper's SHAP figures (Figs. 6-8, read off the PDF): RF and MLP in probability, XGBoost in log-odds.
PAPER_SHAP_XLIM = {"RF": (-0.08, 0.40), "XGBoost": (-1.7, 3.2), "MLP": (-0.13, 0.62)}
STATUS_STYLE = {"matches": (GREEN, "matches its source"), "differs": (RED, "source gives another value"),
                "different_metric": (ORANGE, "source reports another metric")}


def fig1() -> Path:
    lit = pd.read_csv(RES / "paper_extracted" / "fig1_literature.csv")
    src = pd.read_csv(ROOT / "data" / "fig1_sources.csv").set_index("label")
    for label, (cls, value) in LABEL_POINT.items():  # guard against typing a value the figure does not contain
        assert ((lit["complexity_rank"] == cls) & (abs(lit["gain_pct"] - value) < 1e-6)).any(), label
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    rng = np.random.default_rng(0)
    seen = set()
    for label, (cls, value) in LABEL_POINT.items():
        row = src.loc[label]
        color, text = STATUS_STYLE[row["status"]]
        x = cls + rng.uniform(-0.18, 0.18)
        plotted = max(value, 0.0)
        source = pd.to_numeric(row["abs_gain_points"], errors="coerce")
        if row["status"] != "matches" and not np.isnan(source):
            ax.plot([x, x], [plotted, source], color=color, lw=0.8, alpha=0.7)
            ax.scatter(x, source, s=34, facecolor="white", edgecolor=color, linewidth=1.3, zorder=3)
        ax.scatter(x, plotted, s=44, color=color, edgecolor="white", linewidth=0.8, zorder=4,
                   label=None if text in seen else text)
        seen.add(text)
    ax.scatter([], [], s=34, facecolor="white", edgecolor=GREY, linewidth=1.3, label="value in the source")
    ax.set_xticks(range(len(CLASSES)), CLASSES, fontsize=8.5)
    ax.set_xlim(0.5, 4.5)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_ylabel("Gain over the logit as plotted (AUC points)")
    ax.set_title("Fig. 1 of the paper, each point checked in its source", fontsize=11, color=NAVY, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, FIG / "paper_fig1_checked.png")


def radar(scores: pd.DataFrame, title: str, path: Path) -> Path:
    """Darkness scores on a 0-5 radar, models clockwise from the top as in the paper's Fig. 4."""
    angles = np.radians((90 - np.arange(len(MODELS5)) * 360 / len(MODELS5)) % 360)  # keep in [0, 2π): a full disc
    fig, ax = plt.subplots(figsize=(4.2, 4.2), subplot_kw={"projection": "polar"})
    for (cat, label), color in zip(CATS, [NAVY, GREY, ORANGE]):
        vals = scores.loc[MODELS5, cat].to_numpy(float)
        ax.plot(np.append(angles, angles[0]), np.append(vals, vals[0]), color=color, lw=2, label=label)
    ax.set_xticks(angles, MODELS5, fontsize=9)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"], fontsize=7, color=GREY)
    ax.set_title(title, fontsize=10, color=NAVY, pad=14)
    ax.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.32, 1.12))
    return _save(fig, path)


def fig4() -> list[Path]:
    paper = pd.read_csv(RES / "paper_extracted" / "fig4_scores.csv").set_index("model")
    i13 = pd.read_csv(RES / "improvements" / "risk" / "i13_scorecard.csv").drop_duplicates("model").set_index("model")
    ours = i13[[f"z_{c}" for c, _ in CATS]].rename(columns=lambda c: c[2:])
    return [radar(paper, "Paper, Fig. 4 (expert scores, read off the PDF)", FIG / "paper_fig4_radar.png"),
            radar(ours, "Ours: scores by the I13 rule from measured proxies", FIG / "ours_fig4_radar.png")]


def fig9_layout(shares: pd.DataFrame, title: str, path: Path, ylims: dict | None = None) -> Path:
    """Per-document shares by category, one line per document, the average in bold (the paper's Fig. 9 layout)."""
    panels = [("capital", "Regulatory capital"), ("scoring", "Credit scoring"), ("provisions", "Provisioning"),
              ("common", "Common area")]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.6))
    for ax, (use, name) in zip(axes.flat, panels):
        docs = shares[shares["use"] == use]
        for _, row in docs.iterrows():
            ax.plot(range(3), [100 * row[c] for c, _ in CATS], color=BLUE, alpha=0.35, lw=1, marker="o", ms=3)
        mean = docs[[c for c, _ in CATS]].mean()
        ax.plot(range(3), 100 * mean.to_numpy(), color=NAVY, lw=2.6, marker="o", ms=5, label=f"average ({len(docs)} documents)")
        ax.set_xticks(range(3), [lab for _, lab in CATS], fontsize=8)
        ax.set_title(name, fontsize=10, color=NAVY, loc="left")
        ax.set_ylabel("% of words", fontsize=8)
        if ylims:
            ax.set_ylim(0, ylims[use])
        ax.legend(frameon=False, fontsize=7)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(title, fontsize=11, color=NAVY, x=0.02, ha="left")
    fig.tight_layout()
    return _save(fig, path)


def fig9() -> list[Path]:
    paper = pd.read_csv(RES / "paper_extracted" / "fig9_documents.csv")
    paper = paper[~paper["doc_id"].str.startswith("Average")].pivot_table(index=["panel", "doc_id"], columns="category",
                                                                             values="share").reset_index()
    paper = paper.rename(columns={"panel": "use"})
    ours = pd.read_csv(RES / "reproduction" / "nlp" / "document_shares.csv")
    cols = [c for c, _ in CATS]
    ylims = {u: 105 * max(paper.loc[paper["use"] == u, cols].max().max(), ours.loc[ours["use"] == u, cols].max().max())
             for u in ("capital", "scoring", "provisions", "common")}  # same axes per panel for the paper and us
    return [fig9_layout(paper, "Paper, Fig. 9 (values read off the PDF; denominator not stated)", FIG / "paper_fig9_layout.png", ylims),
            fig9_layout(ours, "Ours (share of all words)", FIG / "ours_fig9_layout.png", ylims)]


def shap_figures() -> list[Path]:
    import shap

    short = {"NumberOfTimes90DaysLate": "90+ days late", "NumberOfTime30-59DaysPastDueNotWorse": "30–59 days late",
             "NumberOfTime60-89DaysPastDueNotWorse": "60–89 days late", "RevolvingUtilizationOfUnsecuredLines": "Revolving utilisation",
             "age": "Age", "DebtRatio": "Debt ratio", "MonthlyIncome": "Monthly income",
             "NumberOfOpenCreditLinesAndLoans": "Open credit lines", "NumberRealEstateLoansOrLines": "Real-estate loans",
             "NumberOfDependents": "Dependents"}
    out = []
    for name, paper_fig in (("RF", 6), ("XGBoost", 7), ("MLP", 8)):
        d = np.load(RES / "reproduction" / "models" / f"shap_values_{name}.npz")
        expl = shap.Explanation(values=d["values"], data=d["features"], feature_names=[short[n] for n in d["names"]])
        plt.figure()
        shap.plots.beeswarm(expl, max_display=10, show=False, plot_size=(5.6, 3.8))
        plt.xlim(*PAPER_SHAP_XLIM[name])  # the axis range of the paper's figure, so the two can be read side by side
        plt.title(f"Ours, {name} (paper's Fig. {paper_fig}); SHAP in {d['scale']}", fontsize=10, color=NAVY, loc="left")
        path = FIG / f"ours_shap_{name}.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close("all")
        out.append(path)
    return out


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    paths = [fig1(), *fig4(), *fig9(), *shap_figures()]
    print("\n".join(str(p.relative_to(ROOT)) for p in paths))


if __name__ == "__main__":
    main()
