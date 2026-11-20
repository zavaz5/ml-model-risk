"""Figures for the README and the slides. Each function draws one source only:
the paper's numbers and ours never share a panel (CLAUDE.md, rule 2)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

NAVY, BLUE, ORANGE, GREEN, GREY = "#0F2C68", "#2A78D6", "#EB6834", "#1BAF7A", "#8A8F98"
USE_STYLE = {"capital": (BLUE, "o", "Capital"), "scoring": (ORANGE, "s", "Credit scoring"),
             "provisions": (GREEN, "^", "Provisions")}
plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#555555"})


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def risk_reward(risk: pd.DataFrame, gain: pd.Series, path: Path, title: str, risk_se: pd.DataFrame | None = None,
                gain_se: pd.Series | None = None, xlabel: str = "Supervisory model risk",
                xlim: tuple | None = None, ylim: tuple | None = None) -> Path:
    """One marker per model and use: risk on x, AUC gain over the logit on y (the paper's Fig. 5 layout)."""
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for use, (color, marker, label) in USE_STYLE.items():
        xerr = None if risk_se is None else risk_se[use].reindex(gain.index)
        yerr = None if gain_se is None else gain_se.reindex(gain.index)
        ax.errorbar(risk[use].reindex(gain.index), gain, xerr=xerr, yerr=yerr, fmt=marker, color=color,
                    ms=7, mec="white", mew=1.2, elinewidth=0.9, capsize=2, label=label)
    for model in gain.index:
        x = risk.loc[model].max()
        ax.annotate(model, (x, gain[model]), xytext=(8, -3), textcoords="offset points", fontsize=9, color=NAVY)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("AUC gain over the logit (points)")
    if xlim:  # paired figures share their axes, as the seminar asks
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=11, color=NAVY, loc="left")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    return _save(fig, path)


def weights_bars(table: pd.DataFrame, path: Path, title: str, se: pd.DataFrame | None = None,
                 ylim: tuple | None = None) -> Path:
    """Category weights per use (Table 2 / Fig. 3 layout), in percent of all words."""
    cats = ["statistics", "technology", "conduct"]
    x = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for k, (use, (color, _, label)) in enumerate(USE_STYLE.items()):
        err = None if se is None else 100 * se.loc[use, cats]
        ax.bar(x + (k - 1) * 0.26, 100 * table.loc[use, cats], 0.24, yerr=err, capsize=2, color=color,
               label=label, error_kw={"lw": 0.8})
    ax.set_xticks(x, ["Statistics", "Technology", "Market conduct"])
    ax.set_ylabel("Share of all words (%)")
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=11, color=NAVY, loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, path)


def benchmark(frame: pd.DataFrame, path: Path) -> Path:
    """I1: AUC gain over the logit with 95% paired DeLong intervals, per dataset."""
    data = frame[(frame["model"] != "_summary") & (frame["model"] != "Logit")]
    sets = list(dict.fromkeys(data["dataset"]))
    models = list(dict.fromkeys(data["model"]))
    names = {"gmsc": "Give Me Some Credit", "taiwan_default": "Taiwan default", "german_credit": "German credit"}
    colors = [NAVY, BLUE, ORANGE]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    x = np.arange(len(models))
    for k, s in enumerate(sets):
        d = data[data["dataset"] == s].set_index("model").loc[models]
        ax.errorbar(x + (k - 1) * 0.22, 100 * d["gain"], yerr=[100 * (d["gain"] - d["gain_lo"]), 100 * (d["gain_hi"] - d["gain"])],
                    fmt="o", color=colors[k], ms=6, capsize=2, elinewidth=1, label=names[s])
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xticks(x, models)
    ax.set_ylabel("AUC gain over the logit (points)")
    ax.legend(frameon=False, fontsize=8, ncol=1, loc="lower left")
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, path)


def shap_ranks(ranks: dict[str, list[str]], path: Path, title: str, top: int = 6) -> Path:
    """Top features per model as a rank table image (model columns, rank rows)."""
    short = {"NumberOfTimes90DaysLate": "90+ days late", "NumberOfTime30-59DaysPastDueNotWorse": "30–59 days late",
             "NumberOfTime60-89DaysPastDueNotWorse": "60–89 days late", "RevolvingUtilizationOfUnsecuredLines": "Revolving utilisation",
             "age": "Age", "DebtRatio": "Debt ratio", "MonthlyIncome": "Monthly income",
             "NumberOfOpenCreditLinesAndLoans": "Open credit lines", "NumberRealEstateLoansOrLines": "Real-estate loans",
             "NumberOfDependents": "Dependents"}
    models = list(ranks)
    fig, ax = plt.subplots(figsize=(1.9 * len(models) + 0.6, 0.36 * top + 0.6))
    ax.axis("off")
    cells = [[short.get(ranks[m][r], ranks[m][r]) for m in models] for r in range(top)]
    tab = ax.table(cellText=cells, colLabels=models, rowLabels=[str(r + 1) for r in range(top)], loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1, 1.35)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        if r == 0:
            cell.set_facecolor(NAVY)
            cell.get_text().set_color("white")
    ax.set_title(title, fontsize=11, color=NAVY, loc="left")
    return _save(fig, path)


def fig9_scatter(pairs: pd.DataFrame, path: Path) -> Path:
    """R14: each document's share in the paper's Fig. 9 against ours, one panel per category (log scales)."""
    cats = [("statistics", "Statistics"), ("technology", "Technology"), ("conduct", "Market conduct")]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
    for ax, (c, label) in zip(axes, cats):
        x, y = pairs[("paper", c)].clip(lower=1e-4), pairs[("ours", c)].clip(lower=1e-4)
        ax.scatter(x, y, s=18, color=BLUE, edgecolor="white", linewidth=0.6)
        lo, hi = min(x.min(), y.min()) * 0.8, max(x.max(), y.max()) * 1.2
        ax.plot([lo, hi], [lo, hi], color=GREY, lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(label, fontsize=10, color=NAVY, loc="left")
        ax.set_xlabel("Paper (Fig. 9)")
    axes[0].set_ylabel("Ours")
    fig.tight_layout()
    return _save(fig, path)
