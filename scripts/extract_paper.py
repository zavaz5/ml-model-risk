"""Recover machine-readable data from the paper's PDF (CC BY 4.0, in paper/).

1. Table 11, the risk terminology, from word positions (three columns).
2. Fig. 1, Fig. 5 and Fig. 9 are vector graphics: the markers' coordinates
   are mapped to data values with the axes' tick labels. This gives the
   paper's plotted values exactly, instead of reading them off by eye.

Outputs: data/dictionary.csv (committed: it is the paper's Table 11) and
results/paper_extracted/*.csv.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper" / "alonso_carbo_2022.pdf"
OUT = ROOT / "results" / "paper_extracted"

TABLE11_PAGE = 29  # zero-based: "Page 30 of 35"
COLUMN_EDGES = {"statistics": (110, 255), "technology": (255, 390), "conduct": (390, 520)}


def table11(doc: pymupdf.Document) -> pd.DataFrame:
    """Rebuild Table 11 row by row; a line closer than 10 pt to the previous one continues it."""
    words = [w for w in doc[TABLE11_PAGE].get_text("words") if 115 < w[1] < 720]
    rows = []
    for cat, (lo, hi) in COLUMN_EDGES.items():
        col = sorted((w for w in words if lo <= w[0] < hi), key=lambda w: (round(w[1]), w[0]))
        lines: list[tuple[float, list[str]]] = []
        for w in col:
            text = w[4].replace("​", "")
            if lines and abs(w[1] - lines[-1][0]) < 2:
                lines[-1][1].append(text)
            elif lines and w[1] - lines[-1][0] < 10:  # wrapped entry, e.g. "Independent conditional / expectations"
                lines[-1][1].append(text)
                lines[-1] = (w[1], lines[-1][1])
            else:
                lines.append((w[1], [text]))
        rows += [{"category": cat, "term": " ".join(t)} for _, t in lines]
    return pd.DataFrame(rows)


def _ticks(page: pymupdf.Page, clip: pymupdf.Rect, pattern: str, axis: str):
    """Tick values and their centre coordinates on one axis."""
    vals, pos = [], []
    for w in page.get_text("words", clip=clip):
        if re.fullmatch(pattern, w[4]):
            vals.append(float(w[4].rstrip("%")))
            pos.append((w[0] + w[2]) / 2 if axis == "x" else (w[1] + w[3]) / 2)
    slope, intercept = np.polyfit(pos, vals, 1)
    return lambda c: slope * c + intercept


def _markers(page: pymupdf.Page, clip: pymupdf.Rect, max_size: float = 4.0):
    """Centres and fill colours of the small filled shapes (data markers) inside ``clip``."""
    out = []
    for d in page.get_drawings():
        r = d["rect"]
        if r.width < max_size and r.height < max_size and clip.contains(r) and d.get("fill"):
            out.append(((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2, tuple(round(c, 3) for c in d["fill"])))
    return sorted(set(out))


def fig1(doc: pymupdf.Document) -> pd.DataFrame:
    """Literature review: AUC gain over the logit by algorithmic-complexity class (page 4)."""
    page = doc[3]
    fy = _ticks(page, pymupdf.Rect(130, 80, 150, 240), r"\d+%", "y")
    classes = {152: "Logit", 219: "Regularisation (PCA, Lasso, Ridge)", 286: "Trees, RF, XGBoost",
               352: "Deep neural networks", 419: "RL, CNN, ensembles"}
    # Two papers can sit on the same spot; get_drawings keeps both, so count duplicates.
    rows = []
    for d in page.get_drawings():
        r = d["rect"]
        if r.width < 4 and r.height < 4 and len(d["items"]) == 4 and d.get("fill"):
            x, y = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
            key = min(classes, key=lambda c: abs(c - x))
            rows.append({"complexity_rank": sorted(classes).index(key), "class": classes[key],
                         "gain_pct": round(fy(y), 2)})
    return pd.DataFrame(rows).sort_values(["complexity_rank", "gain_pct"]).reset_index(drop=True)


def fig5(doc: pymupdf.Document) -> pd.DataFrame:
    """Risk-reward plot (page 22): one marker per model and use."""
    page = doc[21]
    plot = pymupdf.Rect(150, 80, 470, 200)
    fx = _ticks(page, pymupdf.Rect(140, 192, 480, 204), r"0\.\d0", "x")
    fy = _ticks(page, pymupdf.Rect(130, 80, 155, 200), r"\d\.00%", "y")
    # Legend markers sit at x ≈ 412, y = 107 / 120 / 132 (credit scoring, provisioning, capital).
    legend = {107: "scoring", 120: "provisions", 132: "capital"}
    colours = {}
    for x, y, c in _markers(page, pymupdf.Rect(405, 100, 420, 140)):
        colours[c] = legend[min(legend, key=lambda k: abs(k - y))]
    rows = []
    for x, y, c in _markers(page, plot):
        if c in colours and not (405 < x < 420 and 100 < y < 140):
            rows.append({"use": colours[c], "risk": round(fx(x), 4), "gain_pct": round(fy(y), 2)})
    frame = pd.DataFrame(rows).drop_duplicates()
    # Name each point by its gain (Table 7 gains are distinct: 0.2, 0.4, 4.2, 5.3, 1.7).
    gains = {"Lasso": 0.2, "CART": 0.4, "RF": 4.2, "XGBoost": 5.3, "MLP": 1.7}
    frame["model"] = [min(gains, key=lambda m: abs(gains[m] - g)) for g in frame["gain_pct"]]
    return frame.sort_values(["use", "model"]).reset_index(drop=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(PDF)
    dictionary = table11(doc)
    dictionary.to_csv(ROOT / "data" / "dictionary.csv", index=False)
    print(dictionary.groupby("category").size().to_string())
    fig1(doc).to_csv(OUT / "fig1_literature.csv", index=False)
    f5 = fig5(doc)
    f5.to_csv(OUT / "fig5_risk_reward.csv", index=False)
    print(f5.to_string(index=False))


if __name__ == "__main__":
    main()
