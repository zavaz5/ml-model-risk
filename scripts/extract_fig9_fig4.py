"""Read the values behind the paper's Fig. 9 (per-document term shares) and Fig. 4 (darkness scores).

Both figures are vector graphics. Fig. 9 has four panels (capital, credit
scoring, provisioning, common area); in each, a document is one coloured
series with a marker per risk category, and the legend pairs each colour
with the document's file label. Values come from the markers' positions,
calibrated on each panel's own axis labels. Fig. 4 is a bar chart; bar
heights are read the same way.

Outputs: results/paper_extracted/fig9_documents.csv, fig4_scores.csv.
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
CATS = ["statistics", "technology", "conduct"]

# (use, y-range of the panel, x of the three category columns, x-bands of the legend markers)
PANELS = [
    ("capital", (140, 270), (245.4, 299.8, 353.8), ((301.0, 302.6), (376.0, 377.0))),
    ("scoring", (295, 415), (240.5, 290.0, 339.0), ((355.0, 358.0),)),
    ("provisions", (435, 560), (240.2, 290.0, 339.4), ((359.0, 361.5),)),
    ("common", (575, 705), (240.0, 289.7, 339.8), ((358.0, 360.5),)),
]


def markers(page: pymupdf.Page) -> list[tuple[float, float, tuple]]:
    out = []
    for d in page.get_drawings():
        r = d["rect"]
        if r.width < 4 and r.height < 4 and d.get("fill"):
            out.append(((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2, tuple(round(c, 3) for c in d["fill"])))
    return out


def lines_text(page: pymupdf.Page) -> list[tuple[float, float, float, str]]:
    out = []
    for b in page.get_text("dict")["blocks"]:
        for line in b.get("lines", []):
            text = " ".join(s["text"] for s in line["spans"]).strip()
            x0, y0, x1, y1 = line["bbox"]
            out.append((x0, (y0 + y1) / 2, x1, text))
    return out


def y_axis(texts, ylo: float, yhi: float):
    """Linear map from page y to value, fitted on the panel's numeric tick labels."""
    ticks = [(y, float(t)) for x0, y, x1, t in texts if ylo <= y <= yhi and x0 < 190 and re.fullmatch(r"-?\d+\.\d+", t)]
    ys, vals = zip(*ticks)
    slope, intercept = np.polyfit(ys, vals, 1)
    return lambda y: slope * y + intercept


def fig9(doc: pymupdf.Document) -> pd.DataFrame:
    page = doc[28]
    mk, texts = markers(page), lines_text(page)
    rows = []
    for use, (ylo, yhi), cols, bands in PANELS:
        fy = y_axis(texts, ylo, yhi)
        labels = [(y, t, x0) for x0, y, x1, t in texts if ylo <= y <= yhi and x0 > 300 and re.match(r"(Cap|Cons|prov|comun|Average)", t)]
        legend = [m for m in mk if ylo <= m[1] <= yhi and any(a <= m[0] <= b for a, b in bands)]
        plot = [m for m in mk if ylo <= m[1] <= yhi and min(abs(m[0] - c) for c in cols) < 1.0]
        for lx, ly, colour in legend:
            near = [lt for lt in labels if 0 < lt[2] - lx < 15] or labels
            label = min(near, key=lambda lt: abs(lt[0] - ly))
            if abs(label[0] - ly) > 4:  # a legend entry whose text the PDF does not carry
                label = (ly, "unlabelled", lx)
            series = [m for m in plot if m[2] == colour]
            for k, c in enumerate(cols):
                pts = [m for m in series if abs(m[0] - c) < 1.0]
                rows.append({"panel": use, "doc_id": label[1], "category": CATS[k],
                             "share": round(float(fy(pts[0][1])), 5) if len(pts) == 1 else np.nan,
                             "markers_found": len(pts)})
    return pd.DataFrame(rows)


def fig4(doc: pymupdf.Document) -> pd.DataFrame:
    """Fig. 4 (page 19) is a radar chart: one axis per model, one polygon per risk category.

    The centre and scale come from the axis labels ("-" is 0, "5.0" is 5) on
    the vertical (Lasso) axis; each polygon vertex is assigned to the nearest
    model axis, and the legend's line colours name the categories.
    """
    page = doc[18]
    texts = lines_text(page)
    zero = next(y for x0, y, x1, t in texts if t == "-" and 280 < x0 < 300)
    five = next(y for x0, y, x1, t in texts if t == "5.0" and 280 < x0 < 300)
    drawings = page.get_drawings()
    centre_x = next(d for d in drawings if len(d["items"]) == 5)["items"][1][1].x  # the Lasso vertex is on the vertical axis
    unit = (zero - five) / 5
    axes = {"Lasso": 90, "CART": 18, "RF": -54, "XGBoost": -126, "MLP": 162}
    legend = sorted((d["items"][0][1].y, tuple(round(c, 3) for c in d["color"])) for d in drawings
                    if len(d["items"]) == 1 and d["rect"].y0 < 115 and d["rect"].x0 > 350)
    names = dict(zip([c for _, c in legend], CATS))
    rows = []
    for d in drawings:
        if len(d["items"]) != 5:
            continue
        category = names[tuple(round(c, 3) for c in d["color"])]
        for item in d["items"]:
            x, y = item[1].x - centre_x, zero - item[1].y
            angle = np.degrees(np.arctan2(y, x))
            model = min(axes, key=lambda m: abs((angle - axes[m] + 180) % 360 - 180))
            rows.append({"category": category, "model": model, "score": round(float(np.hypot(x, y) / unit), 2)})
    return pd.DataFrame(rows).pivot(index="model", columns="category", values="score").reset_index()


def main() -> None:
    doc = pymupdf.open(PDF)
    f9 = fig9(doc)
    f9.to_csv(OUT / "fig9_documents.csv", index=False)
    print(f9.groupby("panel")["markers_found"].agg(["count", "min", "max"]))
    print(f9.pivot_table(index=["panel", "doc_id"], columns="category", values="share").round(4).to_string())
    f4 = fig4(doc)
    f4.to_csv(OUT / "fig4_scores.csv", index=False)
    print(f4.to_string())


if __name__ == "__main__":
    main()
