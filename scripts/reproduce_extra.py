"""The results the first pass left out: Fig. 9, Table 9 and Fig. 4 (pre-registered addendum, R14-R16).

R14  Fig. 9: our per-document shares against the values read off the paper's
     figure. Levels differ by construction (stop words, DEVIATIONS D15), so the
     test is the rank correlation across documents, per category, with a
     bootstrap interval over documents.
R15  Table 9: the sentiment statements of R5, at d = 20.
R16  Fig. 4: the darkness scores the radar chart plots, against Table 5.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from mrisk import paper_numbers as P
from mrisk.uncertainty import BOOT_REPS, BOOT_SEED

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "results" / "paper_extracted"
NLP = ROOT / "results" / "reproduction" / "nlp"
OUT = ROOT / "results" / "reproduction" / "extra.json"
CATS, USES = P.CATEGORIES, P.USES


def fig9_comparison() -> dict:
    paper = pd.read_csv(EXT / "fig9_documents.csv")
    paper = paper[~paper["doc_id"].str.startswith(("Average", "unlabelled"))]
    paper = paper.pivot_table(index="doc_id", columns="category", values="share")
    ours = pd.read_csv(NLP / "document_shares.csv").drop_duplicates("doc_id").set_index("doc_id")[CATS]
    both = paper.index.intersection(ours.index)
    pairs = pd.concat({"paper": paper.loc[both], "ours": ours.loc[both]}, axis=1)
    pairs.to_csv(NLP / "fig9_comparison.csv")
    rng = np.random.default_rng(BOOT_SEED)
    out = {"documents": int(len(both)), "spearman": {}, "level_ratio": {}}
    for c in CATS:
        x, y = pairs[("paper", c)].to_numpy(), pairs[("ours", c)].to_numpy()
        rho = stats.spearmanr(x, y).statistic
        draws = []
        for _ in range(BOOT_REPS):
            k = rng.integers(0, len(x), len(x))
            draws.append(stats.spearmanr(x[k], y[k]).statistic)
        out["spearman"][c] = {"rho": float(rho), "lo": float(np.nanpercentile(draws, 2.5)),
                              "hi": float(np.nanpercentile(draws, 97.5)), "p": float(stats.spearmanr(x, y).pvalue)}
        out["level_ratio"][c] = float(np.nanmedian(np.where(y > 0, x / np.where(y > 0, y, 1), np.nan)))
    rhos = [out["spearman"][c]["rho"] for c in CATS]
    out["verdict"] = ("reproduced" if min(rhos) >= 0.7 else
                      "close" if sum(r >= 0.4 for r in rhos) >= 2 else "not reproduced")
    unl = pd.read_csv(EXT / "fig9_documents.csv").query("doc_id == 'unlabelled'")
    out["unlabelled_series"] = {"panel": unl["panel"].iloc[0], **unl.set_index("category")["share"].round(4).to_dict()} if len(unl) else None
    return out


def table9() -> dict:
    neg = pd.read_csv(NLP / "negatives_d20.csv", index_col=0)
    tech = bool(neg.loc["scoring", "technology"] > neg.drop("scoring")["technology"].max())
    cond = bool(neg.loc["scoring", "conduct"] > neg.drop("scoring")["conduct"].max())
    return {"scoring_highest_technology": tech, "scoring_highest_conduct": cond,
            "verdict": "reproduced" if tech and cond else ("close" if tech or cond else "not reproduced"),
            "ours": neg.round(2).to_dict(orient="index")}


def fig4() -> dict:
    f4 = pd.read_csv(EXT / "fig4_scores.csv").set_index("model")
    table5 = {"statistics": P.TABLE5["z_statistics"], "technology": P.TABLE5["z_technology"],
              "conduct": P.TABLE5["z_conduct"]}
    diffs = [{"model": m, "category": c, "fig4": float(round(f4.loc[m, c])), "table5": table5[c][m],
              "table6": P.TABLE6_Z[c][m]}
             for m in P.MODELS for c in CATS if round(f4.loc[m, c]) != table5[c][m]]
    return {"scores": f4.round(0).astype(int).to_dict(orient="index"), "differences_from_table5": diffs,
            "verdict": "reproduced" if not diffs else "not reproduced"}


def main() -> None:
    out = {"R14": fig9_comparison(), "R15": table9(), "R16": fig4()}
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
