"""Reproduce the paper's scorecard arithmetic (Table 4, Table 6, Fig. 5) and check it for errata.

Nothing here is fitted. The supervisory risk of model m for use j is
Σ_i H̄_ij Z_im (p. 16). We recompute it three ways, each in its own table:

* the paper's Z and the paper's H̄: does the paper's own arithmetic hold?
* the paper's Z and our H̄ (from our corpus): does the ranking survive our weights?
* our gains from Table 7's protocol, for the risk-reward picture (Fig. 5).

Errata found by these checks go to results/reproduction/errata.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mrisk import paper_numbers as P

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "reproduction" / "scorecard"
CATS, USES, MODELS = P.CATEGORIES, P.USES, P.MODELS


def risk(weights: dict, z: dict) -> pd.DataFrame:
    """Σ_i H̄_ij Z_im with H̄ in percent (the paper's Table 6 scale)."""
    return pd.DataFrame({u: {m: sum(weights[u][c] * z[c][m] for c in CATS) for m in MODELS} for u in USES})


def table4_example(weights: dict) -> dict:
    z = P.TABLE4_Z
    return {u: sum(weights[u][c] * z[c] for c in CATS) / (5 * sum(weights[u][c] for c in CATS)) for u in USES}


def implied_z(fig5: pd.DataFrame, model: str, category: str) -> dict:
    """Which Z for (model, category) reproduces the Fig. 5 x-coordinates, other Z as in Table 6."""
    out = {}
    for z_try in range(1, 6):
        z = {c: dict(P.TABLE6_Z[c]) for c in CATS}
        z[category][model] = z_try
        r = risk(P.TABLE2, z) / 100
        pts = fig5[fig5["model"] == model].set_index("use")["risk"]
        out[z_try] = float(max(abs(r.loc[model, u] - pts[u]) for u in USES))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    errata = []

    ex = table4_example(P.TABLE2)
    pd.DataFrame({"recomputed": ex, "paper": P.TABLE4_RISKINESS}).to_csv(OUT / "table4_example.csv")

    t6_paper_z = risk(P.TABLE2, P.TABLE6_Z)
    t6_check = (t6_paper_z - pd.DataFrame(P.TABLE6_RISK)).abs().max().max()
    t6_paper_z.to_csv(OUT / "table6_paper_weights.csv")
    z5 = {"statistics": P.TABLE5["z_statistics"], "technology": P.TABLE5["z_technology"],
          "conduct": P.TABLE5["z_conduct"]}
    t6_table5_z = risk(P.TABLE2, z5)
    if P.TABLE5["z_statistics"]["MLP"] != P.TABLE6_Z["statistics"]["MLP"]:
        errata.append({"id": "E1", "where": "Table 5 vs Table 6 (pp. 18, 20)",
                       "what": "MLP statistics score is 5 in Table 5 and the text but 4 in Table 6",
                       "effect": {u: [float(t6_paper_z.loc['MLP', u]), float(t6_table5_z.loc['MLP', u])] for u in USES}})

    fig5 = pd.read_csv(ROOT / "results" / "paper_extracted" / "fig5_risk_reward.csv")
    fig5_vs_t6 = fig5.assign(table6=[t6_paper_z.loc[m, u] / 100 for m, u in zip(fig5["model"], fig5["use"])])
    fig5_vs_t6["abs_diff"] = (fig5_vs_t6["risk"] - fig5_vs_t6["table6"]).abs()
    fig5_vs_t6.to_csv(OUT / "fig5_vs_table6.csv", index=False)
    bad = fig5_vs_t6[fig5_vs_t6["abs_diff"] > 0.005]
    if len(bad):
        fits = implied_z(fig5, "RF", "statistics")
        errata.append({"id": "E2", "where": "Fig. 5 vs Table 6 (pp. 20, 22)",
                       "what": "Random forest points in Fig. 5 match a statistics score of "
                               f"{min(fits, key=fits.get)}, not the 3 of Tables 5 and 6",
                       "max_abs_diff_by_z": fits, "models_off": sorted(bad["model"].unique().tolist())})

    if t6_paper_z.loc["CART", "provisions"] > t6_paper_z.loc["XGBoost", "provisions"]:
        errata.append({"id": "E3", "where": "Text vs Table 6 (p. 20)",
                       "what": "The text says CART is less risky than XGBoost for provisioning; Table 6 says the opposite",
                       "values": [float(t6_paper_z.loc["CART", "provisions"]), float(t6_paper_z.loc["XGBoost", "provisions"])]})
    for pair, cells in P.TABLE3_BOLD.items():
        for cat in cells:
            if P.TABLE3[pair][cat] >= 0.10:
                errata.append({"id": "E4", "where": "Table 3 (p. 16)",
                               "what": f"p = {P.TABLE3[pair][cat]} ({pair[0]} vs {pair[1]}, {cat}) is marked "
                                       "significant at 10%"})
    sd = P.TABLE5["stability_sd"]
    unstable_text = set(P.CLAIMS["most_unstable"])
    top2 = set(sorted(sd, key=sd.get, reverse=True)[:2])
    if top2 != unstable_text:
        errata.append({"id": "E5", "where": "Text vs Table 5 (p. 18)",
                       "what": f"Text names {' and '.join(sorted(unstable_text))} as least stable; Table 5's two largest SDs are "
                               f"{' and '.join(sorted(top2))}"})
    errata.append({"id": "E6", "where": "p. 17 vs Figs. 6-8",
                   "what": "Text: 11 characteristics and 22 variables with squares; the data have 10 characteristics "
                           "(11 columns with the file's row index) and the SHAP figures show 10 features"})
    dictionary = pd.read_csv(ROOT / "data" / "dictionary.csv")
    sizes = dictionary.drop_duplicates().groupby("category").size().to_dict()
    for cat, n in P.DICTIONARY_SIZE.items():
        if sizes.get(cat) != n:
            errata.append({"id": "E7", "where": "p. 13 vs Table 11", "what": f"{cat}: {n} terms stated, {sizes.get(cat)} listed"})
    errata.append({"id": "E8", "where": "Table 12 vs Fig. 9",
                   "what": "Fig. 9 counts the EBA report on innovative uses of consumer data in the common area "
                           "(Table 12: credit scoring) and has no bar for IIF (2018), which Table 12 lists"})

    # Our weights, if the corpus has been counted.
    ours = ROOT / "results" / "reproduction" / "nlp" / "table2.csv"
    if ours.exists():
        t2 = pd.read_csv(ours, index_col=0)
        weights = {u: {c: 100 * t2.loc[u, c] for c in CATS} for u in USES}
        risk(weights, P.TABLE6_Z).to_csv(OUT / "table6_our_weights.csv")
        pd.Series(table4_example(weights)).to_csv(OUT / "table4_example_our_weights.csv", header=["riskiness"])

    (OUT / "checks.json").write_text(json.dumps({"table6_max_abs_error_with_paper_inputs": float(t6_check),
                                                  "table4_example": ex}, indent=2))
    (ROOT / "results" / "reproduction" / "errata.json").write_text(json.dumps(errata, indent=2))
    print(json.dumps(errata, indent=2))


if __name__ == "__main__":
    main()
