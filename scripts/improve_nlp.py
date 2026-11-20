"""Pre-registered improvements on the text side: I2-I6 (docs/preregistration.md).

I2  Bootstrap intervals for the mentions per document (p. 11).
I3  Document bootstrap of the weights H̄ (Table 2) and a dictionary audit.
I4  Welch degrees of freedom, permutation tests and multiple-testing corrections (Table 3).
I5  Negative-word lift over the base rate (Tables 8-10).
I6  The worked example (Table 4) with the weight uncertainty propagated.

Nothing is fitted; the only "choice" is the trimming rule of I3, fixed in
the pre-registration before the corpus was counted.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from mrisk.corpus import CATEGORIES, MANIFEST, USES, assignment, count_corpus, shares, table2
from mrisk.nlp import negatives_around, term_frequencies
from mrisk.paper_numbers import TABLE4_Z
from mrisk.uncertainty import BOOT_REPS, BOOT_SEED, benjamini_hochberg, holm, permutation_test, welch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "improvements" / "nlp"
STAT_TERMS = {"stability": ["Stability"], "hyperparameters": ["Hyper parameters"],
              "overfitting": ["Over-fitting", "Over-fit"], "dynamic calibration": ["Dynamic calibration"],
              "feature engineering": ["Feature engineering"]}
TRIM_SHARE = 0.10  # pre-registered: drop a term that alone makes up more than 10% of its category


def claims(t2: pd.DataFrame) -> dict[str, bool]:
    """The three statements the paper draws from Table 2 (p. 15)."""
    return {
        "statistics_highest_for_capital": bool(t2.loc["capital", "statistics"] > t2.drop("capital")["statistics"].max()),
        "technology_highest_for_scoring": bool(t2.loc["scoring", "technology"] > t2.drop("scoring")["technology"].max()),
        "conduct_highest_for_scoring": bool(t2.loc["scoring", "conduct"] > t2.drop("scoring")["conduct"].max()),
        "provisions_lowest_total": bool(t2.loc["provisions", "total"] < t2.drop("provisions")["total"].min()),
    }


def resample_documents(h: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Draw documents with replacement within each use and within the common area."""
    parts = []
    for use, group in h.groupby("use"):
        parts.append(group.iloc[rng.integers(0, len(group), len(group))])
    return pd.concat(parts)


def improvement_3(h: pd.DataFrame, counts, docs, freq) -> tuple[dict, pd.DataFrame, list[str]]:
    rng = np.random.default_rng(BOOT_SEED)
    draws = [table2(resample_documents(h, rng)) for _ in range(BOOT_REPS)]
    stack = np.stack([d[CATEGORIES + ["total"]].to_numpy() for d in draws])
    t2 = table2(h)
    se = pd.DataFrame(stack.std(axis=0, ddof=1), index=t2.index, columns=CATEGORIES + ["total"])
    prob = pd.DataFrame([claims(d) for d in draws]).mean().to_dict()
    # Dictionary audit: terms that dominate their category across the whole corpus.
    tot = freq.groupby(["category", "term"])["count"].sum()
    share = tot / tot.groupby(level="category").transform("sum")
    dominant = sorted(f"{c}|{t}" for (c, t), v in share.items() if v > TRIM_SHARE)
    keep = {t for (c, t), v in share.items() if v <= TRIM_SHARE}
    trimmed = table2(shares(counts, docs, terms=keep))
    pd.concat({"estimate": t2, "se": se, "trimmed": trimmed}).to_csv(OUT / "i3_weights.csv")
    np.save(OUT / "i3_weight_draws.npy", stack)
    return {"claim_probability": prob, "claims_point": claims(t2), "claims_trimmed": claims(trimmed),
            "dominant_terms": dominant, "dominant_shares": {k: float(share[tuple(k.split("|"))]) for k in dominant}}, se, dominant


def improvement_2(counts) -> dict:
    out = {}
    for label, names in STAT_TERMS.items():
        per_doc = np.array([sum(c.term_counts.get(f"statistics|{n}", 0) for n in names) for c in counts.values()])
        rng = np.random.default_rng(BOOT_SEED)
        means = [per_doc[rng.integers(0, len(per_doc), len(per_doc))].mean() for _ in range(BOOT_REPS)]
        top3 = np.sort(per_doc)[::-1][:3].sum() / max(per_doc.sum(), 1)
        out[label] = {"mean": float(per_doc.mean()), "lo": float(np.percentile(means, 2.5)),
                      "hi": float(np.percentile(means, 97.5)), "top3_share": float(top3),
                      "documents_with_mention": int((per_doc > 0).sum()), "documents": int(len(per_doc))}
    return out


def improvement_4(h: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b in combinations(USES, 2):
        for cat in CATEGORIES:
            xa, xb = h.loc[h["use"] == a, cat].to_numpy(), h.loc[h["use"] == b, cat].to_numpy()
            w = welch(xa, xb)
            rows.append({"pair": f"{a} vs {b}", "category": cat, "welch_t": w["t"], "welch_df": w["df"],
                         "naive_df": min(len(xa), len(xb)) - 1, "welch_p": w["p"],
                         "permutation_p": permutation_test(xa, xb)})
    frame = pd.DataFrame(rows)
    frame["holm_p"] = holm(frame["permutation_p"].tolist())
    frame["bh_p"] = benjamini_hochberg(frame["permutation_p"].tolist())
    return frame


def lift_table(counts, docs: pd.DataFrame, d: int, ids_by_use: dict[str, list[str]]) -> pd.DataFrame:
    """Negatives per mention divided by what a random word position would show (2d × base rate)."""
    rows = {}
    for use in USES:
        ids = ids_by_use[use]
        base = sum(counts[i].negatives.sum() for i in ids) / sum(counts[i].n_tokens for i in ids)
        row = {}
        for cat in CATEGORIES:
            pairs = np.array([negatives_around(counts[i], cat, d) for i in ids])
            per_mention = pairs[:, 0].sum() / max(pairs[:, 1].sum(), 1)
            row[cat] = per_mention / (2 * d * base)
        rows[use] = row
    return pd.DataFrame(rows).T


def improvement_5(counts, docs: pd.DataFrame, d: int = 10) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    ids = {u: docs.loc[docs["use"] == u, "doc_id"].tolist() for u in USES}
    point = lift_table(counts, docs, d, ids)
    rng = np.random.default_rng(BOOT_SEED)
    draws = []
    for _ in range(BOOT_REPS):
        sample = {u: [v[i] for i in rng.integers(0, len(v), len(v))] for u, v in ids.items()}
        draws.append(lift_table(counts, docs, d, sample))
    stack = np.stack([t.to_numpy() for t in draws])
    se = pd.DataFrame(stack.std(axis=0, ddof=1), index=point.index, columns=point.columns)
    tech = float(np.mean([t.loc["scoring", "technology"] > t.drop("scoring")["technology"].max() for t in draws]))
    cond = float(np.mean([t.loc["scoring", "conduct"] > t.drop("scoring")["conduct"].max() for t in draws]))
    return point, se, tech, cond


def improvement_6(stack: np.ndarray, t2: pd.DataFrame) -> dict:
    """Riskiness of the Table 4 model, Σ H̄ Z / (5 Σ H̄), over the bootstrap draws of H̄."""
    z = np.array([TABLE4_Z[c] for c in CATEGORIES])
    weights = stack[:, :, :3]  # draws × uses × categories
    risk = (weights * z).sum(axis=2) / (5 * weights.sum(axis=2))
    point = (t2[CATEGORIES].to_numpy() * z).sum(axis=1) / (5 * t2[CATEGORIES].to_numpy().sum(axis=1))
    out = {u: {"est": float(point[i]), "se": float(risk[:, i].std(ddof=1))} for i, u in enumerate(USES)}
    out["p_scoring_riskiest"] = float(np.mean(risk[:, 1] > np.maximum(risk[:, 0], risk[:, 2])))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dictionary = pd.read_csv(ROOT / "data" / "dictionary.csv")
    counts = count_corpus(dictionary, ROOT / "data" / "raw" / "lexicon" / "negative-words.txt")
    docs = assignment(pd.read_csv(MANIFEST))
    h = shares(counts, docs)
    freq = term_frequencies([counts[i] for i in dict.fromkeys(docs["doc_id"])])
    summary = {"I2": improvement_2({i: counts[i] for i in dict.fromkeys(docs["doc_id"])})}
    i3, _, _ = improvement_3(h, counts, docs, freq)
    summary["I3"] = i3
    i4 = improvement_4(h)
    i4.to_csv(OUT / "i4_tests.csv", index=False)
    summary["I4"] = {"holm_10pct": int((i4["holm_p"] < 0.10).sum()), "holm_5pct": int((i4["holm_p"] < 0.05).sum()),
                     "bh_10pct": int((i4["bh_p"] < 0.10).sum()), "welch_10pct": int((i4["welch_p"] < 0.10).sum())}
    point, se, p_tech, p_cond = improvement_5(counts, docs)
    pd.concat({"lift": point, "se": se}).to_csv(OUT / "i5_lift.csv")
    summary["I5"] = {"p_scoring_highest_technology": p_tech, "p_scoring_highest_conduct": p_cond}
    stack = np.load(OUT / "i3_weight_draws.npy")
    summary["I6"] = improvement_6(stack, table2(h))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
