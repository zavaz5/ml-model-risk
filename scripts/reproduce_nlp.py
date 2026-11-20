"""Reproduce the paper's phase 2: Table 2, Table 3, Fig. 3, Fig. 9, the
mentions per document (p. 11) and the sentiment robustness check (Tables 8-10).

What is computed: for every regulatory document, the share of its words
that are terms of each risk category (Table 11). Nothing is fitted or chosen.
What is compared: the paper's numbers, in separate tables (never side by side
in one table: the corpora differ in version and text extraction).
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import pandas as pd

from mrisk.corpus import CATEGORIES, MANIFEST, USES, assignment, count_corpus, mentions_per_document, \
    negative_share, negative_table, shares, table2
from mrisk.nlp import term_frequencies, top_contributors
from mrisk.uncertainty import welch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "reproduction" / "nlp"
STAT_TERMS = {  # p. 11; grouped the way the paper names them
    "stability": ["Stability"],
    "hyperparameters": ["Hyper parameters"],
    "overfitting": ["Over-fitting", "Over-fit"],
    "dynamic calibration": ["Dynamic calibration"],
    "feature engineering": ["Feature engineering"],
}


def table3(h: pd.DataFrame) -> pd.DataFrame:
    """Welch t-tests of the per-document shares between each pair of uses (p. 15)."""
    rows = []
    for a, b in combinations(USES, 2):
        for cat in CATEGORIES:
            res = welch(h.loc[h["use"] == a, cat].to_numpy(), h.loc[h["use"] == b, cat].to_numpy())
            rows.append({"pair": f"{a} vs {b}", "category": cat, **res})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dictionary = pd.read_csv(ROOT / "data" / "dictionary.csv")
    counts = count_corpus(dictionary, ROOT / "data" / "raw" / "lexicon" / "negative-words.txt")
    docs = assignment(pd.read_csv(MANIFEST))
    print(docs.groupby("use").size().to_string())

    h = shares(counts, docs)
    h.to_csv(OUT / "document_shares.csv", index=False)
    t2 = table2(h)
    t2.to_csv(OUT / "table2.csv")
    print((100 * t2).round(2).to_string())

    t3 = table3(h)
    t3.to_csv(OUT / "table3.csv", index=False)
    print(t3.round(4).to_string(index=False))

    # Over the documents the computation uses (Fig. 9 assignment), as in improve_nlp.py.
    mpd = mentions_per_document({i: counts[i] for i in dict.fromkeys(docs['doc_id'])}, STAT_TERMS)
    print("mentions per document:", {k: round(v, 2) for k, v in mpd.items()})

    sentiment = {d: negative_table(counts, docs, d) for d in (5, 10, 15, 20)}
    for d, table in sentiment.items():
        table.to_csv(OUT / f"negatives_d{d}.csv")
    neg = negative_share(counts, docs)
    neg.to_csv(OUT / "negative_share.csv", header=["share"])
    print(sentiment[10].round(2).to_string(), "\n", (100 * neg).round(2).to_string())

    freq = term_frequencies(list(counts.values()))
    freq.to_csv(OUT / "term_counts.csv", index=False)
    top_contributors(freq).to_csv(OUT / "top_terms.csv", index=False)

    meta = {"documents": {u: int((docs["use"] == u).sum()) for u in USES + ["common"]},
            "missing": pd.read_csv(MANIFEST).query("status == 'missing'")["doc_id"].tolist(),
            "mentions_per_document": mpd,
            "dictionary_size": dictionary.drop_duplicates().groupby("category").size().to_dict(),
            "tokens_total": int(sum(c.n_tokens for c in counts.values()))}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
