"""Why are our word shares about half of the paper's? A diagnostic, not a tuning step.

Our Table 2 totals (~7% of all words) and negative-word shares (~2%) are
about half the paper's (~14% and ~4%), while every ordering matches. One
explanation would scale every level by the same factor: the paper dividing
by the number of words *after removing stop words*, a common preprocessing
step it does not mention. We recount with scikit-learn's English stop-word
list removed and compare levels. Nothing here feeds our main results.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from mrisk.corpus import CATEGORIES, MANIFEST, USES, assignment, document_text, negative_share, negative_table, \
    shares, table2
from mrisk.nlp import compile_terms, count_document, load_negative_words, tokens

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "reproduction" / "nlp" / "stopword_diagnostic.json"


def main() -> None:
    dictionary = pd.read_csv(ROOT / "data" / "dictionary.csv")
    terms = compile_terms(dictionary)
    negative = load_negative_words(ROOT / "data" / "raw" / "lexicon" / "negative-words.txt")
    manifest = pd.read_csv(MANIFEST)
    docs = assignment(manifest)
    counts, kept_share = {}, []
    for row in manifest[manifest["status"].isin(["found", "substitute"])].drop_duplicates("doc_id").itertuples():
        toks = tokens(document_text(ROOT / row.file))
        kept = [t for t in toks if t not in ENGLISH_STOP_WORDS]
        kept_share.append(len(kept) / len(toks))
        counts[row.doc_id] = count_document(row.doc_id, " ".join(kept), terms, negative)
    t2 = table2(shares(counts, docs))
    neg10 = negative_table(counts, docs, 10)
    neg_share = negative_share(counts, docs)
    out = {"kept_share_mean": float(np.mean(kept_share)),
           "table2_pct": (100 * t2).round(2).to_dict(orient="index"),
           "neg_d10": neg10.round(2).to_dict(orient="index"),
           "neg_share_pct": (100 * neg_share).round(2).to_dict()}
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
