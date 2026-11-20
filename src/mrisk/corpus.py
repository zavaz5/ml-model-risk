"""The regulatory corpus and the paper's phase-2 tables built from it.

Which document belongs to which use: Table 12 and Fig. 9 of the paper
disagree. Fig. 9, which plots the documents the computation actually used,
puts the EBA report on innovative uses of consumer data (Cons_05) in the
common area and has no bar for the IIF's "Explainability in predictive
modelling". We follow Fig. 9 by default (DEVIATIONS.md, D9).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .nlp import (DocumentCounts, category_share, compile_terms, count_document, html_text,
                  load_negative_words, negatives_around, pdf_text)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "corpus_manifest.csv"
CACHE = ROOT / "data" / "processed" / "corpus_counts.pkl"
CATEGORIES = ["statistics", "technology", "conduct"]
USES = ["capital", "scoring", "provisions"]


def assignment(manifest: pd.DataFrame, scheme: str = "fig9") -> pd.DataFrame:
    """Rows (doc_id, use, file) for the documents we have, under the chosen scheme."""
    frame = manifest[manifest["status"].isin(["found", "substitute"])].copy()
    frame["use"] = frame["set"]
    if scheme == "fig9":
        frame.loc[frame["doc_id"] == "Cons_05_EBA_innovative", "use"] = "common"
        frame = frame[frame["doc_id"] != "comun_02_iif_explain"]
    return frame[["doc_id", "use", "file"]].reset_index(drop=True)


def document_text(path: Path) -> str:
    return pdf_text(path) if path.suffix.lower() == ".pdf" else html_text(path)


def count_corpus(dictionary: pd.DataFrame, negative_path: Path, refresh: bool = False) -> dict[str, DocumentCounts]:
    """Count every downloaded document once; cached because PDF parsing is the slow part."""
    manifest = pd.read_csv(MANIFEST)
    key = (tuple(map(tuple, dictionary.to_numpy().tolist())), tuple(manifest["sha256"].fillna("")))
    if CACHE.exists() and not refresh:
        cached = pickle.loads(CACHE.read_bytes())
        if cached["key"] == key:
            return cached["counts"]
    terms = compile_terms(dictionary)
    negative_words = load_negative_words(negative_path)
    counts = {}
    for row in manifest[manifest["status"].isin(["found", "substitute"])].itertuples():
        if row.doc_id in counts:
            continue
        counts[row.doc_id] = count_document(row.doc_id, document_text(ROOT / row.file), terms, negative_words)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(pickle.dumps({"key": key, "counts": counts}))
    return counts


def shares(counts: dict[str, DocumentCounts], docs: pd.DataFrame, terms: set[str] | None = None) -> pd.DataFrame:
    """H_ijk for every document and category (fraction of the document's words)."""
    rows = []
    for doc_id, use in docs[["doc_id", "use"]].itertuples(index=False):
        c = counts[doc_id]
        rows.append({"doc_id": doc_id, "use": use, "n_tokens": c.n_tokens,
                     **{cat: category_share(c, cat, terms) for cat in CATEGORIES}})
    return pd.DataFrame(rows)


def table2(h: pd.DataFrame) -> pd.DataFrame:
    """H̄_ij = mean of H_ijk over use j's documents + mean over the common-area documents (p. 14)."""
    common = h[h["use"] == "common"][CATEGORIES].mean()
    out = h[h["use"].isin(USES)].groupby("use")[CATEGORIES].mean() + common
    out["total"] = out[CATEGORIES].sum(axis=1)
    return out.loc[USES]


def negative_table(counts: dict[str, DocumentCounts], docs: pd.DataFrame, d: int) -> pd.DataFrame:
    """T_ij: negative words within d words of a mention, per mention, pooled over a use's documents (p. 27)."""
    rows = []
    for use in USES:
        ids = docs.loc[docs["use"] == use, "doc_id"]
        row = {"use": use}
        for cat in CATEGORIES:
            pairs = np.array([negatives_around(counts[i], cat, d) for i in ids])
            row[cat] = pairs[:, 0].sum() / max(pairs[:, 1].sum(), 1)
        rows.append(row)
    return pd.DataFrame(rows).set_index("use")


def negative_share(counts: dict[str, DocumentCounts], docs: pd.DataFrame) -> pd.Series:
    """Table 10: negative words as a share of all words, pooled over each use's documents."""
    out = {}
    for use in USES:
        ids = docs.loc[docs["use"] == use, "doc_id"]
        out[use] = sum(counts[i].negatives.sum() for i in ids) / sum(counts[i].n_tokens for i in ids)
    return pd.Series(out)


def mentions_per_document(counts: dict[str, DocumentCounts], terms: dict[str, list[str]]) -> dict[str, float]:
    """Average count per document of a group of terms (p. 11), over all distinct documents."""
    out = {}
    for label, names in terms.items():
        per_doc = [sum(c.term_counts.get(f"statistics|{n}", 0) for n in names) for c in counts.values()]
        out[label] = float(np.mean(per_doc))
    return out
