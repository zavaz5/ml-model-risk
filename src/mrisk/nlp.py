"""Count risk terms in regulatory documents (the paper's phase 2, pp. 13-16, 26-28).

The paper counts how often the terms of each risk category (Table 11)
appear in each document, "plus their lemmatization", relative to the
document's word count. It does not publish code, so these choices are ours
(DEVIATIONS.md, D10-D13):

* text: PyMuPDF plain text; words split across lines with a hyphen are joined;
* tokens: lower-case runs of letters and digits;
* lemmatisation: plural nouns to singular (what NLTK's WordNet lemmatiser
  does with its default noun part of speech, the most common setup);
* a hyphenated or two-word term also matches its closed form
  ("back-testing" matches "backtesting");
* every term is counted independently, so "model" inside "challenger model"
  counts for both terms, as a literal reading of "the number of times a word
  from risk category i appears" implies.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TOKEN = re.compile(r"[a-z0-9]+")
_KEEP = {"bias", "is", "was", "has", "this", "its", "us", "analysis", "basis", "status", "process",
         "access", "loss", "less", "business", "across", "class", "gross", "stress", "crisis",
         "thus", "plus", "bonus", "corpus", "consensus", "focus", "virus", "series", "species",
         "lens", "means", "news", "gas", "ifrs", "gaas", "pas", "ias", "irb", "pds", "lgds"}


def lemma(token: str) -> str:
    """Plural noun to singular; leave everything else alone."""
    if token in _KEEP or len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("sses", "shes", "ches", "xes", "zes")):
        return token[:-2]
    if token.endswith("ses") and token[:-2] in _KEEP:  # biases -> bias
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def pdf_text(path: Path) -> str:
    import pymupdf

    with pymupdf.open(path) as doc:
        return "\n".join(page.get_text() for page in doc)


def html_text(path: Path) -> str:
    raw = path.read_text(errors="ignore")
    raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    return re.sub(r"<[^>]+>", " ", raw)


def tokens(text: str) -> list[str]:
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)  # join words hyphenated at a line break
    return TOKEN.findall(text.lower())


@dataclass(frozen=True)
class Term:
    category: str
    term: str
    variants: tuple[tuple[str, ...], ...]  # each variant is a sequence of lemmatised tokens


def compile_terms(dictionary: pd.DataFrame) -> list[Term]:
    """Turn Table 11 into token patterns; duplicates within a category are dropped."""
    out, seen = [], set()
    for cat, term in dictionary[["category", "term"]].itertuples(index=False):
        key = (cat, term.lower())
        if key in seen:
            continue
        seen.add(key)
        parts = tuple(lemma(t) for t in TOKEN.findall(term.lower()))
        variants = {parts}
        if len(parts) > 1:
            variants.add((lemma("".join(TOKEN.findall(term.lower()))),))
        out.append(Term(cat, term, tuple(sorted(variants))))
    return out


def find_mentions(lemmas: list[str], term: Term) -> list[int]:
    """Start positions of every match of any variant of ``term``."""
    hits = []
    for variant in term.variants:
        k = len(variant)
        first = variant[0]
        for i, tok in enumerate(lemmas):
            if tok == first and tuple(lemmas[i:i + k]) == variant:
                hits.append(i)
    return sorted(set(hits))


@dataclass
class DocumentCounts:
    doc_id: str
    n_tokens: int
    term_counts: dict[str, int]  # "category|term" -> count
    mentions: dict[str, list[int]]  # category -> start positions (for the sentiment windows)
    negatives: np.ndarray  # boolean per token: is it in the negative lexicon?


def count_document(doc_id: str, text: str, terms: list[Term], negative_words: set[str]) -> DocumentCounts:
    toks = tokens(text)
    lemmas = [lemma(t) for t in toks]
    term_counts, mentions = {}, {}
    for term in terms:
        pos = find_mentions(lemmas, term)
        term_counts[f"{term.category}|{term.term}"] = len(pos)
        mentions.setdefault(term.category, []).extend(pos)
    negatives = np.array([t in negative_words for t in toks], dtype=bool)
    return DocumentCounts(doc_id, len(toks), term_counts, mentions, negatives)


def category_share(counts: DocumentCounts, category: str, terms: list[Term] | None = None) -> float:
    """H_ijk = W_ijk / N_k: occurrences of the category's terms per word of the document."""
    w = sum(v for k, v in counts.term_counts.items()
            if k.split("|")[0] == category and (terms is None or k.split("|")[1] in terms))
    return w / counts.n_tokens


def negatives_around(counts: DocumentCounts, category: str, d: int) -> tuple[int, int]:
    """Negative words within ``d`` words on either side of each mention: (sum, number of mentions).

    The paper's T_ij divides the first by the second, pooled over the
    documents of a use (Appendix, p. 27).
    """
    neg = counts.negatives
    cum = np.concatenate([[0], np.cumsum(neg)])
    total = 0
    positions = counts.mentions.get(category, [])
    for p in positions:
        lo, hi = max(0, p - d), min(len(neg), p + 1 + d)
        total += int(cum[hi] - cum[lo]) - int(neg[p])
    return total, len(positions)


def load_negative_words(path: Path) -> set[str]:
    words = set()
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip().lower()
        if line and not line.startswith(";"):
            words.add(line)
    return words


def term_frequencies(all_counts: list[DocumentCounts]) -> pd.DataFrame:
    """One row per document and term: raw count and count per document word."""
    rows = []
    for c in all_counts:
        for key, v in c.term_counts.items():
            cat, term = key.split("|", 1)
            rows.append({"doc_id": c.doc_id, "category": cat, "term": term, "count": v,
                         "per_word": v / c.n_tokens})
    return pd.DataFrame(rows)


def top_contributors(freq: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Which terms make up each category's count: share of the category total."""
    tot = freq.groupby(["category", "term"])["count"].sum().reset_index()
    tot["share"] = tot["count"] / tot.groupby("category")["count"].transform("sum")
    return tot.sort_values(["category", "share"], ascending=[True, False]).groupby("category").head(k)


def word_counter(text: str) -> Counter:
    return Counter(tokens(text))
