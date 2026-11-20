"""Download the public data this project needs (none of it is committed).

1. Give Me Some Credit (Kaggle competition, 2011): ``cs-training.csv``, the
   file the paper's empirical example uses. Kaggle's licence forbids
   redistribution, so the file is fetched with the user's own API token
   (``~/.kaggle/kaggle.json``); the account must have accepted the
   competition rules on kaggle.com.
2. Hu & Liu (2004) opinion lexicon, the negative-word list of the paper's
   sentiment robustness check, from the NLTK data repository.
3. Two further public credit datasets for the multi-dataset benchmark
   (Taiwan credit-card default and German credit), from OpenML.

Every file is checked against a SHA-256 written to ``data/checksums.json`` on
first download, so a later run notices if a source changes.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

import certifi

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CHECKSUMS = ROOT / "data" / "checksums.json"

# python.org builds of Python on macOS ship without a CA bundle.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
CONTEXT = ssl.create_default_context(cafile=certifi.where())

KAGGLE_URL = "https://www.kaggle.com/api/v1/competitions/data/download/GiveMeSomeCredit/cs-training.csv"
LEXICON_URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/opinion_lexicon.zip"
OPENML_IDS = {"taiwan_default": 42477, "german_credit": 31}


def _get(url: str, headers: dict | None = None) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "research-replication", **(headers or {})})
    with urllib.request.urlopen(request, context=CONTEXT, timeout=120) as response:
        return response.read()


def _record(path: Path) -> None:
    """Store the file's SHA-256 the first time; fail if it changes later."""
    sums = json.loads(CHECKSUMS.read_text()) if CHECKSUMS.exists() else {}
    key = str(path.relative_to(ROOT))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if key in sums and sums[key] != digest:
        sys.exit(f"{key}: checksum changed ({sums[key]} -> {digest})")
    sums[key] = digest
    CHECKSUMS.write_text(json.dumps(sums, indent=2, sort_keys=True) + "\n")


def download_gmsc() -> Path:
    """Fetch cs-training.csv with the Kaggle token (basic auth, as the Kaggle CLI does)."""
    out = RAW / "gmsc" / "cs-training.csv"
    if out.exists():
        return out
    token = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())
    auth = base64.b64encode(f"{token['username']}:{token['key']}".encode()).decode()
    payload = _get(KAGGLE_URL, {"Authorization": f"Basic {auth}"})
    if payload[:2] == b"PK":  # Kaggle sometimes serves the file zipped
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            payload = archive.read(archive.namelist()[0])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(payload)
    return out


def download_lexicon() -> Path:
    out = RAW / "lexicon" / "negative-words.txt"
    if out.exists():
        return out
    with zipfile.ZipFile(io.BytesIO(_get(LEXICON_URL))) as archive:
        name = next(n for n in archive.namelist() if n.endswith("negative-words.txt"))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(archive.read(name))
    return out


def download_openml() -> list[Path]:
    """Cache the two benchmark datasets as CSV through scikit-learn's OpenML client."""
    from sklearn.datasets import fetch_openml

    paths = []
    for name, data_id in OPENML_IDS.items():
        out = RAW / "openml" / f"{name}.csv"
        if not out.exists():
            # The pandas ARFF parser breaks on integer category labels under pandas 3.
            frame = fetch_openml(data_id=data_id, as_frame=True, parser="liac-arff",
                                 data_home=str(RAW / "openml" / "cache")).frame
            out.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(out, index=False)
        paths.append(out)
    return paths


def main() -> None:
    paths = [download_lexicon(), *download_openml()]
    try:
        paths.insert(0, download_gmsc())
    except urllib.error.HTTPError as error:
        print(f"Kaggle refused ({error.code}): {error.read()[:200]!r}\n"
              "Accept the rules at https://www.kaggle.com/competitions/GiveMeSomeCredit/rules and rerun.")
    for path in paths:
        _record(path)
        print(f"{path.relative_to(ROOT)}  {path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
