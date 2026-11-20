"""Download the regulatory documents listed in data/corpus_manifest.csv and check their hashes.

The manifest records, for each document of the paper's Table 12, the URL of
the official copy we used, its SHA-256 and notes on the version. A few hosts
refuse scripted downloads (the script says which); fetch those by hand into
data/raw/corpus/<doc_id>.pdf and rerun to verify.
"""
from __future__ import annotations

import hashlib
import os
import ssl
import urllib.request
from pathlib import Path

import certifi
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "corpus_manifest.csv"
CONTEXT = ssl.create_default_context(cafile=certifi.where())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url: str, out: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research replication)"})
    with urllib.request.urlopen(request, context=CONTEXT, timeout=120) as response:
        out.write_bytes(response.read())


def main() -> None:
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    manifest = pd.read_csv(MANIFEST)
    problems = []
    for row in manifest[manifest["status"].isin(["found", "substitute"])].drop_duplicates("file").itertuples():
        out = ROOT / row.file
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            try:
                fetch(row.url, out)
            except Exception as error:  # report and continue: one blocked host should not stop the rest
                problems.append(f"{row.doc_id}: download failed ({error}); fetch {row.url} by hand")
                continue
        if sha256(out) != row.sha256:
            problems.append(f"{row.doc_id}: SHA-256 differs from the manifest (another version of the file?)")
    print("\n".join(problems) or "all documents present and verified")


if __name__ == "__main__":
    main()
