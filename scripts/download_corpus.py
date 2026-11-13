"""Download a Project Gutenberg corpus into data/gutenberg/.

Usage: python scripts/download_corpus.py [--out data/gutenberg] [--target-mb 55]

Fetches plain-text books by id (sequentially, politely, with a pause).
Strips the Project Gutenberg license header/footer.
"""

from __future__ import annotations

import argparse
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
START = re.compile(r"\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.DOTALL)
END = re.compile(r"\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK")


def clean(text: str) -> str:
    m = START.search(text)
    if m:
        text = text[m.end():]
    m = END.search(text)
    if m:
        text = text[: m.start()]
    return text.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/gutenberg"))
    ap.add_argument("--target-mb", type=float, default=55)
    ap.add_argument("--max-id", type=int, default=3000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    total = sum(p.stat().st_size for p in args.out.glob("*.txt"))
    for book_id in range(1, args.max_id + 1):
        if total >= args.target_mb * 2**20:
            break
        dest = args.out / f"pg{book_id}.txt"
        if dest.exists():
            continue
        req = urllib.request.Request(
            URL.format(id=book_id),
            headers={"User-Agent": "findex-course-lab/0.1 (educational)"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
        except (urllib.error.URLError, TimeoutError):
            continue
        try:
            text = clean(raw.decode("utf-8-sig"))
        except UnicodeDecodeError:
            continue
        if len(text) < 20_000:
            continue
        dest.write_text(text, encoding="utf-8")
        total += dest.stat().st_size
        print(f"pg{book_id}: {len(text) // 1024} KiB (total {total / 2**20:.1f} MiB)", flush=True)
        time.sleep(0.5)


if __name__ == "__main__":
    main()
