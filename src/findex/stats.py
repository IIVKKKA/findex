"""Corpus statistics in a single streaming pass.

Usage: python -m findex.stats data/ [--limit N] [--eager]
"""

from __future__ import annotations

import argparse
import time
import tracemalloc
from collections import Counter
from collections.abc import Iterable
from itertools import islice
from pathlib import Path

from .corpus import Document, iter_documents
from .tokenize import tokenize


def lazy_stats(docs: Iterable[Document]) -> tuple[int, int, Counter[str]]:
    """One pass, O(vocabulary) memory."""
    n_docs = 0
    n_tokens = 0
    counts: Counter[str] = Counter()
    for doc in docs:
        n_docs += 1
        for tok in tokenize(doc.text):
            n_tokens += 1
            counts[tok] += 1
    return n_docs, n_tokens, counts


def eager_stats(docs: Iterable[Document]) -> tuple[int, int, Counter[str]]:
    """Deliberately greedy: everything materialized as lists first."""
    all_docs = list(docs)
    all_tokens = [list(tokenize(d.text)) for d in all_docs]
    counts: Counter[str] = Counter()
    for toks in all_tokens:
        counts.update(toks)
    return len(all_docs), sum(len(t) for t in all_tokens), counts


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="findex.stats")
    ap.add_argument("root", type=Path)
    ap.add_argument("--limit", type=int, default=None, help="first N documents only")
    ap.add_argument("--eager", action="store_true", help="use the greedy list version")
    args = ap.parse_args(argv)

    docs = iter_documents(args.root)
    if args.limit is not None:
        docs = islice(docs, args.limit)
    fn = eager_stats if args.eager else lazy_stats

    tracemalloc.start()
    t0 = time.perf_counter()
    n_docs, n_tokens, counts = fn(docs)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"mode:       {'eager' if args.eager else 'lazy'}")
    print(f"documents:  {n_docs}")
    print(f"tokens:     {n_tokens}")
    print(f"vocabulary: {len(counts)}")
    print("top 50 terms:")
    for term, c in counts.most_common(50):
        print(f"  {term:<20} {c}")
    print(f"elapsed:    {elapsed:.2f} s")
    print(f"peak mem:   {peak / 2**20:.1f} MiB (tracemalloc)")


if __name__ == "__main__":
    main()
