"""Lab 8: lab-3 pure-Python scorer vs NumPy, on the real corpus and on a 100x larger synthetic index.

  python scripts/bench_lab8.py index.bin > docs/data/bench8.md
"""

import random
import sys
import timeit
from array import array
from pathlib import Path

from findex.index import ArrayPostings, DocMeta, Index
from findex.npsearch import NumpyIndex
from findex.ranking import match_ids, search
from findex.scoring import BM25
from findex.store import load


def ms(fn, number: int) -> float:
    return min(timeit.repeat(fn, number=number, repeat=5)) / number * 1000


def row(label: str, index: Index, ni: NumpyIndex, query: str, number: int) -> str:
    def py() -> None:
        match_ids.cache_clear()
        search(index, query, BM25(), k=10)

    def npy() -> None:
        ni.search(query, "bm25", 10)

    a, b = ms(py, number), ms(npy, number)
    return f"| {label} | `{query}` | {a:.3f} | {b:.3f} | {a / b:.1f}x |"


def synthetic(n_docs: int = 100_000) -> Index:
    rng = random.Random(1)
    lengths = {d: rng.randint(100, 5000) for d in range(n_docs)}
    meta = {d: DocMeta(f"d{d}", f"doc {d}") for d in range(n_docs)}
    postings: dict[str, ArrayPostings] = {}
    for term, step in (("common", 2), ("mid", 20), ("rare", 5000)):
        ids = array("I", range(0, n_docs, step))
        tfs = array("I", (rng.randint(1, 6) for _ in ids))
        postings[term] = ArrayPostings(ids, tfs)
    return Index(dict(postings), lengths, meta, "array", False)  # type: ignore[arg-type]


print("| Corpus | Query | Lab 3 Python scorer, ms | NumPy, ms | Speed-up |")
print("|---|---|---|---|---|")
real = load(Path(sys.argv[1]))
nreal = NumpyIndex(real)
for label, q in [("real, 91 docs", "the"), ("real, 91 docs", "mersenne"), ("real, 91 docs", "whale OR harpoon OR ship")]:
    print(row(label, real, nreal, q, 200))
big = synthetic()
nbig = NumpyIndex(big)
for label, q in [("synthetic, 100 000 docs", "common"), ("synthetic, 100 000 docs", "rare"),
                 ("synthetic, 100 000 docs", "common OR mid OR rare")]:
    print(row(label, big, nbig, q, 3))
