"""Bytes per posting for each layout when lists are long (1,000,000 postings, 1,000 terms).

On the 91-document corpus most terms have a single posting, so per-term overhead hides
the per-posting cost. This isolates the per-posting cost.
"""

import gc
import tracemalloc
from array import array

from findex.index import ArrayPostings, PlainPosting, Posting

N_TERMS, PER_TERM = 1_000, 1_000


def measure(build):
    gc.collect()
    tracemalloc.start()
    obj = build()
    cur, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del obj
    return cur


def plain():
    return {t: [PlainPosting(d, 3) for d in range(PER_TERM)] for t in range(N_TERMS)}


def slots():
    return {t: [Posting(d, 3) for d in range(PER_TERM)] for t in range(N_TERMS)}


def arrays():
    return {
        t: ArrayPostings(array("I", range(PER_TERM)), array("I", [3] * PER_TERM))
        for t in range(N_TERMS)
    }


total = N_TERMS * PER_TERM
print("| Layout | Total | Bytes per posting |")
print("|---|---|---|")
for name, fn in [("plain dataclass", plain), ("slots=True", slots), ("array('I') pairs", arrays)]:
    b = measure(fn)
    print(f"| {name} | {b / 2**20:.1f} MiB | {b / total:.1f} |")
