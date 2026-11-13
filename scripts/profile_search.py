"""Profile `findex search` (3-term query, full corpus). Usage:
  python scripts/profile_search.py index.bin            # cProfile summary
  py-spy record -f flamegraph -o docs/flame_search.svg -- python scripts/profile_search.py index.bin --no-profile
"""

import cProfile
import pstats
import sys
import time
from pathlib import Path

from findex.ranking import match_ids, search
from findex.scoring import BM25
from findex.store import load

QUERY = "whale OR harpoon OR ship"
N = 15

index = load(Path(sys.argv[1]))
snip = "--snippets" in sys.argv


def one() -> None:
    match_ids.cache_clear()
    search(index, QUERY, BM25(), k=10)


one()  # warm up (file cache)
if "--no-profile" in sys.argv:
    for _ in range(N * 4):
        one()
    raise SystemExit
t0 = time.perf_counter()
for _ in range(N):
    one()
print(f"{(time.perf_counter() - t0) / N * 1000:.1f} ms per search (no profiler)")
pr = cProfile.Profile()
pr.enable()
for _ in range(N):
    one()
pr.disable()
st = pstats.Stats(pr)
st.sort_stats("tottime").print_stats(12)
