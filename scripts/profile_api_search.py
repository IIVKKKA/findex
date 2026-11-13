"""Profile the full search path (rank + 10 snippets from files), like `GET /search`.

  python scripts/profile_api_search.py index.bin
  py-spy record -f flamegraph -o docs/flame_search.svg -- python scripts/profile_api_search.py index.bin --no-profile
"""

import cProfile
import pstats
import sys
import time
from pathlib import Path

from findex.ranking import match_ids
from findex.store import load
from findex.web.app import run_search

QUERY = "whale OR harpoon OR ship"
N = 10
index = load(Path(sys.argv[1]))


def one() -> None:
    match_ids.cache_clear()
    run_search(index, QUERY, "bm25", 10, 1)


one()
if "--no-profile" in sys.argv:
    for _ in range(N * 3):
        one()
    raise SystemExit
t0 = time.perf_counter()
for _ in range(N):
    one()
print(f"{(time.perf_counter() - t0) / N * 1000:.1f} ms per request (rank + snippets)")
pr = cProfile.Profile()
pr.enable()
for _ in range(N):
    one()
pr.disable()
pstats.Stats(pr).sort_stats("tottime").print_stats(10)
