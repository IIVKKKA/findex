"""Lab 5 benchmark: wall / CPU / peak RSS / merge time for executor x workers.

One cell = one fresh process (clean peak-RSS), repeated --reps times, median reported.

  python scripts/bench_lab5.py data/gutenberg --out bench5_gil.json
  python scripts/bench_lab5.py data/gutenberg --executors serial,threads --out bench5_ft.json   # 3.13t
Internal: python scripts/bench_lab5.py data/gutenberg --cell EXECUTOR WORKERS
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

from findex.parallel import ExecutorName, build_parallel

METRICS = ("wall_s", "cpu_s", "rss_mib", "merge_s")


def peak_rss_mib() -> float:
    try:
        import psutil

        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None) or info.rss  # peak working set on Windows
        return peak / 2**20
    except ImportError:
        return float("nan")


def run_cell(root: Path, executor: ExecutorName, workers: int) -> dict[str, Any]:
    stats: dict[str, float] = {}
    cpu0 = time.process_time()
    t0 = time.perf_counter()
    index = build_parallel(root, workers, executor, stats=stats)
    wall = time.perf_counter() - t0
    cpu = time.process_time() - cpu0 + stats["worker_cpu_s"]
    return {
        "executor": executor,
        "workers": workers,
        "wall_s": wall,
        "cpu_s": cpu,
        "rss_mib": peak_rss_mib(),
        "merge_s": stats["merge_s"],
        "terms": len(index),
        "gil": getattr(sys, "_is_gil_enabled", lambda: True)(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--cell", nargs=2, metavar=("EXECUTOR", "WORKERS"))
    ap.add_argument("--executors", default="serial,threads,processes")
    ap.add_argument("--workers", default="1,2,4,8,12")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("bench5.json"))
    args = ap.parse_args()

    if args.cell:
        ex = cast(ExecutorName, args.cell[0])
        print(json.dumps(run_cell(args.root, ex, int(args.cell[1]))))
        return

    cells: list[tuple[str, int]] = []
    for ex in args.executors.split(","):
        for w in [1] if ex == "serial" else [int(x) for x in args.workers.split(",")]:
            cells.append((ex, w))
    results: list[dict[str, Any]] = []
    for ex, w in cells:
        runs: list[dict[str, Any]] = []
        for rep in range(args.reps):
            out = subprocess.run(
                [sys.executable, __file__, str(args.root), "--cell", ex, str(w)],
                capture_output=True, text=True, check=True,
            ).stdout
            runs.append(json.loads(out.strip().splitlines()[-1]))
            print(f"{ex:9} w={w:<2} rep{rep + 1}: wall={runs[-1]['wall_s']:.2f}s", file=sys.stderr, flush=True)
        med: dict[str, Any] = {k: statistics.median(r[k] for r in runs) for k in METRICS}
        med.update(executor=ex, workers=w, gil=runs[0]["gil"], terms=runs[0]["terms"],
                   all_wall=[r["wall_s"] for r in runs])
        results.append(med)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
