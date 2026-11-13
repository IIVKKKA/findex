"""Lab 6 benchmark: same start URL and max_pages, concurrency 1 / 5 / 20 (+ a polite per-host row).

Each cell runs in a fresh process against a fresh mirror (clean RSS, no leftover server state).
  python scripts/bench_lab6.py --out docs/data/bench6.json
Internal: python scripts/bench_lab6.py --cell PORT CONCURRENCY PER_HOST DELAY
"""

import argparse
import asyncio
import json
import statistics
import subprocess
import sys
import time
from typing import Any

import httpx

from findex.crawler import CrawlStats, crawl

MAX_PAGES = 200


def peak_rss_mib() -> float:
    import psutil

    info = psutil.Process().memory_info()
    return (getattr(info, "peak_wset", None) or info.rss) / 2**20


async def run_cell(port: int, concurrency: int, per_host: int, delay: float) -> dict[str, Any]:
    stats = CrawlStats()
    t0 = time.perf_counter()
    limits = httpx.Limits(max_connections=concurrency + 2)
    async with httpx.AsyncClient(follow_redirects=True, limits=limits) as client:
        async for _ in crawl([f"http://127.0.0.1:{port}/page/0"], client, max_pages=MAX_PAGES,
                             concurrency=concurrency, per_host=per_host, min_delay=delay, timeout=5.0, stats=stats):
            pass
    wall = time.perf_counter() - t0
    return {"concurrency": concurrency, "per_host": per_host, "delay": delay, "wall_s": wall,
            "pages": stats.pages, "pages_per_s": stats.pages / wall, "errors": stats.errors,
            "rss_mib": peak_rss_mib()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", nargs=4, metavar=("PORT", "C", "PER_HOST", "DELAY"))
    ap.add_argument("--out", default="docs/data/bench6.json")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--latency", type=float, default=0.05)
    args = ap.parse_args()
    if args.cell:
        port, c, ph, d = int(args.cell[0]), int(args.cell[1]), int(args.cell[2]), float(args.cell[3])
        print(json.dumps(asyncio.run(run_cell(port, c, ph, d))))
        return

    cells = [(1, 1, 0.0), (5, 5, 0.0), (20, 20, 0.0), (20, 2, 0.0), (20, 20, 0.05)]
    results: list[dict[str, Any]] = []
    port = 8800
    for c, ph, d in cells:
        runs: list[dict[str, Any]] = []
        for rep in range(args.reps):
            port += 1
            srv = subprocess.Popen([sys.executable, "-m", "findex.mirror", "--port", str(port), "--pages", "400",
                                    "--latency", str(args.latency), "--source", "data/gutenberg", "--no-faults"],
                                   stdout=subprocess.PIPE, text=True)
            assert srv.stdout is not None
            srv.stdout.readline()  # "mirror on ..." = listening
            try:
                out = subprocess.run([sys.executable, __file__, "--cell", str(port), str(c), str(ph), str(d)],
                                     capture_output=True, text=True, check=True).stdout
            finally:
                srv.terminate()
            runs.append(json.loads(out.strip().splitlines()[-1]))
            print(f"c={c} per_host={ph} delay={d} rep{rep + 1}: {runs[-1]['wall_s']:.2f}s", file=sys.stderr, flush=True)
        med: dict[str, Any] = {k: statistics.median(r[k] for r in runs)
                               for k in ("wall_s", "pages_per_s", "errors", "rss_mib", "pages")}
        med.update(concurrency=c, per_host=ph, delay=d, all_wall=[r["wall_s"] for r in runs])
        results.append(med)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print("| Concurrency | Per-host limit | Min delay | Pages | Wall, s | Pages/s | Errors | Peak RSS, MiB |")
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['concurrency']} | {r['per_host']} | {r['delay']} s | {int(r['pages'])} | {r['wall_s']:.2f} | "
              f"{r['pages_per_s']:.1f} | {int(r['errors'])} | {r['rss_mib']:.0f} |")


if __name__ == "__main__":
    main()
