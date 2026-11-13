"""Lab 7 measurements.

 1. 20 concurrent /search requests: ``def`` vs ``async def`` route, and how long /health takes meanwhile.
 2. oha load tests: 1 worker async def, 1 worker def, 4 workers (def), (+ a deployed URL if given).

  python scripts/bench_lab7.py --index index.bin --oha <path-to-oha.exe> [--url https://your.app] > bench7.md
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from typing import Any

import httpx

QUERY = "whale OR harpoon OR ship"


def start(port: int, workers: int, app: str, index: str) -> subprocess.Popen[bytes]:
    env = {**os.environ, "INDEX_PATH": index, "LOG_LEVEL": "WARNING"}
    cmd = [sys.executable, "-m", "uvicorn", app, "--port", str(port), "--workers", str(workers), "--log-level", "warning"]
    if app.startswith("def_vs_async_app"):
        cmd += ["--app-dir", "scripts"]
    else:
        cmd += ["--factory"]
    proc = subprocess.Popen(cmd, env=env)
    for _ in range(300):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                return proc
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    proc.kill()
    raise RuntimeError("server did not start")


async def concurrent_20(port: int, route: str) -> dict[str, float]:
    base = f"http://127.0.0.1:{port}"
    async with httpx.AsyncClient(timeout=60) as c:
        await c.get(f"{base}{route}", params={"q": QUERY})  # warm-up (fills the lru_cache path)
        health: list[float] = []
        stop = False

        async def probe() -> None:
            while not stop:
                t = time.perf_counter()
                await c.get(f"{base}/health")
                health.append(time.perf_counter() - t)
                await asyncio.sleep(0.01)

        probe_task = asyncio.create_task(probe())
        await asyncio.sleep(0.05)
        t0 = time.perf_counter()
        lat: list[float] = []

        async def one(i: int) -> None:
            t = time.perf_counter()
            await c.get(f"{base}{route}", params={"q": f"{QUERY} OR w{i}"})  # distinct query: no cache hit
            lat.append(time.perf_counter() - t)

        await asyncio.gather(*(one(i) for i in range(20)))
        wall = time.perf_counter() - t0
        stop = True
        await probe_task
    return {"wall_s": wall, "mean_latency_s": sum(lat) / len(lat), "health_max_s": max(health),
            "health_p50_s": sorted(health)[len(health) // 2]}


def oha(oha_path: str, url: str, seconds: int = 15, conns: int = 20) -> dict[str, Any]:
    out = subprocess.run([oha_path, "-z", f"{seconds}s", "-c", str(conns), "--no-tui", "--output-format", "json", url],
                         capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    p = d["latencyPercentiles"]
    return {"rps": d["summary"]["requestsPerSec"], "p50": p["p50"] * 1000, "p95": p["p95"] * 1000,
            "p99": p["p99"] * 1000, "ok": d["statusCodeDistribution"].get("200", 0),
            "total": sum(d["statusCodeDistribution"].values())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="index.bin")
    ap.add_argument("--oha", required=True)
    ap.add_argument("--url", help="deployed base URL for the 4th row")
    ap.add_argument("--seconds", type=int, default=15)
    args = ap.parse_args()

    print("## 20 concurrent requests: `def` vs `async def` (1 worker)\n")
    print("| Route | Wall for 20 requests | Mean latency | /health p50 meanwhile | /health worst meanwhile |")
    print("|---|---|---|---|---|")
    proc = start(8101, 1, "def_vs_async_app:app", args.index)
    try:
        for route in ("/s_def", "/s_async"):
            r = asyncio.run(concurrent_20(8101, route))
            print(f"| `{route}` | {r['wall_s'] * 1000:.0f} ms | {r['mean_latency_s'] * 1000:.0f} ms | "
                  f"{r['health_p50_s'] * 1000:.0f} ms | {r['health_max_s'] * 1000:.0f} ms |", flush=True)
    finally:
        proc.terminate()
        proc.wait()

    rows: list[tuple[str, dict[str, Any]]] = []
    for label, app, workers, route in [
        ("1 worker, `async def`", "def_vs_async_app:app", 1, "/s_async"),
        ("1 worker, `def`", "def_vs_async_app:app", 1, "/s_def"),
        ("4 workers, `def`", "findex.web.app:create_app", 4, "/search"),
    ]:
        port = 8102 + len(rows)
        proc = start(port, workers, app, args.index)
        try:
            url = f"http://127.0.0.1:{port}{route}?q=whale+OR+harpoon+OR+ship&k=10"
            time.sleep(8 * workers)  # /health answers as soon as ONE worker is up; let all of them finish loading
            oha(args.oha, url, 3, 20)  # warm-up run, not recorded
            rows.append((label, oha(args.oha, url, args.seconds)))
        finally:
            proc.terminate()
            proc.wait()
    if args.url:
        rows.append(("deployed service", oha(args.oha, f"{args.url.rstrip('/')}/search?q=whale+OR+harpoon+OR+ship&k=10", args.seconds, 10)))

    print("\n## oha: 20 connections, 15 s, `GET /search?q=whale OR harpoon OR ship&k=10`\n")
    print("| Setup | Requests/s | p50, ms | p95, ms | p99, ms | 200 OK / total |")
    print("|---|---|---|---|---|---|")
    for label, r in rows:
        print(f"| {label} | {r['rps']:.0f} | {r['p50']:.0f} | {r['p95']:.0f} | {r['p99']:.0f} | {r['ok']}/{r['total']} |")


if __name__ == "__main__":
    main()
