"""Lab 2 measurements. Usage: python scripts/bench_lab2.py data/gutenberg > bench_lab2.md"""

from __future__ import annotations

import random
import re
import subprocess
import sys
import tempfile
import timeit
from pathlib import Path

from findex.search import and_merge, and_set, or_merge, or_set
from findex.store import load

ROOT = sys.argv[1]
TMP = Path(tempfile.mkdtemp())


def run(*args: str) -> str:
    return subprocess.run(
        [sys.executable, "-m", *args], capture_output=True, text=True, check=True
    ).stdout


def field(out: str, name: str) -> float:
    return float(re.search(rf"^{name}:\s+([\d.]+)", out, re.MULTILINE).group(1))


def build(layout: str, fmt: str, positions: bool = False) -> dict[str, float]:
    out = TMP / f"{layout}_{fmt}_{positions}.bin"
    args = ["findex.index", ROOT, "--out", str(out), "--layout", layout, "--format", fmt]
    if positions:
        args.append("--positions")
    o = run(*args)
    s = run("findex.search", str(out), "whale")
    return {
        "peak": field(o, "peak mem"),
        "build": field(o, "build"),
        "size": field(o, "file size"),
        "save": field(o, "save"),
        "load": field(s, "load"),
        "path": out,  # type: ignore[dict-item]
    }


print("## A. Memory study (peak during build, tracemalloc)\n")
print("| Postings representation | Peak memory (build) | Index file (pickle) | Load time (pickle) |")
print("|---|---|---|---|")
res = {}
for layout, label in [
    ("dataclass", "`list[Posting]`, plain `@dataclass`"),
    ("slots", "`list[Posting]`, `slots=True`"),
    ("array", "`array('I')` pairs, no objects"),
]:
    r = res[layout] = build(layout, "pickle")
    print(f"| {label} | {r['peak']:.1f} MiB | {r['size']:.2f} MiB | {r['load']:.2f} s |")

print("\n## B. Persistence formats (array layout; slots layout for pickle)\n")
print("| Format | File size | Save time | Load time |")
print("|---|---|---|---|")
rows = [
    ("pickle (slots layout)", res["slots"]),
    ("pickle (array layout)", res["array"]),
    ("binary (array + JSON header)", build("array", "binary")),
]
for name, r in rows:
    print(f"| {name} | {r['size']:.2f} MiB | {r['save']:.2f} s | {r['load']:.2f} s |")

print("\n## C. Positions cost (slots layout, pickle)\n")
print("| Variant | Peak memory (build) | File size |")
print("|---|---|---|")
pos = build("slots", "pickle", positions=True)
print(f"| no positions | {res['slots']['peak']:.1f} MiB | {res['slots']['size']:.2f} MiB |")
print(f"| with positions | {pos['peak']:.1f} MiB | {pos['size']:.2f} MiB |")

print("\n## D. merge vs set (AND/OR of two terms, median of repeats)\n")
index = load(res["array"]["path"])  # type: ignore[arg-type]
by_df = sorted(index.postings, key=index.df)
common = sorted(index.postings, key=index.df, reverse=True)[:2]
rare = [t for t in by_df if index.df(t) == 1][:2]


def bench(fn, a, b, n=2000) -> float:
    return min(timeit.repeat(lambda: fn(a, b), number=n, repeat=5)) / n * 1e6


print("| Case | Lists (len) | Op | merge, us | set, us | Winner |")
print("|---|---|---|---|---|---|")
cases = [
    (f"common: {common[0]}, {common[1]}", common[0], common[1]),
    (f"rare: {rare[0]}, {rare[1]}", rare[0], rare[1]),
    (f"mixed: {common[0]}, {rare[0]}", common[0], rare[0]),
]
for label, t1, t2 in cases:
    a, b = list(index.doc_ids(t1)), list(index.doc_ids(t2))
    for op, fm, fs in [("AND", and_merge, and_set), ("OR", or_merge, or_set)]:
        m, s = bench(fm, a, b), bench(fs, a, b)
        print(f"| {label} | {len(a)}, {len(b)} | {op} | {m:.2f} | {s:.2f} | {'merge' if m < s else 'set'} |")

rng = random.Random(0)
A = sorted(rng.sample(range(2_000_000), 500_000))
B = sorted(rng.sample(range(2_000_000), 500_000))
C = sorted(rng.sample(range(2_000_000), 50))
for label, x, y in [("synthetic 500k x 500k", A, B), ("synthetic 500k x 50", A, C)]:
    for op, fm, fs in [("AND", and_merge, and_set), ("OR", or_merge, or_set)]:
        m, s = bench(fm, x, y, 1) / 1e3, bench(fs, x, y, 1) / 1e3
        print(f"| {label} | {len(x)}, {len(y)} | {op} | {m:.1f} ms | {s:.1f} ms | {'merge' if m < s else 'set'} |")
