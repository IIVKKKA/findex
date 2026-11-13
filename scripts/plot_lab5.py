"""Plot speedup vs workers and print the markdown table for the README.

python scripts/plot_lab5.py bench5_gil.json bench5_ft.json docs/speedup.png
"""

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

Row = dict[str, Any]
gil: list[Row] = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
ft: list[Row] = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
out = Path(sys.argv[3])

base = next(r for r in gil if r["executor"] == "serial")["wall_s"]


def series(rows: list[Row], ex: str) -> tuple[list[int], list[float]]:
    pts = sorted((r["workers"], base / r["wall_s"]) for r in rows if r["executor"] == ex)
    return [p[0] for p in pts], [p[1] for p in pts]


fig, ax = plt.subplots(figsize=(7, 4.5))
xs_max = max(r["workers"] for r in gil)
ax.plot([1, xs_max], [1, xs_max], "k--", label="ideal (linear)")
for label, rows, ex, style in [
    ("threads, 3.13 (GIL)", gil, "threads", "o-"),
    ("processes, 3.13 (GIL)", gil, "processes", "s-"),
    ("threads, 3.13t (no GIL)", ft, "threads", "^-"),
]:
    x, y = series(rows, ex)
    ax.plot(x, y, style, label=label)
ax.set_xlabel("workers")
ax.set_ylabel("speedup vs serial (wall time)")
ax.set_title("findex index: 91 books, 58 MB")
ax.set_xticks([1, 2, 4, 8, 12])
ax.grid(alpha=0.3)
ax.legend()
out.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(out, dpi=130)

print("| Interpreter | Executor | Workers | Wall, s | CPU, s | Peak RSS (parent), MiB | Merge, s | Speedup |")
print("|---|---|---|---|---|---|---|---|")
for name, rows in (("3.13 (GIL)", gil), ("3.13t (no GIL)", ft)):
    for r in sorted(rows, key=lambda r: (r["executor"] != "serial", r["executor"], r["workers"])):
        if name != "3.13 (GIL)" and r["executor"] == "serial":
            pass
        print(
            f"| {name} | {r['executor']} | {r['workers']} | {r['wall_s']:.2f} | {r['cpu_s']:.2f} | "
            f"{r['rss_mib']:.0f} | {r['merge_s']:.2f} | {base / r['wall_s']:.2f}x |"
        )
