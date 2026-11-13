"""Lab 8: precision@5 for BM25 / semantic / hybrid on the 10 lab-3 queries + 5 paraphrases.

Labels were fixed BEFORE running (same book -> doc_id map as README lab 3).
  python scripts/eval_lab8.py index.bin embeddings.npy > docs/data/eval8.md
"""

import sys
import time
from pathlib import Path

from findex.modes import Mode, ranked_hits
from findex.semantic import SemanticIndex
from findex.store import load

# (query, relevant doc_ids, kind)
QUERIES: list[tuple[str, set[int], str]] = [
    ("frankenstein OR monster", {74, 87}, "lab 3"),
    ("paradise OR satan OR eden", {11, 17, 48}, "lab 3"),
    ("internet OR email OR network", {30, 39, 65, 70, 46}, "lab 3"),
    ("census OR population", {20, 28, 5, 38}, "lab 3"),
    ("prime OR primes OR mersenne", {55, 59}, "lab 3"),
    ("lord OR god OR commandments", {0, 21, 8}, "lab 3"),
    ("guillotine OR paris OR revolution", {89, 50}, "lab 3"),
    ("slavery OR freedom OR slaves", {90, 14, 57}, "lab 3"),
    ("anne OR gilbert OR marilla", {35, 42}, "lab 3"),
    ("constitution OR union OR states", {9, 80}, "lab 3"),
    # paraphrases: describe the book without its distinctive words
    ("giant sea mammal hunted by men with spears on wooden boats", {6}, "paraphrase"),
    ("the fallen angel rebels against heaven", {11, 17, 48}, "paraphrase"),
    ("how to send electronic letters across computer networks", {65, 30, 39, 70, 46}, "paraphrase"),
    ("official count of every resident of the nation", {20, 28}, "paraphrase"),
    ("a young orphan girl growing up on a farm in canada", {35, 42}, "paraphrase"),
]

index = load(Path(sys.argv[1]))
sem = SemanticIndex.load(Path(sys.argv[2]))
MODES: list[Mode] = ["lexical", "semantic", "hybrid"]
sem.search("warm up the model")

sums = dict.fromkeys(MODES, 0.0)
sums_by_kind: dict[str, dict[str, float]] = {"lab 3": dict.fromkeys(MODES, 0.0), "paraphrase": dict.fromkeys(MODES, 0.0)}
counts = {"lab 3": 0, "paraphrase": 0}
print("| # | Query | Relevant | BM25 top 5 | P@5 | Semantic top 5 | P@5 | Hybrid top 5 | P@5 |")
print("|---|---|---|---|---|---|---|---|---|")
for n, (q, rel, kind) in enumerate(QUERIES, 1):
    cells = [str(n), f"`{q}`" + (" *(paraphrase)*" if kind == "paraphrase" else ""), ", ".join(map(str, sorted(rel)))]
    counts[kind] += 1
    for mode in MODES:
        hits, _ = ranked_hits(index, q, "bm25", mode, 5, 1, sem)
        ids = [h.doc_id for h in hits]
        p = sum(i in rel for i in ids) / 5
        sums[mode] += p
        sums_by_kind[kind][mode] += p
        cells += [", ".join(map(str, ids)), f"{p:.1f}"]
    print("| " + " | ".join(cells) + " |")
for label, data, n in [("**Mean, 10 lab-3 queries**", sums_by_kind["lab 3"], 10), ("**Mean, 5 paraphrases**", sums_by_kind["paraphrase"], 5),
                       ("**Mean, all 15**", sums, 15)]:
    print(f"| | {label} | | | **{data['lexical'] / n:.2f}** | | **{data['semantic'] / n:.2f}** | | **{data['hybrid'] / n:.2f}** |")

t0 = time.perf_counter()
for _ in range(20):
    sem.search("giant sea mammal hunted by men", 10)
print(f"\nsemantic query: {(time.perf_counter() - t0) / 20 * 1000:.1f} ms (embed query + {len(sem):,} x 384 matrix product + per-doc max)", file=sys.stderr)
