"""Boolean search over a saved index.

Query syntax: ``term term`` is AND; ``OR`` and ``NOT`` are explicit
(upper-case) keywords. Evaluation is plain left-to-right (no precedence;
a real parser comes in lab 3). ``a NOT b`` means ``a AND NOT b``.

Usage: python -m findex.search index.bin "query" [--engine merge|set]
"""

from __future__ import annotations

import argparse
import logging
import time
import tracemalloc
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from .corpus import read_text
from .index import Index
from .store import load
from .tokenize import tokenize

Engine = Literal["merge", "set"]
Op = Literal["AND", "OR"]


# ---- two-pointer merges over sorted sequences --------------------------------
def and_merge(a: Sequence[int], b: Sequence[int]) -> list[int]:
    out: list[int] = []
    i = j = 0
    while i < len(a) and j < len(b):
        x, y = a[i], b[j]
        if x == y:
            out.append(x)
            i += 1
            j += 1
        elif x < y:
            i += 1
        else:
            j += 1
    return out


def or_merge(a: Sequence[int], b: Sequence[int]) -> list[int]:
    out: list[int] = []
    i = j = 0
    while i < len(a) and j < len(b):
        x, y = a[i], b[j]
        if x == y:
            out.append(x)
            i += 1
            j += 1
        elif x < y:
            out.append(x)
            i += 1
        else:
            out.append(y)
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


def not_merge(a: Sequence[int], b: Sequence[int]) -> list[int]:
    """a minus b."""
    out: list[int] = []
    i = j = 0
    while i < len(a):
        if j >= len(b) or a[i] < b[j]:
            out.append(a[i])
            i += 1
        elif a[i] == b[j]:
            i += 1
            j += 1
        else:
            j += 1
    return out


# ---- set alternative -----------------------------------------------------------
def and_set(a: Sequence[int], b: Sequence[int]) -> list[int]:
    return sorted(set(a) & set(b))


def or_set(a: Sequence[int], b: Sequence[int]) -> list[int]:
    return sorted(set(a) | set(b))


def not_set(a: Sequence[int], b: Sequence[int]) -> list[int]:
    return sorted(set(a) - set(b))


_ENGINES = {
    "merge": (and_merge, or_merge, not_merge),
    "set": (and_set, or_set, not_set),
}


class QueryError(ValueError):
    pass


def search(index: Index, query: str, engine: Engine = "merge") -> list[int]:
    """Evaluate ``query`` left to right and return sorted doc_ids."""
    f_and, f_or, f_not = _ENGINES[engine]
    universe = list(range(index.num_docs))
    # first pass: raw whitespace words -> (op, negated, term) operands
    result: Sequence[int] | None = None
    op: Op = "AND"
    negate = False
    words = query.split()
    if words and words[-1] in ("AND", "OR", "NOT"):
        raise QueryError("query ends with an operator")
    for raw in words:
        if raw == "AND":
            op = "AND"
            continue
        if raw == "OR":
            op = "OR"
            continue
        if raw == "NOT":
            negate = True
            continue
        for term in tokenize(raw):
            ids = index.doc_ids(term)
            if result is None:
                result = f_not(universe, ids) if negate else list(ids)
            elif negate:
                result = f_not(result, ids) if op == "AND" else f_or(result, f_not(universe, ids))
            else:
                result = (f_and if op == "AND" else f_or)(result, ids)
            negate = False
            op = "AND"  # implicit AND between adjacent terms
    if result is None:
        raise QueryError("empty query")
    return list(result)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="findex.search")
    ap.add_argument("index", type=Path)
    ap.add_argument("query")
    ap.add_argument("--engine", choices=["merge", "set"], default="merge")
    ap.add_argument("--top", type=int, default=10, help="how many hits to print")
    ap.add_argument(
        "--scorer",
        choices=["none", "tfidf", "bm25"],
        default="none",
        help="none = lab 2 boolean (doc_id order); tfidf/bm25 = ranked, with the full "
        "query language (parentheses, quoted phrases) and snippets",
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="show @timed / cache logs")
    args = ap.parse_args(argv)
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    tracemalloc.start()
    t0 = time.perf_counter()
    index = load(args.index)
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    if args.scorer == "none":
        hits = search(index, args.query, args.engine)
        search_s = time.perf_counter() - t0
        print(f"{len(hits)} hits for {args.query!r} (engine={args.engine})")
        for doc_id in hits[: args.top]:
            m = index.doc_meta[doc_id]
            print(f"  [{doc_id}] {m.title}  ({Path(m.path).name})")
    else:
        from .query import QueryError, parse
        from .ranking import search as ranked_search
        from .scoring import BM25, TfIdf
        from .snippets import make_snippet

        try:
            results = ranked_search(
                index, args.query, TfIdf() if args.scorer == "tfidf" else BM25(), k=args.top
            )
        except QueryError as exc:
            raise SystemExit(f"query error: {exc}") from exc
        search_s = time.perf_counter() - t0
        terms = parse(args.query).terms()
        print(f"top {len(results)} for {args.query!r} (scorer={args.scorer})")
        for r in results:
            m = index.doc_meta[r.doc_id]
            text = read_text(Path(m.path)) or ""
            print(f"  [{r.doc_id}] {r.score:.3f}  {m.title}  ({Path(m.path).name})")
            print(f"      {make_snippet(text, terms)}")
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"load:       {load_s:.3f} s")
    print(f"search:     {search_s * 1000:.3f} ms")
    print(f"peak mem:   {peak / 2**20:.1f} MiB (tracemalloc, load+search)")


if __name__ == "__main__":
    main()
