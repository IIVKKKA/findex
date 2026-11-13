"""Inverted index: term -> postings sorted by doc_id.

Three storage layouts are supported so the memory study (lab 2, M4) can
compare them on the same corpus:

* ``dataclass`` - ``list[PlainPosting]``, ordinary ``@dataclass`` (has ``__dict__``)
* ``slots``     - ``list[Posting]``, ``@dataclass(frozen=True, slots=True)``
* ``array``     - ``ArrayPostings``: two ``array('I')`` (doc_ids, tfs), no objects

Usage: python -m findex.index data/ --out index.bin [--positions] [--layout slots]
"""

from __future__ import annotations

import argparse
import time
import tracemalloc
from array import array
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from itertools import islice
from pathlib import Path
from typing import Literal

from .corpus import Document, iter_documents
from .tokenize import tokenize
from .util import timed

Layout = Literal["dataclass", "slots", "array"]


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass
class PlainPosting:
    """Same fields as Posting but a normal (dict-backed, mutable) dataclass."""

    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DocMeta:
    path: str
    title: str


@dataclass(slots=True)
class ArrayPostings:
    """Columnar postings: no per-posting Python objects."""

    doc_ids: array[int] = field(default_factory=lambda: array("I"))
    tfs: array[int] = field(default_factory=lambda: array("I"))
    # flat offsets, allocated only with --positions; posting i owns tfs[i] entries
    positions: array[int] | None = None


Postings = list[Posting | PlainPosting] | ArrayPostings


class Index(Mapping[str, list[Posting]]):
    """An inverted index that behaves like a read-only ``dict[term, postings]``.

    ``len(index)`` = number of terms, ``term in index``, ``index[term]`` ->
    ``list[Posting]`` (``KeyError`` if absent), ``for term in index``.
    """

    layout: Layout
    has_positions: bool

    def __init__(
        self,
        postings: dict[str, Postings],
        doc_lengths: dict[int, int],
        doc_meta: dict[int, DocMeta],
        layout: Layout = "slots",
        has_positions: bool = False,
    ) -> None:
        self.postings = postings
        self.doc_lengths = doc_lengths
        self.doc_meta = doc_meta
        self.layout = layout
        self.has_positions = has_positions

    # --- Mapping protocol (the dunder methods the lab asks for) ---------------
    def __len__(self) -> int:
        return len(self.postings)

    def __contains__(self, term: object) -> bool:
        # Mapping's default would call __getitem__ and materialise postings
        return term in self.postings

    def __iter__(self) -> Iterator[str]:
        return iter(self.postings)

    def __getitem__(self, term: str) -> list[Posting]:
        p = self.postings[term]  # KeyError for unknown terms
        if not isinstance(p, ArrayPostings):
            return [Posting(x.doc_id, x.tf, x.positions) for x in p]
        out: list[Posting] = []
        off = 0
        has = self.has_positions and p.positions is not None
        for i, d in enumerate(p.doc_ids):
            tf = p.tfs[i]
            pos = tuple(p.positions[off : off + tf]) if has else ()  # type: ignore[index]
            off += tf if has else 0
            out.append(Posting(d, tf, pos))
        return out

    def __repr__(self) -> str:
        return (
            f"Index(terms={len(self)}, docs={self.num_docs}, "
            f"layout={self.layout!r}, positions={self.has_positions})"
        )

    # Mapping defines __eq__ (comparing all items) which would make the index
    # unhashable. Identity semantics are what we want: it is a big singleton
    # object, and lru_cache needs to hash it.
    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)

    # --- corpus-level numbers ----------------------------------------------------
    @property
    def num_docs(self) -> int:
        return len(self.doc_meta)

    @cached_property
    def avg_doc_length(self) -> float:
        return sum(self.doc_lengths.values()) / max(1, len(self.doc_lengths))

    def doc_length(self, doc_id: int) -> int:
        return self.doc_lengths[doc_id]

    def close(self) -> None:
        """Release the big structures (used by ``open_index``)."""
        self.postings = {}
        self.doc_lengths = {}
        self.doc_meta = {}
        self.__dict__.pop("avg_doc_length", None)

    def doc_ids(self, term: str) -> Sequence[int]:
        """Sorted doc_ids containing ``term`` (empty if unknown)."""
        p = self.postings.get(term)
        if p is None:
            return ()
        if isinstance(p, ArrayPostings):
            return p.doc_ids
        return [x.doc_id for x in p]

    def df(self, term: str) -> int:
        p = self.postings.get(term)
        if p is None:
            return 0
        return len(p.doc_ids) if isinstance(p, ArrayPostings) else len(p)


def _title(text: str) -> str:
    """First line that looks like a title (has letters, not "[Illustration]" etc.)."""
    for line in text.splitlines():
        line = line.strip()
        if sum(ch.isalpha() for ch in line) < 3 or line.startswith(("[", "***")):
            continue
        return line[:80]
    return ""


@timed
def build_index(
    docs: Iterable[Document], *, positions: bool = False, layout: Layout = "slots"
) -> Index:
    """Build an index in a single streaming pass over ``docs``.

    doc_ids are increasing, so every postings list is sorted by construction.
    """
    obj_postings: defaultdict[str, list[Posting | PlainPosting]] = defaultdict(list)
    arr_postings: defaultdict[str, ArrayPostings] = defaultdict(ArrayPostings)
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}
    make = Posting if layout == "slots" else PlainPosting

    for doc in docs:
        doc_meta[doc.doc_id] = DocMeta(str(doc.path), _title(doc.text))
        n = 0
        items: list[tuple[str, int, list[int] | None]]
        if positions:
            pos: defaultdict[str, list[int]] = defaultdict(list)
            for i, tok in enumerate(tokenize(doc.text)):
                pos[tok].append(i)
            n = sum(len(v) for v in pos.values())
            items = [(t, len(v), v) for t, v in pos.items()]
        else:
            counts: Counter[str] = Counter()
            for tok in tokenize(doc.text):
                counts[tok] += 1
            n = counts.total()
            items = [(t, c, None) for t, c in counts.items()]
        doc_lengths[doc.doc_id] = n
        # flush this document's counter into the postings
        for term, tf, plist in items:
            if layout == "array":
                ap = arr_postings[term]
                ap.doc_ids.append(doc.doc_id)
                ap.tfs.append(tf)
                if plist is not None:
                    if ap.positions is None:
                        ap.positions = array("I")
                    ap.positions.extend(plist)
            else:
                obj_postings[term].append(
                    make(doc.doc_id, tf, tuple(plist) if plist is not None else ())
                )

    final: dict[str, Postings] = {}
    if layout == "array":
        final.update(arr_postings)
    else:
        final.update(obj_postings)
    return Index(final, doc_lengths, doc_meta, layout, positions)


def main(argv: list[str] | None = None) -> None:
    # `python -m findex.index` runs this file as __main__; use the real module
    # so pickled classes are importable as findex.index.* when loading later.
    from findex import index as real  # noqa: PLW0406
    from findex.store import save

    ap =argparse.ArgumentParser(prog="findex.index")
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--positions", action="store_true", help="store token offsets")
    ap.add_argument("--layout", choices=["dataclass", "slots", "array"], default="slots")
    ap.add_argument("--format", choices=["pickle", "binary"], default="pickle")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    docs = iter_documents(args.root)
    if args.limit is not None:
        docs = islice(docs, args.limit)

    tracemalloc.start()
    t0 = time.perf_counter()
    index = real.build_index(docs, positions=args.positions, layout=args.layout)
    build_s = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    t0 = time.perf_counter()
    save(index, args.out, fmt=args.format)
    save_s = time.perf_counter() - t0

    print(f"documents:  {index.num_docs}")
    print(f"terms:      {len(index.postings)}")
    print(f"layout:     {args.layout}  positions={args.positions}  format={args.format}")
    print(f"build:      {build_s:.2f} s")
    print(f"save:       {save_s:.2f} s")
    print(f"file size:  {args.out.stat().st_size / 2**20:.2f} MiB")
    print(f"peak mem:   {peak / 2**20:.1f} MiB (tracemalloc, build only)")


if __name__ == "__main__":
    main()
