"""Parallel index construction: ``build_partial`` per chunk of files, then ``merge``.

Workers receive *paths* (and a pre-assigned doc_id range), never already-read
documents, so nothing big crosses the process boundary on the way in. Only the
finished partial index travels back (pickled) - that is the price of processes.

doc_ids are assigned up front: chunk *k* owns ``[start_k, start_k + len(chunk_k))``,
so partial indexes never collide and ``merge`` can simply concatenate postings.
Unreadable files become empty documents (keeps ids contiguous); other I/O errors
propagate and fail the whole build with the worker's stack.
"""

import logging
import multiprocessing
import time
from collections.abc import Callable, Sequence
from concurrent.futures import (
    Executor,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from pathlib import Path
from typing import Literal

from .corpus import Document, read_text
from .index import ArrayPostings, DocMeta, Index, Layout, Postings, build_index

log = logging.getLogger(__name__)

ExecutorName = Literal["serial", "threads", "processes"]


def list_corpus(root: Path) -> list[Path]:
    """Corpus files in the same (sorted) order ``iter_documents`` uses."""
    return sorted(root.rglob("*.txt"))


def plan_chunks(paths: Sequence[Path], n_chunks: int) -> list[tuple[int, list[Path]]]:
    """Split ``paths`` into <= ``n_chunks`` contiguous chunks of similar total size.

    Returns ``[(start_doc_id, chunk_paths), ...]``; ids are the global positions.
    """
    n_chunks = max(1, min(n_chunks, len(paths) or 1))
    sizes = [p.stat().st_size for p in paths]
    target = sum(sizes) / n_chunks
    chunks: list[tuple[int, list[Path]]] = []
    start, acc = 0, 0
    current: list[Path] = []
    for i, (p, s) in enumerate(zip(paths, sizes, strict=True)):
        current.append(p)
        acc += s
        remaining_chunks = n_chunks - len(chunks) - 1
        remaining_paths = len(paths) - i - 1
        if remaining_chunks > 0 and (acc >= target or remaining_paths == remaining_chunks):
            chunks.append((start, current))
            start, acc, current = i + 1, 0, []
    if current:
        chunks.append((start, current))
    return chunks


def build_partial(
    paths: Sequence[Path], start_id: int, positions: bool = False, layout: Layout = "array"
) -> Index:
    """Index ``paths`` as documents ``start_id, start_id + 1, ...`` (module-level: picklable)."""
    docs = (
        Document(start_id + i, p, read_text(p) or "")
        for i, p in enumerate(paths)
    )
    return build_index(docs, positions=positions, layout=layout)


def merge(parts: Sequence[Index]) -> Index:
    """Combine partial indexes with disjoint doc_id ranges into one index.

    The parts are consumed (their postings objects are reused, not copied).
    """
    parts = sorted((p for p in parts if p.doc_meta), key=lambda p: min(p.doc_meta))
    if not parts:
        return Index({}, {}, {}, "array", False)
    layout: Layout = parts[0].layout
    has_positions = parts[0].has_positions
    postings: dict[str, Postings] = {}
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}
    prev_max = -1
    for part in parts:
        if part.layout != layout or part.has_positions != has_positions:
            raise ValueError("cannot merge indexes with different layout/positions")
        if min(part.doc_meta) <= prev_max:
            raise ValueError("partial indexes have overlapping doc_id ranges")
        prev_max = max(part.doc_meta)
        doc_lengths.update(part.doc_lengths)
        doc_meta.update(part.doc_meta)
        for term, plist in part.postings.items():
            mine = postings.get(term)
            if mine is None:
                postings[term] = plist
            elif isinstance(mine, ArrayPostings) and isinstance(plist, ArrayPostings):
                mine.doc_ids.extend(plist.doc_ids)
                mine.tfs.extend(plist.tfs)
                if plist.positions is not None:
                    if mine.positions is None:
                        mine.positions = plist.positions
                    else:
                        mine.positions.extend(plist.positions)
            elif isinstance(mine, list) and isinstance(plist, list):
                mine.extend(plist)
    return Index(postings, doc_lengths, doc_meta, layout, has_positions)


Job = tuple[Sequence[Path], int, bool, Layout]


def _job(args: Job) -> tuple[Index, float]:
    """Worker entry point: build one partial index, report this process's CPU seconds."""
    paths, start, positions, layout = args
    cpu0 = time.process_time()
    index = build_partial(paths, start, positions, layout)
    return index, time.process_time() - cpu0


def run_jobs(
    fn: Callable[[Job], tuple[Index, float]],
    jobs: Sequence[Job],
    workers: int,
    executor: ExecutorName,
) -> list[tuple[Index, float]]:
    """Run ``fn`` over ``jobs`` with the chosen executor; the first failure aborts the build."""
    if executor == "serial":
        return [fn(j) for j in jobs]
    pool: Executor
    if executor == "threads":
        pool = ThreadPoolExecutor(max_workers=workers)
    else:
        # spawn: the only start method available on Windows/macOS defaults; workers re-import
        # this module, so everything they need must be importable at module level.
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
    results: list[tuple[Index, float] | None] = [None] * len(jobs)
    with pool:
        futures: dict[Future[tuple[Index, float]], int] = {pool.submit(fn, j): i for i, j in enumerate(jobs)}
        try:
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()  # re-raises the worker's exception (with its traceback)
        except BaseException:
            for f in futures:
                f.cancel()
            raise
    return [r for r in results if r is not None]


def build_parallel(
    root: Path,
    workers: int = 1,
    executor: ExecutorName = "serial",
    positions: bool = False,
    layout: Layout = "array",
    stats: dict[str, float] | None = None,
) -> Index:
    """Build the index of directory ``root`` with ``workers`` workers.

    If ``stats`` is given it is filled with ``merge_s`` (serial merge time) and
    ``worker_cpu_s`` (CPU seconds spent inside worker *processes*; threads share the
    parent's CPU clock, so it is 0 for the other executors).
    """
    paths = list_corpus(root)
    chunks = plan_chunks(paths, workers)
    jobs: list[Job] = [(chunk, start, positions, layout) for start, chunk in chunks]
    log.info("%d files in %d chunks, executor=%s workers=%d", len(paths), len(jobs), executor, workers)
    done = run_jobs(_job, jobs, workers, executor)
    parts = [idx for idx, _ in done]
    t0 = time.perf_counter()
    merged = merge(parts)
    if stats is not None:
        stats["merge_s"] = time.perf_counter() - t0
        stats["worker_cpu_s"] = sum(c for _, c in done) if executor == "processes" else 0.0
    return merged


