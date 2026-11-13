from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex.cli import app
from findex.corpus import iter_documents
from findex.index import Index, build_index
from findex.parallel import (
    ExecutorName,
    Job,
    _job,  # pyright: ignore[reportPrivateUsage]  # the worker entry point is tested directly
    build_parallel,
    build_partial,
    list_corpus,
    merge,
    plan_chunks,
    run_jobs,
)
from findex.store import load, postings_as_objects


def same_index(a: Index, b: Index) -> None:
    assert a.doc_lengths == b.doc_lengths
    assert a.doc_meta == b.doc_meta
    assert a.has_positions == b.has_positions
    assert set(a) == set(b)
    for term in a:
        assert postings_as_objects(a, term) == postings_as_objects(b, term), term


def test_plan_chunks_cover_all_paths_in_order(corpus_dir: Path) -> None:
    paths = list_corpus(corpus_dir)
    for n in (1, 2, 3, 4, 10):
        chunks = plan_chunks(paths, n)
        assert [p for _, c in chunks for p in c] == paths
        assert [s for s, _ in chunks] == [sum(len(c) for _, c in chunks[:i]) for i in range(len(chunks))]
        assert len(chunks) <= min(n, len(paths))


@pytest.mark.parametrize("positions", [False, True])
def test_serial_merge_of_one_partial_equals_lab4_index(corpus_dir: Path, positions: bool) -> None:
    expected = build_index(iter_documents(corpus_dir), positions=positions, layout="array")
    got = merge([build_partial(list_corpus(corpus_dir), 0, positions)])
    same_index(got, expected)


@pytest.mark.parametrize("workers", [2, 3, 4])
@pytest.mark.parametrize("positions", [False, True])
def test_chunked_serial_equals_single(corpus_dir: Path, workers: int, positions: bool) -> None:
    expected = build_index(iter_documents(corpus_dir), positions=positions, layout="array")
    same_index(build_parallel(corpus_dir, workers, "serial", positions), expected)


@pytest.mark.parametrize("layout", ["slots", "dataclass"])
def test_merge_works_for_object_layouts(corpus_dir: Path, layout: str) -> None:
    from typing import cast

    from findex.index import Layout

    lay = cast(Layout, layout)
    expected = build_index(iter_documents(corpus_dir), layout=lay)
    got = build_parallel(corpus_dir, 3, "serial", False, lay)
    same_index(got, expected)


def test_threads_equal_serial(corpus_dir: Path) -> None:
    same_index(
        build_parallel(corpus_dir, 3, "threads"),
        build_parallel(corpus_dir, 1, "serial"),
    )


@pytest.mark.slow
def test_processes_equal_serial(corpus_dir: Path) -> None:
    stats: dict[str, float] = {}
    got = build_parallel(corpus_dir, 2, "processes", stats=stats)
    same_index(got, build_parallel(corpus_dir, 1, "serial"))
    assert stats["merge_s"] >= 0 and stats["worker_cpu_s"] >= 0


def test_merge_rejects_overlapping_ranges(corpus_dir: Path) -> None:
    paths = list_corpus(corpus_dir)
    a = build_partial(paths[:2], 0)
    b = build_partial(paths[1:3], 1)  # ids 1,2 overlap with a's 0,1
    with pytest.raises(ValueError, match="overlapping"):
        merge([a, b])


def test_merge_empty() -> None:
    assert len(merge([])) == 0


@pytest.mark.parametrize("executor", ["serial", "threads"])
def test_worker_failure_propagates(executor: ExecutorName, tmp_path: Path) -> None:
    bad: Job = ([tmp_path / "missing.txt"], 0, False, "array")
    with pytest.raises(FileNotFoundError):
        run_jobs(_job, [bad], 2, executor)


@pytest.mark.slow
def test_worker_failure_in_process_shows_its_stack(tmp_path: Path) -> None:
    bad: Job = ([tmp_path / "missing.txt"], 0, False, "array")
    with pytest.raises(FileNotFoundError) as info:
        run_jobs(_job, [bad], 2, "processes")
    # concurrent.futures chains the remote traceback text as the cause
    assert "build_partial" in str(info.value.__cause__)


def test_cli_parallel_matches_serial(corpus_dir: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    out_a, out_b = tmp_path / "a.bin", tmp_path / "b.bin"
    assert runner.invoke(app, ["index", str(corpus_dir), "-o", str(out_a), "--executor", "serial"]).exit_code == 0
    r = runner.invoke(app, ["index", str(corpus_dir), "-o", str(out_b), "--executor", "threads", "--workers", "3"])
    assert r.exit_code == 0, r.output
    same_index(load(out_a), load(out_b))


def test_cli_limit_falls_back_to_serial(corpus_dir: Path, tmp_path: Path) -> None:
    r = CliRunner().invoke(app, ["index", str(corpus_dir), "-o", str(tmp_path / "x.bin"), "--limit", "2"])
    assert r.exit_code == 0, r.output
    assert "indexed 2 documents" in r.stdout
