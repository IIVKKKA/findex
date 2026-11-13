import random
from pathlib import Path

import pytest

from findex.corpus import iter_documents
from findex.index import DocMeta, Index, Layout, Posting, build_index
from findex.search import (
    QueryError,
    and_merge,
    and_set,
    not_merge,
    not_set,
    or_merge,
    or_set,
    search,
)
from findex.store import load, postings_as_objects, save


def test_posting_and_meta_hashable_and_frozen() -> None:
    s = {Posting(1, 2), Posting(1, 2), DocMeta("p", "t")}
    assert len(s) == 2
    p = Posting(1, 2)
    with pytest.raises(AttributeError):
        p.tf = 5  # type: ignore[misc]  # frozen dataclass: assignment must raise


def test_postings_sorted_with_tf(corpus_dir: Path) -> None:
    idx = build_index(iter_documents(corpus_dir), layout="slots")
    assert [(p.doc_id, p.tf) for p in idx["apple"]] == [(0, 2), (2, 1)]
    for term in idx:
        ids = [p.doc_id for p in idx[term]]
        assert ids == sorted(ids)


def test_doc_lengths_and_meta(index: Index) -> None:
    assert index.doc_length(1) == 7
    assert index.doc_meta[0].path.endswith("a.txt")


def test_positions(corpus_dir: Path) -> None:
    idx = build_index(iter_documents(corpus_dir), positions=True, layout="slots")
    assert idx["apple"][0].positions == (0, 3)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("apple", [0, 2]),
        ("apple banana", [0]),
        ("apple AND date", [2]),
        ("banana OR egg", [0, 1, 2]),
        ("apple NOT banana", [2]),
        ("NOT apple", [1, 3]),
        ("cherry OR NOT apple", [0, 1, 3]),
        ("nosuchword", []),
    ],
)
@pytest.mark.parametrize("engine", ["merge", "set"])
def test_boolean_queries(index: Index, query: str, expected: list[int], engine: str) -> None:
    assert search(index, query, "merge" if engine == "merge" else "set") == expected


@pytest.mark.parametrize("bad", ["", "apple AND", "apple NOT"])
def test_boolean_bad_queries(index: Index, bad: str) -> None:
    with pytest.raises(QueryError):
        search(index, bad)


@pytest.mark.parametrize("fmt", ["pickle", "binary"])
@pytest.mark.parametrize("positions", [False, True])
def test_save_load_roundtrip(corpus_dir: Path, tmp_path: Path, layout: Layout, fmt: str, positions: bool) -> None:
    idx = build_index(iter_documents(corpus_dir), positions=positions, layout=layout)
    out = tmp_path / "idx.bin"
    save(idx, out, "pickle" if fmt == "pickle" else "binary")
    back = load(out)
    assert back.doc_lengths == idx.doc_lengths
    assert back.doc_meta == idx.doc_meta
    for term in idx:
        assert postings_as_objects(back, term) == postings_as_objects(idx, term)
    assert search(back, "apple NOT banana") == [2]


def test_merge_equals_set_on_random_lists() -> None:
    rng = random.Random(1)
    for _ in range(200):
        a = sorted(rng.sample(range(60), rng.randint(0, 30)))
        b = sorted(rng.sample(range(60), rng.randint(0, 30)))
        assert and_merge(a, b) == and_set(a, b)
        assert or_merge(a, b) == or_set(a, b)
        assert not_merge(a, b) == not_set(a, b)
