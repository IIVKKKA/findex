import pytest

from findex.corpus import iter_documents
from findex.index import DocMeta, Posting, build_index
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

DOCS = {
    "a.txt": "apple banana cherry apple",
    "b.txt": "banana cherry date",
    "c.txt": "apple date egg",
    "d.txt": "fig",
}


@pytest.fixture()
def corpus(tmp_path):
    for name, text in DOCS.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path


@pytest.fixture(params=["dataclass", "slots", "array"])
def index(request, corpus):
    return build_index(iter_documents(corpus), layout=request.param)


def test_posting_and_meta_hashable_frozen():
    s = {Posting(1, 2), Posting(1, 2), DocMeta("p", "t")}
    assert len(s) == 2
    with pytest.raises(AttributeError):
        Posting(1, 2).tf = 5  # type: ignore[misc]


def test_postings_sorted_with_tf(corpus):
    idx = build_index(iter_documents(corpus), layout="slots")
    assert idx.postings["apple"] == [Posting(0, 2), Posting(2, 1)]
    assert idx.doc_lengths == {0: 4, 1: 3, 2: 3, 3: 1}
    for plist in idx.postings.values():
        ids = [p.doc_id for p in plist]
        assert ids == sorted(ids)


def test_positions(corpus):
    idx = build_index(iter_documents(corpus), positions=True, layout="slots")
    assert idx.postings["apple"][0].positions == (0, 3)


@pytest.mark.parametrize(
    "query, expected",
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
def test_boolean_queries(index, query, expected, engine):
    assert search(index, query, engine) == expected


@pytest.mark.parametrize("bad", ["", "apple AND", "apple NOT"])
def test_bad_queries(index, bad):
    with pytest.raises(QueryError):
        search(index, bad)


@pytest.mark.parametrize("layout", ["dataclass", "slots", "array"])
@pytest.mark.parametrize("fmt", ["pickle", "binary"])
@pytest.mark.parametrize("positions", [False, True])
def test_save_load_roundtrip(corpus, tmp_path, layout, fmt, positions):
    idx = build_index(iter_documents(corpus), positions=positions, layout=layout)
    out = tmp_path / "idx.bin"
    save(idx, out, fmt)
    back = load(out)
    assert back.doc_lengths == idx.doc_lengths
    assert back.doc_meta == idx.doc_meta
    for term in idx.postings:
        assert postings_as_objects(back, term) == postings_as_objects(idx, term)
    assert search(back, "apple NOT banana") == [2]


def test_merge_equals_set_on_random_lists():
    import random

    rng = random.Random(1)
    for _ in range(200):
        a = sorted(rng.sample(range(60), rng.randint(0, 30)))
        b = sorted(rng.sample(range(60), rng.randint(0, 30)))
        assert and_merge(a, b) == and_set(a, b)
        assert or_merge(a, b) == or_set(a, b)
        assert not_merge(a, b) == not_set(a, b)
