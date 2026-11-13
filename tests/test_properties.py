"""Property-based tests (Hypothesis). They run many examples, so they are marked slow."""

import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from findex.corpus import Document
from findex.index import Layout, build_index
from findex.search import and_merge, and_set, not_merge, not_set, or_merge, or_set
from findex.store import load, postings_as_objects, save

pytestmark = pytest.mark.slow

WORDS = ["a", "b", "c", "dd", "eee", "don't", "well-known", "x1", "été", "кіт"]
doc_texts = st.lists(
    st.lists(st.sampled_from(WORDS), max_size=12).map(" ".join), min_size=1, max_size=6
)
sorted_ids = st.lists(st.integers(0, 200), unique=True, max_size=60).map(sorted)


def docs_of(texts: list[str]) -> list[Document]:
    return [Document(i, Path(f"d{i}.txt"), t) for i, t in enumerate(texts)]


@settings(max_examples=60, deadline=None)
@given(texts=doc_texts, layout=st.sampled_from(["dataclass", "slots", "array"]), positions=st.booleans())
def test_pickle_and_binary_roundtrip(texts: list[str], layout: Layout, positions: bool) -> None:
    idx = build_index(docs_of(texts), positions=positions, layout=layout)
    with tempfile.TemporaryDirectory() as d:
        for fmt in ("pickle", "binary"):
            path = Path(d) / f"i.{fmt}"
            save(idx, path, "pickle" if fmt == "pickle" else "binary")
            back = load(path)
            assert back.doc_lengths == idx.doc_lengths
            assert back.doc_meta == idx.doc_meta
            assert set(back) == set(idx)
            for term in idx:
                assert postings_as_objects(back, term) == postings_as_objects(idx, term)


@settings(max_examples=60, deadline=None)
@given(texts=doc_texts, layout=st.sampled_from(["dataclass", "slots", "array"]))
def test_postings_sorted_and_consistent(texts: list[str], layout: Layout) -> None:
    idx = build_index(docs_of(texts), positions=True, layout=layout)
    per_doc: dict[int, int] = {}
    for term in idx:
        ps = idx[term]
        ids = [p.doc_id for p in ps]
        assert ids == sorted(ids) and len(set(ids)) == len(ids)  # sorted, one posting per doc
        for p in ps:
            assert p.tf == len(p.positions) >= 1
            per_doc[p.doc_id] = per_doc.get(p.doc_id, 0) + p.tf
    for doc_id, length in idx.doc_lengths.items():
        assert per_doc.get(doc_id, 0) == length  # sum of tf == document length


@settings(max_examples=200, deadline=None)
@given(a=sorted_ids, b=sorted_ids)
def test_merge_matches_set_operations(a: list[int], b: list[int]) -> None:
    assert and_merge(a, b) == and_set(a, b) == sorted(set(a) & set(b))
    assert or_merge(a, b) == or_set(a, b) == sorted(set(a) | set(b))
    assert not_merge(a, b) == not_set(a, b) == sorted(set(a) - set(b))
