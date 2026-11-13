from pathlib import Path

import pytest

from findex.corpus import Document
from findex.index import Index, build_index
from findex.npsearch import NumpyIndex, ScorerName
from findex.ranking import match_ids, search
from findex.scoring import BM25, Scorer, TfIdf

QUERIES = ["python", "python OR java", "event loop", '"event loop"', "python NOT java", "the", "fig OR apple OR date",
           "nosuchword", "apple banana"]


def lab3(index: Index, q: str, name: ScorerName, k: int) -> list[tuple[int, float]]:
    scorer: Scorer = BM25() if name == "bm25" else TfIdf()
    match_ids.cache_clear()
    return [(r.doc_id, r.score) for r in search(index, q, scorer, k=k)]


@pytest.mark.parametrize("name", ["bm25", "tfidf"])
@pytest.mark.parametrize("k", [1, 2, 10])
@pytest.mark.parametrize("query", QUERIES)
def test_numpy_order_matches_lab3(array_index: Index, query: str, name: ScorerName, k: int) -> None:
    got = [(r.doc_id, r.score) for r in NumpyIndex(array_index).search(query, name, k)]
    want = lab3(array_index, query, name, k)
    assert [d for d, _ in got] == [d for d, _ in want]  # same order
    assert [s for _, s in got] == pytest.approx([s for _, s in want], rel=1e-9)


def test_ties_are_broken_like_heapq_nlargest() -> None:
    # 12 identical documents: every score ties, so the cut-off at k falls inside a tie group
    docs = [Document(i, Path(f"d{i}.txt"), "alpha beta gamma") for i in range(12)]
    idx = build_index(docs, layout="array")
    for k in (1, 3, 5, 11, 12):
        got = [r.doc_id for r in NumpyIndex(idx).search("alpha", "bm25", k)]
        want = [d for d, _ in lab3(idx, "alpha", "bm25", k)]
        assert got == want == list(range(11, 11 - k, -1))  # larger doc_id wins ties


def test_numpy_postings_are_zero_copy_views(array_index: Index) -> None:
    ni = NumpyIndex(array_index)
    ids, tf = ni.postings("python") or (None, None)
    assert ids is not None and tf is not None
    assert ids.dtype.name == "int32" and not ids.flags["OWNDATA"]
    assert ids.tolist() == [0, 2, 3]


def test_requires_array_layout(corpus_dir: Path) -> None:
    from findex.corpus import iter_documents

    with pytest.raises(ValueError, match="array"):
        NumpyIndex(build_index(iter_documents(corpus_dir), layout="slots"))


@pytest.mark.skipif(not Path("index.bin").exists(), reason="needs the full-corpus index.bin")
def test_order_matches_lab3_on_full_corpus() -> None:
    from findex.store import load

    index = load(Path("index.bin"))
    ni = NumpyIndex(index)
    for q in ["whale OR harpoon OR ship", "paradise OR satan OR eden", "the", "mersenne", '"white whale"']:
        for name in ("bm25", "tfidf"):
            got = [r.doc_id for r in ni.search(q, name, 10)]
            want = [d for d, _ in lab3(index, q, name, 10)]
            assert got == want, (q, name)
