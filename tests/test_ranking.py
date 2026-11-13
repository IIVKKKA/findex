import logging

import pytest

from findex.corpus import iter_documents
from findex.index import Index, Posting, build_index
from findex.query import And, Not, Or, Phrase, QueryError, Term, parse
from findex.ranking import SearchResult, match_ids, search
from findex.scoring import BM25, Scorer, TfIdf
from findex.snippets import make_snippet
from findex.store import open_index, save
from findex.util import timed

DOCS = {
    "a.txt": "python async await event loop python",
    "b.txt": "java event loop threads",
    "c.txt": "python java the the the the the",
    "d.txt": "the quick brown fox jumps over the lazy dog " * 40 + "python",
}


@pytest.fixture()
def index(tmp_path) -> Index:
    for n, t in DOCS.items():
        (tmp_path / n).write_text(t, encoding="utf-8")
    return build_index(iter_documents(tmp_path), positions=True, layout="array")


def test_index_mapping_api(index):
    assert len(index) == len(set(" ".join(DOCS.values()).split()))
    assert "python" in index and "ruby" not in index
    assert [p.doc_id for p in index["python"]] == [0, 2, 3]
    assert "java" in list(index)
    with pytest.raises(KeyError):
        index["ruby"]
    assert index.num_docs == 4
    assert index.df("python") == 3
    assert index.avg_doc_length == sum(index.doc_lengths.values()) / 4
    assert "Index(" in repr(index)


def test_open_index_closes_even_on_error(index, tmp_path):
    path = tmp_path / "i.bin"
    save(index, path, "binary")
    holder = []
    with pytest.raises(RuntimeError), open_index(path) as idx:
        holder.append(idx)
        assert len(idx) > 0
        raise RuntimeError("boom")
    assert len(holder[0]) == 0  # released in finally; exception not swallowed


@pytest.mark.parametrize(
    "query, tree",
    [
        ("a", Term("a")),
        ("a b", And(Term("a"), Term("b"))),
        ("a AND b", And(Term("a"), Term("b"))),
        ("a OR b c", Or(Term("a"), And(Term("b"), Term("c")))),
        ("(a OR b) c", And(Or(Term("a"), Term("b")), Term("c"))),
        ("python AND (async OR await) NOT java",
         And(Term("python"), Or(Term("async"), Term("await")), Not(Term("java")))),
        ('"Event Loop" x', And(Phrase("event", "loop"), Term("x"))),
        ("NOT a", Not(Term("a"))),
    ],
)
def test_parser_trees(query, tree):
    assert parse(query) == tree


@pytest.mark.parametrize("bad", ["", "   ", "(a", "a)", "a OR", "AND a", '"abc', "NOT"])
def test_parser_errors(bad):
    with pytest.raises(QueryError):
        parse(bad)


def test_operator_overloads():
    a, b, c = Term("a"), Term("b"), Term("c")
    assert a & b & c == And(a, b, c)
    assert a | b == Or(a, b)
    assert ~a == Not(a)


def test_evaluate_and_phrase(index):
    assert parse("python NOT java").evaluate(index) == {0, 3}
    assert parse('"event loop"').evaluate(index) == {0, 1}
    assert parse('"loop event"').evaluate(index) == set()
    assert parse('"python async"').evaluate(index) == {0}


def test_phrase_requires_positions(tmp_path):
    (tmp_path / "a.txt").write_text("a b", encoding="utf-8")
    idx = build_index(iter_documents(tmp_path))
    with pytest.raises(QueryError):
        parse('"a b"').evaluate(idx)


def test_scorers_satisfy_protocol_without_inheritance():
    for s in (TfIdf(), BM25()):
        assert isinstance(s.score, object)
        assert not (s is None)
    scorers: list[Scorer] = [TfIdf(), BM25(k1=1.2, b=0.5)]
    assert len(scorers) == 2


@pytest.mark.parametrize("scorer", [TfIdf(), BM25()])
def test_rare_term_beats_common(index, scorer):
    # doc 2 has "java" (rarer) + lots of "the"; doc 3 has many "the" + "python"
    p_rare = Posting(2, 1)
    p_common = Posting(2, 1)
    assert scorer.score("async", p_rare, index) > scorer.score("python", p_common, index)


def test_bm25_saturation_and_length_norm(index):
    bm = BM25()
    s1 = bm.score("python", Posting(0, 1), index)
    s20 = bm.score("python", Posting(0, 20), index)
    s40 = bm.score("python", Posting(0, 40), index)
    assert s20 < s1 * (bm.k1 + 1)  # bounded by saturation
    assert s40 - s20 < 0.1 * s20
    short = bm.score("python", Posting(2, 1), index)  # 7 tokens
    long_ = bm.score("python", Posting(3, 1), index)  # ~361 tokens
    assert short > long_


def test_search_ranking_and_topk(index):
    res = search(index, "python", BM25(), k=2)
    assert len(res) == 2
    assert res == sorted(res, reverse=True)
    assert res[0].doc_id == 0  # tf=2 in a short doc
    assert isinstance(res[0], SearchResult)
    assert SearchResult(1.0, 1) < SearchResult(2.0, 0)


def test_cache_hit_logged(index, caplog):
    match_ids.cache_clear()
    with caplog.at_level(logging.INFO):
        search(index, "event loop")
        search(index, "event loop")
    text = caplog.text
    assert "cache miss" in text and "cache hit" in text
    assert "search took" in text


def test_timed_preserves_metadata():
    @timed
    def f(x: int) -> int:
        """doc"""
        return x

    assert f.__name__ == "f" and f.__doc__ == "doc" and f(3) == 3


def test_snippet_highlight():
    text = "x " * 100 + "The Event Loop runs forever " + "y " * 100
    s = make_snippet(text, ["event", "loop"], width=30)
    assert "**Event**" in s and "**Loop**" in s
    assert s.startswith("...") and s.endswith("...")
    assert len(s) < 120
    assert make_snippet("no match here", ["zzz"]) == "no match here"
