import logging
from pathlib import Path

import pytest

from findex.index import Index, Posting
from findex.query import And, Node, Not, Or, Phrase, QueryError, Term, parse
from findex.ranking import SearchResult, match_ids, search
from findex.scoring import BM25, Scorer, TfIdf
from findex.snippets import make_snippet
from findex.store import open_index
from findex.util import timed

from .conftest import DOCS


def test_index_mapping_api(index: Index) -> None:
    vocab = {w for text in DOCS.values() for w in text.split()}
    assert len(index) == len(vocab)
    assert "python" in index and "ruby" not in index
    assert [p.doc_id for p in index["python"]] == [0, 2, 3]
    assert "java" in list(index)
    with pytest.raises(KeyError):
        index["ruby"]
    assert index.num_docs == 4
    assert index.df("python") == 3
    assert index.avg_doc_length == sum(index.doc_lengths.values()) / 4
    assert "Index(" in repr(index)


def test_open_index_closes_even_on_error(index_file: Path) -> None:
    holder: list[Index] = []
    with pytest.raises(RuntimeError), open_index(index_file) as idx:
        holder.append(idx)
        assert len(idx) > 0
        raise RuntimeError("boom")
    assert len(holder[0]) == 0  # released in finally; exception not swallowed


@pytest.mark.parametrize(
    ("query", "tree"),
    [
        ("a", Term("a")),
        ("a b", And(Term("a"), Term("b"))),
        ("a AND b", And(Term("a"), Term("b"))),
        ("a OR b c", Or(Term("a"), And(Term("b"), Term("c")))),
        ("(a OR b) c", And(Or(Term("a"), Term("b")), Term("c"))),
        (
            "python AND (async OR await) NOT java",
            And(Term("python"), Or(Term("async"), Term("await")), Not(Term("java"))),
        ),
        ('"Event Loop" x', And(Phrase("event", "loop"), Term("x"))),
        ("NOT a", Not(Term("a"))),
    ],
)
def test_parser_trees(query: str, tree: Node) -> None:
    assert parse(query) == tree


@pytest.mark.parametrize("bad", ["", "   ", "(a", "a)", "a OR", "AND a", '"abc', "NOT"])
def test_parser_errors(bad: str) -> None:
    with pytest.raises(QueryError):
        parse(bad)


def test_operator_overloads() -> None:
    a, b, c = Term("a"), Term("b"), Term("c")
    assert a & b & c == And(a, b, c)
    assert a | b == Or(a, b)
    assert ~a == Not(a)


def test_evaluate_and_phrase(index: Index) -> None:
    assert parse("python NOT java").evaluate(index) == {0, 3}
    assert parse('"event loop"').evaluate(index) == {0, 1}
    assert parse('"loop event"').evaluate(index) == set()
    assert parse('"python async"').evaluate(index) == {0}


def test_phrase_requires_positions(corpus_dir: Path) -> None:
    from findex.corpus import iter_documents
    from findex.index import build_index

    idx = build_index(iter_documents(corpus_dir))
    with pytest.raises(QueryError):
        parse('"apple banana"').evaluate(idx)


def test_scorers_are_interchangeable_without_inheritance() -> None:
    # Neither class inherits from Scorer; pyright checks this list structurally.
    scorers: list[Scorer] = [TfIdf(), BM25(k1=1.2, b=0.5)]
    assert all(callable(s.score) for s in scorers)
    assert Scorer not in TfIdf.__mro__ and Scorer not in BM25.__mro__


@pytest.mark.parametrize("scorer", [TfIdf(), BM25()])
def test_rare_term_beats_common(index: Index, scorer: Scorer) -> None:
    assert scorer.score("async", Posting(2, 1), index) > scorer.score("python", Posting(2, 1), index)


def test_bm25_saturation(index: Index) -> None:
    bm = BM25()
    s1 = bm.score("python", Posting(0, 1), index)
    s20 = bm.score("python", Posting(0, 20), index)
    s40 = bm.score("python", Posting(0, 40), index)
    assert s20 < s1 * (bm.k1 + 1)
    assert s40 - s20 < 0.1 * s20


def test_bm25_length_normalisation(index: Index) -> None:
    bm = BM25()
    short = bm.score("python", Posting(2, 1), index)  # 9 tokens
    long_ = bm.score("python", Posting(3, 1), index)  # ~360 tokens
    assert short > long_


def test_search_ranking_and_topk(index: Index) -> None:
    res = search(index, "python", BM25(), k=2)
    assert len(res) == 2
    assert res == sorted(res, reverse=True)
    assert isinstance(res[0], SearchResult)
    assert SearchResult(1.0, 1) < SearchResult(2.0, 0)


def test_cache_hit_logged(index: Index, caplog: pytest.LogCaptureFixture) -> None:
    match_ids.cache_clear()
    with caplog.at_level(logging.INFO):
        search(index, "event loop")
        search(index, "event loop")
    assert "cache miss" in caplog.text and "cache hit" in caplog.text
    assert "search took" in caplog.text


def test_timed_preserves_metadata() -> None:
    @timed
    def f(x: int) -> int:
        """doc"""
        return x

    assert f.__name__ == "f" and f.__doc__ == "doc" and f(3) == 3


def test_timed_logs_even_on_exception(caplog: pytest.LogCaptureFixture) -> None:
    @timed
    def boom() -> None:
        raise ValueError("x")

    with caplog.at_level(logging.INFO), pytest.raises(ValueError, match="x"):
        boom()
    assert "boom took" in caplog.text


def test_snippet_highlight() -> None:
    text = "x " * 100 + "The Event Loop runs forever " + "y " * 100
    s = make_snippet(text, ["event", "loop"], width=30)
    assert "**Event**" in s and "**Loop**" in s
    assert s.startswith("...") and s.endswith("...")
    assert len(s) < 120
    assert make_snippet("no match here", ["zzz"]) == "no match here"
