import inspect
import unicodedata
from pathlib import Path

import pytest

from findex.corpus import iter_documents
from findex.tokenize import tokenize


def toks(s: str) -> list[str]:
    return list(tokenize(s))


def test_is_generator() -> None:
    assert inspect.isgeneratorfunction(tokenize)
    assert inspect.isgeneratorfunction(iter_documents)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello WORLD Straße", ["hello", "world", "strasse"]),  # casefold, not lower
        ("Привіт, СВІТЕ!", ["привіт", "світе"]),  # Cyrillic
        ("Ünïcödé ÀÉÎ", ["ünïcödé", "àéî"]),  # accented Latin
        ("a, b; c... d?!", ["a", "b", "c", "d"]),  # punctuation
        ("", []),
        ("   \n\t ", []),
        ("Don't 'quote' п’ять п'ять", ["don't", "quote", "п'ять", "п'ять"]),  # apostrophes
        ("well-known - dash -x y-", ["well-known", "dash", "x", "y"]),  # hyphens
        ("abc123 3.14 1990-2000 a_b", ["abc123", "3", "14", "1990-2000", "a", "b"]),  # digits
    ],
)
def test_tokenize_cases(text: str, expected: list[str]) -> None:
    assert toks(text) == expected


def test_combining_mark_is_normalised() -> None:
    decomposed = "café"
    assert toks(decomposed) == ["café"]
    assert toks(decomposed) == toks(unicodedata.normalize("NFC", decomposed))


def test_iter_documents_is_lazy_and_tolerates_bad_bytes(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("one", encoding="utf-8")
    (tmp_path / "b.txt").write_bytes(b"\xff\xfe\x00bad")  # not UTF-8: falls back to cp1252
    it = iter_documents(tmp_path)
    first = next(it)
    assert first.doc_id == 0 and first.text == "one"
    assert len(list(it)) == 1


def test_iter_documents_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "docs.jsonl"
    p.write_text('{"text": "alpha"}\n\nnot json\n{"nope": 1}\n{"text": "beta"}\n', encoding="utf-8")
    docs = list(iter_documents(p))
    assert [d.text for d in docs] == ["alpha", "beta"]  # bad lines are skipped, not fatal


def test_lazy_and_eager_stats_agree(tmp_path: Path) -> None:
    from findex.stats import eager_stats, lazy_stats

    (tmp_path / "a.txt").write_text("the cat the dog", encoding="utf-8")
    (tmp_path / "b.txt").write_text("a cat", encoding="utf-8")
    lazy = lazy_stats(iter_documents(tmp_path))
    eager = eager_stats(iter_documents(tmp_path))
    assert lazy == eager
    assert lazy[0] == 2 and lazy[1] == 6 and lazy[2]["the"] == 2
