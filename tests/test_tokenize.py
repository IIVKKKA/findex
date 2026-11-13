import inspect
import unicodedata

from findex.corpus import iter_documents
from findex.tokenize import tokenize


def toks(s: str) -> list[str]:
    return list(tokenize(s))


def test_is_generator():
    assert inspect.isgeneratorfunction(tokenize)
    assert inspect.isgeneratorfunction(iter_documents)


def test_mixed_case():
    assert toks("Hello WORLD Straße") == ["hello", "world", "strasse"]


def test_cyrillic():
    assert toks("Привіт, СВІТЕ!") == ["привіт", "світе"]


def test_combining_mark():
    decomposed = "café"
    assert toks(decomposed) == ["café"]
    assert toks(decomposed) == toks(unicodedata.normalize("NFC", decomposed))


def test_punctuation():
    assert toks("a, b; c... d?!") == ["a", "b", "c", "d"]


def test_empty():
    assert toks("") == []


def test_apostrophes():
    assert toks("Don't 'quote' п’ять п'ять") == ["don't", "quote", "п'ять", "п'ять"]


def test_hyphens():
    assert toks("well-known - dash -x y-") == ["well-known", "dash", "x", "y"]


def test_digits():
    assert toks("abc123 3.14 1990-2000 a_b") == ["abc123", "3", "14", "1990-2000", "a", "b"]


def test_iter_documents_lazy(tmp_path):
    (tmp_path / "a.txt").write_text("one", encoding="utf-8")
    (tmp_path / "b.txt").write_bytes(b"\xff\xfe\x00bad")  # cp1252 fallback works
    it = iter_documents(tmp_path)
    first = next(it)
    assert first.doc_id == 0 and first.text == "one"
