"""Search modes shared by the CLI and the web app: lexical (BM25/TF-IDF), semantic, hybrid (RRF)."""

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from .corpus import read_text
from .index import Index
from .npsearch import NumpyIndex
from .query import QueryError, parse
from .ranking import SearchResult, match_ids, present_terms
from .ranking import search as py_search
from .scoring import BM25, Scorer, TfIdf
from .semantic import SemanticHit, SemanticIndex, rrf
from .snippets import make_snippet

Mode = Literal["lexical", "semantic", "hybrid"]
ScorerName = Literal["bm25", "tfidf"]
POOL = 50  # how many candidates each ranking contributes to the hybrid fusion
_OPERATORS = re.compile(r'[()"]|\b(?:AND|OR|NOT)\b')


@dataclass
class RankedHit:
    doc_id: int
    score: float
    snippet: str


def plain_query(q: str) -> str:
    """The query as natural language: operators, quotes and parentheses removed."""
    return " ".join(_OPERATORS.sub(" ", q).split())


def as_or_query(q: str) -> str:
    """A bag-of-words query becomes ``a OR b OR c``; queries that already use operators are kept."""
    if _OPERATORS.search(q):
        return q
    words = plain_query(q).split()
    return " OR ".join(words) if len(words) > 1 else q


def doc_text(index: Index, doc_id: int) -> str:
    path = Path(index.doc_meta[doc_id].path)
    return (read_text(path) or "") if path.is_file() else ""


@lru_cache(maxsize=4)
def numpy_for(index: Index) -> NumpyIndex | None:
    """NumPy view of the index (built once per index; ``None`` if it is not in the array layout)."""
    return NumpyIndex(index) if index.layout == "array" else None


def lexical_top(index: Index, q: str, scorer: ScorerName, n: int) -> list[SearchResult]:
    npi = numpy_for(index)
    if npi is not None:
        return npi.search(q, scorer, n)
    s: Scorer = TfIdf() if scorer == "tfidf" else BM25()
    return py_search(index, q, s, k=n)


def chunk_snippet(index: Index, hit: SemanticHit, length: int = 260) -> str:
    text = doc_text(index, hit.doc_id)
    piece = " ".join(text[hit.chunk_offset : hit.chunk_offset + length * 2].split())
    return piece[:length] + ("..." if len(piece) > length else "")


def lexical_snippet(index: Index, doc_id: int, terms: set[str]) -> str:
    return make_snippet(doc_text(index, doc_id), present_terms(index, doc_id, terms) or terms)


def ranked_hits(
    index: Index,
    q: str,
    scorer: ScorerName = "bm25",
    mode: Mode = "lexical",
    k: int = 10,
    page: int = 1,
    sem: SemanticIndex | None = None,
) -> tuple[list[RankedHit], int]:
    """Hits for one page and the total number of candidates. Raises ``QueryError`` / ``LookupError``."""
    lo, hi = (page - 1) * k, page * k
    if mode == "lexical":
        tree = parse(q)
        total = len(match_ids(index, q))
        terms = tree.terms()
        top = lexical_top(index, q, scorer, hi)
        return [RankedHit(r.doc_id, round(r.score, 4), lexical_snippet(index, r.doc_id, terms)) for r in top[lo:hi]], total

    if sem is None:
        raise LookupError("semantic search is unavailable: no embeddings loaded")
    sem_hits = sem.search(plain_query(q), k=max(hi, POOL))
    by_id = {h.doc_id: h for h in sem_hits}
    if mode == "semantic":
        page_hits = sem_hits[lo:hi]
        return [RankedHit(h.doc_id, round(h.score, 4), chunk_snippet(index, h)) for h in page_hits], len(sem.doc_ids)

    # hybrid: BM25 over the bag of words + semantic ranking, merged with Reciprocal Rank Fusion
    lex_q = as_or_query(q)
    lex: list[SearchResult] = []
    terms: set[str] = set()
    try:
        lex = lexical_top(index, lex_q, scorer, max(hi, POOL))
        terms = parse(lex_q).terms()
    except QueryError:
        pass
    fused = rrf([[r.doc_id for r in lex], [h.doc_id for h in sem_hits]])
    out: list[RankedHit] = []
    for doc_id, score in fused[lo:hi]:
        if doc_id in {r.doc_id for r in lex} and terms:
            snippet = lexical_snippet(index, doc_id, terms)
        else:
            snippet = chunk_snippet(index, by_id[doc_id]) if doc_id in by_id else ""
        out.append(RankedHit(doc_id, round(score, 5), snippet))
    return out, len(fused)


def n_pages(total: int, k: int) -> int:
    return max(1, math.ceil(total / k))
