"""Ranked retrieval: parse -> evaluate -> score matched docs -> top-k."""

from __future__ import annotations

import heapq
import logging
from bisect import bisect_left
from dataclasses import dataclass, field
from functools import lru_cache

from .index import Index
from .query import parse
from .scoring import BM25, Scorer
from .util import timed

log = logging.getLogger(__name__)


@dataclass(frozen=True, order=True)
class SearchResult:
    # order=True compares fields in order: score first, then doc_id.
    score: float
    doc_id: int
    title: str = field(default="", compare=False)


@lru_cache(maxsize=256)
def match_ids(index: Index, query: str) -> frozenset[int]:
    """query string -> set of matching doc ids (cached; Index hashes by identity)."""
    return frozenset(parse(query).evaluate(index))


@timed
def search(index: Index, query: str, scorer: Scorer | None = None, k: int = 10) -> list[SearchResult]:
    """Top-``k`` results for ``query``, best first, ranked only over the matched set."""
    scorer = scorer or BM25()
    hits_before = match_ids.cache_info().hits
    matched = match_ids(index, query)
    log.info(
        "query %r: %d matches (%s)",
        query,
        len(matched),
        "cache hit" if match_ids.cache_info().hits > hits_before else "cache miss",
    )
    scores: dict[int, float] = dict.fromkeys(matched, 0.0)
    for term in parse(query).terms():
        if term not in index:
            continue
        for posting in index[term]:
            if posting.doc_id in scores:
                scores[posting.doc_id] += scorer.score(term, posting, index)
    results = (
        SearchResult(score, doc_id, index.doc_meta[doc_id].title) for doc_id, score in scores.items()
    )
    # nlargest is O(n log k) and keeps only k items; sorted()[:k] is O(n log n)
    return heapq.nlargest(k, results)


def present_terms(index: Index, doc_id: int, terms: set[str]) -> set[str]:
    """The query terms that actually occur in ``doc_id`` (binary search in each sorted postings list).

    Lets the snippet code stop as soon as one window covers all terms *this document has*
    instead of scanning the whole book looking for terms it does not contain (lab 8).
    """
    out: set[str] = set()
    for t in terms:
        ids = index.doc_ids(t)
        i = bisect_left(ids, doc_id)
        if i < len(ids) and ids[i] == doc_id:
            out.add(t)
    return out
