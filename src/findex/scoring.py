"""Ranking functions behind a structural ``Scorer`` protocol.

Both scorers are interchangeable without sharing a base class (duck typing).

TF-IDF
    score(t, d) = (1 + ln tf) * ln(N / df)        (sublinear tf, natural log)
    A term that occurs in every document has idf 0 and contributes nothing.

BM25 (k1=1.5, b=0.75)
    idf(t)      = ln(1 + (N - df + 0.5) / (df + 0.5))        (Lucene variant, >= 0)
    score(t, d) = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from .index import Index, Posting


class Scorer(Protocol):
    def score(self, term: str, posting: Posting, index: Index) -> float: ...


class TfIdf:
    def score(self, term: str, posting: Posting, index: Index) -> float:
        df = index.df(term)
        if df == 0 or posting.tf == 0:
            return 0.0
        return (1 + math.log(posting.tf)) * math.log(index.num_docs / df)


@dataclass(frozen=True)
class BM25:
    k1: float = 1.5
    b: float = 0.75

    def score(self, term: str, posting: Posting, index: Index) -> float:
        df = index.df(term)
        if df == 0:
            return 0.0
        n = index.num_docs
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        norm = 1 - self.b + self.b * index.doc_length(posting.doc_id) / index.avg_doc_length
        tf = posting.tf
        return idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
