"""NumPy ranking: postings as int32 arrays, BM25/TF-IDF as array expressions, top-k via argpartition.

``NumpyIndex`` views the ``array('I')`` postings of an ``Index`` (array layout) as ``np.int32``
without copying. Scoring a term is then a handful of vectorised operations over its whole
postings list instead of a Python call per posting; the best-k are picked with
``np.argpartition`` (O(n)) and only those k are sorted.

Result order is identical to ``findex.ranking.search``: score descending, ties by larger doc_id
(the order of ``SearchResult(order=True)`` under ``heapq.nlargest``).
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import numpy.typing as npt

from .index import ArrayPostings, Index
from .query import parse
from .ranking import SearchResult

I32 = npt.NDArray[np.int32]
F64 = npt.NDArray[np.float64]
ScorerName = Literal["bm25", "tfidf"]


class NumpyIndex:
    def __init__(self, index: Index) -> None:
        if index.layout != "array":
            raise ValueError("NumpyIndex needs an index in the 'array' layout")
        self.index = index
        self.n_docs = index.num_docs
        lengths = np.zeros(max(index.doc_lengths, default=-1) + 1, dtype=np.int32)
        for d, n in index.doc_lengths.items():
            lengths[d] = n
        self.doc_lengths: I32 = lengths
        self.avgdl = float(index.avg_doc_length)
        self._postings: dict[str, tuple[I32, I32]] = {}
        for term, p in index.postings.items():
            assert isinstance(p, ArrayPostings)
            # array('I') is uint32; values stay far below 2**31, so reinterpret as int32 (zero-copy view)
            self._postings[term] = (
                np.frombuffer(p.doc_ids, dtype=np.uint32).view(np.int32),
                np.frombuffer(p.tfs, dtype=np.uint32).view(np.int32),
            )

    def postings(self, term: str) -> tuple[I32, I32] | None:
        return self._postings.get(term)

    def scores(self, terms: set[str], scorer: ScorerName = "bm25", k1: float = 1.5, b: float = 0.75) -> F64:
        """Dense score vector (length = max doc_id + 1) summed over ``terms``."""
        out = np.zeros(len(self.doc_lengths), dtype=np.float64)
        n = self.n_docs
        for term in terms:
            p = self._postings.get(term)
            if p is None:
                continue
            ids, tf = p
            df = len(ids)
            tf_f = tf.astype(np.float64)
            if scorer == "bm25":
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                norm = 1 - b + b * self.doc_lengths[ids] / self.avgdl
                out[ids] += idf * tf_f * (k1 + 1) / (tf_f + k1 * norm)
            else:
                out[ids] += (1 + np.log(tf_f)) * math.log(n / df)
        return out

    def search(self, query: str, scorer: ScorerName = "bm25", k: int = 10) -> list[SearchResult]:
        tree = parse(query)
        matched = np.fromiter(sorted(tree.evaluate(self.index)), dtype=np.int64)
        if matched.size == 0:
            return []
        scores = self.scores(tree.terms(), scorer)[matched]
        if matched.size > k:
            # O(n) selection with argpartition: find the k-th best score, keep everything at or above it
            # (ties at the cut-off are all kept), then sort just that handful.
            threshold = scores[np.argpartition(-scores, k - 1)[k - 1]]
            keep = np.flatnonzero(scores >= threshold)
        else:
            keep = np.arange(matched.size)
        ids, sc = matched[keep], scores[keep]
        order = np.lexsort((-ids, -sc))[:k]  # score desc, then doc_id desc (== heapq.nlargest on SearchResult)
        meta = self.index.doc_meta
        return [SearchResult(float(sc[i]), int(ids[i]), meta[int(ids[i])].title) for i in order]
