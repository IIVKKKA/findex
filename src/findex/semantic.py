"""Semantic search: chunk embeddings, cosine similarity as a matrix product, RRF hybrid.

* ``embed_corpus`` splits every document into overlapping chunks, embeds them with a small
  sentence-embedding model (all-MiniLM-L6-v2 via ``fastembed``/ONNX, 384 dims), L2-normalises
  the vectors and saves a float32 matrix (``.npy``) plus chunk -> (doc_id, char offset) tables.
* ``SemanticIndex.search`` embeds the query, multiplies the matrix by the query vector (for
  unit vectors that *is* cosine similarity) and scores each document by its best chunk.
* ``rrf`` merges two rankings (e.g. BM25 and semantic) with Reciprocal Rank Fusion.
"""

import logging
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .corpus import Document

log = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
F32 = npt.NDArray[np.float32]
I32 = npt.NDArray[np.int32]

_model: Any = None


def get_model() -> Any:
    """Load the embedding model once per process (first call downloads ~90 MB, then cached)."""
    global _model
    if _model is None:
        from fastembed import TextEmbedding

        _model = TextEmbedding(MODEL_NAME)
    return _model


def embed_texts(texts: Sequence[str], batch_size: int = 64) -> F32:
    vecs = np.array(list(get_model().embed(list(texts), batch_size=batch_size)), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    unit: F32 = (vecs / np.maximum(norms, 1e-12)).astype(np.float32)  # unit length: dot product == cosine
    return unit


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> Iterator[tuple[int, str]]:
    """Yield ``(char_offset, chunk)`` of about ``size`` characters, cut at whitespace."""
    n = len(text)
    pos = 0
    while pos < n:
        end = min(n, pos + size)
        if end < n:
            cut = text.rfind(" ", pos + size // 2, end)
            end = cut if cut != -1 else end
        chunk = " ".join(text[pos:end].split())
        if len(chunk) >= 40:
            yield pos, chunk
        if end >= n:
            break
        pos = max(end - overlap, pos + 1)


@dataclass
class SemanticHit:
    doc_id: int
    score: float
    chunk_offset: int  # character offset of the best-matching chunk in the document


class SemanticIndex:
    def __init__(self, matrix: F32, chunk_doc: I32, chunk_offset: I32, embed: Callable[[Sequence[str]], F32] | None = None) -> None:
        self.matrix = matrix
        self.chunk_doc = chunk_doc
        self.chunk_offset = chunk_offset
        self._embed = embed or embed_texts
        # chunks are stored in doc order, so each document owns a contiguous slice
        self.doc_ids, self.starts = np.unique(chunk_doc, return_index=True)

    def __len__(self) -> int:
        return len(self.matrix)

    def doc_scores(self, query_vec: F32) -> F32:
        """Best-chunk cosine similarity per document (aligned with ``self.doc_ids``)."""
        sims: F32 = self.matrix @ query_vec  # (n_chunks,)  one BLAS matrix-vector product
        return np.maximum.reduceat(sims, self.starts)

    def search(self, query: str, k: int = 10) -> list[SemanticHit]:
        q = self._embed([query])[0]
        best = self.doc_scores(q)
        k = min(k, len(best))
        top = np.argpartition(-best, k - 1)[:k]
        top = top[np.argsort(-best[top], kind="stable")]
        out: list[SemanticHit] = []
        for i in top:
            lo = int(self.starts[i])
            hi = int(self.starts[i + 1]) if i + 1 < len(self.starts) else len(self.matrix)
            j = lo + int(np.argmax(self.matrix[lo:hi] @ q))
            out.append(SemanticHit(int(self.doc_ids[i]), float(best[i]), int(self.chunk_offset[j])))
        return out

    def save(self, path: Path) -> None:
        np.save(path, self.matrix)
        np.savez(path.with_suffix(".meta.npz"), chunk_doc=self.chunk_doc, chunk_offset=self.chunk_offset)

    @classmethod
    def load(cls, path: Path, embed: Callable[[Sequence[str]], F32] | None = None, mmap: bool = True) -> "SemanticIndex":
        matrix: F32 = np.load(path, mmap_mode="r" if mmap else None)
        meta = np.load(path.with_suffix(".meta.npz"))
        return cls(matrix, meta["chunk_doc"], meta["chunk_offset"], embed)


def embed_corpus(
    docs: Iterable[Document],
    embed: Callable[[Sequence[str]], F32] = embed_texts,
    batch: int = 512,
    progress: Callable[[int], None] | None = None,
    max_chunks_per_doc: int | None = None,
) -> SemanticIndex:
    """Chunk and embed documents in order; the matrix is built batch by batch.

    ``max_chunks_per_doc`` keeps an evenly spaced sample of each long document's chunks
    (embedding is the slow step: ~25 chunks/s on the CPU used for the README numbers).
    """
    mats: list[F32] = []
    chunk_doc: list[int] = []
    chunk_off: list[int] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            mats.append(embed(buf))
            buf.clear()
            if progress:
                progress(len(chunk_doc))

    for doc in docs:
        chunks = list(chunk_text(doc.text))
        if max_chunks_per_doc is not None and len(chunks) > max_chunks_per_doc:
            picks = [int(i) for i in np.linspace(0, len(chunks) - 1, max_chunks_per_doc)]
            chunks = [chunks[i] for i in picks]
        for off, chunk in chunks:
            buf.append(chunk)
            chunk_doc.append(doc.doc_id)
            chunk_off.append(off)
            if len(buf) >= batch:
                flush()
    flush()
    matrix = np.concatenate(mats) if mats else np.zeros((0, 384), dtype=np.float32)
    return SemanticIndex(matrix, np.array(chunk_doc, dtype=np.int32), np.array(chunk_off, dtype=np.int32), embed)


def rrf(rankings: Sequence[Sequence[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank), rank starting at 1."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
