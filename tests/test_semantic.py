import zlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from findex.corpus import Document
from findex.semantic import SemanticIndex, chunk_text, embed_corpus, rrf

DIM = 64


def fake_embed(texts: Sequence[str]) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
    """Deterministic bag-of-words hashing 'embedding' (no model download needed in tests)."""
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in t.lower().split():
            out[i, zlib.crc32(w.encode()) % DIM] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return (out / np.maximum(norms, 1e-12)).astype(np.float32)


DOCS = [
    Document(0, Path("a.txt"), "the whale swam in the deep sea " * 40 + "harpoon " * 30),
    Document(1, Path("b.txt"), "prime numbers are divisible only by one and themselves " * 50),
    Document(2, Path("c.txt"), "a girl lived on a green farm with her adoptive family " * 30),
]


def test_chunk_text_offsets_and_overlap() -> None:
    text = ("word " * 600).strip()
    chunks = list(chunk_text(text, size=300, overlap=50))
    assert len(chunks) > 5
    for off, chunk in chunks:
        assert 40 <= len(chunk) <= 300
        assert chunk.split()[0] == "word" and text[off:].lstrip().startswith("word")
    offsets = [o for o, _ in chunks]
    assert offsets == sorted(offsets) and len(set(offsets)) == len(offsets)


def test_chunk_text_tiny_text_is_dropped() -> None:
    assert list(chunk_text("too short")) == []


@pytest.fixture
def sem() -> SemanticIndex:
    return embed_corpus(DOCS, embed=fake_embed, batch=7)


def test_embed_corpus_shapes_and_unit_norm(sem: SemanticIndex) -> None:
    assert sem.matrix.shape[1] == DIM and sem.matrix.dtype == np.float32
    assert np.allclose(np.linalg.norm(sem.matrix, axis=1), 1.0, atol=1e-5)
    assert sem.doc_ids.tolist() == [0, 1, 2]
    assert list(sem.starts) == sorted(sem.starts)
    assert len(sem.chunk_doc) == len(sem.chunk_offset) == len(sem.matrix)


@pytest.mark.parametrize(
    ("query", "winner"),
    [("whale harpoon sea", 0), ("divisible prime numbers", 1), ("girl on a farm", 2)],
)
def test_search_ranks_by_best_chunk(sem: SemanticIndex, query: str, winner: int) -> None:
    hits = sem.search(query, k=3)
    assert hits[0].doc_id == winner
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert len(hits) == 3 and {h.doc_id for h in hits} == {0, 1, 2}


def test_search_k_larger_than_corpus_and_chunk_offset(sem: SemanticIndex) -> None:
    hits = sem.search("whale", k=50)
    assert len(hits) == 3
    assert hits[0].chunk_offset >= 0


def test_doc_scores_is_max_over_chunks(sem: SemanticIndex) -> None:
    q = fake_embed(["whale"])[0]
    sims = sem.matrix @ q
    expected = [sims[sem.chunk_doc == d].max() for d in sem.doc_ids]
    assert sem.doc_scores(q) == pytest.approx(expected)


def test_max_chunks_per_doc_samples_evenly() -> None:
    full = embed_corpus(DOCS, embed=fake_embed)
    small = embed_corpus(DOCS, embed=fake_embed, max_chunks_per_doc=3)
    assert len(small) == sum(min(3, int((full.chunk_doc == d).sum())) for d in range(3)) < len(full)
    assert small.doc_ids.tolist() == [0, 1, 2]


def test_save_load_roundtrip_with_mmap(sem: SemanticIndex, tmp_path: Path) -> None:
    path = tmp_path / "emb.npy"
    sem.save(path)
    back = SemanticIndex.load(path, embed=fake_embed, mmap=True)
    assert isinstance(back.matrix, np.memmap)
    assert np.array_equal(np.asarray(back.matrix), sem.matrix)
    assert [h.doc_id for h in back.search("prime numbers")] == [h.doc_id for h in sem.search("prime numbers")]


def test_rrf_basic_properties() -> None:
    fused = rrf([[1, 2, 3], [3, 2, 1]])
    scores = dict(fused)
    assert scores[2] == pytest.approx(2 / 62)  # middle in both lists
    assert scores[1] == pytest.approx(1 / 61 + 1 / 63) == pytest.approx(scores[3])
    # a document that is first in one list and absent from the other still beats one that is low in both
    assert rrf([[9, 1, 2, 3], [8]])[0][0] in (9, 8)
    assert [d for d, _ in rrf([[1, 2], [1, 2]])] == [1, 2]
    assert rrf([]) == []
