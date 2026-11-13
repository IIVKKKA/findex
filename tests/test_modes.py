from pathlib import Path

import pytest

from findex.corpus import iter_documents
from findex.index import Index
from findex.modes import as_or_query, plain_query, ranked_hits
from findex.semantic import SemanticIndex, embed_corpus

from .test_semantic import fake_embed


@pytest.fixture
def sem(corpus_dir: Path) -> SemanticIndex:
    return embed_corpus(iter_documents(corpus_dir), embed=fake_embed)


@pytest.mark.parametrize(
    ("raw", "plain", "or_q"),
    [
        ('whale AND (harpoon OR ship) NOT "moby dick"', "whale harpoon ship moby dick", 'whale AND (harpoon OR ship) NOT "moby dick"'),
        ("giant sea mammal", "giant sea mammal", "giant OR sea OR mammal"),
        ("single", "single", "single"),
        ("  spaced   out ", "spaced out", "spaced OR out"),
    ],
)
def test_query_helpers(raw: str, plain: str, or_q: str) -> None:
    assert plain_query(raw) == plain
    assert as_or_query(raw) == or_q


def test_lexical_mode_uses_numpy_and_keeps_order(array_index: Index) -> None:
    hits, total = ranked_hits(array_index, "python", "bm25", "lexical", 2, 1)
    assert total == 3 and hits[0].doc_id == 0 and len(hits) == 2
    assert "**python**" in hits[0].snippet.lower()
    page2, _ = ranked_hits(array_index, "python", "bm25", "lexical", 2, 2)
    assert len(page2) == 1 and {h.doc_id for h in hits} | {page2[0].doc_id} == {0, 2, 3}


def test_semantic_mode_returns_best_chunk_snippets(array_index: Index, sem: SemanticIndex) -> None:
    hits, total = ranked_hits(array_index, "event loop threads", "bm25", "semantic", 3, 1, sem)
    assert total == 4 and hits[0].doc_id == 1  # doc b: "... java event loop threads"
    assert hits[0].snippet and all(h.score <= 1.0 + 1e-6 for h in hits)


def test_hybrid_mode_fuses_both_rankings(array_index: Index, sem: SemanticIndex) -> None:
    hits, total = ranked_hits(array_index, "async await event loop", "bm25", "hybrid", 4, 1, sem)
    assert hits[0].doc_id == 0 and total >= 2
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_semantic_without_embeddings_raises_lookup_error(array_index: Index) -> None:
    with pytest.raises(LookupError, match="no embeddings"):
        ranked_hits(array_index, "x", "bm25", "semantic", 3, 1, None)


def test_cli_semantic_mode_needs_embeddings(index_file: Path) -> None:
    from typer.testing import CliRunner

    from findex.cli import app

    r = CliRunner().invoke(app, ["search", str(index_file), "python", "--mode", "semantic"])
    assert r.exit_code == 1 and r.stderr.startswith("error: semantic/hybrid mode needs --embeddings")
