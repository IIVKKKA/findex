"""Shared fixtures: a tiny corpus on disk and the index built from it."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from findex.corpus import iter_documents
from findex.index import Index, Layout, build_index

DOCS = {
    "a.txt": "apple banana cherry apple python async await event loop python",
    "b.txt": "banana cherry date java event loop threads",
    "c.txt": "apple date egg python java the the the the the",
    "d.txt": "fig " + "the quick brown fox jumps over the lazy dog " * 40 + "python",
}

LAYOUTS: list[Layout] = ["dataclass", "slots", "array"]


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    for name, text in DOCS.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path


@pytest.fixture(params=LAYOUTS)
def layout(request: pytest.FixtureRequest) -> Layout:
    return request.param


@pytest.fixture
def index(corpus_dir: Path, layout: Layout) -> Index:
    """Index with positions in every storage layout (tests run once per layout)."""
    return build_index(iter_documents(corpus_dir), positions=True, layout=layout)


@pytest.fixture
def array_index(corpus_dir: Path) -> Index:
    return build_index(iter_documents(corpus_dir), positions=True, layout="array")


@pytest.fixture
def index_file(array_index: Index, tmp_path: Path) -> Iterator[Path]:
    from findex.store import save

    path = tmp_path / "idx.bin"
    save(array_index, path, "binary")
    yield path
