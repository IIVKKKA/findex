# pyright: basic
# (pytest-benchmark ships no type information; the rest of the test-suite stays strict.)
"""pytest-benchmark: CI prints these timings on every push, so a slowdown is visible at once.

Run only these:  uv run pytest tests/test_benchmarks.py --benchmark-only
Compare runs:    --benchmark-save=base   then   --benchmark-compare=base --benchmark-compare-fail=mean:25%
"""

from pathlib import Path

import pytest
from pytest_benchmark.fixture import BenchmarkFixture  # pyright: ignore[reportMissingTypeStubs]

from findex.corpus import Document
from findex.index import Index, build_index
from findex.modes import ranked_hits
from findex.npsearch import NumpyIndex
from findex.ranking import match_ids, search
from findex.scoring import BM25

WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau"]


@pytest.fixture(scope="module")
def docs() -> list[Document]:
    out: list[Document] = []
    for i in range(60):
        text = " ".join(WORDS[(i * 7 + j * 3) % len(WORDS)] + (str(j % 13) if j % 5 == 0 else "") for j in range(1500))
        out.append(Document(i, Path(f"d{i}.txt"), text))
    return out


@pytest.fixture(scope="module")
def index(docs: list[Document]) -> Index:
    return build_index(docs, layout="array")


def test_benchmark_build_index(benchmark: BenchmarkFixture, docs: list[Document]) -> None:
    result = benchmark(build_index, docs, layout="array")
    assert result.num_docs == 60


def test_benchmark_search_lab3_scorer(benchmark: BenchmarkFixture, index: Index) -> None:
    def run() -> int:
        match_ids.cache_clear()
        return len(search(index, "alpha OR beta OR gamma", BM25(), k=10))

    assert benchmark(run) == 10


def test_benchmark_search_numpy(benchmark: BenchmarkFixture, index: Index) -> None:
    ni = NumpyIndex(index)
    assert len(benchmark(ni.search, "alpha OR beta OR gamma", "bm25", 10)) == 10


def test_benchmark_ranked_hits_with_snippets(benchmark: BenchmarkFixture, index: Index) -> None:
    hits, _ = benchmark(ranked_hits, index, "alpha OR beta", "bm25", "lexical", 10, 1)
    assert len(hits) == 10
