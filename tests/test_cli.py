import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex import __version__
from findex.cli import app

runner = CliRunner()


def test_help_lists_subcommands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("index", "search", "stats"):
        assert cmd in result.stdout


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_index_then_search_then_stats(corpus_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "i.bin"
    r = runner.invoke(app, ["index", str(corpus_dir), "--out", str(out), "--positions", "--executor", "serial"])
    assert r.exit_code == 0, r.output
    assert "indexed 4 documents" in r.stdout
    assert out.is_file()

    r = runner.invoke(app, ["search", str(out), "python NOT java", "-k", "5"])
    assert r.exit_code == 0, r.output
    assert "results for" in r.stdout

    r = runner.invoke(app, ["stats", str(out), "--top", "3"])
    assert r.exit_code == 0
    assert "documents:       4" in r.stdout
    assert "positions: True" in r.stdout


def test_search_json_goes_to_stdout_only(index_file: Path) -> None:
    r = runner.invoke(app, ["-vv", "search", str(index_file), '"event loop"', "--json"])
    assert r.exit_code == 0
    data = json.loads(r.stdout)  # stdout is pure JSON: diagnostics are on stderr
    assert {row["doc_id"] for row in data} == {0, 1}
    assert {"doc_id", "score", "title", "path", "snippet"} <= set(data[0])


def test_search_scorer_option(index_file: Path) -> None:
    for scorer in ("tfidf", "bm25"):
        r = runner.invoke(app, ["search", str(index_file), "python", "--scorer", scorer, "--json"])
        assert r.exit_code == 0
        assert len(json.loads(r.stdout)) == 3


def test_missing_index_is_one_line_error(tmp_path: Path) -> None:
    r = runner.invoke(app, ["search", str(tmp_path / "nope.bin"), "x"])
    assert r.exit_code == 1
    assert r.stderr.strip() == f"error: index file not found: {tmp_path / 'nope.bin'}"
    assert "Traceback" not in r.output


def test_bad_query_is_one_line_error(index_file: Path) -> None:
    r = runner.invoke(app, ["search", str(index_file), "(a OR"])
    assert r.exit_code == 1
    assert r.stderr.startswith("error: bad query")
    assert r.stderr.count("\n") == 1


def test_corrupt_index_is_clean_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"this is not an index")
    r = runner.invoke(app, ["stats", str(bad)])
    assert r.exit_code == 1
    assert r.stderr.startswith("error: cannot read index")


def test_missing_corpus(tmp_path: Path) -> None:
    r = runner.invoke(app, ["index", str(tmp_path / "missing"), "--out", str(tmp_path / "o.bin")])
    assert r.exit_code == 1
    assert "corpus not found" in r.stderr


def test_verbosity_levels_set_logging(index_file: Path, caplog: pytest.LogCaptureFixture) -> None:
    import logging

    from findex.cli import configure_logging

    configure_logging(0)
    assert logging.getLogger().level == logging.WARNING
    configure_logging(1)
    assert logging.getLogger().level == logging.INFO
    configure_logging(2)
    assert logging.getLogger().level == logging.DEBUG
    configure_logging(0)
