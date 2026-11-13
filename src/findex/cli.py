"""The ``findex`` command: index / search / stats.

Diagnostics go through ``logging`` to stderr (configured once, in the callback
below); results go to stdout. Expected failures (missing index, bad query) print a
single ``error: ...`` line to stderr and exit with status 1, with no traceback.
"""

import json
import logging
import os
import pickle
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from . import __version__
from .corpus import Document, iter_documents, read_text
from .index import Index, Layout, build_index
from .parallel import ExecutorName, build_parallel
from .query import QueryError, parse
from .ranking import search as ranked_search
from .scoring import BM25, Scorer, TfIdf
from .snippets import make_snippet
from .store import Format, load, save

log = logging.getLogger("findex")
app = typer.Typer(help="findex - a small search engine.", no_args_is_help=True, add_completion=False)
err_console = Console(stderr=True)

ScorerName = Literal["tfidf", "bm25"]


def _fail(message: str) -> typer.Exit:
    """Print a one-line error to stderr and return an Exit(1) to raise."""
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code=1)


def configure_logging(verbosity: int) -> None:
    """Set up the root logger once: WARNING by default, -v INFO, -vv DEBUG."""
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(
        stream=sys.stderr,
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )


def _version(value: bool) -> None:
    if value:
        typer.echo(f"findex {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    verbose: Annotated[int, typer.Option("--verbose", "-v", count=True, help="-v info, -vv debug")] = 0,
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show version and exit.")
    ] = False,
) -> None:
    configure_logging(verbose)


def _load_index(path: Path) -> Index:
    if not path.is_file():
        raise _fail(f"index file not found: {path}")
    try:
        return load(path)
    except (OSError, EOFError, ValueError, pickle.UnpicklingError, AttributeError, ImportError) as exc:
        raise _fail(f"cannot read index {path}: {exc}") from exc


def _count_docs(root: Path) -> int | None:
    if root.is_dir():
        return sum(1 for _ in root.rglob("*.txt"))
    return None  # jsonl: unknown without a pass


def _with_progress(docs: Iterable[Document], total: int | None) -> Iterator[Document]:
    with Progress(
        TextColumn("[bold]indexing"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=err_console,
        transient=False,
    ) as progress:
        task = progress.add_task("docs", total=total)
        for doc in docs:
            yield doc
            progress.advance(task)


@app.command()
def index(
    root: Annotated[Path, typer.Argument(help="Directory of .txt files or a .jsonl file")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Where to write the index")] = Path("index.bin"),
    positions: Annotated[bool, typer.Option(help="Store token positions (needed for phrases)")] = False,
    layout: Annotated[Layout, typer.Option(help="Postings storage layout")] = "array",
    fmt: Annotated[Format, typer.Option("--format", help="File format")] = "binary",
    limit: Annotated[int | None, typer.Option(help="Only the first N documents")] = None,
    workers: Annotated[int, typer.Option(min=1, help="Parallel workers [default: CPU count]")] = os.cpu_count() or 1,
    executor: Annotated[
        ExecutorName, typer.Option(help="serial | threads | processes (fastest on a GIL build, see README)")
    ] = "processes",
) -> None:
    """Build an index from a corpus and save it."""
    if not root.exists():
        raise _fail(f"corpus not found: {root}")
    if executor != "serial" and root.is_dir() and limit is None:
        log.info("parallel build: executor=%s workers=%d", executor, workers)
        pidx = build_parallel(root, workers, executor, positions, layout)
        save(pidx, out, fmt)
        typer.echo(f"indexed {pidx.num_docs} documents, {len(pidx)} terms -> {out}")
        return
    total = _count_docs(root)
    docs: Iterable[Document] = iter_documents(root)
    if limit is not None:
        total = min(total, limit) if total is not None else limit
        docs = (d for i, d in enumerate(docs) if i < limit)
    log.info("building index from %s (positions=%s, layout=%s)", root, positions, layout)
    idx = build_index(_with_progress(docs, total), positions=positions, layout=layout)
    save(idx, out, fmt)
    log.info("saved %s", out)
    typer.echo(f"indexed {idx.num_docs} documents, {len(idx)} terms -> {out}")


@app.command()
def search(
    index_path: Annotated[Path, typer.Argument(help="Index file")],
    query: Annotated[str, typer.Argument(help='e.g. \'whale AND (harpoon OR ship) NOT "moby dick"\'')],
    scorer: Annotated[ScorerName, typer.Option(help="Ranking function")] = "bm25",
    top: Annotated[int, typer.Option("--top", "-k", min=1, help="Number of results")] = 10,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output on stdout")] = False,
) -> None:
    """Ranked search with snippets."""
    idx = _load_index(index_path)
    scorer_obj: Scorer = TfIdf() if scorer == "tfidf" else BM25()
    try:
        results = ranked_search(idx, query, scorer_obj, k=top)
        terms = parse(query).terms()
    except QueryError as exc:
        raise _fail(f"bad query: {exc}") from exc

    rows: list[dict[str, object]] = []
    for r in results:
        path = idx.doc_meta[r.doc_id].path
        text = read_text(Path(path)) if Path(path).is_file() else None
        snippet = make_snippet(text, terms) if text else ""
        rows.append(
            {"doc_id": r.doc_id, "score": round(r.score, 4), "title": r.title, "path": path, "snippet": snippet}
        )
    if as_json:
        typer.echo(json.dumps(rows, ensure_ascii=False, indent=2))
        return

    table = Table(title=f"{len(rows)} results for {query!r} ({scorer})", show_lines=True)
    table.add_column("#", justify="right")
    table.add_column("score", justify="right")
    table.add_column("title")
    table.add_column("snippet", overflow="fold")
    for n, row in enumerate(rows, 1):
        snippet = str(row["snippet"]).replace("**", "")
        table.add_row(str(n), f"{row['score']}", str(row["title"]), snippet)
    Console().print(table)


@app.command()
def stats(
    index_path: Annotated[Path, typer.Argument(help="Index file")],
    top: Annotated[int, typer.Option(help="How many most frequent terms to list")] = 10,
) -> None:
    """Show corpus and index statistics."""
    idx = _load_index(index_path)
    typer.echo(f"documents:       {idx.num_docs}")
    typer.echo(f"terms:           {len(idx)}")
    typer.echo(f"avg doc length:  {idx.avg_doc_length:.1f} tokens")
    typer.echo(f"layout:          {idx.layout}   positions: {idx.has_positions}")
    common = sorted(idx, key=idx.df, reverse=True)[:top]
    typer.echo("top terms by document frequency:")
    for term in common:
        typer.echo(f"  {term:<20} df={idx.df(term)}")


if __name__ == "__main__":
    app()
