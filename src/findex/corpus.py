"""Lazy document stream: read corpus files one at a time."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

log = logging.getLogger(__name__)


class Document(NamedTuple):
    doc_id: int
    path: Path
    text: str


def _read_text(path: Path) -> str | None:
    """Read a file as UTF-8, falling back to cp1252; None if undecodable."""
    raw = path.read_bytes()
    for enc in ("utf-8", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    log.warning("cannot decode %s, skipping", path)
    return None


def _iter_txt(root: Path) -> Iterator[Document]:
    doc_id = 0
    for path in sorted(root.rglob("*.txt")):
        try:
            text = _read_text(path)
        except OSError as exc:
            log.warning("cannot read %s: %s", path, exc)
            continue
        if text is None:
            continue
        yield Document(doc_id, path, text)
        doc_id += 1


def _iter_jsonl(path: Path) -> Iterator[Document]:
    doc_id = 0
    with path.open(encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj["text"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                log.warning("%s:%d bad record: %s", path, lineno, exc)
                continue
            yield Document(doc_id, path, text)
            doc_id += 1


def iter_documents(root: Path) -> Iterator[Document]:
    """Yield documents one by one without loading the corpus into memory.

    ``root`` is either a directory tree of ``.txt`` files (found with
    ``rglob``) or a single JSON-lines file with a ``text`` field per line.
    doc_ids are assigned sequentially in sorted-path order.
    """
    root = Path(root)
    if root.is_file():
        yield from _iter_jsonl(root)
    else:
        yield from _iter_txt(root)
