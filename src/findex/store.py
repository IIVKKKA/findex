"""Persist and reload an Index: pickle, plus a binary array-based format.

Binary layout (little-endian native ``array`` bytes)::

    b"FIDX1" | u32 header_len | JSON header | doc_ids array of doc_lengths |
    for each term (header order): doc_ids | tfs | [positions]

The header holds doc_meta, the term list with posting counts, and flags.
The binary format always loads into the ``array`` layout.
"""

from __future__ import annotations

import json
import pickle
import struct
from array import array
from pathlib import Path
from typing import BinaryIO, Literal

from .index import ArrayPostings, DocMeta, Index, Posting

MAGIC = b"FIDX1"
Format = Literal["pickle", "binary"]


def save(index: Index, path: Path, fmt: Format = "pickle") -> None:
    if fmt == "pickle":
        with open(path, "wb") as fh:
            pickle.dump(index, fh, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        _save_binary(index, path)


def load(path: Path) -> Index:
    """Load an index; the format is detected from the first bytes.

    SECURITY: ``pickle.load`` can execute arbitrary code embedded in the
    file (a pickle may call any importable callable through ``__reduce__``).
    Never load an index file from an untrusted source - use the binary
    format, which only reads numbers and JSON, for files you did not make.
    """
    with open(path, "rb") as fh:
        head = fh.read(len(MAGIC))
        fh.seek(0)
        if head == MAGIC:
            return _load_binary(fh)
        return pickle.load(fh)


def _as_arrays(p) -> tuple[array, array, array]:
    if isinstance(p, ArrayPostings):
        return p.doc_ids, p.tfs, p.positions if p.positions is not None else array("I")
    ids = array("I", (x.doc_id for x in p))
    tfs = array("I", (x.tf for x in p))
    pos = array("I")
    for x in p:
        pos.extend(x.positions)
    return ids, tfs, pos


def _save_binary(index: Index, path: Path) -> None:
    terms = list(index.postings)
    header = {
        "positions": index.has_positions,
        "doc_meta": {str(i): [m.path, m.title] for i, m in index.doc_meta.items()},
        "terms": [[t, index.df(t)] for t in terms],
    }
    hb = json.dumps(header, ensure_ascii=False).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(MAGIC + struct.pack("<I", len(hb)) + hb)
        ids = sorted(index.doc_lengths)
        array("I", ids).tofile(fh)
        array("I", (index.doc_lengths[i] for i in ids)).tofile(fh)
        for t in terms:
            d, tf, pos = _as_arrays(index.postings[t])
            d.tofile(fh)
            tf.tofile(fh)
            if index.has_positions:
                pos.tofile(fh)


def _read_array(fh: BinaryIO, n: int) -> array:
    a = array("I")
    if n:
        a.fromfile(fh, n)
    return a


def _load_binary(fh: BinaryIO) -> Index:
    fh.read(len(MAGIC))
    (hlen,) = struct.unpack("<I", fh.read(4))
    header = json.loads(fh.read(hlen).decode("utf-8"))
    meta = {int(i): DocMeta(*v) for i, v in header["doc_meta"].items()}
    n_docs = len(meta)
    ids = _read_array(fh, n_docs)
    lens = _read_array(fh, n_docs)
    doc_lengths = dict(zip(ids, lens, strict=True))
    has_pos = header["positions"]
    postings: dict[str, ArrayPostings] = {}
    for term, n in header["terms"]:
        d = _read_array(fh, n)
        tf = _read_array(fh, n)
        pos = _read_array(fh, sum(tf)) if has_pos else None
        postings[term] = ArrayPostings(d, tf, pos)
    return Index(postings, doc_lengths, meta, "array", has_pos)


def postings_as_objects(index: Index, term: str) -> list[Posting]:
    """Materialise Posting objects for one term (any layout)."""
    p = index.postings.get(term)
    if p is None:
        return []
    if not isinstance(p, ArrayPostings):
        return [Posting(x.doc_id, x.tf, x.positions) for x in p]
    out, off = [], 0
    for i, d in enumerate(p.doc_ids):
        tf = p.tfs[i]
        has = index.has_positions and p.positions is not None
        pos = tuple(p.positions[off : off + tf]) if has else ()  # type: ignore[index]
        off += tf if has else 0
        out.append(Posting(d, tf, pos))
    return out
