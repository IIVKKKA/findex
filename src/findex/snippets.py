"""Snippets: a window around the best match with highlighted terms."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator

_APOS = "'’ʼ"


def _pattern(terms: Iterable[str]) -> re.Pattern[str] | None:
    alts: list[str] = []
    for t in sorted(set(terms), key=len, reverse=True):
        alts.append("".join(f"[{_APOS}]" if ch == "'" else re.escape(ch) for ch in t))
    if not alts:
        return None
    return re.compile(rf"(?<!\w)(?:{'|'.join(alts)})(?!\w)", re.IGNORECASE)


def _is_word(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def _find_term(low: str, term: str) -> Iterator[tuple[int, int, str]]:
    """Whole-word occurrences of ``term`` in ``low`` using str.find (C speed, no regex)."""
    n = len(term)
    pos = low.find(term)
    while pos != -1:
        end = pos + n
        if (pos == 0 or not _is_word(low[pos - 1])) and (end == len(low) or not _is_word(low[end])):
            yield pos, end, term
        pos = low.find(term, pos + 1)


CHUNK = 1 << 16


def _iter_matches(text: str, terms: set[str], max_scan: int) -> Iterator[tuple[int, int, str]]:
    """(start, end, term) of every whole-word match, in text order, produced lazily.

    Fast path: lower-case 64 KB at a time and ``str.find`` each term (~GB/s), so the scan can stop
    early without lower-casing a multi-MB book. The case-insensitive alternation regex this replaces
    ran at ~25 MB/s (lab 8 profiling). Terms with apostrophes use the regex; so does any chunk
    whose lower() changed its length (rare Unicode special cases).
    """
    if any("'" in t for t in terms):
        pat = _pattern(terms)
        if pat is not None:
            for m in pat.finditer(text, 0, max_scan):
                yield m.start(), m.end(), m.group().casefold()
        return
    margin = max(len(t) for t in terms) + 1
    for base in range(0, min(len(text), max_scan), CHUNK):
        lo = max(0, base - 1)  # one char of left context for the word-boundary check
        seg_text = text[lo : base + CHUNK + margin]
        seg = seg_text.lower()
        if len(seg) != len(seg_text):
            pat = _pattern(terms)
            if pat is not None:
                for m in pat.finditer(text, base, max_scan):
                    yield m.start(), m.end(), m.group().casefold()
            return
        found = sorted(m for t in terms for m in _find_term(seg, t))
        for s, e, t in found:
            if base <= lo + s < base + CHUNK:  # each match belongs to exactly one chunk
                yield lo + s, lo + e, t


def make_snippet(
    text: str,
    terms: Iterable[str],
    width: int = 80,
    mark: tuple[str, str] = ("**", "**"),
    max_scan: int = 400_000,
) -> str:
    """Return ~``width`` chars either side of the best match, terms wrapped in ``mark``.

    The "best" window is the one that covers the most distinct query terms. Matches are
    streamed lazily; the scan stops as soon as a window covers *every* query term, or after
    ``max_scan`` characters (trade-off: a better window later in a huge book is not looked for).
    """
    terms = set(terms)
    if any(not t.isascii() for t in terms) and not unicodedata.is_normalized("NFC", text):
        text = unicodedata.normalize("NFC", text)
    target = len(terms)

    it = _iter_matches(text, terms, max_scan) if terms else iter(())
    matches: list[tuple[int, int, str]] = []
    done = False

    def fill(upto: int) -> None:
        nonlocal done
        while not done and (not matches or matches[-1][0] <= upto):
            nxt = next(it, None)
            if nxt is None:
                done = True
            else:
                matches.append(nxt)

    fill(-1)
    if not matches:
        return " ".join(text[: 2 * width].split())
    best_i, best_n = 0, -1
    seen: Counter[str] = Counter()
    lo_i = hi_i = 0
    i = 0
    while i < len(matches):
        start_i = matches[i][0]
        fill(start_i + width)
        while hi_i < len(matches) and matches[hi_i][0] <= start_i + width:
            seen[matches[hi_i][2]] += 1
            hi_i += 1
        while matches[lo_i][0] < start_i - width:
            key = matches[lo_i][2]
            seen[key] -= 1
            if not seen[key]:
                del seen[key]
            lo_i += 1
        if len(seen) > best_n:
            best_i, best_n = i, len(seen)
            if best_n >= target:
                break  # cannot do better than covering every query term
        i += 1
    c_start, c_end, _ = matches[best_i]
    lo = max(0, c_start - width)
    hi = min(len(text), c_end + width)
    window = text[lo:hi]
    pat = _pattern(terms)
    out = pat.sub(lambda m: f"{mark[0]}{m.group()}{mark[1]}", window) if pat else window
    out = " ".join(out.split())
    return ("..." if lo > 0 else "") + out + ("..." if hi < len(text) else "")
