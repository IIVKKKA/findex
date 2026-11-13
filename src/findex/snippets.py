"""Snippets: a window around the best match with highlighted terms."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable

_APOS = "'’ʼ"


def _pattern(terms: Iterable[str]) -> re.Pattern[str] | None:
    alts: list[str] = []
    for t in sorted(set(terms), key=len, reverse=True):
        alts.append("".join(f"[{_APOS}]" if ch == "'" else re.escape(ch) for ch in t))
    if not alts:
        return None
    return re.compile(rf"(?<!\w)(?:{'|'.join(alts)})(?!\w)", re.IGNORECASE)


def make_snippet(
    text: str, terms: Iterable[str], width: int = 80, mark: tuple[str, str] = ("**", "**")
) -> str:
    """Return ~``width`` chars either side of the best match, terms wrapped in ``mark``.

    The "best" window is the one that covers the most distinct query terms.
    """
    text = unicodedata.normalize("NFC", text)
    pat = _pattern(terms)
    if pat is None:
        return " ".join(text[: 2 * width].split())
    matches = list(pat.finditer(text))
    if not matches:
        return " ".join(text[: 2 * width].split())

    # Sliding window over the (sorted) matches: O(M) instead of comparing every pair (O(M^2) was
    # a real problem on long books, found while load-testing lab 7).
    best_i, best_n = 0, -1
    seen: Counter[str] = Counter()
    lo_i = hi_i = 0
    for i, m in enumerate(matches):
        while hi_i < len(matches) and matches[hi_i].start() <= m.start() + width:
            seen[matches[hi_i].group().casefold()] += 1
            hi_i += 1
        while matches[lo_i].start() < m.start() - width:
            key = matches[lo_i].group().casefold()
            seen[key] -= 1
            if not seen[key]:
                del seen[key]
            lo_i += 1
        if len(seen) > best_n:
            best_i, best_n = i, len(seen)
    centre = matches[best_i]
    lo = max(0, centre.start() - width)
    hi = min(len(text), centre.end() + width)
    window = text[lo:hi]
    out = pat.sub(lambda m: f"{mark[0]}{m.group()}{mark[1]}", window)
    out = " ".join(out.split())
    return ("..." if lo > 0 else "") + out + ("..." if hi < len(text) else "")
