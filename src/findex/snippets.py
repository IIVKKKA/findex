"""Snippets: a window around the best match with highlighted terms."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

_APOS = "'’ʼ"


def _pattern(terms: Iterable[str]) -> re.Pattern[str] | None:
    alts = []
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

    best_i, best_n = 0, -1
    for i, m in enumerate(matches):
        seen = {
            m2.group().casefold()
            for m2 in matches
            if abs(m2.start() - m.start()) <= width
        }
        if len(seen) > best_n:
            best_i, best_n = i, len(seen)
    centre = matches[best_i]
    lo = max(0, centre.start() - width)
    hi = min(len(text), centre.end() + width)
    window = text[lo:hi]
    out = pat.sub(lambda m: f"{mark[0]}{m.group()}{mark[1]}", window)
    out = " ".join(out.split())
    return ("..." if lo > 0 else "") + out + ("..." if hi < len(text) else "")
