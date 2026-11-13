"""Streaming tokenizer.

Policy
------
1. NFC normalization (``unicodedata.normalize``) so ``e`` + U+0301 == ``é``.
2. ``str.casefold()`` (not ``lower()``): ``Straße`` -> ``strasse``.
3. ``re.finditer`` over ``WORD_RE`` (lazy; never builds a list of matches).

``WORD_RE`` = letters/digits/underscore-free words (``[^\\W_]+``) optionally
joined by a single inner connector:

* Apostrophes (``'``, ``’``, ``ʼ``) inside a word are kept, and normalised to
  ``'``: ``don't`` -> ``don't``, ``п'ять`` -> ``п'ять``. Leading/trailing
  apostrophes (quotes) are dropped.
* Hyphens inside a word are kept: ``well-known`` -> ``well-known``. A
  leading/trailing hyphen or a dash with spaces is a separator.
* Digits are part of words: ``abc123``, ``3.14`` is split into ``3`` and
  ``14`` (the dot is not a connector); ``1990-2000`` stays one token.
* Underscore is a separator.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator

_APOS = "'’ʼ"
WORD_RE = re.compile(r"[^\W_]+(?:['-][^\W_]+)*")  # text was already mapped to the ASCII apostrophe
_APOS_TABLE = str.maketrans({"’": "'", "ʼ": "'"})


def tokenize(text: str) -> Iterator[str]:
    """Yield normalized tokens from ``text`` lazily."""
    # apostrophes are unified once per text (a per-token translate() cost ~27 % of indexing time)
    text = unicodedata.normalize("NFC", text).casefold().translate(_APOS_TABLE)
    for m in WORD_RE.finditer(text):
        yield m.group()
