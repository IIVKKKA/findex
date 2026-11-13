"""Query language: recursive-descent parser producing a tree of nodes.

Grammar::

    or_expr  -> and_expr ('OR' and_expr)*
    and_expr -> not_expr ('AND'? not_expr)*        # adjacent terms = implicit AND
    not_expr -> 'NOT' atom | atom
    atom     -> WORD | "quoted phrase" | '(' or_expr ')'

``AND`` binds tighter than ``OR``, so ``a OR b c`` parses to
``Or(a, And(b, c))``. Words are normalised with the same ``tokenize`` as the
index, so query terms match index terms.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import TYPE_CHECKING

from .tokenize import tokenize

if TYPE_CHECKING:
    from .index import Index


class QueryError(ValueError):
    """Syntax error in a query string."""


class Node:
    """Base class: operator overloads build trees, subclasses evaluate."""

    def evaluate(self, index: Index) -> set[int]:
        raise NotImplementedError

    def terms(self) -> set[str]:
        """Terms that contribute to ranking (everything outside NOT)."""
        raise NotImplementedError

    def __and__(self, other: Node) -> And:
        return And(*_flat(self, other, And))

    def __or__(self, other: Node) -> Or:
        return Or(*_flat(self, other, Or))

    def __invert__(self) -> Not:
        return Not(self)


def _flat(a: Node, b: Node, kind: type) -> list[Node]:
    out: list[Node] = []
    for n in (a, b):
        out.extend(n.children if isinstance(n, kind) else [n])  # type: ignore[attr-defined]
    return out


class Term(Node):
    def __init__(self, text: str) -> None:
        self.text = text

    def evaluate(self, index: Index) -> set[int]:
        return set(index.doc_ids(self.text))

    def terms(self) -> set[str]:
        return {self.text}

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Term) and self.text == other.text

    def __hash__(self) -> int:
        return hash(("Term", self.text))

    def __repr__(self) -> str:
        return f"Term({self.text!r})"


class Phrase(Node):
    def __init__(self, *words: str) -> None:
        self.words = tuple(words)

    def evaluate(self, index: Index) -> set[int]:
        if not index.has_positions:
            raise QueryError("phrase queries need an index built with --positions")
        if not self.words:
            return set()
        candidates = set(index.doc_ids(self.words[0]))
        for w in self.words[1:]:
            candidates &= set(index.doc_ids(w))
        if not candidates:
            return set()
        per_word = [{p.doc_id: p.positions for p in index[w] if p.doc_id in candidates} for w in self.words]
        hits: set[int] = set()
        for doc in candidates:
            starts = set(per_word[0][doc])
            for offset, pos in enumerate(per_word[1:], 1):
                starts &= {p - offset for p in pos[doc]}
                if not starts:
                    break
            if starts:
                hits.add(doc)
        return hits

    def terms(self) -> set[str]:
        return set(self.words)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Phrase) and self.words == other.words

    def __hash__(self) -> int:
        return hash(("Phrase", self.words))

    def __repr__(self) -> str:
        return f"Phrase{self.words!r}"


class _NAry(Node):
    def __init__(self, *children: Node) -> None:
        self.children = tuple(children)

    def terms(self) -> set[str]:
        out: set[str] = set()
        for c in self.children:
            out |= c.terms()
        return out

    def __eq__(self, other: object) -> bool:
        return type(other) is type(self) and self.children == other.children  # type: ignore[attr-defined]

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.children))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(map(repr, self.children))})"


class And(_NAry):
    def evaluate(self, index: Index) -> set[int]:
        sets = sorted((c.evaluate(index) for c in self.children), key=len)
        out = sets[0]
        for s in sets[1:]:
            out = out & s
        return out


class Or(_NAry):
    def evaluate(self, index: Index) -> set[int]:
        out: set[int] = set()
        for c in self.children:
            out |= c.evaluate(index)
        return out


class Not(Node):
    def __init__(self, child: Node) -> None:
        self.child = child

    def evaluate(self, index: Index) -> set[int]:
        return set(range(index.num_docs)) - self.child.evaluate(index)

    def terms(self) -> set[str]:
        return set()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Not) and self.child == other.child

    def __hash__(self) -> int:
        return hash(("Not", self.child))

    def __repr__(self) -> str:
        return f"Not({self.child!r})"


# ---- lexer + parser ------------------------------------------------------------
_TOKEN_RE = re.compile(r'\s*(?:(?P<paren>[()])|"(?P<phrase>[^"]*)"|(?P<word>[^\s()"]+))')


def _lex(query: str) -> Iterator[tuple[str, str]]:
    pos = 0
    query = query.rstrip()
    while pos < len(query):
        m = _TOKEN_RE.match(query, pos)
        if m is None:
            raise QueryError(f"unbalanced quote near position {pos}")
        pos = m.end()
        if m.group("paren"):
            yield "paren", m.group("paren")
        elif m.group("phrase") is not None:
            yield "phrase", m.group("phrase")
        else:
            yield "word", m.group("word")


class _Parser:
    def __init__(self, query: str) -> None:
        self.toks = list(_lex(query))
        self.i = 0

    def peek(self) -> tuple[str, str] | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self) -> tuple[str, str]:
        tok = self.toks[self.i]
        self.i += 1
        return tok

    def is_kw(self, kw: str) -> bool:
        t = self.peek()
        return t is not None and t == ("word", kw)

    def parse(self) -> Node:
        if not self.toks:
            raise QueryError("empty query")
        node = self.or_expr()
        if self.peek() is not None:
            raise QueryError(f"unexpected {self.peek()[1]!r}")  # type: ignore[index]
        return node

    def or_expr(self) -> Node:
        parts = [self.and_expr()]
        while self.is_kw("OR"):
            self.take()
            parts.append(self.and_expr())
        return parts[0] if len(parts) == 1 else Or(*parts)

    def and_expr(self) -> Node:
        parts = [self.not_expr()]
        while True:
            if self.is_kw("AND"):
                self.take()
                parts.append(self.not_expr())
            elif self.peek() is not None and not self.is_kw("OR") and self.peek() != ("paren", ")"):
                parts.append(self.not_expr())
            else:
                break
        return parts[0] if len(parts) == 1 else And(*parts)

    def not_expr(self) -> Node:
        if self.is_kw("NOT"):
            self.take()
            return Not(self.atom())
        return self.atom()

    def atom(self) -> Node:
        t = self.peek()
        if t is None:
            raise QueryError("query ends unexpectedly")
        kind, val = self.take()
        if kind == "paren":
            if val == ")":
                raise QueryError("unexpected ')'")
            node = self.or_expr()
            if self.peek() != ("paren", ")"):
                raise QueryError("missing ')'")
            self.take()
            return node
        if kind == "phrase":
            words = tuple(tokenize(val))
            if not words:
                raise QueryError("empty phrase")
            return Phrase(*words)
        if val in ("AND", "OR"):
            raise QueryError(f"operator {val} needs an operand")
        toks = list(tokenize(val))
        if not toks:
            raise QueryError(f"no searchable term in {val!r}")
        return Term(toks[0]) if len(toks) == 1 else And(*(Term(x) for x in toks))


def parse(query: str) -> Node:
    """Parse ``query`` into a tree of nodes (raises ``QueryError``)."""
    return _Parser(query).parse()
