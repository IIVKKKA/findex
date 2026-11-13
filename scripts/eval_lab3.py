"""Lab 3 evaluation: sanity checks, P@5 for TF-IDF vs BM25, cache/timing evidence.

Usage: python scripts/eval_lab3.py index.bin > eval_lab3.md
Labels were written BEFORE running the scorers (doc_id -> data/gutenberg/pgN.txt).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from findex.index import Posting
from findex.query import parse
from findex.ranking import match_ids, search
from findex.scoring import BM25, TfIdf
from findex.snippets import make_snippet
from findex.store import open_index

# query -> doc_ids that SHOULD be in the top 5 (hand-labelled from knowing the books)
QUERIES: list[tuple[str, set[int]]] = [
    ("frankenstein OR monster", {74, 87}),                       # Frankenstein, The Monster Men
    ("paradise OR satan OR eden", {11, 17, 48}),                 # Paradise Lost x2, Paradise Regained
    ("internet OR email OR network", {30, 39, 65, 70, 46}),      # internet guides / EMAIL 101 / NREN
    ("census OR population", {20, 28, 5, 38}),                   # 1990 census x2, CIA factbooks
    ("prime OR primes OR mersenne", {55, 59}),                   # prime-number files
    ("lord OR god OR commandments", {0, 21, 8}),                 # KJV Bible x2, Book of Mormon
    ("guillotine OR paris OR revolution", {89, 50}),             # Tale of Two Cities, Scarlet Pimpernel
    ("slavery OR freedom OR slaves", {90, 14, 57}),              # Douglass x2, Black experience
    ("anne OR gilbert OR marilla", {35, 42}),                    # Anne of Green Gables / of the Island
    ("constitution OR union OR states", {9, 80}),                # Federalist, Lincoln inaugural
]


def p_at_5(results, relevant: set[int]) -> float:
    return sum(r.doc_id in relevant for r in results[:5]) / 5


def main(path: str) -> None:
    with open_index(Path(path)) as index:
        print(f"index: {index!r}, avg_doc_length={index.avg_doc_length:.0f}\n")

        # ---- Table 2: P@5 -------------------------------------------------------
        print("## Precision@5\n")
        print("| # | Query | Relevant docs | TF-IDF top 5 | TF-IDF P@5 | BM25 top 5 | BM25 P@5 |")
        print("|---|---|---|---|---|---|---|")
        sums = {"tfidf": 0.0, "bm25": 0.0}
        for n, (q, rel) in enumerate(QUERIES, 1):
            row = [f"{n}", f"`{q}`", ", ".join(map(str, sorted(rel)))]
            for name, scorer in (("tfidf", TfIdf()), ("bm25", BM25())):
                res = search(index, q, scorer, k=5)
                p = p_at_5(res, rel)
                sums[name] += p
                row += [", ".join(str(r.doc_id) for r in res), f"{p:.1f}"]
            print("| " + " | ".join(row) + " |")
        print(f"| | **Mean** | | | **{sums['tfidf'] / len(QUERIES):.2f}** | | **{sums['bm25'] / len(QUERIES):.2f}** |")

        # ---- Table 1: sanity checks ---------------------------------------------
        print("\n## Sanity checks (BM25 k1=1.5 b=0.75)\n")
        bm = BM25()
        # 1. rare beats common: pick one doc per term
        rare = "mersenne"
        common = "the"
        r_doc = index[rare][0]
        c_post = index[common][0]
        s_rare = bm.score(rare, r_doc, index)
        s_common = bm.score(common, c_post, index)
        print(f"1. rare vs common: score({rare!r}, doc {r_doc.doc_id}, tf={r_doc.tf}) = {s_rare:.3f}; "
              f"score({common!r}, doc {c_post.doc_id}, tf={c_post.tf}) = {s_common:.3f}")
        res = search(index, "the OR mersenne", bm, k=3)
        print(f"   query `the OR mersenne` top-3 doc_ids: {[r.doc_id for r in res]} "
              f"(df(mersenne)={index.df('mersenne')}, df(the)={index.df('the')})")
        # 2. saturation at average length
        some_doc = min(index.doc_lengths, key=lambda d: abs(index.doc_lengths[d] - index.avg_doc_length))
        print(f"2. saturation for 'whale' in doc {some_doc} (length {index.doc_length(some_doc)} ~ average):")
        prev = 0.0
        for tf in (1, 2, 5, 10, 20, 40):
            s = bm.score("whale", Posting(some_doc, tf), index)
            print(f"   tf={tf:>2}: {s:.3f}  (+{s - prev:.3f})")
            prev = s
        # 3. length normalisation
        short = min(index.doc_lengths, key=index.doc_lengths.__getitem__)
        long_ = max(index.doc_lengths, key=index.doc_lengths.__getitem__)
        s_s = bm.score("whale", Posting(short, 1), index)
        s_l = bm.score("whale", Posting(long_, 1), index)
        print(f"3. one occurrence of 'whale': shortest doc {short} ({index.doc_length(short)} tokens) = {s_s:.3f}; "
              f"longest doc {long_} ({index.doc_length(long_)} tokens) = {s_l:.3f}")

        # ---- Table 3: cache + timing logs ---------------------------------------
        print("\n## Cache and @timed evidence\n```")
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)
        match_ids.cache_clear()
        q = 'python OR "event loop" OR whale'
        search(index, q)
        search(index, q)
        logging.getLogger().removeHandler(handler)
        print(f"{match_ids.cache_info()}\n```")

        # ---- snippet example ----------------------------------------------------
        print("\n## Snippet example\n")
        q = '(whale OR harpoon) "white whale"'
        res = search(index, q, k=1)
        r = res[0]
        text = Path(index.doc_meta[r.doc_id].path).read_text(encoding="utf-8", errors="replace")
        print(f"Query `{q}` -> doc {r.doc_id} {r.title!r} score {r.score:.3f}\n")
        print("> " + make_snippet(text, parse(q).terms()))


if __name__ == "__main__":
    main(sys.argv[1])
