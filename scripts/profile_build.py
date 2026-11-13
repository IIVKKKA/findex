"""Profile building the index of the first N books (serial). py-spy: see README."""

import sys
from itertools import islice
from pathlib import Path

from findex.corpus import iter_documents
from findex.index import build_index

n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
for _ in range(2):
    build_index(islice(iter_documents(Path("data/gutenberg")), n), layout="array")
