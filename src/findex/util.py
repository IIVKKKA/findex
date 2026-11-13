"""Small shared helpers: the ``@timed`` decorator."""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable

timing_log = logging.getLogger("findex.timing")


def timed[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """Log the elapsed milliseconds of every call (also when it raises)."""

    @functools.wraps(fn)  # keeps __name__/__doc__, so logs and help() stay useful
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            ms = (time.perf_counter() - t0) * 1000
            timing_log.info("%s took %.2f ms", fn.__qualname__, ms)

    return wrapper
