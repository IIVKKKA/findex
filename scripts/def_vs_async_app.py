"""The real findex app plus two experimental twin routes, used only by bench_lab7.py.

/s_def   - plain ``def``  : FastAPI runs it in the thread pool, the event loop stays free
/s_async - ``async def``  : the CPU-bound search runs ON the event loop and blocks everything
"""

from typing import Annotated

from fastapi import Query

from findex.web.app import IndexDep, create_app, run_search
from findex.web.models import SearchOut

app = create_app()


@app.get("/s_def", response_model=SearchOut)
def s_def(index: IndexDep, q: Annotated[str, Query()]) -> SearchOut:
    return run_search(index, q, "bm25", 10, 1)


@app.get("/s_async", response_model=SearchOut)
async def s_async(index: IndexDep, q: Annotated[str, Query()]) -> SearchOut:
    return run_search(index, q, "bm25", 10, 1)
