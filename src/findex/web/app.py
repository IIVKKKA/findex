"""FastAPI application: JSON API under /search, /docs/{id}, /stats, /health and an HTML UI at /.

* ``lifespan`` loads the index once (from ``Settings.index_path``) and releases it on shutdown.
* Search routes are plain ``def``: the work is CPU-bound, so FastAPI runs them in a thread pool
  and the event loop keeps serving /health etc. (measured against ``async def`` in the README).
* Pydantic models appear only at the boundary (``models.py``); inside we use the lab 2-3 dataclasses.
* Any unexpected exception is logged with a request id; the client gets a 500 with no stack trace.
"""

import html
import logging
import math
import re
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from markupsafe import Markup

from .. import __version__
from ..corpus import read_text
from ..index import Index
from ..query import QueryError, parse
from ..ranking import match_ids
from ..ranking import search as ranked_search
from ..scoring import BM25, Scorer, TfIdf
from ..snippets import make_snippet
from ..store import load
from .models import DocOut, ErrorOut, HealthOut, HitOut, ScorerName, SearchOut, StatsOut
from .settings import Settings, get_settings

log = logging.getLogger("findex.web")
TEMPLATES = Jinja2Templates(directory=Path(__file__).parent / "templates")


def _mark(text: str) -> Markup:
    """Escape, then turn **x** (our snippet highlight) into <mark>x</mark>."""
    return Markup(re.sub(r"\*\*(.+?)\*\*", r"<mark>\1</mark>", html.escape(text)))


TEMPLATES.env.filters["mark"] = _mark


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    settings: Settings = app.state.settings
    app.state.index = None
    if settings.index_path.is_file():
        t0 = time.perf_counter()
        app.state.index = await run_in_threadpool(load, settings.index_path)  # keep the loop responsive
        idx: Index = app.state.index
        log.info("loaded %s: %d docs, %d terms in %.2fs", settings.index_path, idx.num_docs, len(idx),
                 time.perf_counter() - t0)
    else:
        log.error("index file %s not found; /health will report 503", settings.index_path)
    yield
    if app.state.index is not None:
        app.state.index.close()


def get_index(request: Request) -> Index:
    index: Index | None = request.app.state.index
    if index is None:
        raise HTTPException(status_code=503, detail="index not loaded yet")
    return index


IndexDep = Annotated[Index, Depends(get_index)]


def _scorer(name: ScorerName) -> Scorer:
    return TfIdf() if name == "tfidf" else BM25()


def _doc_text(index: Index, doc_id: int) -> str:
    path = Path(index.doc_meta[doc_id].path)
    return (read_text(path) or "") if path.is_file() else ""


def run_search(index: Index, q: str, scorer: ScorerName, k: int, page: int) -> SearchOut:
    try:
        tree = parse(q)
        total = len(match_ids(index, q))
        top = ranked_search(index, q, _scorer(scorer), k=page * k)
    except QueryError as exc:
        raise HTTPException(status_code=422, detail=f"bad query: {exc}") from exc
    terms = tree.terms()
    hits = [
        HitOut(doc_id=r.doc_id, score=round(r.score, 4), title=r.title,
               snippet=make_snippet(_doc_text(index, r.doc_id), terms))
        for r in top[(page - 1) * k : page * k]
    ]
    return SearchOut(query=q, scorer=scorer, page=page, per_page=k, total=total,
                     pages=max(1, math.ceil(total / k)), results=hits)


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(
        title="findex", version=__version__, lifespan=lifespan,
        docs_url="/api/docs", openapi_url="/api/openapi.json", redoc_url=None,
    )
    app.state.settings = settings or get_settings()
    app.state.index = None
    logging.basicConfig(level=app.state.settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    @app.middleware("http")
    async def request_id_and_errors(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = rid
        try:
            response = await call_next(request)
        except Exception:
            log.exception("unhandled error on %s %s [request_id=%s]", request.method, request.url.path, rid)
            response = JSONResponse(ErrorOut(detail="Internal Server Error", request_id=rid).model_dump(),
                                    status_code=500)
        response.headers["X-Request-ID"] = rid
        return response

    # ---- JSON API -------------------------------------------------------------------
    @app.get("/search", response_model=SearchOut, responses={422: {"model": ErrorOut}, 503: {"model": ErrorOut}})
    def search(
        index: IndexDep,
        q: Annotated[str, Query(min_length=1, max_length=200, description="Query: AND/OR/NOT, (), quoted phrases")],
        k: Annotated[int, Query(ge=1, le=50, description="Results per page")] = 10,
        scorer: ScorerName = "bm25",
        page: Annotated[int, Query(ge=1, le=100)] = 1,
    ) -> SearchOut:
        return run_search(index, q, scorer, k, page)

    @app.get("/docs/{doc_id}", response_model=DocOut, responses={404: {"model": ErrorOut}})
    def get_doc(doc_id: int, index: IndexDep) -> DocOut:
        if doc_id not in index.doc_meta:
            raise HTTPException(status_code=404, detail=f"no document {doc_id}")
        meta = index.doc_meta[doc_id]
        return DocOut(doc_id=doc_id, title=meta.title, path=meta.path, length=index.doc_length(doc_id),
                      text=_doc_text(index, doc_id))

    @app.get("/stats", response_model=StatsOut)
    def stats(index: IndexDep) -> StatsOut:
        return StatsOut(documents=index.num_docs, terms=len(index), avg_doc_length=round(index.avg_doc_length, 1),
                        layout=index.layout, positions=index.has_positions)

    @app.get("/health", response_model=HealthOut, responses={503: {"model": HealthOut}})
    def health(request: Request, response: Response) -> HealthOut:
        index: Index | None = request.app.state.index
        if index is None:
            response.status_code = 503
            return HealthOut(status="loading")
        return HealthOut(status="ok", documents=index.num_docs)

    # ---- HTML UI --------------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def home(
        request: Request,
        index: IndexDep,
        q: Annotated[str, Query(max_length=200)] = "",
        scorer: ScorerName = "bm25",
        page: Annotated[int, Query(ge=1, le=100)] = 1,
    ) -> HTMLResponse:
        result: SearchOut | None = None
        error = ""
        if q.strip():
            try:
                result = run_search(index, q, scorer, 10, page)
            except HTTPException as exc:
                error = str(exc.detail)
        ctx = {"q": q, "scorer": scorer, "result": result, "error": error, "docs": index.num_docs,
               "terms": len(index), "version": __version__}
        return TEMPLATES.TemplateResponse(request, "index.html", ctx, status_code=422 if error else 200)

    @app.get("/view/{doc_id}", response_class=HTMLResponse)
    def view_doc(request: Request, doc_id: int, index: IndexDep, q: str = "") -> HTMLResponse:
        if doc_id not in index.doc_meta:
            raise HTTPException(status_code=404, detail=f"no document {doc_id}")
        meta = index.doc_meta[doc_id]
        text = _doc_text(index, doc_id)
        shown = text[:60_000]
        terms: set[str] = set()
        try:
            terms = parse(q).terms() if q.strip() else set()
        except QueryError:
            terms = set()
        body = make_snippet(shown, terms, width=len(shown)) if terms and shown else " ".join(shown.split())
        ctx = {"doc_id": doc_id, "title": meta.title, "length": index.doc_length(doc_id), "body": body,
               "truncated": len(text) > len(shown), "q": q, "version": __version__}
        return TEMPLATES.TemplateResponse(request, "doc.html", ctx)

    return app
