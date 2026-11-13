import asyncio
import itertools
import json
import logging
import random
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import aclosing
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from findex.cli import app
from findex.corpus import iter_documents
from findex.crawler import (
    CrawlStats,
    Page,
    RobotsCache,
    backoff_delay,
    canonicalize,
    crawl,
    extract,
    fetch,
    parse_retry_after,
)
from findex.mirror import serve

Handler = Callable[[httpx.Request], httpx.Response | asyncio.Future[httpx.Response]]


def client_for(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True, max_redirects=5)


class FakeSleep:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


# ---- canonicalisation --------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("HTTP://Example.COM/a", "http://example.com/a"),
        ("http://example.com/a#section", "http://example.com/a"),
        ("http://example.com/a/", "http://example.com/a"),
        ("http://example.com", "http://example.com/"),
        ("http://example.com/", "http://example.com/"),
        ("http://example.com//a///b", "http://example.com/a/b"),
        ("http://example.com/p?b=2&a=1", "http://example.com/p?a=1&b=2"),
        ("http://example.com:80/x", "http://example.com/x"),
        ("https://example.com:443/x", "https://example.com/x"),
        ("http://example.com:8080/x", "http://example.com:8080/x"),
    ],
)
def test_canonicalize(raw: str, expected: str) -> None:
    assert canonicalize(raw) == expected


def test_canonicalize_is_idempotent() -> None:
    url = "HTTP://Ex.com:80/A//b/?z=1&y=2#frag"
    assert canonicalize(canonicalize(url)) == canonicalize(url)


# ---- Retry-After / backoff ---------------------------------------------------------------
def test_parse_retry_after() -> None:
    assert parse_retry_after("5") == 5.0
    assert parse_retry_after(None) is None
    assert parse_retry_after("soon") is None
    assert parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT", now=0) == pytest.approx(1445412480 - 0)
    assert parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0  # in the past


def test_backoff_is_jittered_and_capped() -> None:
    rng = random.Random(0)
    for attempt in range(10):
        d = backoff_delay(attempt, rng, base=0.5, cap=4.0)
        assert 0 <= d <= min(4.0, 0.5 * 2**attempt)


# ---- fetch ---------------------------------------------------------------------------------
def test_fetch_retries_429_and_honours_retry_after() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "7"})
        return httpx.Response(200, text="ok", headers={"content-type": "text/html"})

    sleep = FakeSleep()

    async def run() -> None:
        async with client_for(handler) as c:
            r = await fetch(c, "http://h/x", sleep=sleep, rng=random.Random(1))
        assert r.ok and r.attempts == 2 and r.body == "ok"

    asyncio.run(run())
    assert len(sleep.calls) == 1 and sleep.calls[0] >= 7.0


def test_fetch_gives_up_after_three_retries() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503)

    sleep = FakeSleep()

    async def run() -> None:
        async with client_for(handler) as c:
            r = await fetch(c, "http://h/x", sleep=sleep, rng=random.Random(1))
        assert r.status == 503 and not r.ok and r.attempts == 4

    asyncio.run(run())
    assert calls == 4 and len(sleep.calls) == 3  # 1 try + 3 retries, 3 sleeps


@pytest.mark.parametrize("status", [400, 401, 403, 404, 410])
def test_fetch_does_not_retry_other_4xx(status: int) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status)

    async def run() -> None:
        async with client_for(handler) as c:
            r = await fetch(c, "http://h/x", sleep=FakeSleep())
        assert r.status == status and r.attempts == 1

    asyncio.run(run())
    assert calls == 1


def test_fetch_timeout_is_retried_then_reported() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(1)
        return httpx.Response(200)

    sleep = FakeSleep()

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            r = await fetch(c, "http://h/x", timeout=0.02, sleep=sleep, rng=random.Random(1))
        assert r.status is None and "timeout" in r.error and r.attempts == 4

    asyncio.run(run())
    assert len(sleep.calls) == 3


def test_fetch_redirect_loop_is_not_retried() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        other = "/b" if request.url.path == "/a" else "/a"
        return httpx.Response(302, headers={"Location": other})

    async def run() -> None:
        async with client_for(handler) as c:
            r = await fetch(c, "http://h/a", sleep=FakeSleep())
        assert r.status is None and r.error == "redirect loop" and r.attempts == 1

    asyncio.run(run())


# ---- robots.txt ----------------------------------------------------------------------------
def test_robots_fetched_once_per_host_and_obeyed() -> None:
    robots_hits: dict[str, int] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            host = request.url.host
            robots_hits[host] = robots_hits.get(host, 0) + 1
            return httpx.Response(200, text="User-agent: *\nDisallow: /private/\n")
        return httpx.Response(200)

    async def run() -> None:
        async with client_for(handler) as c:
            rc = RobotsCache(c)
            urls = [f"http://a.test/page/{i}" for i in range(20)] + ["http://a.test/private/x", "http://b.test/private/y"]
            res = await asyncio.gather(*(rc.allowed(u) for u in urls))
            assert res[:20] == [True] * 20 and res[20] is False and res[21] is False
            assert rc.fetches == 2

    asyncio.run(run())
    assert robots_hits == {"a.test": 1, "b.test": 1}  # cached per host, even under concurrency


@pytest.mark.parametrize(("status", "allowed"), [(404, True), (410, True), (500, False), (403, False)])
def test_robots_status_policy(status: int, allowed: bool) -> None:
    async def run() -> bool:
        async with client_for(lambda r: httpx.Response(status)) as c:
            return await RobotsCache(c).allowed("http://h.test/page")

    assert asyncio.run(run()) is allowed


# ---- extraction -----------------------------------------------------------------------------
def test_extract_title_text_and_links() -> None:
    html = (
        "<html><head><title> My Title </title><style>p{x:y}</style></head><body><h1>Hello</h1>"
        '<script>var secret=1</script><p>some <b>bold</b> text</p><a href="/rel">r</a>'
        '<a href="mailto:a@b.c">m</a><a href="http://other.test/x">o</a></body></html>'
    )
    title, text, links = extract(html, "http://h.test/dir/page")
    assert title == "My Title"
    assert "secret" not in text and "p{x:y}" not in text
    assert "Hello" in text and "some bold text" in text
    assert links == ["http://h.test/rel", "http://other.test/x"]


# ---- crawl ----------------------------------------------------------------------------------
def make_site(n: int = 12, extra: dict[str, tuple[int, str]] | None = None) -> tuple[Callable[[httpx.Request], httpx.Response], dict[str, int]]:
    hits: dict[str, int] = {}
    extra = extra or {}

    def page(i: int) -> str:
        links = "".join(
            f'<a href="/p/{j}{suffix}">x</a>'
            for j, suffix in [((i + 1) % n, "#frag"), ((i + 2) % n, "/"), ((i + 3) % n, "")]
        )
        return (
            f"<html><title>Page {i}</title><body>text {i} {links}"
            '<a href="http://elsewhere.test/out">ext</a><a href="/private/secret">no</a></body></html>'
        )

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        hits[path] = hits.get(path, 0) + 1
        if path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /private/\n")
        if path in extra:
            status, body = extra[path]
            return httpx.Response(status, text=body, headers={"content-type": "text/html"})
        if path.startswith("/p/"):
            return httpx.Response(200, text=page(int(path.split("/")[2])), headers={"content-type": "text/html"})
        return httpx.Response(404)

    return handler, hits


async def collect(
    client: httpx.AsyncClient, start: str, **kw: float | bool | CrawlStats
) -> list[Page]:
    out: list[Page] = []
    async with aclosing(crawl([start], client, **kw)) as pages:  # type: ignore[arg-type]  # kwargs forwarded as-is
        async for p in pages:
            out.append(p)
    return out


def test_crawl_visits_each_canonical_url_once() -> None:
    handler, hits = make_site(12)

    async def run() -> list[Page]:
        async with client_for(handler) as c:
            return await collect(c, "http://site.test/p/0", max_pages=100, concurrency=4, per_host=4)

    pages = asyncio.run(run())
    assert len(pages) == 12
    assert len({p.url for p in pages}) == 12
    assert all(hits[f"/p/{i}"] + hits.get(f"/p/{i}/", 0) == 1 for i in range(12))  # #frag and / variants deduped
    assert "/private/secret" not in hits and "/out" not in hits  # robots + same-host rule
    assert hits["/robots.txt"] == 1


def test_crawl_stops_at_max_pages_and_cancels_workers() -> None:
    handler, _ = make_site(50)
    stats = CrawlStats()

    async def run() -> list[Page]:
        async with client_for(handler) as c:
            pages = await collect(c, "http://site.test/p/0", max_pages=7, concurrency=3, per_host=3, stats=stats)
        await asyncio.sleep(0)  # let cancelled tasks settle
        return pages

    pages = asyncio.run(run())
    assert len(pages) == 7 and stats.pages == 7
    assert stats.skipped_robots >= 0


def test_crawl_respects_global_and_per_host_limits() -> None:
    state = {"now": 0, "max": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        state["now"] += 1
        state["max"] = max(state["max"], state["now"])
        await asyncio.sleep(0.01)
        state["now"] -= 1
        i = int(request.url.path.split("/")[2])
        links = "".join(f'<a href="/p/{j}">x</a>' for j in range(i * 3 + 1, i * 3 + 4) if j < 40)
        return httpx.Response(200, text=f"<html><title>{i}</title><body>{links}</body></html>",
                              headers={"content-type": "text/html"})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            await collect(c, "http://site.test/p/0", max_pages=40, concurrency=10, per_host=3)

    asyncio.run(run())
    assert state["max"] <= 3  # the per-host semaphore is the tighter bound
    assert state["max"] >= 2  # ... and we really did run concurrently


def test_crawl_min_delay_between_requests_to_one_host() -> None:
    times: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        times.append(time.perf_counter())
        i = int(request.url.path.split("/")[2])
        links = "".join(f'<a href="/p/{j}">x</a>' for j in range(i + 1, i + 3))
        return httpx.Response(200, text=f"<html><title>{i}</title><body>{links}</body></html>",
                              headers={"content-type": "text/html"})

    async def run() -> None:
        async with client_for(handler) as c:
            await collect(c, "http://site.test/p/0", max_pages=6, concurrency=6, per_host=6, min_delay=0.05)

    asyncio.run(run())
    gaps = [b - a for a, b in itertools.pairwise(times)]
    assert gaps and min(gaps) >= 0.04  # ~50 ms apart despite 6 concurrent slots


def test_crawl_does_not_block_the_event_loop(caplog: pytest.LogCaptureFixture) -> None:
    handler, _ = make_site(30)

    async def run() -> None:
        async with client_for(handler) as c:
            await collect(c, "http://site.test/p/0", max_pages=30, concurrency=5, per_host=5)

    with caplog.at_level(logging.WARNING, logger="asyncio"):
        asyncio.run(run(), debug=True)  # debug mode logs "Executing <Task...> took N seconds" for slow steps
    assert not [r for r in caplog.records if "took" in r.getMessage()]


# ---- end to end against the local mirror, through the CLI ----------------------------------------------
@pytest.fixture
def mirror_url() -> Iterator[str]:
    srv = serve(port=0, n_pages=30, latency=0.0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/"
    srv.shutdown()
    srv.server_close()


def test_cli_crawl_then_index_then_search(mirror_url: str, tmp_path: Path) -> None:
    runner = CliRunner()
    out, log = tmp_path / "pages.jsonl", tmp_path / "crawl.log"
    r = runner.invoke(app, ["crawl", mirror_url, "-o", str(out), "--max-pages", "15", "--concurrency", "4",
                            "--log-file", str(log)])
    assert r.exit_code == 0, r.output
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 15 and all(row["status"] == 200 for row in rows)
    text = log.read_text(encoding="utf-8")
    assert "200 http://127.0.0.1" in text
    assert "/private/" not in "".join(row["url"] for row in rows)

    # lab 1's iter_documents reads the JSONL; the event loop is long gone by the time we index
    assert len(list(iter_documents(out))) == 15
    idx_path = tmp_path / "i.bin"
    assert runner.invoke(app, ["index", str(out), "-o", str(idx_path)]).exit_code == 0
    r = runner.invoke(app, ["search", str(idx_path), "synthetic", "--json"])
    assert r.exit_code == 0 and json.loads(r.stdout)
