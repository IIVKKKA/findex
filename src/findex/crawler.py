"""Polite asyncio crawler: robots.txt, retries with backoff, per-host limits.

Only crawl what you are allowed to: your own site, documentation whose robots.txt
permits it, or a local mirror. The User-Agent is honest and carries a contact URL.
"""

import asyncio
import email.utils
import logging
import random
import time
import urllib.robotparser
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import (
    parse_qsl,
    quote,
    unquote,
    urldefrag,
    urljoin,
    urlsplit,
    urlunsplit,
)

import httpx
from selectolax import lexbor as _selectolax

log = logging.getLogger("findex.crawl")
HTMLParser: Any = _selectolax.LexborHTMLParser  # selectolax ships no type stubs

USER_AGENT = "findex-crawler/0.6 (+https://github.com/IIVKKKA/findex; educational project)"
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_RETRIES = 3


# --- URL canonicalisation -----------------------------------------------------------
def canonicalize(url: str) -> str:
    """One spelling per page: lower-case scheme/host, no fragment, no default port,
    no trailing slash (except the root), sorted query parameters."""
    url, _ = urldefrag(url)
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    port = parts.port
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        host = f"{host}:{port}"
    path = quote(unquote(parts.path), safe="/:@!$&'()*+,;=-._~")
    while "//" in path:
        path = path.replace("//", "/")
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    if not path:
        path = "/"
    query = "&".join(f"{k}={v}" for k, v in sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return urlunsplit((scheme, host, path, query, ""))


def host_of(url: str) -> str:
    return urlsplit(url).netloc


# --- robots.txt -----------------------------------------------------------------------
class RobotsCache:
    """robots.txt per host, fetched once and cached.

    404/410 -> everything allowed; 5xx or network failure -> nothing allowed (conservative).
    """

    def __init__(self, client: httpx.AsyncClient, user_agent: str = USER_AGENT) -> None:
        self._client = client
        self._ua = user_agent
        self._parsers: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self.fetches = 0

    async def _load(self, base: str) -> urllib.robotparser.RobotFileParser | None:
        self.fetches += 1
        rp = urllib.robotparser.RobotFileParser()
        try:
            resp = await self._client.get(f"{base}/robots.txt", headers={"User-Agent": self._ua})
        except httpx.HTTPError as exc:
            log.warning("robots.txt for %s unreachable (%s): treating as disallow-all", base, exc)
            return None
        if resp.status_code in (401, 403) or resp.status_code >= 500:
            log.warning("robots.txt for %s -> %d: treating as disallow-all", base, resp.status_code)
            return None
        if resp.status_code >= 400:
            rp.parse([])  # no robots.txt: allow everything
        else:
            rp.parse(resp.text.splitlines())
        return rp

    async def _parser(self, url: str) -> urllib.robotparser.RobotFileParser | None:
        parts = urlsplit(url)
        base = f"{parts.scheme}://{parts.netloc}"
        lock = self._locks.setdefault(base, asyncio.Lock())
        async with lock:  # concurrent first requests to one host fetch robots.txt once
            if base not in self._parsers:
                self._parsers[base] = await self._load(base)
            return self._parsers[base]

    async def allowed(self, url: str) -> bool:
        rp = await self._parser(url)
        return rp is not None and rp.can_fetch(self._ua, url)

    async def crawl_delay(self, url: str) -> float:
        rp = await self._parser(url)
        delay = rp.crawl_delay(self._ua) if rp else None
        return float(delay) if delay else 0.0


# --- fetch with retries -------------------------------------------------------------------
@dataclass
class FetchResult:
    url: str
    status: int | None  # None: no HTTP response (timeout, redirect loop, network error)
    body: str = ""
    content_type: str = ""
    nbytes: int = 0
    elapsed: float = 0.0
    attempts: int = 1
    error: str = ""
    final_url: str = ""

    @property
    def ok(self) -> bool:
        return self.status is not None and 200 <= self.status < 300


def parse_retry_after(value: str | None, now: float | None = None) -> float | None:
    """Retry-After: delta-seconds or an HTTP date."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        dt = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, dt.timestamp() - (time.time() if now is None else now))


def backoff_delay(attempt: int, rng: random.Random, base: float = 0.5, cap: float = 30.0) -> float:
    """Exponential backoff with full jitter: uniform(0, min(cap, base * 2**attempt))."""
    return rng.uniform(0, min(cap, base * 2**attempt))


Sleep = Callable[[float], Awaitable[None]]


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 10.0,
    user_agent: str = USER_AGENT,
    sleep: Sleep = asyncio.sleep,
    rng: random.Random | None = None,
    backoff_base: float = 0.5,
) -> FetchResult:
    """GET ``url``; retry only 429/5xx/timeout (at most ``MAX_RETRIES`` times), never other 4xx."""
    rng = rng or random.Random()
    t_start = time.perf_counter()
    result = FetchResult(url=url, status=None)
    for attempt in range(MAX_RETRIES + 1):
        result.attempts = attempt + 1
        retry_after: float | None = None
        try:
            async with asyncio.timeout(timeout):
                resp = await client.get(url, headers={"User-Agent": user_agent})
            result.status = resp.status_code
            result.error = ""
            result.final_url = str(resp.url)
            result.content_type = resp.headers.get("content-type", "")
            result.body = resp.text
            result.nbytes = len(resp.content)
            if resp.status_code not in RETRY_STATUSES:
                break
            retry_after = parse_retry_after(resp.headers.get("retry-after"))
            result.error = f"HTTP {resp.status_code}"
        except TimeoutError:
            result.status, result.error = None, f"timeout after {timeout}s"
        except httpx.TooManyRedirects:
            result.status, result.error = None, "redirect loop"
            break  # retrying a loop will not help
        except httpx.HTTPError as exc:
            result.status, result.error = None, f"{type(exc).__name__}: {exc}"
            break
        if attempt == MAX_RETRIES:
            break
        wait = max(retry_after or 0.0, backoff_delay(attempt, rng, backoff_base))
        log.warning("retry %d/%d for %s (%s), sleeping %.2fs", attempt + 1, MAX_RETRIES, url, result.error, wait)
        await sleep(wait)
    result.elapsed = time.perf_counter() - t_start
    return result


# --- HTML extraction -------------------------------------------------------------------------
def extract(html: str, base_url: str) -> tuple[str, str, list[str]]:
    """Return (title, visible text, absolute links) of an HTML page."""
    tree = HTMLParser(html)
    title_node = tree.css_first("title")
    title: str = title_node.text(strip=True) if title_node else ""
    links: list[str] = []
    for a in tree.css("a[href]"):
        href: str | None = a.attributes.get("href")
        if href and not href.startswith(("mailto:", "javascript:", "tel:")):
            links.append(urljoin(base_url, href))
    for bad in tree.css("script, style, noscript"):
        bad.decompose()
    body = tree.body
    text: str = body.text(separator=" ", strip=True) if body else ""
    return title, " ".join(text.split()), links


# --- crawl ------------------------------------------------------------------------------------
@dataclass
class Page:
    url: str
    title: str
    text: str
    status: int
    nbytes: int
    elapsed: float
    depth: int


@dataclass
class CrawlStats:
    """Live counters (read by the CLI's rich display)."""

    pages: int = 0
    errors: int = 0
    skipped_robots: int = 0
    in_flight: int = 0
    queued: int = 0
    started: float = field(default_factory=time.perf_counter)

    @property
    def pages_per_s(self) -> float:
        return self.pages / max(1e-9, time.perf_counter() - self.started)


async def crawl(
    start_urls: list[str],
    client: httpx.AsyncClient,
    *,
    max_pages: int = 200,
    concurrency: int = 5,
    per_host: int = 5,
    min_delay: float = 0.0,
    timeout: float = 10.0,
    same_host_only: bool = True,
    stats: CrawlStats | None = None,
    backoff_base: float = 0.5,
) -> AsyncGenerator[Page]:
    """Async generator yielding pages as they arrive.

    A queue of URLs feeds ``concurrency`` workers inside a TaskGroup; one global
    semaphore bounds all in-flight requests and one semaphore per host bounds each
    host; ``min_delay`` is the minimum pause between two requests to the same host.
    On ``max_pages`` (or when the generator is closed) the workers are cancelled.
    """
    stats = stats if stats is not None else CrawlStats()
    robots = RobotsCache(client)
    seen: set[str] = set()
    frontier: asyncio.Queue[tuple[str, int]] = asyncio.Queue()
    results: asyncio.Queue[Page | None] = asyncio.Queue()
    global_sem = asyncio.Semaphore(concurrency)
    host_sems: dict[str, asyncio.Semaphore] = {}
    host_next: dict[str, float] = {}
    host_locks: dict[str, asyncio.Lock] = {}
    allowed_hosts = {host_of(canonicalize(u)) for u in start_urls}

    def enqueue(url: str, depth: int) -> None:
        canon = canonicalize(url)
        if canon in seen or not canon.startswith(("http://", "https://")):
            return
        if same_host_only and host_of(canon) not in allowed_hosts:
            return
        seen.add(canon)
        frontier.put_nowait((canon, depth))
        stats.queued = frontier.qsize()

    async def polite_wait(host: str, url: str) -> None:
        delay = max(min_delay, await robots.crawl_delay(url))
        if delay <= 0:
            return
        lock = host_locks.setdefault(host, asyncio.Lock())
        async with lock:  # serialise the "when may I go next" bookkeeping per host
            now = time.perf_counter()
            wait = host_next.get(host, now) - now
            host_next[host] = max(now, host_next.get(host, now)) + delay
        if wait > 0:
            await asyncio.sleep(wait)

    async def process(url: str, depth: int) -> None:
        host = host_of(url)
        if not await robots.allowed(url):
            stats.skipped_robots += 1
            log.info("SKIP robots.txt %s", url)
            return
        sem = host_sems.setdefault(host, asyncio.Semaphore(per_host))
        async with global_sem, sem:
            await polite_wait(host, url)
            stats.in_flight += 1
            try:
                res = await fetch(client, url, timeout=timeout, backoff_base=backoff_base)
            finally:
                stats.in_flight -= 1
        status = res.status if res.status is not None else "ERR"
        log.info("%s %s bytes=%d %.0fms attempts=%d %s", status, url, res.nbytes, res.elapsed * 1000,
                 res.attempts, res.error)
        if not res.ok or "html" not in res.content_type:
            if not res.ok:
                stats.errors += 1
            return
        title, text, links = extract(res.body, res.final_url or url)
        for link in links:
            enqueue(link, depth + 1)
        await results.put(Page(url, title, text, res.status or 0, res.nbytes, res.elapsed, depth))

    async def worker() -> None:
        while True:
            url, depth = await frontier.get()
            try:
                await process(url, depth)
            except Exception:
                stats.errors += 1
                log.exception("unexpected error while processing %s", url)
            finally:
                stats.queued = frontier.qsize()
                frontier.task_done()

    async def closer() -> None:
        await frontier.join()  # queue drained and every in-flight URL finished
        await results.put(None)

    for u in start_urls:
        enqueue(u, 0)
    async with asyncio.TaskGroup() as tg:
        workers = [tg.create_task(worker()) for _ in range(concurrency)]
        closer_task = tg.create_task(closer())
        try:
            while stats.pages < max_pages:
                page = await results.get()
                if page is None:
                    break
                stats.pages += 1
                yield page
        finally:
            closer_task.cancel()
            for w in workers:
                w.cancel()  # stop everything on max_pages, on exhaustion, or if the consumer quits
