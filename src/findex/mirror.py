"""A local mirror site for crawler tests and benchmarks (lab 6).

Serves N generated pages (chunks of books if ``--source`` points to a text corpus, otherwise
synthetic text) on 127.0.0.1 with a configurable per-request latency, a robots.txt, and a few
deliberate failure paths:

  /flaky/N     503 the first two times, then 200          (retry + backoff)
  /limited/N   429 + ``Retry-After: 1`` the first time     (Retry-After handling)
  /loop/a      redirects to /loop/b and back               (redirect loop)
  /slow        answers after 5 s                           (timeout)
  /private/x   disallowed by robots.txt

Usage: python -m findex.mirror --port 8765 --pages 400 --latency 0.05 [--source data/gutenberg]
"""

import argparse
import html
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

WORDS = ["index", "search", "query", "token", "corpus", "document", "ranking", "posting", "term", "list", "merge", "score", "cache", "async", "await", "loop", "thread", "process", "lock", "queue", "stream", "file", "path", "byte", "memory"]


def make_pages(n: int, source: Path | None, seed: int = 7) -> list[tuple[str, str]]:
    """(title, text) for page 0..n-1."""
    rng = random.Random(seed)
    books = sorted(source.glob("*.txt")) if source and source.is_dir() else []
    pages: list[tuple[str, str]] = []
    if books:
        per_book = max(1, n // min(len(books), 40) + 1)
        for b in books[:40]:
            text = b.read_text(encoding="utf-8", errors="replace")
            title = next((ln.strip() for ln in text.splitlines() if sum(c.isalpha() for c in ln) > 3), b.stem)
            step = max(1, len(text) // (per_book + 1))
            for part in range(per_book):
                chunk = text[step * (part + 1) : step * (part + 1) + 2500]
                pages.append((f"{title[:60]} - part {part + 1}", " ".join(chunk.split())))
        pages = pages[:n]
    while len(pages) < n:
        words = " ".join(rng.choice(WORDS) for _ in range(250))
        pages.append((f"Synthetic page {len(pages)}", words))
    return pages


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # silence default access log
        pass

    def _send(self, status: int, body: str = "", ctype: str = "text/html; charset=utf-8",
              headers: dict[str, str] | None = None) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        srv = cast(MirrorServer, self.server)
        srv.count_request(path)
        if path == "/robots.txt":
            self._send(200, "User-agent: *\nDisallow: /private/\n", "text/plain")
            return
        time.sleep(srv.latency)
        if path in ("/", ""):
            self._send(200, srv.home())
        elif path.startswith("/page/"):
            try:
                n = int(path.split("/")[2])
            except ValueError:
                n = -1
            if 0 <= n < len(srv.pages):
                self._send(200, srv.render(n))
            else:
                self._send(404, "<html><title>404</title><body>no such page</body></html>")
        elif path.startswith("/flaky/"):
            if srv.nth_request(path) <= 2:
                self._send(503, "<html><body>try later</body></html>")
            else:
                self._send(200, "<html><title>Flaky page</title><body>Finally served after retries.</body></html>")
        elif path.startswith("/limited/"):
            if srv.nth_request(path) == 1:
                self._send(429, "slow down", "text/plain", {"Retry-After": "1"})
            else:
                self._send(200, "<html><title>Rate limited page</title><body>Served after Retry-After.</body></html>")
        elif path in ("/loop/a", "/loop/b"):
            self._send(302, "", headers={"Location": "/loop/b" if path == "/loop/a" else "/loop/a"})
        elif path == "/slow":
            time.sleep(5)
            self._send(200, "<html><title>Slow</title><body>very slow</body></html>")
        else:
            self._send(404, "<html><title>404</title><body>not found</body></html>")


class MirrorServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128

    def __init__(self, port: int, pages: list[tuple[str, str]], latency: float, faults: bool = True) -> None:
        super().__init__(("127.0.0.1", port), Handler)
        self.pages = pages
        self.latency = latency
        self.faults = faults
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()
        self.rng = random.Random(11)
        self.links = [self.rng.sample(range(len(pages)), 3) for _ in pages]

    def count_request(self, path: str) -> None:
        with self._lock:
            self._counts[path] = self._counts.get(path, 0) + 1

    def nth_request(self, path: str) -> int:
        with self._lock:
            return self._counts.get(path, 0)

    def home(self) -> str:
        links = ['<a href="/page/0">start</a>']
        if not self.faults:
            return "<html><title>Mirror home</title><body>" + links[0] + "</body></html>"
        links += [f'<a href="/flaky/{i}">flaky {i}</a>' for i in range(2)]
        links += [f'<a href="/limited/{i}">limited {i}</a>' for i in range(2)]
        links += ['<a href="/loop/a">loop</a>', '<a href="/slow">slow</a>', '<a href="/private/x">private</a>']
        return "<html><title>Mirror home</title><body><h1>Mirror</h1>" + " ".join(links) + "</body></html>"

    def render(self, n: int) -> str:
        title, text = self.pages[n]
        nxt = (n + 1) % len(self.pages)
        prv = (n - 1) % len(self.pages)
        extra = "".join(f'<a href="/page/{k}/">see also {k}</a> ' for k in self.links[n])
        return (
            f"<html><head><title>{html.escape(title)}</title><style>p{{color:red}}</style></head><body>"
            f"<h1>{html.escape(title)}</h1><p>{html.escape(text)}</p>"
            f'<nav><a href="/page/{nxt}#top">next</a> <a href="/page/{prv}">prev</a> '
            f'<a href="HTTP://127.0.0.1:{self.server_address[1]}/page/{nxt}">next again</a> {extra}'
            f'<a href="https://example.com/elsewhere">external</a> <a href="/">home</a></nav>'
            f"<script>var x = 'not text';</script></body></html>"
        )


def serve(port: int = 8765, n_pages: int = 400, latency: float = 0.05, source: Path | None = None,
          faults: bool = True) -> MirrorServer:
    return MirrorServer(port, make_pages(n_pages, source), latency, faults)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--pages", type=int, default=400)
    ap.add_argument("--latency", type=float, default=0.05, help="seconds per request")
    ap.add_argument("--source", type=Path, default=None, help="directory of .txt books to slice into pages")
    ap.add_argument("--no-faults", action="store_true", help="home page does not link to the failure paths")
    args = ap.parse_args()
    srv = serve(args.port, args.pages, args.latency, args.source, faults=not args.no_faults)
    print(f"mirror on http://127.0.0.1:{args.port}/ ({args.pages} pages, latency {args.latency}s)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
