# pyright: basic
# (Starlette's TestClient exposes partially-unknown types in strict mode; the rest of the tests stay strict.)
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from findex.index import Index
from findex.store import save
from findex.web.app import create_app, get_index
from findex.web.settings import Settings


@pytest.fixture
def client(array_index: Index) -> Iterator[TestClient]:
    """App with NO index file: the index comes from dependency_overrides."""
    app = create_app(Settings(index_path=Path("does-not-exist.bin")))
    app.dependency_overrides[get_index] = lambda: array_index
    with TestClient(app) as c:
        yield c


def test_search_ok(client: TestClient) -> None:
    r = client.get("/search", params={"q": "python", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 3 and body["per_page"] == 2 and body["pages"] == 2 and body["page"] == 1
    assert len(body["results"]) == 2
    assert body["results"][0]["score"] >= body["results"][1]["score"]
    assert "**python**" in body["results"][0]["snippet"].lower()


def test_search_pagination_covers_all_results(client: TestClient) -> None:
    ids: list[int] = []
    for page in (1, 2):
        r = client.get("/search", params={"q": "python", "k": 2, "page": page})
        ids += [h["doc_id"] for h in r.json()["results"]]
    assert sorted(ids) == [0, 2, 3]
    assert client.get("/search", params={"q": "python", "k": 2, "page": 3}).json()["results"] == []


def test_search_phrase_and_scorers(client: TestClient) -> None:
    for scorer in ("bm25", "tfidf"):
        r = client.get("/search", params={"q": '"event loop"', "scorer": scorer})
        assert r.status_code == 200
        assert {h["doc_id"] for h in r.json()["results"]} == {0, 1}


@pytest.mark.parametrize(
    "params",
    [
        {},  # q is required
        {"q": ""},
        {"q": "x", "k": 0},
        {"q": "x", "k": 51},
        {"q": "x", "page": 0},
        {"q": "x", "scorer": "magic"},
        {"q": "x" * 201},
    ],
)
def test_search_validation_422(client: TestClient, params: dict[str, str | int]) -> None:
    assert client.get("/search", params=params).status_code == 422


def test_search_bad_query_is_422_with_message(client: TestClient) -> None:
    r = client.get("/search", params={"q": "(a OR"})
    assert r.status_code == 422 and r.json()["detail"].startswith("bad query")


def test_doc_found_and_404(client: TestClient) -> None:
    r = client.get("/docs/0")
    assert r.status_code == 200
    body = r.json()
    assert body["doc_id"] == 0 and "apple" in body["text"] and body["length"] == 10
    r = client.get("/docs/999")
    assert r.status_code == 404 and "999" in r.json()["detail"]
    assert client.get("/docs/abc").status_code == 422


def test_stats(client: TestClient) -> None:
    body = client.get("/stats").json()
    assert body["documents"] == 4 and body["positions"] is True and body["layout"] == "array"


def test_health_503_when_index_not_loaded(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 503 and r.json()["status"] == "loading"


def test_search_503_without_any_index() -> None:
    with TestClient(create_app(Settings(index_path=Path("nope.bin")))) as c:
        assert c.get("/search", params={"q": "x"}).status_code == 503
        assert c.get("/health").status_code == 503


def test_health_ok_when_lifespan_loads_index(array_index: Index, tmp_path: Path) -> None:
    path = tmp_path / "i.bin"
    save(array_index, path, "binary")
    with TestClient(create_app(Settings(index_path=path))) as c:
        r = c.get("/health")
        assert r.status_code == 200 and r.json() == {"status": "ok", "documents": 4}
        assert c.get("/search", params={"q": "fig"}).json()["total"] == 1


def test_html_home_and_results(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200 and "<form" in r.text and "Example queries" in r.text
    r = client.get("/", params={"q": "python NOT java"})
    assert r.status_code == 200 and "<mark>" in r.text and "matching document" in r.text
    r = client.get("/", params={"q": "(a OR"})
    assert r.status_code == 422 and "bad query" in r.text


def test_html_escapes_user_input(client: TestClient) -> None:
    r = client.get("/", params={"q": '<script>alert(1)</script>'})
    assert "<script>alert(1)</script>" not in r.text


def test_view_document(client: TestClient) -> None:
    r = client.get("/view/0", params={"q": "python"})
    assert r.status_code == 200 and "<mark>python</mark>" in r.text.lower()
    assert client.get("/view/999").status_code == 404


def test_unexpected_error_is_500_with_request_id_and_no_stack(
    array_index: Index, caplog: pytest.LogCaptureFixture
) -> None:
    def boom() -> Index:
        raise RuntimeError("secret internal detail")

    app = create_app(Settings(index_path=Path("x.bin")))
    app.dependency_overrides[get_index] = boom
    with TestClient(app, raise_server_exceptions=False) as c, caplog.at_level(logging.ERROR):
        r = c.get("/search", params={"q": "x"}, headers={"X-Request-ID": "req-123"})
    assert r.status_code == 500
    assert r.json() == {"detail": "Internal Server Error", "request_id": "req-123"}
    assert r.headers["x-request-id"] == "req-123"
    assert "secret internal detail" not in r.text and "Traceback" not in r.text
    assert "req-123" in caplog.text and "secret internal detail" in caplog.text  # the log has the details


def test_response_carries_generated_request_id(client: TestClient) -> None:
    assert len(client.get("/stats").headers["x-request-id"]) == 12


def test_openapi_docs_moved_so_docs_route_is_ours(client: TestClient) -> None:
    assert client.get("/api/docs").status_code == 200
    assert client.get("/docs/1").json()["doc_id"] == 1


def test_semantic_and_hybrid_modes_via_dependency_override(array_index: Index, corpus_dir: Path) -> None:
    from findex.corpus import iter_documents
    from findex.semantic import embed_corpus
    from findex.web.app import get_semantic

    from .test_semantic import fake_embed

    sem = embed_corpus(iter_documents(corpus_dir), embed=fake_embed)
    app = create_app(Settings(index_path=Path("none.bin")))
    app.dependency_overrides[get_index] = lambda: array_index
    app.dependency_overrides[get_semantic] = lambda: sem
    with TestClient(app) as c:
        for mode in ("semantic", "hybrid"):
            r = c.get("/search", params={"q": "event loop threads", "mode": mode, "k": 3})
            assert r.status_code == 200 and r.json()["mode"] == mode and r.json()["results"][0]["doc_id"] in (0, 1)
        page = c.get("/", params={"q": "event loop threads", "mode": "semantic"})
        assert page.status_code == 200 and 'value="semantic" selected' in page.text
        assert c.get("/search", params={"q": "x", "mode": "bogus"}).status_code == 422


def test_semantic_mode_503_when_no_embeddings_loaded(client: TestClient) -> None:
    r = client.get("/search", params={"q": "python", "mode": "semantic"})
    assert r.status_code == 503 and "no embeddings" in r.json()["detail"]
