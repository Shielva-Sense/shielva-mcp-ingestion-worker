"""RAG source adapters — SSRF guards, URL/API/DB readers, json path."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import src.sources as sources
from src.sources import (
    _assert_db_host,
    _assert_public_url,
    _host_is_public,
    _json_path,
    _row_to_text,
    fetch_url,
    read_api,
    read_database,
)


# ── SSRF guards ───────────────────────────────────────────────────────────────
def test_host_is_public_rejects_private(monkeypatch):
    monkeypatch.setattr(sources.socket, "getaddrinfo", lambda h, p: [(2, 1, 6, "", ("10.0.0.1", 0))])
    assert _host_is_public("internal") is False


def test_host_is_public_accepts_public(monkeypatch):
    monkeypatch.setattr(sources.socket, "getaddrinfo", lambda h, p: [(2, 1, 6, "", ("93.184.216.34", 0))])
    assert _host_is_public("example.com") is True


def test_host_is_public_dns_failure_is_not_public(monkeypatch):
    def boom(*a, **k):
        raise OSError("dns")

    monkeypatch.setattr(sources.socket, "getaddrinfo", boom)
    assert _host_is_public("x") is False


def test_assert_public_url_rejects_bad_scheme():
    with pytest.raises(ValueError):
        _assert_public_url("ftp://example.com/x")


def test_assert_public_url_rejects_no_host():
    with pytest.raises(ValueError):
        _assert_public_url("http://")


def test_assert_public_url_rejects_private(monkeypatch):
    monkeypatch.setattr(sources, "_host_is_public", lambda h: False)
    with pytest.raises(ValueError):
        _assert_public_url("http://169.254.169.254/latest")


def test_assert_db_host_allows_in_dev():
    # module loaded with ENVIRONMENT=development -> _IS_DEV True -> always allowed
    _assert_db_host("localhost")  # no raise


def test_assert_db_host_blocks_private_in_prod(monkeypatch):
    monkeypatch.setattr(sources, "_IS_DEV", False)
    monkeypatch.setattr(sources, "_host_is_public", lambda h: False)
    with pytest.raises(ValueError):
        _assert_db_host("db.internal")


# ── json path + row formatting ────────────────────────────────────────────────
def test_json_path_navigates_nested():
    data = {"data": {"items": [1, 2, 3]}}
    assert _json_path(data, "data.items") == [1, 2, 3]
    assert _json_path(data, None) is data
    assert _json_path(data, "data.missing") is None
    assert _json_path([1, 2], "x") is None


def test_row_to_text_skips_none():
    assert _row_to_text({"a": 1, "b": None, "c": "x"}) == "a: 1\nc: x"


# ── fetch_url ─────────────────────────────────────────────────────────────────
async def test_fetch_url_single_page(monkeypatch):
    monkeypatch.setattr(sources, "_assert_public_url", lambda u: None)

    resp = MagicMock()
    resp.text = "<html>page</html>"
    resp.raise_for_status = MagicMock()
    client = AsyncMock()
    client.get = AsyncMock(return_value=resp)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    with patch.object(sources.httpx, "AsyncClient", return_value=client):
        docs = await fetch_url("http://example.com")
    assert docs == [("http://example.com", "<html>page</html>", "html")]


# ── read_api ──────────────────────────────────────────────────────────────────
async def test_read_api_list_via_json_path(monkeypatch):
    monkeypatch.setattr(sources, "_assert_public_url", lambda u: None)
    resp = MagicMock()
    resp.json = MagicMock(return_value={"results": [{"a": 1}, {"a": 2}]})
    resp.raise_for_status = MagicMock()
    client = AsyncMock()
    client.request = AsyncMock(return_value=resp)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    with patch.object(sources.httpx, "AsyncClient", return_value=client):
        docs = await read_api("http://api.x/y", json_path="results")
    assert len(docs) == 2
    assert docs[0][2] == "json"


async def test_read_api_non_json_falls_back_to_text(monkeypatch):
    monkeypatch.setattr(sources, "_assert_public_url", lambda u: None)
    resp = MagicMock()
    resp.json = MagicMock(side_effect=ValueError("not json"))
    resp.text = "plain body"
    resp.raise_for_status = MagicMock()
    client = AsyncMock()
    client.request = AsyncMock(return_value=resp)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    with patch.object(sources.httpx, "AsyncClient", return_value=client):
        docs = await read_api("http://api.x/y")
    assert docs == [("http://api.x/y", "plain body", "text")]


async def test_read_api_single_object(monkeypatch):
    monkeypatch.setattr(sources, "_assert_public_url", lambda u: None)
    resp = MagicMock()
    resp.json = MagicMock(return_value={"a": 1})
    resp.raise_for_status = MagicMock()
    client = AsyncMock()
    client.request = AsyncMock(return_value=resp)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    with patch.object(sources.httpx, "AsyncClient", return_value=client):
        docs = await read_api("http://api.x/y")
    assert len(docs) == 1
    assert docs[0][2] == "json"


# ── read_database ─────────────────────────────────────────────────────────────
async def test_read_database_requires_select_for_sql():
    with pytest.raises(ValueError):
        await read_database("postgres", "postgresql://localhost/db", query="DELETE FROM t")


async def test_read_database_unsupported_type():
    with pytest.raises(ValueError):
        await read_database("cassandra", "x://h/db")


async def test_read_database_sql_reads_rows(monkeypatch):
    monkeypatch.setattr(sources, "_assert_db_host", lambda h: None)

    result_proxy = MagicMock()
    result_proxy.keys.return_value = ["id", "name"]
    result_proxy.fetchmany.return_value = [(1, "Alice"), (2, "Bob")]
    conn = MagicMock()
    conn.execute.return_value = result_proxy
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    engine = MagicMock()
    engine.connect.return_value = conn

    monkeypatch.setattr(sources, "_get_pooled_engine", lambda uri: engine)
    docs = await read_database("postgres", "postgresql://h/db", query="SELECT * FROM t")
    assert len(docs) == 2
    assert "id: 1" in docs[0][1]
    assert docs[1][0] == "row 2"


async def test_read_database_mongo_requires_collection():
    with pytest.raises(ValueError):
        await read_database("mongodb", "mongodb://h/db", collection=None)


async def test_fetch_url_crawl_bfs(monkeypatch):
    monkeypatch.setattr(sources, "_assert_public_url", lambda u: None)
    monkeypatch.setattr(sources, "_host_is_public", lambda h: True)

    pages = {
        "http://site.com/": '<a href="/a">A</a><a href="http://other.com/x">ext</a>',
        "http://site.com/a": "<p>leaf</p>",
    }

    def make_resp(url):
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"content-type": "text/html"}
        resp.text = pages.get(url, "")
        return resp

    client = AsyncMock()
    client.get = AsyncMock(side_effect=lambda url, headers=None: make_resp(url))
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    with patch.object(sources.httpx, "AsyncClient", return_value=client):
        docs = await fetch_url("http://site.com/", crawl=True, max_pages=10, max_depth=1)
    urls = {d[0] for d in docs}
    assert "http://site.com/" in urls
    assert "http://site.com/a" in urls  # same-host link followed
    # cross-host link is never fetched
    assert "http://other.com/x" not in urls


async def test_fetch_url_crawl_skips_non_html(monkeypatch):
    monkeypatch.setattr(sources, "_assert_public_url", lambda u: None)
    monkeypatch.setattr(sources, "_host_is_public", lambda h: True)
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "application/json"}
    resp.text = "{}"
    client = AsyncMock()
    client.get = AsyncMock(return_value=resp)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    with patch.object(sources.httpx, "AsyncClient", return_value=client):
        docs = await fetch_url("http://site.com/", crawl=True)
    assert docs == []  # non-html content skipped


async def test_read_database_mongo_reads_documents(monkeypatch):
    monkeypatch.setattr(sources, "_assert_db_host", lambda h: None)

    class FakeCursor:
        def __init__(self, docs):
            self._docs = docs

        def limit(self, n):
            return self

        def __aiter__(self):
            async def gen():
                for d in self._docs:
                    yield d

            return gen()

    collection = MagicMock()
    collection.find.return_value = FakeCursor([{"name": "A"}, {"name": "B"}])
    db = MagicMock()
    db.__getitem__.return_value = collection
    client = MagicMock()
    client.get_default_database.return_value = db
    client.close = MagicMock()

    fake_motor = MagicMock()
    fake_motor.AsyncIOMotorClient = MagicMock(return_value=client)
    with patch.dict("sys.modules", {"motor": MagicMock(), "motor.motor_asyncio": fake_motor}):
        docs = await read_database("mongodb", "mongodb://h/db", collection="things", query='{"active": true}')
    assert len(docs) == 2
    assert docs[0][0] == "things #1"
    client.close.assert_called_once()


async def test_read_database_mongo_invalid_query_ignored(monkeypatch):
    monkeypatch.setattr(sources, "_assert_db_host", lambda h: None)

    class FakeCursor:
        def limit(self, n):
            return self

        def __aiter__(self):
            async def gen():
                yield {"x": 1}

            return gen()

    collection = MagicMock()
    collection.find.return_value = FakeCursor()
    db = MagicMock()
    db.__getitem__.return_value = collection
    client = MagicMock()
    client.get_default_database.return_value = db
    fake_motor = MagicMock()
    fake_motor.AsyncIOMotorClient = MagicMock(return_value=client)
    with patch.dict("sys.modules", {"motor": MagicMock(), "motor.motor_asyncio": fake_motor}):
        docs = await read_database("mongo", "mongodb://h/db", collection="c", query="not-json")
    assert len(docs) == 1  # bad query -> empty filter, still reads


async def test_read_database_mongo_no_default_db(monkeypatch):
    monkeypatch.setattr(sources, "_assert_db_host", lambda h: None)
    client = MagicMock()
    client.get_default_database.return_value = None
    fake_motor = MagicMock()
    fake_motor.AsyncIOMotorClient = MagicMock(return_value=client)
    with patch.dict("sys.modules", {"motor": MagicMock(), "motor.motor_asyncio": fake_motor}):
        with pytest.raises(ValueError):
            await read_database("mongodb", "mongodb://h", collection="c")


def test_get_pooled_engine_is_cached(monkeypatch):
    sources._ENGINE_CACHE.clear()
    created = {"n": 0}

    def fake_create_engine(uri, **kw):
        created["n"] += 1
        return MagicMock()

    fake_sa = MagicMock()
    fake_sa.create_engine = fake_create_engine
    with patch.dict("sys.modules", {"sqlalchemy": fake_sa}):
        e1 = sources._get_pooled_engine("postgresql://h/db")
        e2 = sources._get_pooled_engine("postgresql://h/db")
    assert e1 is e2
    assert created["n"] == 1
