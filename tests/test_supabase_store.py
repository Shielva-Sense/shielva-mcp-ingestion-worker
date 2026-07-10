"""Supabase (vecs/pgvector) vector store — collection naming, CRUD, SQL helpers.

The ``vecs`` client and DB session are mocked; every method's real logic
(name derivation, record shaping, result mapping, error fallbacks) is exercised.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from src.vectorstore import SearchResult, VectorDocument
from src.vectorstore.supabase_store import SupabaseVectorStore


def _store() -> SupabaseVectorStore:
    return SupabaseVectorStore(db_url="postgresql://h/db", collection_prefix="kb_", embedding_dim=4)


def _session_client(sess):
    """Build a fake vecs client whose Session() context manager yields ``sess``."""
    client = MagicMock()

    @contextmanager
    def _session():
        yield sess

    client.Session = _session
    return client


# ── collection name derivation ────────────────────────────────────────────────
def test_collection_name_short_replaces_dashes():
    s = _store()
    name = s._get_collection_name("tenant-a", "kb-1")
    assert name == "kb_tenant_a_kb_1"


def test_collection_name_long_is_hashed():
    s = _store()
    long_tenant = "t" * 60
    name = s._get_collection_name(long_tenant, "kb-xyz")
    assert name.startswith("kb_")
    assert len(name) <= 63


# ── connect / close ───────────────────────────────────────────────────────────
async def test_connect_success(monkeypatch):
    import src.vectorstore.supabase_store as mod

    fake_client = MagicMock()
    monkeypatch.setattr(mod.vecs, "create_client", lambda url: fake_client)
    s = _store()
    await s.connect()
    assert s._client is fake_client


async def test_connect_failure_raises(monkeypatch):
    import src.vectorstore.supabase_store as mod

    def boom(url):
        raise RuntimeError("no db")

    monkeypatch.setattr(mod.vecs, "create_client", boom)
    s = _store()
    with pytest.raises(RuntimeError):
        await s.connect()


async def test_close_disconnects():
    s = _store()
    s._client = MagicMock()
    await s.close()
    s._client.disconnect.assert_called_once()


# ── create / delete collection ────────────────────────────────────────────────
async def test_create_collection(monkeypatch):
    s = _store()
    s._client = MagicMock()

    async def fake_index(t, k):
        return None

    monkeypatch.setattr(s, "create_text_index", fake_index)
    name = await s.create_collection("t", "kb")
    assert name == s._get_collection_name("t", "kb")
    s._client.get_or_create_collection.assert_called_once()


async def test_create_collection_error(monkeypatch):
    s = _store()
    s._client = MagicMock()
    s._client.get_or_create_collection.side_effect = RuntimeError("fail")
    with pytest.raises(RuntimeError):
        await s.create_collection("t", "kb")


async def test_delete_collection_true_and_false():
    s = _store()
    s._client = MagicMock()
    assert await s.delete_collection("t", "kb") is True
    s._client.delete_collection.side_effect = RuntimeError("x")
    assert await s.delete_collection("t", "kb") is False


# ── upsert ────────────────────────────────────────────────────────────────────
async def test_upsert_builds_records():
    s = _store()
    coll = MagicMock()
    s._client = MagicMock()
    s._client.get_or_create_collection.return_value = coll
    docs = [VectorDocument(id="c1", content="body", embedding=[0.1] * 4, metadata={"m": 1})]
    n = await s.upsert("t", "kb", docs)
    assert n == 1
    records = coll.upsert.call_args.kwargs["records"]
    rid, vec, meta = records[0]
    assert rid == "c1"
    assert meta["content"] == "body"
    assert meta["tenant_id"] == "t"


async def test_upsert_error_raises():
    s = _store()
    coll = MagicMock()
    coll.upsert.side_effect = RuntimeError("db")
    s._client = MagicMock()
    s._client.get_or_create_collection.return_value = coll
    with pytest.raises(RuntimeError):
        await s.upsert("t", "kb", [VectorDocument(id="c", content="x", embedding=[0.0] * 4)])


# ── search ────────────────────────────────────────────────────────────────────
async def test_search_maps_results():
    s = _store()
    coll = MagicMock()
    coll.query.return_value = [("c1", {}), ("c2", {})]
    rec1 = MagicMock(id="c1", metadata={"content": "hello", "tenant_id": "t", "kb_id": "kb", "x": 1})
    rec2 = MagicMock(id="c2", metadata={"content": "world"})
    coll.fetch.return_value = [rec1, rec2]
    s._client = MagicMock()
    s._client.get_collection.return_value = coll
    results = await s.search("t", ["kb"], [0.1] * 4, top_k=5)
    assert len(results) == 2
    assert isinstance(results[0], SearchResult)
    assert results[0].content == "hello"
    assert "tenant_id" not in results[0].metadata  # internal keys stripped


async def test_search_skips_missing_collection():
    s = _store()
    s._client = MagicMock()
    s._client.get_collection.side_effect = KeyError("missing")
    assert await s.search("t", ["kb"], [0.1] * 4) == []


# ── raw-SQL helpers (Session mocked) ──────────────────────────────────────────
async def test_create_text_index_executes_ddl():
    s = _store()
    sess = MagicMock()
    s._client = _session_client(sess)
    await s.create_text_index("t", "kb")
    assert sess.execute.call_count == 2  # FTS + HNSW
    sess.commit.assert_called_once()


async def test_create_text_index_swallows_errors():
    s = _store()
    sess = MagicMock()
    sess.execute.side_effect = RuntimeError("no ddl")
    s._client = _session_client(sess)
    await s.create_text_index("t", "kb")  # must not raise


async def test_list_documents_maps_rows():
    s = _store()
    sess = MagicMock()
    sess.execute.return_value = [("doc-1", "Title", 3, 2048), ("doc-2", "", 1, 0)]
    s._client = _session_client(sess)
    docs = await s.list_documents("t", "kb")
    assert docs[0] == {"document_id": "doc-1", "title": "Title", "chunks": 3, "bytes": 2048}
    assert docs[1]["title"] == "doc-2"  # empty title falls back to id


async def test_list_documents_error_returns_empty():
    s = _store()
    sess = MagicMock()
    sess.execute.side_effect = RuntimeError("no table")
    s._client = _session_client(sess)
    assert await s.list_documents("t", "kb") == []


async def test_kb_storage_reads_counts():
    s = _store()
    sess = MagicMock()
    row = MagicMock()
    row.__getitem__ = lambda self, i: [5, 42, 9999][i]
    sess.execute.return_value.first.return_value = row
    s._client = _session_client(sess)
    out = await s.kb_storage("t", "kb")
    assert out == {"documents": 5, "chunks": 42, "file_bytes": 9999}


async def test_kb_storage_error_returns_zeros():
    s = _store()
    sess = MagicMock()
    sess.execute.side_effect = RuntimeError("x")
    s._client = _session_client(sess)
    assert await s.kb_storage("t", "kb") == {"documents": 0, "chunks": 0, "file_bytes": 0}


async def test_list_chunks_with_and_without_doc_filter():
    s = _store()
    sess = MagicMock()
    sess.execute.return_value = [("id1", "text", "doc1", "Title", 0)]
    s._client = _session_client(sess)
    out = await s.list_chunks("t", "kb", document_id="doc1")
    assert out[0]["id"] == "id1"
    assert out[0]["content"] == "text"
    # without doc filter
    out2 = await s.list_chunks("t", "kb")
    assert out2[0]["document_id"] == "doc1"


async def test_delete_chunk_returns_rowcount():
    s = _store()
    sess = MagicMock()
    sess.execute.return_value.rowcount = 1
    s._client = _session_client(sess)
    assert await s.delete_chunk("t", "kb", "chunk-1") == 1


async def test_delete_chunk_error_returns_zero():
    s = _store()
    sess = MagicMock()
    sess.execute.side_effect = RuntimeError("x")
    s._client = _session_client(sess)
    assert await s.delete_chunk("t", "kb", "chunk-1") == 0


async def test_update_chunk_true_and_false():
    s = _store()
    sess = MagicMock()
    sess.execute.return_value.rowcount = 1
    s._client = _session_client(sess)
    assert await s.update_chunk("t", "kb", "c1", "new text", [0.1, 0.2]) is True

    sess2 = MagicMock()
    sess2.execute.return_value.rowcount = 0
    s._client = _session_client(sess2)
    assert await s.update_chunk("t", "kb", "c1", "new", [0.1]) is False


async def test_update_chunk_error_returns_false():
    s = _store()
    sess = MagicMock()
    sess.execute.side_effect = RuntimeError("x")
    s._client = _session_client(sess)
    assert await s.update_chunk("t", "kb", "c1", "x", [0.1]) is False


async def test_keyword_search_maps_rows():
    s = _store()
    sess = MagicMock()
    sess.execute.return_value = [("c1", {"content": "hit", "tenant_id": "t"}, 0.9)]
    s._client = _session_client(sess)
    results = await s.keyword_search("t", ["kb"], "query")
    assert results[0].content == "hit"
    assert results[0].score == 0.9
    assert "tenant_id" not in results[0].metadata


async def test_keyword_search_error_continues():
    s = _store()
    sess = MagicMock()
    sess.execute.side_effect = RuntimeError("no fts")
    s._client = _session_client(sess)
    assert await s.keyword_search("t", ["kb"], "q") == []


async def test_delete_by_ids():
    s = _store()
    coll = MagicMock()
    s._client = MagicMock()
    s._client.get_collection.return_value = coll
    n = await s.delete_by_ids("t", "kb", ["a", "b", "c"])
    assert n == 3
    coll.delete.assert_called_once()


async def test_get_collection_info_counts():
    s = _store()
    sess = MagicMock()
    sess.execute.return_value.scalar.return_value = 12
    client = _session_client(sess)
    client.get_collection = MagicMock(return_value=MagicMock())
    s._client = client
    info = await s.get_collection_info("t", "kb")
    assert info["vector_count"] == 12


async def test_get_collection_info_missing_collection():
    s = _store()
    s._client = MagicMock()
    s._client.get_collection.side_effect = KeyError("missing")
    info = await s.get_collection_info("t", "kb")
    assert info["vector_count"] == 0


async def test_delete_by_filter_builds_where():
    s = _store()
    sess = MagicMock()
    sess.execute.return_value.rowcount = 4
    s._client = _session_client(sess)
    n = await s.delete_by_filter("t", "kb", {"document_id": "d1"})
    assert n == 4


async def test_delete_by_filter_error_returns_zero():
    s = _store()
    sess = MagicMock()
    sess.execute.side_effect = RuntimeError("x")
    s._client = _session_client(sess)
    assert await s.delete_by_filter("t", "kb", {"document_id": "d1"}) == 0
