"""Vector indexer — batch upsert, delete, mock path, index ensure."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.indexer import IndexedChunk, IndexerConfig, VectorIndexer


def _chunk(i: int) -> dict:
    return {
        "id": f"c{i}",
        "document_id": "d1",
        "content": f"body {i}",
        "embedding": [0.1, 0.2],
        "metadata": {"k": "v"},
        "chunk_index": i,
    }


async def test_index_chunks_empty_returns_empty():
    idx = VectorIndexer(vector_store=MagicMock())
    assert await idx.index_chunks([], "t", "kb") == []


async def test_index_chunks_upserts_and_creates_index():
    store = MagicMock()
    store.connect = AsyncMock()
    store.upsert = AsyncMock()
    store.create_text_index = AsyncMock()
    idx = VectorIndexer(config=IndexerConfig(batch_size=2), vector_store=store)

    results = await idx.index_chunks([_chunk(0), _chunk(1), _chunk(2)], "t", "kb")
    assert len(results) == 3
    assert all(r.success for r in results)
    # 3 chunks / batch 2 => 2 upsert batches
    assert store.upsert.await_count == 2
    assert store.create_text_index.await_count == 2


async def test_index_chunks_batch_failure_marks_failed():
    store = MagicMock()
    store.connect = AsyncMock()
    store.upsert = AsyncMock(side_effect=RuntimeError("db down"))
    idx = VectorIndexer(vector_store=store)
    results = await idx.index_chunks([_chunk(0)], "t", "kb")
    assert len(results) == 1
    assert results[0].success is False
    assert results[0].error == "db down"


async def test_index_batch_mock_when_no_store():
    idx = VectorIndexer.__new__(VectorIndexer)
    idx.config = IndexerConfig()
    idx.vector_store = None
    results = await idx._index_batch([_chunk(0)], "t", "kb")
    assert results[0].vector_id == "vec-c0"
    assert results[0].success


async def test_index_ensure_failure_is_swallowed():
    store = MagicMock()
    store.connect = AsyncMock()
    store.upsert = AsyncMock()
    store.create_text_index = AsyncMock(side_effect=RuntimeError("index oops"))
    idx = VectorIndexer(vector_store=store)
    results = await idx.index_chunks([_chunk(0)], "t", "kb")
    assert results[0].success  # upsert still succeeded


async def test_delete_by_document_and_kb():
    store = MagicMock()
    store.delete_by_filter = AsyncMock(return_value=5)
    store.delete_collection = AsyncMock(return_value=True)
    idx = VectorIndexer(vector_store=store)
    assert await idx.delete_by_document("d1", "t", "kb") == 5
    assert await idx.delete_by_kb("t", "kb") is True
    store.delete_by_filter.assert_awaited_once()


async def test_delete_returns_defaults_when_no_store():
    idx = VectorIndexer.__new__(VectorIndexer)
    idx.config = IndexerConfig()
    idx.vector_store = None
    assert await idx.delete_by_document("d", "t", "kb") == 0
    assert await idx.delete_by_kb("t", "kb") is True


def test_default_indexer_lazily_builds_supabase_store(monkeypatch):
    import src.vectorstore as vs

    made = {}

    class FakeStore:
        def __init__(self, db_url, collection_prefix):
            made["db_url"] = db_url
            made["prefix"] = collection_prefix

    monkeypatch.setattr(vs, "SupabaseVectorStore", FakeStore)
    idx = VectorIndexer(config=IndexerConfig(host="postgresql://h/db"))
    assert isinstance(idx.vector_store, FakeStore)
    assert made["prefix"] == "shielva_kb_"


def test_indexed_chunk_dataclass_defaults():
    ic = IndexedChunk(id="a", document_id="d", vector_id="v")
    assert ic.success is True
    assert ic.error is None
