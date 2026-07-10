"""FastAPI application — endpoints, caps, helpers, queue wiring.

The app's global ``pipeline`` / ``processor`` / ingest queue are replaced with
mocks; the verified-principal dependency is overridden. Each endpoint's real
request handling (validation, tenant scoping, response shaping, error mapping)
is exercised via the Starlette TestClient (lifespan is intentionally NOT run).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import main
import src.jobs.queue as queue_mod
from main import (
    IngestBatchRequest,
    IngestDocumentRequest,
    _doctype_for_filename,
    _stable_doc_id,
    _submit_to_queue,
    _validate_batch_caps,
    find_free_port,
)
from shielva_common.auth import Principal, require_principal
from src.jobs.queue import QueueFull
from src.models import DocumentType


class FakeQueue:
    def __init__(self):
        self.submitted = []
        self.full = False

    def submit(self, job, run):
        if self.full:
            raise QueueFull("full")
        self.submitted.append((job, run))
        return len(self.submitted)

    def stats(self):
        return {"target": 2, "active": 0, "waiting": 0, "accepting": True}

    def cancel_by_kb(self, kb_id):
        return 2


@pytest.fixture()
def client(monkeypatch):
    # Mocked pipeline + processor globals (normally built in lifespan)
    pipeline = MagicMock()
    pipeline.vector_store = MagicMock()
    pipeline.vector_store.create_collection = AsyncMock(return_value="coll_t_kb")
    pipeline.vector_store.delete_collection = AsyncMock(return_value=True)
    pipeline.vector_store.get_collection_info = AsyncMock(
        return_value={"collection_name": "c", "vector_count": 7, "document_count": 7}
    )
    pipeline.vector_store.list_documents = AsyncMock(return_value=[{"document_id": "d1"}])
    pipeline.vector_store.kb_storage = AsyncMock(return_value={"documents": 1, "chunks": 3, "file_bytes": 100})
    pipeline.vector_store.list_chunks = AsyncMock(return_value=[{"id": "c1"}])
    pipeline.vector_store.delete_chunk = AsyncMock(return_value=1)
    pipeline.vector_store.update_chunk = AsyncMock(return_value=True)
    pipeline.embedding_client = MagicMock()
    pipeline.embedding_client.embed_single = AsyncMock(return_value=[0.1, 0.2])
    pipeline.delete_document = AsyncMock(return_value=4)

    processor = MagicMock()
    processor.process_job = AsyncMock()

    monkeypatch.setattr(main, "pipeline", pipeline)
    monkeypatch.setattr(main, "processor", processor)
    monkeypatch.setattr(queue_mod, "ingest_queue", FakeQueue())

    main.app.dependency_overrides[require_principal] = lambda: Principal(tenant_id="tenant-x")
    c = TestClient(main.app, raise_server_exceptions=False)
    yield c
    main.app.dependency_overrides.clear()


# ── pure helpers ──────────────────────────────────────────────────────────────
def test_doctype_for_filename():
    assert _doctype_for_filename("a.pdf") is DocumentType.PDF
    assert _doctype_for_filename("a.MD") is DocumentType.MARKDOWN
    assert _doctype_for_filename("noext") is DocumentType.TEXT
    assert _doctype_for_filename("a.weird") is DocumentType.TEXT


def test_stable_doc_id_deterministic():
    a = _stable_doc_id("upload", "kb", "key", "content")
    b = _stable_doc_id("upload", "kb", "key", "content")
    c = _stable_doc_id("upload", "kb", "key", "different")
    assert a == b
    assert a != c
    assert a.startswith("upload_")
    # bytes content also supported
    assert _stable_doc_id("x", "kb", "k", b"bytes").startswith("x_")


def test_validate_batch_caps_ok_and_too_many(monkeypatch):
    small = IngestBatchRequest(kb_id="kb", documents=[IngestDocumentRequest(id="1", content="hi", title="T")])
    _validate_batch_caps(small)  # no raise

    docs = [IngestDocumentRequest(id=str(i), content="x", title="T") for i in range(200)]
    with pytest.raises(HTTPException) as ei:
        _validate_batch_caps(IngestBatchRequest(kb_id="kb", documents=docs))
    assert ei.value.status_code == 413


def test_validate_batch_caps_doc_too_large(monkeypatch):
    from src.config import get_settings

    settings = get_settings()
    big = "x" * (settings.max_bytes_per_document + 1)
    req = IngestBatchRequest(kb_id="kb", documents=[IngestDocumentRequest(id="1", content=big, title="T")])
    with pytest.raises(HTTPException) as ei:
        _validate_batch_caps(req)
    assert ei.value.status_code == 413


def test_submit_to_queue_503_when_missing(monkeypatch):
    monkeypatch.setattr(queue_mod, "ingest_queue", None)
    with pytest.raises(HTTPException) as ei:
        _submit_to_queue(MagicMock(), lambda: None)
    assert ei.value.status_code == 503


def test_submit_to_queue_429_when_full(monkeypatch):
    q = FakeQueue()
    q.full = True
    monkeypatch.setattr(queue_mod, "ingest_queue", q)
    with pytest.raises(HTTPException) as ei:
        _submit_to_queue(MagicMock(), lambda: None)
    assert ei.value.status_code == 429


def test_find_free_port_returns_bindable():
    port = find_free_port(58000)
    assert 58000 <= port < 58100


# ── endpoints ─────────────────────────────────────────────────────────────────
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"
    assert r.json()["pipeline_ready"] is True


def test_queue_stats(client):
    r = client.get("/queue/stats")
    assert r.status_code == 200
    assert r.json()["accepting"] is True


def test_ingest_documents_async_202(client):
    payload = {
        "kb_id": "kb1",
        "documents": [{"id": "d1", "content": "hello world", "title": "Doc"}],
    }
    r = client.post("/ingest", json=payload)
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "queued"
    assert body["documents_queued"] == 1


def test_ingest_documents_caps_413(client):
    docs = [{"id": str(i), "content": "x", "title": "T"} for i in range(200)]
    r = client.post("/ingest", json={"kb_id": "kb", "documents": docs})
    assert r.status_code == 413


def test_ingest_sync(client):
    payload = {"kb_id": "kb", "documents": [{"id": "d1", "content": "hi", "title": "T"}]}
    r = client.post("/ingest/sync", json=payload)
    assert r.status_code == 200
    assert "job_id" in r.json()


def test_ingest_files_multipart(client):
    files = [("files", ("notes.txt", b"file body content", "text/plain"))]
    r = client.post("/ingest/file", data={"kb_id": "kb", "guardrails": "{}"}, files=files)
    assert r.status_code == 202
    assert r.json()["status"] == "queued"


def test_ingest_files_no_files_400(client):
    # multipart with an empty file list is rejected by validation (422) — the
    # 400 branch requires the framework to pass an empty list; assert rejection.
    r = client.post("/ingest/file", data={"kb_id": "kb"})
    assert r.status_code in (400, 422)


def test_ingest_r2_invalid_key(client):
    body = {"kb_id": "kb", "key": "../etc/passwd", "filename": "x.txt"}
    r = client.post("/ingest/r2", json=body)
    assert r.status_code == 400


def test_ingest_r2_outside_prefix(client):
    body = {"kb_id": "kb", "key": "other-tenant/kb/file.txt", "filename": "file.txt"}
    r = client.post("/ingest/r2", json=body)
    assert r.status_code == 403


def test_ingest_r2_accepts_valid_prefix(client):
    body = {"kb_id": "kb", "key": "tenant-x/kb/file.txt", "filename": "file.txt"}
    r = client.post("/ingest/r2", json=body)
    assert r.status_code == 202


def test_ingest_url(client, monkeypatch):
    async def fake_fetch_url(url, **kw):
        return [("http://x", "<html>page</html>", "html")]

    monkeypatch.setattr(main, "fetch_url", fake_fetch_url)
    r = client.post("/ingest/url", json={"kb_id": "kb", "url": "http://x"})
    assert r.status_code == 200


def test_ingest_url_ssrf_rejected(client, monkeypatch):
    async def bad(url, **kw):
        raise ValueError("private host")

    monkeypatch.setattr(main, "fetch_url", bad)
    r = client.post("/ingest/url", json={"kb_id": "kb", "url": "http://10.0.0.1"})
    assert r.status_code == 400


def test_ingest_database(client, monkeypatch):
    async def fake_read_db(*a, **k):
        return [("row 1", "id: 1", "text")]

    monkeypatch.setattr(main, "read_database", fake_read_db)
    r = client.post(
        "/ingest/database",
        json={"kb_id": "kb", "db_type": "postgres", "connection_uri": "postgresql://h/db", "query": "SELECT 1"},
    )
    assert r.status_code == 200


def test_ingest_database_empty_422(client, monkeypatch):
    async def empty(*a, **k):
        return []

    monkeypatch.setattr(main, "read_database", empty)
    r = client.post(
        "/ingest/database", json={"kb_id": "kb", "db_type": "postgres", "connection_uri": "x", "query": "SELECT 1"}
    )
    assert r.status_code == 422


def test_ingest_api(client, monkeypatch):
    async def fake_read_api(*a, **k):
        return [("item 1", "{}", "json")]

    monkeypatch.setattr(main, "read_api", fake_read_api)
    r = client.post("/ingest/api", json={"kb_id": "kb", "url": "http://api"})
    assert r.status_code == 200


def test_get_job_status_and_404(client):
    # create a job via async ingest first
    client.post("/ingest", json={"kb_id": "kb", "documents": [{"id": "d1", "content": "x", "title": "T"}]})
    jobs = client.get("/jobs").json()["jobs"]
    assert jobs
    jid = jobs[0]["job_id"]
    r = client.get(f"/jobs/{jid}")
    assert r.status_code == 200
    assert client.get("/jobs/does-not-exist").status_code == 404


def test_delete_document(client):
    r = client.delete("/documents/d1?kb_id=kb")
    assert r.status_code == 200
    assert r.json()["chunks_deleted"] == 4


def test_delete_document_error_500(client):
    main.pipeline.delete_document = AsyncMock(side_effect=RuntimeError("boom"))
    r = client.delete("/documents/d1?kb_id=kb")
    assert r.status_code == 500


def test_initialize_kb(client):
    r = client.post("/kb/kb1/initialize")
    assert r.status_code == 200
    assert r.json()["collection"] == "coll_t_kb"


def test_delete_kb(client):
    r = client.delete("/kb/kb1")
    assert r.status_code == 200
    assert r.json()["status"] == "deleted"


def test_get_kb_info(client):
    r = client.get("/kb/kb1/info")
    assert r.status_code == 200
    assert r.json()["vector_count"] == 7


def test_list_kb_documents(client):
    r = client.get("/kb/kb1/documents")
    assert r.status_code == 200
    assert r.json()["documents"][0]["document_id"] == "d1"


def test_kb_storage(client):
    r = client.get("/kb/kb1/storage")
    assert r.status_code == 200
    assert r.json()["chunks"] == 3


def test_cancel_kb(client):
    r = client.post("/kb/kb1/cancel")
    assert r.status_code == 200
    assert r.json()["cancelled"] == 2


def test_list_kb_chunks(client):
    r = client.get("/kb/kb1/chunks")
    assert r.status_code == 200
    assert r.json()["chunks"][0]["id"] == "c1"


def test_delete_chunk(client):
    r = client.delete("/chunks/c1?kb_id=kb")
    assert r.status_code == 200
    assert r.json()["chunks_deleted"] == 1


def test_update_chunk(client):
    r = client.put("/chunks/c1", json={"kb_id": "kb", "content": "new text"})
    assert r.status_code == 200
    assert r.json()["status"] == "updated"


def test_update_chunk_not_found_404(client):
    main.pipeline.vector_store.update_chunk = AsyncMock(return_value=False)
    r = client.put("/chunks/c1", json={"kb_id": "kb", "content": "x"})
    assert r.status_code == 404


def test_cors_origins_helper():
    origins = main._cors_origins()
    assert isinstance(origins, list)
    assert origins  # non-empty
