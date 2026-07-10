"""Data models + core exception hierarchy + FastAPI error handlers."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.core.error_handlers import install_exception_handlers
from src.core.errors import (
    IntegrationException,
    RuntimeException,
    ShielvaException,
    TechnicalException,
)
from src.models import (
    Chunk,
    ChunkingStrategy,
    Document,
    DocumentType,
    IngestionJob,
)


def test_document_type_and_strategy_values():
    assert DocumentType.PDF.value == "pdf"
    assert DocumentType("json") is DocumentType.JSON
    assert ChunkingStrategy.RECURSIVE.value == "recursive"


def test_document_defaults():
    doc = Document(id="1", tenant_id="t", kb_id="kb", content="c", title="T")
    assert doc.doc_type is DocumentType.TEXT
    assert doc.metadata == {}
    assert doc.source_url is None
    assert doc.created_at is not None


def test_chunk_defaults():
    ch = Chunk(id="c1", document_id="d1", content="body")
    assert ch.embedding is None
    assert ch.chunk_index == 0
    assert ch.metadata == {}


def test_ingestion_job_defaults():
    job = IngestionJob(job_id="j", tenant_id="t", kb_id="kb")
    assert job.status == "pending"
    assert job.errors == []
    assert job.kb_file_bytes == 0


def test_exception_class_attributes():
    assert IntegrationException("x").status_code == 502
    assert IntegrationException("x").retryable is True
    assert RuntimeException("x").status_code == 400
    assert TechnicalException("x").retryable is False
    exc = ShielvaException("boom", detail="because")
    assert exc.message == "boom"
    assert exc.detail == "because"
    assert str(exc) == "boom"


def _client_with_handlers() -> TestClient:
    app = FastAPI()
    install_exception_handlers(app)

    @app.get("/shielva")
    async def _shielva():
        raise IntegrationException("integration down", detail="upstream 500")

    @app.get("/http")
    async def _http():
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="missing")

    @app.get("/boom")
    async def _boom():
        raise ValueError("unexpected")

    @app.get("/validate/{n}")
    async def _validate(n: int):
        return {"n": n}

    return TestClient(app, raise_server_exceptions=False)


def test_shielva_exception_handler_maps_status_and_body():
    client = _client_with_handlers()
    r = client.get("/shielva")
    assert r.status_code == 502
    body = r.json()["error"]
    assert body["code"] == "INTEGRATION_ERROR"
    assert body["retryable"] is True
    assert body["detail"] == "upstream 500"


def test_http_exception_handler():
    r = _client_with_handlers().get("/http")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "HTTP_ERROR"


def test_validation_exception_handler():
    r = _client_with_handlers().get("/validate/not-an-int")
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "VALIDATION_ERROR"


def test_generic_exception_handler():
    r = _client_with_handlers().get("/boom")
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "INTERNAL_ERROR"
