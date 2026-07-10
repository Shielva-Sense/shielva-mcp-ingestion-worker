"""Ingestion pipeline — guardrails + full ingest_document / ingest_batch."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models import Document, DocumentType, IngestionJob
from src.pipeline import IngestionPipeline, apply_guardrails


# ── guardrails ────────────────────────────────────────────────────────────────
def test_apply_guardrails_noop_without_config():
    assert apply_guardrails("hello", {}) == "hello"
    assert apply_guardrails("", {"redact_pii": True}) == ""


def test_apply_guardrails_redacts_pii():
    text = "email a@b.com phone 123-456-7890 ssn 123-45-6789"
    out = apply_guardrails(text, {"redact_pii": True})
    assert "[REDACTED_EMAIL]" in out
    assert "[REDACTED_PHONE]" in out
    assert "[REDACTED_SSN]" in out
    assert "a@b.com" not in out


def test_apply_guardrails_excludes_keyword_lines():
    text = "keep this line\nsecret password here\nkeep this too"
    out = apply_guardrails(text, {"exclude_keywords": ["password"]})
    assert "secret password" not in out
    assert "keep this line" in out


def _pipeline(embed_client, indexer):
    return IngestionPipeline(
        vector_store=MagicMock(),
        embedding_client=embed_client,
        chunker=None,
        indexer=indexer,
        fetcher=MagicMock(),
    )


async def test_ingest_document_full_flow(fake_embedding_client, fake_indexer):
    pipe = _pipeline(fake_embedding_client, fake_indexer)
    doc = Document(
        id="d1", tenant_id="t", kb_id="kb", content="Hello world. " * 60, title="T", doc_type=DocumentType.TEXT
    )
    n = await pipe.ingest_document(doc)
    assert n > 0
    assert fake_indexer.calls  # indexer received chunks


async def test_ingest_document_applies_guardrails(fake_embedding_client, fake_indexer):
    pipe = _pipeline(fake_embedding_client, fake_indexer)
    doc = Document(
        id="d2",
        tenant_id="t",
        kb_id="kb",
        content="contact me at secret@evil.com anytime for the deal",
        title="T",
        doc_type=DocumentType.TEXT,
        metadata={"_guardrails": {"redact_pii": True}},
    )
    await pipe.ingest_document(doc)
    assert "secret@evil.com" not in doc.content
    assert "[REDACTED_EMAIL]" in doc.content


async def test_ingest_document_fetches_when_no_content(fake_embedding_client, fake_indexer):
    pipe = _pipeline(fake_embedding_client, fake_indexer)
    fetch_res = MagicMock()
    fetch_res.content = b"fetched body text that is reasonably long"
    pipe.fetcher = MagicMock()
    pipe.fetcher.fetch = AsyncMock(return_value=fetch_res)
    doc = Document(
        id="d3", tenant_id="t", kb_id="kb", content="", title="T", source_url="http://x", doc_type=DocumentType.TEXT
    )
    n = await pipe.ingest_document(doc)
    pipe.fetcher.fetch.assert_awaited_once()
    assert n >= 1


async def test_ingest_document_empty_content_no_chunks(fake_embedding_client, fake_indexer):
    pipe = _pipeline(fake_embedding_client, fake_indexer)
    doc = Document(id="d4", tenant_id="t", kb_id="kb", content="", title="T", doc_type=DocumentType.TEXT)
    n = await pipe.ingest_document(doc)
    assert n == 0


async def test_ingest_document_reraises_parser_error(fake_embedding_client, fake_indexer):
    pipe = _pipeline(fake_embedding_client, fake_indexer)
    doc = Document(
        id="d5", tenant_id="t", kb_id="kb", content="x" * 100, title="T", doc_type=DocumentType.PDF
    )  # PDF parser on garbage -> error

    with pytest.raises(Exception):
        await pipe.ingest_document(doc)


async def test_ingest_batch_tracks_progress(fake_embedding_client, fake_indexer):
    pipe = _pipeline(fake_embedding_client, fake_indexer)
    progress = []

    async def on_progress(job):
        progress.append(job.documents_processed)

    docs = [
        Document(
            id=f"d{i}",
            tenant_id="t",
            kb_id="kb",
            content="Some text here. " * 40,
            title="T",
            doc_type=DocumentType.TEXT,
        )
        for i in range(3)
    ]
    job = IngestionJob(job_id="j", tenant_id="t", kb_id="kb")
    out = await pipe.ingest_batch(docs, job, on_progress=on_progress)
    assert out.status == "completed"
    assert out.documents_processed == 3
    assert out.chunks_created > 0
    assert progress  # callback fired


async def test_ingest_batch_records_failures(fake_embedding_client, fake_indexer):
    pipe = _pipeline(fake_embedding_client, fake_indexer)
    docs = [Document(id="bad", tenant_id="t", kb_id="kb", content="x" * 50, title="T", doc_type=DocumentType.PDF)]
    job = IngestionJob(job_id="j2", tenant_id="t", kb_id="kb")
    out = await pipe.ingest_batch(docs, job)
    assert out.documents_failed == 1
    assert out.errors


async def test_delete_document_delegates(fake_embedding_client, fake_indexer):
    pipe = _pipeline(fake_embedding_client, fake_indexer)
    n = await pipe.delete_document("d1", "t", "kb")
    assert n == 3  # FakeIndexer.delete_by_document
