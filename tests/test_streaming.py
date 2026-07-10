"""Streaming ingest — rolling chunker, guardrail line filter, R2/PDF/office streams."""

from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import src.streaming as streaming
from src.models import Document, DocumentType
from src.streaming import (
    RollingChunker,
    _GuardrailLineFilter,
    _pull_batch,
    _TextChunkIngestor,
    stream_ingest_office_r2,
    stream_ingest_pdf_r2,
    stream_ingest_r2,
)


# ── RollingChunker ────────────────────────────────────────────────────────────
def test_rolling_chunker_emits_fixed_size_with_overlap():
    rc = RollingChunker(chunk_size=5, overlap=1, min_chunk_size=1)
    out = list(rc.feed("abcdefghij"))
    assert out[0] == (0, "abcde")
    # overlap keeps last char -> next window starts at index 4
    assert all(len(content) == 5 for _, content in out)


def test_rolling_chunker_overlap_capped_below_size():
    rc = RollingChunker(chunk_size=3, overlap=10, min_chunk_size=1)
    assert rc.overlap == 2  # capped to size-1 to avoid infinite loop


def test_rolling_chunker_flush_emits_remainder():
    rc = RollingChunker(chunk_size=10, overlap=0, min_chunk_size=2)
    list(rc.feed("abc"))  # under size, nothing emitted yet
    out = list(rc.flush())
    assert out == [(0, "abc")]


def test_rolling_chunker_flush_drops_tiny_tail():
    rc = RollingChunker(chunk_size=10, overlap=0, min_chunk_size=5)
    list(rc.feed("ab"))
    assert list(rc.flush()) == []


# ── guardrail line filter ─────────────────────────────────────────────────────
def test_guardrail_filter_passthrough_without_rules():
    gf = _GuardrailLineFilter(None)
    assert gf.feed("anything") == "anything"
    assert gf.flush() == ""


def test_guardrail_filter_redacts_complete_lines():
    gf = _GuardrailLineFilter({"redact_pii": True})
    out = gf.feed("email a@b.com here\npartial")
    assert "[REDACTED_EMAIL]" in out
    # trailing partial line is buffered, flushed at the end
    tail = gf.flush()
    assert "partial" in tail


def test_guardrail_filter_force_flushes_pathological_line(monkeypatch):
    monkeypatch.setattr(_GuardrailLineFilter, "_MAX_PARTIAL", 5)
    gf = _GuardrailLineFilter({"redact_pii": True})
    out = gf.feed("abcdefghij")  # no newline, exceeds max -> force flush
    assert "abcdefghij" in out
    assert gf.flush() == ""


# ── _TextChunkIngestor ────────────────────────────────────────────────────────
def _fake_pipeline(embed_client, indexer, chunk_size=8, overlap=0, min_size=1):
    chunker = SimpleNamespace(chunk_size=chunk_size, chunk_overlap=overlap, min_chunk_size=min_size)
    return SimpleNamespace(chunker=chunker, embedding_client=embed_client, indexer=indexer)


async def test_text_chunk_ingestor_feeds_and_finalizes(fake_embedding_client, fake_indexer):
    doc = Document(
        id="d1",
        tenant_id="t",
        kb_id="kb",
        content="",
        title="T",
        source_url="http://x",
        metadata={"a": 1, "_guardrails": {}},
    )
    pipe = _fake_pipeline(fake_embedding_client, fake_indexer, chunk_size=6, min_size=1)
    ing = _TextChunkIngestor(document=doc, guardrails=None, pipeline=pipe, embed_batch=2)
    await ing.feed("abcdefghijkl")  # 12 chars -> 2 chunks of 6
    total = await ing.finalize()
    assert total >= 2
    assert fake_indexer.calls
    # base_meta strips _guardrails, keeps title/source_url/a
    assert "_guardrails" not in ing.base_meta
    assert ing.base_meta["title"] == "T"
    assert ing.base_meta["a"] == 1


async def test_text_chunk_ingestor_flush_noop_on_empty(fake_embedding_client, fake_indexer):
    doc = Document(id="d1", tenant_id="t", kb_id="kb", content="", title="T")
    pipe = _fake_pipeline(fake_embedding_client, fake_indexer)
    ing = _TextChunkIngestor(document=doc, guardrails=None, pipeline=pipe, embed_batch=4)
    total = await ing.finalize()
    assert total == 0


# ── stream_ingest_r2 ──────────────────────────────────────────────────────────
def _mock_r2_body(chunks):
    body = MagicMock()
    seq = list(chunks) + [b""]
    body.read.side_effect = seq
    return body


async def test_stream_ingest_r2_text(monkeypatch, fake_embedding_client, fake_indexer):
    body = _mock_r2_body([b"Hello streaming world. ", b"More text here too."])
    client = MagicMock()
    client.get_object.return_value = {"Body": body}
    monkeypatch.setattr(streaming, "_r2_client", lambda: client, raising=False)
    # streaming imports _r2_client from src.fetcher inside the function
    import src.fetcher as fetcher

    monkeypatch.setattr(fetcher, "_r2_client", lambda: client)

    doc = Document(id="d1", tenant_id="t", kb_id="kb", content="", title="T", doc_type=DocumentType.TEXT)
    pipe = _fake_pipeline(fake_embedding_client, fake_indexer, chunk_size=8, min_size=1)
    n = await stream_ingest_r2(bucket="b", key="k", document=doc, guardrails=None, pipeline=pipe, embed_batch=2)
    assert n > 0
    body.close.assert_called_once()


async def test_stream_ingest_pdf_r2(monkeypatch, tmp_path, fake_embedding_client, fake_indexer):
    fitz = pytest.importorskip("fitz")
    # build a real 1-page pdf
    d = fitz.open()
    page = d.new_page()
    page.insert_text((72, 72), "Streamed PDF page content here for chunking")
    pdf_bytes = d.tobytes()
    d.close()

    body = _mock_r2_body([pdf_bytes])
    client = MagicMock()
    client.get_object.return_value = {"Body": body}
    import src.fetcher as fetcher

    monkeypatch.setattr(fetcher, "_r2_client", lambda: client)

    doc = Document(id="d1", tenant_id="t", kb_id="kb", content="", title="T", doc_type=DocumentType.PDF)
    pipe = _fake_pipeline(fake_embedding_client, fake_indexer, chunk_size=10, min_size=1)
    n = await stream_ingest_pdf_r2(bucket="b", key="k", document=doc, guardrails=None, pipeline=pipe, embed_batch=4)
    assert n > 0


def test_pull_batch_stops_at_stopiteration():
    gen = iter(["a", "b"])
    assert _pull_batch(gen, 5) == ["a", "b"]


async def test_stream_ingest_office_docx(monkeypatch, fake_embedding_client, fake_indexer):
    docx = pytest.importorskip("docx")
    d = docx.Document()
    for i in range(10):
        d.add_paragraph(f"Paragraph number {i} with some words in it")
    buf = io.BytesIO()
    d.save(buf)
    docx_bytes = buf.getvalue()

    body = _mock_r2_body([docx_bytes])
    client = MagicMock()
    client.get_object.return_value = {"Body": body}
    import src.fetcher as fetcher

    monkeypatch.setattr(fetcher, "_r2_client", lambda: client)

    doc = Document(id="d1", tenant_id="t", kb_id="kb", content="", title="T", doc_type=DocumentType.DOCX)
    pipe = _fake_pipeline(fake_embedding_client, fake_indexer, chunk_size=20, min_size=1)
    n = await stream_ingest_office_r2(bucket="b", key="k", document=doc, guardrails=None, pipeline=pipe, embed_batch=4)
    assert n > 0


async def test_stream_ingest_office_xlsx(monkeypatch, fake_embedding_client, fake_indexer):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    for i in range(10):
        ws.append([f"row{i}", f"value{i}"])
    buf = io.BytesIO()
    wb.save(buf)
    xlsx_bytes = buf.getvalue()

    body = _mock_r2_body([xlsx_bytes])
    client = MagicMock()
    client.get_object.return_value = {"Body": body}
    import src.fetcher as fetcher

    monkeypatch.setattr(fetcher, "_r2_client", lambda: client)

    doc = Document(id="d1", tenant_id="t", kb_id="kb", content="", title="T", doc_type=DocumentType.XLSX)
    pipe = _fake_pipeline(fake_embedding_client, fake_indexer, chunk_size=15, min_size=1)
    n = await stream_ingest_office_r2(bucket="b", key="k", document=doc, guardrails=None, pipeline=pipe, embed_batch=4)
    assert n > 0
