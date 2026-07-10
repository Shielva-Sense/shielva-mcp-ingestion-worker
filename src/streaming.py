"""Streaming ingest — constant-RAM ingestion of large files from R2.

The normal path loads the whole object into ``Document.content`` and chunks the full
string → O(file size) RAM. This streams the object and runs a bounded generator chain so
peak RAM stays flat (~ one text unit + one embed batch) for any file size:

    text source  →  line-buffered guardrails  →  rolling fixed-size chunker  →
    micro-batch embed  →  index  →  drop

Two text sources share that core (``_TextChunkIngestor``):
  • **Text-like** (txt/md/html/csv/json/log): R2 read in 1 MiB byte windows → incremental
    UTF-8 decode. The file is never fully held anywhere.
  • **PDF**: streamed to a temp file (bytes on disk, not RAM), opened with PyMuPDF once,
    then **one page at a time** (fitz loads page content on demand) → per-page text. RAM
    stays flat regardless of page count.

Office formats (docx/xlsx) still need a SAX reader — until then the caller full-loads
them. All paths produce the SAME pgvector rows as the normal pipeline: chunk id
``{doc_id}_chunk_{idx}``, metadata ``{title, source_url, **doc.metadata}``, the worker's
``chunk_size``/overlap, and identical guardrail redaction/exclusion.

Robustness:
  • UTF-8 boundary safety via an ``IncrementalDecoder`` (multi-byte char split across
    windows is never corrupted).
  • PII never half-redacted: guardrails run on COMPLETE lines (partial trailing line
    buffered across units).
  • Backpressure: embed+index every ``embed_batch`` chunks, then drop — a fast reader
    can't outrun the embedder.
  • No-newline pathological stream can't grow the line buffer unbounded (capped).
  • PDF temp file is always cleaned up; doc handle used sequentially (fitz isn't
    concurrency-safe, so page extraction is serialized through ``to_thread``).
"""

from __future__ import annotations

import asyncio
import codecs
from typing import Any, Dict, List, Optional

import structlog

from src.cleaner import TextCleaner
from src.models import Document, DocumentType
from src.pipeline import apply_guardrails

log = structlog.get_logger(__name__)

# Text-like formats that can be decoded + chunked incrementally (constant RAM).
STREAMABLE_DOCTYPES = {
    DocumentType.TEXT,
    DocumentType.MARKDOWN,
    DocumentType.HTML,
    DocumentType.CSV,
    DocumentType.JSON,
}

_READ_WINDOW = 1024 * 1024  # 1 MiB byte windows pulled from R2
_DEFAULT_EMBED_BATCH = 64  # chunks embedded+indexed per flush (bounds RAM)


class RollingChunker:
    """Fixed-size char chunker with overlap, fed incrementally. Mirrors the worker's
    FIXED_SIZE chunking (``chunk_size``/``chunk_overlap``/``min_chunk_size``) but never
    holds more than ~``chunk_size`` of text. Yields ``(chunk_index, content)``."""

    def __init__(self, chunk_size: int, overlap: int, min_chunk_size: int):
        self.chunk_size = max(1, int(chunk_size))
        self.overlap = max(0, min(int(overlap), self.chunk_size - 1))  # < size, else infinite loop
        self.min_chunk_size = max(1, int(min_chunk_size))
        self._buf = ""
        self._idx = 0

    def feed(self, text: str):
        if text:
            self._buf += text
        while len(self._buf) >= self.chunk_size:
            yield self._idx, self._buf[: self.chunk_size]
            self._idx += 1
            self._buf = self._buf[self.chunk_size - self.overlap :]

    def flush(self):
        if len(self._buf) >= self.min_chunk_size:
            yield self._idx, self._buf
            self._idx += 1
        self._buf = ""


class _GuardrailLineFilter:
    """Apply ingest guardrails (PII redaction + keyword line-exclusion) on COMPLETE lines
    as text streams in. Buffers the partial trailing line across units so a line split
    across reads/pages is evaluated whole — so PII is never half-redacted."""

    _MAX_PARTIAL = 256 * 1024  # force-flush a pathological no-newline stream

    def __init__(self, guardrails: Optional[Dict[str, Any]]):
        self._gr = guardrails or {}
        self._partial = ""

    def feed(self, text: str) -> str:
        if not self._gr:
            return text  # no guardrails → pass-through, no buffering needed
        self._partial += text
        out = ""
        if "\n" in self._partial:
            head, self._partial = self._partial.rsplit("\n", 1)
            out = apply_guardrails(head + "\n", self._gr)
        if len(self._partial) > self._MAX_PARTIAL:
            out += apply_guardrails(self._partial, self._gr)
            self._partial = ""
        return out

    def flush(self) -> str:
        if not self._partial:
            return ""
        out = apply_guardrails(self._partial, self._gr) if self._gr else self._partial
        self._partial = ""
        return out


class _TextChunkIngestor:
    """Shared core: consume a stream of text units → guardrails → rolling chunker →
    micro-batch embed+index → drop. Bounded memory. Produces rows identical to the
    normal pipeline."""

    def __init__(self, *, document: Document, guardrails: Optional[Dict[str, Any]], pipeline: Any, embed_batch: int):
        self.document = document
        self.pipeline = pipeline
        self.embed_batch = max(1, int(embed_batch))
        self.chunker = RollingChunker(
            chunk_size=getattr(pipeline.chunker, "chunk_size", 512),
            overlap=getattr(pipeline.chunker, "chunk_overlap", 50),
            min_chunk_size=getattr(pipeline.chunker, "min_chunk_size", 10),
        )
        self.gfilter = _GuardrailLineFilter(guardrails)
        self.base_meta: Dict[str, Any] = {
            "title": document.title,
            "source_url": document.source_url,
            **{k: v for k, v in (document.metadata or {}).items() if k != "_guardrails"},
        }
        self._batch: List[Dict[str, Any]] = []
        self.total = 0

    async def _flush(self) -> None:
        if not self._batch:
            return
        texts = [c["content"] for c in self._batch]
        embeddings = await self.pipeline.embedding_client.embed(texts)
        for c, emb in zip(self._batch, embeddings):
            c["embedding"] = emb
        await self.pipeline.indexer.index_chunks(
            chunks=self._batch,
            tenant_id=self.document.tenant_id,
            kb_id=self.document.kb_id,
        )
        self.total += len(self._batch)
        self._batch = []

    async def _emit(self, idx: int, content: str) -> None:
        self._batch.append(
            {
                "id": f"{self.document.id}_chunk_{idx}",
                "document_id": self.document.id,
                "content": content,
                "embedding": None,
                "metadata": dict(self.base_meta),
                "chunk_index": idx,
            }
        )
        if len(self._batch) >= self.embed_batch:
            await self._flush()

    async def feed(self, text: str) -> None:
        """Feed one text unit (a decoded byte window, or one PDF page's text)."""
        cleaned = TextCleaner.clean(self.gfilter.feed(text))
        for idx, piece in self.chunker.feed(cleaned):
            await self._emit(idx, piece)

    async def finalize(self) -> int:
        for idx, piece in self.chunker.feed(TextCleaner.clean(self.gfilter.flush())):
            await self._emit(idx, piece)
        for idx, piece in self.chunker.flush():
            await self._emit(idx, piece)
        await self._flush()
        return self.total


async def stream_ingest_r2(
    *,
    bucket: str,
    key: str,
    document: Document,
    guardrails: Optional[Dict[str, Any]],
    pipeline: Any,
    embed_batch: int = _DEFAULT_EMBED_BATCH,
) -> int:
    """Stream-ingest a TEXT-like R2 object. Constant RAM; never fully loads the file."""
    from src.fetcher import _r2_client

    ing = _TextChunkIngestor(document=document, guardrails=guardrails, pipeline=pipeline, embed_batch=embed_batch)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="ignore")

    def _open_body():
        return _r2_client().get_object(Bucket=bucket, Key=key)["Body"]

    body = await asyncio.to_thread(_open_body)
    try:
        while True:
            raw = await asyncio.to_thread(body.read, _READ_WINDOW)
            if not raw:
                break
            await ing.feed(decoder.decode(raw))
        tail = decoder.decode(b"", final=True)
        if tail:
            await ing.feed(tail)
        n = await ing.finalize()
    finally:
        try:
            body.close()
        except Exception:  # noqa: BLE001
            pass
    log.info("stream_ingest_r2_done", kb_id=document.kb_id, key=key, chunks=n)
    return n


async def stream_ingest_pdf_r2(
    *,
    bucket: str,
    key: str,
    document: Document,
    guardrails: Optional[Dict[str, Any]],
    pipeline: Any,
    embed_batch: int = _DEFAULT_EMBED_BATCH,
) -> int:
    """Stream a PDF from R2 to a temp file (bytes on disk, not RAM), then ingest it ONE
    PAGE AT A TIME (fitz loads page content on demand). RAM stays flat (~one page of text
    + one embed batch) for any page count. Temp file is always removed."""
    import os
    import tempfile
    from src.fetcher import _r2_client

    ing = _TextChunkIngestor(document=document, guardrails=guardrails, pipeline=pipeline, embed_batch=embed_batch)

    def _download_to_temp() -> str:
        body = _r2_client().get_object(Bucket=bucket, Key=key)["Body"]
        fd, path = tempfile.mkstemp(suffix=".pdf")
        try:
            with os.fdopen(fd, "wb") as fh:
                while True:
                    chunk = body.read(_READ_WINDOW)
                    if not chunk:
                        break
                    fh.write(chunk)  # bytes land on disk, not RAM
        finally:
            try:
                body.close()
            except Exception:  # noqa: BLE001
                pass
        return path

    path = await asyncio.to_thread(_download_to_temp)
    doc = None
    try:
        import fitz  # PyMuPDF — same lib the normal PDF parser uses

        doc = await asyncio.to_thread(fitz.open, path)
        page_count = doc.page_count
        for i in range(page_count):
            # fitz isn't concurrency-safe — extract pages serially off the event loop.
            text = await asyncio.to_thread(lambda idx=i: doc.load_page(idx).get_text("text") or "")
            if text:
                await ing.feed(text + "\n")  # page boundary = newline (guardrail line safety)
        n = await ing.finalize()
        log.info("stream_ingest_pdf_r2_done", kb_id=document.kb_id, key=key, pages=page_count, chunks=n)
        return n
    finally:
        if doc is not None:
            try:
                await asyncio.to_thread(doc.close)
            except Exception:  # noqa: BLE001
                pass
        try:
            os.unlink(path)
        except Exception:  # noqa: BLE001
            pass


# Office (docx/xlsx) — ZIP-of-XML; SAX/streaming readers, never the whole DOM.
OFFICE_STREAMABLE_DOCTYPES = {DocumentType.DOCX, DocumentType.XLSX}

_DOCX_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _pull_batch(gen, k: int) -> List[str]:
    """Pull up to k units from a sync generator (run off-thread to amortise hops)."""
    out: List[str] = []
    for _ in range(k):
        try:
            out.append(next(gen))
        except StopIteration:
            break
    return out


def _iter_docx_paragraphs(path: str):
    """Yield one paragraph of text per <w:p>, SAX-streaming word/document.xml. Memory is
    bounded: lxml.iterparse parses incrementally and we .clear() each paragraph subtree
    (plus drop processed siblings) so the DOM never accumulates."""
    import zipfile
    from lxml import etree

    with zipfile.ZipFile(path) as z:
        if "word/document.xml" not in z.namelist():
            return
        with z.open("word/document.xml") as fh:
            para: List[str] = []
            for _ev, elem in etree.iterparse(fh, events=("end",)):
                tag = elem.tag
                if tag == _DOCX_W + "t":
                    para.append(elem.text or "")
                elif tag == _DOCX_W + "tab":
                    para.append("\t")
                elif tag == _DOCX_W + "p":
                    yield "".join(para) + "\n"
                    para = []
                    elem.clear()
                    while elem.getprevious() is not None:
                        del elem.getparent()[0]


def _iter_xlsx_rows(path: str):
    """Yield one tab-joined row of text per sheet row, streaming via openpyxl read_only
    (SAX-based; never loads the whole sheet)."""
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        for ws in wb.worksheets:
            for row in ws.iter_rows(values_only=True):
                cells = [str(c) for c in row if c is not None]
                if cells:
                    yield "\t".join(cells) + "\n"
    finally:
        try:
            wb.close()
        except Exception:  # noqa: BLE001
            pass


async def stream_ingest_office_r2(
    *,
    bucket: str,
    key: str,
    document: Document,
    guardrails: Optional[Dict[str, Any]],
    pipeline: Any,
    embed_batch: int = _DEFAULT_EMBED_BATCH,
) -> int:
    """Stream a DOCX/XLSX from R2 to a temp file (bytes on disk), then SAX/stream its XML
    one paragraph (docx) or row (xlsx) at a time → the shared chunk/embed/index core. RAM
    stays flat regardless of document size. Temp file always removed."""
    import os
    import tempfile
    from src.fetcher import _r2_client

    ing = _TextChunkIngestor(document=document, guardrails=guardrails, pipeline=pipeline, embed_batch=embed_batch)
    is_xlsx = document.doc_type == DocumentType.XLSX

    def _download_to_temp(suffix: str) -> str:
        body = _r2_client().get_object(Bucket=bucket, Key=key)["Body"]
        fd, path = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as fh:
                while True:
                    chunk = body.read(_READ_WINDOW)
                    if not chunk:
                        break
                    fh.write(chunk)
        finally:
            try:
                body.close()
            except Exception:  # noqa: BLE001
                pass
        return path

    path = await asyncio.to_thread(_download_to_temp, ".xlsx" if is_xlsx else ".docx")
    try:
        gen = _iter_xlsx_rows(path) if is_xlsx else _iter_docx_paragraphs(path)
        while True:
            # Pull a batch of units off-thread (sync parser), feed them async. Bounded:
            # ~256 small text units + one embed batch in flight at a time.
            units = await asyncio.to_thread(_pull_batch, gen, 256)
            if not units:
                break
            for unit in units:
                await ing.feed(unit)
        n = await ing.finalize()
        log.info(
            "stream_ingest_office_r2_done", kb_id=document.kb_id, key=key, kind="xlsx" if is_xlsx else "docx", chunks=n
        )
        return n
    finally:
        try:
            os.unlink(path)
        except Exception:  # noqa: BLE001
            pass
