"""Read one PDF against a field list — and do nothing else with it.

The ingest path reads a document AND indexes it: extraction is a fork off the
same parse, and the text then goes on to be chunked, embedded and written to the
vector store. That is right for a knowledge base and wrong for an identity
document, a bank statement or a company registry extract, which must never
exist as vectors and must never sit unencrypted in a bucket.

So this is the fork without the rest of the road:

* the bytes arrive in the request and live only in this process's memory;
* nothing is written to R2, the vector store, the job store or a temp file;
* the text layer is read if there is one, otherwise the pages are rendered with
  `pages.render` and read by the workspace's own model — the same
  `extract_document` the ingest fork runs, so the prompt rules are not copied;
* nothing a document says is logged: only counts, a status and the tenant.

The caller gets the fields back in the response and owns what happens to them.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import structlog

from . import BLIND, RETRY, ExtractionResult, extract_document

logger = structlog.get_logger(__name__)

#: A category that asks for more than this is not a document category.
MAX_FIELDS = 50
_MAX_KEY_CHARS = 64
_MAX_TEXT_CHARS = 200
_MAX_CATEGORY_CHARS = 120

#: Outcomes a caller acts on. A model outage is not one of them: it is a 503, so
#: a caller's retry logic sees an HTTP failure rather than a "result".
STATUS_READ = "read"
STATUS_UNREADABLE = "unreadable"
#: The workspace's model cannot read images — a configuration fact, not the
#: document's fault. The caller should route the document to a person.
STATUS_UNSUPPORTED = "unsupported"


class ExtractRejected(ValueError):
    """The request is refused before any model is called. `status` is the HTTP code."""

    def __init__(self, message: str, *, status: int) -> None:
        super().__init__(message)
        self.status = status


class ModelUnavailable(RuntimeError):
    """The model call failed in a way worth retrying later."""


def parse_fields(raw: str) -> list[dict[str, Any]]:
    """The caller's field spec, validated. Only the keys the prompt uses survive."""
    try:
        items = json.loads(raw or "")
    except (TypeError, ValueError) as exc:
        raise ExtractRejected("fields must be a JSON array", status=400) from exc
    if not isinstance(items, list) or not items:
        raise ExtractRejected("fields must be a non-empty JSON array", status=400)
    if len(items) > MAX_FIELDS:
        raise ExtractRejected(f"at most {MAX_FIELDS} fields may be asked for", status=400)
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            raise ExtractRejected("each field must be an object", status=400)
        key = item.get("key")
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", key):
            raise ExtractRejected("each field needs a key of letters, digits and underscores", status=400)
        field: dict[str, Any] = {"key": key[:_MAX_KEY_CHARS]}
        for name in ("label", "type", "hint"):
            value = item.get(name)
            if isinstance(value, str) and value:
                field[name] = value[:_MAX_TEXT_CHARS]
        if item.get("required") is True:
            field["required"] = True
        out.append(field)
    return out


def _open_pdf(raw: bytes, max_pages: int) -> int:
    """Page count, after refusing anything that is not a readable, bounded PDF."""
    import fitz  # pymupdf — already the worker's PDF parser

    try:
        doc = fitz.open(stream=raw, filetype="pdf")
    except Exception as exc:
        raise ExtractRejected("the body is not a readable PDF", status=415) from exc
    try:
        if doc.needs_pass or doc.is_encrypted:
            raise ExtractRejected("password-protected PDFs cannot be read", status=415)
        count = doc.page_count
    finally:
        doc.close()
    if count < 1:
        raise ExtractRejected("the PDF has no pages", status=415)
    if count > max_pages:
        raise ExtractRejected(f"PDFs of more than {max_pages} pages are not read", status=413)
    return count


def validate_pdf(raw: bytes, *, max_bytes: int, max_pages: int) -> int:
    if not raw:
        raise ExtractRejected("the PDF is empty", status=400)
    if len(raw) > max_bytes:
        raise ExtractRejected(f"the PDF is larger than {max_bytes} bytes", status=413)
    if not raw.startswith(b"%PDF-"):
        raise ExtractRejected("the body is not a PDF", status=415)
    return _open_pdf(raw, max_pages)


async def read_pdf(
    raw: bytes,
    *,
    category_name: str,
    fields: list[dict[str, Any]],
    complete: Any,
    max_bytes: int,
    max_pages: int,
) -> dict[str, Any]:
    """Validate, read, and return the response body. Raises `ExtractRejected` / `ModelUnavailable`."""
    from ..cleaner import TextCleaner
    from ..parser import PDFParser

    name = (category_name or "").strip()[:_MAX_CATEGORY_CHARS]
    if not name:
        raise ExtractRejected("category_name is required", status=400)
    page_count = await asyncio.to_thread(validate_pdf, raw, max_bytes=max_bytes, max_pages=max_pages)
    text = TextCleaner.clean(await PDFParser().parse(raw))
    result: ExtractionResult = await extract_document(
        text=text,
        category_name=name,
        fields=fields,
        complete=complete,
        raw_bytes=raw,
    )
    if result.reason == RETRY:
        raise ModelUnavailable(result.failure)
    if not result.failure:
        status = STATUS_READ
    elif result.reason == BLIND:
        status = STATUS_UNSUPPORTED
    else:
        status = STATUS_UNREADABLE
    logger.info("document_extract_only", pages=page_count, status=status, fields_read=len(result.values))
    return {"status": status, "pages": page_count, **result.as_payload()}
