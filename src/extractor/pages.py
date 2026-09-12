"""Turn a page nobody could read into a picture the model can.

🚨 This is the whole of our OCR, and it is deliberately not OCR. A scanned PDF
has no text layer, so the parser returns header junk and the extractor refuses
to guess. The obvious next step is a hosted OCR vendor — and it is the wrong
one: every model this platform provisions reads images as first-class input, so
the page can simply be RENDERED and handed to the model the workspace already
pays for. That removes a vendor, a key, a per-page bill and a second pipeline,
and the call is metered like every other LLM call because it IS one.

PyMuPDF does the rendering and was already a dependency — it is what parses
every PDF that arrives. Nothing new is installed to make this work.

What this module will NOT do is decide whether the tenant's model can see. That
question has one right answer and it lives in the provisioned configuration, not
in a list of model names copied into the worker: such a list is correct until the
platform owner adds a provider, and then it is a silent wrong answer. The call is
attempted and a refusal becomes a row that says so.
"""

from __future__ import annotations

import base64

import structlog

logger = structlog.get_logger(__name__)

#: 150 DPI. A scanned A4 page lands near 1240×1754, which is enough for a model
#: to read body text and a stamped total. Higher costs image tokens for detail
#: that changes no answer; lower starts losing the small print that invoices put
#: their terms in.
DPI = 150

#: 🚨 A bound on COST, not on correctness. Each page is its own image and its own
#: tokens, so a 400-page scanned bundle would be one enormous bill for a category
#: that wants six fields. A document longer than this is read as far as this and
#: the row says it was truncated — the same contract the text path already has
#: for a long document.
MAX_PAGES = 8


def render(raw: bytes) -> tuple[list[str], bool]:
    """Pages of a PDF as ``data:image/png;base64,…`` URIs.

    Returns ``(images, truncated)``. An empty list means this cannot be turned
    into pictures — not a PDF, encrypted, corrupt, or PyMuPDF missing — and the
    caller falls back to saying the page could not be read. Never raises: a
    document that defeats the renderer is one row's problem.
    """
    if not raw:
        return [], False
    try:
        import fitz  # pymupdf — already present for parsing
    except ModuleNotFoundError:  # pragma: no cover — packaged dependency
        logger.warning("page_render_unavailable")
        return [], False

    try:
        doc = fitz.open(stream=raw, filetype="pdf")
    except Exception as exc:
        # Not a PDF, or one we cannot open. An image file arriving directly is
        # handled by the caller, which already has its bytes and media type.
        logger.info("page_render_not_a_pdf", error=str(exc)[:160])
        return [], False

    images: list[str] = []
    try:
        total = doc.page_count
        for index in range(min(total, MAX_PAGES)):
            try:
                pix = doc.load_page(index).get_pixmap(dpi=DPI)
                images.append("data:image/png;base64," + base64.b64encode(pix.tobytes("png")).decode("ascii"))
            except Exception as exc:
                # One bad page does not lose the others — a reviewer reading a
                # six-page contract still wants the five that rendered.
                logger.info("page_render_page_failed", page=index, error=str(exc)[:160])
        truncated = total > MAX_PAGES
        if images:
            logger.info("page_render_ok", pages=len(images), of=total, truncated=truncated)
        return images, truncated
    finally:
        try:
            doc.close()
        except Exception:
            pass


def image_part(data_uri: str) -> dict[str, object]:
    """One image as an OpenAI content part.

    The shape is the OpenAI one because that is what LiteLLM forwards to every
    provider it supports, so nothing here is specific to the model configured.
    """
    return {"type": "image_url", "image_url": {"url": data_uri}}
