"""Read a parsed document against a category, and produce a row.

🚨 Sits on the OTHER branch of the fork from chunk-and-embed, and before
guardrails. That order is the point: `apply_guardrails` redacts email, phone,
SSN and card before chunking, and those four are the payload of KYC, invoices,
loans and claims. A knowledge base is right to strip them; an extractor cannot
run downstream of that and find anything.

The model is the tenant's own — resolved through the same provisioned
configuration that already decides their LLM, STT and TTS. Nothing here names a
provider.
"""

from __future__ import annotations

from typing import Any

import structlog

from . import mapping, pages, prompt

logger = structlog.get_logger(__name__)

__all__ = ["ExtractionResult", "extract_document", "mapping", "pages", "prompt"]


class ExtractionResult:
    """What reading one document produced.

    A failure is a RESULT, not an exception. A batch of fifty documents with
    one scan among them must produce fifty rows — forty-nine extracted and one
    saying why it could not be — rather than a raised error that loses the
    batch and tells a recruiter nothing about which file was the problem.
    """

    __slots__ = ("failure", "note", "unmapped", "values")

    def __init__(
        self,
        *,
        values: dict[str, Any] | None = None,
        unmapped: dict[str, Any] | None = None,
        failure: str = "",
        note: str = "",
    ) -> None:
        self.values = values or {}
        self.unmapped = unmapped or {}
        #: Set only when nothing could be read. Shown to a person as-is.
        self.failure = failure
        #: A caveat about an otherwise usable row — a truncated document.
        self.note = note

    @property
    def ok(self) -> bool:
        return not self.failure

    def as_payload(self) -> dict[str, Any]:
        return {
            "values": self.values,
            "unmapped": self.unmapped,
            "failure": self.failure,
            "note": self.note,
        }


#: What a row says when the page is a scan AND the workspace's model cannot see.
#:
#: 🚨 Named as the MODEL's limit, not as ours. "Scanned documents are not
#: supported" sends somebody looking for a feature to switch on; the truth is
#: that the feature is here and the configured model does not read images, which
#: is one choice away on the Models screen.
VISION_UNAVAILABLE = (
    "This looks like a scanned page. It was rendered and sent to be read as an image, "
    "but the model configured for this workspace cannot read images — choose one that "
    "can on the Models screen, or upload a digital copy."
)


async def extract_document(
    *,
    text: str,
    category_name: str,
    fields: list[dict[str, Any]],
    complete,
    raw_bytes: bytes = b"",
) -> ExtractionResult:
    """Read one document. `complete` is an async (prompt, images=None) -> str.

    Passed in rather than constructed here so this function stays testable
    without a network, and so the worker's choice of client — which must be the
    tenant's provisioned model — is made once, at the call site that knows the
    tenant.

    `raw_bytes` is the document as it arrived, needed only for the scanned path:
    the parser has already replaced the content with text by the time this runs,
    and a scan's text is nothing.
    """
    if mapping.looks_unreadable(text):
        # 🚨 No text to read — so read the PICTURES instead of guessing.
        #
        # Sending 30 characters of header junk to a model produces a confident
        # empty object, and a row saying "nothing found" when the truth is
        # "nobody could read this" sends a reviewer looking for a value that was
        # never legible. But refusing outright was only right while nothing
        # could read a scan: every model this platform provisions reads images,
        # so the page is rendered and handed to the workspace's own model. No
        # OCR vendor, no second key, and the usage is metered like any other
        # completion because it IS one.
        logger.info("document_unreadable", chars=len(text or ""))
        return await _extract_from_images(
            raw_bytes=raw_bytes,
            category_name=category_name,
            fields=fields,
            complete=complete,
        )

    body, truncated = prompt.clip(text)
    try:
        raw = await complete(prompt.build(category_name=category_name, fields=fields, text=body))
    except Exception as exc:
        # The row says the reading failed, and the batch continues. A model
        # outage must cost the documents it touched, not the whole run.
        logger.warning("document_extraction_failed", error=str(exc)[:200])
        return ExtractionResult(failure="This document could not be read just now. Try it again.")

    found = mapping.parse_model_json(raw)
    if not found:
        logger.info("document_extraction_empty")
        return ExtractionResult(
            failure="Nothing could be read from this document.",
        )

    split = mapping.map_to_category(found, [str(f.get("key") or "") for f in fields])
    return ExtractionResult(
        values=split["values"],
        unmapped=split["unmapped"],
        note=prompt.TRUNCATION_NOTE if truncated else "",
    )


#: A 400/415/422 naming the modality is the provider saying "I cannot see".
#: Matched on the RESPONSE, never on a list of model names: such a list is
#: correct until the platform owner adds a provider, and then it is a silent
#: wrong answer in a service that has no business knowing which models exist.
_BLIND_HINTS = ("image", "vision", "multimodal", "modality", "unsupported content", "invalid_image")


def _reads_as_cannot_see(error: str) -> bool:
    low = (error or "").lower()
    return any(hint in low for hint in _BLIND_HINTS)


async def _extract_from_images(
    *,
    raw_bytes: bytes,
    category_name: str,
    fields: list[dict[str, Any]],
    complete,
) -> ExtractionResult:
    """Read a scan by rendering its pages and asking the model to look."""
    images, truncated = pages.render(raw_bytes)
    if not images:
        # Not a PDF, encrypted, corrupt, or the bytes were never kept. The
        # original message is right for all of those: there was nothing to read
        # and nothing to render.
        return ExtractionResult(failure=mapping.SCANNED_MESSAGE)

    try:
        raw = await complete(
            prompt.build_for_images(
                category_name=category_name,
                fields=fields,
                pages=len(images),
                truncated=truncated,
            ),
            images,
        )
    except Exception as exc:
        detail = str(exc)[:300]
        if _reads_as_cannot_see(detail):
            logger.info("document_vision_unsupported", error=detail)
            return ExtractionResult(failure=VISION_UNAVAILABLE)
        logger.warning("document_vision_failed", error=detail)
        return ExtractionResult(failure="This document could not be read just now. Try it again.")

    found = mapping.parse_model_json(raw)
    if not found:
        # The model looked and reported nothing. Distinct from a failure: a
        # blank or illegible scan is a real outcome, and saying so is honest.
        logger.info("document_vision_empty", pages=len(images))
        return ExtractionResult(failure="Nothing could be read from the scanned pages of this document.")

    split = mapping.map_to_category(found, [str(f.get("key") or "") for f in fields])
    return ExtractionResult(
        values=split["values"],
        unmapped=split["unmapped"],
        note=prompt.TRUNCATION_NOTE if truncated else "",
    )
