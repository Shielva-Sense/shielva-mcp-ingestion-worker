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

from . import mapping, prompt

logger = structlog.get_logger(__name__)

__all__ = ["ExtractionResult", "extract_document", "mapping", "prompt"]


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


async def extract_document(
    *,
    text: str,
    category_name: str,
    fields: list[dict[str, Any]],
    complete,
) -> ExtractionResult:
    """Read one document. `complete` is an async (prompt) -> str.

    Passed in rather than constructed here so this function stays testable
    without a network, and so the worker's choice of client — which must be the
    tenant's provisioned model — is made once, at the call site that knows the
    tenant.
    """
    if mapping.looks_unreadable(text):
        # 🚨 Refused BEFORE the model call. Sending 30 characters of header
        # junk produces a confident empty object, and a row that says "nothing
        # found" when the truth is "nobody could read this" sends a reviewer
        # looking for a missing value that was never legible.
        logger.info("document_unreadable", chars=len(text or ""))
        return ExtractionResult(failure=mapping.SCANNED_MESSAGE)

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
