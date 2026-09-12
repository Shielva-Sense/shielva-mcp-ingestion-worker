"""Turning what a model read off a document into a row of a category.

The model is asked for everything it can find. This module decides which of
those findings answers a field the category declared, and what happens to the
rest — which is the whole design, so it is a pure function with no client, no
network and no I/O, and it is where the tests live.

🚨 NOTHING IS DISCARDED. A value the category never asked for goes to
`unmapped` with its page reference, not to the floor. That half of the document
is what lets a reviewer notice the schema is missing a field, and the tenth
invoice from a vendor is where you learn what the first should have asked for.
An extractor that returned only the declared keys would make the review screen
a data-entry form instead.

🚨 KEYS ARE MATCHED LOOSELY, VALUES ARE NOT. A model asked for `vendor_name`
answers with "Vendor Name", "vendorName" or "vendor name" depending on the day
and the document, and treating those as three different findings would file a
correct answer as unmapped and then report the field as missing. So keys are
normalised before matching. Values are never touched: what the row stores is
exactly the string that came off the page.
"""

from __future__ import annotations

import json
import re
from typing import Any

#: Below this many word characters, a parse produced nothing worth sending to a
#: model. Almost always a scan with no text layer.
#:
#: 🚨 The number is small on purpose. A one-line delivery note is a real
#: document; a 40-page scanned contract that yields 30 characters of header
#: junk is not. Erring low means a thin-but-real document is extracted (and
#: comes back mostly empty, which a reviewer can see and fix) rather than
#: refused with a message about scanning that is simply wrong.
MIN_TEXT_CHARS = 24

#: What a row says when there was nothing to read. Shown to a person, so it
#: names the likely cause and what it means, rather than "extraction failed".
SCANNED_MESSAGE = (
    "This looks like a scanned page with no readable text. "
    "Scanned documents are not read yet — upload a digital copy, or re-run it once OCR is switched on."
)

_WORD = re.compile(r"\w")
_KEY_NOISE = re.compile(r"[^a-z0-9]+")

#: A model asked for JSON returns JSON inside a fence about a third of the
#: time, whatever the instruction says.
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def looks_unreadable(text: str) -> bool:
    """Whether a parse produced too little to be worth a model call."""
    return len(_WORD.findall(text or "")) < MIN_TEXT_CHARS


def normalise_key(raw: str) -> str:
    """`Vendor Name` / `vendorName` / `vendor-name` → `vendor_name`.

    Used on BOTH sides of the match, so a category field and a model answer are
    compared on the same footing. Splits camelCase first, or `vendorName`
    collapses to `vendorname` and never meets `vendor_name`.
    """
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(raw or ""))
    return _KEY_NOISE.sub("_", text.lower()).strip("_")


def parse_model_json(raw: str) -> dict[str, Any]:
    """The object a model meant to return, however it wrapped it.

    Returns {} rather than raising: a model that answered in prose has
    extracted nothing, which is a row that needs review — not an exception that
    loses the other forty-nine documents in the batch.
    """
    text = (raw or "").strip()
    if not text:
        return {}
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    else:
        # A model sometimes prefixes a sentence before the object. Take from
        # the first brace to the last, which is the object either way.
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
    try:
        got = json.loads(text)
    except (ValueError, TypeError):
        return {}
    return got if isinstance(got, dict) else {}


def _one(raw: Any) -> tuple[str, float, str]:
    """One finding as (value, confidence, page_ref), in either shape a model uses.

    The rich shape is `{"value": …, "confidence": …, "page": …}`; the bare
    shape is the value itself. Both are accepted because insisting on one
    produces an empty row the day a model simplifies its own output.
    """
    if isinstance(raw, dict):
        value = raw.get("value")
        conf = raw.get("confidence")
        page = raw.get("page") or raw.get("page_ref") or ""
        try:
            confidence = max(0.0, min(1.0, float(conf))) if conf is not None else 0.0
        except (TypeError, ValueError):
            confidence = 0.0
        return (_flatten(value), confidence, str(page or ""))
    return (_flatten(raw), 0.0, "")


def _flatten(value: Any) -> str:
    """A value as the string the row stores.

    🚨 Never reformatted. An amount stays "₹1,23,456.00" and a date stays "3
    Apr 2026", because the row is the evidence: a reviewer checks it against
    the page, and a value this module tidied is one they cannot check.

    Lists and objects are the exception — they have no printed form to
    preserve, so they are JSON-encoded rather than dropped. A line-items array
    belongs in `unmapped` as something a reviewer can see, not nowhere.
    """
    if value is None or isinstance(value, bool):
        return "" if value is None else ("yes" if value else "no")
    if isinstance(value, (str, int, float)):
        return str(value).strip()
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def map_to_category(found: dict[str, Any], field_keys: list[str]) -> dict[str, dict[str, Any]]:
    """Split a model's findings into what the category asked for and what it did not.

    Returns `{"values": {...}, "unmapped": {...}}`, each entry
    `{"value", "confidence", "page_ref"}` — the shape a collected row stores.

    A finding whose value is blank is dropped from BOTH sides: a key with no
    value behind it is not an answer, and putting it in `unmapped` would fill a
    reviewer's screen with empty rows suggesting fields that do not exist.
    """
    by_normal = {normalise_key(k): k for k in field_keys}
    values: dict[str, dict[str, Any]] = {}
    unmapped: dict[str, dict[str, Any]] = {}

    for raw_key, raw_val in (found or {}).items():
        value, confidence, page_ref = _one(raw_val)
        if not value:
            continue
        entry = {"value": value, "confidence": confidence, "page_ref": page_ref}
        declared = by_normal.get(normalise_key(raw_key))
        if declared is not None:
            # 🚨 Stored under the CATEGORY's key, not the model's spelling.
            # Otherwise a row answers `Vendor Name` while the schema, the
            # export and every downstream flow read `vendor_name`.
            values[declared] = entry
        else:
            # Normalised here too, so "Transport Docket" and "transport_docket"
            # from two documents become one promotable column rather than two.
            unmapped[normalise_key(raw_key) or raw_key] = entry

    return {"values": values, "unmapped": unmapped}
