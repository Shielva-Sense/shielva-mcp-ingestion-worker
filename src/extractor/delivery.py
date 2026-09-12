"""Sending extracted rows to core-api, which owns the store.

🚨 The worker does NOT write these rows itself, and that is a security
boundary, not an accident. A collect row holds what the knowledge base redacts
— a PAN, an account number, a date of birth — and it is sealed with the
tenant's own key. That key lives in core-api, which already derives it for
screening answers and call recordings. Giving the worker the master key so it
could seal its own writes would put the fleet's most sensitive derivation in a
service whose job is parsing untrusted files.

So the worker sends plaintext over the in-cluster network to one endpoint, and
core-api seals it on arrival. Same shape as the ingest callback, and authorised
the same way: a capability URL the enqueuing side minted, signed over the
tenant and kb it is for, so a URL issued for one workspace cannot be replayed
against another.
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog

from ..retry import RetryConfig, retry_async
from .client import _tls_verify

logger = structlog.get_logger(__name__)

#: A batch of rows is one request. Bounded because a thousand-document ingest
#: would otherwise be a single enormous body, and a failed delivery would lose
#: all of it rather than one slice.
BATCH = 50


async def deliver_rows(*, callback_url: str, tenant_id: str, collect_id: str, rows: list[dict[str, Any]]) -> int:
    """Post extracted rows to core-api. Returns how many were accepted.

    🚨 Retried, unlike a progress tick. This is the ONLY path by which an
    extraction result reaches the store — the worker keeps nothing and core-api
    never polls — so a dropped delivery is a document that was read, paid for
    in tokens, and then silently lost. The ingest result callback already
    learned this lesson; the comment at the top of `jobs/webhook.py` is about
    exactly that failure.
    """
    if not callback_url or not rows:
        return 0

    sent = 0
    for start in range(0, len(rows), BATCH):
        slice_ = rows[start : start + BATCH]

        async def _post(body: list[dict[str, Any]] = slice_) -> None:
            async with httpx.AsyncClient(verify=_tls_verify(), timeout=60.0) as client:
                r = await client.post(
                    callback_url,
                    json={"tenant_id": tenant_id, "collect_id": collect_id, "rows": body},
                )
                if r.status_code >= 300:
                    raise RuntimeError(f"rows callback returned {r.status_code}")

        try:
            await retry_async(_post, config=RetryConfig(max_retries=3, base_delay=1.0))
            sent += len(slice_)
        except Exception as exc:
            # Loud, and names the count: these rows are gone, and the only
            # evidence they ever existed is this line.
            logger.error(
                "extracted_rows_delivery_failed",
                tenant_id=tenant_id,
                collect_id=collect_id,
                lost=len(slice_),
                error=str(exc)[:200],
            )
    if sent:
        logger.info("extracted_rows_delivered", tenant_id=tenant_id, collect_id=collect_id, rows=sent)
    return sent
