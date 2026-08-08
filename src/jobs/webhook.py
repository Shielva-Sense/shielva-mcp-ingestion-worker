"""
Job completion / progress webhook.

When a job carries a ``webhook_url`` (set by core-api when it enqueues the ingest),
the worker POSTs the job's current stats there — used for live progress (during
``process_job``) and once more when the job reaches a terminal state (fired by the
queue worker loop). core-api's callback endpoint turns these into the SSE events the
Knowledge UI already listens to (``ingesting`` → ``ready`` / ``failed``).

Two deliveries, two reliability contracts:

  * :func:`notify_webhook` — PROGRESS. Genuinely best-effort and single-shot. A
    dropped progress tick costs a UI refresh; the terminal delivery corrects it.
  * :func:`deliver_result` — TERMINAL. Retried with bounded exponential backoff,
    and records the outcome on the job. This is the ONLY path by which an ingest
    result reaches core-api (it never polls ``/jobs/{id}``), so dropping it leaves
    the KB stuck at ``status=failed, chunks=None`` even though the vectors indexed
    and are queryable — the system's own state made inconsistent by one warning
    line. A result that was never delivered is not a completed job.

TLS verification uses the internal CA bundle — core-api serves the dev self-signed
cert, so a bare client (the old behaviour) failed the handshake and silently dropped
the callback.
"""

from __future__ import annotations

from typing import Any, Dict

import httpx
import structlog
from shielva_common.tls import internal_ca_verify

from ..config import get_settings
from ..models import IngestionJob
from ..retry import RetryConfig, retry_async

logger = structlog.get_logger(__name__)


class WebhookRetryable(Exception):
    """Delivery failed in a way a later attempt could plausibly fix.

    Transport errors (DNS, refused connection, TLS, timeout) and receiver-side
    faults (5xx, 429). A core-api rollout or a brief network blip lands here.
    """


class WebhookPermanent(Exception):
    """The receiver rejected the payload (4xx other than 429).

    Retrying re-sends the same bytes to the same URL, so it cannot succeed —
    a bad callback token or a malformed body needs a config/code fix.
    """


def _payload(job: IngestionJob) -> Dict[str, Any]:
    return {
        "job_id": job.job_id,
        "kb_id": job.kb_id,
        "tenant_id": job.tenant_id,
        "status": job.status,
        "documents_total": job.documents_total,
        "documents_processed": job.documents_processed,
        "documents_failed": job.documents_failed,
        "chunks_created": job.chunks_created,
        "kb_file_bytes": job.kb_file_bytes,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "errors": job.errors,
    }


async def _post(job: IngestionJob, *, timeout: float) -> None:
    """One POST attempt. Raises :class:`WebhookRetryable` / :class:`WebhookPermanent`."""
    job.delivery_attempts += 1
    try:
        async with httpx.AsyncClient(verify=internal_ca_verify(), timeout=timeout) as client:
            resp = await client.post(
                job.webhook_url,
                json=_payload(job),
                headers={"X-Tenant-ID": job.tenant_id},
            )
    except Exception as exc:  # noqa: BLE001 — every transport failure is retryable
        raise WebhookRetryable(f"{type(exc).__name__}: {exc}") from exc

    if resp.status_code < 300:
        return
    detail = f"HTTP {resp.status_code}: {resp.text[:200]}"
    if resp.status_code >= 500 or resp.status_code == 429:
        raise WebhookRetryable(detail)
    raise WebhookPermanent(detail)


async def notify_webhook(job: IngestionJob) -> None:
    """Best-effort single POST of ``job`` stats — used for live PROGRESS ticks.

    Never raises: a dropped progress update must not fail or retry the ingest.
    """
    if not job.webhook_url:
        return
    try:
        await _post(job, timeout=get_settings().webhook_timeout)
    except Exception as exc:  # noqa: BLE001 — progress is best-effort
        logger.warning("ingest_webhook_failed", url=job.webhook_url, error=str(exc))


async def deliver_result(job: IngestionJob) -> bool:
    """Deliver ``job``'s TERMINAL result, retrying transient failures.

    Sets ``job.delivery_status`` and returns True when the result landed. On
    False the caller owns the undelivered job (see the queue's redelivery
    sweep) — the ingest itself is untouched, only its reporting failed.
    """
    if not job.webhook_url:
        job.delivery_status = "skipped"
        return True

    settings = get_settings()
    config = RetryConfig(
        max_retries=max(0, settings.webhook_max_attempts - 1),
        base_delay=settings.webhook_base_delay,
        max_delay=settings.webhook_max_delay,
        retryable_exceptions=(WebhookRetryable,),
    )

    try:
        # WebhookPermanent isn't in retryable_exceptions, so it propagates on the
        # first attempt instead of burning the backoff budget.
        await retry_async(_post, job, config=config, timeout=settings.webhook_timeout)
    except Exception as exc:  # noqa: BLE001 — the caller decides what to do next
        job.delivery_status = "undelivered"
        job.delivery_error = str(exc)[:300]
        # ERROR, not warning: the ingest result exists but nothing downstream
        # knows it. Stable event name — alert on it (README § Result delivery).
        logger.error(
            "ingest_result_undelivered",
            job_id=job.job_id,
            kb_id=job.kb_id,
            tenant_id=job.tenant_id,
            url=job.webhook_url,
            job_status=job.status,
            chunks_created=job.chunks_created,
            attempts=job.delivery_attempts,
            permanent=isinstance(exc, WebhookPermanent),
            error=job.delivery_error,
        )
        return False

    job.delivery_status = "delivered"
    job.delivery_error = None
    logger.info(
        "ingest_result_delivered",
        job_id=job.job_id,
        kb_id=job.kb_id,
        job_status=job.status,
        attempts=job.delivery_attempts,
    )
    return True
