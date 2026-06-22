"""
Job completion / progress webhook.

When a job carries a ``webhook_url`` (set by core-api when it enqueues the ingest),
the worker POSTs the job's current stats there — used for live progress (during
``process_job``) and once more when the job reaches a terminal state (fired by the
queue worker loop). core-api's callback endpoint turns these into the SSE events the
Knowledge UI already listens to (``ingesting`` → ``ready`` / ``failed``).

TLS verification uses the internal CA bundle — core-api serves the dev self-signed
cert, so a bare client (the old behaviour) failed the handshake and silently dropped
the callback.
"""
from __future__ import annotations

import httpx
import structlog
from shielva_common.tls import internal_ca_verify

from ..models import IngestionJob

logger = structlog.get_logger(__name__)


async def notify_webhook(job: IngestionJob) -> None:
    """Best-effort POST of ``job`` stats to ``job.webhook_url``. No-op when unset.

    Never raises — a webhook failure must not fail or retry the ingest itself.
    """
    if not job.webhook_url:
        return

    payload = {
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
    try:
        async with httpx.AsyncClient(verify=internal_ca_verify(), timeout=15.0) as client:
            resp = await client.post(
                job.webhook_url,
                json=payload,
                headers={"X-Tenant-ID": job.tenant_id},
            )
        if resp.status_code >= 300:
            logger.warning(
                "ingest_webhook_non_2xx",
                url=job.webhook_url,
                status=resp.status_code,
                body=resp.text[:200],
            )
    except Exception as exc:  # noqa: BLE001 — webhook is best-effort
        logger.warning("ingest_webhook_failed", url=job.webhook_url, error=str(exc))
