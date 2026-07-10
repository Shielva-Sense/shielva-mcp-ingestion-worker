"""
Job Processor
Handles the execution of ingestion jobs.
"""

import structlog
from typing import List

from ..pipeline import IngestionPipeline
from ..models import Document, IngestionJob
from .webhook import notify_webhook

logger = structlog.get_logger(__name__)


class JobProcessor:
    """
    Processes ingestion jobs.

    Executes the ingestion pipeline for a given job and set of documents.
    """

    def __init__(self, pipeline: IngestionPipeline):
        self.pipeline = pipeline

    async def process_job(self, job: IngestionJob, documents: List[Document]):
        """
        Process a job with the given documents.
        """
        logger.info("Processing job", job_id=job.job_id, docs=len(documents))

        job.status = "processing"

        # The FINAL completion/failure webhook is fired exactly once by the queue's
        # worker loop (every ingest path goes through it). Here we only emit live
        # PROGRESS updates via on_progress so the UI can show server-side progress.
        try:
            await self.pipeline.ingest_batch(documents, job, on_progress=notify_webhook)
            job.status = "completed"
            logger.info("Job execution completed", job_id=job.job_id)
        except Exception as e:
            job.status = "failed"
            job.errors.append(str(e))
            logger.error("Job execution failed", job_id=job.job_id, error=str(e))
        finally:
            if job.completed_at is None:
                from datetime import datetime as _dt

                job.completed_at = _dt.utcnow()
            # Snapshot the KB's ABSOLUTE file-byte total so core-api can store it
            # (cheap single query; lets the Knowledge page skip a per-KB worker call).
            try:
                stats = await self.pipeline.vector_store.kb_storage(job.tenant_id, job.kb_id)
                job.kb_file_bytes = int(stats.get("file_bytes", 0))
            except Exception as exc:  # noqa: BLE001 — stats are best-effort, never fail the job
                logger.warning("kb_file_bytes snapshot failed", job_id=job.job_id, error=str(exc))
