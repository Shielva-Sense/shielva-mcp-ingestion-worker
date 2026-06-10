from __future__ import annotations

from src.core.errors import IntegrationException, RuntimeException


class IngestionJobNotFound(RuntimeException):
    error_code = "INGESTION_JOB_NOT_FOUND"
    status_code = 404

    def __init__(self, job_id: str) -> None:
        super().__init__(f"Ingestion job not found: {job_id}")


class PipelineFailed(IntegrationException):
    error_code = "PIPELINE_FAILED"

    def __init__(self, job_id: str, reason: str) -> None:
        super().__init__(f"Ingestion pipeline failed for job {job_id}: {reason}")
