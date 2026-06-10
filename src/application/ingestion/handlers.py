from __future__ import annotations

import uuid

import structlog

from src.domain.ingestion.entities import IngestionJob
from src.domain.ingestion.ports import IIngestionJobRepository
from src.domain.ingestion.value_objects import JobId, JobStatus, TenantId

from .commands import SubmitIngestionJobCommand

logger = structlog.get_logger(__name__)


class SubmitIngestionJobHandler:
    def __init__(self, repo: IIngestionJobRepository) -> None:
        self._repo = repo

    async def handle(self, cmd: SubmitIngestionJobCommand) -> IngestionJob:
        job = IngestionJob(
            id=JobId(str(uuid.uuid4())),
            tenant_id=TenantId(cmd.tenant_id),
            kb_id=cmd.kb_id,
            status=JobStatus.PENDING,
            documents_total=len(cmd.documents),
            webhook_url=cmd.webhook_url,
        )
        saved = await self._repo.save(job)
        logger.info("ingestion_job_submitted", job_id=saved.id, tenant_id=cmd.tenant_id)
        return saved
