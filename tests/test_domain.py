"""DDD layer — entities, value objects, exceptions, handler, adapters, uow."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.application.ingestion.commands import SubmitIngestionJobCommand
from src.application.ingestion.handlers import SubmitIngestionJobHandler
from src.domain.ingestion.entities import Document as DDoc
from src.domain.ingestion.entities import IngestionJob as DJob
from src.domain.ingestion.exceptions import IngestionJobNotFound, PipelineFailed
from src.domain.ingestion.value_objects import JobId, JobStatus, TenantId
from src.infrastructure.persistence.uow import UnitOfWork
from src.infrastructure.persistence.vectorstore_impl import VectorStoreImpl


def test_value_objects_and_status_enum():
    assert JobStatus.PENDING.value == "pending"
    assert JobStatus("completed") is JobStatus.COMPLETED
    assert JobId("abc") == "abc"
    assert TenantId("t") == "t"


def test_entities_defaults():
    job = DJob(id=JobId("j"), tenant_id=TenantId("t"), kb_id="kb")
    assert job.status is JobStatus.PENDING
    assert job.errors == []
    doc = DDoc(id="d", tenant_id=TenantId("t"), kb_id="kb", content="c", title="T")
    assert doc.doc_type == "text"
    assert doc.metadata == {}


def test_exceptions_carry_codes():
    nf = IngestionJobNotFound("job-9")
    assert nf.status_code == 404
    assert nf.error_code == "INGESTION_JOB_NOT_FOUND"
    assert "job-9" in nf.message
    pf = PipelineFailed("job-1", "boom")
    assert pf.error_code == "PIPELINE_FAILED"
    assert pf.status_code == 502
    assert "boom" in pf.message


async def test_submit_handler_persists_job():
    repo = MagicMock()
    saved_holder = {}

    async def save(job):
        saved_holder["job"] = job
        return job

    repo.save = save
    handler = SubmitIngestionJobHandler(repo)
    cmd = SubmitIngestionJobCommand(tenant_id="t", kb_id="kb", documents=[{"a": 1}, {"b": 2}], webhook_url="http://cb")
    job = await handler.handle(cmd)
    assert job.documents_total == 2
    assert job.status is JobStatus.PENDING
    assert job.webhook_url == "http://cb"
    assert saved_holder["job"] is job


async def test_unit_of_work_transaction_yields():
    uow = UnitOfWork()
    async with uow.transaction():
        pass  # no-op placeholder must not raise


async def test_vectorstore_impl_delegates_upsert_and_delete():
    store = MagicMock()
    store.connect = AsyncMock()
    store.close = AsyncMock()
    store.upsert = AsyncMock()
    store.delete = AsyncMock(return_value=True)
    impl = VectorStoreImpl(store)

    await impl.connect()
    await impl.close()
    store.connect.assert_awaited_once()
    store.close.assert_awaited_once()

    docs = [
        DDoc(id="d1", tenant_id=TenantId("t"), kb_id="kb", content="c1", title="T1"),
        DDoc(id="d2", tenant_id=TenantId("t"), kb_id="kb", content="c2", title="T2"),
    ]
    count = await impl.upsert(TenantId("t"), docs)
    assert count == 2
    assert store.upsert.await_count == 2

    ok = await impl.delete(TenantId("t"), "d1")
    assert ok is True
