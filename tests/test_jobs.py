"""Job manager, processor, and completion webhook."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import src.jobs.webhook as webhook_mod
from src.jobs.manager import JobManager
from src.jobs.processor import JobProcessor
from src.jobs.webhook import notify_webhook
from src.models import Document, IngestionJob


# ── JobManager ────────────────────────────────────────────────────────────────
def test_create_and_get_job_scoped_by_tenant():
    mgr = JobManager()
    job = mgr.create_job("tenant-a", "kb1", documents_count=2, webhook_url="http://cb")
    assert job.status == "queued"
    assert job.documents_total == 2
    assert mgr.get_job(job.job_id, "tenant-a") is job
    # cross-tenant read is denied
    assert mgr.get_job(job.job_id, "tenant-b") is None
    assert mgr.get_job("missing", "tenant-a") is None


def test_list_jobs_filters_by_status():
    mgr = JobManager()
    mgr.create_job("t", "kb", 1)
    b = mgr.create_job("t", "kb", 1)
    mgr.update_status(b.job_id, "completed")
    all_jobs = mgr.list_jobs("t")
    assert len(all_jobs) == 2
    completed = mgr.list_jobs("t", status="completed")
    assert [j.job_id for j in completed] == [b.job_id]
    assert mgr.list_jobs("other") == []


def test_update_status_sets_completed_at_and_error():
    mgr = JobManager()
    job = mgr.create_job("t", "kb", 1)
    mgr.update_status(job.job_id, "failed", error="boom")
    assert job.status == "failed"
    assert job.completed_at is not None
    assert "boom" in job.errors
    # unknown job is a no-op
    mgr.update_status("nope", "completed")


# ── JobProcessor ──────────────────────────────────────────────────────────────
async def test_process_job_success_snapshots_kb_bytes():
    pipeline = MagicMock()
    pipeline.ingest_batch = AsyncMock()
    pipeline.vector_store = MagicMock()
    pipeline.vector_store.kb_storage = AsyncMock(return_value={"file_bytes": 4096})
    proc = JobProcessor(pipeline)
    job = IngestionJob(job_id="j", tenant_id="t", kb_id="kb")
    docs = [Document(id="d", tenant_id="t", kb_id="kb", content="x", title="T")]
    await proc.process_job(job, docs)
    assert job.status == "completed"
    assert job.kb_file_bytes == 4096
    assert job.completed_at is not None


async def test_process_job_failure_marks_failed():
    pipeline = MagicMock()
    pipeline.ingest_batch = AsyncMock(side_effect=RuntimeError("pipeline broke"))
    pipeline.vector_store = MagicMock()
    pipeline.vector_store.kb_storage = AsyncMock(return_value={"file_bytes": 0})
    proc = JobProcessor(pipeline)
    job = IngestionJob(job_id="j2", tenant_id="t", kb_id="kb")
    await proc.process_job(job, [])
    assert job.status == "failed"
    assert any("pipeline broke" in e for e in job.errors)


async def test_process_job_kb_bytes_snapshot_best_effort():
    pipeline = MagicMock()
    pipeline.ingest_batch = AsyncMock()
    pipeline.vector_store = MagicMock()
    pipeline.vector_store.kb_storage = AsyncMock(side_effect=RuntimeError("stats down"))
    proc = JobProcessor(pipeline)
    job = IngestionJob(job_id="j3", tenant_id="t", kb_id="kb")
    await proc.process_job(job, [])
    # completion is still recorded even though stats failed
    assert job.status == "completed"
    assert job.kb_file_bytes == 0


# ── webhook ───────────────────────────────────────────────────────────────────
async def test_notify_webhook_noop_without_url():
    job = IngestionJob(job_id="j", tenant_id="t", kb_id="kb")  # webhook_url None
    await notify_webhook(job)  # must not raise / must not POST


async def test_notify_webhook_posts_payload(monkeypatch):
    posted = {}

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, headers=None):
            posted["url"] = url
            posted["json"] = json
            posted["headers"] = headers
            resp = MagicMock()
            resp.status_code = 200
            return resp

    monkeypatch.setattr(webhook_mod.httpx, "AsyncClient", FakeClient)
    job = IngestionJob(job_id="j", tenant_id="t", kb_id="kb", status="completed", webhook_url="http://cb/hook")
    await notify_webhook(job)
    assert posted["url"] == "http://cb/hook"
    assert posted["json"]["job_id"] == "j"
    assert posted["headers"]["X-Tenant-ID"] == "t"


async def test_notify_webhook_logs_non_2xx(monkeypatch):
    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, headers=None):
            resp = MagicMock()
            resp.status_code = 500
            resp.text = "server error"
            return resp

    monkeypatch.setattr(webhook_mod.httpx, "AsyncClient", FakeClient)
    job = IngestionJob(job_id="j", tenant_id="t", kb_id="kb", webhook_url="http://cb")
    await notify_webhook(job)  # must swallow, not raise


async def test_notify_webhook_swallows_exceptions(monkeypatch):
    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            raise RuntimeError("connect failed")

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(webhook_mod.httpx, "AsyncClient", FakeClient)
    job = IngestionJob(job_id="j", tenant_id="t", kb_id="kb", webhook_url="http://cb")
    await notify_webhook(job)  # best-effort: no raise
