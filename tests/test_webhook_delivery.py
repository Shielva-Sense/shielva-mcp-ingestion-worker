"""Terminal-result delivery — retry, classification, and job delivery state.

Regression cover for the ARC incident (2026-08-08): a KB indexed 129/129 chunks,
the completion callback could not connect, and the worker logged one warning and
recorded the job as completed — leaving the KB at `status=failed, chunks=None`
with no signal anywhere that the result had never been delivered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import pytest

import src.jobs.webhook as webhook_mod
from src.jobs.webhook import WebhookPermanent, WebhookRetryable, deliver_result, notify_webhook
from src.models import IngestionJob

WEBHOOK = "https://core-api.internal/api/v1/knowledge/internal/ingest-callback?token=t"


@dataclass
class _FakeSettings:
    """Same shape as the delivery slice of IngestionSettings, but instant."""

    webhook_max_attempts: int = 3
    webhook_base_delay: float = 0.0
    webhook_max_delay: float = 0.0
    webhook_timeout: float = 1.0


@pytest.fixture()
def fast_settings(monkeypatch: pytest.MonkeyPatch) -> _FakeSettings:
    settings = _FakeSettings()
    monkeypatch.setattr(webhook_mod, "get_settings", lambda: settings)
    return settings


def _job(status: str = "completed", url: Optional[str] = WEBHOOK) -> IngestionJob:
    return IngestionJob(
        job_id="job-1",
        tenant_id="tenant-1",
        kb_id="kb-1",
        status=status,
        chunks_created=129,
        documents_processed=1,
        webhook_url=url,
    )


class _FakeResponse:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


class _FakeClient:
    """Stands in for httpx.AsyncClient — yields queued outcomes in order."""

    def __init__(self, outcomes: List[Any]) -> None:
        self._outcomes = outcomes
        self.posts: List[dict] = []

    def __call__(self, **_kw: Any) -> "_FakeClient":
        return self

    async def __aenter__(self) -> "_FakeClient":
        return self

    async def __aexit__(self, *_exc: Any) -> bool:
        return False

    async def post(self, url: str, json: dict, headers: dict) -> _FakeResponse:
        self.posts.append({"url": url, "json": json, "headers": headers})
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _install_client(monkeypatch: pytest.MonkeyPatch, outcomes: List[Any]) -> _FakeClient:
    client = _FakeClient(outcomes)
    monkeypatch.setattr(webhook_mod.httpx, "AsyncClient", client)
    return client


# ── _post: outcome classification ────────────────────────────────────────────
async def test_post_transport_error_is_retryable(monkeypatch, fast_settings):
    _install_client(monkeypatch, [ConnectionRefusedError("connection refused")])
    with pytest.raises(WebhookRetryable):
        await webhook_mod._post(_job(), timeout=1.0)


@pytest.mark.parametrize("code", [500, 502, 503, 429])
async def test_post_receiver_faults_are_retryable(monkeypatch, fast_settings, code):
    _install_client(monkeypatch, [_FakeResponse(code, "nope")])
    with pytest.raises(WebhookRetryable):
        await webhook_mod._post(_job(), timeout=1.0)


@pytest.mark.parametrize("code", [400, 401, 404])
async def test_post_client_errors_are_permanent(monkeypatch, fast_settings, code):
    _install_client(monkeypatch, [_FakeResponse(code, "bad token")])
    with pytest.raises(WebhookPermanent):
        await webhook_mod._post(_job(), timeout=1.0)


async def test_post_sends_the_terminal_stats(monkeypatch, fast_settings):
    client = _install_client(monkeypatch, [_FakeResponse(200)])
    job = _job()
    await webhook_mod._post(job, timeout=1.0)
    body = client.posts[0]["json"]
    assert body["kb_id"] == "kb-1"
    assert body["status"] == "completed"
    assert body["chunks_created"] == 129
    assert client.posts[0]["headers"]["X-Tenant-ID"] == "tenant-1"


# ── deliver_result: the terminal contract ────────────────────────────────────
async def test_deliver_result_success_marks_delivered(monkeypatch, fast_settings):
    _install_client(monkeypatch, [_FakeResponse(200)])
    job = _job()
    assert await deliver_result(job) is True
    assert job.delivery_status == "delivered"
    assert job.delivery_attempts == 1
    assert job.delivery_error is None


async def test_deliver_result_retries_transient_failure(monkeypatch, fast_settings):
    _install_client(
        monkeypatch,
        [ConnectionRefusedError("blip"), _FakeResponse(503), _FakeResponse(200)],
    )
    job = _job()
    assert await deliver_result(job) is True
    assert job.delivery_status == "delivered"
    assert job.delivery_attempts == 3


async def test_deliver_result_exhausted_marks_undelivered(monkeypatch, fast_settings):
    """The incident: ingest succeeded, callback never landed. The job must NOT
    read as a clean completion — the pipeline status stays truthful, and the
    delivery outcome is recorded alongside it."""
    _install_client(monkeypatch, [ConnectionRefusedError("no route") for _ in range(3)])
    job = _job()
    assert await deliver_result(job) is False
    assert job.status == "completed"  # the vectors really are indexed
    assert job.delivery_status == "undelivered"
    assert job.delivery_attempts == fast_settings.webhook_max_attempts
    assert "no route" in (job.delivery_error or "")


async def test_deliver_result_does_not_retry_permanent_failure(monkeypatch, fast_settings):
    client = _install_client(monkeypatch, [_FakeResponse(401, "invalid ingest callback token")])
    job = _job()
    assert await deliver_result(job) is False
    assert job.delivery_status == "undelivered"
    assert len(client.posts) == 1  # re-sending the same bytes cannot help


async def test_deliver_result_without_webhook_url_is_skipped(fast_settings):
    job = _job(url=None)
    assert await deliver_result(job) is True
    assert job.delivery_status == "skipped"
    assert job.delivery_attempts == 0


# ── notify_webhook: progress stays best-effort ───────────────────────────────
async def test_progress_webhook_is_single_shot_and_never_raises(monkeypatch, fast_settings):
    client = _install_client(monkeypatch, [ConnectionRefusedError("blip")])
    job = _job(status="processing")
    await notify_webhook(job)  # must not raise
    assert len(client.posts) == 1
    assert job.delivery_status == "pending"  # progress never decides the verdict
