"""Elastic ingest queue — submit/dispatch/complete, cancel, scaling, stats."""

from __future__ import annotations

import asyncio

import pytest

import src.jobs.queue as queue_mod
from src.jobs.queue import IngestQueue, QueueFull
from src.models import IngestionJob


def _job(jid="j1", kb="kb1") -> IngestionJob:
    return IngestionJob(job_id=jid, tenant_id="t", kb_id=kb)


def _make_queue(**kw) -> IngestQueue:
    params = dict(
        initial=2, min_workers=1, max_workers=4, waiting_max=3, load_high=1.0, load_low=0.7, control_interval=0.05
    )
    params.update(kw)
    return IngestQueue(**params)


async def _drain(seconds=0.2):
    await asyncio.sleep(seconds)


async def test_submit_before_start_raises():
    q = _make_queue()
    with pytest.raises(RuntimeError):
        q.submit(_job(), lambda: asyncio.sleep(0))


async def test_submit_and_run_completes():
    q = _make_queue()
    q.start()
    done = asyncio.Event()

    async def run():
        done.set()

    depth = q.submit(_job(), run)
    assert depth >= 1
    await asyncio.wait_for(done.wait(), timeout=2)
    await _drain(0.1)
    await q.stop()


async def test_queue_full_raises():
    # zero effective concurrency by never freeing slots: use a blocking run
    q = _make_queue(initial=1, min_workers=1, max_workers=1, waiting_max=1)
    q.start()
    gate = asyncio.Event()

    async def blocker():
        await gate.wait()

    q.submit(_job("running"), blocker)  # occupies the single worker
    await _drain(0.1)
    q.submit(_job("waiting"), blocker)  # fills the waiting queue (max 1)
    await _drain(0.05)
    with pytest.raises(QueueFull):
        q.submit(_job("overflow"), blocker)
    gate.set()
    await q.stop()


async def test_cancel_running_job():
    q = _make_queue(initial=1, min_workers=1, max_workers=1)
    q.start()
    started = asyncio.Event()

    async def long_run():
        started.set()
        await asyncio.sleep(30)

    job = _job("cancelme")
    q.submit(job, long_run)
    await asyncio.wait_for(started.wait(), timeout=2)
    disp = q.cancel("cancelme")
    assert disp == "cancelling"
    await _drain(0.2)
    assert job.status == "cancelled"
    await q.stop()


async def test_cancel_unknown_job():
    q = _make_queue()
    q.start()
    assert q.cancel("ghost") == "not_found"
    await q.stop()


async def test_cancel_queued_job_before_dispatch():
    q = _make_queue(initial=1, min_workers=1, max_workers=1, waiting_max=5)
    q.start()
    gate = asyncio.Event()

    async def blocker():
        await gate.wait()

    q.submit(_job("running"), blocker)
    await _drain(0.1)
    queued = _job("queued-one")
    q.submit(queued, blocker)
    disp = q.cancel("queued-one")
    assert disp == "queued"
    gate.set()
    await _drain(0.2)
    assert queued.status == "cancelled"
    await q.stop()


async def test_cancel_by_kb():
    q = _make_queue(initial=1, min_workers=1, max_workers=1)
    q.start()
    gate = asyncio.Event()

    async def blocker():
        await gate.wait()

    q.submit(_job("a", kb="shared"), blocker)
    q.submit(_job("b", kb="shared"), blocker)
    await _drain(0.1)
    n = q.cancel_by_kb("shared")
    assert n >= 1
    gate.set()
    await q.stop()


async def test_crashing_job_marked_failed():
    q = _make_queue()
    q.start()

    async def boom():
        raise ValueError("kaboom")

    job = _job("crash")
    q.submit(job, boom)
    await _drain(0.2)
    assert job.status == "failed"
    assert any("kaboom" in e for e in job.errors)
    await q.stop()


async def test_control_loop_scales_up_on_backlog(monkeypatch):
    q = _make_queue(initial=1, min_workers=1, max_workers=3, control_interval=0.05)
    monkeypatch.setattr(q, "_load_ratio", lambda: 0.0)  # headroom
    q.start()
    gate = asyncio.Event()

    async def blocker():
        await gate.wait()

    # create backlog
    for i in range(3):
        q.submit(_job(f"b{i}"), blocker)
    await _drain(0.3)
    assert q._target > 1  # scaled up under backlog + headroom
    gate.set()
    await q.stop()


async def test_control_loop_scales_down_under_load(monkeypatch):
    q = _make_queue(initial=3, min_workers=1, max_workers=3, control_interval=0.05)
    monkeypatch.setattr(q, "_load_ratio", lambda: 5.0)  # heavy load
    q.start()
    await _drain(0.3)
    assert q._target < 3
    await q.stop()


def test_load_ratio_handles_missing_getloadavg(monkeypatch):
    q = _make_queue()
    monkeypatch.setattr(queue_mod.os, "getloadavg", lambda: (_ for _ in ()).throw(OSError()))
    assert q._load_ratio() == 0.0


async def test_stats_shape():
    q = _make_queue()
    q.start()
    s = q.stats()
    assert {"target", "min", "max", "active", "waiting", "waiting_max", "accepting"} <= set(s)
    assert s["accepting"] is True
    await q.stop()


async def test_double_start_is_idempotent():
    q = _make_queue()
    q.start()
    d1 = q._dispatcher
    q.start()  # no-op
    assert q._dispatcher is d1
    await q.stop()


# ── terminal-result delivery ─────────────────────────────────────────────────
# A finished job whose callback never lands leaves core-api's KB inconsistent
# with the vector store. The queue must hold on to it, not drop it.


def _delivery_stub(outcomes):
    """Replacement for deliver_result — pops the next True/False outcome."""
    calls = []

    async def _deliver(job):
        calls.append(job.job_id)
        ok = outcomes.pop(0) if outcomes else True
        job.delivery_status = "delivered" if ok else "undelivered"
        return ok

    _deliver.calls = calls
    return _deliver


async def test_undelivered_result_is_parked_not_dropped(monkeypatch):
    monkeypatch.setattr(queue_mod, "deliver_result", _delivery_stub([False]))
    q = _make_queue(redelivery_interval=30)  # sweeper must not fire during the test
    q.start()
    job = _job("stranded")
    q.submit(job, lambda: asyncio.sleep(0))
    await _drain(0.2)

    assert job.delivery_status == "undelivered"
    stats = q.stats()
    assert stats["undelivered"] == 1
    assert stats["undelivered_total"] == 1
    assert stats["delivered_total"] == 0
    await q.stop()


async def test_delivered_result_is_not_parked(monkeypatch):
    monkeypatch.setattr(queue_mod, "deliver_result", _delivery_stub([True]))
    q = _make_queue(redelivery_interval=30)
    q.start()
    q.submit(_job("clean"), lambda: asyncio.sleep(0))
    await _drain(0.2)

    stats = q.stats()
    assert stats["undelivered"] == 0
    assert stats["delivered_total"] == 1
    await q.stop()


async def test_redelivery_sweep_heals_a_stranded_result(monkeypatch):
    # Fails on the completion path, succeeds on the sweep — a core-api rollout.
    deliver = _delivery_stub([False, True])
    monkeypatch.setattr(queue_mod, "deliver_result", deliver)
    q = _make_queue(redelivery_interval=0.05)
    q.start()
    job = _job("heals")
    q.submit(job, lambda: asyncio.sleep(0))
    await _drain(0.4)

    assert len(deliver.calls) >= 2  # retried without anyone re-uploading
    assert job.delivery_status == "delivered"
    assert q.stats()["undelivered"] == 0
    await q.stop()


async def test_redelivery_abandons_after_max_age(monkeypatch):
    monkeypatch.setattr(queue_mod, "deliver_result", _delivery_stub([False, False, False]))
    q = _make_queue(redelivery_interval=0.05, redelivery_max_age=0.0)  # instantly stale
    q.start()
    q.submit(_job("too-old"), lambda: asyncio.sleep(0))
    await _drain(0.3)

    stats = q.stats()
    assert stats["undelivered"] == 0
    assert stats["abandoned_total"] == 1
    await q.stop()


async def test_redelivery_keeps_original_failure_time(monkeypatch):
    """Re-parking must not reset the age, or a permanently unreachable
    receiver would keep the job alive forever."""
    monkeypatch.setattr(queue_mod, "deliver_result", _delivery_stub([False, False]))
    q = _make_queue(redelivery_interval=30)
    q.start()
    job = _job("keeps-age")
    q.submit(job, lambda: asyncio.sleep(0))
    await _drain(0.2)
    first_failed_at = q._undelivered[job.job_id][1]

    await q._deliver(job)  # a second failure
    assert q._undelivered[job.job_id][1] == first_failed_at
    assert q.stats()["undelivered_total"] == 1  # counted once, not per attempt
    await q.stop()
