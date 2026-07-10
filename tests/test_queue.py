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
