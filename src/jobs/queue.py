"""
Elastic, load-aware async ingest executor.

Ingestion is asynchronous: an HTTP request enqueues a job and returns 202 immediately
(status="queued") — it is never held until the embed/index finishes. Three queues:

  • Waiting queue  — bounded ``asyncio.Queue`` (cap ``waiting_max``, default 100). The
                     backlog. Full → :meth:`submit` raises :class:`QueueFull` → 429.
  • Work pool      — a DYNAMIC number of concurrently-running job tasks (the "active
                     slots"). The size is NOT hardcoded: a controller floats the
                     ``target`` between ``min_workers`` and ``max_workers`` based on the
                     live 1-minute system load average per core — scale up when there's
                     backlog AND CPU headroom, scale down when the box is loaded (system
                     -wide load, so we yield to OTHER VM processes too). Downscale is
                     graceful: running jobs always finish; the dispatcher simply stops
                     admitting new ones until in-flight drops below the new target.
  • Completion queue — finished jobs (pointers) awaiting response delivery. Workers hand
                     a job off here and immediately pick up the next — they are NOT held
                     by the webhook round-trip. Dedicated notifier coroutines drain it
                     and deliver the response (webhook → core-api → SSE).
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Awaitable, Callable, List, Optional, Set

import structlog

from ..models import IngestionJob
from .webhook import notify_webhook

logger = structlog.get_logger(__name__)

RunFn = Callable[[], Awaitable[None]]


class QueueFull(Exception):
    """The waiting queue is at capacity — caller should back off (HTTP 429)."""


class _QueuedWork:
    __slots__ = ("job", "run", "enqueued_at")

    def __init__(self, job: IngestionJob, run: RunFn):
        self.job = job
        self.run = run
        self.enqueued_at = datetime.utcnow()


class IngestQueue:
    """Elastic work pool (dynamic concurrency) draining a bounded waiting queue,
    with a decoupled completion queue for response delivery."""

    _NOTIFIERS = 2  # completion-queue consumers (a completion is a fast HTTP POST)

    def __init__(
        self,
        *,
        initial: int,
        min_workers: int,
        max_workers: int,
        waiting_max: int,
        load_high: float,
        load_low: float,
        control_interval: float,
    ):
        self.waiting_max = max(1, int(waiting_max))
        self._min = max(1, int(min_workers))
        self._max = max(self._min, int(max_workers))
        self._target = min(self._max, max(self._min, int(initial)))
        self._load_high = float(load_high)
        self._load_low = float(load_low)
        self._interval = float(control_interval)
        self._cpus = os.cpu_count() or 4

        self._queue: "asyncio.Queue[_QueuedWork]" = asyncio.Queue(maxsize=self.waiting_max)
        self._completion: "asyncio.Queue[IngestionJob]" = asyncio.Queue()
        self._inflight: Set[asyncio.Task] = set()
        # Cancellation bookkeeping: the running task per job, every not-yet-completed
        # job (for kb→job lookup), and job_ids a caller has asked to cancel.
        self._task_by_job: Dict[str, asyncio.Task] = {}
        self._active_jobs: Dict[str, IngestionJob] = {}
        self._cancelled: Set[str] = set()
        self._slot_freed = asyncio.Event()
        self._dispatcher: Optional[asyncio.Task] = None
        self._controller: Optional[asyncio.Task] = None
        self._notifiers: List[asyncio.Task] = []
        self._last_load_ratio = 0.0
        self._started = False

    # ── lifecycle ───────────────────────────────────────────────────────
    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._dispatcher = asyncio.create_task(self._dispatch_loop(), name="ingest-dispatcher")
        self._controller = asyncio.create_task(self._control_loop(), name="ingest-controller")
        for i in range(self._NOTIFIERS):
            self._notifiers.append(asyncio.create_task(self._notifier_loop(i), name=f"ingest-notifier-{i}"))
        logger.info(
            "ingest_queue_started",
            initial_target=self._target, min=self._min, max=self._max,
            waiting_max=self.waiting_max, cpus=self._cpus,
            load_low=self._load_low, load_high=self._load_high,
        )

    async def stop(self) -> None:
        tasks = [self._dispatcher, self._controller, *self._notifiers, *self._inflight]
        for t in tasks:
            if t is not None:
                t.cancel()
        for t in tasks:
            if t is None:
                continue
            try:
                await t
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001 — never let shutdown raise
                pass
        self._inflight.clear()
        self._notifiers.clear()
        self._started = False
        logger.info("ingest_queue_stopped")

    # ── enqueue ─────────────────────────────────────────────────────────
    def submit(self, job: IngestionJob, run: RunFn) -> int:
        """Enqueue work for ``job``. Returns the 1-based waiting-queue depth.
        Raises :class:`QueueFull` when the waiting queue is at capacity (→ 429)."""
        if not self._started:
            raise RuntimeError("IngestQueue.submit() before start()")
        try:
            self._queue.put_nowait(_QueuedWork(job, run))
        except asyncio.QueueFull as exc:
            raise QueueFull(f"ingest waiting queue full ({self.waiting_max}); retry shortly") from exc
        self._active_jobs[job.job_id] = job
        job.status = "queued"
        waiting = self._queue.qsize()
        logger.info("ingest_job_queued", job_id=job.job_id, kb_id=job.kb_id,
                    waiting=waiting, active=len(self._inflight), target=self._target)
        return waiting

    # ── dispatcher: admit up to `target` concurrent jobs ────────────────
    async def _dispatch_loop(self) -> None:
        while True:
            # Respect the current (dynamic) target. If full, wait for a slot to free
            # OR the target to grow — the 0.5 s timeout re-checks both without a signal.
            while len(self._inflight) >= self._target:
                self._slot_freed.clear()
                try:
                    await asyncio.wait_for(self._slot_freed.wait(), timeout=0.5)
                except asyncio.TimeoutError:
                    pass
            item = await self._queue.get()
            task = asyncio.create_task(self._run_one(item))
            setattr(task, "_shielva_job_id", item.job.job_id)
            self._inflight.add(task)
            self._task_by_job[item.job.job_id] = task
            task.add_done_callback(self._on_job_done)

    def _on_job_done(self, task: asyncio.Task) -> None:
        self._inflight.discard(task)
        jid = getattr(task, "_shielva_job_id", None)
        if jid is not None:
            self._task_by_job.pop(jid, None)
        self._slot_freed.set()  # free a slot → wake the dispatcher

    async def _run_one(self, item: _QueuedWork) -> None:
        job = item.job
        try:
            # Cancelled while still waiting in the queue → skip processing entirely.
            if job.job_id in self._cancelled:
                job.status = "cancelled"
                job.errors.append("ingestion cancelled before it started")
                if job.completed_at is None:
                    job.completed_at = datetime.utcnow()
                logger.info("ingest_job_cancelled_pre_run", job_id=job.job_id, kb_id=job.kb_id)
                return
            job.status = "processing"
            await item.run()
        except asyncio.CancelledError:
            if job.job_id in self._cancelled:
                # USER-requested kill of a running job → mark + complete, swallow the
                # cancellation (it was intentional, not a shutdown).
                job.status = "cancelled"
                job.errors.append("ingestion cancelled by user")
                if job.completed_at is None:
                    job.completed_at = datetime.utcnow()
                logger.info("ingest_job_cancelled", job_id=job.job_id, kb_id=job.kb_id)
                return
            # Shutdown cancellation → propagate so stop() can drain cleanly.
            job.status = "failed"
            job.errors.append("worker cancelled (shutdown)")
            raise
        except Exception as exc:  # noqa: BLE001 — one bad job must not kill the pool
            job.status = "failed"
            job.errors.append(str(exc))
            if job.completed_at is None:
                job.completed_at = datetime.utcnow()
            logger.error("ingest_job_crashed", job_id=job.job_id, error=str(exc))
        finally:
            self._queue.task_done()
            self._active_jobs.pop(job.job_id, None)
            self._cancelled.discard(job.job_id)
            # Hand off to the completion queue and return — NOT held by the response
            # delivery. Unbounded queue → put_nowait never blocks. (A cancelled job
            # still reports completion so core-api flips the KB out of "ingesting".)
            self._completion.put_nowait(job)

    # ── cancellation ────────────────────────────────────────────────────
    def cancel(self, job_id: str) -> str:
        """Request cancellation of a job.

        Returns the disposition: ``'cancelling'`` (a running task was signalled),
        ``'queued'`` (still waiting — it'll be skipped when dispatched), or
        ``'not_found'`` (unknown / already completed).
        """
        job = self._active_jobs.get(job_id)
        if job is None:
            return "not_found"
        self._cancelled.add(job_id)
        task = self._task_by_job.get(job_id)
        if task is not None and not task.done():
            task.cancel()
            return "cancelling"
        return "queued"

    def cancel_by_kb(self, kb_id: str) -> int:
        """Cancel every active (queued or running) ingestion job for a KB.
        Returns the number of jobs signalled."""
        count = 0
        for jid, job in list(self._active_jobs.items()):
            if job.kb_id == kb_id and job.status not in ("ready", "failed", "cancelled"):
                if self.cancel(jid) != "not_found":
                    count += 1
        return count

    # ── controller: float `target` with system load ────────────────────
    def _load_ratio(self) -> float:
        try:
            return os.getloadavg()[0] / self._cpus
        except (OSError, AttributeError):
            return 0.0  # platform without loadavg → don't throttle on it

    async def _control_loop(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            ratio = self._load_ratio()
            self._last_load_ratio = ratio
            waiting = self._queue.qsize()
            active = len(self._inflight)
            old = self._target
            reason = None

            if ratio >= self._load_high and self._target > self._min:
                # Box is loaded (incl. other VM processes) → release a slot.
                self._target -= 1
                reason = "cpu_pressure"
            elif waiting > 0 and ratio < self._load_low and self._target < self._max:
                # Backlog + CPU headroom → grow.
                self._target += 1
                reason = "backlog_headroom"
            elif waiting == 0 and active < self._target and self._target > self._min:
                # Idle headroom → shed unused capacity toward the floor.
                self._target -= 1
                reason = "idle_release"

            if reason is not None:
                self._slot_freed.set()  # if we grew, let the dispatcher admit more now
                logger.info("ingest_pool_scaled", **{
                    "from": old, "to": self._target, "reason": reason,
                    "load_ratio": round(ratio, 2), "waiting": waiting, "active": active,
                })

    # ── completion / response delivery ──────────────────────────────────
    async def _notifier_loop(self, idx: int) -> None:
        while True:
            job = await self._completion.get()
            try:
                await notify_webhook(job)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — delivery is best-effort
                logger.warning("ingest_completion_notify_failed", job_id=job.job_id, error=str(exc))
            finally:
                self._completion.task_done()

    # ── introspection ───────────────────────────────────────────────────
    def stats(self) -> dict:
        return {
            "target": self._target,            # current concurrency target (dynamic)
            "min": self._min,
            "max": self._max,
            "active": len(self._inflight),     # jobs running right now
            "waiting": self._queue.qsize(),    # backlog
            "waiting_max": self.waiting_max,
            "completing": self._completion.qsize(),  # finished, awaiting delivery
            "notifiers": self._NOTIFIERS,
            "cpus": self._cpus,
            "load1": round(self._last_load_ratio * self._cpus, 2),
            "load_ratio": round(self._last_load_ratio, 2),
            "accepting": self._queue.qsize() < self.waiting_max,
        }


# Module singleton — instantiated + started in the app lifespan (needs a running loop).
ingest_queue: Optional[IngestQueue] = None
