"""Retry backoff + circuit breaker."""

from __future__ import annotations

import asyncio

import pytest

from src.retry import (
    CircuitBreaker,
    RetryConfig,
    calculate_delay,
    retry_async,
    with_retry,
)


def test_calculate_delay_grows_exponentially_and_caps():
    cfg = RetryConfig(base_delay=1.0, exponential_base=2.0, max_delay=8.0, jitter=0.0)
    assert calculate_delay(0, cfg) == 1.0
    assert calculate_delay(1, cfg) == 2.0
    assert calculate_delay(2, cfg) == 4.0
    assert calculate_delay(10, cfg) == 8.0  # capped


def test_calculate_delay_never_negative_with_jitter():
    cfg = RetryConfig(base_delay=1.0, jitter=1.0)
    for _ in range(50):
        assert calculate_delay(0, cfg) >= 0.0


async def test_retry_async_succeeds_after_transient_failures():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("transient")
        return "ok"

    cfg = RetryConfig(max_retries=5, base_delay=0.0, jitter=0.0)
    result = await retry_async(flaky, config=cfg)
    assert result == "ok"
    assert calls["n"] == 3


async def test_retry_async_raises_after_exhaustion():
    async def always_fail():
        raise KeyError("nope")

    cfg = RetryConfig(max_retries=2, base_delay=0.0, jitter=0.0, retryable_exceptions=(KeyError,))
    with pytest.raises(KeyError):
        await retry_async(always_fail, config=cfg)


async def test_with_retry_decorator():
    state = {"n": 0}

    @with_retry(RetryConfig(max_retries=3, base_delay=0.0, jitter=0.0))
    async def sometimes(x):
        state["n"] += 1
        if state["n"] < 2:
            raise RuntimeError("retry me")
        return x * 2

    assert await sometimes(5) == 10


async def test_circuit_breaker_opens_after_threshold():
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=30.0)

    async def boom():
        raise ValueError("down")

    for _ in range(2):
        with pytest.raises(ValueError):
            await cb.call(boom)
    assert cb.state == "OPEN"
    # While OPEN, calls are rejected fast without invoking func
    with pytest.raises(Exception):
        await cb.call(boom)


async def test_circuit_breaker_half_open_recovers_to_closed():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.0, half_open_max_calls=3)

    async def boom():
        raise ValueError("x")

    with pytest.raises(ValueError):
        await cb.call(boom)
    assert cb.state == "OPEN"

    async def ok():
        return "recovered"

    # recovery_timeout=0 -> immediately transitions to HALF_OPEN then CLOSED on success
    result = await cb.call(ok)
    assert result == "recovered"
    assert cb.state == "CLOSED"


async def test_circuit_breaker_reset():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=999.0)

    async def boom():
        raise ValueError("x")

    with pytest.raises(ValueError):
        await cb.call(boom)
    assert cb.state == "OPEN"
    cb.reset()
    assert cb.state == "CLOSED"
