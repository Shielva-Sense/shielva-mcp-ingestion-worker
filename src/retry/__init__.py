"""
Retry Module
Retry logic with exponential backoff.
"""

from typing import TypeVar, Callable, Any
from functools import wraps
import asyncio
import random
import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class RetryConfig:
    """Retry configuration."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: float = 0.1,
        retryable_exceptions: tuple = (Exception,),
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate delay with exponential backoff and jitter."""
    delay = min(config.base_delay * (config.exponential_base**attempt), config.max_delay)

    # Add jitter
    jitter = delay * config.jitter * random.uniform(-1, 1)

    return max(0, delay + jitter)


async def retry_async(func: Callable[..., T], *args, config: RetryConfig = None, **kwargs) -> T:
    """
    Execute async function with retry logic.

    Args:
        func: Async function to execute
        *args: Function arguments
        config: Retry configuration
        **kwargs: Function keyword arguments

    Returns:
        Function result

    Raises:
        Last exception if all retries fail
    """
    config = config or RetryConfig()
    last_exception = None

    for attempt in range(config.max_retries + 1):
        try:
            return await func(*args, **kwargs)

        except config.retryable_exceptions as e:
            last_exception = e

            if attempt < config.max_retries:
                delay = calculate_delay(attempt, config)
                logger.warning(
                    "Retry scheduled", attempt=attempt + 1, max_retries=config.max_retries, delay=delay, error=str(e)
                )
                await asyncio.sleep(delay)
            else:
                logger.error("All retries exhausted", attempts=attempt + 1, error=str(e))

    raise last_exception


def with_retry(config: RetryConfig = None):
    """
    Decorator for adding retry logic to async functions.

    Usage:
        @with_retry(RetryConfig(max_retries=3))
        async def my_function():
            ...
    """
    config = config or RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await retry_async(func, *args, config=config, **kwargs)

        return wrapper

    return decorator


class CircuitBreaker:
    """
    Circuit breaker for preventing cascading failures.

    States:
    - CLOSED: Normal operation
    - OPEN: Failing, reject requests
    - HALF_OPEN: Testing if recovered
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0, half_open_max_calls: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = "CLOSED"
        self._failure_count = 0
        self._last_failure_time = 0
        self._half_open_calls = 0

    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function through circuit breaker."""
        import time

        if self._state == "OPEN":
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                self._state = "HALF_OPEN"
                self._half_open_calls = 0
            else:
                raise Exception("Circuit breaker is OPEN")

        if self._state == "HALF_OPEN":
            if self._half_open_calls >= self.half_open_max_calls:
                raise Exception("Circuit breaker HALF_OPEN limit reached")
            self._half_open_calls += 1

        try:
            result = await func(*args, **kwargs)

            # Success - reset or close
            if self._state == "HALF_OPEN":
                self._state = "CLOSED"
            self._failure_count = 0

            return result

        except Exception:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._failure_count >= self.failure_threshold:
                self._state = "OPEN"
                logger.warning("Circuit breaker OPENED")

            raise

    @property
    def state(self) -> str:
        """Get current circuit breaker state."""
        return self._state

    def reset(self):
        """Reset circuit breaker to closed state."""
        self._state = "CLOSED"
        self._failure_count = 0


__all__ = ["RetryConfig", "retry_async", "with_retry", "calculate_delay", "CircuitBreaker"]
