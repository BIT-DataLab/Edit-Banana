"""
Retry decorator with exponential backoff and flexible configuration.

Provides:
- retry: Configurable retry decorator with backoff strategies
- retry_with_defaults: Convenience decorator with sensible defaults
- RetryContext: Manual retry control for complex scenarios
- get_retry_stats/reset_retry_stats: Global retry statistics
"""

import time
import asyncio
import functools
from typing import Optional, Tuple, Type, Callable, Any, Union
from enum import Enum

from ..exceptions import EditBananaException


# Global retry statistics
_retry_stats = {
    "total_attempts": 0,
    "successful_retries": 0,
    "failed_retries": 0,
}


def get_retry_stats() -> dict:
    """Get global retry statistics."""
    return _retry_stats.copy()


def reset_retry_stats():
    """Reset global retry statistics."""
    _retry_stats["total_attempts"] = 0
    _retry_stats["successful_retries"] = 0
    _retry_stats["failed_retries"] = 0


def _calculate_delay(attempt: int, base_delay: float, max_delay: float, strategy: str) -> float:
    """Calculate delay for a given attempt based on strategy."""
    if strategy == "fixed":
        delay = base_delay
    elif strategy == "linear":
        delay = base_delay * attempt
    elif strategy == "exponential":
        delay = base_delay * (2 ** (attempt - 1))
    else:
        delay = base_delay

    return min(delay, max_delay)


def _should_retry_exception(
    exc: Exception,
    exceptions_to_retry: Tuple[Type[Exception], ...],
    should_retry_fn: Optional[Callable[[Exception], bool]]
) -> bool:
    """Determine if an exception should trigger a retry."""
    # Check if exception type matches
    if not isinstance(exc, exceptions_to_retry):
        return False

    # Check EditBananaException retry_allowed flag
    if isinstance(exc, EditBananaException) and not exc.retry_allowed:
        return False

    # Apply custom predicate if provided
    if should_retry_fn is not None and not should_retry_fn(exc):
        return False

    return True


class RetryContext:
    """
    Manual retry control for complex scenarios.

    Example:
        with RetryContext(max_retries=3) as ctx:
            for attempt in ctx.attempts():
                try:
                    result = process()
                    ctx.success()
                except Exception as e:
                    ctx.failure(e)
                    if not ctx.should_retry():
                        raise
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_strategy: str = "exponential"
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_strategy = backoff_strategy
        self._attempt = 0
        self._succeeded = False
        self._failures = []

    def attempts(self):
        """Generator that yields attempt numbers."""
        for attempt in range(self.max_retries + 1):
            self._attempt = attempt
            _retry_stats["total_attempts"] += 1
            yield attempt
            if self._succeeded:
                break

    def success(self):
        """Mark the current attempt as successful."""
        self._succeeded = True
        if self._failures:
            _retry_stats["successful_retries"] += 1

    def failure(self, exc: Exception):
        """Mark the current attempt as failed."""
        self._failures.append(exc)
        _retry_stats["failed_retries"] += 1

    def should_retry(self) -> bool:
        """Check if we should retry based on current state."""
        if self._succeeded:
            return False
        return self._attempt < self.max_retries

    def get_delay(self) -> float:
        """Get the delay for the next retry."""
        return _calculate_delay(
            self._attempt + 1,  # Next attempt
            self.base_delay,
            self.max_delay,
            self.backoff_strategy
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_strategy: str = "exponential",
    exceptions_to_retry: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int, float], None]] = None,
    should_retry: Optional[Callable[[Exception], bool]] = None
):
    """
    Retry decorator with configurable backoff strategies.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay cap (seconds)
        backoff_strategy: 'fixed', 'linear', or 'exponential'
        exceptions_to_retry: Tuple of exception types to retry on
        on_retry: Callback called on each retry: (exc, attempt, delay)
        should_retry: Optional predicate to filter exceptions

    Returns:
        Decorated function that will retry on failure
    """
    def decorator(func):
        is_async = asyncio.iscoroutinefunction(func)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                _retry_stats["total_attempts"] += 1

                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        _retry_stats["successful_retries"] += 1
                    return result
                except Exception as e:
                    last_exception = e

                    # Check if we should retry this exception
                    if not _should_retry_exception(e, exceptions_to_retry, should_retry):
                        raise

                    # Check if we've exhausted retries
                    if attempt >= max_retries:
                        _retry_stats["failed_retries"] += 1
                        raise

                    # Calculate delay
                    delay = _calculate_delay(attempt + 1, base_delay, max_delay, backoff_strategy)

                    # Call callback if provided
                    if on_retry:
                        on_retry(e, attempt + 1, delay)

                    # Wait before retry
                    time.sleep(delay)

            # Should never reach here, but just in case
            raise last_exception if last_exception else Exception("Retry failed")

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                _retry_stats["total_attempts"] += 1

                try:
                    result = await func(*args, **kwargs)
                    if attempt > 0:
                        _retry_stats["successful_retries"] += 1
                    return result
                except Exception as e:
                    last_exception = e

                    # Check if we should retry this exception
                    if not _should_retry_exception(e, exceptions_to_retry, should_retry):
                        raise

                    # Check if we've exhausted retries
                    if attempt >= max_retries:
                        _retry_stats["failed_retries"] += 1
                        raise

                    # Calculate delay
                    delay = _calculate_delay(attempt + 1, base_delay, max_delay, backoff_strategy)

                    # Call callback if provided
                    if on_retry:
                        on_retry(e, attempt + 1, delay)

                    # Wait before retry
                    await asyncio.sleep(delay)

            # Should never reach here, but just in case
            raise last_exception if last_exception else Exception("Retry failed")

        return async_wrapper if is_async else sync_wrapper
    return decorator


def retry_with_defaults(
    _func: Optional[Callable] = None,
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0
):
    """
    Convenience decorator with sensible defaults for EditBanana.

    Can be used with or without parentheses:
        @retry_with_defaults
        def my_func(): ...

        @retry_with_defaults(max_retries=5)
        def my_func(): ...
    """
    def decorator(func):
        return retry(
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            backoff_strategy="exponential",
            exceptions_to_retry=(EditBananaException,)
        )(func)

    if _func is not None:
        return decorator(_func)
    return decorator
