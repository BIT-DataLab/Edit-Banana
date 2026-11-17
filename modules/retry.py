"""
Retry decorator with exponential backoff for EditBanana.

Provides intelligent retry logic that respects exception types and
uses exponential backoff to handle transient failures.
"""

import time
import functools
import logging
from typing import Callable, Optional, Tuple, Type

from modules.exceptions import EditBananaException

# Configure logging
logger = logging.getLogger(__name__)


def retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
    on_give_up: Optional[Callable[[Exception], None]] = None,
):
    """
    Decorator that adds retry logic with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay between retries in seconds (default: 1.0)
        max_delay: Maximum delay between retries in seconds (default: 60.0)
        exponential_base: Base for exponential calculation (default: 2.0)
        exceptions: Tuple of exception types to catch and retry (default: Exception)
        on_retry: Optional callback called on each retry with (exception, attempt_number)
        on_give_up: Optional callback called when max retries exceeded

    Returns:
        Decorated function with retry logic

    Example:
        @retry(max_retries=3, base_delay=1.0)
        def flaky_operation():
            # Might fail transiently
            pass

        @retry(
            max_retries=5,
            exceptions=(SegmentationError, LLMProcessingError),
            on_retry=lambda e, n: logger.warning(f"Retry {n}: {e}")
        )
        def process_with_ai():
            # AI operations that might timeout
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    # Check if this is EditBananaException and retry is not allowed
                    if isinstance(e, EditBananaException) and not e.retry_allowed:
                        logger.debug(
                            f"Not retrying {func.__name__}: {e.error_code} "
                            f"(retry_allowed=False)"
                        )
                        raise

                    # Don't retry on the last attempt
                    if attempt >= max_retries:
                        logger.warning(
                            f"{func.__name__} failed after {max_retries} retries: {e}"
                        )
                        if on_give_up:
                            on_give_up(e)
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay
                    )

                    # Add jitter to prevent thundering herd
                    jitter = delay * 0.1 * (0.5 - hash(str(time.time())) % 100 / 100)
                    actual_delay = delay + jitter

                    logger.info(
                        f"{func.__name__} attempt {attempt + 1}/{max_retries + 1} "
                        f"failed: {e}. Retrying in {actual_delay:.2f}s..."
                    )

                    if on_retry:
                        try:
                            on_retry(e, attempt + 1)
                        except Exception as callback_error:
                            logger.warning(f"on_retry callback failed: {callback_error}")

                    time.sleep(actual_delay)

            # Should never reach here, but just in case
            if last_exception:
                raise last_exception

        # Attach retry configuration for introspection
        wrapper._retry_config = {
            "max_retries": max_retries,
            "base_delay": base_delay,
            "max_delay": max_delay,
            "exponential_base": exponential_base,
            "exceptions": exceptions,
        }

        return wrapper
    return decorator


def retry_with_editbanana_defaults(
    max_retries: int = 3,
    base_delay: float = 1.0,
):
    """
    Convenience decorator with EditBanana-aware defaults.

    Automatically respects the retry_allowed flag on EditBananaException.
    Retries on common recoverable errors: SegmentationError, LLMProcessingError, TimeoutError.

    Args:
        max_retries: Maximum retry attempts (default: 3)
        base_delay: Initial delay in seconds (default: 1.0)

    Example:
        @retry_with_editbanana_defaults(max_retries=3)
        def process_image():
            # Will retry on recoverable errors, not on OCR errors
            pass
    """
    from modules.exceptions import (
        SegmentationError,
        LLMProcessingError,
        TimeoutError,
    )

    return retry(
        max_retries=max_retries,
        base_delay=base_delay,
        exceptions=(SegmentationError, LLMProcessingError, TimeoutError),
    )


def is_retryable(exception: Exception) -> bool:
    """
    Check if an exception supports retry logic.

    For EditBananaException, checks the retry_allowed flag.
    For other exceptions, returns True (assume retryable).

    Args:
        exception: The exception to check

    Returns:
        True if the exception supports retry, False otherwise
    """
    if isinstance(exception, EditBananaException):
        return exception.retry_allowed
    return True  # Unknown exceptions assumed retryable


def get_retry_info(func: Callable) -> Optional[dict]:
    """
    Get retry configuration from a decorated function.

    Args:
        func: The potentially decorated function

    Returns:
        Dict with retry config or None if not decorated
    """
    return getattr(func, '_retry_config', None)
