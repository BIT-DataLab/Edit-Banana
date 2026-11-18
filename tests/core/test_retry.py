"""
Tests for retry decorator.

Covers:
- Retry with exponential backoff
- Fixed and linear backoff strategies
- Exception filtering
- Callback invocation
- Retry statistics
- Async function support
"""

import pytest
import time
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.core.retry import retry, retry_with_defaults, RetryContext, get_retry_stats, reset_retry_stats
from modules.exceptions import EditBananaException, LLMProcessingError, FileValidationError


class TestRetryDecorator:
    """Test retry decorator functionality."""

    def test_success_no_retry(self):
        """Test successful function is not retried."""
        call_count = 0

        @retry(max_retries=3)
        def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = success_func()
        assert result == "success"
        assert call_count == 1  # No retries on success

    def test_retry_on_exception(self):
        """Test function is retried on exception."""
        call_count = 0

        @retry(max_retries=3, base_delay=0.01)
        def fail_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise EditBananaException("Temporary error")
            return "success"

        result = fail_func()
        assert result == "success"
        assert call_count == 3  # 2 failures + 1 success

    def test_max_retries_exceeded(self):
        """Test exception raised when max retries exceeded."""
        call_count = 0

        @retry(max_retries=2, base_delay=0.01)
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise EditBananaException("Persistent error")

        with pytest.raises(EditBananaException) as exc_info:
            always_fail()

        assert call_count == 3  # Initial + 2 retries
        assert "Persistent error" in str(exc_info.value)

    def test_no_retry_for_non_matching_exception(self):
        """Test no retry for exceptions not in exceptions_to_retry."""
        call_count = 0

        @retry(max_retries=3, base_delay=0.01, exceptions_to_retry=(ValueError,))
        def raise_type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("Wrong type")

        with pytest.raises(TypeError):
            raise_type_error()

        assert call_count == 1  # No retry for non-matching exception

    def test_retry_callback_invoked(self):
        """Test retry callback is called on each retry."""
        callback_calls = []

        def on_retry(exc, attempt, delay):
            callback_calls.append((type(exc).__name__, attempt, delay))

        call_count = 0

        @retry(max_retries=2, base_delay=0.01, on_retry=on_retry)
        def fail_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise EditBananaException("Error")
            return "success"

        fail_twice()

        assert len(callback_calls) == 2
        assert callback_calls[0][0] == "EditBananaException"
        assert callback_calls[0][1] == 1  # First retry attempt
        assert callback_calls[1][1] == 2  # Second retry attempt

    def test_should_retry_predicate(self):
        """Test custom should_retry predicate."""
        call_count = 0

        def should_retry(exc):
            # Only retry if error message contains "please retry"
            return "please retry" in str(exc).lower()

        @retry(max_retries=3, base_delay=0.01, should_retry=should_retry)
        def conditional_fail():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise EditBananaException("Please retry this")
            raise EditBananaException("Do not retry this")

        with pytest.raises(EditBananaException) as exc_info:
            conditional_fail()

        assert call_count == 2  # Initial + 1 retry
        assert "Do not retry this" in str(exc_info.value)

    def test_retry_with_editbanana_exception_retry_flag(self):
        """Test retry respects EditBananaException.retry_allowed flag."""
        call_count = 0

        @retry(max_retries=3, base_delay=0.01)
        def fail_with_no_retry():
            nonlocal call_count
            call_count += 1
            # FileValidationError has retry_allowed=False
            raise FileValidationError("Invalid file")

        with pytest.raises(FileValidationError):
            fail_with_no_retry()

        assert call_count == 1  # No retry because retry_allowed=False


class TestBackoffStrategies:
    """Test different backoff strategies."""

    def test_fixed_backoff(self):
        """Test fixed delay backoff."""
        delays = []

        def on_retry(exc, attempt, delay):
            delays.append(delay)

        call_count = 0

        @retry(max_retries=3, base_delay=0.05, backoff_strategy="fixed", on_retry=on_retry)
        def fail_three_times():
            nonlocal call_count
            call_count += 1
            if call_count < 4:
                raise EditBananaException("Error")
            return "success"

        fail_three_times()

        # Fixed backoff: all delays should be the same
        assert all(d == pytest.approx(0.05, abs=0.01) for d in delays)

    def test_linear_backoff(self):
        """Test linear delay backoff."""
        delays = []

        def on_retry(exc, attempt, delay):
            delays.append(delay)

        call_count = 0

        @retry(max_retries=3, base_delay=0.05, backoff_strategy="linear", on_retry=on_retry)
        def fail_three_times():
            nonlocal call_count
            call_count += 1
            if call_count < 4:
                raise EditBananaException("Error")
            return "success"

        fail_three_times()

        # Linear backoff: delays should be base_delay * attempt
        assert delays[0] == pytest.approx(0.05, abs=0.01)   # 0.05 * 1
        assert delays[1] == pytest.approx(0.10, abs=0.01)   # 0.05 * 2
        assert delays[2] == pytest.approx(0.15, abs=0.01)   # 0.05 * 3

    def test_exponential_backoff(self):
        """Test exponential delay backoff."""
        delays = []

        def on_retry(exc, attempt, delay):
            delays.append(delay)

        call_count = 0

        @retry(max_retries=3, base_delay=0.05, backoff_strategy="exponential", on_retry=on_retry)
        def fail_three_times():
            nonlocal call_count
            call_count += 1
            if call_count < 4:
                raise EditBananaException("Error")
            return "success"

        fail_three_times()

        # Exponential backoff: delays should be base_delay * 2^(attempt-1)
        assert delays[0] == pytest.approx(0.05, abs=0.01)   # 0.05 * 2^0
        assert delays[1] == pytest.approx(0.10, abs=0.01)   # 0.05 * 2^1
        assert delays[2] == pytest.approx(0.20, abs=0.01)   # 0.05 * 2^2

    def test_max_delay_cap(self):
        """Test delay is capped at max_delay."""
        delays = []

        def on_retry(exc, attempt, delay):
            delays.append(delay)

        call_count = 0

        @retry(max_retries=5, base_delay=1.0, max_delay=2.0, backoff_strategy="exponential", on_retry=on_retry)
        def fail_five_times():
            nonlocal call_count
            call_count += 1
            if call_count < 6:
                raise EditBananaException("Error")
            return "success"

        fail_five_times()

        # All delays should be capped at 2.0
        assert all(d <= 2.0 for d in delays)


class TestRetryWithDefaults:
    """Test retry_with_defaults convenience decorator."""

    def test_without_parentheses(self):
        """Test usage without parentheses."""
        call_count = 0

        @retry_with_defaults
        def fail_once():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise EditBananaException("Error")
            return "success"

        result = fail_once()
        assert result == "success"

    def test_with_parentheses(self):
        """Test usage with parentheses."""
        call_count = 0

        @retry_with_defaults(max_retries=2)
        def fail_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise EditBananaException("Error")
            return "success"

        result = fail_twice()
        assert result == "success"
        assert call_count == 3


class TestRetryContext:
    """Test RetryContext for manual retry control."""

    def test_successful_attempt(self):
        """Test successful attempt through context."""
        with RetryContext(max_retries=3) as ctx:
            for attempt in ctx.attempts():
                ctx.success()

        assert ctx._succeeded is True
        assert len(ctx._failures) == 0

    def test_failed_attempts(self):
        """Test failed attempts through context."""
        with pytest.raises(ValueError, match="Error"):
            with RetryContext(max_retries=2) as ctx:
                for attempt in ctx.attempts():
                    try:
                        raise ValueError("Error")
                    except ValueError as e:
                        ctx.failure(e)
                        if not ctx.should_retry():
                            raise

        assert ctx._succeeded is False
        assert len(ctx._failures) == 3  # Initial + 2 retries

    def test_should_retry_logic(self):
        """Test should_retry returns correct value."""
        ctx = RetryContext(max_retries=2)

        # First attempt
        next(ctx.attempts())
        assert ctx.should_retry() is True  # Not succeeded, attempt 0 < max_retries

        ctx.success()
        assert ctx.should_retry() is False  # Succeeded

    def test_get_delay_exponential(self):
        """Test delay calculation in context."""
        ctx = RetryContext(max_retries=3, base_delay=1.0, backoff_strategy="exponential")

        delays = []
        for _ in ctx.attempts():
            delays.append(ctx.get_delay())

        # Delays: 1.0, 2.0, 4.0 (base_delay * 2^attempt)
        assert delays[0] == 1.0  # 1.0 * 2^0
        assert delays[1] == 2.0  # 1.0 * 2^1
        assert delays[2] == 4.0  # 1.0 * 2^2


class TestAsyncRetry:
    """Test retry with async functions."""

    @pytest.mark.asyncio
    async def test_async_success_no_retry(self):
        """Test async successful function is not retried."""
        call_count = 0

        @retry(max_retries=3)
        async def async_success():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await async_success()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_async_retry_on_exception(self):
        """Test async function is retried on exception."""
        call_count = 0

        @retry(max_retries=3, base_delay=0.01)
        async def async_fail():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise EditBananaException("Error")
            return "success"

        result = await async_fail()
        assert result == "success"
        assert call_count == 3


class TestRetryStats:
    """Test global retry statistics."""

    def setup_method(self):
        """Reset stats before each test."""
        reset_retry_stats()

    def test_initial_stats(self):
        """Test initial stats are zero."""
        stats = get_retry_stats()
        assert stats["total_attempts"] == 0
        assert stats["successful_retries"] == 0
        assert stats["failed_retries"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
