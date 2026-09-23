"""
Tests for retry module.

Tests exponential backoff logic, retry_allowed flag handling, and edge cases.
"""

import time
import pytest
from unittest.mock import Mock, patch

from modules.retry import retry, retry_with_editbanana_defaults, is_retryable, get_retry_info
from modules.exceptions import (
    EditBananaException,
    SegmentationError,
    OCRParsingError,
    TimeoutError,
    ErrorSeverity,
)


class TestRetryDecorator:
    """Test the retry decorator functionality."""

    def test_successful_call_no_retry(self):
        """Successful calls should not trigger retries."""
        mock_func = Mock(return_value="success")

        @retry(max_retries=3)
        def target():
            return mock_func()

        result = target()
        assert result == "success"
        assert mock_func.call_count == 1

    def test_retry_on_failure_then_success(self):
        """Should retry on failure and return result when succeeds."""
        mock_func = Mock(side_effect=[ValueError("fail"), "success"])

        @retry(max_retries=3, base_delay=0.01)
        def target():
            return mock_func()

        result = target()
        assert result == "success"
        assert mock_func.call_count == 2

    def test_max_retries_exceeded(self):
        """Should raise exception when max retries exceeded."""
        mock_func = Mock(side_effect=ValueError("always fails"))

        @retry(max_retries=2, base_delay=0.01)
        def target():
            return mock_func()

        with pytest.raises(ValueError, match="always fails"):
            target()

        assert mock_func.call_count == 3  # Initial + 2 retries

    def test_respects_editbanana_retry_allowed(self):
        """Should not retry EditBananaException with retry_allowed=False."""
        error = OCRParsingError("OCR failed")  # retry_allowed=False
        mock_func = Mock(side_effect=error)

        @retry(max_retries=3, base_delay=0.01)
        def target():
            return mock_func()

        with pytest.raises(OCRParsingError) as exc_info:
            target()

        assert exc_info.value.retry_allowed is False
        assert mock_func.call_count == 1  # No retries

    def test_retries_editbanana_retry_allowed(self):
        """Should retry EditBananaException with retry_allowed=True."""
        error = TimeoutError("Timeout")  # retry_allowed=True
        mock_func = Mock(side_effect=[error, error, "success"])

        @retry(max_retries=3, base_delay=0.01)
        def target():
            return mock_func()

        result = target()
        assert result == "success"
        assert mock_func.call_count == 3

    def test_specific_exception_types(self):
        """Should only catch specified exception types."""
        mock_func = Mock(side_effect=ValueError("wrong type"))

        @retry(max_retries=3, exceptions=(TypeError,), base_delay=0.01)
        def target():
            return mock_func()

        with pytest.raises(ValueError):
            target()

        assert mock_func.call_count == 1

    def test_on_retry_callback(self):
        """Should call on_retry callback on each retry."""
        callback = Mock()
        mock_func = Mock(side_effect=[ValueError("fail"), ValueError("fail"), "success"])

        @retry(max_retries=3, base_delay=0.01, on_retry=callback)
        def target():
            return mock_func()

        target()
        assert callback.call_count == 2  # Called on 1st and 2nd failure

    def test_on_give_up_callback(self):
        """Should call on_give_up callback when max retries exceeded."""
        callback = Mock()
        mock_func = Mock(side_effect=ValueError("always fails"))

        @retry(max_retries=2, base_delay=0.01, on_give_up=callback)
        def target():
            return mock_func()

        with pytest.raises(ValueError):
            target()

        assert callback.call_count == 1

    def test_exponential_backoff_timing(self):
        """Should wait exponentially longer between retries."""
        mock_func = Mock(side_effect=[ValueError("fail"), ValueError("fail"), "success"])
        sleep_times = []

        with patch('time.sleep', side_effect=lambda x: sleep_times.append(x)):
            @retry(max_retries=3, base_delay=1.0, exponential_base=2.0)
            def target():
                return mock_func()

            target()

        assert len(sleep_times) == 2
        # First retry: ~1s, Second retry: ~2s (with jitter)
        assert 0.9 < sleep_times[0] < 1.2  # 1.0 + jitter
        assert 1.8 < sleep_times[1] < 2.4  # 2.0 + jitter

    def test_max_delay_cap(self):
        """Should cap delay at max_delay."""
        mock_func = Mock(side_effect=[ValueError("fail")] * 5 + ["success"])
        sleep_times = []

        with patch('time.sleep', side_effect=lambda x: sleep_times.append(x)):
            @retry(max_retries=5, base_delay=1.0, max_delay=3.0, exponential_base=2.0)
            def target():
                return mock_func()

            target()

        # With base=1, exp=2: delays would be 1, 2, 4, 8, 16
        # But max_delay=3, so all should be capped around 3
        assert all(t <= 3.5 for t in sleep_times)

    def test_function_metadata_preserved(self):
        """Should preserve function name and docstring."""
        @retry(max_retries=3)
        def example_function():
            """Example docstring."""
            return 42

        assert example_function.__name__ == "example_function"
        assert example_function.__doc__ == "Example docstring."

    def test_get_retry_info(self):
        """Should expose retry configuration."""
        @retry(max_retries=5, base_delay=2.0)
        def target():
            pass

        info = get_retry_info(target)
        assert info is not None
        assert info["max_retries"] == 5
        assert info["base_delay"] == 2.0


class TestRetryWithEditBananaDefaults:
    """Test the EditBanana-aware retry decorator."""

    def test_retries_recoverable_errors(self):
        """Should retry on SegmentationError and TimeoutError."""
        mock_func = Mock(side_effect=[SegmentationError("fail"), "success"])

        @retry_with_editbanana_defaults(max_retries=3, base_delay=0.01)
        def target():
            return mock_func()

        result = target()
        assert result == "success"
        assert mock_func.call_count == 2

    def test_does_not_retry_ocr_errors(self):
        """Should not retry OCR errors."""
        error = OCRParsingError("OCR failed")
        mock_func = Mock(side_effect=error)

        @retry_with_editbanana_defaults(max_retries=3, base_delay=0.01)
        def target():
            return mock_func()

        with pytest.raises(OCRParsingError):
            target()

        assert mock_func.call_count == 1


class TestIsRetryable:
    """Test the is_retryable function."""

    def test_editbanana_retry_allowed_true(self):
        """Should return True for retry_allowed=True."""
        error = SegmentationError("test")
        assert is_retryable(error) is True

    def test_editbanana_retry_allowed_false(self):
        """Should return False for retry_allowed=False."""
        error = OCRParsingError("test")
        assert is_retryable(error) is False

    def test_regular_exception(self):
        """Should return True for regular exceptions."""
        assert is_retryable(ValueError("test")) is True
        assert is_retryable(RuntimeError("test")) is True

    def test_critical_severity_not_retryable(self):
        """Critical severity should not be retryable."""
        error = EditBananaException(
            "test",
            severity=ErrorSeverity.CRITICAL
        )
        assert is_retryable(error) is False


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_max_retries(self):
        """Should not retry with max_retries=0."""
        mock_func = Mock(side_effect=ValueError("fail"))

        @retry(max_retries=0, base_delay=0.01)
        def target():
            return mock_func()

        with pytest.raises(ValueError):
            target()

        assert mock_func.call_count == 1

    def test_very_small_base_delay(self):
        """Should handle very small base delays."""
        mock_func = Mock(side_effect=[ValueError("fail"), "success"])

        @retry(max_retries=3, base_delay=0.001)
        def target():
            return mock_func()

        result = target()
        assert result == "success"

    def test_callback_exception_handled(self):
        """Should handle exceptions in callbacks gracefully."""
        def bad_callback(e, n):
            raise RuntimeError("callback failed")

        mock_func = Mock(side_effect=[ValueError("fail"), "success"])

        @retry(max_retries=3, base_delay=0.01, on_retry=bad_callback)
        def target():
            return mock_func()

        # Should not raise despite callback failure
        result = target()
        assert result == "success"
