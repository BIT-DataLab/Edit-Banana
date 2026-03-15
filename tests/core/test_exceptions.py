"""
Tests for exception hierarchy.

Covers:
- Exception creation and attributes
- Severity levels and retry_allowed logic
- to_dict serialization
- Partial result error with data preservation
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.exceptions import (
    ErrorSeverity,
    EditBananaException,
    SegmentationError,
    OCRParsingError,
    LLMProcessingError,
    FileValidationError,
    TimeoutError,
    XMLGenerationError,
    ArrowProcessingError,
    ProcessingPartialResultError,
)


class TestErrorSeverity:
    """Test ErrorSeverity enum."""

    def test_severity_values(self):
        """Test severity enum has expected values."""
        assert ErrorSeverity.CRITICAL.value == "critical"
        assert ErrorSeverity.RECOVERABLE.value == "recoverable"
        assert ErrorSeverity.WARNING.value == "warning"


class TestEditBananaException:
    """Test base exception class."""

    def test_basic_creation(self):
        """Test basic exception creation."""
        exc = EditBananaException("Something went wrong")
        assert exc.message == "Something went wrong"
        assert exc.severity == ErrorSeverity.RECOVERABLE
        assert exc.retry_allowed is True
        assert exc.error_code is not None

    def test_custom_severity(self):
        """Test exception with custom severity."""
        exc = EditBananaException(
            "Critical error",
            severity=ErrorSeverity.CRITICAL
        )
        assert exc.severity == ErrorSeverity.CRITICAL
        # Critical errors should not allow retry
        assert exc.retry_allowed is False

    def test_custom_error_code(self):
        """Test custom error code."""
        exc = EditBananaException(
            "Error",
            error_code="CUSTOM_ERROR"
        )
        assert exc.error_code == "CUSTOM_ERROR"

    def test_auto_generated_error_code(self):
        """Test auto-generated error code from class name."""
        exc = EditBananaException("Error")
        assert "ERROR" in exc.error_code

    def test_context_storage(self):
        """Test context data storage."""
        context = {"file": "test.png", "size": 1024}
        exc = EditBananaException("Error", context=context)
        assert exc.context == context
        assert exc.context["file"] == "test.png"

    def test_retry_allowed_override(self):
        """Test retry_allowed can be overridden."""
        # Even RECOVERABLE can have retry disallowed
        exc = EditBananaException(
            "Error",
            severity=ErrorSeverity.RECOVERABLE,
            retry_allowed=False
        )
        assert exc.retry_allowed is False

    def test_to_dict_serialization(self):
        """Test to_dict method."""
        exc = EditBananaException(
            "Test error",
            severity=ErrorSeverity.WARNING,
            error_code="TEST_ERROR",
            context={"key": "value"},
            retry_allowed=False
        )
        data = exc.to_dict()
        assert data["message"] == "Test error"
        assert data["severity"] == "warning"
        assert data["error_code"] == "TEST_ERROR"
        assert data["retry_allowed"] is False
        assert data["context"] == {"key": "value"}

    def test_str_representation_with_context(self):
        """Test string representation with context."""
        exc = EditBananaException(
            "Error occurred",
            context={"file": "test.png"}
        )
        str_repr = str(exc)
        assert "Error occurred" in str_repr
        assert "test.png" in str_repr

    def test_str_representation_without_context(self):
        """Test string representation without context."""
        exc = EditBananaException("Error occurred")
        str_repr = str(exc)
        assert str_repr == "[EDIT_BANANA_EXCEPTION_ERROR] Error occurred"


class TestSpecificExceptions:
    """Test specific exception types."""

    def test_segmentation_error(self):
        """Test SegmentationError defaults."""
        exc = SegmentationError("Segmentation failed")
        assert exc.severity == ErrorSeverity.RECOVERABLE
        assert exc.error_code == "SEGMENTATION_ERROR"
        assert exc.retry_allowed is True

    def test_ocr_parsing_error_no_retry(self):
        """Test OCRParsingError does not allow retry."""
        exc = OCRParsingError("OCR failed")
        assert exc.severity == ErrorSeverity.WARNING
        assert exc.error_code == "OCR_ERROR"
        assert exc.retry_allowed is False  # OCR rarely benefits from retry

    def test_llm_processing_error(self):
        """Test LLMProcessingError."""
        exc = LLMProcessingError("API timeout")
        assert exc.severity == ErrorSeverity.RECOVERABLE
        assert exc.error_code == "LLM_ERROR"
        assert exc.retry_allowed is True

    def test_file_validation_error_critical(self):
        """Test FileValidationError is critical and no retry."""
        exc = FileValidationError("Invalid file format")
        assert exc.severity == ErrorSeverity.CRITICAL
        assert exc.error_code == "FILE_VALIDATION_ERROR"
        assert exc.retry_allowed is False

    def test_timeout_error(self):
        """Test TimeoutError."""
        exc = TimeoutError("Processing timeout")
        assert exc.severity == ErrorSeverity.RECOVERABLE
        assert exc.error_code == "TIMEOUT_ERROR"
        assert exc.retry_allowed is True

    def test_xml_generation_error(self):
        """Test XMLGenerationError."""
        exc = XMLGenerationError("XML parse error")
        assert exc.severity == ErrorSeverity.WARNING
        assert exc.error_code == "XML_ERROR"
        assert exc.retry_allowed is False

    def test_arrow_processing_error(self):
        """Test ArrowProcessingError."""
        exc = ArrowProcessingError("Arrow detection failed")
        assert exc.severity == ErrorSeverity.WARNING
        assert exc.error_code == "ARROW_ERROR"
        assert exc.retry_allowed is False


class TestProcessingPartialResultError:
    """Test ProcessingPartialResultError."""

    def test_partial_result_storage(self):
        """Test partial result is stored."""
        partial = [{"id": 1}, {"id": 2}]
        exc = ProcessingPartialResultError(
            "Partial success",
            partial_result=partial,
            failed_elements=["element_3"]
        )
        assert exc.partial_result == partial
        assert len(exc.partial_result) == 2

    def test_partial_result_in_context(self):
        """Test failed elements stored in context."""
        exc = ProcessingPartialResultError(
            "Partial success",
            partial_result=[{"id": 1}],
            failed_elements=["element_2", "element_3"]
        )
        assert exc.context["failed_elements"] == ["element_2", "element_3"]
        assert exc.context["partial_element_count"] == 1

    def test_partial_result_no_retry(self):
        """Test partial result error does not allow retry."""
        exc = ProcessingPartialResultError("Partial success")
        assert exc.retry_allowed is False
        assert exc.severity == ErrorSeverity.WARNING

    def test_partial_result_defaults(self):
        """Test partial result with defaults."""
        exc = ProcessingPartialResultError("Partial success")
        assert exc.partial_result is None
        assert exc.context["failed_elements"] == []
        assert exc.context["partial_element_count"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
