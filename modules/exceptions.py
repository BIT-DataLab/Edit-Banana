"""
EditBanana exception hierarchy.

Provides structured error types with severity levels for proper error handling,
retry logic, and user-friendly error messages.
"""

from enum import Enum
from typing import Optional, Dict, Any


class ErrorSeverity(Enum):
    """Error severity levels for determining recovery strategies."""
    CRITICAL = "critical"      # Unrecoverable, requires manual intervention
    RECOVERABLE = "recoverable"  # Can retry or degrade gracefully
    WARNING = "warning"        # Non-blocking, logged but operation continues


class EditBananaException(Exception):
    """Base exception for all EditBanana errors.

    Attributes:
        message: Human-readable error description
        severity: ErrorSeverity level
        error_code: Unique error identifier for client handling
        context: Additional context data for debugging
        retry_allowed: Whether this error supports retry logic
    """

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.RECOVERABLE,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        retry_allowed: bool = True
    ):
        self.message = message
        self.severity = severity
        self.error_code = error_code or self._generate_error_code()
        self.context = context or {}
        self.retry_allowed = retry_allowed and severity != ErrorSeverity.CRITICAL
        super().__init__(self.message)

    def _generate_error_code(self) -> str:
        """Generate default error code from class name."""
        import re
        # Convert CamelCase to SCREAMING_SNAKE_CASE with _ERROR suffix
        name = self.__class__.__name__
        # Insert underscore before uppercase letters (except first)
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        result = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).upper()
        # Ensure _ERROR suffix
        if not result.endswith('_ERROR'):
            result += '_ERROR'
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "severity": self.severity.value,
            "retry_allowed": self.retry_allowed,
            "context": self.context,
        }

    def __str__(self) -> str:
        if self.context:
            return f"[{self.error_code}] {self.message} (context: {self.context})"
        return f"[{self.error_code}] {self.message}"


class SegmentationError(EditBananaException):
    """Raised when SAM3 segmentation fails."""

    def __init__(
        self,
        message: str = "Image segmentation failed",
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            severity=ErrorSeverity.RECOVERABLE,
            error_code="SEGMENTATION_ERROR",
            context=context,
            **kwargs
        )


class OCRParsingError(EditBananaException):
    """Raised when text extraction/OCR fails."""

    def __init__(
        self,
        message: str = "Text extraction failed",
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            severity=ErrorSeverity.WARNING,
            error_code="OCR_ERROR",
            context=context,
            retry_allowed=False,  # OCR failures rarely benefit from retry
            **kwargs
        )


class LLMProcessingError(EditBananaException):
    """Raised when VLM/LLM API calls fail."""

    def __init__(
        self,
        message: str = "AI processing failed",
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            severity=ErrorSeverity.RECOVERABLE,
            error_code="LLM_ERROR",
            context=context,
            **kwargs
        )


class FileValidationError(EditBananaException):
    """Raised when input file validation fails."""

    def __init__(
        self,
        message: str = "Invalid file",
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            severity=ErrorSeverity.CRITICAL,
            error_code="FILE_VALIDATION_ERROR",
            context=context,
            retry_allowed=False,  # Invalid file won't become valid on retry
            **kwargs
        )


class TimeoutError(EditBananaException):
    """Raised when processing exceeds time limit."""

    def __init__(
        self,
        message: str = "Processing timeout",
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            severity=ErrorSeverity.RECOVERABLE,
            error_code="TIMEOUT_ERROR",
            context=context,
            **kwargs
        )


class XMLGenerationError(EditBananaException):
    """Raised when XML fragment generation fails."""

    def __init__(
        self,
        message: str = "XML generation failed",
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            severity=ErrorSeverity.WARNING,
            error_code="XML_ERROR",
            context=context,
            retry_allowed=False,
            **kwargs
        )


class ArrowProcessingError(EditBananaException):
    """Raised when arrow detection or connection fails."""

    def __init__(
        self,
        message: str = "Arrow processing failed",
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            severity=ErrorSeverity.WARNING,
            error_code="ARROW_ERROR",
            context=context,
            retry_allowed=False,
            **kwargs
        )


class ProcessingPartialResultError(EditBananaException):
    """Raised when processing partially succeeds with some elements failed.

    This exception carries partial results so they can be preserved.
    """

    def __init__(
        self,
        message: str = "Processing completed with partial results",
        partial_result: Optional[Any] = None,
        failed_elements: Optional[list] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        ctx = context or {}
        ctx["failed_elements"] = failed_elements or []
        ctx["partial_element_count"] = len(partial_result) if partial_result else 0

        super().__init__(
            message=message,
            severity=ErrorSeverity.WARNING,
            error_code="PARTIAL_RESULT_ERROR",
            context=ctx,
            retry_allowed=False,
            **kwargs
        )
        self.partial_result = partial_result
