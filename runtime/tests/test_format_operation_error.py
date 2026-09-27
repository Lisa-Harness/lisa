"""Tests for the typed ``OperationExecutionError`` hierarchy and
``format_operation_error`` type branches.

The registry raises typed subclasses so ``format_operation_error`` can
produce actionable user-facing guidance:

- ``OperationTimeoutError``    → time-budget guidance.
- ``OperationContentError``    → empty/invalid output guidance.
- ``OperationDegradedError``    → partial-artifact guidance.
- ``OperationExecutionError``   → base-class fallback (legacy message).
"""

from __future__ import annotations

from kernel.operations.registry import (
    OperationContentError,
    OperationDegradedError,
    OperationExecutionError,
    OperationTimeoutError,
    OperationValidationError,
)
from runtime.commands._routing import format_operation_error


class TestExceptionHierarchy:
    def test_timeout_error_is_execution_error(self) -> None:
        """``OperationTimeoutError`` is an ``OperationExecutionError`` subclass."""
        exc = OperationTimeoutError("Operation 'x' timed out")
        assert isinstance(exc, OperationExecutionError)
        assert isinstance(exc, RuntimeError)

    def test_content_error_is_execution_error(self) -> None:
        """``OperationContentError`` is an ``OperationExecutionError`` subclass."""
        exc = OperationContentError("empty distillation output")
        assert isinstance(exc, OperationExecutionError)

    def test_degraded_error_is_execution_error(self) -> None:
        """``OperationDegradedError`` is an ``OperationExecutionError`` subclass."""
        exc = OperationDegradedError("truncated after 3 attempts")
        assert isinstance(exc, OperationExecutionError)

    def test_validation_error_is_value_error(self) -> None:
        """``OperationValidationError`` stays a ``ValueError`` subclass."""
        exc = OperationValidationError("bad metadata")
        assert isinstance(exc, ValueError)
        assert not isinstance(exc, OperationExecutionError)


class TestFormatOperationError:
    def test_timeout_error_gets_time_budget_guidance(self) -> None:
        """A typed timeout error surfaces the time-budget guidance."""
        exc = OperationTimeoutError("Operation 'forge' timed out")
        text = format_operation_error(exc)
        assert "timed out" in text.lower()
        assert "time budget" in text.lower()

    def test_content_error_gets_empty_output_guidance(self) -> None:
        """A content error surfaces the empty/invalid-output guidance."""
        exc = OperationContentError("empty distillation output")
        text = format_operation_error(exc)
        assert "empty" in text.lower() or "invalid" in text.lower()
        assert "prompt" in text.lower() or "model" in text.lower()

    def test_degraded_error_gets_partial_artifact_guidance(self) -> None:
        """A degraded error surfaces the partial-artifact guidance."""
        exc = OperationDegradedError("truncated after 3 attempts")
        text = format_operation_error(exc)
        assert "partial" in text.lower() or "warning" in text.lower()
        assert "review" in text.lower()

    def test_base_execution_error_falls_back_to_message(self) -> None:
        """A base ``OperationExecutionError`` with no type match returns the message."""
        exc = OperationExecutionError("Operation 'x' failed: unknown reason")
        text = format_operation_error(exc)
        assert "Operation 'x' failed: unknown reason" in text

    def test_base_execution_error_with_timeout_string_still_gets_guidance(self) -> None:
        """A base-class error whose message contains 'timed out' still gets guidance."""
        exc = OperationExecutionError("Operation 'x' timed out")
        text = format_operation_error(exc)
        assert "time budget" in text.lower()
