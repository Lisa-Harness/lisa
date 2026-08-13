"""Tests for the ``OperationResult.status`` / ``error_detail`` contract.

The ``status`` field is the typed success/degradation contract on
``kernel.models.OperationResult``:

- ``"success"``  — clean result; the turn runner records an ``EffectReceipt``.
- ``"degraded"`` — completed with warnings (partial / truncated); no receipt.
- ``"error"``    — operation failed but chose not to raise; no receipt.

``is_success`` is the convenience property returning ``status == "success"``.
The field defaults to ``"success"`` so every existing ``OperationResult(...)``
construction (100+ call sites) preserves the legacy behavior.
"""

from __future__ import annotations

from kernel.models import OperationResult


class TestOperationResultStatus:
    def test_status_defaults_to_success(self) -> None:
        """An OperationResult with no explicit status is a clean success."""
        result = OperationResult(output_text="done")
        assert result.status == "success"
        assert result.is_success is True

    def test_error_detail_defaults_to_none(self) -> None:
        """``error_detail`` is None on a clean success result."""
        result = OperationResult(output_text="done")
        assert result.error_detail is None

    def test_degraded_result_is_not_success(self) -> None:
        """A degraded result (truncated/partial) is not a clean success."""
        result = OperationResult(
            output_text="Document forged (truncated).",
            status="degraded",
            error_detail="continuation exhausted: 3 empty passes",
        )
        assert result.status == "degraded"
        assert result.is_success is False
        assert result.error_detail == "continuation exhausted: 3 empty passes"

    def test_error_result_is_not_success(self) -> None:
        """An error-status result is not a clean success."""
        result = OperationResult(
            output_text="Operation failed.",
            status="error",
            error_detail="storage write failed: permission denied",
        )
        assert result.status == "error"
        assert result.is_success is False

    def test_status_serializes_to_json(self) -> None:
        """``status`` and ``error_detail`` serialize in model_dump()."""
        result = OperationResult(
            output_text="done",
            status="degraded",
            error_detail="truncated",
        )
        dumped = result.model_dump(mode="json")
        assert dumped["status"] == "degraded"
        assert dumped["error_detail"] == "truncated"

    def test_success_round_trips_through_serialization(self) -> None:
        """A default result round-trips through model_dump / model_validate."""
        original = OperationResult(output_text="done")
        dumped = original.model_dump(mode="json")
        restored = OperationResult.model_validate(dumped)
        assert restored.status == "success"
        assert restored.is_success is True
        assert restored.error_detail is None

    def test_extra_dict_is_independent_of_status(self) -> None:
        """``extra`` carries structured context alongside ``status``."""
        result = OperationResult(
            output_text="done",
            status="degraded",
            extra={"continuation": {"was_truncated": True, "attempts": 3}},
        )
        assert result.extra["continuation"]["attempts"] == 3
        assert result.is_success is False