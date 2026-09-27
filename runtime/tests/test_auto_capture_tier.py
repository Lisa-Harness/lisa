"""Tests for auto-capture salience → tier classification.

``classify_memory_tier`` maps the trigger salience onto a retention tier so
workflow-state captures (write-effect receipts, soft commitments) expire as
``daily``/``weekly`` instead of polluting ``long_term`` recall for a year.
"""

from __future__ import annotations

import pytest
from runtime.turn_hooks import (
    AUTO_CAPTURE_TIERS,
    DEFAULT_TIER_THRESHOLD_DAILY,
    DEFAULT_TIER_THRESHOLD_WEEKLY,
    classify_memory_tier,
    should_auto_capture,
)


class TestClassifyMemoryTier:
    """Salience band → retention tier mapping."""

    @pytest.mark.parametrize(
        ("salience", "expected"),
        [
            # Below the daily band: workflow state — daily (7-day TTL).
            (0.0, "daily"),
            (0.5, "daily"),
            (0.6, "daily"),  # write-effect tool calls land here
            (0.64, "daily"),
            # Weekly band (30-day TTL): soft commitments ("I prefer").
            (0.65, "weekly"),
            (0.7, "weekly"),  # "from now on" / "decided"
            (0.79, "weekly"),
            # Long-term band (365-day TTL): only approvals + firm commitments.
            (0.8, "long_term"),
            (0.9, "long_term"),
            (1.0, "long_term"),
        ],
    )
    def test_default_thresholds(self, salience: float, expected: str) -> None:
        assert classify_memory_tier(salience) == expected

    def test_daily_threshold_is_lower_than_weekly(self) -> None:
        assert DEFAULT_TIER_THRESHOLD_DAILY < DEFAULT_TIER_THRESHOLD_WEEKLY

    def test_returns_only_auto_capture_tiers(self) -> None:
        # permanent is reserved for explicit curation, never auto-capture.
        for salience in (0.0, 0.5, 0.7, 0.8, 1.0):
            assert classify_memory_tier(salience) in AUTO_CAPTURE_TIERS

    def test_custom_thresholds_override_defaults(self) -> None:
        # Lowering the weekly threshold promotes mid-band saliences to long_term.
        assert classify_memory_tier(0.7, weekly_threshold=0.65) == "long_term"
        # Raising the daily threshold demotes soft commitments to daily.
        assert classify_memory_tier(0.7, daily_threshold=0.75) == "daily"

    def test_negative_salience_clamps_to_daily(self) -> None:
        # A malformed negative salience must not silently promote past daily.
        assert classify_memory_tier(-1.0) == "daily"

    def test_boundary_at_daily_threshold_is_weekly(self) -> None:
        # The daily band is exclusive at the top; the threshold itself is weekly.
        assert classify_memory_tier(DEFAULT_TIER_THRESHOLD_DAILY) == "weekly"

    def test_boundary_at_weekly_threshold_is_long_term(self) -> None:
        assert classify_memory_tier(DEFAULT_TIER_THRESHOLD_WEEKLY) == "long_term"


class TestShouldAutoCaptureSalienceFeedsTier:
    """The salience ``should_auto_capture`` returns must classify sensibly."""

    def test_write_effect_receipt_is_not_long_term(self) -> None:
        # A write-effect receipt fires the write trigger at salience 0.6,
        # which must classify as daily (workflow state), not long_term.
        messages: list[dict[str, object]] = []
        receipts = [{"capability_id": "coding_edit_file", "effect": "write", "artifact_id": ""}]
        captured, salience = should_auto_capture(
            "update the file", "done", messages, effect_receipts=receipts
        )
        assert captured is True
        assert salience == 0.6
        assert classify_memory_tier(salience) == "daily"

    def test_read_only_receipt_does_not_trigger(self) -> None:
        # A read-only receipt (effect="read") must not fire the write trigger.
        messages: list[dict[str, object]] = []
        receipts = [{"capability_id": "coding_read_file", "effect": "read", "artifact_id": ""}]
        captured, salience = should_auto_capture(
            "look at the file", "done", messages, effect_receipts=receipts
        )
        assert captured is False
        assert salience == 0.0

    def test_soft_commitment_is_not_long_term(self) -> None:
        # "I prefer" fires at salience 0.6 → daily, not long_term.
        messages: list[dict[str, object]] = []
        captured, salience = should_auto_capture(
            "I prefer we use option A", "acknowledged", messages
        )
        assert captured is True
        assert salience == 0.6
        assert classify_memory_tier(salience) == "daily"

    def test_firm_commitment_is_weekly(self) -> None:
        # "decided" fires at salience 0.7 → weekly.
        messages: list[dict[str, object]] = []
        captured, salience = should_auto_capture(
            "I decided to go with option A", "acknowledged", messages
        )
        assert captured is True
        assert salience == 0.7
        assert classify_memory_tier(salience) == "weekly"

    def test_approval_granted_is_long_term(self) -> None:
        # An explicit approval fires at salience 0.8 → long_term.
        messages = [
            {"role": "tool", "content": "approved"},
        ]
        captured, salience = should_auto_capture("proceed", "done", messages)
        assert captured is True
        assert salience == 0.8
        assert classify_memory_tier(salience) == "long_term"
