"""Tests for the provider-owned ``truncated`` signal derivation helpers.

These tests cover the three pure helper functions that derive the
authoritative ``truncated: bool | None`` field on ``StubCompletion`` from
each provider's native termination signal. The helpers live in
``hosts.core.providers`` and are unit-testable without mocking the full
async ``complete_live_*`` chain.

Contract:
- ``True``  = model stopped before completing (hit max_tokens / incomplete).
- ``False`` = clean stop.
- ``None``  = adapter genuinely cannot tell.
"""

from __future__ import annotations

from hosts.core.providers import (
    derive_anthropic_truncated,
    derive_openai_compatible_truncated,
    derive_responses_api_truncated,
)

# ── OpenAI-compatible (incl. Ollama) ─────────────────────────────────


class TestOpenAICompatibleTruncated:
    def test_length_is_truncated(self) -> None:
        assert derive_openai_compatible_truncated(
            finish_reason="length", completion_tokens=500, requested_tokens=500
        ) is True

    def test_stop_is_not_truncated(self) -> None:
        assert derive_openai_compatible_truncated(
            finish_reason="stop", completion_tokens=100, requested_tokens=500
        ) is False

    def test_tool_calls_is_not_truncated(self) -> None:
        assert derive_openai_compatible_truncated(
            finish_reason="tool_calls", completion_tokens=100, requested_tokens=500
        ) is False

    def test_none_with_near_budget_is_truncated(self) -> None:
        # Ollama-defensive: missing signal + ≥90% of budget → truncation.
        assert derive_openai_compatible_truncated(
            finish_reason=None, completion_tokens=460, requested_tokens=500
        ) is True

    def test_none_with_low_usage_is_unknown(self) -> None:
        # Missing signal but tokens well under budget → cannot tell.
        assert derive_openai_compatible_truncated(
            finish_reason=None, completion_tokens=100, requested_tokens=500
        ) is None

    def test_none_with_zero_requested_is_unknown(self) -> None:
        assert derive_openai_compatible_truncated(
            finish_reason=None, completion_tokens=100, requested_tokens=0
        ) is None

    def test_unknown_string_is_unknown(self) -> None:
        assert derive_openai_compatible_truncated(
            finish_reason="content_filter", completion_tokens=100, requested_tokens=500
        ) is None

    def test_exactly_90_percent_threshold_is_truncated(self) -> None:
        # Boundary: completion_tokens == 90% of requested → truncated.
        assert derive_openai_compatible_truncated(
            finish_reason=None, completion_tokens=450, requested_tokens=500
        ) is True

    def test_below_90_percent_threshold_is_unknown(self) -> None:
        assert derive_openai_compatible_truncated(
            finish_reason=None, completion_tokens=449, requested_tokens=500
        ) is None


# ── Responses API (Azure Codex) ───────────────────────────────────────


class TestResponsesApiTruncated:
    def test_incomplete_is_truncated(self) -> None:
        assert derive_responses_api_truncated(status="incomplete") is True

    def test_completed_is_not_truncated(self) -> None:
        assert derive_responses_api_truncated(status="completed") is False

    def test_failed_is_unknown(self) -> None:
        assert derive_responses_api_truncated(status="failed") is None

    def test_none_is_unknown(self) -> None:
        assert derive_responses_api_truncated(status=None) is None


# ── Anthropic ─────────────────────────────────────────────────────────


class TestAnthropicTruncated:
    def test_max_tokens_is_truncated(self) -> None:
        assert derive_anthropic_truncated(stop_reason="max_tokens") is True

    def test_max_sequence_is_truncated(self) -> None:
        assert derive_anthropic_truncated(stop_reason="max_sequence") is True

    def test_end_turn_is_not_truncated(self) -> None:
        assert derive_anthropic_truncated(stop_reason="end_turn") is False

    def test_stop_sequence_is_not_truncated(self) -> None:
        assert derive_anthropic_truncated(stop_reason="stop_sequence") is False

    def test_tool_use_is_not_truncated(self) -> None:
        assert derive_anthropic_truncated(stop_reason="tool_use") is False

    def test_none_is_unknown(self) -> None:
        assert derive_anthropic_truncated(stop_reason=None) is None

    def test_unknown_string_is_unknown(self) -> None:
        assert derive_anthropic_truncated(stop_reason="model_error") is None

    def test_case_insensitive(self) -> None:
        # Anthropic stop_reason is normalized to lowercase before calling.
        assert derive_anthropic_truncated(stop_reason="Max_Tokens") is True
        assert derive_anthropic_truncated(stop_reason="END_TURN") is False