"""Tests for the provider-owned truncated signal on StubCompletion."""

from __future__ import annotations

from host.providers import (
    derive_anthropic_truncated,
    derive_openai_compatible_truncated,
    derive_responses_api_truncated,
)


class TestOpenAICompatible:
    def test_length_is_truncated(self) -> None:
        assert (
            derive_openai_compatible_truncated(
                finish_reason="length", completion_tokens=100, requested_tokens=100
            )
            is True
        )

    def test_stop_is_not_truncated(self) -> None:
        assert (
            derive_openai_compatible_truncated(
                finish_reason="stop", completion_tokens=50, requested_tokens=100
            )
            is False
        )

    def test_tool_calls_is_not_truncated(self) -> None:
        assert (
            derive_openai_compatible_truncated(
                finish_reason="tool_calls", completion_tokens=50, requested_tokens=100
            )
            is False
        )

    def test_none_with_near_budget_is_truncated(self) -> None:
        assert (
            derive_openai_compatible_truncated(
                finish_reason=None, completion_tokens=95, requested_tokens=100
            )
            is True
        )

    def test_none_with_low_budget_is_none(self) -> None:
        assert (
            derive_openai_compatible_truncated(
                finish_reason=None, completion_tokens=10, requested_tokens=100
            )
            is None
        )

    def test_empty_string_treated_as_none(self) -> None:
        assert (
            derive_openai_compatible_truncated(
                finish_reason="", completion_tokens=10, requested_tokens=100
            )
            is None
        )

    def test_unknown_value_is_none(self) -> None:
        assert (
            derive_openai_compatible_truncated(
                finish_reason="content_filter", completion_tokens=50, requested_tokens=100
            )
            is None
        )


class TestResponsesAPI:
    def test_incomplete_is_truncated(self) -> None:
        assert derive_responses_api_truncated(status="incomplete") is True

    def test_completed_is_not_truncated(self) -> None:
        assert derive_responses_api_truncated(status="completed") is False

    def test_failed_is_not_truncated(self) -> None:
        assert derive_responses_api_truncated(status="failed") is False

    def test_none_is_none(self) -> None:
        assert derive_responses_api_truncated(status=None) is None


class TestAnthropic:
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

    def test_none_is_none(self) -> None:
        assert derive_anthropic_truncated(stop_reason=None) is None
