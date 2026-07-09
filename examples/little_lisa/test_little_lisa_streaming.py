from __future__ import annotations

import pytest

from . import main as little_lisa


def test_parse_streaming_option_accepts_mode_and_optional_delay() -> None:
    assert little_lisa.parse_streaming_option("none") == ("none", 0)
    assert little_lisa.parse_streaming_option("stage") == ("stage", 0)
    assert little_lisa.parse_streaming_option("text:15") == ("text", 15)
    assert little_lisa.parse_streaming_option("hybrid:3") == ("hybrid", 3)


def test_parse_streaming_option_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="mode must be one of"):
        little_lisa.parse_streaming_option("tokens")

    with pytest.raises(ValueError, match="non-negative integer"):
        little_lisa.parse_streaming_option("text:-1")

    with pytest.raises(ValueError, match="non-negative integer"):
        little_lisa.parse_streaming_option("hybrid:abc")


def test_initialize_runtime_applies_streaming_mode_and_delay() -> None:
    delay_ms = 12

    little_lisa._initialize_runtime(
        debug=False,
        log_path="",
        user_slug="",
        state_snapshot_path="",
        streaming=f"hybrid:{delay_ms}",
    )

    assert little_lisa._RUNTIME_STATE["streaming_mode"] == "hybrid"
    assert little_lisa._RUNTIME_STATE["streaming_char_delay_ms"] == delay_ms
