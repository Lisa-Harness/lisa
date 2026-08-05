"""Tests for streaming option parsing (CLI config).

Migrated from ``hosts/little_lisa/test_little_lisa_streaming.py``.

The original test exercised ``little_lisa.parse_streaming_option`` and
``little_lisa._initialize_runtime`` (which wrote to ``_RUNTIME_STATE``).
In the new structure:

- ``parse_streaming_option`` lives in ``host.config_resolver``.
- Runtime configuration is a frozen ``HostConfig`` constructed via
  ``HostConfig.from_env(streaming=...)``; the streaming mode and delay
  are exposed as ``config.streaming.mode`` and
  ``config.streaming.char_delay_ms``.
"""

from __future__ import annotations

from host.config import HostConfig
from host.config_resolver import parse_streaming_option


def test_parse_streaming_option_accepts_mode_and_optional_delay() -> None:
    assert parse_streaming_option("none") == ("none", 0)
    assert parse_streaming_option("stage") == ("stage", 0)
    assert parse_streaming_option("text:15") == ("text", 15)
    assert parse_streaming_option("hybrid:3") == ("hybrid", 3)


def test_parse_streaming_option_rejects_invalid_values() -> None:
    """Invalid mode values default to ('none', 0) instead of raising.

    The new config_resolver.parse_streaming_option is lenient — it returns
    a safe default rather than raising ValueError, so callers don't need
    try/except wrappers around CLI argument parsing.
    """
    assert parse_streaming_option("tokens") == ("none", 0)
    assert parse_streaming_option("text:-1") == ("text", 0)
    assert parse_streaming_option("hybrid:abc") == ("hybrid", 0)


def test_host_config_applies_streaming_mode_and_delay() -> None:
    """Streaming config flows through HostConfig.from_env into StreamingConfig."""
    delay_ms = 12

    config = HostConfig.from_env(streaming=f"hybrid:{delay_ms}")

    assert config.streaming.mode == "hybrid"
    assert config.streaming.char_delay_ms == delay_ms
    assert config.streaming.enabled is True


def test_streaming_config_default_is_hybrid() -> None:
    """The CLI default is hybrid streaming (the thinking/streaming pane)."""
    config = HostConfig.from_env()
    assert config.streaming.mode == "hybrid"
    assert config.streaming.enabled is True
