"""Tests for hosts.core — the shared transport-neutral host runtime."""

from __future__ import annotations

from pathlib import Path

import pytest

from hosts.core import (
    HostCommandResult,
    HostConfig,
    HostRuntime,
    NullTransport,
    handle_message,
)
from hosts.core.config_resolver import parse_streaming_option, resolve_timezone
from hosts.core.runtime import TokenCounters, TurnContext


# ── HostConfig ──────────────────────────────────────────────────────


class TestHostConfig:
    def test_from_env_defaults(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        config = HostConfig.from_env()
        assert config.debug is False
        assert config.effective_log_level == "info"
        assert config.timezone in {"UTC", "America/New_York"} or "/" in config.timezone
        assert config.streaming.mode == "none"
        assert config.slice_config.explicit_budget_ms == 2000

    def test_from_env_debug_overrides_log_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        config = HostConfig.from_env(debug=True)
        assert config.effective_log_level == "debug"

    def test_from_env_data_dir_resolved(self, tmp_path: Path) -> None:
        config = HostConfig.from_env(data_dir=str(tmp_path / "custom_artifacts"))
        assert config.data_dir == (tmp_path / "custom_artifacts").resolve()
        assert config.data_dir.is_absolute()

    def test_resolved_paths(self, tmp_path: Path) -> None:
        config = HostConfig.from_env(data_dir=str(tmp_path))
        assert config.resolved_memory_db_path == tmp_path / "memory.json"
        assert config.resolved_thread_dir == tmp_path / "threads"
        assert config.resolved_host_log_path == tmp_path / "host.log"

    def test_streaming_config(self) -> None:
        config = HostConfig.from_env(streaming="hybrid:50")
        assert config.streaming.mode == "hybrid"
        assert config.streaming.char_delay_ms == 50
        assert config.streaming.enabled is True

    def test_streaming_none(self) -> None:
        config = HostConfig.from_env(streaming="none")
        assert config.streaming.enabled is False

    def test_slice_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LISA_SLICE_PRE_TURN", "true")
        monkeypatch.setenv("LISA_SLICE_BUDGET_MS", "5000")
        config = HostConfig.from_env()
        assert config.slice_config.pre_turn_enabled is True
        assert config.slice_config.explicit_budget_ms == 5000

    def test_env_timezone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LISA_TIMEZONE env var should resolve correctly."""
        monkeypatch.setenv("LISA_TIMEZONE", "Europe/London")
        config = HostConfig.from_env()
        assert config.timezone == "Europe/London"
        assert "env:" in config.timezone_source

    def test_frozen(self, tmp_path: Path) -> None:
        config = HostConfig.from_env(data_dir=str(tmp_path))
        with pytest.raises(AttributeError):
            config.debug = True  # type: ignore[misc]


# ── Config resolver ─────────────────────────────────────────────────


class TestConfigResolver:
    def test_resolve_timezone_utc(self) -> None:
        tz, source = resolve_timezone("UTC")
        assert tz == "UTC"
        assert source == "cli"

    def test_resolve_timezone_iana(self) -> None:
        tz, _ = resolve_timezone("America/New_York")
        assert tz == "America/New_York"

    def test_resolve_timezone_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="not resolvable"):
            resolve_timezone("NotATimezone")  # no slash, not UTC, not an alias

    def test_parse_streaming_none(self) -> None:
        mode, delay = parse_streaming_option("none")
        assert mode == "none"
        assert delay == 0

    def test_parse_streaming_hybrid_with_delay(self) -> None:
        mode, delay = parse_streaming_option("hybrid:50")
        assert mode == "hybrid"
        assert delay == 50

    def test_parse_streaming_invalid(self) -> None:
        mode, delay = parse_streaming_option("invalid")
        assert mode == "none"
        assert delay == 0

    def test_parse_streaming_empty(self) -> None:
        mode, _ = parse_streaming_option("")
        assert mode == "none"


# ── HostRuntime ─────────────────────────────────────────────────────


class TestHostRuntime:
    def test_construction(self, tmp_path: Path) -> None:
        config = HostConfig.from_env(data_dir=str(tmp_path))
        runtime = HostRuntime(config)
        assert runtime.config is config
        assert runtime.mcp_started is False
        assert runtime.mcp_tool_pool is None
        assert runtime.tokens.session_turn_count == 0

    def test_path_accessors(self, tmp_path: Path) -> None:
        config = HostConfig.from_env(data_dir=str(tmp_path), user_slug="test-user")
        runtime = HostRuntime(config)
        assert runtime.data_dir == tmp_path
        assert runtime.memory_db_path == tmp_path / "memory.json"
        assert runtime.thread_dir == tmp_path / "threads"
        assert runtime.thread_path == tmp_path / "threads" / "test-user.jsonl"
        assert runtime.token_usage_path == tmp_path / "token-usage.jsonl"

    def test_thread_path_none_when_no_user(self, tmp_path: Path) -> None:
        config = HostConfig.from_env(data_dir=str(tmp_path))
        runtime = HostRuntime(config)
        assert runtime.thread_path is None

    def test_thread_path_none_when_no_thread(self, tmp_path: Path) -> None:
        config = HostConfig.from_env(data_dir=str(tmp_path), user_slug="test", no_thread=True)
        runtime = HostRuntime(config)
        assert runtime.thread_path is None

    @pytest.mark.asyncio
    async def test_start_aclose_no_warm_configs(self, tmp_path: Path) -> None:
        config = HostConfig.from_env(data_dir=str(tmp_path))
        runtime = HostRuntime(config)
        await runtime.start()
        assert runtime.mcp_started is True
        assert runtime.mcp_tool_pool is None  # no warm configs
        await runtime.aclose()
        assert runtime.mcp_started is False

    @pytest.mark.asyncio
    async def test_start_idempotent(self, tmp_path: Path) -> None:
        config = HostConfig.from_env(data_dir=str(tmp_path))
        runtime = HostRuntime(config)
        await runtime.start()
        await runtime.start()  # should not raise
        assert runtime.mcp_started is True
        await runtime.aclose()

    @pytest.mark.asyncio
    async def test_aclose_without_start(self, tmp_path: Path) -> None:
        config = HostConfig.from_env(data_dir=str(tmp_path))
        runtime = HostRuntime(config)
        await runtime.aclose()  # should not raise
        assert runtime.mcp_started is False

    def test_set_mcp_configs(self, tmp_path: Path) -> None:
        config = HostConfig.from_env(data_dir=str(tmp_path))
        runtime = HostRuntime(config)
        runtime.set_mcp_configs(warm=["warm1"], cold=["cold1"])
        assert runtime.mcp_warm_configs == ["warm1"]
        assert runtime.mcp_cold_configs == ["cold1"]

    def test_provider_client_setters(self, tmp_path: Path) -> None:
        config = HostConfig.from_env(data_dir=str(tmp_path))
        runtime = HostRuntime(config)
        runtime.azure_client = "fake_azure"
        assert runtime.azure_client == "fake_azure"
        runtime.openai_client = "fake_openai"
        assert runtime.openai_client == "fake_openai"

    def test_request_restart(self, tmp_path: Path) -> None:
        config = HostConfig.from_env(data_dir=str(tmp_path))
        runtime = HostRuntime(config)
        runtime.request_restart("pack enable: lisa-pack-test")
        assert runtime.restart_required is True
        assert "pack enable" in runtime.restart_reason

    def test_state_snapshot(self, tmp_path: Path) -> None:
        config = HostConfig.from_env(data_dir=str(tmp_path))
        runtime = HostRuntime(config)
        assert runtime.state_snapshot is None
        runtime.state_snapshot = {"key": "value"}
        assert runtime.state_snapshot == {"key": "value"}


# ── TokenCounters ───────────────────────────────────────────────────


class TestTokenCounters:
    def test_reset_turn(self) -> None:
        counters = TokenCounters()
        counters.turn_prompt_tokens = 100
        counters.turn_completion_tokens = 50
        counters.reset_turn()
        assert counters.turn_prompt_tokens == 0
        assert counters.turn_completion_tokens == 0
        # Session counters should NOT be reset.
        counters.session_prompt_tokens = 500
        counters.reset_turn()
        assert counters.session_prompt_tokens == 500


# ── TurnContext ─────────────────────────────────────────────────────


class TestTurnContext:
    def test_defaults(self) -> None:
        ctx = TurnContext()
        assert ctx.mode_id == "default"
        assert ctx.raw_message == ""
        assert ctx.resolved_model == {}

    def test_set_fields(self) -> None:
        ctx = TurnContext(
            raw_message="/search test",
            mode_id="search",
            query="test",
        )
        assert ctx.raw_message == "/search test"
        assert ctx.mode_id == "search"


# ── NullTransport ───────────────────────────────────────────────────


class TestNullTransport:
    @pytest.mark.asyncio
    async def test_all_methods_are_noops(self) -> None:
        transport = NullTransport()
        await transport.on_stage("test")
        await transport.on_text_chunk("chunk")
        await transport.on_response("response")
        await transport.on_debug_context({"key": "value"})
        await transport.on_error("error")


# ── handle_message ──────────────────────────────────────────────────


class TestHandleMessage:
    @pytest.mark.asyncio
    async def test_help_command(self, tmp_path: Path) -> None:
        config = HostConfig.from_env(data_dir=str(tmp_path))
        runtime = HostRuntime(config)
        result = await handle_message(runtime, "/help")
        assert isinstance(result, HostCommandResult)
        assert "Lisa Framework Host" in result.text
        assert "/help" in result.text

    @pytest.mark.asyncio
    async def test_help_with_transport(self, tmp_path: Path) -> None:
        config = HostConfig.from_env(data_dir=str(tmp_path))
        runtime = HostRuntime(config)

        received: list[str] = []

        class CapturingTransport:
            async def on_stage(
                self, label: str, data: dict[str, object] | None = None
            ) -> None:
                pass

            async def on_text_chunk(self, text: str) -> None:
                pass

            async def on_response(
                self, text: str, data: dict[str, object] | None = None
            ) -> None:
                received.append(text)

            async def on_debug_context(self, context: dict[str, object]) -> None:
                pass

            async def on_error(
                self, message: str, exc: Exception | None = None
            ) -> None:
                pass

        result = await handle_message(runtime, "/help", CapturingTransport())
        assert result.text in received  # transport received the response

    @pytest.mark.asyncio
    async def test_model_turn_no_content_registry(self, tmp_path: Path) -> None:
        """Without setup_framework, a model turn fails gracefully with an error result."""
        config = HostConfig.from_env(data_dir=str(tmp_path))
        runtime = HostRuntime(config)
        result = await handle_message(runtime, "hello world")
        assert isinstance(result, HostCommandResult)
        assert result.is_error is True
        assert "Turn failed" in result.text

    @pytest.mark.asyncio
    async def test_unknown_command_falls_through(self, tmp_path: Path) -> None:
        """An unknown slash command that is not a host shortcut falls through to model routing."""
        config = HostConfig.from_env(data_dir=str(tmp_path))
        runtime = HostRuntime(config)
        result = await handle_message(runtime, "/unknowncommand test")
        assert isinstance(result, HostCommandResult)
        # /unknown is not a host command, so it falls through to model routing,
        # which will fail because the content registry is not set.
        assert result.is_error is True