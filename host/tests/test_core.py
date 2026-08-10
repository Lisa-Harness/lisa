"""Tests for host — the shared transport-neutral host runtime."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from host import (
    HostCommandResult,
    HostConfig,
    HostRuntime,
    NullTransport,
)
from host.config_resolver import parse_streaming_option, resolve_timezone
from host.handle_message import handle_message
from host.runtime import TokenCounters, TurnContext

# ── HostConfig ──────────────────────────────────────────────────────


class TestHostConfig:
    def test_from_env_defaults(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        config = HostConfig.from_env()
        assert config.debug is False
        assert config.effective_log_level == "info"
        assert config.timezone in {"UTC", "America/New_York"} or "/" in config.timezone
        assert config.streaming.mode == "hybrid"
        assert config.slice_config.explicit_budget_ms == 2000
        assert config.execution_policy.max_model_calls == 20
        assert config.execution_policy.max_capability_calls == 30
        assert config.execution_policy.max_total_tokens == 50_000
        assert config.execution_policy.max_wall_ms == 300_000

    def test_from_env_debug_overrides_log_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        config = HostConfig.from_env(debug=True)
        assert config.effective_log_level == "debug"

    def test_from_env_data_dir_resolved(self, tmp_path: Path) -> None:
        config = HostConfig.from_env(data_dir=str(tmp_path / "custom_artifacts"))
        assert config.data_dir == (tmp_path / "custom_artifacts").resolve()
        assert config.data_dir.is_absolute()

    def test_resolved_paths(self, tmp_path: Path) -> None:
        config = HostConfig.from_env(data_dir=str(tmp_path))
        assert config.resolved_thread_dir == tmp_path / "threads"
        assert config.resolved_host_log_path == tmp_path / "host.jsonl"

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
        assert mode == "hybrid"


# ── HostRuntime ─────────────────────────────────────────────────────


class TestHostRuntime:
    def test_construction(self, tmp_path: Path) -> None:
        config = HostConfig.from_env(data_dir=str(tmp_path))
        runtime = HostRuntime(config)
        assert runtime.config is config
        assert runtime.mcp_started is False
        assert runtime.mcp_tool_pool is None
        assert runtime.tokens.session_turn_count == 0
        assert runtime.execution_policy.max_model_calls == 20
        assert runtime.execution_policy.max_capability_calls == 30

    def test_path_accessors(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        config = HostConfig.from_env(data_dir=str(tmp_path), user_slug="test-user")
        runtime = HostRuntime(config)
        assert runtime.data_dir == tmp_path
        assert runtime.thread_dir == tmp_path / "threads"
        assert runtime.thread_path == tmp_path / "threads" / "test-user"
        assert runtime.token_usage_path == tmp_path / "token-usage.jsonl"

    def test_thread_path_none_when_no_user(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        config = HostConfig.from_env(data_dir=str(tmp_path))
        runtime = HostRuntime(config)
        assert runtime.thread_path is None

    def test_thread_path_none_when_no_thread(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
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

    @pytest.mark.asyncio
    async def test_deferred_start_then_aclose_owns_cancel_scope(self, tmp_path: Path) -> None:
        """Deferred-start + aclose must not trip anyio's cross-task rule.

        Regression for the /quit crash:
        ``CancelledError: Cancelled via cancel scope ... by <Task pending
        name='Task-136' coro=<HostRuntime.aclose() ...>>``.

        The MCP ``stdio_client`` enters an ``anyio.create_task_group()``
        whose cancel scope binds to the task that entered it. When the
        pool is started via ``start_pool_deferred()``, that task is the
        deferred-start task. ``aclose()`` must therefore tear the pool
        down in that SAME task — not in the caller's task (and not in an
        ``asyncio.shield`` task, which was the original bug). This test
        installs a fake ``_connect`` that enters a real anyio task group
        so the cross-task guard is exercised for real.
        """
        from contextlib import AsyncExitStack
        from unittest.mock import patch

        import anyio
        from kernel.tools.mcp_client import MCPClientConfig, MCPToolPool

        runtime = HostRuntime(HostConfig.from_env(data_dir=str(tmp_path)))
        runtime.set_mcp_configs(
            warm=[
                MCPClientConfig(
                    server_id="w",
                    command="python",
                    args=["-c", "pass"],
                    warm=True,
                    required=False,
                )
            ],
            cold=[],
        )

        closed_by: dict[str, str] = {}

        async def _fake_connect(self: MCPToolPool, config: MCPClientConfig) -> None:
            """Mimic stdio_client: enter an anyio task group whose children
            block on a memory stream and exit promptly when it closes (just
            like the real stdout_reader/stdin_writer exit when the MCP
            streams close on teardown). Records which task closes the stack
            so the test can assert teardown ran in the owning task."""

            send, recv = anyio.create_memory_object_stream(0)

            async def _reader() -> None:
                async with recv:
                    async for _ in recv:
                        pass

            stack = AsyncExitStack()
            await stack.__aenter__()
            tg = await stack.enter_async_context(anyio.create_task_group())
            tg.start_soon(_reader)

            async def _record_close() -> None:
                closed_by[config.server_id] = asyncio.current_task().get_name()

            stack.push_async_callback(_record_close)
            async with self._connect_lock:
                self._child_stacks[config.server_id] = stack
            self._sessions[config.server_id] = object()
            # Keep the send stream open on the stack so closing the stack
            # (aclose) closes it, which unblocks _reader and lets the task
            # group exit cleanly — exactly how stdio_client's finally closes
            # its streams.
            stack.push_async_callback(send.aclose)

        # Skip memory/mcp alias registration (no real content roots in this
        # tmp_path). Patch them out so _build_pool only exercises the pool.
        with (
            patch.object(MCPToolPool, "_connect", new=_fake_connect),
            patch("host.memory_aliases.register_memory_aliases", new=lambda *a, **k: None),
            patch("host.mcp_pool.register_mcp_aliases", new=lambda *a, **k: None),
        ):
            task = runtime.start_pool_deferred()
            task.set_name("mcp-deferred-start")
            await runtime.ensure_pool_ready()
            assert runtime.mcp_started is True
            assert runtime.mcp_tool_pool is not None
            caller_task = asyncio.current_task()
            assert caller_task is not task
            # aclose runs from a DIFFERENT task than the deferred-start task.
            # The fix delegates teardown to the deferred-start task so the
            # anyio cancel scope is exited in the task that entered it. The
            # stack's close callback must therefore record the deferred-start
            # task, not the caller. Without the fix, MCPToolPool.aclose would
            # close the stack from the caller's task and anyio would raise
            # (swallowed by MCPToolPool.aclose's broad except, leaving the
            # callback recorded under the wrong task / not recorded at all).
            await runtime.aclose()
            assert task.done()
            assert runtime.mcp_tool_pool is None
            assert runtime.mcp_started is False
            # The teardown ran in the deferred-start task, not the caller.
            assert "w" in closed_by
            assert closed_by["w"] == "mcp-deferred-start"

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
            async def on_stage(self, label: str, data: dict[str, object] | None = None) -> None:
                pass

            async def on_text_chunk(self, text: str) -> None:
                pass

            async def on_response(self, text: str, data: dict[str, object] | None = None) -> None:
                received.append(text)

            async def on_debug_context(self, context: dict[str, object]) -> None:
                pass

            async def on_error(self, message: str, exc: Exception | None = None) -> None:
                pass

        result = await handle_message(runtime, "/help", CapturingTransport())
        assert result.text in received  # transport received the response

    @pytest.mark.asyncio
    async def test_help_pack_subcommand(self, tmp_path: Path) -> None:
        """/help pack returns the dedicated pack-command guidance."""
        config = HostConfig.from_env(data_dir=str(tmp_path))
        runtime = HostRuntime(config)
        result = await handle_message(runtime, "/help pack")
        assert isinstance(result, HostCommandResult)
        assert "## /pack" in result.text
        assert "/pack list" in result.text
        assert "/pack install" in result.text

    @pytest.mark.asyncio
    async def test_help_models_subcommand(self, tmp_path: Path) -> None:
        """/help models delegates to the models help text."""
        config = HostConfig.from_env(data_dir=str(tmp_path))
        runtime = HostRuntime(config)
        result = await handle_message(runtime, "/help models")
        assert isinstance(result, HostCommandResult)
        assert "## /models" in result.text
        assert "/models discover" in result.text

    @pytest.mark.asyncio
    async def test_help_mcp_subcommand(self, tmp_path: Path) -> None:
        """/help mcp returns the MCP introspection guidance."""
        config = HostConfig.from_env(data_dir=str(tmp_path))
        runtime = HostRuntime(config)
        result = await handle_message(runtime, "/help mcp")
        assert isinstance(result, HostCommandResult)
        assert "## /mcp" in result.text
        assert "/mcp <server_id>" in result.text

    @pytest.mark.asyncio
    async def test_help_unknown_subcommand(self, tmp_path: Path) -> None:
        """/help <unknown> falls back to a pointer rather than the overview."""
        config = HostConfig.from_env(data_dir=str(tmp_path))
        runtime = HostRuntime(config)
        result = await handle_message(runtime, "/help nosuchcommand")
        assert isinstance(result, HostCommandResult)
        assert "No help available" in result.text
        assert "/help" in result.text

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

    @pytest.mark.asyncio
    async def test_models_help_command(self, tmp_path: Path) -> None:
        config = HostConfig.from_env(data_dir=str(tmp_path))
        runtime = HostRuntime(config)
        result = await handle_message(runtime, "/models")
        assert isinstance(result, HostCommandResult)
        assert "## /models" in result.text
        assert "/models discover" in result.text

    @pytest.mark.asyncio
    async def test_models_show_without_manifest(self, tmp_path: Path) -> None:
        config = HostConfig.from_env(data_dir=str(tmp_path))
        runtime = HostRuntime(config)
        result = await handle_message(runtime, "/models show")
        assert isinstance(result, HostCommandResult)
        assert "No active models manifest is available yet." in result.text

    @pytest.mark.asyncio
    async def test_intake_command(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """/intake runs one intake tick in-process and reports the result."""
        config = HostConfig.from_env(data_dir=str(tmp_path))
        runtime = HostRuntime(config)

        captured: dict[str, Path] = {}

        class _StubState:
            context_items: list[object] = []

        async def _fake_run_intake(*, data_dir: Path) -> _StubState:
            captured["data_dir"] = data_dir
            stub = _StubState()
            stub.context_items = [{"item_id": "state:signal:github"}]
            return stub

        monkeypatch.setattr(
            "host.intake.run_intake",
            _fake_run_intake,
        )

        result = await handle_message(runtime, "/intake")
        assert isinstance(result, HostCommandResult)
        assert result.is_error is False
        assert "Intake tick complete" in result.text
        assert "1 context item(s)" in result.text
        assert "state.json" in result.text
        assert captured["data_dir"] == tmp_path
        assert runtime.current_mode_id == "intake"

    @pytest.mark.asyncio
    async def test_intake_command_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """/intake surfaces failures without crashing the session."""
        config = HostConfig.from_env(data_dir=str(tmp_path))
        runtime = HostRuntime(config)

        async def _boom(*, data_dir: Path) -> None:
            raise RuntimeError("collector exploded")

        monkeypatch.setattr("host.intake.run_intake", _boom)

        result = await handle_message(runtime, "/intake")
        assert isinstance(result, HostCommandResult)
        assert "Intake tick failed" in result.text
        assert "collector exploded" in result.text

    @pytest.mark.asyncio
    async def test_models_discover_then_apply(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from host.model_discovery import (
            DiscoveredModel,
            DiscoveryReport,
            ProbeResult,
            ProbeStatus,
        )

        async def _fake_discover_models() -> DiscoveryReport:
            return DiscoveryReport(
                results=(
                    ProbeResult(
                        provider="ollama",
                        status=ProbeStatus.OK,
                        models=(
                            DiscoveredModel(model_id="gemma3:latest", provider="ollama"),
                            DiscoveredModel(
                                model_id="deepseek-r1:latest",
                                provider="ollama",
                                reasoning_model=True,
                            ),
                        ),
                    ),
                )
            )

        monkeypatch.setattr("host.commands.discover_models", _fake_discover_models)

        config = HostConfig.from_env(data_dir=str(tmp_path))
        runtime = HostRuntime(config)

        class _FakeStoragePool:
            async def call_tool(
                self,
                tool_id: str,
                arguments: dict[str, object],
                *,
                server_id: str | None = None,
            ) -> object:
                from types import SimpleNamespace

                if server_id != "lisa-storage":
                    return SimpleNamespace(
                        output_text=f"Unknown MCP server '{server_id}'.",
                        is_error=True,
                    )

                which = str(arguments.get("which") or "proposal").strip().lower()
                if which == "active":
                    path = runtime.models_path
                else:
                    path = runtime.models_proposal_path

                if tool_id == "read_model_manifest":
                    if not path.is_file():
                        return SimpleNamespace(
                            output_text=json.dumps(
                                {"error": f"manifest not found: {which} ({path.name})"}
                            ),
                            is_error=False,
                        )
                    return SimpleNamespace(
                        output_text=path.read_text(encoding="utf-8"), is_error=False
                    )

                if tool_id == "write_model_manifest":
                    content = str(arguments.get("content") or "")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content, encoding="utf-8")
                    return SimpleNamespace(
                        output_text=json.dumps(
                            {
                                "ok": True,
                                "which": which,
                                "bytes": len(content.encode("utf-8")),
                            }
                        ),
                        is_error=False,
                    )

                return SimpleNamespace(
                    output_text=json.dumps({"error": f"unsupported tool: {tool_id}"}),
                    is_error=False,
                )

        runtime._mcp_tool_pool = _FakeStoragePool()
        runtime._mcp_started = True

        discover_result = await handle_message(runtime, "/models discover")
        assert "proposal_written" in discover_result.text
        assert "presets" in discover_result.text
        assert runtime.models_proposal_path.exists()

        # Proposal should be minimal — no preset annotation comments.
        # Presets live in sdk/router/README.md, not the generated file.
        proposal_content = runtime.models_proposal_path.read_text(encoding="utf-8")
        assert "cost-conscious" not in proposal_content
        assert "preset" not in proposal_content.lower()

        apply_result = await handle_message(runtime, "/models apply")
        assert "Applied models proposal." in apply_result.text
        assert runtime.models_path.exists()

        # Active manifest should be comment-free (yaml.safe_load strips them).
        active_content = runtime.models_path.read_text(encoding="utf-8")
        assert "cost-conscious" not in active_content
