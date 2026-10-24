"""Tests for host — the shared transport-neutral host runtime."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from runtime.config_resolver import parse_streaming_option, resolve_timezone
from runtime.execution_policy import default_interactive_execution_policy
from runtime.handle_message import handle_message
from runtime.runtime import TokenCounters, TurnContext
from sdk.layout import DATA_DIR, LOGS_DIR

from runtime import (
    HostCommandResult,
    HostConfig,
    HostRuntime,
    NullTransport,
)

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
        assert config.execution_policy == default_interactive_execution_policy()

    def test_from_env_debug_overrides_log_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        config = HostConfig.from_env(debug=True)
        assert config.effective_log_level == "debug"

    def test_prompt_cache_enabled_defaults_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Prompt caching defaults to enabled (no env var set)."""
        monkeypatch.delenv("LISA_PROMPT_CACHE_ENABLED", raising=False)
        config = HostConfig.from_env()
        assert config.prompt_cache_enabled is True

    def test_prompt_cache_enabled_disabled_by_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``LISA_PROMPT_CACHE_ENABLED=false`` is the immediate rollback switch."""
        monkeypatch.setenv("LISA_PROMPT_CACHE_ENABLED", "false")
        config = HostConfig.from_env()
        assert config.prompt_cache_enabled is False

    def test_from_env_data_dir_resolved(self, tmp_path: Path) -> None:
        config = HostConfig.from_env(data_dir=str(tmp_path / "custom_artifacts"))
        assert config.data_dir == (tmp_path / "custom_artifacts").resolve()
        assert config.data_dir.is_absolute()

    def test_resolved_paths(self, tmp_path: Path) -> None:
        config = HostConfig.from_env(data_dir=str(tmp_path))
        assert config.resolved_thread_dir == tmp_path / DATA_DIR / "threads"
        assert config.resolved_host_log_path == tmp_path / LOGS_DIR / "runtime.jsonl"

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

    def test_thread_segment_turns_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LISA_THREAD_SEGMENT_TURNS flows into thread_segment_turns.

        Pinned because ``thread_segment_turns`` is the only thread-store
        knob HostConfig still owns: it is read by every
        ``append_thread_entry`` call site with a runtime in scope
        (see ``handle_message``, ``approval.orchestration``, and the
        ``_log._append_thread_entry`` wrapper).
        """
        config = HostConfig.from_env()
        assert config.thread_segment_turns == 256
        monkeypatch.setenv("LISA_THREAD_SEGMENT_TURNS", "64")
        config = HostConfig.from_env()
        assert config.thread_segment_turns == 64

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
        # The runtime resolves a copy: input paths kept, identity from the session plan.
        assert runtime.config.data_dir == config.data_dir
        assert runtime.config.surface_type == runtime.session_plan.manifest.surface.type
        assert runtime.resources.mcp_started is False
        assert runtime.resources.mcp_tool_pool is None
        assert runtime.session.tokens.session_turn_count == 0
        # Session requests may only tighten the host defaults.
        defaults = default_interactive_execution_policy()
        assert runtime.execution_policy.max_model_calls <= defaults.max_model_calls
        assert runtime.execution_policy.max_capability_calls <= defaults.max_capability_calls

    def test_path_accessors(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        config = HostConfig.from_env(data_dir=str(tmp_path), user_slug="test-user")
        runtime = HostRuntime(config)
        assert runtime.paths.data_dir == tmp_path
        assert runtime.paths.thread_dir == tmp_path / DATA_DIR / "threads"
        assert runtime.paths.thread_path == tmp_path / DATA_DIR / "threads" / "test-user"
        assert runtime.paths.token_usage_path == tmp_path / LOGS_DIR / "token-usage.jsonl"

    def test_thread_path_defaults_to_session_name_when_no_user(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        config = HostConfig.from_env(data_dir=str(tmp_path))
        runtime = HostRuntime(config)
        session_name = runtime.session_plan.manifest.name
        assert runtime.config.user_slug == session_name
        assert runtime.paths.thread_path == runtime.paths.thread_dir / session_name

    def test_thread_path_none_when_no_thread(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        config = HostConfig.from_env(data_dir=str(tmp_path), user_slug="test", no_thread=True)
        runtime = HostRuntime(config)
        assert runtime.paths.thread_path is None

    @pytest.mark.asyncio
    async def test_start_aclose_no_warm_configs(self, tmp_path: Path) -> None:
        config = HostConfig.from_env(data_dir=str(tmp_path))
        runtime = HostRuntime(config)
        await runtime.start()
        assert runtime.resources.mcp_started is True
        assert runtime.resources.mcp_tool_pool is None  # no warm configs
        await runtime.aclose()
        assert runtime.resources.mcp_started is False

    @pytest.mark.asyncio
    async def test_start_idempotent(self, tmp_path: Path) -> None:
        config = HostConfig.from_env(data_dir=str(tmp_path))
        runtime = HostRuntime(config)
        await runtime.start()
        await runtime.start()  # should not raise
        assert runtime.resources.mcp_started is True
        await runtime.aclose()

    @pytest.mark.asyncio
    async def test_aclose_without_start(self, tmp_path: Path) -> None:
        config = HostConfig.from_env(data_dir=str(tmp_path))
        runtime = HostRuntime(config)
        await runtime.aclose()  # should not raise
        assert runtime.resources.mcp_started is False

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
        runtime.resources.set_mcp_configs(
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
            patch("runtime.ambient_resolvers.register_ambient_resolvers", new=lambda *a, **k: None),
            patch("runtime.mcp_pool.register_mcp_aliases", new=lambda *a, **k: None),
        ):
            task = runtime.start_pool_deferred()
            task.set_name("mcp-deferred-start")
            await runtime.ensure_pool_ready()
            assert runtime.resources.mcp_started is True
            assert runtime.resources.mcp_tool_pool is not None
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
            assert runtime.resources.mcp_tool_pool is None
            assert runtime.resources.mcp_started is False
            # The teardown ran in the deferred-start task, not the caller.
            assert "w" in closed_by
            assert closed_by["w"] == "mcp-deferred-start"

    def test_set_mcp_configs(self, tmp_path: Path) -> None:
        config = HostConfig.from_env(data_dir=str(tmp_path))
        runtime = HostRuntime(config)
        runtime.resources.set_mcp_configs(warm=["warm1"], cold=["cold1"])
        assert runtime.resources.mcp_warm_configs == ["warm1"]
        assert runtime.resources.mcp_cold_configs == ["cold1"]

    def test_provider_client_setters(self, tmp_path: Path) -> None:
        config = HostConfig.from_env(data_dir=str(tmp_path))
        runtime = HostRuntime(config)
        runtime.providers.azure_client = "fake_azure"
        assert runtime.providers.azure_client == "fake_azure"
        runtime.providers.openai_client = "fake_openai"
        assert runtime.providers.openai_client == "fake_openai"

    def test_request_restart(self, tmp_path: Path) -> None:
        config = HostConfig.from_env(data_dir=str(tmp_path))
        runtime = HostRuntime(config)
        runtime.session.request_restart("pack enable: lisa-pack-test")
        assert runtime.session.restart_required is True
        assert "pack enable" in runtime.session.restart_reason

    def test_state_snapshot(self, tmp_path: Path) -> None:
        config = HostConfig.from_env(data_dir=str(tmp_path))
        runtime = HostRuntime(config)
        assert runtime.session.state_snapshot is None
        runtime.session.state_snapshot = {"key": "value"}
        assert runtime.session.state_snapshot == {"key": "value"}


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
        assert "Lisa Harness Host" in result.text
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
            context_items: ClassVar[list[object]] = []

        async def _fake_run_intake(*, data_dir: Path) -> _StubState:
            captured["data_dir"] = data_dir
            stub = _StubState()
            stub.context_items = [{"item_id": "state:signal:github"}]
            return stub

        monkeypatch.setattr(
            "runtime.intake.run_intake",
            _fake_run_intake,
        )

        result = await handle_message(runtime, "/intake")
        assert isinstance(result, HostCommandResult)
        assert result.is_error is False
        assert "Intake tick complete" in result.text
        assert "1 context item(s)" in result.text
        assert "state.json" in result.text
        assert captured["data_dir"] == tmp_path
        assert runtime.session.current_mode_id == "intake"

    @pytest.mark.asyncio
    async def test_intake_command_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """/intake surfaces failures without crashing the session."""
        config = HostConfig.from_env(data_dir=str(tmp_path))
        runtime = HostRuntime(config)

        async def _boom(*, data_dir: Path) -> None:
            raise RuntimeError("collector exploded")

        monkeypatch.setattr("runtime.intake.run_intake", _boom)

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
        from runtime.model_discovery import (
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

        monkeypatch.setattr("runtime.commands.discover_models", _fake_discover_models)

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
                    path = runtime.paths.models_path
                else:
                    path = runtime.paths.models_proposal_path

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

        runtime.resources._mcp_tool_pool = _FakeStoragePool()
        runtime.resources._mcp_started = True

        discover_result = await handle_message(runtime, "/models discover")
        assert "proposal_written" in discover_result.text
        assert "presets" in discover_result.text
        assert runtime.paths.models_proposal_path.exists()

        # Proposal should be minimal — no preset annotation comments.
        # Presets live in sdk/router/README.md, not the generated file.
        proposal_content = runtime.paths.models_proposal_path.read_text(encoding="utf-8")
        assert "cost-conscious" not in proposal_content
        assert "preset" not in proposal_content.lower()

        apply_result = await handle_message(runtime, "/models apply")
        assert "Applied models proposal." in apply_result.text
        assert runtime.paths.models_path.exists()

        # Active manifest should be comment-free (yaml.safe_load strips them).
        active_content = runtime.paths.models_path.read_text(encoding="utf-8")
        assert "cost-conscious" not in active_content

    @pytest.mark.asyncio
    async def test_models_apply_refuses_broken_router_cel(
        self,
        tmp_path: Path,
    ) -> None:
        """/models apply refuses a proposal whose router rules do not compile.

        The 2026-09-19 incident: a hand-edited proposal carried a CEL list
        literal with a trailing comma before ``]``. The apply wrote it to
        the active manifest; the router then failed to compile on every
        turn (per-turn warning, silent static-resolution fallback) while
        the operator believed routing was active. Apply must fail fast and
        name the broken rule; the active manifest must be untouched.
        """
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
                if server_id != "lisa-storage":
                    return SimpleNamespace(
                        output_text=f"Unknown MCP server '{server_id}'.",
                        is_error=True,
                    )

                which = str(arguments.get("which") or "proposal").strip().lower()
                if which == "active":
                    path = runtime.paths.models_path
                else:
                    path = runtime.paths.models_proposal_path

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

        runtime.resources._mcp_tool_pool = _FakeStoragePool()
        runtime.resources._mcp_started = True

        # Seed the proposal directly with the incident's broken rule: a
        # trailing comma inside the CEL list literal is a parse error.
        broken_proposal = (
            "# generated-by: lisa /models discover\n"
            "version: 1\n"
            "mode_models: {default: glm-5.3:cloud}\n"
            "router:\n"
            "  backend: lisa-router\n"
            "  rules:\n"
            "    - when: >-\n"
            "        route.operation_id in [\n"
            '          "markdown_forge_document",\n'
            "        ]\n"
            "      use: gpt-5.6-terra\n"
            "models:\n"
            "- model_id: glm-5.3:cloud\n"
            "  provider: ollama\n"
            "  display_name: Glm 5.3:Cloud\n"
            "  capabilities: [text]\n"
            "- model_id: gpt-5.6-terra\n"
            "  provider: openai\n"
            "  display_name: GPT 5.6 Terra\n"
            "  capabilities: [text]\n"
        )
        runtime.paths.models_proposal_path.parent.mkdir(parents=True, exist_ok=True)
        runtime.paths.models_proposal_path.write_text(broken_proposal, encoding="utf-8")

        apply_result = await handle_message(runtime, "/models apply")

        # Apply refuses with the CEL error naming the broken rule…
        assert "Failed to apply proposal" in apply_result.text
        assert "router rules do not compile" in apply_result.text
        assert "router rule 1" in apply_result.text
        # …and the active manifest was NOT written.
        assert not runtime.paths.models_path.exists()

        # A fixed rule applies cleanly.
        fixed_proposal = broken_proposal.replace(
            '"markdown_forge_document",\n        ]',
            '"markdown_forge_document"\n        ]',
        )
        runtime.paths.models_proposal_path.write_text(fixed_proposal, encoding="utf-8")
        apply_result = await handle_message(runtime, "/models apply")
        assert "Applied models proposal." in apply_result.text
        assert runtime.paths.models_path.exists()
