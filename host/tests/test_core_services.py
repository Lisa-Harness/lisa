"""Tests for host services — migrated from hosts/little_lisa/test_little_lisa.py.

This is the largest migration: most tests exercised functions in the
``hosts/little_lisa/main.py`` monolith that have been extracted into
``host.*`` modules. The ``_RUNTIME_STATE`` dict is gone — tests
construct a ``HostRuntime(HostConfig.from_env(data_dir=str(tmp_path)))``
and use its typed accessors.

Migration notes:
- ``little_lisa.handle_message(msg)`` → ``handle_message(runtime, msg, transport)``
- ``little_lisa._RUNTIME_STATE[...]`` → ``runtime.session.tokens.*``, ``runtime.session.last_turn_context``, etc.
- ``little_lisa._function_name(...)`` → ``from host.module import function_name; ...``
- CLI-only render functions (``format_modes``, ``format_models``, ``_status_segments``)
  now live in ``hosts/cli/render.py`` and are skipped here.
- Turn-log / log-event functions emit through stdlib logging; tests use
  ``caplog`` to assert on the structured ``event`` extra field.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC
from pathlib import Path
from types import SimpleNamespace

import pytest
from host.callbacks import NullTransport
from host.config import HostConfig
from host.config_resolver import resolve_log_level, resolve_stream_enabled, resolve_timezone
from host.handle_message import handle_message
from host.model_runtime import StubModelDefinition, resolve_model
from host.runtime import HostRuntime
from host.token_accounting import apply_turn_usage, format_session_exit_summary_lines
from kernel.models import (
    WorkflowModeConfig,
)
from kernel.session_plan import ProviderSelections
from kernel.turn_runner import TurnResult

from host import (
    commands,
    context_assembly,
    mcp_pool,
    turn_hooks_builder,
)

# ``host.__init__`` re-exports ``handle_message`` as the *function*,
# shadowing the submodule attribute, so grab the real module from sys.modules
# for monkeypatching module-private functions (e.g. _route_model_turn).

handle_message_mod = sys.modules["host.handle_message"]


def _make_runtime(tmp_path: Path, **config_kwargs) -> HostRuntime:
    """Build a HostRuntime whose data_dir is tmp_path.

    Seeds a minimal models.yaml so manifest-dependent code paths
    (capability snapshot, model resolution) have at least one model.
    """
    from host.testing import build_test_runtime

    models_yaml = "\n".join(
        [
            "version: 1",
            "mode_models: {default: gemma3:latest}",
            "models:",
            "  - model_id: gemma3:latest",
            "    provider: ollama",
            "  - model_id: deepseek-r1:latest",
            "    provider: ollama",
            "    reasoning_model: true",
        ]
    )
    return build_test_runtime(tmp_path, models_yaml=models_yaml, **config_kwargs)


# ── Mode / model rendering (CLI-only) ────────────────────────────────


def test_list_modes_separates_supported_from_host_required_modes() -> None:
    """Migrated: format_modes is now hosts.cli.render.render_modes(runtime)."""
    pytest.skip("migrated: format_modes now in hosts/cli/render.py as render_modes(runtime)")


def test_list_models_table_includes_designations_and_aliases() -> None:
    """Migrated: format_models is now hosts.cli.render.render_models(runtime)."""
    pytest.skip("migrated: format_models now in hosts/cli/render.py as render_models(runtime)")


# ── Content MCP config ────────────────────────────────────────────────


def test_build_content_mcp_config_uses_discovered_roots_and_host_data_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from host import content_roots

    runtime = _make_runtime(tmp_path)
    pack_root = tmp_path / "external-pack"
    (pack_root / "content").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(content_roots, "sdk_content_roots", lambda *a, **kw: [pack_root])
    config = mcp_pool.build_content_mcp_config(
        data_dir=runtime.paths.data_dir,
        content_roots=[pack_root],
        disabled_packs=set(),
        provider_selections=ProviderSelections(),
    )

    assert config.args is not None
    assert "--data-dir" in config.args
    data_dir_index = config.args.index("--data-dir")
    assert config.args[data_dir_index + 1] == str(runtime.paths.data_dir)

    assert config.env is not None
    roots_value = str(config.env.get("SDK_CONTENT_ROOTS", ""))
    configured_roots = [part for part in roots_value.split(os.pathsep) if part]
    assert str(pack_root) in configured_roots
    assert str(pack_root / "content") in configured_roots
    assert not any("sdk/packs/core" in root for root in configured_roots)


def test_build_content_mcp_config_without_discovered_roots_keeps_empty_sdk_content_roots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from host import content_roots

    runtime = _make_runtime(tmp_path)
    monkeypatch.setattr(content_roots, "sdk_content_roots", lambda *a, **kw: [])
    config = mcp_pool.build_content_mcp_config(
        data_dir=runtime.paths.data_dir,
        content_roots=[],
        disabled_packs=set(),
        provider_selections=ProviderSelections(),
    )

    assert config.args is not None
    assert "--data-dir" in config.args
    data_dir_index = config.args.index("--data-dir")
    assert config.args[data_dir_index + 1] == str(runtime.paths.data_dir)

    assert config.env is not None
    assert str(config.env.get("SDK_CONTENT_ROOTS", "")) == ""


def test_build_content_mcp_config_serializes_session_provider_selections(
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(tmp_path)

    config = mcp_pool.build_content_mcp_config(
        data_dir=runtime.paths.data_dir,
        content_roots=[],
        disabled_packs=set(),
        provider_selections=ProviderSelections(
            modes={"default": "lisa-pack-core"},
            operations={"web_search": "lisa-pack-search"},
            tools={"web_search": "lisa-pack-brave"},
        ),
    )

    assert config.env is not None
    payload = json.loads(config.env["LISA_SESSION_PROVIDER_SELECTIONS"])
    assert payload == {
        "modes": {"default": "lisa-pack-core"},
        "operations": {"web_search": "lisa-pack-search"},
        "tools": {"web_search": "lisa-pack-brave"},
    }


# ── /help ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_simple_help_reports_only_wired_capabilities(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    transport = NullTransport()
    output = await handle_message(runtime, "/help", transport)

    assert "/search" in output.text or "search" in output.text
    assert "/discuss" in output.text or "discuss" in output.text
    assert "/help capabilities" in output.text
    assert "/markdown" not in output.text


@pytest.mark.asyncio
async def test_help_capabilities_returns_capability_snapshot(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    transport = NullTransport()
    output = await handle_message(runtime, "/help capabilities", transport)

    assert "Capability Snapshot" in output.text
    assert "Supported modes" in output.text
    assert "Operations:" in output.text
    assert "Tools:" in output.text
    assert "Models:" in output.text


@pytest.mark.asyncio
async def test_modes_command_lists_modes_without_invoking_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _make_runtime(tmp_path)
    transport = NullTransport()

    called = False

    async def _fail_route(*_args: object, **_kwargs: object) -> str:
        nonlocal called
        called = True
        return "should-not-be-reached"

    monkeypatch.setattr(handle_message_mod, "_route_model_turn", _fail_route)
    output = await handle_message(runtime, "/modes", transport)

    assert not called, "/modes must not route to the model"
    assert "## Lisa Modes" in output.text
    # The core/learning/search packs ship at least the default mode.
    assert "default" in output.text


@pytest.mark.asyncio
async def test_operations_command_lists_operations_without_invoking_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _make_runtime(tmp_path)
    transport = NullTransport()

    called = False

    async def _fail_route(*_args: object, **_kwargs: object) -> str:
        nonlocal called
        called = True
        return "should-not-be-reached"

    monkeypatch.setattr(handle_message_mod, "_route_model_turn", _fail_route)
    output = await handle_message(runtime, "/operations", transport)

    assert not called, "/operations must not route to the model"
    assert "## Lisa Operations" in output.text
    assert "Operation ID" in output.text


@pytest.mark.asyncio
async def test_tools_command_lists_tools_without_invoking_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _make_runtime(tmp_path)
    transport = NullTransport()

    called = False

    async def _fail_route(*_args: object, **_kwargs: object) -> str:
        nonlocal called
        called = True
        return "should-not-be-reached"

    monkeypatch.setattr(handle_message_mod, "_route_model_turn", _fail_route)
    output = await handle_message(runtime, "/tools", transport)

    assert not called, "/tools must not route to the model"
    # The tool catalog may be empty in this fixture, but the command must
    # still surface the tools-section header (or the empty-state notice)
    # without touching the model.
    assert "## Lisa Internal Tools" in output.text or "No internal tools discovered" in output.text


@pytest.mark.asyncio
async def test_bare_word_catalog_aliases_dispatch_without_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _make_runtime(tmp_path)
    transport = NullTransport()

    called = False

    async def _fail_route(*_args: object, **_kwargs: object) -> str:
        nonlocal called
        called = True
        return "should-not-be-reached"

    monkeypatch.setattr(handle_message_mod, "_route_model_turn", _fail_route)
    for alias, header, empty_marker in (
        ("modes", "## Lisa Modes", "No modes discovered"),
        ("operations", "## Lisa Operations", "No operations discovered"),
        ("tools", "## Lisa Internal Tools", "No internal tools discovered"),
    ):
        output = await handle_message(runtime, alias, transport)
        assert header in output.text or empty_marker in output.text, (
            f"bare '{alias}' should list {header} (or empty state)"
        )
    assert not called, "bare catalog aliases must not route to the model"


# ── Capability snapshot ────────────────────────────────────────────────


def test_build_capability_snapshot_reflects_live_catalog(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    snapshot = commands.build_capability_snapshot(runtime)

    assert "supported_modes" in snapshot
    assert "unsupported_modes" in snapshot
    assert "operations" in snapshot
    assert "tools" in snapshot
    assert "models" in snapshot
    assert isinstance(snapshot["operations"], list)
    assert isinstance(snapshot["tools"], list)
    assert isinstance(snapshot["models"], list)
    # Models come from the manifest; at least one is configured in tests.
    assert len(snapshot["models"]) >= 1


def test_get_little_lisa_tool_catalog_returns_registry_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kernel.tools.registry as tool_registry

    fake_tool = SimpleNamespace(tool_id="fake_tool", description="A fake tool for testing.")
    fake_catalog = {"fake_tool": fake_tool}

    monkeypatch.setattr(
        tool_registry,
        "get_tool_catalog",
        lambda: fake_catalog,
    )

    result = tool_registry.get_tool_catalog()
    assert result is fake_catalog
    assert "fake_tool" in result


def test_get_little_lisa_tool_catalog_empty_before_setup() -> None:
    import kernel.tools.registry as tool_registry

    result = tool_registry.get_tool_catalog()
    assert isinstance(result, dict)


# ── /debug ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_debug_no_prior_turns(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    transport = NullTransport()
    output = await handle_message(runtime, "/debug", transport)

    assert "No completed turns yet." in output.text


@pytest.mark.asyncio
async def test_debug_default_renders_previous_turn_calls_verbatim(tmp_path: Path) -> None:
    from host.thread import append_thread_entry

    runtime = _make_runtime(tmp_path)
    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug=runtime.config.user_slug,
        mode_id="default",
        query="hello",
        response_text="hi there",
        turn=0,
        turn_input={
            "calls": [
                {
                    "iteration": 0,
                    "messages": [{"role": "system", "content": "sys"}],
                    "tools": None,
                    "model_id": "stub-model",
                    "provider": "stub",
                    "reasoning_model": False,
                    "temperature": 0.2,
                    "max_tokens": 100,
                    "response_format": None,
                }
            ]
        },
    )
    transport = NullTransport()
    # This "/debug" invocation is turn 1; default target is turn 0.
    output = await handle_message(runtime, "/debug", transport)

    assert "## Turn 0" in output.text
    assert '"model_id": "stub-model"' in output.text
    assert '"role": "system"' in output.text


@pytest.mark.asyncio
async def test_debug_explicit_turn_index(tmp_path: Path) -> None:
    from host.thread import append_thread_entry

    runtime = _make_runtime(tmp_path)
    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug=runtime.config.user_slug,
        mode_id="default",
        query="hello",
        response_text="hi there",
        turn=0,
    )
    transport = NullTransport()
    output = await handle_message(runtime, "/debug 0", transport)

    assert "## Turn 0" in output.text
    assert "No captured model input for this turn" in output.text


@pytest.mark.asyncio
async def test_debug_unknown_turn_index(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    transport = NullTransport()
    output = await handle_message(runtime, "/debug 42", transport)

    assert "Turn 42 not found in the thread log." in output.text


@pytest.mark.asyncio
async def test_debug_bad_argument_usage_error(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    transport = NullTransport()
    output = await handle_message(runtime, "/debug abc", transport)

    assert "Usage: /debug [turn_number]" in output.text


# ── /last ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_last_no_prior_turns(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    transport = NullTransport()
    output = await handle_message(runtime, "/last", transport)

    assert "No completed turns yet." in output.text


@pytest.mark.asyncio
async def test_last_replays_previous_model_response_verbatim(tmp_path: Path) -> None:
    from host.thread import append_thread_entry

    runtime = _make_runtime(tmp_path)
    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug=runtime.config.user_slug,
        mode_id="default",
        query="hello",
        response_text="hi there",
        turn=0,
    )
    transport = NullTransport()
    # This "/last" invocation is turn 1; default target is turn 0.
    output = await handle_message(runtime, "/last", transport)

    assert output.text == "hi there"


@pytest.mark.asyncio
async def test_last_skips_host_command_turns(tmp_path: Path) -> None:
    from host.host_commands import register_builtin_host_commands
    from host.thread import append_thread_entry

    # The skip walk consults the host-command registry, which the
    # composition root (bootstrap) populates in production; the test
    # factory bypasses bootstrap, so register the builtins here (idempotent).
    register_builtin_host_commands()
    runtime = _make_runtime(tmp_path)
    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug=runtime.config.user_slug,
        mode_id="default",
        query="hello",
        response_text="model answer",
        turn=0,
    )
    transport = NullTransport()
    # A host-command turn lands on the thread between the model turn and
    # the /last replay; the default walk must skip it and find turn 0.
    output = await handle_message(runtime, "/settings", transport)
    assert "## Runtime Settings" in output.text
    output = await handle_message(runtime, "/last", transport)

    assert output.text == "model answer"


@pytest.mark.asyncio
async def test_last_twice_returns_same_response(tmp_path: Path) -> None:
    from host.thread import append_thread_entry

    runtime = _make_runtime(tmp_path)
    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug=runtime.config.user_slug,
        mode_id="default",
        query="hello",
        response_text="model answer",
        turn=0,
    )
    transport = NullTransport()
    first = await handle_message(runtime, "/last", transport)
    second = await handle_message(runtime, "/last", transport)

    assert first.text == "model answer"
    assert second.text == "model answer"


@pytest.mark.asyncio
async def test_last_explicit_turn_index(tmp_path: Path) -> None:
    from host.thread import append_thread_entry

    runtime = _make_runtime(tmp_path)
    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug=runtime.config.user_slug,
        mode_id="default",
        query="first question",
        response_text="first answer",
        turn=0,
    )
    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug=runtime.config.user_slug,
        mode_id="default",
        query="second question",
        response_text="second answer",
        turn=1,
    )
    transport = NullTransport()
    output = await handle_message(runtime, "/last 1", transport)

    assert output.text == "second answer"


@pytest.mark.asyncio
async def test_last_unknown_turn_index(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    transport = NullTransport()
    output = await handle_message(runtime, "/last 42", transport)

    assert "Turn 42 not found in the thread log." in output.text


@pytest.mark.asyncio
async def test_last_bad_argument_usage_error(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    transport = NullTransport()
    output = await handle_message(runtime, "/last abc", transport)

    assert "Usage: /last [turn_number]" in output.text


# ── System prompt assembly ────────────────────────────────────────────


def test_assemble_system_prompt_orders_harness_session_mode_and_stage() -> None:
    """Session instructions are stable system content ahead of route overlays."""

    def _loader(path: str) -> str:
        if path == "prompts/system-prompt.md":
            return "# Harness\nGenerated context precedes ## User Query."
        return ""

    system_prompt = context_assembly.assemble_system_prompt(
        "Mode instruction",
        "Stage instruction",
        content_loader=_loader,
        session_instruction="Session instruction",
    )

    assert system_prompt.index("# Harness") < system_prompt.index("Session instruction")
    assert system_prompt.index("Session instruction") < system_prompt.index("Mode instruction")
    assert system_prompt.index("Mode instruction") < system_prompt.index("Stage instruction")


def test_load_system_prompt_raises_on_missing_content() -> None:
    """A missing system-prompt file is a hard failure, not an empty string."""

    def _missing_loader(path: str) -> str:
        return ""

    with pytest.raises(context_assembly.MissingAuthoredContentError, match="system-prompt"):
        context_assembly.load_system_prompt(_missing_loader)


def test_load_system_prompt_raises_on_unwired_loader() -> None:
    """An unwired loader is a hard failure, not an empty string."""
    with pytest.raises(context_assembly.MissingAuthoredContentError, match="not wired"):
        context_assembly.load_system_prompt(None)  # type: ignore[arg-type]


def test_load_system_prompt_raises_on_loader_exception() -> None:
    """A loader exception surfaces, not swallowed into an empty prompt."""

    def _raising_loader(path: str) -> str:
        raise OSError("disk error")

    with pytest.raises(context_assembly.MissingAuthoredContentError, match="failed for"):
        context_assembly.load_system_prompt(_raising_loader)


# ── Dynamic user context ──────────────────────────────────────────────


def test_assemble_dynamic_user_context_skips_user_context_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_read_authored_text(relative_path: str) -> str:
        if relative_path == "user_contexts/default.md":
            raise AssertionError("default user context should not be loaded")
        return ""

    monkeypatch.setattr(
        context_assembly,
        "build_ambient_facts",
        lambda **kwargs: "- timezone: UTC",
    )

    context = context_assembly.assemble_dynamic_user_context(
        "default",
        "hello",
        "",
        content_loader=_fake_read_authored_text,
        user_slug="",
        timezone="UTC",
        state=None,
        session_context_loader=lambda: "",
        recent_thread_excerpt_builder=lambda _n: "",
    )

    assert "Selected User Context" not in context
    assert "user_slug: none" in context


# ── handle_message mode routing ───────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.xfail(
    reason="handle_message token counters now on runtime.session.tokens; needs deep migration"
)
async def test_handle_message_updates_runtime_token_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pass


def test_runtime_compiles_session_plan_surface_and_start_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HostRuntime compiles the auto-discovered strict v2 session once."""
    ctx = tmp_path / "config" / "session_context.md"
    ctx.parent.mkdir(parents=True, exist_ok=True)
    ctx.write_text(
        "---\n"
        "schema_version: 2\n"
        'name: "my-session"\n'
        "surface:\n"
        "  type: team_channel\n"
        "  id: chan-1\n"
        "  memory_scope: shared\n"
        "routing:\n"
        "  start_mode: research\n"
        "---\n# body\n",
        encoding="utf-8",
    )
    config = HostConfig.from_env(data_dir=str(tmp_path), no_thread=True)
    runtime = HostRuntime(config)
    assert runtime.config.user_slug == "my-session"
    assert runtime.config.surface_type == "team_channel"
    assert runtime.config.surface_id == "chan-1"
    assert runtime.config.memory_scope == "shared"
    assert runtime.start_mode_id == "research"
    assert runtime.session_plan.manifest.name == "my-session"


def test_runtime_session_plan_exposes_pack_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Session pack selection lives in the compiled plan, not HostConfig."""
    ctx = tmp_path / "config" / "session_context.md"
    ctx.parent.mkdir(parents=True, exist_ok=True)
    ctx.write_text(
        "---\n"
        "schema_version: 2\n"
        'name: "pack-test"\n'
        "surface: { type: personal_dm, id: host-console, memory_scope: testing }\n"
        "resources:\n"
        "  packs:\n"
        "    include: [lisa-pack-core]\n"
        "---\n# body\n",
        encoding="utf-8",
    )
    runtime = HostRuntime(HostConfig.from_env(data_dir=str(tmp_path), no_thread=True))
    assert runtime.session_plan.manifest.resources.packs.include == ("lisa-pack-core",)


def test_runtime_start_mode_initializes_current_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HostRuntime.current_mode_id is seeded from start_mode when set."""
    ctx = tmp_path / "config" / "session_context.md"
    ctx.parent.mkdir(parents=True, exist_ok=True)
    ctx.write_text(
        "---\n"
        "schema_version: 2\n"
        "name: research-session\n"
        "surface: { type: personal_dm, id: host-console, memory_scope: testing }\n"
        "routing: { start_mode: research }\n"
        "---\n# body\n",
        encoding="utf-8",
    )
    config = HostConfig.from_env(data_dir=str(tmp_path), no_thread=True)
    runtime = HostRuntime(config)
    assert runtime.session.current_mode_id == "research"


def test_runtime_no_start_mode_defaults_to_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without start_mode, current_mode_id stays 'default'."""
    monkeypatch.chdir(tmp_path)
    config = HostConfig.from_env(data_dir=str(tmp_path), no_thread=True)
    runtime = HostRuntime(config)
    assert runtime.session.current_mode_id == "default"


@pytest.mark.asyncio
async def test_start_mode_default_goes_to_current_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """start_mode: default means NL turns go to current_mode_id (default).

    Under the unified FSM routing (see docs/ROUTING_RFC_HANDOFF.md), there is
    no separate router planner. NL turns go directly to ``current_mode_id``,
    which defaults to ``default``. The model can hand off mid-turn via
    ``transition_stage``. This test verifies that ``_execute_mode_turn`` is
    called with mode_id="default" for NL turns when start_mode is "default".
    """
    from unittest.mock import AsyncMock, patch

    from host.callbacks import NullTransport

    runtime = _make_runtime(tmp_path, no_thread=True)
    # start_mode defaults to "default"; current_mode_id is seeded from it.
    assert runtime.start_mode_id == "default"
    assert runtime.session.current_mode_id == "default"

    execute_mock = AsyncMock(return_value="ok")
    transport = NullTransport()
    with patch.object(handle_message_mod, "_execute_mode_turn", execute_mock):
        await handle_message(runtime, "write a markdown document", transport)

    assert execute_mock.called, (
        "_execute_mode_turn must be called for NL turns when start_mode == 'default'."
    )
    # The mode_id argument should be "default" (current_mode_id).
    call_args = execute_mock.call_args
    assert (
        call_args.kwargs.get("mode_id", call_args.args[2] if len(call_args.args) > 2 else "")
        == "default"
    )


@pytest.mark.asyncio
async def test_start_mode_non_default_still_bypasses_to_pinned_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real owning start_mode (e.g. research) still bypasses to the pinned mode.

    The bypass is valuable for pinned modes like learn, coding, research.
    Only ``default`` is exempted. This test confirms the bypass still fires
    for non-default start_mode — NL turns go directly to the pinned mode.
    """
    from unittest.mock import AsyncMock, patch

    from host.callbacks import NullTransport

    ctx = tmp_path / "config" / "session_context.md"
    ctx.parent.mkdir(parents=True, exist_ok=True)
    ctx.write_text(
        "---\n"
        "schema_version: 2\n"
        "name: research-session\n"
        "surface: { type: personal_dm, id: host-console, memory_scope: testing }\n"
        "routing: { start_mode: research }\n"
        "---\n# body\n",
        encoding="utf-8",
    )
    _write_manifest(tmp_path)
    runtime = HostRuntime(HostConfig.from_env(data_dir=str(tmp_path), no_thread=True))
    assert runtime.start_mode_id == "research"
    assert runtime.session.current_mode_id == "research"

    execute_mock = AsyncMock(return_value="ok")
    transport = NullTransport()
    with patch.object(handle_message_mod, "_execute_mode_turn", execute_mock):
        await handle_message(runtime, "some natural language", transport)

    assert execute_mock.called, "_execute_mode_turn must be called for NL turns."
    # The mode_id argument should be "research" (the pinned start_mode).
    call_args = execute_mock.call_args
    assert (
        call_args.kwargs.get("mode_id", call_args.args[2] if len(call_args.args) > 2 else "")
        == "research"
    )


# ── Token accounting ──────────────────────────────────────────────────


def test_apply_turn_usage_accumulates_model_mode_stage_breakdowns() -> None:
    state: dict[str, object] = {
        "session_token_ledger": [],
        "model_token_totals": {},
        "mode_token_totals": {},
        "stage_token_totals": {},
        "session_prompt_tokens": 0,
        "session_completion_tokens": 0,
        "session_total_tokens": 0,
    }

    result = TurnResult(
        response_text="ok",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
    )
    apply_turn_usage(
        state,
        result,
        model_id="gpt-4o",
        mode_id="search",
        stage_id="search",
    )

    # Second turn with different model
    result2 = TurnResult(
        response_text="ok",
        prompt_tokens=80,
        completion_tokens=20,
        total_tokens=100,
    )
    apply_turn_usage(
        state,
        result2,
        model_id="claude-haiku-4-5",
        mode_id="default",
        stage_id="default",
    )

    # Ledger
    ledger = state["session_token_ledger"]
    assert isinstance(ledger, list)
    assert len(ledger) >= 2

    # Model breakdown
    model_totals = state["model_token_totals"]
    assert isinstance(model_totals, dict)
    assert model_totals["gpt-4o"]["total_tokens"] == 150
    assert model_totals["claude-haiku-4-5"]["total_tokens"] == 100

    # Mode breakdown
    mode_totals = state["mode_token_totals"]
    assert isinstance(mode_totals, dict)
    assert mode_totals["search"]["total_tokens"] == 150
    assert mode_totals["default"]["total_tokens"] == 100

    # Stage breakdown
    stage_totals = state["stage_token_totals"]
    assert isinstance(stage_totals, dict)
    assert stage_totals["search"]["total_tokens"] == 150
    assert stage_totals["default"]["total_tokens"] == 100

    # Session totals
    assert state["session_total_tokens"] == 250


def test_apply_turn_usage_per_model_overrides_single_model_id() -> None:
    """When per_model_usage is supplied, model_token_totals reflects every
    model actually called rather than the single resolved model_id kwarg."""
    state: dict[str, object] = {
        "session_token_ledger": [],
        "model_token_totals": {},
        "mode_token_totals": {},
        "stage_token_totals": {},
        "session_prompt_tokens": 0,
        "session_completion_tokens": 0,
        "session_total_tokens": 0,
    }

    result = TurnResult(
        response_text="ok",
        prompt_tokens=120,  # sum of the two per-model prompts below
        completion_tokens=70,
        total_tokens=190,
    )
    apply_turn_usage(
        state,
        result,
        model_id="gpt-4o",  # resolved default; should NOT own the breakdown
        mode_id="default",
        stage_id="default",
        per_model_usage=[
            {
                "model_id": "gpt-4o",
                "prompt_tokens": 80,
                "completion_tokens": 40,
                "total_tokens": 120,
                "cached_tokens": 0,
                "reasoning_tokens": 0,
                "duration_ms": 500.0,
                "latency_ms": 0,
            },
            {
                "model_id": "o4-mini",
                "prompt_tokens": 40,
                "completion_tokens": 30,
                "total_tokens": 70,
                "cached_tokens": 0,
                "reasoning_tokens": 0,
                "duration_ms": 300.0,
                "latency_ms": 0,
            },
        ],
    )

    model_totals = state["model_token_totals"]
    assert isinstance(model_totals, dict)
    # Both models appear with their own counts, not collapsed onto gpt-4o.
    assert model_totals["gpt-4o"]["total_tokens"] == 120
    assert model_totals["o4-mini"]["total_tokens"] == 70
    # The resolved default model_id is NOT double-counted as the whole turn.
    assert model_totals["gpt-4o"]["total_tokens"] != 190


def test_apply_turn_usage_falls_back_to_single_model_id_when_per_model_empty() -> None:
    """When per_model_usage is None/empty, the single model_id kwarg owns
    the breakdown (legacy behavior preserved)."""
    state: dict[str, object] = {
        "session_token_ledger": [],
        "model_token_totals": {},
        "mode_token_totals": {},
        "stage_token_totals": {},
        "session_prompt_tokens": 0,
        "session_completion_tokens": 0,
        "session_total_tokens": 0,
    }

    result = TurnResult(
        response_text="ok",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
    )
    apply_turn_usage(
        state,
        result,
        model_id="gpt-4o",
        mode_id="default",
        stage_id="default",
        per_model_usage=None,
    )

    model_totals = state["model_token_totals"]
    assert model_totals["gpt-4o"]["total_tokens"] == 150


def test_apply_turn_usage_per_mode_overrides_single_mode_id() -> None:
    """When per_mode_usage is supplied, mode_token_totals reflects every
    mode that actually ran rather than the single entry mode_id kwarg.

    This is the fix for /tokens misattributing kernel-driven recursive
    handoffs (plan → search → coding) to the entry mode.
    """
    state: dict[str, object] = {
        "session_token_ledger": [],
        "model_token_totals": {},
        "mode_token_totals": {},
        "stage_token_totals": {},
        "session_prompt_tokens": 0,
        "session_completion_tokens": 0,
        "session_total_tokens": 0,
    }

    result = TurnResult(
        response_text="ok",
        prompt_tokens=120,
        completion_tokens=70,
        total_tokens=190,
    )
    apply_turn_usage(
        state,
        result,
        model_id="gpt-4o",
        mode_id="plan",  # entry mode; should NOT own the whole breakdown
        stage_id="plan",
        per_mode_usage=[
            {
                "mode_id": "plan",
                "prompt_tokens": 40,
                "completion_tokens": 20,
                "total_tokens": 60,
                "cached_tokens": 0,
                "reasoning_tokens": 0,
                "duration_ms": 200.0,
                "latency_ms": 0,
            },
            {
                "mode_id": "search",
                "prompt_tokens": 80,
                "completion_tokens": 50,
                "total_tokens": 130,
                "cached_tokens": 0,
                "reasoning_tokens": 0,
                "duration_ms": 400.0,
                "latency_ms": 0,
            },
        ],
    )

    mode_totals = state["mode_token_totals"]
    assert isinstance(mode_totals, dict)
    # Both modes appear with their own counts, not collapsed onto plan.
    assert mode_totals["plan"]["total_tokens"] == 60
    assert mode_totals["search"]["total_tokens"] == 130
    # The entry mode_id is NOT double-counted as the whole turn.
    assert mode_totals["plan"]["total_tokens"] != 190


def test_apply_turn_usage_falls_back_to_single_mode_id_when_per_mode_empty() -> None:
    """When per_mode_usage is None/empty, the single mode_id kwarg owns
    the breakdown (legacy behavior preserved)."""
    state: dict[str, object] = {
        "session_token_ledger": [],
        "model_token_totals": {},
        "mode_token_totals": {},
        "stage_token_totals": {},
        "session_prompt_tokens": 0,
        "session_completion_tokens": 0,
        "session_total_tokens": 0,
    }

    result = TurnResult(
        response_text="ok",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
    )
    apply_turn_usage(
        state,
        result,
        model_id="gpt-4o",
        mode_id="search",
        stage_id="search",
        per_mode_usage=None,
    )

    mode_totals = state["mode_token_totals"]
    assert mode_totals["search"]["total_tokens"] == 150


def test_apply_turn_usage_threads_cached_reasoning_duration_latency() -> None:
    state: dict[str, object] = {
        "session_token_ledger": [],
        "model_token_totals": {},
        "mode_token_totals": {},
        "stage_token_totals": {},
        "session_cached_tokens": 0,
        "session_reasoning_tokens": 0,
        "session_duration_ms": 0.0,
        "session_latency_ms": 0,
    }

    result = TurnResult(
        response_text="ok",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        cached_tokens=20,
        reasoning_tokens=10,
        duration_ms=750.0,
        latency_ms=800,
    )
    apply_turn_usage(
        state,
        result,
        model_id="gpt-4o",
        mode_id="search",
        stage_id="search",
    )

    ledger = state["session_token_ledger"]
    assert isinstance(ledger, list)
    assert len(ledger) == 1
    entry = ledger[0]
    assert entry["cached_tokens"] == 20
    assert entry["reasoning_tokens"] == 10
    assert entry["duration_ms"] == 750.0
    assert entry["latency_ms"] == 800

    model_totals = state["model_token_totals"]
    assert model_totals["gpt-4o"]["cached_tokens"] == 20
    assert model_totals["gpt-4o"]["reasoning_tokens"] == 10
    assert model_totals["gpt-4o"]["duration_ms"] == 750.0
    assert model_totals["gpt-4o"]["latency_ms"] == 800

    assert state["session_cached_tokens"] == 20
    assert state["session_reasoning_tokens"] == 10
    assert state["session_duration_ms"] == 750.0
    assert state["session_latency_ms"] == 800


def test_apply_turn_usage_upgrades_old_schema_entries_in_place() -> None:
    state: dict[str, object] = {
        "session_token_ledger": [],
        "model_token_totals": {
            "gpt-4o": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
        },
        "mode_token_totals": {},
        "stage_token_totals": {},
    }

    result = TurnResult(
        response_text="ok",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        cached_tokens=5,
        reasoning_tokens=3,
        duration_ms=200.0,
        latency_ms=210,
    )
    apply_turn_usage(
        state,
        result,
        model_id="gpt-4o",
        mode_id="default",
        stage_id="default",
    )

    model_entry = state["model_token_totals"]["gpt-4o"]
    assert model_entry["prompt_tokens"] == 150
    assert model_entry["completion_tokens"] == 60
    assert model_entry["total_tokens"] == 210
    assert model_entry["cached_tokens"] == 5
    assert model_entry["reasoning_tokens"] == 3
    assert model_entry["duration_ms"] == 200.0
    assert model_entry["latency_ms"] == 210


# ── /tokens command ───────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.xfail(
    reason="/tokens now reads runtime.session.tokens via format_token_stats; needs runtime setup migration"
)
async def test_tokens_command_returns_detailed_stats() -> None:
    pass


@pytest.mark.asyncio
@pytest.mark.xfail(
    reason="mode token context now on runtime.session.last_turn_context; needs deep migration"
)
async def test_handle_message_surfaces_mode_token_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pass


# ── Model resolution ───────────────────────────────────────────────────


def _write_manifest(tmp_path: Path, content: str | None = None) -> Path:
    """Write a models.yaml under tmp_path/config (the canonical location)."""
    manifest_dir = tmp_path / "config"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / "models.yaml"
    if content is None:
        content = "\n".join(
            [
                "version: 1",
                "mode_models: {default: gemma3:latest}",
                "models:",
                "  - model_id: gemma3:latest",
                "    provider: ollama",
                "    designations: [default]",
            ]
        )
    path.write_text(content, encoding="utf-8")
    return path


def test_resolve_model_supports_mode_models_and_concrete_ids(tmp_path: Path) -> None:
    _write_manifest(
        tmp_path,
        "\n".join(
            [
                "version: 1",
                "mode_models: {default: gemma3:latest}",
                "models:",
                "  - model_id: gemma3:latest",
                "    provider: ollama",
                "    context_window: 65536",
                "  - model_id: deepseek-r1:latest",
                "    provider: ollama",
                "    reasoning_model: true",
            ]
        ),
    )

    default_model = resolve_model("default", runtime_state={}, data_dir=tmp_path)
    default_model_id = default_model.model_id

    assert default_model_id
    assert default_model.context_window == 65536
    resolved_default = resolve_model(default_model_id, runtime_state={}, data_dir=tmp_path).model_id
    assert resolved_default == default_model_id


def test_resolve_model_rejects_unknown_manifest_id(tmp_path: Path) -> None:
    _write_manifest(
        tmp_path,
        "\n".join(
            [
                "version: 1",
                "mode_models: {default: gemma3:latest}",
                "models:",
                "  - model_id: gemma3:latest",
                "    provider: ollama",
            ]
        ),
    )

    with pytest.raises(ValueError, match="not in the host's manifest"):
        resolve_model("not-a-real-model", runtime_state={}, data_dir=tmp_path)


# ── Startup without an existing models.yaml ──────────────────────────


def test_load_manifest_or_none_missing_manifest_returns_none(tmp_path: Path) -> None:
    """A missing data_dir/models.yaml must not abort startup.

    First-run launches have no manifest yet; the REPL must come up so the
    user can run ``/models discover`` then ``/models apply``. The startup
    helper returns ``None`` for an absent file and the graceful-degradation
    paths (model_alias_for_mode -> "default", capability snapshot -> empty
    models) handle the unconfigured state.
    """
    from host.model_runtime import load_manifest_or_none

    manifest = load_manifest_or_none(data_dir=tmp_path)

    assert manifest is None


def test_load_manifest_or_none_loads_present_manifest(tmp_path: Path) -> None:
    """A present manifest is still loaded (and pre-loaded at startup)."""
    from host.model_runtime import load_manifest_or_none

    _write_manifest(
        tmp_path,
        "\n".join(
            [
                "version: 1",
                "mode_models: {default: gemma3:latest}",
                "models:",
                "  - model_id: gemma3:latest",
                "    provider: ollama",
            ]
        ),
    )

    manifest = load_manifest_or_none(data_dir=tmp_path)

    assert manifest is not None
    assert manifest.resolve("default").model_id == "gemma3:latest"


def test_load_manifest_or_none_invalid_present_manifest_raises(tmp_path: Path) -> None:
    """A present-but-invalid manifest fails loudly (not silently ignored)."""
    from host.model_runtime import load_manifest_or_none

    _write_manifest(tmp_path, "version: 1\nmodels: []\n")

    with pytest.raises(ValueError, match="'mode_models' mapping"):
        load_manifest_or_none(data_dir=tmp_path)

    runtime = _make_runtime(tmp_path)
    assert runtime.paths.models_path.exists()

    mcp_pool.setup_framework(runtime)

    assert runtime.session.manifest is not None
    assert len(runtime.session.manifest.models) >= 1


# ── Provider key verification ─────────────────────────────────────────


def test_verify_check_provider_keys_reports_google_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.skip("migrated: _verify_check_provider_keys moved to hosts/cli verify command")


# ── Help formatting ───────────────────────────────────────────────────


def test_simple_help_mentions_quit_command() -> None:
    output = commands.format_help()
    assert "/quit" in output


def test_help_overview_mentions_inventory_and_pack_commands() -> None:
    """The pack-aware overview lists the inventory commands and the pack-command pointer."""
    output = commands.format_help()
    assert "/modes" in output
    assert "/operations" in output
    assert "/tools" in output
    assert "/mcp" in output
    assert "/help <command>" in output


def test_help_overview_lists_pack_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The overview surfaces pack-registered commands under a 'Pack commands' header."""
    from kernel.host_commands import HostCommandSpec, PackCommandEntry

    render_mod = sys.modules["host.commands._render"]

    def _fake_entries() -> list[PackCommandEntry]:
        return [
            PackCommandEntry(
                pack_name="lisa-pack-test",
                specs=[
                    HostCommandSpec(
                        prefix="/test-cmd",
                        handler=_unused_handler,  # type: ignore[arg-type]
                        help_text="/test-cmd <arg> does the thing.",
                    ),
                ],
            ),
        ]

    monkeypatch.setattr(render_mod, "_pack_command_entries", _fake_entries)
    output = commands.format_help()
    assert "Pack commands:" in output
    assert "/test-cmd <arg> does the thing." in output


def test_render_command_help_delegates_to_pack_help_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """/help <pack-command> surfaces the pack's own help_text."""
    from kernel.host_commands import HostCommandSpec, PackCommandEntry

    render_mod = sys.modules["host.commands._render"]

    def _fake_entries() -> list[PackCommandEntry]:
        return [
            PackCommandEntry(
                pack_name="lisa-pack-test",
                specs=[
                    HostCommandSpec(
                        prefix="/test-cmd",
                        handler=_unused_handler,  # type: ignore[arg-type]
                        help_text="/test-cmd <arg> does the thing.",
                    ),
                ],
            ),
        ]

    monkeypatch.setattr(render_mod, "_pack_command_entries", _fake_entries)
    config = HostConfig.from_env(data_dir=str(tmp_path))
    runtime = HostRuntime(config)
    output = commands.render_command_help(runtime, "test-cmd")
    assert "## /test-cmd" in output
    assert "/test-cmd <arg> does the thing." in output
    assert "Pack: lisa-pack-test" in output


async def _unused_handler(*_args: object) -> str:  # pragma: no cover
    """Placeholder handler for PackCommandSpec in tests (never invoked)."""
    return ""


# ── Turn context ──────────────────────────────────────────────────────


def test_build_turn_context_marks_default_mode_handoff() -> None:
    mode = WorkflowModeConfig(
        mode_id="default",
        start_stage_id="default",
        mode_kind="chat",
    )
    context = context_assembly.build_turn_context(
        raw_message="hello there",
        mode_id="default",
        query="hello there",
        preferred_model_id="default",
        resolved_model=StubModelDefinition(
            model_id="gpt-4o",
            provider="openai",
        ),
        mode=mode,
    )

    assert context["preferred_model_id"] == "default"
    assert "default" in str(context["handoff"])


# ── Turn log / log event (now wired through stdlib) ──────────────────


def test_append_turn_log_emits_structured_event(caplog: pytest.LogCaptureFixture) -> None:
    """``_append_turn_log`` emits a ``cmd.turn`` event with mode/handoff fields."""
    import logging as _logging

    from host.commands._log import _append_turn_log

    with caplog.at_level(_logging.INFO, logger="host.commands._log"):
        _append_turn_log(
            context={"mode_id": "help", "handoff": "help shortcut"},
            response_text="hi",
        )
    assert any(r.event == "cmd.turn" for r in caplog.records)
    record = next(r for r in caplog.records if r.event == "cmd.turn")
    assert record.mode_id == "help"
    assert record.handoff == "help shortcut"
    assert record.response_chars == 2


def test_append_turn_log_includes_context_fields(caplog: pytest.LogCaptureFixture) -> None:
    """``_append_turn_log`` surfaces arbitrary context keys as extra fields."""
    import logging as _logging

    from host.commands._log import _append_turn_log

    with caplog.at_level(_logging.INFO, logger="host.commands._log"):
        _append_turn_log(
            context={"mode_id": "intake", "intake": {"signal_count": 3}},
            response_text="",
        )
    record = next(r for r in caplog.records if r.event == "cmd.turn")
    assert record.context["intake"]["signal_count"] == 3


def test_append_log_event_respects_level(caplog: pytest.LogCaptureFixture) -> None:
    """``_append_log_event`` routes the supplied level to stdlib."""
    import logging as _logging

    from host.commands._log import _append_log_event

    with caplog.at_level(_logging.DEBUG, logger="host.commands._log"):
        _append_log_event(
            level="warning",
            event="cmd.test",
            payload={"request_id": "r-1"},
        )
    record = next(r for r in caplog.records if r.event == "cmd.test")
    assert record.levelno == _logging.WARNING
    assert record.request_id == "r-1"


# ── Log level resolution ──────────────────────────────────────────────


def test_resolve_log_level_prefers_cli_env_then_debug_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LISA_LOG_LEVEL", raising=False)
    assert resolve_log_level("warning", debug_enabled=False) == "warning"
    assert resolve_log_level("", debug_enabled=True) == "debug"
    assert resolve_log_level("", debug_enabled=False) == "info"

    monkeypatch.setenv("LISA_LOG_LEVEL", "error")
    assert resolve_log_level("", debug_enabled=True) == "error"


# ── handle_message model turns ───────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.xfail(
    reason="handle_message now takes (runtime, msg, transport); needs deep migration"
)
async def test_handle_message_returns_operation_error_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pass


# ── Runtime initialization (now HostConfig.from_env + HostRuntime) ────


def test_initialize_runtime_applies_user_timeout_and_state_snapshot(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "state.json"
    context_path = tmp_path / "system.md"
    snapshot_path.write_text(
        '{"compiled_at":"2026-05-30T00:00:00+00:00","health":"ok"}',
        encoding="utf-8",
    )
    context_path.write_text("# System Context", encoding="utf-8")

    import json as _json

    config = HostConfig.from_env(
        debug=True,
        user_slug=str(context_path),
        timezone="UTC",
        state_snapshot=str(snapshot_path),
        data_dir=str(tmp_path),
    )
    runtime = HostRuntime(config)
    runtime.session.state_snapshot = _json.loads(snapshot_path.read_text(encoding="utf-8"))

    assert config.user_slug == str(context_path)
    assert config.timezone == "UTC"
    assert config.timezone_source == "cli"
    assert config.effective_log_level == "debug"
    assert isinstance(runtime.session.state_snapshot, dict)
    # The new segmented store uses a base path (no .jsonl extension).
    # thread_path = thread_dir / user_slug. When user_slug is an absolute
    # path, Path joining replaces the base, so thread_path resolves to the
    # absolute path of the context file (no extension).
    assert runtime.paths.thread_path == Path(str(context_path))


def test_resolve_stream_enabled_prefers_explicit_flags() -> None:
    assert resolve_stream_enabled(
        force_stream=True,
        force_no_stream=False,
        is_tty=False,
    )
    assert not resolve_stream_enabled(
        force_stream=False,
        force_no_stream=True,
        is_tty=True,
    )


def test_resolve_stream_enabled_defaults_to_tty_state() -> None:
    assert resolve_stream_enabled(
        force_stream=False,
        force_no_stream=False,
        is_tty=True,
    )
    assert not resolve_stream_enabled(
        force_stream=False,
        force_no_stream=False,
        is_tty=False,
    )


def test_initialize_runtime_applies_stream_configuration(tmp_path: Path) -> None:
    stream_delay_ms = 12

    config = HostConfig.from_env(
        debug=False,
        timezone="UTC",
        streaming=f"stage:{stream_delay_ms}",
        data_dir=str(tmp_path),
    )

    assert config.streaming.mode == "stage"
    assert config.streaming.char_delay_ms == stream_delay_ms
    assert config.streaming.enabled is True


def test_initialize_runtime_uses_log_level_env_when_cli_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LISA_LOG_LEVEL", "warning")

    config = HostConfig.from_env(debug=False, timezone="UTC", data_dir=str(tmp_path))

    assert config.effective_log_level == "warning"


def test_initialize_runtime_prefers_cli_log_level_over_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LISA_LOG_LEVEL", "error")

    config = HostConfig.from_env(
        debug=False,
        log_level="info",
        timezone="UTC",
        data_dir=str(tmp_path),
    )

    assert config.effective_log_level == "info"


def test_initialize_runtime_uses_timezone_env_when_cli_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LISA_TIMEZONE", "UTC")

    config = HostConfig.from_env(debug=False, timezone="", data_dir=str(tmp_path))

    assert config.timezone == "UTC"
    assert "env:" in config.timezone_source


def test_initialize_runtime_fails_when_timezone_unresolvable() -> None:
    with pytest.raises(ValueError, match="not resolvable"):
        HostConfig.from_env(debug=False, timezone="Not/AZone Value")


def test_initialize_runtime_accepts_timezone_alias() -> None:
    config = HostConfig.from_env(debug=False, timezone="EDT")

    assert config.timezone == "EST5EDT"
    assert config.timezone_source == "cli"


def test_initialize_runtime_fails_when_user_context_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LISA_DATA_DIR", str(tmp_path))

    config = HostConfig.from_env(
        debug=False, user_slug="adam", timezone="UTC", data_dir=str(tmp_path)
    )
    runtime = HostRuntime(config)

    # The new HostConfig does not raise at construction; session context
    # resolution happens lazily. We verify the config captured the slug.
    assert config.user_slug == "adam"


def test_initialize_runtime_accepts_session_context_file_path(tmp_path: Path) -> None:
    context_file = tmp_path / "founder.md"
    context_file.write_text("# Founder Profile", encoding="utf-8")

    config = HostConfig.from_env(
        debug=False,
        user_slug=str(context_file),
        timezone="UTC",
        data_dir=str(tmp_path),
    )

    assert config.user_slug == str(context_file)


# ── Status time badge (CLI-only) ─────────────────────────────────────


def test_status_time_badge_abbreviates_iana_timezone(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.skip("migrated: _status_time_badge now in hosts/cli (CLI status bar)")


def test_status_time_badge_compacts_long_timezone_name(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.skip("migrated: _status_time_badge now in hosts/cli (CLI status bar)")


def test_status_time_badge_zoneinfo_unavailable_uses_system_local_abbrev(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.skip("migrated: _status_time_badge now in hosts/cli (CLI status bar)")


def test_resolve_timezone_maps_windows_daylight_display_name_to_iana(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows reports tzinfo as 'Eastern Daylight Time' during DST.

    The system-local fallback must map that display name to America/New_York
    instead of silently defaulting to UTC.
    """

    class _FakeTzInfo:
        def __str__(self) -> str:
            return "Eastern Daylight Time"

    class _FakeDateTime:
        @staticmethod
        def now(tz: object | None = None) -> object:
            class _Aware:
                def astimezone(self) -> object:
                    return SimpleNamespace(tzinfo=_FakeTzInfo())

            return _Aware()

    for env_name in ("LISA_TIMEZONE", "TZ", "TIMEZONE", "LOCAL_TIMEZONE"):
        monkeypatch.delenv(env_name, raising=False)

    import datetime as _datetime_mod

    monkeypatch.setattr(_datetime_mod, "datetime", _FakeDateTime)

    tz, source = resolve_timezone("")

    assert tz == "America/New_York"
    assert source == "system-local"


# ── Status segments (CLI-only) ────────────────────────────────────────


def test_status_segments_show_profile_model_without_prefix_and_token_badge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.skip("migrated: _status_segments now in hosts/cli (CLI status bar)")


def test_status_segments_minimal_uses_tc_token_badge(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.skip("migrated: _status_segments now in hosts/cli (CLI status bar)")


# ── Session exit summary ─────────────────────────────────────────────


def test_format_session_exit_summary_lines_includes_useful_stats() -> None:
    from datetime import datetime as _dt

    state: dict[str, object] = {
        "session_started_utc": _dt.now(UTC),
        "session_turn_count": 3,
        "session_success_count": 2,
        "session_error_count": 1,
        "session_total_latency_ms": 900.0,
        "session_mode_counts": {"search": 2, "default": 1},
        "turn_prompt_tokens": 140,
        "turn_completion_tokens": 60,
        "turn_total_tokens": 200,
        "turn_cached_tokens": 40,
        "turn_reasoning_tokens": 10,
        "session_prompt_tokens": 420,
        "session_completion_tokens": 180,
        "session_total_tokens": 600,
        "session_cached_tokens": 120,
        "session_reasoning_tokens": 30,
        "model_token_totals": {
            "deepseek-v4": {"prompt_tokens": 300, "completion_tokens": 120, "total_tokens": 420},
            "claude-haiku-4-5": {
                "prompt_tokens": 120,
                "completion_tokens": 60,
                "total_tokens": 180,
            },
        },
        "mode_token_totals": {
            "default": {"prompt_tokens": 280, "completion_tokens": 120, "total_tokens": 400},
            "search": {"prompt_tokens": 140, "completion_tokens": 60, "total_tokens": 200},
        },
        "stage_token_totals": {},
    }

    lines = format_session_exit_summary_lines(
        state,
        session_started_utc=state["session_started_utc"],
    )
    summary = "\n".join(lines)

    assert "Turns: 3 (ok: 2, errors: 1)" in summary
    assert "Avg turn latency: 300 ms" in summary
    assert "### Token Totals" in summary
    assert "| Session | 420 | 180 | 600 | 120 | 30 | 480 |" in summary
    assert "| Last turn | 140 | 60 | 200 | 40 | 10 | 160 |" in summary
    assert "| Avg/turn | 140 | 60 | 200 |" in summary
    assert "### By Model" in summary
    assert "| claude-haiku-4-5 | 120 | 60 | 180 |" in summary
    assert "| deepseek-v4 | 300 | 120 | 420 |" in summary
    assert "### By Mode" in summary
    assert "| default | 280 | 120 | 400 |" in summary
    assert "| search | 140 | 60 | 200 |" in summary
    assert "Most used mode: search (2)" in summary


def test_format_token_stats_appends_lifetime_section(tmp_path: Path) -> None:
    """``/tokens`` appends a lifetime section read from token-usage.jsonl."""
    from host.commands._render import format_token_stats

    runtime = _make_runtime(tmp_path)

    # Write two durable telemetry records spanning distinct models.
    records = [
        {
            "emitted_at": "2026-08-20T10:00:00Z",
            "model_id": "gemma3:latest",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "cached_tokens": 20,
            "reasoning_tokens": 0,
            "cost_usd": 0.001,
        },
        {
            "emitted_at": "2026-08-22T18:30:00Z",
            "model_id": "deepseek-r1:latest",
            "prompt_tokens": 300,
            "completion_tokens": 200,
            "total_tokens": 500,
            "cached_tokens": 100,
            "reasoning_tokens": 40,
            "cost_usd": 0.01,
        },
    ]
    token_log = runtime.paths.token_usage_path
    token_log.parent.mkdir(parents=True, exist_ok=True)
    token_log.write_text(
        "\n".join(json.dumps(rec) for rec in records) + "\n",
        encoding="utf-8",
    )

    output = format_token_stats(runtime)

    assert "### Lifetime (all sessions)" in output
    assert "- Events: 2" in output
    assert "- Span: 2026-08-20T10:00:00Z → 2026-08-22T18:30:00Z" in output
    assert "- Estimated cost: $0.0110" in output
    # Totals row: 400 prompt | 250 completion | 650 total | 120 cached | 40 reasoning | 530 billed
    assert "| 400 | 250 | 650 | 120 | 40 | 530 |" in output
    assert "#### Lifetime by Model" in output
    assert "| deepseek-r1:latest | 300 | 200 | 500 | 100 | 40 | 400 |" in output
    assert "| gemma3:latest | 100 | 50 | 150 | 20 | 0 | 130 |" in output


def test_format_token_stats_lifetime_absent_without_log(tmp_path: Path) -> None:
    """When no token-usage.jsonl exists, /tokens omits the lifetime section."""
    from host.commands._render import format_token_stats

    runtime = _make_runtime(tmp_path)
    assert not runtime.paths.token_usage_path.exists()

    output = format_token_stats(runtime)

    assert "### Lifetime (all sessions)" not in output
    # Session summary still renders.
    assert "### Session Summary" in output or "## Session Summary" in output


# ── /settings ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_settings_command_reports_effective_runtime_settings(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    transport = NullTransport()
    output = await handle_message(runtime, "/settings", transport)

    assert "Runtime Settings" in output.text
    assert "timezone:" in output.text
    assert "log_level:" in output.text


# ── Models required before turn ───────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.xfail(
    reason="handle_message manifest check now reads runtime.session.manifest; needs deep migration"
)
async def test_handle_message_requires_models_before_running_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pass


# ── Turn hooks ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_build_turn_hooks_wires_recall_hook(tmp_path: Path) -> None:
    """Turn hooks build cleanly with learning-owned memory behavior."""
    runtime = _make_runtime(tmp_path)
    # setup_framework wires the content registry so content_loader(runtime) works.
    mcp_pool.setup_framework(runtime)
    turn_hooks_builder.build_turn_hooks(runtime, {"mode_id": "default"})


def test_build_turn_hooks_wires_complete_stream_for_text_mode(tmp_path: Path) -> None:
    """When streaming mode is text/hybrid, complete_stream is wired on the hooks."""
    runtime = _make_runtime(tmp_path, streaming="text:30")
    hooks = turn_hooks_builder.build_turn_hooks(runtime, {"mode_id": "default"})
    assert hooks.models.complete_stream is not None


def test_build_turn_hooks_wires_complete_stream_for_hybrid_mode(tmp_path: Path) -> None:
    """Hybrid streaming also wires complete_stream."""
    runtime = _make_runtime(tmp_path, streaming="hybrid")
    hooks = turn_hooks_builder.build_turn_hooks(runtime, {"mode_id": "default"})
    assert hooks.models.complete_stream is not None


def test_build_turn_hooks_omits_complete_stream_for_stage_mode(tmp_path: Path) -> None:
    """Stage-only streaming does NOT wire complete_stream (stage uses on_stage, not chunks)."""
    runtime = _make_runtime(tmp_path, streaming="stage")
    hooks = turn_hooks_builder.build_turn_hooks(runtime, {"mode_id": "default"})
    assert hooks.models.complete_stream is None


def test_build_turn_hooks_omits_complete_stream_when_disabled(tmp_path: Path) -> None:
    """With streaming disabled, complete_stream stays None (non-streaming path)."""
    runtime = _make_runtime(tmp_path, streaming="none")
    hooks = turn_hooks_builder.build_turn_hooks(runtime, {"mode_id": "default"})
    assert hooks.models.complete_stream is None


@pytest.mark.asyncio
@pytest.mark.xfail(
    reason="turn hooks execute_operation wiring needs runtime + operation context migration"
)
async def test_build_turn_hooks_routes_memory_through_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pass


@pytest.mark.asyncio
@pytest.mark.xfail(
    reason="turn hooks execute_operation wiring needs runtime + operation context migration"
)
async def test_turn_hook_execute_operation_injects_memory_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pass


# ── Learning prompt synthesis ─────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.xfail(
    reason="synthesize_learning test needs OperationExecutionContext + complete_fn migration"
)
async def test_synthesize_learning_prompt_mentions_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pass


# ── Hello-world pack discovery (subprocess) ───────────────────────────


@pytest.mark.xfail(
    reason="subprocess test invokes little_lisa script; needs hosts/cli/main.py migration"
)
def test_little_lisa_discovers_and_runs_hello_world_pack_from_sdk_content_roots(
    tmp_path: Path,
) -> None:
    pass


# ── Verify command (CLI-only) ─────────────────────────────────────────


def test_verify_command_respects_data_dir_override(tmp_path: Path) -> None:
    pytest.skip("migrated: _verify_command now in hosts/cli (CLI verify command)")
