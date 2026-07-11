from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import httpx
import pytest
from kernel.model_manifest import ModelManifest
from kernel.operations.registry import OperationExecutionError
from kernel.turn_runner import TurnResult
from pydantic import BaseModel

from . import main as little_lisa


@pytest.fixture(scope="module", autouse=True)
def _registered_framework() -> None:
    little_lisa.setup_framework()


def test_list_modes_separates_supported_from_host_required_modes() -> None:
    output = little_lisa.format_modes()

    assert "Supported Little Lisa modes:" in output
    assert "- search (action):" in output
    assert "- thinkabout (workflow):" in output
    assert "- thinkabout: requires" not in output


def test_list_models_table_includes_designations_and_aliases() -> None:
    table = little_lisa.format_models()

    assert table.title == "Little Lisa Models"
    # Verify columns exist by checking header row text
    header_text = " ".join(str(column.header) for column in table.columns)
    assert "Model" in header_text
    assert "Provider" in header_text
    assert "Designations" in header_text
    assert "Aliases" in header_text
    assert "Reasoning" in header_text
    # At least one model row should be present (the default manifest has models)
    assert len(table.rows) >= 1


def test_build_content_mcp_config_uses_discovered_roots_and_host_data_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pack_root = tmp_path / "external-pack"
    (pack_root / "content").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(little_lisa, "_sdk_content_roots", lambda: [pack_root])
    monkeypatch.setattr(
        little_lisa,
        "_load_content_provider_preferences",
        lambda _path: {"modes": {}, "operations": {}, "tools": {}},
    )

    config = little_lisa._build_content_mcp_config()

    assert config.args is not None
    assert "--data-dir" in config.args
    data_dir_index = config.args.index("--data-dir")
    assert config.args[data_dir_index + 1] == str(little_lisa._DEFAULT_DATA_DIR)

    assert config.env is not None
    roots_value = str(config.env.get("SDK_CONTENT_ROOTS", ""))
    configured_roots = [part for part in roots_value.split(os.pathsep) if part]
    assert str(pack_root) in configured_roots
    assert str(pack_root / "content") in configured_roots
    assert not any("sdk_packs/core" in root for root in configured_roots)


def test_build_content_mcp_config_without_discovered_roots_keeps_empty_sdk_content_roots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(little_lisa, "_sdk_content_roots", lambda: [])
    monkeypatch.setattr(
        little_lisa,
        "_load_content_provider_preferences",
        lambda _path: {"modes": {}, "operations": {}, "tools": {}},
    )

    config = little_lisa._build_content_mcp_config()

    assert config.args is not None
    assert "--data-dir" in config.args
    data_dir_index = config.args.index("--data-dir")
    assert config.args[data_dir_index + 1] == str(little_lisa._DEFAULT_DATA_DIR)

    assert config.env is not None
    assert str(config.env.get("SDK_CONTENT_ROOTS", "")) == ""


def test_resolve_desktop_token_uses_generator_when_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(little_lisa, "generate_desktop_token", lambda: "generated-token")

    assert little_lisa._resolve_desktop_token("") == "generated-token"
    assert little_lisa._resolve_desktop_token("  fixed-token  ") == "fixed-token"


def test_build_desktop_subprocess_args_forwards_runtime_settings() -> None:
    args = little_lisa._build_desktop_subprocess_args(
        host="127.0.0.1",
        port=8765,
        token="desktop-token",
        debug=True,
        log_level="debug",
        log="artifacts/turns.jsonl",
        no_log=True,
        data_dir="artifacts",
        memory_db="artifacts/memory.json",
        thread_dir="artifacts/threads",
        no_thread=True,
        session_context="adam",
        timezone="UTC",
        state_snapshot="artifacts/state.json",
        streaming="hybrid:4",
        host_log="artifacts/custom-host.log",
    )

    assert args[:7] == [
        "desktop",
        "--host",
        "127.0.0.1",
        "--port",
        "8765",
        "--token",
        "desktop-token",
    ]
    assert "--no-open-browser" in args
    assert "--reload-models" not in args
    assert "--debug" in args
    assert "--no-log" in args
    assert "--session-context" in args
    assert "adam" in args
    assert "--streaming" in args
    assert "hybrid:4" in args
    assert "--host-log" in args
    assert "artifacts/custom-host.log" in args


@pytest.mark.asyncio
async def test_simple_help_reports_only_wired_capabilities() -> None:
    with TemporaryDirectory() as temp_dir:
        output = await little_lisa.handle_message("/help")

    assert "/search <request>" in output
    assert "/discuss <request>" in output
    assert "/help capabilities" in output
    assert "/markdown" not in output


@pytest.mark.asyncio
async def test_help_capabilities_returns_capability_snapshot() -> None:
    with TemporaryDirectory() as temp_dir:
        output = await little_lisa.handle_message("/help capabilities")

    assert "Capability Snapshot" in output
    assert "Supported modes" in output


@pytest.mark.asyncio
async def test_debug_context_reports_last_turn_sections() -> None:
    little_lisa._RUNTIME_STATE["last_turn_context"] = {
        "turn_input": {
            "assembled_user_payload": {
                "pre_turn_briefing": "brief",
                "memory_recall": "memory",
                "dynamic_user_context": "ctx",
                "user_query": "query",
            }
        }
    }

    with TemporaryDirectory() as temp_dir:
        output = await little_lisa.handle_message("/debug context")

    assert "Last Turn Context" in output
    assert "pre_turn_briefing" in output
    assert "brief" in output


@pytest.mark.asyncio
async def test_debug_frontdoor_reports_last_planner_payload() -> None:
    little_lisa._RUNTIME_STATE["last_frontdoor_debug"] = {
        "selected_mode_id": "thinkabout",
        "planner_reasoning": "explicit think-about wording",
        "fallback_used": False,
    }

    with TemporaryDirectory() as temp_dir:
        output = await little_lisa.handle_message("/debug frontdoor")

    assert "Front-Door Debug" in output
    assert "thinkabout" in output
    assert "planner_reasoning" in output


def test_assemble_system_prompt_includes_turn_input_contract() -> None:
    system_prompt = little_lisa._assemble_system_prompt("Mode instruction", None)

    assert "Mode instruction" in system_prompt
    assert "## User Query" in system_prompt


def test_load_frontdoor_fast_path_verifier_prompt_reads_authored_asset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    little_lisa._load_frontdoor_fast_path_verifier_prompt.cache_clear()

    def _fake_read_authored_text(relative_path: str) -> str:
        if relative_path == "frontdoor/fast_path_verifier.md":
            return "loaded fast-path verifier prompt"
        raise AssertionError(f"Unexpected authored content path: {relative_path}")

    monkeypatch.setattr(little_lisa, "_read_authored_text", _fake_read_authored_text)

    assert (
        little_lisa._load_frontdoor_fast_path_verifier_prompt()
        == "loaded fast-path verifier prompt"
    )


def test_load_frontdoor_fast_path_verifier_prompt_returns_empty_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    little_lisa._load_frontdoor_fast_path_verifier_prompt.cache_clear()

    def _missing_authored_text(relative_path: str) -> str:
        raise FileNotFoundError(relative_path)

    monkeypatch.setattr(little_lisa, "_read_authored_text", _missing_authored_text)

    assert little_lisa._load_frontdoor_fast_path_verifier_prompt() == ""


def test_assemble_dynamic_user_context_skips_user_context_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(little_lisa._RUNTIME_STATE, "user_slug", "")

    def _fake_read_authored_text(relative_path: str) -> str:
        if relative_path == "user_contexts/default.md":
            raise AssertionError("default user context should not be loaded")
        return ""

    monkeypatch.setattr(little_lisa, "_read_authored_text", _fake_read_authored_text)

    context = little_lisa._assemble_dynamic_user_context("default", "hello", "")

    assert "Selected User Context" not in context
    assert "user_slug: none" in context


@pytest.mark.asyncio
async def test_plan_frontdoor_route_uses_markdown_prompt_and_frontdoor_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    captured_aliases: list[str] = []

    def _fake_resolve_model(alias: str) -> little_lisa.StubModelDefinition:
        captured_aliases.append(alias)
        captured["model_alias"] = alias
        return little_lisa.StubModelDefinition(model_id="gpt-4o", provider="openai")

    async def _fake_complete_live(**kwargs: object) -> little_lisa.StubCompletion:
        captured.update(kwargs)
        return little_lisa.StubCompletion(
            content=(
                '{"selected_mode_id":"search","normalized_query":"cloud costs",'
                '"reasoning":"route to search","clarification_needed":false,'
                '"clarification_question":"","confidence_score":0.97}'
            ),
            tool_calls=[],
        )

    monkeypatch.setattr(little_lisa, "_resolve_model", _fake_resolve_model)
    monkeypatch.setattr(little_lisa, "_complete_live", _fake_complete_live)

    decision = await little_lisa._plan_frontdoor_route(
        raw_message="find cloud costs",
        conversation_excerpt_markdown="recent context",
    )

    assert "front-door" in captured_aliases
    assert "## Routable Modes" in str(captured["messages"][0]["content"])
    assert "## User Query" in str(captured["messages"][0]["content"])
    assert decision.selected_mode_id == "search"
    assert decision.normalized_query == "cloud costs"
    debug_payload = little_lisa._last_frontdoor_debug()
    assert isinstance(debug_payload, dict)
    assert debug_payload.get("fallback_used") is False
    assert debug_payload.get("planner_reasoning") == "route to search"


@pytest.mark.asyncio
async def test_plan_frontdoor_route_falls_back_when_completion_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _failing_complete_live(**kwargs: object) -> little_lisa.StubCompletion:
        _ = kwargs
        raise ValueError("invalid planner payload")

    monkeypatch.setattr(little_lisa, "_complete_live", _failing_complete_live)

    decision = await little_lisa._plan_frontdoor_route(
        raw_message="I'd like to go in depth about memory systems",
        conversation_excerpt_markdown="",
    )

    assert decision.selected_mode_id == "default"
    assert decision.normalized_query == "I'd like to go in depth about memory systems"
    debug_payload = little_lisa._last_frontdoor_debug()
    assert isinstance(debug_payload, dict)
    assert debug_payload.get("fallback_used") is True
    assert debug_payload.get("fallback_reason") == "planner_completion_error"


@pytest.mark.asyncio
async def test_plan_frontdoor_route_falls_back_when_completion_returns_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _invalid_json_complete_live(**kwargs: object) -> little_lisa.StubCompletion:
        _ = kwargs
        return little_lisa.StubCompletion(content='{\n  "reasoning": "The', tool_calls=[])

    monkeypatch.setattr(little_lisa, "_complete_live", _invalid_json_complete_live)

    decision = await little_lisa._plan_frontdoor_route(
        raw_message="memory architectures",
        conversation_excerpt_markdown="",
    )

    assert decision.selected_mode_id == "default"
    assert decision.normalized_query == "memory architectures"
    debug_payload = little_lisa._last_frontdoor_debug()
    assert isinstance(debug_payload, dict)
    assert debug_payload.get("fallback_used") is True
    assert debug_payload.get("fallback_reason") == "planner_output_validation_error"


@pytest.mark.asyncio
async def test_plan_frontdoor_route_retries_without_response_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_call_count = 2
    calls: list[dict[str, object]] = []

    async def _flaky_complete_live(**kwargs: object) -> little_lisa.StubCompletion:
        calls.append(dict(kwargs))
        if len(calls) == 1:
            raise RuntimeError("response_format is not supported")
        return little_lisa.StubCompletion(
            content=(
                '{"selected_mode_id":"search","normalized_query":"agent popularity",'
                '"reasoning":"route to search","clarification_needed":false,'
                '"clarification_question":"","confidence_score":0.84}'
            ),
            tool_calls=[],
        )

    monkeypatch.setattr(little_lisa, "_complete_live", _flaky_complete_live)

    decision = await little_lisa._plan_frontdoor_route(
        raw_message="what autonomous agents are popular",
        conversation_excerpt_markdown="",
    )

    assert len(calls) == expected_call_count
    assert calls[0].get("response_format") == little_lisa.FrontDoorRouteDecision
    assert calls[1].get("response_format") is None
    assert calls[0].get("max_tokens") == 512
    assert calls[1].get("max_tokens") == 512
    assert decision.selected_mode_id == "search"
    assert decision.normalized_query == "agent popularity"
    debug_payload = little_lisa._last_frontdoor_debug()
    assert isinstance(debug_payload, dict)
    assert debug_payload.get("fallback_used") is False
    assert debug_payload.get("structured_output_retry_used") is True
    assert debug_payload.get("structured_output_retry_succeeded") is True
    assert "RuntimeError" in str(debug_payload.get("planner_completion_error") or "")


@pytest.mark.asyncio
async def test_plan_frontdoor_route_retries_when_structured_output_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: structured output call returns empty content, retry recovers."""
    expected_call_count = 2
    calls: list[dict[str, object]] = []

    async def _empty_then_valid_complete_live(
        **kwargs: object,
    ) -> little_lisa.StubCompletion:
        calls.append(dict(kwargs))
        if len(calls) == 1:
            return little_lisa.StubCompletion(content="", tool_calls=[])
        return little_lisa.StubCompletion(
            content=(
                '{"selected_mode_id":"discuss","normalized_query":"lets discuss '
                'the latest suggestions","reasoning":"The user explicitly asks '
                'to discuss","clarification_needed":false,'
                '"clarification_question":"","confidence_score":0.91}'
            ),
            tool_calls=[],
        )

    monkeypatch.setattr(little_lisa, "_complete_live", _empty_then_valid_complete_live)

    decision = await little_lisa._plan_frontdoor_route(
        raw_message="lets discuss the latest suggestions",
        conversation_excerpt_markdown="",
    )

    assert len(calls) == expected_call_count
    assert calls[0].get("response_format") == little_lisa.FrontDoorRouteDecision
    assert calls[1].get("response_format") is None
    assert calls[0].get("max_tokens") == 512
    assert calls[1].get("max_tokens") == 512
    assert decision.selected_mode_id == "discuss"
    assert decision.normalized_query == "lets discuss the latest suggestions"
    debug_payload = little_lisa._last_frontdoor_debug()
    assert isinstance(debug_payload, dict)
    assert debug_payload.get("fallback_used") is False
    assert debug_payload.get("structured_output_retry_used") is True
    assert debug_payload.get("structured_output_retry_succeeded") is True
    assert "EOF while parsing" in str(debug_payload.get("planner_output_validation_error") or "")


@pytest.mark.asyncio
async def test_plan_frontdoor_route_falls_back_when_both_attempts_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both structured and retry calls return empty content — fallback to default."""
    expected_call_count = 2
    calls: list[dict[str, object]] = []

    async def _always_empty_complete_live(**kwargs: object) -> little_lisa.StubCompletion:
        calls.append(dict(kwargs))
        return little_lisa.StubCompletion(content="", tool_calls=[])

    monkeypatch.setattr(little_lisa, "_complete_live", _always_empty_complete_live)

    decision = await little_lisa._plan_frontdoor_route(
        raw_message="some query",
        conversation_excerpt_markdown="",
    )

    assert len(calls) == expected_call_count
    assert decision.selected_mode_id == "default"
    assert decision.normalized_query == "some query"
    debug_payload = little_lisa._last_frontdoor_debug()
    assert isinstance(debug_payload, dict)
    assert debug_payload.get("fallback_used") is True
    assert debug_payload.get("fallback_reason") == "planner_output_validation_error"
    assert debug_payload.get("structured_output_retry_used") is True
    assert debug_payload.get("structured_output_retry_succeeded") is False


@pytest.mark.asyncio
async def test_handle_message_uses_frontdoor_route_planner_for_natural_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route_calls: list[dict[str, object]] = []
    captured: dict[str, object] = {}

    async def _fake_plan_frontdoor_route(**kwargs: object) -> little_lisa.FrontDoorRouteDecision:
        route_calls.append(dict(kwargs))
        return little_lisa.FrontDoorRouteDecision(
            selected_mode_id="search",
            normalized_query="cloud costs",
            reasoning="route to search",
        )

    async def _fake_run_turn(**kwargs: object) -> TurnResult:
        captured.update(kwargs)
        return TurnResult(response_text="search")

    monkeypatch.setattr(little_lisa, "_plan_frontdoor_route", _fake_plan_frontdoor_route)
    monkeypatch.setattr(little_lisa, "run_turn", _fake_run_turn)

    with TemporaryDirectory() as temp_dir:
        output = await little_lisa.handle_message("find cloud costs")

    assert output == "search"
    assert route_calls and route_calls[0]["raw_message"] == "find cloud costs"
    assert captured["mode_id"] == "search"
    assert captured["preferred_model_id"] == "default"


@pytest.mark.asyncio
async def test_handle_message_updates_runtime_token_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_plan_frontdoor_route(**kwargs: object) -> little_lisa.FrontDoorRouteDecision:
        _ = kwargs
        return little_lisa.FrontDoorRouteDecision(
            selected_mode_id="default",
            normalized_query="hello",
            reasoning="default route",
        )

    async def _fake_run_turn(**kwargs: object) -> TurnResult:
        _ = kwargs
        return TurnResult(
            response_text="ok",
            prompt_tokens=120,
            completion_tokens=30,
            total_tokens=150,
        )

    monkeypatch.setattr(little_lisa, "_plan_frontdoor_route", _fake_plan_frontdoor_route)
    monkeypatch.setattr(little_lisa, "run_turn", _fake_run_turn)

    monkeypatch.setitem(little_lisa._RUNTIME_STATE, "session_prompt_tokens", 10)
    monkeypatch.setitem(little_lisa._RUNTIME_STATE, "session_completion_tokens", 5)
    monkeypatch.setitem(little_lisa._RUNTIME_STATE, "session_total_tokens", 15)
    monkeypatch.setitem(little_lisa._RUNTIME_STATE, "turn_prompt_tokens", 0)
    monkeypatch.setitem(little_lisa._RUNTIME_STATE, "turn_completion_tokens", 0)
    monkeypatch.setitem(little_lisa._RUNTIME_STATE, "turn_total_tokens", 0)

    with TemporaryDirectory() as temp_dir:
        _ = temp_dir
        output = await little_lisa.handle_message("hello")

    assert output == "ok"
    assert little_lisa._RUNTIME_STATE["turn_prompt_tokens"] == 120
    assert little_lisa._RUNTIME_STATE["turn_completion_tokens"] == 30
    assert little_lisa._RUNTIME_STATE["turn_total_tokens"] == 150
    assert little_lisa._RUNTIME_STATE["session_prompt_tokens"] == 130
    assert little_lisa._RUNTIME_STATE["session_completion_tokens"] == 35


def test_resolve_session_context_reads_display_name_from_front_matter() -> None:
    from tempfile import TemporaryDirectory  # noqa: PLC0415 — test-scoped import

    with TemporaryDirectory() as td:
        path = Path(td) / "ctx.md"
        path.write_text('---\ndisplay_name: "Adam Migus"\n---\n# hello\n', encoding="utf-8")
        name, ref, is_file = little_lisa._resolve_session_context(str(path))

    assert name == "Adam Migus"
    assert ref == str(path)
    assert is_file is True


def test_apply_turn_usage_accumulates_model_mode_stage_breakdowns() -> None:
    little_lisa._RUNTIME_STATE["session_token_ledger"] = []
    little_lisa._RUNTIME_STATE["model_token_totals"] = {}
    little_lisa._RUNTIME_STATE["mode_token_totals"] = {}
    little_lisa._RUNTIME_STATE["stage_token_totals"] = {}
    little_lisa._RUNTIME_STATE["session_prompt_tokens"] = 0
    little_lisa._RUNTIME_STATE["session_completion_tokens"] = 0
    little_lisa._RUNTIME_STATE["session_total_tokens"] = 0

    result = TurnResult(
        response_text="ok",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
    )
    little_lisa._apply_turn_usage(
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
    little_lisa._apply_turn_usage(
        result2,
        model_id="claude-haiku-4-5",
        mode_id="default",
        stage_id="default",
    )

    # Ledger
    ledger = little_lisa._RUNTIME_STATE["session_token_ledger"]
    assert len(ledger) == 2
    assert ledger[0]["model_id"] == "gpt-4o"
    assert ledger[0]["mode_id"] == "search"
    assert ledger[0]["stage_id"] == "search"
    assert ledger[1]["model_id"] == "claude-haiku-4-5"

    # Model breakdown
    model_totals = little_lisa._RUNTIME_STATE["model_token_totals"]
    assert model_totals["gpt-4o"]["total_tokens"] == 150
    assert model_totals["claude-haiku-4-5"]["total_tokens"] == 100

    # Mode breakdown
    mode_totals = little_lisa._RUNTIME_STATE["mode_token_totals"]
    assert mode_totals["search"]["total_tokens"] == 150
    assert mode_totals["default"]["total_tokens"] == 100

    # Stage breakdown
    stage_totals = little_lisa._RUNTIME_STATE["stage_token_totals"]
    assert stage_totals["search"]["total_tokens"] == 150
    assert stage_totals["default"]["total_tokens"] == 100

    # Session totals
    assert little_lisa._RUNTIME_STATE["session_total_tokens"] == 250


def test_apply_turn_usage_threads_cached_reasoning_duration_latency() -> None:
    """Sub-counters and time fields flow through to the ledger and breakdowns."""
    little_lisa._RUNTIME_STATE["session_token_ledger"] = []
    little_lisa._RUNTIME_STATE["model_token_totals"] = {}
    little_lisa._RUNTIME_STATE["mode_token_totals"] = {}
    little_lisa._RUNTIME_STATE["stage_token_totals"] = {}
    little_lisa._RUNTIME_STATE["session_cached_tokens"] = 0
    little_lisa._RUNTIME_STATE["session_reasoning_tokens"] = 0
    little_lisa._RUNTIME_STATE["session_duration_ms"] = 0.0
    little_lisa._RUNTIME_STATE["session_latency_ms"] = 0

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
    little_lisa._apply_turn_usage(
        result,
        model_id="gpt-4o",
        mode_id="search",
        stage_id="search",
    )

    # Ledger entry carries every field.
    ledger = little_lisa._RUNTIME_STATE["session_token_ledger"]
    assert len(ledger) == 1
    entry = ledger[0]
    assert entry["cached_tokens"] == 20
    assert entry["reasoning_tokens"] == 10
    assert entry["duration_ms"] == 750.0
    assert entry["latency_ms"] == 800

    # Per-model breakdown carries every field.
    model_totals = little_lisa._RUNTIME_STATE["model_token_totals"]
    assert model_totals["gpt-4o"]["cached_tokens"] == 20
    assert model_totals["gpt-4o"]["reasoning_tokens"] == 10
    assert model_totals["gpt-4o"]["duration_ms"] == 750.0
    assert model_totals["gpt-4o"]["latency_ms"] == 800

    # Session-level sub-counter and time totals accumulate.
    assert little_lisa._RUNTIME_STATE["session_cached_tokens"] == 20
    assert little_lisa._RUNTIME_STATE["session_reasoning_tokens"] == 10
    assert little_lisa._RUNTIME_STATE["session_duration_ms"] == 750.0
    assert little_lisa._RUNTIME_STATE["session_latency_ms"] == 800


def test_apply_turn_usage_upgrades_old_schema_entries_in_place() -> None:
    """Pre-existing breakdown dicts with the old 3-key schema get upgraded."""
    little_lisa._RUNTIME_STATE["session_token_ledger"] = []
    # Simulate a stale entry from a prior session that only had 3 keys.
    little_lisa._RUNTIME_STATE["model_token_totals"] = {
        "gpt-4o": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
    }
    little_lisa._RUNTIME_STATE["mode_token_totals"] = {}
    little_lisa._RUNTIME_STATE["stage_token_totals"] = {}

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
    # Must not raise KeyError on the stale entry.
    little_lisa._apply_turn_usage(
        result,
        model_id="gpt-4o",
        mode_id="default",
        stage_id="default",
    )

    model_entry = little_lisa._RUNTIME_STATE["model_token_totals"]["gpt-4o"]
    assert model_entry["prompt_tokens"] == 150
    assert model_entry["completion_tokens"] == 60
    assert model_entry["total_tokens"] == 210
    assert model_entry["cached_tokens"] == 5
    assert model_entry["reasoning_tokens"] == 3
    assert model_entry["duration_ms"] == 200.0
    assert model_entry["latency_ms"] == 210


@pytest.mark.asyncio
async def test_tokens_command_returns_detailed_stats() -> None:
    little_lisa._RUNTIME_STATE["session_turn_count"] = 2
    little_lisa._RUNTIME_STATE["session_success_count"] = 2
    little_lisa._RUNTIME_STATE["session_error_count"] = 0
    little_lisa._RUNTIME_STATE["session_total_latency_ms"] = 500.0
    little_lisa._RUNTIME_STATE["session_prompt_tokens"] = 180
    little_lisa._RUNTIME_STATE["session_completion_tokens"] = 70
    little_lisa._RUNTIME_STATE["session_total_tokens"] = 250
    little_lisa._RUNTIME_STATE["turn_prompt_tokens"] = 80
    little_lisa._RUNTIME_STATE["turn_completion_tokens"] = 20
    little_lisa._RUNTIME_STATE["turn_total_tokens"] = 100
    little_lisa._RUNTIME_STATE["model_token_totals"] = {
        "gpt-4o": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        "claude-haiku-4-5": {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100},
    }
    little_lisa._RUNTIME_STATE["mode_token_totals"] = {
        "search": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        "default": {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100},
    }
    little_lisa._RUNTIME_STATE["stage_token_totals"] = {}
    little_lisa._RUNTIME_STATE["session_mode_counts"] = {"search": 1, "default": 1}

    with TemporaryDirectory() as temp_dir:
        _ = temp_dir
        output = await little_lisa.handle_message("/tokens")

    assert "### Token Totals" in output
    assert "| Session | 180 | 70 | 250 |" in output
    assert "### By Model" in output
    assert "| gpt-4o | 100 | 50 | 150 |" in output
    assert "| claude-haiku-4-5 | 80 | 20 | 100 |" in output
    assert "### By Mode" in output
    assert "| search | 100 | 50 | 150 |" in output


@pytest.mark.asyncio
async def test_handle_message_surfaces_frontdoor_and_mode_token_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_plan_frontdoor_route(**kwargs: object) -> little_lisa.FrontDoorRouteDecision:
        _ = kwargs
        return little_lisa.FrontDoorRouteDecision(
            selected_mode_id="default",
            normalized_query="hello",
            reasoning="default route",
        )

    async def _fake_run_turn(**kwargs: object) -> TurnResult:
        _ = kwargs
        return TurnResult(
            response_text="ok",
            tool_call_count=1,
            iteration_count=2,
            prompt_tokens=120,
            completion_tokens=30,
            total_tokens=150,
        )

    monkeypatch.setattr(little_lisa, "_plan_frontdoor_route", _fake_plan_frontdoor_route)
    monkeypatch.setattr(little_lisa, "run_turn", _fake_run_turn)
    monkeypatch.setitem(
        little_lisa._RUNTIME_STATE,
        "last_frontdoor_debug",
        {
            "resolved_model": {
                "model_id": "claude-haiku-4-5",
                "provider": "anthropic",
                "reasoning_model": False,
            },
            "planner_prompt_tokens": 22,
            "planner_completion_tokens": 11,
            "planner_total_tokens": 33,
        },
    )

    with TemporaryDirectory() as temp_dir:
        output = await little_lisa.handle_message("hello")

    assert output == "ok"
    context = little_lisa._last_turn_context()
    assert isinstance(context, dict)
    frontdoor_route = context.get("frontdoor_route")
    assert isinstance(frontdoor_route, dict)
    assert frontdoor_route["resolved_model_id"] == "claude-haiku-4-5"
    assert frontdoor_route["resolved_model_provider"] == "anthropic"
    assert frontdoor_route["prompt_tokens"] == 22
    assert frontdoor_route["completion_tokens"] == 11
    assert frontdoor_route["total_tokens"] == 33

    completion = context.get("completion")
    assert isinstance(completion, dict)
    assert completion["tool_call_count"] == 1
    assert completion["iteration_count"] == 2
    assert completion["prompt_tokens"] == 120
    assert completion["completion_tokens"] == 30
    assert completion["total_tokens"] == 150


def test_resolve_model_supports_designations_and_concrete_ids() -> None:
    front_door_model_id = little_lisa.resolve_little_lisa_model("front-door").model_id
    default_model_id = little_lisa.resolve_little_lisa_model("default").model_id
    deep_model_id = little_lisa.resolve_little_lisa_model("deep").model_id

    assert front_door_model_id
    assert default_model_id
    assert deep_model_id
    resolved_front_door = little_lisa.resolve_little_lisa_model(front_door_model_id).model_id
    assert resolved_front_door == front_door_model_id


def test_resolve_model_rejects_unknown_manifest_id() -> None:
    with pytest.raises(ValueError, match="not in the Little Lisa manifest"):
        little_lisa.resolve_little_lisa_model("not-a-real-model")


def test_normalize_ollama_host_to_base_url_accepts_host_without_scheme() -> None:
    normalized = little_lisa._normalize_ollama_host_to_base_url("127.0.0.1:1143")
    assert normalized == "http://127.0.0.1:1143/v1/"


def test_verify_check_provider_keys_reports_google_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITTLE_LISA_GOOGLE_API_KEY", "AIza-test-key")

    status, detail = little_lisa._verify_check_provider_keys()

    assert status == little_lisa._VERIFY_PASS
    assert "Google Gemini:" in detail


def test_simple_help_mentions_quit_command() -> None:
    output = little_lisa._format_help()
    assert "/quit exits the interactive console loop." in output


def test_build_turn_context_marks_frontdoor_handoff() -> None:
    mode = little_lisa.WorkflowModeConfig(
        mode_id="default",
        start_stage_id="default",
        mode_kind="chat",
    )
    context = little_lisa._build_turn_context(
        raw_message="hello there",
        mode_id="default",
        query="hello there",
        deep_route=False,
        preferred_model_id="front-door",
        resolved_model=little_lisa.StubModelDefinition(
            model_id="gpt-4o",
            provider="openai",
        ),
        mode=mode,
    )

    assert context["preferred_model_id"] == "front-door"
    assert "front-door" in str(context["handoff"])


def test_append_turn_log_writes_jsonl(tmp_path: Path) -> None:
    log_path = tmp_path / "little_lisa.log.jsonl"
    little_lisa._RUNTIME_STATE["log_path"] = log_path
    little_lisa._RUNTIME_STATE["log_level"] = "info"
    little_lisa._RUNTIME_STATE["debug"] = False
    context = {
        "timestamp_utc": "2026-05-30T00:00:00+00:00",
        "raw_message": "hello",
        "handoff": "natural-language intent parsed by front-door, executing on default model",
        "mode_id": "default",
        "mode_kind": "chat",
        "query": "hello",
    }

    little_lisa._append_turn_log(context=context, response_text="world")

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["event"] == "turn.completed"
    assert payload["level"] == "info"
    assert payload["type"] == "little_lisa_turn"
    assert payload["context"]["handoff"].startswith("natural-language")
    assert payload["response"]["text"] == "world"
    assert payload["route"]["mode_id"] == "default"
    assert payload["mode_model"] == ""
    assert payload["frontdoor_model"] == ""
    assert payload["mode_total_tokens"] == 0
    assert payload["frontdoor_total_tokens"] == 0
    assert payload["completion_summary"]["tokens"] == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    assert payload["runtime"]["log_level"] == "info"


def test_append_turn_log_flattens_model_and_token_fields(tmp_path: Path) -> None:
    log_path = tmp_path / "little_lisa.log.jsonl"
    little_lisa._RUNTIME_STATE["log_path"] = log_path
    little_lisa._RUNTIME_STATE["log_level"] = "info"
    little_lisa._RUNTIME_STATE["debug"] = False
    context = {
        "timestamp_utc": "2026-05-30T00:00:00+00:00",
        "raw_message": "summarize our chat",
        "mode_id": "default",
        "mode_kind": "chat",
        "query": "summarize our chat",
        "resolved_model": {
            "model_id": "claude-haiku-4-5",
            "provider": "anthropic",
            "reasoning_model": False,
        },
        "frontdoor_debug": {
            "resolved_model": {
                "model_id": "claude-haiku-4-5",
                "provider": "anthropic",
                "reasoning_model": False,
            },
            "planner_prompt_tokens": 22,
            "planner_completion_tokens": 11,
            "planner_total_tokens": 33,
            "planner_completion_attempts": [{"attempt": "primary_structured"}],
        },
        "completion": {
            "tool_call_count": 0,
            "iteration_count": 1,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        },
    }

    little_lisa._append_turn_log(context=context, response_text="done")

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["mode_model"] == "anthropic:claude-haiku-4-5"
    assert payload["frontdoor_model"] == "anthropic:claude-haiku-4-5"
    assert payload["mode_prompt_tokens"] == 100
    assert payload["mode_completion_tokens"] == 50
    assert payload["mode_total_tokens"] == 150
    assert payload["frontdoor_prompt_tokens"] == 22
    assert payload["frontdoor_completion_tokens"] == 11
    assert payload["frontdoor_total_tokens"] == 33
    assert payload["frontdoor_completion_attempt_count"] == 1


def test_append_log_event_respects_runtime_log_threshold(tmp_path: Path) -> None:
    log_path = tmp_path / "little_lisa.log.jsonl"
    little_lisa._RUNTIME_STATE["log_path"] = log_path
    little_lisa._RUNTIME_STATE["log_level"] = "error"

    little_lisa._append_log_event(
        level="debug",
        event="debug.should_be_filtered",
        payload={"value": 1},
    )
    assert not log_path.exists()

    little_lisa._append_log_event(
        level="error",
        event="error.should_be_written",
        payload={"value": 2},
    )

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["event"] == "error.should_be_written"
    assert payload["level"] == "error"


def test_resolve_log_level_prefers_cli_env_then_debug_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LITTLE_LISA_LOG_LEVEL", raising=False)
    assert little_lisa._resolve_log_level("warning", debug_enabled=False) == "warning"
    assert little_lisa._resolve_log_level("", debug_enabled=True) == "debug"
    assert little_lisa._resolve_log_level("", debug_enabled=False) == "info"

    monkeypatch.setenv("LITTLE_LISA_LOG_LEVEL", "error")
    assert little_lisa._resolve_log_level("", debug_enabled=True) == "error"


@pytest.mark.asyncio
async def test_complete_live_routes_azure_openai_with_completion_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeMessage:
        def __init__(self) -> None:
            self.content = "azure"
            self.tool_calls = []

    class _FakeChoice:
        def __init__(self) -> None:
            self.message = _FakeMessage()

    class _FakeResponse:
        def __init__(self) -> None:
            self.choices = [_FakeChoice()]

    class _FakeCompletions:
        async def create(self, **kwargs: object) -> _FakeResponse:
            captured.update(kwargs)
            return _FakeResponse()

    class _FakeChat:
        def __init__(self) -> None:
            self.completions = _FakeCompletions()

    class _FakeClient:
        def __init__(self) -> None:
            self.chat = _FakeChat()

    def _fake_azure_client_instance() -> _FakeClient:
        return _FakeClient()

    monkeypatch.setattr(little_lisa, "_azure_client_instance", _fake_azure_client_instance)

    response = await little_lisa._complete_live(
        messages=[{"role": "user", "content": "hello"}],
        model_def=little_lisa.StubModelDefinition(
            model_id="gpt-chat-latest",
            provider="azure_openai",
        ),
        max_tokens=256,
        temperature=0.2,
        tools=None,
        response_format=None,
    )

    assert response.content == "azure"
    assert captured["model"] == "gpt-chat-latest"
    assert captured["max_completion_tokens"] == 256
    assert "max_tokens" not in captured


@pytest.mark.asyncio
async def test_complete_live_routes_anthropic_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def _fake_complete_anthropic(**kwargs: object) -> little_lisa.StubCompletion:
        captured.update(kwargs)
        return little_lisa.StubCompletion(content="anthropic", tool_calls=[])

    monkeypatch.setattr(little_lisa, "_complete_anthropic", _fake_complete_anthropic)

    response = await little_lisa._complete_live(
        messages=[{"role": "user", "content": "hello"}],
        model_def=little_lisa.StubModelDefinition(
            model_id="claude-3-5-sonnet",
            provider="anthropic",
        ),
        max_tokens=256,
        temperature=0.2,
        tools=None,
        response_format=None,
    )

    assert response.content == "anthropic"
    assert captured["model_def"] == little_lisa.StubModelDefinition(
        model_id="claude-3-5-sonnet",
        provider="anthropic",
        reasoning_model=False,
    )


@pytest.mark.asyncio
async def test_complete_anthropic_surfaces_model_not_found_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeAnthropicClient:
        async def post(
            self,
            path: str,
            *,
            headers: dict[str, str],
            json: dict[str, object],
        ) -> httpx.Response:
            _ = headers
            _ = json
            request = httpx.Request("POST", f"https://api.anthropic.com{path}")
            return httpx.Response(
                status_code=404,
                request=request,
                json={
                    "type": "error",
                    "error": {
                        "type": "not_found_error",
                        "message": "model: claude-haiku-4.5-20240930",
                    },
                },
            )

    monkeypatch.setenv("LITTLE_LISA_ANTHROPIC_API_KEY", "anthropic-test-key")

    def _fake_anthropic_http_client_instance() -> _FakeAnthropicClient:
        return _FakeAnthropicClient()

    monkeypatch.setattr(
        little_lisa,
        "_anthropic_http_client_instance",
        _fake_anthropic_http_client_instance,
    )

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await little_lisa._complete_anthropic(
            messages=[{"role": "user", "content": "hello"}],
            model_def=little_lisa.StubModelDefinition(
                model_id="claude-haiku-4.5-20240930",
                provider="anthropic",
            ),
            max_tokens=128,
            temperature=0.0,
            tools=None,
            response_format=None,
        )

    error_message = str(exc_info.value)
    assert "not_found_error" in error_message
    assert "claude-haiku-4.5-20240930" in error_message
    assert "Check the model_id in models.yaml" in error_message


@pytest.mark.asyncio
async def test_complete_live_routes_ollama_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeMessage:
        def __init__(self) -> None:
            self.content = "ollama"
            self.tool_calls = []

    class _FakeChoice:
        def __init__(self) -> None:
            self.message = _FakeMessage()

    class _FakeResponse:
        def __init__(self) -> None:
            self.choices = [_FakeChoice()]

    class _FakeCompletions:
        async def create(self, **kwargs: object) -> _FakeResponse:
            captured.update(kwargs)
            return _FakeResponse()

    class _FakeChat:
        def __init__(self) -> None:
            self.completions = _FakeCompletions()

    class _FakeClient:
        def __init__(self) -> None:
            self.chat = _FakeChat()

    def _fake_ollama_client_instance() -> _FakeClient:
        return _FakeClient()

    monkeypatch.setattr(little_lisa, "_ollama_client_instance", _fake_ollama_client_instance)

    response = await little_lisa._complete_live(
        messages=[{"role": "user", "content": "hello"}],
        model_def=little_lisa.StubModelDefinition(
            model_id="llama3.1",
            provider="ollama",
        ),
        max_tokens=256,
        temperature=0.2,
        tools=None,
        response_format=None,
    )

    assert response.content == "ollama"
    assert captured["model"] == "llama3.1"


@pytest.mark.asyncio
async def test_complete_live_routes_google_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeMessage:
        def __init__(self) -> None:
            self.content = "google"
            self.tool_calls = []

    class _FakeChoice:
        def __init__(self) -> None:
            self.message = _FakeMessage()

    class _FakeResponse:
        def __init__(self) -> None:
            self.choices = [_FakeChoice()]

    class _FakeCompletions:
        async def create(self, **kwargs: object) -> _FakeResponse:
            captured.update(kwargs)
            return _FakeResponse()

    class _FakeChat:
        def __init__(self) -> None:
            self.completions = _FakeCompletions()

    class _FakeClient:
        def __init__(self) -> None:
            self.chat = _FakeChat()

    def _fake_google_client_instance() -> _FakeClient:
        return _FakeClient()

    monkeypatch.setattr(little_lisa, "_google_client_instance", _fake_google_client_instance)

    response = await little_lisa._complete_live(
        messages=[{"role": "user", "content": "hello"}],
        model_def=little_lisa.StubModelDefinition(
            model_id="gemini-2.5-flash",
            provider="google",
        ),
        max_tokens=256,
        temperature=0.2,
        tools=None,
        response_format=None,
    )

    assert response.content == "google"
    assert captured["model"] == "gemini-2.5-flash"


@pytest.mark.asyncio
async def test_complete_live_normalizes_fenced_json_structured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StructuredResponse(BaseModel):
        output_text: str
        source_urls: list[str] = []

    class _FakeMessage:
        def __init__(self) -> None:
            self.content = (
                "```json\n"
                "{\n"
                '  "output_text": "Fresh grounded summary",\n'
                '  "source_urls": ["https://example.com/hermes"]\n'
                "}\n"
                "```"
            )
            self.tool_calls: list[dict[str, object]] = []

    class _FakeChoice:
        def __init__(self) -> None:
            self.message = _FakeMessage()

    class _FakeResponse:
        def __init__(self) -> None:
            self.choices = [_FakeChoice()]

    class _FakeCompletions:
        async def create(self, **kwargs: object) -> _FakeResponse:
            _ = kwargs
            return _FakeResponse()

    class _FakeChat:
        def __init__(self) -> None:
            self.completions = _FakeCompletions()

    class _FakeClient:
        def __init__(self) -> None:
            self.chat = _FakeChat()

    def _fake_ollama_client_instance() -> _FakeClient:
        return _FakeClient()

    monkeypatch.setattr(little_lisa, "_ollama_client_instance", _fake_ollama_client_instance)

    response = await little_lisa._complete_live(
        messages=[{"role": "user", "content": "hello"}],
        model_def=little_lisa.StubModelDefinition(
            model_id="llama3.1",
            provider="ollama",
        ),
        max_tokens=256,
        temperature=0.2,
        tools=None,
        response_format=_StructuredResponse,
    )

    parsed = _StructuredResponse.model_validate_json(response.content)
    assert parsed.output_text == "Fresh grounded summary"
    assert parsed.source_urls == ["https://example.com/hermes"]


@pytest.mark.asyncio
async def test_complete_live_preserves_non_json_structured_output_for_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StructuredResponse(BaseModel):
        facts: list[str]

    class _FakeMessage:
        def __init__(self) -> None:
            self.content = (
                "- The bot memory architecture should keep durable strategic facts only.\n"
                "- Keep memory concise and avoid transcript storage.\n"
                "- Persist only missing components."
            )
            self.tool_calls: list[dict[str, object]] = []

    class _FakeChoice:
        def __init__(self) -> None:
            self.message = _FakeMessage()

    class _FakeResponse:
        def __init__(self) -> None:
            self.choices = [_FakeChoice()]

    class _FakeCompletions:
        async def create(self, **kwargs: object) -> _FakeResponse:
            _ = kwargs
            return _FakeResponse()

    class _FakeChat:
        def __init__(self) -> None:
            self.completions = _FakeCompletions()

    class _FakeClient:
        def __init__(self) -> None:
            self.chat = _FakeChat()

    def _fake_ollama_client_instance() -> _FakeClient:
        return _FakeClient()

    monkeypatch.setattr(little_lisa, "_ollama_client_instance", _fake_ollama_client_instance)

    response = await little_lisa._complete_live(
        messages=[{"role": "user", "content": "summarize memory"}],
        model_def=little_lisa.StubModelDefinition(
            model_id="llama3.1",
            provider="ollama",
        ),
        max_tokens=256,
        temperature=0.2,
        tools=None,
        response_format=_StructuredResponse,
    )

    assert response.content.startswith("- The bot memory architecture")


@pytest.mark.asyncio
async def test_handle_message_uses_explicit_frontdoor_model_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def _fake_plan_frontdoor_route(**kwargs: object) -> little_lisa.FrontDoorRouteDecision:
        _ = kwargs
        return little_lisa.FrontDoorRouteDecision(
            selected_mode_id="default",
            normalized_query="hello there",
            reasoning="stay on default",
        )

    async def _fake_run_turn(**kwargs: object) -> TurnResult:
        captured.update(kwargs)
        return TurnResult(response_text="frontdoor")

    monkeypatch.setattr(little_lisa, "_plan_frontdoor_route", _fake_plan_frontdoor_route)
    monkeypatch.setattr(little_lisa, "run_turn", _fake_run_turn)

    with TemporaryDirectory() as temp_dir:
        output = await little_lisa.handle_message("hello there")

    assert output == "frontdoor"
    assert captured["preferred_model_id"] == "default"


@pytest.mark.asyncio
async def test_handle_message_does_not_force_frontdoor_model_for_explicit_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def _fake_run_turn(**kwargs: object) -> TurnResult:
        captured.update(kwargs)
        return TurnResult(response_text="search")

    monkeypatch.setattr(little_lisa, "run_turn", _fake_run_turn)

    with TemporaryDirectory() as temp_dir:
        output = await little_lisa.handle_message("/search cloud cost")

    assert output == "search"
    assert captured["preferred_model_id"] == "default"


@pytest.mark.asyncio
async def test_handle_message_deep_route_uses_deep_designation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def _fake_run_turn(**kwargs: object) -> TurnResult:
        captured.update(kwargs)
        return TurnResult(response_text="deep")

    monkeypatch.setattr(little_lisa, "run_turn", _fake_run_turn)

    with TemporaryDirectory() as temp_dir:
        output = await little_lisa.handle_message("/deep evaluate this")

    assert output == "deep"
    assert captured["mode_id"] == "default"
    assert captured["preferred_model_id"] == "deep"
    assert captured["query"] == "evaluate this"


@pytest.mark.asyncio
async def test_handle_message_returns_operation_error_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _failing_run_turn(**kwargs: object) -> TurnResult:
        _ = kwargs
        raise OperationExecutionError("Operation 'web_search' timed out")

    monkeypatch.setattr(little_lisa, "run_turn", _failing_run_turn)

    with TemporaryDirectory() as temp_dir:
        output = await little_lisa.handle_message("/search cloud cost")

    assert "Operation 'web_search' timed out" in output
    assert "Little Lisa kept running" in output


def test_initialize_runtime_applies_user_timeout_and_state_snapshot(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "state.json"
    context_path = tmp_path / "system.md"
    snapshot_path.write_text(
        '{"compiled_at":"2026-05-30T00:00:00+00:00","health":"ok"}',
        encoding="utf-8",
    )
    context_path.write_text("# System Context", encoding="utf-8")

    little_lisa._initialize_runtime(
        debug=True,
        user_slug=str(context_path),
        timezone="UTC",
        state_snapshot_path=str(snapshot_path),
    )

    assert little_lisa._RUNTIME_STATE["user_slug"] == "system"
    assert little_lisa._RUNTIME_STATE["session_context_ref"] == str(context_path)
    assert little_lisa._RUNTIME_STATE["session_context_active"] is True
    assert little_lisa._RUNTIME_STATE["timezone"] == "UTC"
    assert little_lisa._RUNTIME_STATE["timezone_source"] == "cli"
    assert little_lisa._RUNTIME_STATE["log_level"] == "debug"
    assert isinstance(little_lisa._RUNTIME_STATE["state_snapshot"], dict)
    assert little_lisa._RUNTIME_STATE["thread_log_path"] == (
        little_lisa._runtime_data_dir() / "threads" / "system.jsonl"
    )


def test_resolve_stream_enabled_prefers_explicit_flags() -> None:
    assert little_lisa.resolve_stream_enabled(
        force_stream=True,
        force_no_stream=False,
        is_tty=False,
    )
    assert not little_lisa.resolve_stream_enabled(
        force_stream=False,
        force_no_stream=True,
        is_tty=True,
    )


def test_resolve_stream_enabled_defaults_to_tty_state() -> None:
    assert little_lisa.resolve_stream_enabled(
        force_stream=False,
        force_no_stream=False,
        is_tty=True,
    )
    assert not little_lisa.resolve_stream_enabled(
        force_stream=False,
        force_no_stream=False,
        is_tty=False,
    )


def test_initialize_runtime_applies_stream_configuration() -> None:
    stream_delay_ms = 12

    little_lisa._initialize_runtime(
        debug=False,
        user_slug="",
        timezone="UTC",
        state_snapshot_path="",
        streaming=f"stage:{stream_delay_ms}",
    )

    assert little_lisa._RUNTIME_STATE["stream_enabled"] is True
    assert little_lisa._RUNTIME_STATE["stream_mode"] == "stage"
    assert little_lisa._RUNTIME_STATE["stream_char_delay_ms"] == stream_delay_ms


def test_initialize_runtime_uses_log_level_env_when_cli_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITTLE_LISA_LOG_LEVEL", "warning")

    little_lisa._initialize_runtime(
        debug=False,
        user_slug="",
        timezone="UTC",
        state_snapshot_path="",
    )

    assert little_lisa._RUNTIME_STATE["log_level"] == "warning"


def test_initialize_runtime_prefers_cli_log_level_over_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITTLE_LISA_LOG_LEVEL", "error")

    little_lisa._initialize_runtime(
        debug=False,
        log_level="info",
        user_slug="",
        timezone="UTC",
        state_snapshot_path="",
    )

    assert little_lisa._RUNTIME_STATE["log_level"] == "info"


def test_initialize_runtime_uses_timezone_env_when_cli_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITTLE_LISA_TIMEZONE", "UTC")

    little_lisa._initialize_runtime(
        debug=False,
        user_slug="",
        timezone="",
        state_snapshot_path="",
    )

    assert little_lisa._RUNTIME_STATE["timezone"] == "UTC"
    assert little_lisa._RUNTIME_STATE["timezone_source"] == "env:LITTLE_LISA_TIMEZONE"


def test_initialize_runtime_fails_when_timezone_unresolvable() -> None:
    with pytest.raises(ValueError, match="not resolvable"):
        little_lisa._initialize_runtime(
            debug=False,
            user_slug="",
            timezone="Not/AZone Value",
            state_snapshot_path="",
        )


def test_initialize_runtime_accepts_timezone_alias() -> None:
    little_lisa._initialize_runtime(
        debug=False,
        user_slug="",
        timezone="EDT",
        state_snapshot_path="",
    )

    assert little_lisa._RUNTIME_STATE["timezone"] == "EST5EDT"
    assert little_lisa._RUNTIME_STATE["timezone_source"] == "cli"


def test_initialize_runtime_fails_when_user_context_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITTLE_LISA_DATA_DIR", str(tmp_path))

    with pytest.raises(ValueError, match="Session context 'adam' was not found"):
        little_lisa._initialize_runtime(
            debug=False,
            user_slug="adam",
            timezone="UTC",
            state_snapshot_path="",
        )


def test_initialize_runtime_accepts_session_context_file_path(tmp_path: Path) -> None:
    context_file = tmp_path / "founder.md"
    context_file.write_text("# Founder Profile", encoding="utf-8")

    little_lisa._initialize_runtime(
        debug=False,
        user_slug=str(context_file),
        timezone="UTC",
        state_snapshot_path="",
    )

    assert little_lisa._RUNTIME_STATE["user_slug"] == "founder"
    assert little_lisa._RUNTIME_STATE["session_context_ref"] == str(context_file)
    assert little_lisa._RUNTIME_STATE["session_context_active"] is True


def test_initialize_runtime_rejects_session_context_path_without_md(tmp_path: Path) -> None:
    context_file = tmp_path / "founder.txt"
    context_file.write_text("not markdown", encoding="utf-8")

    with pytest.raises(ValueError, match=r"must end with '\.md'"):
        little_lisa._initialize_runtime(
            debug=False,
            user_slug=str(context_file),
            timezone="UTC",
            state_snapshot_path="",
        )


def test_status_time_badge_abbreviates_iana_timezone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(little_lisa._RUNTIME_STATE, "timezone", "America/New_York")

    label, _ = little_lisa._status_time_badge()

    assert "America/New_York" not in label
    assert "🕒" in label


def test_status_time_badge_compacts_long_timezone_name(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeDateTime:
        @staticmethod
        def now(tz: object | None = None) -> _FakeStatusDateTime:
            _ = tz
            return _FakeStatusDateTime()

    class _FakeStatusDateTime:
        def tzname(self) -> str:
            return "Eastern Daylight Time"

        def dst(self) -> object:
            return object()

        def strftime(self, _fmt: str) -> str:
            return "13:24:16"

    monkeypatch.setitem(little_lisa._RUNTIME_STATE, "timezone", "America/New_York")
    monkeypatch.setattr(little_lisa, "datetime", _FakeDateTime)

    label, _ = little_lisa._status_time_badge()

    assert "EDT" in label
    assert "EASTERN DAYLIGHT TIME" not in label


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

    for env_name in ("LITTLE_LISA_TIMEZONE", "TZ", "TIMEZONE", "LOCAL_TIMEZONE"):
        monkeypatch.delenv(env_name, raising=False)

    monkeypatch.setattr(little_lisa, "datetime", _FakeDateTime)

    tz, source = little_lisa._resolve_timezone("")

    assert tz == "America/New_York"
    assert source == "system-local"


def test_status_segments_show_profile_model_without_prefix_and_token_badge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(little_lisa._RUNTIME_STATE, "user_slug", "")
    monkeypatch.setitem(little_lisa._RUNTIME_STATE, "debug", False)
    monkeypatch.setitem(little_lisa._RUNTIME_STATE, "current_mode_id", "default")
    monkeypatch.setitem(little_lisa._RUNTIME_STATE, "current_stage_id", "-")
    monkeypatch.setitem(little_lisa._RUNTIME_STATE, "next_model_provider", "ollama")
    monkeypatch.setitem(little_lisa._RUNTIME_STATE, "next_model_id", "deepseek-v4")
    monkeypatch.setitem(little_lisa._RUNTIME_STATE, "turn_total_tokens", 128)
    monkeypatch.setitem(little_lisa._RUNTIME_STATE, "session_total_tokens", 1024)

    balanced_labels = [label for label, _ in little_lisa._status_segments()]
    assert any(label == "ollama:deepseek-v4" for label in balanced_labels)
    assert not any(
        label.startswith("model ") or label.startswith("m ") for label in balanced_labels
    )
    assert any(label == "tokens 128/1024" for label in balanced_labels)


def test_status_segments_minimal_uses_tc_token_badge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(little_lisa._RUNTIME_STATE, "debug", False)
    monkeypatch.setitem(little_lisa._RUNTIME_STATE, "current_mode_id", "default")
    monkeypatch.setitem(little_lisa._RUNTIME_STATE, "current_stage_id", "-")
    monkeypatch.setitem(little_lisa._RUNTIME_STATE, "next_model_provider", "ollama")
    monkeypatch.setitem(little_lisa._RUNTIME_STATE, "next_model_id", "deepseek-v4")
    monkeypatch.setitem(little_lisa._RUNTIME_STATE, "turn_total_tokens", 32)
    monkeypatch.setitem(little_lisa._RUNTIME_STATE, "session_total_tokens", 640)

    labels = [
        label for label, _ in little_lisa._status_segments(little_lisa._STATUS_COMPACT_MINIMAL)
    ]

    assert any(label == "tc 32/640" for label in labels)


def test_format_session_exit_summary_lines_includes_useful_stats() -> None:
    little_lisa._RUNTIME_STATE["session_started_utc"] = little_lisa.datetime.now(little_lisa.UTC)
    little_lisa._RUNTIME_STATE["session_turn_count"] = 3
    little_lisa._RUNTIME_STATE["session_success_count"] = 2
    little_lisa._RUNTIME_STATE["session_error_count"] = 1
    little_lisa._RUNTIME_STATE["session_total_latency_ms"] = 900.0
    little_lisa._RUNTIME_STATE["session_mode_counts"] = {"search": 2, "default": 1}
    little_lisa._RUNTIME_STATE["next_model_provider"] = "ollama"
    little_lisa._RUNTIME_STATE["next_model_id"] = "deepseek-v4"
    little_lisa._RUNTIME_STATE["turn_prompt_tokens"] = 140
    little_lisa._RUNTIME_STATE["turn_completion_tokens"] = 60
    little_lisa._RUNTIME_STATE["turn_total_tokens"] = 200
    little_lisa._RUNTIME_STATE["session_prompt_tokens"] = 420
    little_lisa._RUNTIME_STATE["session_completion_tokens"] = 180
    little_lisa._RUNTIME_STATE["session_total_tokens"] = 600
    little_lisa._RUNTIME_STATE["model_token_totals"] = {
        "deepseek-v4": {"prompt_tokens": 300, "completion_tokens": 120, "total_tokens": 420},
        "claude-haiku-4-5": {"prompt_tokens": 120, "completion_tokens": 60, "total_tokens": 180},
    }
    little_lisa._RUNTIME_STATE["mode_token_totals"] = {
        "default": {"prompt_tokens": 280, "completion_tokens": 120, "total_tokens": 400},
        "search": {"prompt_tokens": 140, "completion_tokens": 60, "total_tokens": 200},
    }
    little_lisa._RUNTIME_STATE["stage_token_totals"] = {}

    lines = little_lisa._format_session_exit_summary_lines()
    summary = "\n".join(lines)

    assert "Turns: 3 (ok: 2, errors: 1)" in summary
    assert "Avg turn latency: 300 ms" in summary
    assert "### Token Totals" in summary
    assert "| Session | 420 | 180 | 600 |" in summary
    assert "| Last turn | 140 | 60 | 200 |" in summary
    assert "| Avg/turn | 140 | 60 | 200 |" in summary
    assert "### By Model" in summary
    assert "| claude-haiku-4-5 | 120 | 60 | 180 |" in summary
    assert "| deepseek-v4 | 300 | 120 | 420 |" in summary
    assert "### By Mode" in summary
    assert "| default | 280 | 120 | 400 |" in summary
    assert "| search | 140 | 60 | 200 |" in summary
    assert "Most used mode: search (2)" in summary


@pytest.mark.asyncio
async def test_settings_command_reports_effective_runtime_settings() -> None:
    with TemporaryDirectory() as temp_dir:
        output = await little_lisa.handle_message("/settings")

    assert "Runtime Settings" in output
    assert "timezone:" in output
    assert "log_level:" in output


@pytest.mark.asyncio
async def test_handle_message_requires_models_before_running_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty_manifest = ModelManifest(models=(), mode_models={})
    monkeypatch.setattr(little_lisa, "_load_manifest", lambda: empty_manifest)

    with TemporaryDirectory() as temp_dir:
        with pytest.raises(ValueError, match="requires at least one model"):
            await little_lisa.handle_message("hello")


@pytest.mark.asyncio
async def test_build_turn_hooks_wires_recall_hook() -> None:
    """Turn hooks build cleanly with learning-owned memory behavior.

    The little-lisa host still provides MCP-backed memory aliases, while the
    learning pack owns the explicit `recall_memory` operation behavior.
    """
    little_lisa._build_turn_hooks({"mode_id": "default"})


@pytest.mark.asyncio
async def test_build_turn_hooks_routes_memory_through_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the model calls `recall_memory` via the operation surface, the
    operation context it receives has the live tool executor and defaults
    injected; the learning pack owns the recall behavior itself.
    """
    captured: dict[str, object] = {}

    async def _fake_execute_operation(
        operation_id: str,
        ctx: little_lisa.OperationExecutionContext,
    ) -> little_lisa.OperationResult:
        captured["operation_id"] = operation_id
        captured["mode_id"] = ctx.mode_id
        captured["query"] = ctx.query
        captured["has_introspect_fn"] = callable(ctx.introspect_fn)
        captured["default_limit"] = ctx.memory_recall_default_limit
        return little_lisa.OperationResult(output_text="ok")

    monkeypatch.setattr(little_lisa, "execute_operation", _fake_execute_operation)
    hooks = little_lisa._build_turn_hooks({"mode_id": "default"})

    ctx = little_lisa.OperationExecutionContext(mode_id="default", query="hello")
    result = await hooks.execute_operation("recall_memory", ctx)

    assert result.output_text == "ok"
    assert captured["operation_id"] == "recall_memory"
    assert captured["mode_id"] == "default"
    assert captured["query"] == "hello"
    assert captured["has_introspect_fn"] is True
    assert captured["default_limit"] == 8


@pytest.mark.asyncio
async def test_turn_hook_execute_operation_injects_memory_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def _fake_execute_operation(
        operation_id: str,
        ctx: little_lisa.OperationExecutionContext,
    ) -> little_lisa.OperationResult:
        captured["operation_id"] = operation_id
        captured["has_introspect_fn"] = callable(ctx.introspect_fn)
        captured["default_limit"] = ctx.memory_recall_default_limit
        return little_lisa.OperationResult(output_text="ok")

    monkeypatch.setattr(little_lisa, "execute_operation", _fake_execute_operation)
    hooks = little_lisa._build_turn_hooks({"mode_id": "default"})

    ctx = little_lisa.OperationExecutionContext(mode_id="default", query="hello")
    result = await hooks.execute_operation("recall_memory", ctx)

    assert result.output_text == "ok"
    assert captured["operation_id"] == "recall_memory"
    assert captured["has_introspect_fn"] is True
    assert captured["default_limit"] == 8


@pytest.mark.asyncio
async def test_synthesize_learning_prompt_mentions_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synthesize_learning = pytest.importorskip(
        "sdk_packs.learning.operations.memory.synthesize_learning"
    )
    captured: dict[str, object] = {}

    async def _fake_execute_tool_fn(tool_id: str, ctx: object) -> object:
        captured["tool_id"] = tool_id
        captured["tool_params"] = getattr(ctx, "parameters", {})
        return little_lisa.ToolResult(tool_id=tool_id, output_text="Memory stored: 1")

    async def _fake_complete_fn(**kwargs: object) -> object:
        captured["messages"] = kwargs["messages"]
        captured["response_format"] = kwargs["response_format"]
        return SimpleNamespace(content='{"facts":["Keep the distilled learning concise."]}')

    def _fake_resolve_model_fn(model_id: str) -> little_lisa.StubModelDefinition:
        _ = model_id
        return little_lisa.StubModelDefinition(
            model_id="gpt-chat-latest",
            provider="azure_openai",
        )

    ctx = little_lisa.OperationExecutionContext(
        mode_id="default",
        query="",
        response="",
        params={"content": "Test learning content"},
        complete_fn=_fake_complete_fn,
        resolve_model_fn=_fake_resolve_model_fn,
        execute_tool_fn=_fake_execute_tool_fn,
        preferred_model_id="gpt-chat-latest",
    )

    result = await synthesize_learning.execute(ctx)

    assert result.output_text.startswith("✅ Learning stored:")
    assert "json" in str(captured["messages"][0]["content"]).lower()
    assert captured["response_format"] is synthesize_learning.DistilledLearning
    assert captured["tool_id"] == "memory.add"


def test_little_lisa_discovers_and_runs_hello_world_pack_from_sdk_content_roots(
    tmp_path: Path,
) -> None:
    pack_root = tmp_path / "hello_world_pack"
    (pack_root / "modes" / "hello").mkdir(parents=True)
    (pack_root / "operations").mkdir(parents=True)
    (pack_root / "__init__.py").write_text("", encoding="utf-8")
    (pack_root / "modes" / "hello" / "_mode.md").write_text(
        textwrap.dedent(
            """\
            ---
            mode_id: hello
            start_stage_id: hello
            mode_kind: action
            require_query: false
            command_pattern: '^(?:/)?hello(?:[\\s,:;]+(?P<query>.+))?$'
            entry_operation_ids: [hello_operation]
            allowed_operation_ids: [hello_operation]
            description: "Hello world test mode."
            ---

            # Hello
            """
        ),
        encoding="utf-8",
    )
    (pack_root / "operations" / "hello_operation.yml").write_text(
        textwrap.dedent(
            """\
            operation_id: hello_operation
            description: Say hello from a custom pack.
            mode_allowlist: [hello]
            timeout_seconds: 30
            """
        ),
        encoding="utf-8",
    )
    (pack_root / "operations" / "hello_operation.py").write_text(
        textwrap.dedent(
            """\
            from kernel.models import OperationExecutionContext, OperationResult


            async def execute(ctx: OperationExecutionContext) -> OperationResult:
                output_text = f"Hello from custom pack: {ctx.query or 'hello'}"
                return OperationResult(output_text=output_text)
            """
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["SDK_CONTENT_ROOTS"] = str(pack_root)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in [str(tmp_path), env.get("PYTHONPATH", "")] if part
    )
    framework_root = Path(little_lisa.__file__).resolve().parents[2]
    little_lisa_script = Path(little_lisa.__file__).resolve()

    list_result = subprocess.run(
        [sys.executable, str(little_lisa_script), "--list-modes"],
        capture_output=True,
        check=True,
        cwd=framework_root,
        env=env,
        text=True,
    )
    run_result = subprocess.run(
        [sys.executable, str(little_lisa_script), "--once", "/hello framework"],
        capture_output=True,
        check=True,
        cwd=framework_root,
        env=env,
        text=True,
    )

    assert "- hello (action): Hello world test mode." in list_result.stdout
    assert "Hello from custom pack: framework" in run_result.stdout


def test_initialize_runtime_prefers_artifact_session_context(tmp_path: Path) -> None:
    session_file = tmp_path / "session_context.md"
    session_file.write_text('---\ndisplay_name: "Artifact User"\n---\n# hello\n', encoding="utf-8")

    # Initialize runtime with empty user_slug; data_dir points to tmp_path
    little_lisa._initialize_runtime(
        debug=False,
        log_level="",
        data_dir=str(tmp_path),
        memory_db="",
        thread_dir="",
        no_thread=False,
        user_slug="",
        timezone="",
        state_snapshot_path="",
        streaming="none",
    )

    assert little_lisa._RUNTIME_STATE["user_slug"] == "Artifact User"
    assert little_lisa._RUNTIME_STATE["session_context_ref"] == str(session_file)
    assert little_lisa._RUNTIME_STATE["session_context_active"] is True


def test_verify_command_respects_data_dir_override(tmp_path: Path) -> None:
    """Regression: verify must honor --data-dir instead of defaulting."""
    custom_dir = tmp_path / "custom_data"
    custom_dir.mkdir()
    (custom_dir / "memory.json").write_text("[]", encoding="utf-8")

    little_lisa._verify_command(
        data_dir=str(custom_dir),
        memory_db="",
        thread_dir="",
    )

    assert little_lisa._RUNTIME_STATE["data_dir"] == custom_dir
    assert little_lisa._RUNTIME_STATE["memory_db_path"] == custom_dir / "memory.json"
