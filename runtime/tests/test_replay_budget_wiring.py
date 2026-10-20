"""Turn-start replay working-set budget wiring (residency by allocation, R2).

Pins that ``build_turn_hooks`` wires ``prior_messages`` to bound the
replayed tool-result working set by the session plan's
``context.history.replay_max_bytes`` knob (tighten-only against the host
``context_budget_replay_max_bytes`` ceiling). Without an explicit ceiling,
replay stays full until the kernel allocates the complete model request.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from runtime.testing import build_test_runtime
from runtime.turn_hooks_builder import build_turn_hooks


def _seed_open_task(thread_path: Path, n_results: int = 6, result_size: int = 6000) -> None:
    """Append one paused turn whose six sub-threshold results accumulate.

    Each 6000-byte result is below the 8192 inline threshold, so per-result
    projection keeps them verbatim; only a working-set budget collapses
    them. Six results = 36000 bytes of resident tool content.
    """
    from runtime.thread import append_thread_entry

    calls = [
        {"id": f"c{i}", "type": "function", "function": {"name": "read", "arguments": "{}"}}
        for i in range(n_results)
    ]
    transcript = [
        {"role": "user", "content": "scan"},
        {"role": "assistant", "content": "", "tool_calls": calls},
        *[
            {"role": "tool", "tool_call_id": f"c{i}", "content": "x" * result_size}
            for i in range(n_results)
        ],
    ]
    append_thread_entry(
        thread_path=thread_path,
        user_slug="user",
        mode_id="worker",
        query="scan",
        response_text="paused",
        provenance={"source": "observed", "operator_exit": "paused"},
        transcript=transcript,
    )


def _tool_bytes(messages: list[dict[str, object]]) -> int:
    return sum(
        len(str(m.get("content") or "").encode("utf-8"))
        for m in messages
        if m.get("role") == "tool"
    )


def _runtime_with_history_knob(tmp_path: Path, replay_max_bytes: int | None) -> object:
    """Build a runtime whose compiled session history knob is forced.

    Each case explicitly selects a ceiling or defers to request allocation.
    """
    runtime = build_test_runtime(tmp_path, user_slug="user", surface_type="dm", surface_id="s")
    plan = runtime.session_plan
    history = plan.manifest.context.history.model_copy(
        update={"replay_max_bytes": replay_max_bytes}
    )
    context = plan.manifest.context.model_copy(update={"history": history})
    manifest = plan.manifest.model_copy(update={"context": context})
    runtime.session_plan = plan.model_copy(update={"manifest": manifest})
    return runtime


def test_prior_messages_bounds_working_set_to_history_knob(tmp_path: Path) -> None:
    """The session knob tightens the replay working set: 36000 verbatim
    bytes collapse to ref-stubs under a 12288-byte budget."""
    runtime = _runtime_with_history_knob(tmp_path, replay_max_bytes=12288)
    _seed_open_task(runtime.paths.thread_path)

    hooks = build_turn_hooks(runtime, context=None, transport=None)

    messages = hooks.context.prior_messages()
    assert _tool_bytes(messages) <= 12288
    stub_refs = [
        json.loads(str(m["content"]))["ref"]
        for m in messages
        if m.get("role") == "tool" and str(m.get("content") or "").startswith('{"ref"')
    ]
    assert stub_refs, "expected over-budget results to replay as ref-stubs"


def test_prior_messages_core_default_session_defers_to_request_budget(tmp_path: Path) -> None:
    """The core default admits complete replay for whole-request allocation."""
    runtime = build_test_runtime(tmp_path, user_slug="user", surface_type="dm", surface_id="s")
    assert runtime.session_plan.manifest.context.history.replay_max_bytes is None
    _seed_open_task(runtime.paths.thread_path)

    hooks = build_turn_hooks(runtime, context=None, transport=None)

    messages = hooks.context.prior_messages()
    assert _tool_bytes(messages) == 36000
    assert hooks.models.count_request_tokens is not None


def test_prior_messages_host_ceiling_when_knob_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicitly-absent knob keeps the host ceiling: a 36000-byte set
    stays verbatim (36000 < 65536 default), exactly as before this feature."""
    runtime = _runtime_with_history_knob(tmp_path, replay_max_bytes=None)
    _seed_open_task(runtime.paths.thread_path)

    hooks = build_turn_hooks(runtime, context=None, transport=None)

    messages = hooks.context.prior_messages()
    assert _tool_bytes(messages) == 36000


def test_prior_messages_knob_cannot_exceed_host_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tighten-only: a knob above the host ceiling is capped back to it.
    A 131072-byte knob cannot loosen a 12288-byte env ceiling."""
    monkeypatch.setenv("LISA_CONTEXT_BUDGET_REPLAY_MAX_BYTES", "12288")
    runtime = _runtime_with_history_knob(tmp_path, replay_max_bytes=131072)
    assert runtime.config.context_budget_replay_max_bytes == 12288
    _seed_open_task(runtime.paths.thread_path)

    hooks = build_turn_hooks(runtime, context=None, transport=None)

    messages = hooks.context.prior_messages()
    assert _tool_bytes(messages) <= 12288


def test_default_replay_and_live_results_share_actual_remaining_window(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from kernel.turn_phases._turn_state import _TurnState
    from kernel.turn_runner import _project_provider_messages
    from runtime.thread import load_result_object

    runtime = _runtime_with_history_knob(tmp_path, replay_max_bytes=None)
    _seed_open_task(runtime.paths.thread_path, result_size=30_000)
    hooks = build_turn_hooks(runtime, context=None, transport=None)
    replay = hooks.context.prior_messages()
    assert _tool_bytes(replay) == 180_000, "default replay must not pre-trim to 64KB"
    messages = [{"role": "system", "content": "fixed"}, *replay]
    state = _TurnState()
    large = SimpleNamespace(model_id="unknown", context_window=131_072)
    fitted = _project_provider_messages(state, hooks, messages, large, max_tokens=4096)
    assert fitted == messages, "ample window must preserve all admitted evidence"
    assert state.request_context_budget.reduced is False

    small = SimpleNamespace(model_id="unknown", context_window=32_768)
    pressured = _project_provider_messages(state, hooks, messages, small, max_tokens=4096)
    budget = state.request_context_budget
    assert budget.reduced is True
    assert budget.input_tokens + budget.margin_tokens + budget.completion_tokens <= 32_768
    assert _tool_bytes(pressured) < 180_000
    references = [
        json.loads(str(m["content"]))["ref"]
        for m in pressured
        if m.get("role") == "tool" and str(m.get("content") or "").startswith('{"ref"')
    ]
    assert references
    assert all(
        load_result_object(runtime.paths.thread_path, ref)["content"] == "x" * 30_000
        for ref in references
    )
    restored = _TurnState()
    restored.provider_messages = pressured
    restored.projected_message_count = len(messages)
    expanded_window = _project_provider_messages(restored, hooks, messages, large, max_tokens=4096)
    assert expanded_window == pressured, "growing capacity must not rewrite old references"
