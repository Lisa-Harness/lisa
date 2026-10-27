"""Tests for ``/workflow`` — inspecting and concluding the active workflow.

The command exists because a plan had only two endings: finish every step, or
supersede it with a new plan. An abandoned plan therefore stayed active
forever, and because the derived view is injected into every turn's briefing
and steers routing, it kept misdirecting handoffs long after the operator had
moved on. ``/workflow conclude`` retires it durably by appending a thread
entry, which is the same source of truth the view derives from.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from kernel.workflow_view import WORKFLOW_CONCLUSION_KEY, derive_workflow_view
from runtime.commands._workflow import handle_workflow_command
from runtime.thread import append_thread_entry, load_thread_range

_PLAN = """\
## Plan

1. **coding** — Implement the fix
   _writes: coding_commit_
2. **explainer** — Write the summary
   _writes: markdown_forge_document_
"""


def _runtime(tmp_path: Path) -> Any:
    """Minimal runtime stub for the host-command surface."""

    class _Session:
        current_mode_id = "default"
        current_stage_id = "-"
        next_model_id = "-"

        def set_last_turn_context(self, _context: Any) -> None:
            return None

    return SimpleNamespace(
        paths=SimpleNamespace(thread_path=tmp_path / "thread"),
        config=SimpleNamespace(user_slug="user", thread_segment_turns=256),
        session=_Session(),
    )


def _seed_plan(runtime: Any, *, turn: int = 0) -> None:
    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug="user",
        mode_id="planner",
        query="make a plan",
        response_text=_PLAN,
        turn=turn,
    )


def _entries(runtime: Any) -> list[dict[str, Any]]:
    return load_thread_range(
        thread_path=runtime.paths.thread_path,
        start_turn=0,
        end_turn=9999,
    )


def _active(runtime: Any) -> Any:
    return derive_workflow_view(_entries(runtime))


def test_status_without_a_workflow_says_so(tmp_path: Path) -> None:
    out = handle_workflow_command(_runtime(tmp_path), "/workflow", "/workflow")
    assert "No workflow is active" in out
    assert "planner" in out, "the message should say how a workflow starts"


def test_status_lists_steps_and_the_conclude_hint(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    _seed_plan(runtime)

    out = handle_workflow_command(runtime, "/workflow status", "/workflow status")

    assert "## Workflow progress" in out
    assert "[ ] coding" in out
    assert "[ ] explainer" in out
    assert "0 of 2 step(s) complete" in out
    assert "/workflow conclude" in out


def test_status_marks_completed_steps(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    _seed_plan(runtime, turn=0)
    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug="user",
        mode_id="coding",
        query="implement",
        response_text="Committed.",
        effect_receipts=[{"capability_id": "coding_commit", "effect": "write"}],
        turn=1,
    )

    out = handle_workflow_command(runtime, "/workflow status", "/workflow status")

    assert "[x] coding" in out
    assert "[ ] explainer" in out
    assert "1 of 2 step(s) complete" in out


def test_conclude_retires_the_workflow(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    _seed_plan(runtime)
    assert _active(runtime) is not None

    out = handle_workflow_command(
        runtime, "/workflow conclude no longer needed", "/workflow conclude no longer needed"
    )

    assert "Concluded" in out
    assert "no longer needed" in out
    assert _active(runtime) is None, "the plan must stop deriving after a conclusion"


def test_conclude_reports_which_steps_never_ran(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    _seed_plan(runtime)
    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug="user",
        mode_id="coding",
        query="implement",
        response_text="Committed.",
        effect_receipts=[{"capability_id": "coding_commit", "effect": "write"}],
        turn=1,
    )

    out = handle_workflow_command(runtime, "/workflow conclude", "/workflow conclude")

    assert "1 of 2 step(s) complete" in out
    assert "Not run: explainer" in out


def test_conclusion_is_recorded_in_the_thread(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    _seed_plan(runtime)

    handle_workflow_command(runtime, "/workflow conclude shipped elsewhere", "/workflow conclude")

    concluding = [
        entry
        for entry in _entries(runtime)
        if isinstance(entry.get("provenance"), dict)
        and WORKFLOW_CONCLUSION_KEY in entry["provenance"]
    ]
    assert len(concluding) == 1
    assert concluding[0]["provenance"][WORKFLOW_CONCLUSION_KEY] == "shipped elsewhere"
    assert concluding[0]["mode_id"] == "workflow"


def test_conclusion_survives_a_fresh_derivation(tmp_path: Path) -> None:
    """The marker is durable thread data, not in-process state."""
    runtime = _runtime(tmp_path)
    _seed_plan(runtime)
    handle_workflow_command(runtime, "/workflow conclude done", "/workflow conclude done")

    # Re-read the thread from disk as a restarted host would.
    raw = (tmp_path / "thread.current.jsonl").read_text(encoding="utf-8")
    entries = [json.loads(line) for line in raw.splitlines() if line.strip()]
    assert derive_workflow_view(entries) is None


def test_conclude_without_a_workflow_is_harmless(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)

    out = handle_workflow_command(runtime, "/workflow conclude", "/workflow conclude")

    assert "No workflow is active" in out
    assert not _entries(runtime), "nothing should be recorded when nothing was active"


def test_alias_subcommands_conclude(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    _seed_plan(runtime)

    out = handle_workflow_command(runtime, "/workflow abandon", "/workflow abandon")

    assert "Concluded" in out
    assert _active(runtime) is None


def test_unknown_subcommand_reports_usage_without_concluding(tmp_path: Path) -> None:
    """A typo must not look like it did something."""
    runtime = _runtime(tmp_path)
    _seed_plan(runtime)

    out = handle_workflow_command(runtime, "/workflow conclud", "/workflow conclud")

    assert "Unrecognized subcommand" in out
    assert "conclud" in out
    assert _active(runtime) is not None, "a typo must not retire the workflow"


def test_new_plan_after_conclusion_starts_a_fresh_workflow(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    _seed_plan(runtime, turn=0)
    handle_workflow_command(runtime, "/workflow conclude", "/workflow conclude")

    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug="user",
        mode_id="planner",
        query="new plan",
        response_text="## Plan\n\n1. **thinker** — Think it over\n   _reads_\n",
        turn=3,
    )

    view = _active(runtime)
    assert view is not None
    assert [step.mode_id for step in view.declared_steps] == ["thinker"]


@pytest.mark.parametrize("subcommand", ["", "status", "show"])
def test_status_subcommands_never_mutate(tmp_path: Path, subcommand: str) -> None:
    runtime = _runtime(tmp_path)
    _seed_plan(runtime)
    before = len(_entries(runtime))

    handle_workflow_command(runtime, f"/workflow {subcommand}".strip(), "/workflow")

    assert len(_entries(runtime)) == before
    assert _active(runtime) is not None


@pytest.mark.asyncio
async def test_workflow_command_is_reachable_through_host_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``/workflow`` must resolve before the model-turn path.

    Host commands are dispatched before routing, so this also pins that the
    prefix does not shadow a mode command (``/plan`` could not be used for
    this reason — it is the planner mode's own ``command_pattern``).
    """
    import runtime.handle_message as hm

    runtime = _runtime(tmp_path)
    runtime.paths.data_dir = tmp_path
    _seed_plan(runtime)

    class _Transport:
        def __init__(self) -> None:
            self.responses: list[str] = []

        async def on_response(self, text: str, data: Any = None) -> None:
            self.responses.append(text)

    async def _no_pool() -> None:
        return None

    runtime.ensure_pool_ready = _no_pool
    runtime.host_commands = SimpleNamespace()
    transport = _Transport()

    result = await hm._dispatch_host_command(
        runtime, "/workflow status", "/workflow status", "/workflow status", transport
    )

    assert result is not None, "/workflow must be handled as a host command"
    assert "## Workflow progress" in result.text
