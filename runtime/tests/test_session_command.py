"""Tests for ``/session`` — inspecting and repairing session-plan identity across pauses.

The command surfaces the content-addressed session archive: whether the
latest paused turn's saved digest matches the live plan, what is on disk,
and how to restore an archived source. These tests lock the operator-facing
verdicts and the restore path, including that the restore takes effect on
the *next* boot (a per-message host boots fresh each turn).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from kernel.session_plan import compile_session_plan
from runtime.commands._session import handle_session_command
from runtime.session_archive import archive_session_source, sessions_dir
from runtime.thread import append_thread_entry

_V1 = """\
---
schema_version: 2
name: sessioncmd
surface:
  type: personal_dm
  id: sessioncmd
  memory_scope: personal
execution:
  max_capability_calls: 150
---
Paused body.
"""

_V2 = _V1.replace("150", "500")


def _digest_of(text: str, uri: Path) -> str:
    return compile_session_plan(text=text, source_uri=str(uri)).plan_digest


def _runtime(tmp_path: Path, *, digest: str = "") -> Any:
    """Minimal runtime stub for the host-command surface.

    ``digest`` stands in for the booted plan's digest; the real ``HostRuntime``
    seam is covered by the boot-hook test.
    """
    return SimpleNamespace(
        session_plan=SimpleNamespace(
            plan_digest=digest or _digest_of(_V1, tmp_path / "config" / "session_context.md"),
            source_uri=str(tmp_path / "config" / "session_context.md"),
        ),
        paths=SimpleNamespace(
            thread_path=tmp_path / "threads" / "t",
            data_dir=tmp_path,
        ),
        config=SimpleNamespace(user_slug="user", thread_segment_turns=256),
    )


def _seed_paused_turn(runtime: Any, digest: str, *, turn: int = 3) -> None:
    """Append a paused thread entry whose continuation stamps ``digest``."""
    from kernel.turn_continuation import TurnContinuation
    from kernel.turn_hooks import TurnIdentity
    from runtime.turn_continuation import persist_continuation

    identity = TurnIdentity(session_plan_digest=digest)
    saved = TurnContinuation(
        identity=identity,
        mode_id="default",
        query="original request",
        state_json="{}",
        context_json="{}",
        mode_config_json="{}",
        segments_json="[]",
        preferred_model_id=None,
        max_iterations=3,
        per_step_max_iterations=0,
        per_step_max_tool_calls=0,
        turn_budget=None,
        execution_policy=None,
        suppress_routing_hint=False,
        delegate_depth=0,
        mode_instruction="",
        reason="cap",
    )
    ref = persist_continuation(runtime.paths.thread_path, saved)
    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug=runtime.config.user_slug,
        mode_id="default",
        query="original request",
        response_text="capped",
        provenance={
            "source": "observed",
            "operator_exit": "paused",
            "turn_continuation_ref": ref,
        },
        turn=turn,
    )


def test_status_with_no_paused_turn_says_so(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    out = handle_session_command(_runtime(tmp_path), "/session", "/session")
    assert "## Session plan" in out
    assert "Latest paused turn: none" in out


def test_status_reports_resumable_when_digests_match(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    uri = tmp_path / "config" / "session_context.md"
    uri.write_text(_V1, encoding="utf-8")
    runtime = _runtime(tmp_path)
    archive_session_source(str(uri), _V1, data_dir=tmp_path)
    _seed_paused_turn(runtime, runtime.session_plan.plan_digest)

    out = handle_session_command(runtime, "/session status", "/session status")
    assert "**resumable**" in out
    assert "Send" in out


def test_status_reports_blocked_when_live_differs_and_names_the_repair(
    tmp_path: Path,
) -> None:
    (tmp_path / "config").mkdir()
    uri = tmp_path / "config" / "session_context.md"
    saved_digest = _digest_of(_V1, uri)
    archive_session_source(str(uri), _V1, data_dir=tmp_path)
    runtime = _runtime(tmp_path, digest=_digest_of(_V2, uri))
    _seed_paused_turn(runtime, saved_digest)

    out = handle_session_command(runtime, "/session", "/session")
    assert "**blocked**" in out
    assert f"/session restore {saved_digest[:12]}" in out, "the verdict names the repair"


def test_archive_lists_entries_and_marks_the_live_digest(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    uri = tmp_path / "config" / "session_context.md"
    archive_session_source(str(uri), _V1, data_dir=tmp_path)
    runtime = _runtime(tmp_path, digest=_digest_of(_V1, uri))

    out = handle_session_command(runtime, "/session archive", "/session archive")
    assert "## Session archive" in out
    assert "live digest" in out
    assert str(uri) in out


def test_restore_subcommand_restores_the_archived_source(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    uri = tmp_path / "config" / "session_context.md"
    digest = _digest_of(_V1, uri)
    archive_session_source(str(uri), _V1, data_dir=tmp_path)
    runtime = _runtime(tmp_path)
    uri.write_text(_V2, encoding="utf-8")

    out = handle_session_command(
        runtime, f"/session restore {digest[:12]}", f"/session restore {digest[:12]}"
    )
    assert "verified" in out
    assert uri.read_text(encoding="utf-8") == _V1
    assert uri.with_suffix(".md.bak").read_text(encoding="utf-8") == _V2


def test_archive_with_empty_inventory_prints_the_guidance(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    out = handle_session_command(_runtime(tmp_path), "/session archive", "/session archive")
    assert "The archive is empty" in out


def test_archive_flags_a_missing_source_file(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    uri = tmp_path / "config" / "session_context.md"
    digest = _digest_of(_V1, uri)
    archive_session_source(str(uri), _V1, data_dir=tmp_path)
    runtime = _runtime(tmp_path, digest=_digest_of(_V2, uri))
    (tmp_path / "state" / "sessions" / f"{digest}.md").unlink()

    out = handle_session_command(runtime, "/session archive", "/session archive")
    assert "**source file missing**" in out


def test_restore_failure_reports_the_verdict_not_a_crash(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    out = handle_session_command(_runtime(tmp_path), "/session restore ff", "/session restore ff")
    assert "## Session restore failed" in out
    assert "No archived session source matches" in out


def test_simple_module_routes_session_command_to_the_lazily_imported_handler(
    tmp_path: Path,
) -> None:
    """The ``_simple`` delegation reaches ``_session`` through the lazy import."""
    from runtime.commands._simple import handle_session_command as simple_handle

    (tmp_path / "config").mkdir()
    runtime = _runtime(tmp_path)
    out = simple_handle(runtime, "/session archive", "/session archive")
    assert "## Session archive" in out


def test_restore_without_a_digest_prints_usage(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    out = handle_session_command(_runtime(tmp_path), "/session restore", "/session restore")
    assert "Usage:" in out


def test_unknown_subcommand_prints_usage_without_side_effects(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    out = handle_session_command(_runtime(tmp_path), "/session nuke", "/session nuke")
    assert "Usage:" in out
    assert "Unrecognized subcommand" in out
    assert not sessions_dir(tmp_path).exists()


def test_broken_continuation_reference_is_skipped_not_fatal(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    runtime = _runtime(tmp_path)
    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug=runtime.config.user_slug,
        mode_id="default",
        query="original request",
        response_text="capped",
        provenance={
            "source": "observed",
            "operator_exit": "paused",
            "turn_continuation_ref": "rdeadbeef",
        },
        turn=5,
    )
    out = handle_session_command(runtime, "/session", "/session")
    assert "Latest paused turn: none" in out, "an unreadable ref must not fail the command"


def test_paused_scan_skips_entries_without_continuation_refs(tmp_path: Path) -> None:
    """Normal (non-paused) thread entries are skipped in the latest-pause scan."""
    (tmp_path / "config").mkdir()
    runtime = _runtime(tmp_path)
    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug=runtime.config.user_slug,
        mode_id="default",
        query="regular question",
        response_text="answered",
        provenance={"source": "observed"},
        turn=2,
    )
    out = handle_session_command(runtime, "/session", "/session")
    assert "Latest paused turn: none" in out


def test_paused_scan_skips_a_none_saved_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A load that resolves to ``None`` (empty ref) does not crash the status scan."""
    import runtime.turn_continuation as continuation_mod

    (tmp_path / "config").mkdir()
    runtime = _runtime(tmp_path)
    _seed_paused_turn(runtime, runtime.session_plan.plan_digest)
    monkeypatch.setattr(continuation_mod, "load_continuation", lambda *_a, **_k: None)

    out = handle_session_command(runtime, "/session", "/session")
    assert "Latest paused turn: none" in out
