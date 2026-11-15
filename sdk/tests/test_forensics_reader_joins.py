"""Reader-side joins for the Evidence Envelope keys (Spec 1 S5).

Row-level discipline: every ledger test drives a real kernel writer (the
``TokenUsageTelemetryHandler`` / ``DecisionTelemetryHandler`` appenders) or
seeds the durable thread-entry JSONL the host writes, then reads the rows
back through :class:`~sdk.forensics.reader.ForensicReader`. Builders prove
nothing here: the row on disk is what the join consumes.

Covers the S5 acceptance set: per-call cost grouping, the dispatch/
tool_call join, the delegation tree, top-level lineage absence, the
resumed-continuation turn id, consecutive-run-manifest changes, and the
schema-version contract (legacy rows counted and joined, newer rows
reported, excluded, and never silently dropped).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from kernel.decision_telemetry import DecisionTelemetryHandler
from kernel.models import ToolExecutionTelemetry
from kernel.token_usage import TokenUsageTelemetryHandler, build_token_usage_event
from sdk.forensics import ForensicReader, render_bundle
from sdk.forensics._logrec import check_schema_version
from sdk.forensics._schema import SCHEMA_VERSION_CURRENT


def _entry(turn: int, **extra: object) -> dict:
    """One thread entry as the host writes it, optionally with lineage keys."""
    entry: dict = {
        "timestamp_utc": f"2026-11-01T0{turn}:00:00+00:00",
        "turn": turn,
        "user_slug": "adam",
        "mode_id": "exec",
        "query": f"q{turn}",
        "response_text": "r",
    }
    entry.update(extra)
    return entry


def _seed_thread(threads: Path, entries: list[dict]) -> None:
    threads.mkdir(parents=True, exist_ok=True)
    (threads / "adam.current.jsonl").write_text(
        "\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8"
    )


def _reader(data_dir: Path) -> ForensicReader:
    return ForensicReader(
        data_dir,
        thread_dir=data_dir / "data" / "threads",
        user_slug="adam",
        resolve_redaction_from_env=False,
    )


def _spend_token_rows(path: Path, events: list) -> None:
    """Write token rows through the real durable handler, not a fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = TokenUsageTelemetryHandler(path)
    for event in events:
        asyncio.run(handler.publish(event))


def test_streaming_fallback_rows_group_per_call(tmp_path: Path) -> None:
    """Two calls in one iteration read as two spends with distinct attempts."""
    _seed_thread(tmp_path / "data" / "threads", [_entry(0)])
    primary = build_token_usage_event(
        provider="p",
        model_id="m",
        prompt_tokens=100,
        completion_tokens=10,
        total_tokens=110,
        turn_index=0,
        user_slug="adam",
        call_id="call-primary",
        iteration=1,
        attempt="primary",
        request_ref="r" + "a" * 10,
    )
    fallback = build_token_usage_event(
        provider="p",
        model_id="m",
        prompt_tokens=900,
        completion_tokens=20,
        total_tokens=920,
        turn_index=0,
        user_slug="adam",
        call_id="call-fallback",
        iteration=1,
        attempt="stream_tool_fallback",
        request_ref="r" + "b" * 10,
    )
    _spend_token_rows(tmp_path / "logs" / "token-usage.jsonl", [primary, fallback])
    bundle = _reader(tmp_path).bundle(turns="last")
    turn = bundle.turns[0]
    assert [c.call_id for c in turn.call_costs] == ["call-primary", "call-fallback"]
    assert [c.attempt for c in turn.call_costs] == ["primary", "stream_tool_fallback"]
    assert [c.request_ref for c in turn.call_costs] == ["r" + "a" * 10, "r" + "b" * 10]
    assert turn.total_tokens == 1030
    assert turn.call_costs[0].iteration == 1
    out = render_bundle(bundle)
    assert "cost (per call):" in out
    assert "call call-primary" in out
    assert "attempt stream_tool_fallback" in out


def test_dispatch_row_names_the_tool_call_join(tmp_path: Path) -> None:
    """A published tool.telemetry row joins its dispatch to its tool call."""
    _seed_thread(tmp_path / "data" / "threads", [_entry(0)])
    handler = DecisionTelemetryHandler(tmp_path / "logs" / "decisions.jsonl")
    payload = ToolExecutionTelemetry(
        target_id="fetch_page",
        target_type="tool",
        status="error",
        duration_ms=812,
        error_message="timeout",
        turn_index=0,
        user_slug="adam",
        thread_store_slug="adam",
        dispatch_id="disp-1",
        tool_call_id="toolu_abc",
        call_id="call-1",
        iteration=2,
    ).model_dump(mode="json")
    asyncio.run(handler.publish_tool_telemetry(payload))
    bundle = _reader(tmp_path).bundle(turns="last")
    dispatch = bundle.turns[0].dispatches[0]
    assert dispatch.dispatch_id == "disp-1"
    assert dispatch.tool_call_id == "toolu_abc"
    out = render_bundle(bundle)
    assert "-> tool_call toolu_abc" in out


def test_delegate_child_joins_under_parent(tmp_path: Path) -> None:
    """A child entry's lineage keys build the tree under its parent turn."""
    _seed_thread(
        tmp_path / "data" / "threads",
        [
            _entry(0, turn_id="turn-top0"),
            _entry(1, turn_id="turn-parent"),
            _entry(
                2,
                turn_id="turn-child",
                parent_turn_id="turn-parent",
                parent_dispatch_id="disp-9",
            ),
        ],
    )
    bundle = _reader(tmp_path).bundle(turns="0-2")
    top = bundle.turns[1]
    assert top.turn_id == "turn-parent"
    assert not top.parent_turn_id and not top.parent_dispatch_id
    child = bundle.turns[2]
    assert child.turn_id == "turn-child"
    assert child.parent_turn_id == "turn-parent"
    assert child.parent_dispatch_id == "disp-9"
    roots = bundle.delegation_tree
    assert [n.turn_index for n in roots] == [0, 1]
    parent_node = roots[1]
    assert [c.turn_index for c in parent_node.children] == [2]
    assert parent_node.children[0].parent_dispatch_id == "disp-9"
    out = render_bundle(bundle)
    assert "### Delegation tree" in out
    assert "- turn 1 (turn_id=turn-parent)" in out
    assert ("  - turn 2 (turn_id=turn-child) -> parent turn-parent via dispatch disp-9") in out


def test_pack_reload_flags_consecutive_manifest_change(tmp_path: Path) -> None:
    """A run_ref/mode_digest difference between adjacent turns is flagged."""
    _seed_thread(
        tmp_path / "data" / "threads",
        [
            _entry(0, run_ref="run-a", mode_digest="md-a"),
            _entry(1, run_ref="run-a", mode_digest="md-a"),
            _entry(2, run_ref="run-b", mode_digest="md-b"),
        ],
    )
    bundle = _reader(tmp_path).bundle(turns="0-2")
    changes = bundle.run_manifest_changes
    assert len(changes) == 1
    change = changes[0]
    assert change.turn_index == 2
    assert change.previous_run_ref == "run-a"
    assert change.current_run_ref == "run-b"
    assert change.previous_mode_digest == "md-a"
    assert change.current_mode_digest == "md-b"
    out = render_bundle(bundle)
    assert "### Run manifest changes" in out
    assert "at turn 2: run-a / md-a -> run-b / md-b" in out


def test_resumed_continuation_keeps_turn_id(tmp_path: Path) -> None:
    """A continuation entry names the same execution, and roots stay roots."""
    _seed_thread(
        tmp_path / "data" / "threads",
        [_entry(0, turn_id="turn-resumed"), _entry(1, turn_id="turn-resumed")],
    )
    bundle = _reader(tmp_path).bundle(turns="0-1")
    assert bundle.turns[0].turn_id == bundle.turns[1].turn_id == "turn-resumed"
    assert all(not t.parent_turn_id and not t.parent_dispatch_id for t in bundle.turns)
    assert [n.turn_index for n in bundle.delegation_tree] == [0, 1]


def test_legacy_rows_still_join_and_are_counted(tmp_path: Path) -> None:
    """Rows predating schema_version stamping parse, join, and are counted."""
    _seed_thread(tmp_path / "data" / "threads", [_entry(0)])
    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    dispatch_row = {
        "topic": "tool.telemetry",
        "turn_index": 0,
        "user_slug": "adam",
        "target_id": "fetch_page",
        "target_type": "tool",
        "status": "ok",
        "duration_ms": 5,
        "logged_at": 1793376030.0,
    }
    token_row = {
        "turn_index": 0,
        "user_slug": "adam",
        "prompt_tokens": 10,
        "total_tokens": 10,
        "emitted_at": "2026-11-01T00:00:00+00:00",
    }
    (logs / "decisions.jsonl").write_text(json.dumps(dispatch_row) + "\n", encoding="utf-8")
    (logs / "token-usage.jsonl").write_text(json.dumps(token_row) + "\n", encoding="utf-8")
    bundle = _reader(tmp_path).bundle(turns="last")
    assert bundle.sources.legacy_row_count == 2
    assert bundle.turns[0].dispatches[0].target_id == "fetch_page"
    assert bundle.turns[0].total_tokens == 10
    assert any("predate schema_version stamping" in note for note in bundle.notes)


def test_newer_version_rows_reported_not_joined(tmp_path: Path) -> None:
    """Rows newer than the reader understands are counted, not silently dropped."""
    _seed_thread(tmp_path / "data" / "threads", [_entry(0)])
    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    dispatch_row = {
        "topic": "tool.telemetry",
        "schema_version": SCHEMA_VERSION_CURRENT + 1,
        "turn_index": 0,
        "user_slug": "adam",
        "target_id": "fetch_page",
        "target_type": "tool",
        "status": "ok",
        "duration_ms": 5,
    }
    token_row = {
        "schema_version": SCHEMA_VERSION_CURRENT + 5,
        "turn_index": 0,
        "user_slug": "adam",
        "prompt_tokens": 10,
        "total_tokens": 10,
        "emitted_at": "2026-11-01T00:00:00+00:00",
    }
    (logs / "decisions.jsonl").write_text(json.dumps(dispatch_row) + "\n", encoding="utf-8")
    (logs / "token-usage.jsonl").write_text(json.dumps(token_row) + "\n", encoding="utf-8")
    bundle = _reader(tmp_path).bundle(turns="last")
    assert bundle.sources.unreadable_row_count == 2
    assert not bundle.turns[0].dispatches
    assert bundle.turns[0].total_tokens == 0
    assert any("newer than this reader understands" in note for note in bundle.notes)


def test_legacy_and_per_call_cost_rendering_mix(tmp_path: Path) -> None:
    """Rows with a call id group per call; unnamed rows keep the aggregate."""
    _seed_thread(tmp_path / "data" / "threads", [_entry(0)])
    unnamed = build_token_usage_event(
        provider="p",
        model_id="m",
        prompt_tokens=5,
        total_tokens=5,
        turn_index=0,
        user_slug="adam",
    )
    named = build_token_usage_event(
        provider="p",
        model_id="m",
        prompt_tokens=7,
        total_tokens=7,
        turn_index=0,
        user_slug="adam",
        call_id="call-x",
        iteration=2,
        attempt="primary",
    )
    _spend_token_rows(tmp_path / "logs" / "token-usage.jsonl", [unnamed, named])
    bundle = _reader(tmp_path).bundle(turns="last")
    turn = bundle.turns[0]
    assert len(turn.costs) == 2
    assert [c.call_id for c in turn.call_costs] == ["call-x"]
    out = render_bundle(bundle)
    assert "cost (per call):" in out
    assert "legacy rows without a call id: 5 tokens" in out


def test_check_schema_version_contract() -> None:
    """Absent/unknown stays legacy; a higher version is reported unreadable."""
    assert check_schema_version({}) == (0, False)
    assert check_schema_version({"schema_version": 0}) == (0, False)
    assert check_schema_version({"schema_version": None}) == (0, False)
    assert check_schema_version({"schema_version": SCHEMA_VERSION_CURRENT}) == (
        SCHEMA_VERSION_CURRENT,
        False,
    )
    assert check_schema_version({"schema_version": str(SCHEMA_VERSION_CURRENT)}) == (
        SCHEMA_VERSION_CURRENT,
        False,
    )
    assert check_schema_version({"schema_version": SCHEMA_VERSION_CURRENT + 1}) == (
        SCHEMA_VERSION_CURRENT + 1,
        True,
    )
    assert check_schema_version({"schema_version": "junk"}) == (0, False)
    assert check_schema_version({"schema_version": -2}) == (0, False)
    assert check_schema_version({"schema_version": True}) == (0, False)
