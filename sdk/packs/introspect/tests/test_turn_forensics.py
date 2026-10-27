"""Tests for the ``turn_forensics`` operation.

The operation is thin, so the tests focus on the contract the model depends
on: scope comes from the execution context (not from parameters), a missing
data root is reported as unknown evidence rather than a crash, and every
parameter reaches the reader intact.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from kernel.models import OperationExecutionContext
from pydantic import ValidationError
from sdk.packs.introspect.operations.forensics.turn_forensics import Parameters, execute
from sdk.thread._backend import _segment_paths


def _write_thread_entry(thread: Path, entry: dict) -> None:
    paths = _segment_paths(thread)
    paths.current.parent.mkdir(parents=True, exist_ok=True)
    with paths.current.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _seed(data_dir: Path, *, user: str = "u", turns: int = 2) -> None:
    thread = data_dir / "data" / "threads" / user
    now = datetime.now(UTC).isoformat()
    for turn in range(turns):
        _write_thread_entry(
            thread,
            {
                "timestamp_utc": now,
                "mode_id": "coding_execute",
                "query": f"query {turn}",
                "response_text": f"response {turn}",
                "turn": turn,
            },
        )
    _write_jsonl(
        data_dir / "logs" / "decisions.jsonl",
        [
            record
            for turn in range(turns)
            for record in (
                {
                    "topic": "turn.decision",
                    "turn_index": turn,
                    "action_type": "ContinueAction",
                    "iteration": 1,
                },
                {
                    "topic": "tool.telemetry",
                    "turn_index": turn,
                    "target_id": "read_file",
                    "target_type": "tool",
                    "status": "success",
                },
            )
        ],
    )


_PACK_ROOT = Path(__file__).resolve().parents[1] / "src" / "sdk" / "packs" / "introspect"


def _authored_loader(relative_path: str) -> str:
    """Read authored pack content the way a wired host does.

    ``require_authored_text`` deliberately raises when no loader is wired, so a
    context that can reach the unavailable-state message needs a real one.
    """
    return (_PACK_ROOT / relative_path).read_text(encoding="utf-8")


def _ctx(data_dir: Path | None, *, user: str = "u", **params: object) -> OperationExecutionContext:
    from kernel.seams.host_adapter import build_callable_host

    return OperationExecutionContext(
        operation_id="turn_forensics",
        user_slug=user,
        data_dir=str(data_dir) if data_dir is not None else "",
        params=dict(params),
        host=build_callable_host(read_authored_text_fn=_authored_loader),
    )


@pytest.mark.asyncio
async def test_execute_reads_the_latest_turn(tmp_path: Path) -> None:
    """The default selection is the most recent recorded turn."""
    _seed(tmp_path, turns=3)

    result = await execute(_ctx(tmp_path))

    assert result.status == "success"
    assert "Turn 2" in result.output_text
    assert result.output_json["turns"][0]["turn_index"] == 2
    assert result.extra["turn_count"] == 1


@pytest.mark.asyncio
async def test_execute_passes_a_range_through(tmp_path: Path) -> None:
    """``turns`` reaches the reader unmodified."""
    _seed(tmp_path, turns=4)

    result = await execute(_ctx(tmp_path, turns="1-2"))

    assert [t["turn_index"] for t in result.output_json["turns"]] == [1, 2]


@pytest.mark.asyncio
async def test_missing_data_dir_is_reported_as_unknown_not_clean(tmp_path: Path) -> None:
    """No data root means no evidence — the model must not read it as 'fine'."""
    result = await execute(_ctx(None))

    assert result.status == "error"
    assert "unavailable" in result.output_text
    assert "not a clean" in result.output_text.replace("**", "")
    assert result.output_json["available"] is False
    assert result.output_json["reason"] == "missing_data_dir"


@pytest.mark.asyncio
async def test_scope_comes_from_context_not_parameters(tmp_path: Path) -> None:
    """A parameter cannot redirect the read to another user's evidence."""
    _seed(tmp_path, user="alice", turns=1)
    _write_thread_entry(
        tmp_path / "data" / "threads" / "bob",
        {"mode_id": "default", "query": "bob secret", "turn": 0},
    )

    result = await execute(
        OperationExecutionContext(
            operation_id="turn_forensics",
            user_slug="alice",
            data_dir=str(tmp_path),
            # A hostile/imprecise caller tries to name the other user.
            params={"turns": "last", "user_slug": "bob"},
        )
    )

    assert "query 0" in result.output_text
    assert "bob secret" not in result.output_text


@pytest.mark.asyncio
async def test_max_records_is_clamped_by_the_schema(tmp_path: Path) -> None:
    """An out-of-range cap is rejected at the parameter boundary."""
    _seed(tmp_path)
    with pytest.raises(ValidationError):
        Parameters.model_validate({"max_records": 10_000})


@pytest.mark.asyncio
async def test_include_raw_notes_that_redaction_was_bypassed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The opt-out is visible in the bundle, not silent."""
    monkeypatch.setenv("LISA_THREAD_REDACTION_ENABLED", "1")
    _seed(tmp_path)

    result = await execute(_ctx(tmp_path, include_raw=True))

    assert any("redaction is bypassed" in note for note in result.extra["notes"])
    assert result.status == "success"


@pytest.mark.asyncio
async def test_redaction_is_applied_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the opt-out, thread text is redacted per the deployment policy."""
    monkeypatch.setenv("LISA_THREAD_REDACTION_ENABLED", "1")
    thread = tmp_path / "data" / "threads" / "u"
    _write_thread_entry(
        thread,
        {
            "mode_id": "default",
            "query": "why",
            "response_text": "I've hit my iteration budget in explainer (cap 12)",
            "turn": 0,
        },
    )

    result = await execute(_ctx(tmp_path))

    assert "hit my iteration budget" not in result.output_text


@pytest.mark.asyncio
async def test_missing_ledger_is_surfaced_in_the_output(tmp_path: Path) -> None:
    """A turn with no ledger evidence says so instead of showing a clean result."""
    thread = tmp_path / "data" / "threads" / "u"
    _write_thread_entry(thread, {"mode_id": "default", "query": "q", "turn": 0})

    result = await execute(_ctx(tmp_path))

    assert "decisions log: unavailable" in result.output_text
    assert result.extra["degraded"] is True


@pytest.mark.asyncio
async def test_focus_does_not_hide_evidence(tmp_path: Path) -> None:
    """Every focus still returns the dispatches and decisions."""
    _seed(tmp_path, turns=1)

    for focus in ("all", "blocked", "receipts", "cost", "latency"):
        result = await execute(_ctx(tmp_path, focus=focus))
        assert result.output_json["turns"][0]["dispatches"], f"{focus} dropped dispatches"
        assert result.output_json["turns"][0]["decisions"], f"{focus} dropped decisions"


@pytest.mark.asyncio
async def test_parameter_defaults_are_safe(tmp_path: Path) -> None:
    """Defaults: last turn, everything, redacted, no raw fields."""
    _seed(tmp_path, turns=2)

    result = await execute(_ctx(tmp_path))

    assert [t["turn_index"] for t in result.output_json["turns"]] == [1]
    assert not any("redaction is bypassed" in note for note in result.extra["notes"])


@pytest.mark.asyncio
async def test_output_is_markdown_with_a_verification_pointer(tmp_path: Path) -> None:
    """The model is always told how to check a claim against the record."""
    _seed(tmp_path, turns=1)

    result = await execute(_ctx(tmp_path))

    assert "## Turn forensics" in result.output_text
    assert "thread_read_range" in result.output_text
