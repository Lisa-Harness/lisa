"""Tests for the ``/forensics`` host command.

The command exists so an operator and the model cannot be shown two different
stories about one turn, so the tests pin the contract between them: the
operator view is the same renderer over the same reader, and every failure
mode is reported rather than smoothed over.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from runtime.commands import handle_forensics_command, render_forensics
from runtime.thread import set_thread_redaction_policy
from sdk.thread import ThreadRedactionPolicy
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


@pytest.fixture(autouse=True)
def _reset_redaction() -> object:
    """The redaction policy is a module-level singleton; reset it per test."""
    set_thread_redaction_policy(None)
    yield
    set_thread_redaction_policy(None)


def _seed(runtime: object, *, turns: int = 2) -> None:
    """Write a thread store and ledger for ``turns`` turns under the runtime."""
    base = runtime.paths.thread_path
    assert base is not None
    now = datetime.now(UTC).isoformat()
    for turn in range(turns):
        _write_thread_entry(
            base,
            {
                "timestamp_utc": now,
                "mode_id": "coding_execute",
                "query": f"query {turn}",
                "response_text": f"response {turn}",
                "turn": turn,
            },
        )
    _write_jsonl(
        runtime.paths.decisions_path,
        [
            {
                "topic": "turn.decision",
                "turn_index": turn,
                "action_type": "ContinueAction",
                "iteration": 1,
            }
            for turn in range(turns)
        ]
        + [
            {
                "topic": "tool.telemetry",
                "turn_index": turn,
                "target_id": "read_file",
                "target_type": "tool",
                "status": "success",
            }
            for turn in range(turns)
        ],
    )


def _runtime(tmp_path: Path) -> object:
    from runtime.testing import build_test_runtime

    models = "\n".join(
        [
            "version: 1",
            "mode_models: {default: gemma3:latest}",
            "models:",
            "  - model_id: gemma3:latest",
            "    provider: ollama",
        ]
    )
    return build_test_runtime(tmp_path, models_yaml=models, user_slug="u")


def test_default_argument_reads_the_latest_turn(tmp_path: Path) -> None:
    """No argument means the most recent recorded turn."""
    runtime = _runtime(tmp_path)
    _seed(runtime, turns=3)

    out = render_forensics(runtime, "")

    assert "Turn 2" in out
    assert "Turn 0" not in out


def test_explicit_turn_reads_that_turn(tmp_path: Path) -> None:
    """A numeric argument selects one turn."""
    runtime = _runtime(tmp_path)
    _seed(runtime, turns=3)

    out = render_forensics(runtime, "1")

    assert "Turn 1" in out


def test_range_reads_the_inclusive_window(tmp_path: Path) -> None:
    """``start-end`` includes both endpoints."""
    runtime = _runtime(tmp_path)
    _seed(runtime, turns=4)

    out = render_forensics(runtime, "1-2")

    assert "Turn 1" in out
    assert "Turn 2" in out


def test_invalid_selection_is_reported_not_rejected(tmp_path: Path) -> None:
    """A bad range is a note in the bundle, not an exception."""
    runtime = _runtime(tmp_path)
    _seed(runtime)

    out = render_forensics(runtime, "nonsense")

    assert "Read notes" in out
    assert "unparseable" in out
    assert "No turn entries matched" in out


def test_missing_sources_are_named(tmp_path: Path) -> None:
    """An empty data root says what it could not read."""
    runtime = _runtime(tmp_path)

    out = render_forensics(runtime, "last")

    assert "thread store: unavailable" in out
    assert "decisions log: unavailable" in out
    assert "not absent" in out


def test_operator_view_applies_the_host_redaction_policy(tmp_path: Path) -> None:
    """The operator view is redacted like every other projection."""
    runtime = _runtime(tmp_path)
    base = runtime.paths.thread_path
    assert base is not None
    _write_thread_entry(
        base,
        {
            "mode_id": "default",
            "query": "why",
            "response_text": "I've hit my iteration budget in explainer (cap 12)",
            "turn": 0,
        },
    )
    set_thread_redaction_policy(
        ThreadRedactionPolicy.from_patterns((r"I've hit my iteration budget[^\n]*",))
    )

    out = render_forensics(runtime, "last")

    assert "hit my iteration budget" not in out


def test_renders_through_the_shared_formatter(tmp_path: Path) -> None:
    """The command output is exactly ``render_bundle``'s output.

    This is the contract that keeps the operator view and the model view from
    diverging: one reader, one formatter, two entry points.
    """
    from sdk.forensics import ForensicReader, render_bundle

    runtime = _runtime(tmp_path)
    _seed(runtime, turns=1)

    reader = ForensicReader(
        runtime.paths.data_dir,
        thread_dir=runtime.paths.thread_dir,
        user_slug=runtime.config.user_slug,
        redaction=None,
    )
    expected = render_bundle(reader.bundle(turns="last"))

    # Compare everything after the generated-at header, which is time-dependent.
    actual = render_forensics(runtime, "last")
    assert actual.split("\n", 1)[1] == expected.split("\n", 1)[1]


def test_command_appends_a_thread_entry(tmp_path: Path) -> None:
    """The operator's own query is recorded, like every other host command."""
    runtime = _runtime(tmp_path)
    _seed(runtime, turns=1)

    response = handle_forensics_command(runtime, "/forensics last", "/forensics last")

    assert response.startswith("## Turn forensics")
    stored = runtime.paths.thread_path
    assert stored is not None
    entries = [
        json.loads(line)
        for line in (stored.with_name(f"{stored.name}.current.jsonl"))
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert entries[-1]["mode_id"] == "forensics"
    assert entries[-1]["query"] == "/forensics last"
