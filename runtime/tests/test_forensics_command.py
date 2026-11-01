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

#: Minimal model manifest for a test runtime.
_MODELS_YAML = "\n".join(
    [
        "version: 1",
        "mode_models: {default: gemma3:latest}",
        "models:",
        "  - model_id: gemma3:latest",
        "    provider: ollama",
    ]
)


def _write_thread_entry(thread: Path, entry: dict) -> None:
    paths = _segment_paths(thread)
    paths.current.parent.mkdir(parents=True, exist_ok=True)
    with paths.current.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write ledger rows, appending so a second conversation's rows coexist.

    The ledgers are shared by every conversation of a data root, so seeding
    two conversations must not truncate the first one's rows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


@pytest.fixture(autouse=True)
def _reset_redaction() -> object:
    """The redaction policy is a module-level singleton; reset it per test."""
    set_thread_redaction_policy(None)
    yield
    set_thread_redaction_policy(None)


def _seed(runtime: object, *, turns: int = 2) -> None:
    """Write a thread store and ledger for ``turns`` turns under the runtime.

    Ledger rows carry the plain ``user_slug`` because that is what the
    telemetry writer stamps: the shared ledgers are scoped by owner, while
    the thread store may be isolated per conversation. A reader that scoped
    the ledgers by the store stem instead would silently join no evidence.
    Rows also carry ``thread_store_slug``, so one thread's rows stay off
    another's turns.
    """
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
                "user_slug": runtime.config.user_slug,
                "thread_store_slug": runtime.config.thread_slug,
                "action_type": f"ContinueAction-{runtime.config.thread_slug}",
                "iteration": 1,
                "emitted_at": now,
            }
            for turn in range(turns)
        ]
        + [
            {
                "topic": "tool.telemetry",
                "turn_index": turn,
                "user_slug": runtime.config.user_slug,
                "thread_store_slug": runtime.config.thread_slug,
                "target_id": f"read_file-{runtime.config.thread_slug}",
                "target_type": "tool",
                "status": "success",
                "emitted_at": now,
            }
            for turn in range(turns)
        ],
    )


def _runtime(tmp_path: Path) -> object:
    from runtime.testing import build_test_runtime

    return build_test_runtime(tmp_path, models_yaml=_MODELS_YAML, user_slug="u")


def _conversation_runtime(tmp_path: Path, *, conversation_id: str = "conv-alpha") -> object:
    """Build a runtime whose thread store is isolated per conversation."""
    from runtime.testing import build_test_runtime

    return build_test_runtime(
        tmp_path,
        models_yaml=_MODELS_YAML,
        user_slug="u",
        conversation_id=conversation_id,
    )


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


def test_forensics_reads_the_conversation_scoped_thread_store(tmp_path: Path) -> None:
    """``/forensics`` must read the store the runtime writes to.

    Regression: with ``conversation_id`` set, the runtime writes the thread
    store to ``<user_slug>--<conversation_id>`` but ``/forensics`` resolved
    the store from ``config.user_slug`` alone, so the operator's view of a
    turn read a different (usually empty) store than the one the turn wrote.
    """
    runtime = _conversation_runtime(tmp_path)
    _seed(runtime, turns=2)

    # The runtime writes to the conversation-scoped store.
    assert runtime.paths.thread_path is not None
    assert runtime.paths.thread_path.name == "u--conv-alpha"

    out = render_forensics(runtime, "last")

    assert "Turn 1" in out


def test_forensics_joins_the_ledger_for_a_conversation_scoped_store(tmp_path: Path) -> None:
    """The store stem and the ledger owner are different values.

    The thread store is isolated per conversation, but the shared ledgers are
    stamped with the plain user slug. Scoping the ledgers by the store stem
    would join no evidence at all — silently, with no note — so this pins
    both halves: the store is found by its stem and the ledger is still
    scoped by its owner.
    """
    runtime = _conversation_runtime(tmp_path)
    _seed(runtime, turns=2)

    out = render_forensics(runtime, "last")

    assert "Turn 1" in out
    # The decision and dispatch rows are joined onto the turn.
    assert "ContinueAction" in out
    assert "read_file" in out
    # And no row was dropped for want of an owner match.
    assert "predate per-user attribution" not in out


def test_forensics_does_not_join_another_conversation(tmp_path: Path) -> None:
    """A conversation's bundle must not read another conversation's store.

    Two conversations of one user share a data root. Reading the store by
    the user slug alone would blend them, so each must see only its own.
    """
    first = _conversation_runtime(tmp_path, conversation_id="conv-alpha")
    _seed(first, turns=1)

    second = _conversation_runtime(tmp_path, conversation_id="conv-beta")
    _seed(second, turns=1)

    alpha_out = render_forensics(first, "last")
    beta_out = render_forensics(second, "last")

    assert "query 0" in alpha_out
    assert "query 0" in beta_out
    # Both stores hold a turn 0; each view shows its own single turn.
    assert alpha_out.count("### Turn ") == 1
    assert beta_out.count("### Turn ") == 1


def test_forensics_does_not_join_another_conversations_ledger(tmp_path: Path) -> None:
    """A conversation's bundle must not show another conversation's telemetry.

    Turn indices are per-thread, so two conversations of one user both count
    from zero and their ledger rows collide on the same index. The owner
    alone cannot tell them apart, so the rows carry the conversation too.
    """
    first = _conversation_runtime(tmp_path, conversation_id="conv-alpha")
    _seed(first, turns=1)

    second = _conversation_runtime(tmp_path, conversation_id="conv-beta")
    _seed(second, turns=1)

    alpha_out = render_forensics(first, "last")
    beta_out = render_forensics(second, "last")

    # Each conversation's own telemetry is present.
    assert "ContinueAction-u--conv-alpha" in alpha_out
    assert "ContinueAction-u--conv-beta" in beta_out
    # And the other conversation's is not.
    assert "read_file-u--conv-alpha" in alpha_out
    assert "read_file-u--conv-beta" in beta_out
    assert "conv-beta" not in alpha_out
    assert "conv-alpha" not in beta_out
