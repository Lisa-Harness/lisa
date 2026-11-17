"""Scripted model for compaction_spill_high_pressure.scenario.yml.

Verifies that spill engages under pressure by repeatedly requesting large
``thread_read_recent`` payloads in one turn. The script succeeds once any
recorded ``thread_read_recent`` outcome has been compacted to a spill preview.
"""

from __future__ import annotations

from runtime.testing.scripted import Reply, ScriptError, ScriptTurn

_SEED_TEXT = (
    "seeded history line for compaction coverage; "
    "this text is intentionally long to inflate read_recent payload size. "
) * 5


def _tool_name(turn: ScriptTurn, suffix: str) -> str:
    for name in sorted(turn.tool_names):
        if name.rsplit(".", 1)[-1] == suffix:
            return name
    raise ScriptError(f"{suffix} not offered; tools={sorted(turn.tool_names)}")


def _scenario_prompt(turn: ScriptTurn) -> str:
    marker = "## User Query"
    for message in turn.request.messages:
        if message.get("role") != "user":
            continue
        text = str(message.get("content", ""))
        if marker in text:
            return text.split(marker, 1)[1].strip().lower()
    return turn.user_text.lower().strip()


def auxiliary(turn: ScriptTurn) -> Reply:
    if turn.request.response_format is not None:
        return Reply.text("{}")
    return Reply.text("")


def respond(turn: ScriptTurn) -> Reply:
    user = _scenario_prompt(turn)
    read_recent = _tool_name(turn, "thread_read_recent")

    if user.startswith("seed "):
        return Reply.text(_SEED_TEXT)

    if user != "probe high pressure":
        raise ScriptError(f"unexpected user turn: {turn.user_text!r}")

    outcomes = [outcome for outcome in turn.outcomes() if outcome.name == read_recent]
    if any(outcome.text.startswith("[spilled tool result") for outcome in outcomes):
        return Reply.text("confirmed spill preview under pressure")

    if len(outcomes) >= 8:
        lengths = [len(outcome.text) for outcome in outcomes]
        raise ScriptError(
            f"expected spill preview under pressure, none seen after "
            f"{len(outcomes)} calls; lengths={lengths}"
        )

    if turn.count(read_recent) == 0:
        return Reply.call(
            read_recent,
            limit=50,
            include_transcript=True,
            budget_bytes=32768,
        )
    return Reply.call(
        read_recent,
        limit=50,
        include_transcript=True,
        budget_bytes=32768,
    )
