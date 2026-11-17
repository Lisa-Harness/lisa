"""Scripted model for compaction_spill_low_pressure.scenario.yml.

Verifies that a large tool result can remain inline when the turn is not under
soft-trigger pressure. The script seeds thread history, then calls
``thread_read_recent`` with include_transcript + max budget to produce a large
result (>8KiB). The turn fails if the result arrives as a spill preview.
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

    if user != "probe low pressure":
        raise ScriptError(f"unexpected user turn: {turn.user_text!r}")

    if turn.count(read_recent) == 0:
        return Reply.call(
            read_recent,
            limit=50,
            include_transcript=True,
            budget_bytes=32768,
        )

    last = turn.last(read_recent)
    if last is None:
        raise ScriptError("thread_read_recent call missing outcome")
    if last.text.startswith("[spilled tool result"):
        raise ScriptError("expected inline large result under low pressure, got spilled preview")
    if len(last.text) < 9000:
        raise ScriptError(
            f"thread_read_recent output too small for spill check: {len(last.text)} chars"
        )
    return Reply.text("confirmed inline large thread_read_recent output")
