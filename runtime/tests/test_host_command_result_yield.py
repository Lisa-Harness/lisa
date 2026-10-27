"""Tests for the yield surface on ``HostCommandResult``.

A turn that ends in the model's ``yield`` terminal action (the model asked
the operator a question instead of finishing) must be structurally
distinguishable from a completed answer: ``HostCommandResult.yielded_question``
carries the question and ``data["yielded"]`` mirrors it for panel transports.
One-shot hosts (``lisa once``) rely on this to fail fast instead of writing
the question into redirected output as if it were the requested artifact.
"""

from __future__ import annotations

from dataclasses import fields as dc_fields

from runtime.types import HostCommandResult


def test_host_command_result_has_yielded_question_field() -> None:
    """The dataclass carries the typed yielded_question field (default empty)."""
    names = {f.name for f in dc_fields(HostCommandResult)}
    assert "yielded_question" in names
    assert "operator_exit" in names
    # Default: an empty question means the turn completed normally.
    result = HostCommandResult(text="done")
    assert result.yielded_question == ""
    assert result.data is None


def test_host_command_result_yield_shape_roundtrip() -> None:
    """A yielded result carries the question on both the field and data."""
    result = HostCommandResult(
        text="I need the box's current network prefixes.",
        yielded_question="Which networks?",
        data={"yielded": "Which networks?"},
    )
    assert result.yielded_question == "Which networks?"
    assert result.data == {"yielded": "Which networks?"}


def test_handle_message_yield_payload_carries_context_and_mode() -> None:
    """A yielded turn's data carries context, target mode, and remediation."""
    # Contract under test: the runtime seam stamps the kernel's YieldAction
    # fields into data so hosts render a formatted yield (why + where to
    # reply) instead of parsing prose. Verified against the real builder.
    # pylint: disable=import-outside-toplevel
    from kernel.models import YieldAction
    from runtime.handle_message import _yield_payload

    payload = _yield_payload(
        YieldAction(
            question="Which networks?",
            context="recovery stall blocked the step",
            target_mode_id="coding_execute",
        ),
        remediation_lines=["/retry item-1"],
    )
    assert payload == {
        "yielded": "Which networks?",
        "yield_context": "recovery stall blocked the step",
        "yield_mode_id": "coding_execute",
        "remediation": ["/retry item-1"],
    }


def test_stall_yield_footnotes_the_escape_commands_in_the_text() -> None:
    """A stall yield must surface its slash commands in the response text.

    Regression for the 2026-10 turns 486-488 livelock. A recovery stall
    blocks the turn and yields; the only way out is a slash command. Those
    commands were built and placed in ``data["remediation"]`` but never in
    ``response_text``, so a transport rendering text alone showed the stall
    with no way to act.

    A plain-sentence reply cannot clear a stall: the affirmative-resume path
    only fires when an approval is pending, and these turns carried no
    ``pending_approval_request_id``. The operator answered "approved",
    "proceed", and "let's pick up where we left off" — three turns, three
    identical yields, every reply correct and none able to escape.
    """
    from runtime.handle_message import (
        _append_yield_remediation,
    )

    question = "A previous step stalled and needs your input to continue. Stall: approval required."
    commands = ["/skip abc123", "/cancel abc123"]

    footnoted = _append_yield_remediation(question, commands)

    # The question is preserved verbatim; the escape is added below it.
    assert footnoted.startswith(question)
    assert "/skip abc123" in footnoted
    assert "/cancel abc123" in footnoted

    # Idempotent: a thread entry that already carries the footnote (fed back
    # through the recent-conversation digest) must not double it.
    assert _append_yield_remediation(footnoted, commands) == footnoted

    # No remediation -> the text is untouched.
    assert _append_yield_remediation(question, []) == question


def test_with_yield_remediation_only_touches_yielded_turns() -> None:
    """A completed turn's text is never decorated."""
    from types import SimpleNamespace

    from runtime.handle_message import _with_yield_remediation

    done = SimpleNamespace(response_text="All finished.", yielded=None)
    assert _with_yield_remediation(done, ["/skip x"]) == "All finished."

    yielded = SimpleNamespace(
        response_text="Stalled.",
        yielded=object(),
    )
    assert "/skip x" in _with_yield_remediation(yielded, ["/skip x"])
