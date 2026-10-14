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
