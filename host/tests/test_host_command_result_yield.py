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

from host.types import HostCommandResult


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