"""Tests that catalog descriptions cannot become implicit system prompts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from kernel.ipsar_fsm import (
    MissingModeInstructionError,
    get_required_mode_intro,
    get_required_stage_instruction,
)

from kernel import ipsar_fsm


def test_get_mode_system_intro_requires_authored_instruction(
    monkeypatch,
) -> None:
    """The lenient lookup returns '' for empty body (membership-check contract)."""
    monkeypatch.setattr(
        ipsar_fsm,
        "get_workflow_catalog",
        lambda: {
            "metadata-only": SimpleNamespace(
                instruction_body="",
                description="This must not become model instruction.",
            )
        },
    )

    assert ipsar_fsm.get_mode_system_intro("metadata-only") == ""


def test_get_required_mode_intro_raises_on_empty_body(monkeypatch) -> None:
    """The strict lookup fails loudly instead of returning ''."""
    monkeypatch.setattr(
        ipsar_fsm,
        "get_workflow_catalog",
        lambda: {
            "metadata-only": SimpleNamespace(
                instruction_body="",
                description="This must not become model instruction.",
            )
        },
    )

    with pytest.raises(MissingModeInstructionError, match="empty instruction_body"):
        get_required_mode_intro("metadata-only")


def test_get_required_mode_intro_raises_on_unknown_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        ipsar_fsm,
        "get_workflow_catalog",
        lambda: {},
    )

    with pytest.raises(MissingModeInstructionError, match="not in the workflow catalog"):
        get_required_mode_intro("nonexistent")


def test_get_required_mode_intro_returns_body_when_present(monkeypatch) -> None:
    monkeypatch.setattr(
        ipsar_fsm,
        "get_workflow_catalog",
        lambda: {
            "coach": SimpleNamespace(
                instruction_body="# Coach Mode\nGuide the user.",
            ),
        },
    )

    assert get_required_mode_intro("coach") == "# Coach Mode\nGuide the user."


def test_get_required_stage_instruction_raises_on_empty_body(monkeypatch) -> None:
    monkeypatch.setattr(
        ipsar_fsm,
        "get_workflow_catalog",
        lambda: {
            "coach": SimpleNamespace(
                instruction_body="# Coach",
                stages={
                    "inquire": SimpleNamespace(instruction_body=""),
                },
            ),
        },
    )

    with pytest.raises(MissingModeInstructionError, match="empty"):
        get_required_stage_instruction("coach", "inquire")


def test_get_required_stage_instruction_raises_on_unknown_stage(monkeypatch) -> None:
    monkeypatch.setattr(
        ipsar_fsm,
        "get_workflow_catalog",
        lambda: {
            "coach": SimpleNamespace(
                instruction_body="# Coach",
                stages={},
            ),
        },
    )

    with pytest.raises(MissingModeInstructionError, match="not defined"):
        get_required_stage_instruction("coach", "missing")


def test_get_required_stage_instruction_raises_on_unknown_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        ipsar_fsm,
        "get_workflow_catalog",
        lambda: {},
    )

    with pytest.raises(MissingModeInstructionError, match="not in the workflow"):
        get_required_stage_instruction("nonexistent", "inquire")


def test_get_required_stage_instruction_returns_body_when_present(monkeypatch) -> None:
    monkeypatch.setattr(
        ipsar_fsm,
        "get_workflow_catalog",
        lambda: {
            "coach": SimpleNamespace(
                instruction_body="# Coach",
                stages={
                    "inquire": SimpleNamespace(instruction_body="Ask clarifying questions."),
                },
            ),
        },
    )

    assert get_required_stage_instruction("coach", "inquire") == "Ask clarifying questions."
