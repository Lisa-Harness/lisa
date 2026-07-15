"""Tests that catalog descriptions cannot become implicit system prompts."""

from __future__ import annotations

from types import SimpleNamespace

from kernel import ipsar_fsm


def test_get_mode_system_intro_requires_authored_instruction(
    monkeypatch,
) -> None:
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