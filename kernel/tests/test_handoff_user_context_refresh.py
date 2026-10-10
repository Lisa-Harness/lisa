"""Regression: the turn's own capability block follows the active mode after a handoff.

The 2026-10-10 turn-664 incident (p920): a bare "proceed" entered in
``default``, the model handed off mid-turn to ``coding_execute``, and the
recompiled tool surface carried all 36 coding tools — but the turn's own
user message still displayed the turn-start capability block assembled for
the entry mode ("mode: default / available_tools: 16"). The model read its
own authoritative context, concluded the coding tools were absent after the
handoff, and reported a harness defect instead of implementing the change.
The write-contract backstop then armed and burned two further completions.

The turn loop's rule stands: the kernel never synthesizes new ``user``
messages. These tests pin that the turn's own user message is REFRESHED —
the capability block recomputed against the active mode — after a mid-turn
handoff, and that no replayed history is touched.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from kernel.models import WorkflowModeConfig
from kernel.turn_hooks import ContextHooks, ModelHooks, RecallHooks
from kernel.turn_runner import TurnHooks, run_turn


@pytest.mark.asyncio
async def test_user_capability_context_refreshes_after_mid_turn_handoff() -> None:
    """After a mid-turn handoff, the turn's own user message must carry the
    active mode's capability block — not the entry mode's.

    The turn runner rebuilds the tool surface and recompiles the capability
    manifest on every handoff; the user message must be reassembled with the
    recompiled manifest so the newest user state matches the function schema
    the model can actually call.
    """
    calls: list[dict[str, Any]] = []

    async def complete(**kwargs: Any) -> Any:
        messages = kwargs["messages"]
        calls.append({"messages": messages, "tools": kwargs.get("tools")})
        user_messages = [m for m in messages if m.get("role") == "user"]
        system_has_handoff = "## Handoff Context" in str(messages[0].get("content", ""))
        if len(calls) == 1:
            # Entry mode: hand off mid-turn.
            return SimpleNamespace(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "transition_stage",
                            "arguments": json.dumps(
                                {"target_stage_id": "search", "rationale": "implement the change"}
                            ),
                        },
                    }
                ],
            )
        # Post-handoff: the newest user message must describe the ACTIVE mode.
        assert system_has_handoff, "post-handoff iteration must carry the handoff context"
        assert user_messages, "post-handoff iteration must have a user message"
        newest_user = user_messages[-1]["content"]
        assert "mode: search" in newest_user, (
            "turn's own capability block must name the active mode after the handoff; "
            f"newest user message: {newest_user!r}"
        )
        assert "mode: default" not in newest_user, (
            "entry-mode capability block must not survive the handoff as the newest user state"
        )
        return SimpleNamespace(content="Implemented the change in search mode.")

    default_cfg = _chat_mode_config(mode_id="default")
    search_cfg = _chat_mode_config(mode_id="search", satisfies_intents=["research"])
    catalog = {"default": default_cfg, "search": search_cfg}

    hooks = _hooks_with_recorded_user_context(complete)

    with _patched_catalog(catalog):
        result = await run_turn(mode_id="default", query="proceed", hooks=hooks)

    assert result.handoff_to == "search"
    assert len(calls) >= 2, "the post-handoff completion must run"


@pytest.mark.asyncio
async def test_user_capability_context_not_duplicated_by_handoff() -> None:
    """The refreshed user message replaces the turn's user message — it must
    not grow a second capability block or a second query line.
    """

    async def complete(**kwargs: Any) -> Any:
        messages = kwargs["messages"]
        if not any("## Handoff Context" in str(m.get("content", "")) for m in messages[:1]):
            return SimpleNamespace(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "transition_stage",
                            "arguments": '{"target_stage_id": "search"}',
                        },
                    }
                ],
            )
        user_messages = [m for m in messages if m.get("role") == "user"]
        content = str(user_messages[-1].get("content", ""))
        assert content.count("Capability Context (live this turn)") <= 1, content
        assert content.count("## User Query") == 1, content
        return SimpleNamespace(content="Done in search mode.")

    default_cfg = _chat_mode_config(mode_id="default")
    search_cfg = _chat_mode_config(mode_id="search", satisfies_intents=["research"])
    catalog = {"default": default_cfg, "search": search_cfg}

    hooks = _hooks_with_recorded_user_context(complete)

    with _patched_catalog(catalog):
        result = await run_turn(mode_id="default", query="proceed", hooks=hooks)

    assert result.handoff_to == "search"


def _chat_mode_config(
    *,
    mode_id: str,
    satisfies_intents: list[str] | None = None,
) -> WorkflowModeConfig:
    return WorkflowModeConfig(
        mode_id=mode_id,
        start_stage_id="none",
        mode_kind="chat",
        routable=True,
        satisfies_intents=list(satisfies_intents or []),
    )


def _hooks_with_recorded_user_context(
    complete: Any,
    *,
    recall_enabled: bool = False,
) -> TurnHooks:
    """Hooks that record every assemble_user_context call for assertions."""

    def assemble_user(
        mode_id: str,
        _query: str,
        _pre_turn_briefing: str,
        capability_manifest: Any = None,
    ) -> str:
        del capability_manifest  # rendered by the host in production
        return f"## Capability Context (live this turn)\n\n- mode: {mode_id}\n"

    return TurnHooks(
        models=ModelHooks(complete=complete),
        context=ContextHooks(assemble_user_context=assemble_user),
        recall=RecallHooks(enabled=recall_enabled),
    )


def _patched_catalog(catalog: dict[str, WorkflowModeConfig]):
    from contextlib import ExitStack
    from unittest.mock import patch

    stack = ExitStack()
    stack.enter_context(patch("kernel.turn_runner.get_workflow_catalog", return_value=catalog))
    stack.enter_context(patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=catalog))
    stack.enter_context(
        patch(
            "kernel.turn_runner.get_mode_system_intro",
            lambda mid, _c=catalog: (
                _c[mid].instruction_body if _c.get(mid) and _c[mid].instruction_body else mid
            ),
        )
    )
    stack.enter_context(
        patch(
            "kernel.ipsar_fsm.get_mode_system_intro",
            lambda mid, _c=catalog: (
                _c[mid].instruction_body if _c.get(mid) and _c[mid].instruction_body else mid
            ),
        )
    )
    return stack
