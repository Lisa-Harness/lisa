"""Tests for the ``plan`` action mode and ``plan_workflow`` operation.

Covers the plan-mode routing escalation in
``kernel.turn_runner._compute_capability_routing_hint`` and the
``plan_workflow`` runtime operation in
``sdk.packs.core.operations.planning.plan_workflow``.

These tests stub the workflow catalog so they do not require the
``lisa-catalog`` MCP server or the external pack submodules — they run
purely in-process. This keeps them independent of the
``kernel/tests/conftest_catalog.py`` autouse fixture that spawns the
real catalog (which requires the external packs installed as importable
packages).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import patch

from kernel.models import (
    CapabilityManifest,
    OperationExecutionContext,
    WorkflowModeConfig,
)


def _mode_config(
    *,
    mode_id: str = "default",
    satisfies_intents: list[str] | None = None,
    routing_intent_tags: list[str] | None = None,
    routing_intent_keywords: dict[str, list[str]] | None = None,
    routable: bool = True,
    description: str = "",
) -> WorkflowModeConfig:
    """A minimal ``WorkflowModeConfig`` for routing-hint tests."""
    return WorkflowModeConfig(
        mode_id=mode_id,
        start_stage_id="none",
        mode_kind="chat",
        routable=routable,
        description=description,
        satisfies_intents=list(satisfies_intents) if satisfies_intents else [],
        routing_intent_tags=list(routing_intent_tags) if routing_intent_tags else [],
        routing_intent_keywords=dict(routing_intent_keywords) if routing_intent_keywords else {},
    )


def _empty_manifest() -> CapabilityManifest:
    return CapabilityManifest(generated_at="2026-01-01T00:00:00Z", entries=[])


def _full_catalog() -> dict[str, WorkflowModeConfig]:
    """A representative catalog with search, coding, markdown, learn, plan."""
    return {
        "default": _mode_config(mode_id="default"),
        "search": _mode_config(
            mode_id="search", satisfies_intents=["research"],
            routing_intent_tags=["research"],
            routing_intent_keywords={"research": ["search", "latest", "news"]},
            description="Web research mode.",
        ),
        "coding": _mode_config(
            mode_id="coding", satisfies_intents=["coding", "review"],
            routing_intent_tags=["coding", "review"],
            routing_intent_keywords={
                "coding": ["code", "coding", "codebase"],
                "review": ["review", "audit", "check", "inspect"],
            },
            description="Coding workflow mode.",
        ),
        "markdown": _mode_config(
            mode_id="markdown", satisfies_intents=["write"],
            routing_intent_tags=["write"],
            routing_intent_keywords={"write": ["write", "markdown", "document", "report"]},
            description="Markdown document mode.",
        ),
        "learn": _mode_config(
            mode_id="learn", satisfies_intents=["learn"],
            routing_intent_tags=["learn"],
            routing_intent_keywords={"learn": ["remember", "note", "capture"]},
            description="Memory capture mode.",
        ),
        "plan": _mode_config(
            mode_id="plan", satisfies_intents=["plan"],
            routing_intent_tags=["plan"],
            routing_intent_keywords={"plan": ["plan", "workflow", "multi-step", "chain"]},
            description="Workflow planner mode.",
        ),
    }


# ── Routing hint escalation ────────────────────────────────────────────────


def test_routing_hint_escalates_to_plan_for_sequential_multi_intent() -> None:
    """A sequential query with 2+ gap intents escalates to ``plan``."""
    from kernel.turn_runner import _compute_capability_routing_hint

    catalog = _full_catalog()
    q = (
        "do a web search for python coding best practices then review the "
        "code for async issues then write a markdown summary"
    )
    with patch("kernel.turn_runner.get_workflow_catalog", return_value=catalog):
        hint = _compute_capability_routing_hint(
            mode_id="default", query=q, capability_manifest=_empty_manifest(),
            routing_targets=["search", "coding", "markdown", "plan"],
            mode_config=catalog["default"],
        )

    assert hint != ""
    assert "Routing Hint" in hint
    assert "transition_stage" in hint
    assert "plan" in hint
    # Escalation path, not a direct first-step handoff or fork.
    assert "fork:" not in hint
    # Must route to plan, not directly to the first step (search).
    assert "search" not in hint


def test_routing_hint_escalates_to_plan_for_three_plus_intents_non_sequential() -> None:
    """A non-sequential query with 3+ gap intents also escalates to ``plan``."""
    from kernel.turn_runner import _compute_capability_routing_hint

    catalog = _full_catalog()
    q = "search the latest news remember the key points and write a markdown"
    with patch("kernel.turn_runner.get_workflow_catalog", return_value=catalog):
        hint = _compute_capability_routing_hint(
            mode_id="default", query=q, capability_manifest=_empty_manifest(),
            routing_targets=["search", "learn", "markdown", "plan"],
            mode_config=catalog["default"],
        )

    assert hint != ""
    assert "plan" in hint
    assert "fork:" not in hint


def test_routing_hint_escalates_to_plan_for_two_intents_sequential() -> None:
    """A 2-intent sequential query escalates to ``plan`` (2 + sequential)."""
    from kernel.turn_runner import _compute_capability_routing_hint

    catalog = _full_catalog()
    q = "search for the latest rust news then write a markdown summary"
    with patch("kernel.turn_runner.get_workflow_catalog", return_value=catalog):
        hint = _compute_capability_routing_hint(
            mode_id="default", query=q, capability_manifest=_empty_manifest(),
            routing_targets=["search", "markdown", "plan"],
            mode_config=catalog["default"],
        )

    assert hint != ""
    assert "plan" in hint
    assert "fork:" not in hint


def test_routing_hint_does_not_escalate_to_plan_when_plan_not_routable() -> None:
    """When ``plan`` is not among routing targets, the existing fork logic runs."""
    from kernel.turn_runner import _compute_capability_routing_hint

    catalog = {
        "learn": _full_catalog()["learn"],
        "markdown": _full_catalog()["markdown"],
        "default": _full_catalog()["default"],
    }
    q = "remember this and write a markdown of it"
    with patch("kernel.turn_runner.get_workflow_catalog", return_value=catalog):
        hint = _compute_capability_routing_hint(
            mode_id="default", query=q, capability_manifest=_empty_manifest(),
            routing_targets=["learn", "markdown"],
            mode_config=catalog["default"],
        )

    assert hint != ""
    assert "fork:" in hint
    assert "plan" not in hint


def test_routing_hint_does_not_escalate_to_plan_for_simple_query() -> None:
    """A simple single-intent query routes directly, no plan escalation."""
    from kernel.turn_runner import _compute_capability_routing_hint

    catalog = {
        "search": _full_catalog()["search"],
        "plan": _full_catalog()["plan"],
        "default": _full_catalog()["default"],
    }
    q = "search for the latest rust news"
    with patch("kernel.turn_runner.get_workflow_catalog", return_value=catalog):
        hint = _compute_capability_routing_hint(
            mode_id="default", query=q, capability_manifest=_empty_manifest(),
            routing_targets=["search", "plan"],
            mode_config=catalog["default"],
        )

    assert hint != ""
    assert "search" in hint
    assert "plan" not in hint


# ── discuss mode intent detection (Phase 1 metadata) ───────────────────────


def test_discuss_mode_detects_reasoning_intent_from_new_keywords() -> None:
    """The ``discuss`` mode's new ``routing_intent_keywords`` make
    reasoning-cue queries matchable by the proactive hint detector."""
    from kernel.turn_runner import _detect_query_intents

    discuss = _mode_config(
        mode_id="discuss", satisfies_intents=["reasoning"],
        routing_intent_tags=["reasoning"],
        routing_intent_keywords={
            "reasoning": [
                "analyze", "reason", "compare", "evaluate", "discuss",
                "debate", "think", "consider", "weigh", "pros", "cons",
                "tradeoff", "swot",
            ],
        },
    )
    with patch("kernel.turn_runner.get_workflow_catalog", return_value={"discuss": discuss}):
        assert _detect_query_intents(
            query="analyze the tradeoffs between X and Y",
            routing_targets=["discuss"],
        ) == {"reasoning"}
        assert _detect_query_intents(
            query="do a SWOT analysis of our position",
            routing_targets=["discuss"],
        ) == {"reasoning"}
        assert _detect_query_intents(
            query="hello world", routing_targets=["discuss"],
        ) == set()


# ── plan_workflow operation ────────────────────────────────────────────────


class _FakeCompletion:
    def __init__(self, content: str) -> None:
        self.content = content
        self.tool_calls: list[Any] = []


def _make_ctx(
    *,
    query: str,
    complete_fn: Any | None = None,
    resolve_model_fn: Any | None = None,
) -> OperationExecutionContext:
    return OperationExecutionContext(
        query=query,
        mode_id="plan",
        complete_fn=complete_fn,
        resolve_model_fn=resolve_model_fn or (lambda mid: type("M", (), {"model_id": "default"})()),
    )


def test_plan_workflow_returns_structured_target() -> None:
    """``plan_workflow.execute`` calls the model and returns the parsed
    ``target_stage_id`` in ``output_json`` and the rationale in
    ``output_text``."""
    from sdk.packs.core.operations.planning.plan_workflow import execute

    payload = {"target_stage_id": "search", "rationale": "research intent detected"}

    async def fake_complete(
        messages, model_def, max_tokens, temperature, response_format=None, tools=None,
    ):
        return _FakeCompletion(json.dumps(payload))

    ctx = _make_ctx(query="search for the latest rust news", complete_fn=fake_complete)
    catalog = _full_catalog()
    with (
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=catalog),
        patch(
            "sdk.packs.core.operations.planning.plan_workflow.get_workflow_catalog",
            return_value=catalog,
        ),
    ):
        result = asyncio.run(execute(ctx))

    assert result.output_json["target_stage_id"] == "search"
    assert result.output_text == "research intent detected"


def test_plan_workflow_handles_empty_query() -> None:
    """An empty query returns a no-op result without calling the model."""
    from sdk.packs.core.operations.planning.plan_workflow import execute

    ctx = _make_ctx(query="")
    result = asyncio.run(execute(ctx))
    assert "empty" in result.output_text.lower()
    assert result.output_json["target_stage_id"] == ""


def test_plan_workflow_handles_missing_model_wiring() -> None:
    """When ``complete_fn`` is missing, the operation returns a no-target
    fallback instead of raising."""
    from sdk.packs.core.operations.planning.plan_workflow import execute

    ctx = _make_ctx(query="search for the latest news", complete_fn=None)
    result = asyncio.run(execute(ctx))
    assert result.output_json["target_stage_id"] == ""
    assert "complete_fn" in result.output_text or "model" in result.output_text.lower()