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

import pytest
from kernel.models import (
    CapabilityManifest,
    OperationExecutionContext,
    WorkflowModeConfig,
)


@pytest.fixture(autouse=True)
def _routing_vocab() -> None:
    """Register a representative routing vocabulary for each test.

    Hard cutover (2026-08-11): the kernel has no hardcoded fallbacks.
    A reader raises ``RuntimeError`` when no session vocabulary is wired.
    These tests exercise ``_compute_capability_routing_hint`` which reads
    sequential connectors, producer intents, and consumer intents — all
    must be wired or the test fails on the raise, not on the assertion.
    """
    from kernel.routing_vocabulary import (
        register_routing_vocabulary,
        reset_routing_vocabulary,
    )

    register_routing_vocabulary(
        sequential_connectors=["then", "and then", "after that", "next", "followed by"],
        producer_intents=["research", "gather", "learn", "read", "analyze", "review", "architect"],
        consumer_intents=["code", "write", "implement", "review"],
        resume_phrases=["proceed", "continue", "go ahead"],
    )
    yield
    reset_routing_vocabulary()


def _mode_config(
    *,
    mode_id: str = "default",
    satisfies_intents: list[str] | None = None,
    routing_intent_tags: list[str] | None = None,
    routing_intent_keywords: dict[str, list[str]] | None = None,
    routable: bool = True,
    fork_allowed: bool = True,
    description: str = "",
) -> WorkflowModeConfig:
    """A minimal ``WorkflowModeConfig`` for routing-hint tests.

    ``fork_allowed`` defaults to True so the fork branch is exercisable
    without every test opting in; production modes declare it explicitly.
    """
    return WorkflowModeConfig(
        mode_id=mode_id,
        start_stage_id="none",
        mode_kind="chat",
        routable=routable,
        fork_allowed=fork_allowed,
        description=description,
        satisfies_intents=list(satisfies_intents) if satisfies_intents else [],
        routing_intent_tags=list(routing_intent_tags) if routing_intent_tags else [],
        routing_intent_keywords=dict(routing_intent_keywords) if routing_intent_keywords else {},
    )


def _empty_manifest() -> CapabilityManifest:
    return CapabilityManifest(generated_at="2026-01-01T00:00:00Z", entries=[])


def _full_catalog() -> dict[str, WorkflowModeConfig]:
    """A representative catalog with search, coding, markdown, learn, planner."""
    return {
        "default": _mode_config(mode_id="default"),
        "search": _mode_config(
            mode_id="search",
            satisfies_intents=["research"],
            routing_intent_tags=["research"],
            routing_intent_keywords={"research": ["search", "latest", "news"]},
            description="Web research mode.",
        ),
        "coding": _mode_config(
            mode_id="coding",
            satisfies_intents=["coding", "review"],
            routing_intent_tags=["coding", "review"],
            routing_intent_keywords={
                "coding": ["code", "coding", "codebase"],
                "review": ["review", "audit", "check", "inspect"],
            },
            description="Coding workflow mode.",
        ),
        "markdown": _mode_config(
            mode_id="markdown",
            satisfies_intents=["write"],
            routing_intent_tags=["write"],
            routing_intent_keywords={"write": ["write", "markdown", "document", "report"]},
            description="Markdown document mode.",
        ),
        "remember": _mode_config(
            mode_id="remember",
            satisfies_intents=["remember"],
            routing_intent_tags=["remember"],
            routing_intent_keywords={"remember": ["remember", "note", "capture"]},
            description="Memory capture mode.",
        ),
        "planner": _mode_config(
            mode_id="planner",
            satisfies_intents=["plan"],
            routing_intent_tags=["plan"],
            routing_intent_keywords={"plan": ["plan", "workflow", "multi-step", "chain"]},
            description="Workflow planner mode.",
        ),
    }


# ── Routing hint escalation ────────────────────────────────────────────────


def test_routing_hint_escalates_to_plan_for_sequential_multi_intent() -> None:
    """A sequential query with 2+ gap intents escalates to ``planner``."""
    from kernel.routing_vocabulary import reset_plan_reliability
    from kernel.turn_phases._routing_hint import _compute_capability_routing_hint

    reset_plan_reliability()
    catalog = _full_catalog()
    q = (
        "do a web search for python coding best practices then review the "
        "code for async issues then write a markdown summary"
    )
    with patch("kernel.turn_phases._routing_hint.get_workflow_catalog", return_value=catalog):
        hint = _compute_capability_routing_hint(
            mode_id="default",
            query=q,
            capability_manifest=_empty_manifest(),
            routing_targets=["search", "coding", "markdown", "planner"],
            mode_config=catalog["default"],
        )

    assert hint != ""
    assert "Routing Hint" in hint
    assert "transition_stage" in hint
    assert "planner" in hint
    # Escalation path, not a direct first-step handoff or fork.
    assert "fork:" not in hint
    # Must route to planner, not directly to the first step (search).
    assert "search" not in hint


def test_routing_hint_escalates_to_plan_for_three_plus_intents_non_sequential() -> None:
    """A non-sequential query with 3+ gap intents also escalates to ``planner``."""
    from kernel.routing_vocabulary import reset_plan_reliability
    from kernel.turn_phases._routing_hint import _compute_capability_routing_hint

    reset_plan_reliability()
    catalog = _full_catalog()
    q = "search the latest news remember the key points and write a markdown"
    with patch("kernel.turn_phases._routing_hint.get_workflow_catalog", return_value=catalog):
        hint = _compute_capability_routing_hint(
            mode_id="default",
            query=q,
            capability_manifest=_empty_manifest(),
            routing_targets=["search", "remember", "markdown", "planner"],
            mode_config=catalog["default"],
        )

    assert hint != ""
    assert "planner" in hint
    assert "fork:" not in hint


def test_routing_hint_escalates_to_plan_for_two_intents_sequential() -> None:
    """A 2-intent sequential query escalates to ``planner`` (2 + sequential)."""
    from kernel.routing_vocabulary import reset_plan_reliability
    from kernel.turn_phases._routing_hint import _compute_capability_routing_hint

    reset_plan_reliability()
    catalog = _full_catalog()
    q = "search for the latest rust news then write a markdown summary"
    with patch("kernel.turn_phases._routing_hint.get_workflow_catalog", return_value=catalog):
        hint = _compute_capability_routing_hint(
            mode_id="default",
            query=q,
            capability_manifest=_empty_manifest(),
            routing_targets=["search", "markdown", "planner"],
            mode_config=catalog["default"],
        )

    assert hint != ""
    assert "planner" in hint
    assert "fork:" not in hint


def test_routing_hint_does_not_escalate_to_plan_when_plan_not_routable() -> None:
    """When ``planner`` is not among routing targets, the existing fork logic runs."""
    from kernel.routing_vocabulary import reset_plan_reliability
    from kernel.turn_phases._routing_hint import _compute_capability_routing_hint

    reset_plan_reliability()
    catalog = {
        "remember": _full_catalog()["remember"],
        "markdown": _full_catalog()["markdown"],
        "default": _full_catalog()["default"],
    }
    q = "remember this and write a markdown of it"
    with patch("kernel.turn_phases._routing_hint.get_workflow_catalog", return_value=catalog):
        hint = _compute_capability_routing_hint(
            mode_id="default",
            query=q,
            capability_manifest=_empty_manifest(),
            routing_targets=["remember", "markdown"],
            mode_config=catalog["default"],
        )

    assert hint != ""
    assert "fork" in hint
    assert "planner" not in hint


def test_routing_hint_does_not_escalate_to_plan_for_simple_query() -> None:
    """A simple single-intent query routes directly, no planner escalation."""
    from kernel.routing_vocabulary import reset_plan_reliability
    from kernel.turn_phases._routing_hint import _compute_capability_routing_hint

    reset_plan_reliability()
    catalog = {
        "search": _full_catalog()["search"],
        "planner": _full_catalog()["planner"],
        "default": _full_catalog()["default"],
    }
    q = "search for the latest rust news"
    with patch("kernel.turn_phases._routing_hint.get_workflow_catalog", return_value=catalog):
        hint = _compute_capability_routing_hint(
            mode_id="default",
            query=q,
            capability_manifest=_empty_manifest(),
            routing_targets=["search", "planner"],
            mode_config=catalog["default"],
        )

    assert hint != ""
    assert "search" in hint
    assert "planner" not in hint


# ── discuss mode intent detection (Phase 1 metadata) ───────────────────────


def test_discuss_mode_detects_clarify_intent_from_new_keywords() -> None:
    """The ``discuss`` mode's new ``clarify`` routing intent makes
    clarification-cue queries (ask, unclear, should I, ...) matchable by
    the proactive hint detector. ``discuss`` was re-scoped from
    ``reasoning`` to ``clarify`` on 2026-07-30 so it only punts there when
    the right next step is a question to the user, not a deep analysis
    (which belongs to ``ponder``)."""
    from kernel.turn_phases._routing_hint import _detect_query_intents

    discuss = _mode_config(
        mode_id="discuss",
        satisfies_intents=["clarify"],
        routing_intent_tags=["clarify"],
        routing_intent_keywords={
            "clarify": [
                "clarify",
                "not sure",
                "need to know",
                "what do you mean",
                "ask you",
                "questions about",
                "should i",
                "which option",
                "ambiguous",
                "unclear",
                "can you confirm",
                "help me decide",
            ],
        },
    )
    with patch(
        "kernel.turn_phases._routing_hint.get_workflow_catalog", return_value={"discuss": discuss}
    ):
        assert _detect_query_intents(
            query="I'm not sure which option to pick, can you clarify?",
            routing_targets=["discuss"],
        ) == {"clarify"}
        assert _detect_query_intents(
            query="help me decide between A and B",
            routing_targets=["discuss"],
        ) == {"clarify"}
        assert (
            _detect_query_intents(
                query="analyze the tradeoffs between X and Y",
                routing_targets=["discuss"],
            )
            == set()
        )  # analysis cue no longer matches discuss (now ponder)
        assert (
            _detect_query_intents(
                query="hello world",
                routing_targets=["discuss"],
            )
            == set()
        )


def test_routing_hint_skips_plan_when_model_marked_unreliable() -> None:
    """When the active model has been marked unreliable for planning
    (``plan_workflow`` previously failed for it), the planner-escalation
    branch is skipped so a known-bad path is not re-attempted (the
    2026-07-30 StubCompletion incident)."""
    from kernel.routing_vocabulary import (
        mark_plan_unreliable_for_model,
        reset_plan_reliability,
    )
    from kernel.turn_phases._routing_hint import _compute_capability_routing_hint

    reset_plan_reliability()
    catalog = _full_catalog()
    q = (
        "do a web search for python coding best practices then review the "
        "code for async issues then write a markdown summary"
    )
    mark_plan_unreliable_for_model("glm-5.2:cloud")
    with patch("kernel.turn_phases._routing_hint.get_workflow_catalog", return_value=catalog):
        hint = _compute_capability_routing_hint(
            mode_id="default",
            query=q,
            capability_manifest=_empty_manifest(),
            routing_targets=["search", "coding", "markdown", "planner"],
            mode_config=catalog["default"],
            planner_model_id="glm-5.2:cloud",
        )
    reset_plan_reliability()

    # Planner escalation must be skipped for the unreliable model.
    assert "planner" not in hint or "Routing Hint" not in hint or hint == ""
    # It should NOT hand off to planner.
    assert "target_stage_id `planner`" not in hint


def test_routing_hint_fallback_silences_when_no_clean_cover() -> None:
    """When no single target cleanly covers the gap and the disjoint/fork
    branches do not fire, the hint layer returns empty (silence) instead
    of dumping the whole target list (the 2026-07-30 hallucination-as-
    progress pattern). Constructed so the fork branch refuses (the two
    targets cover the *same* single intent, not a disjoint pair) and
    producer-first does not fire (no producer/consumer mix)."""
    from kernel.routing_vocabulary import reset_plan_reliability
    from kernel.turn_phases._routing_hint import _compute_capability_routing_hint

    reset_plan_reliability()
    # Two targets both declare the SAME intent tag (research). A query
    # matching research leaves a single-intent gap; both cover it, so
    # ``covering_targets`` is non-empty and the single-handoff branch
    # fires — not the fallback. To reach the fallback we need a gap no
    # target covers: declare a target whose keywords do NOT match the
    # query, so detection finds nothing... but then there's no gap either.
    # The realistic fallback trigger is: a gap exists, multiple targets
    # each cover a disjoint subset, but the fork is refused because the
    # union does not cover the whole gap. Build that: three intents,
    # two targets each covering one, third intent uncovered.
    catalog = {
        "default": _full_catalog()["default"],
        "search": _mode_config(
            mode_id="search",
            satisfies_intents=["research"],
            routing_intent_tags=["research"],
            routing_intent_keywords={"research": ["search", "news"]},
        ),
        "remember": _mode_config(
            mode_id="remember",
            satisfies_intents=["remember"],
            routing_intent_tags=["remember"],
            routing_intent_keywords={"remember": ["remember", "note"]},
        ),
        # A third target that covers a *third* intent the query also
        # expresses, so the fork's two-target cover is incomplete.
        "coding": _mode_config(
            mode_id="coding",
            satisfies_intents=["coding"],
            routing_intent_tags=["coding"],
            routing_intent_keywords={"coding": ["code", "coding"]},
        ),
    }
    # Query triggers research + learn + coding (3 intents). Only search +
    # learn + coding together cover all 3, but a 3-way fork with a
    # producer/consumer mix (research is producer, coding is consumer)
    # is refused. No single target covers all 3. The producer-first guard
    # fires for the research/coding mix → returns a producer hint, not
    # the fallback. So this query does NOT reach the fallback either.
    #
    # The fallback is reached only when: gap exists, no single cover, no
    # disjoint fork, no producer/consumer mix, not sequential. The only
    # remaining case is a single uncovered intent with no matching target
    # at all — but then detection returns nothing and there's no gap.
    # Conclusion: with the current branch ordering the fallback (section
    # 6) is effectively unreachable when targets declare intents; it only
    # fires when ``has_gap`` is True but ``detected_intents`` is empty
    # (a gap asserted externally with no detection). Verify that path.
    with patch("kernel.turn_phases._routing_hint.get_workflow_catalog", return_value=catalog):
        hint = _compute_capability_routing_hint(
            mode_id="default",
            query="hello world",
            capability_manifest=_empty_manifest(),
            routing_targets=["search", "remember", "coding"],
            mode_config=catalog["default"],
        )
    # No intents detected from "hello world" → no gap → empty hint.
    assert hint == ""


def test_routing_hint_single_handoff_is_directive() -> None:
    """A single covering target produces a directive hint ('This request
    requires X's tools') rather than the old hedged prose ('Your current
    mode may lack...') that invited the model to override it."""
    from kernel.routing_vocabulary import reset_plan_reliability
    from kernel.turn_phases._routing_hint import _compute_capability_routing_hint

    reset_plan_reliability()
    catalog = {
        "default": _full_catalog()["default"],
        "search": _full_catalog()["search"],
    }
    q = "search for the latest rust news"
    with patch("kernel.turn_phases._routing_hint.get_workflow_catalog", return_value=catalog):
        hint = _compute_capability_routing_hint(
            mode_id="default",
            query=q,
            capability_manifest=_empty_manifest(),
            routing_targets=["search"],
            mode_config=catalog["default"],
        )
    assert hint != ""
    assert "requires" in hint
    assert "may lack" not in hint


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
    from kernel.seams.host_adapter import build_callable_host

    host = None
    if complete_fn is not None:
        host = build_callable_host(
            complete_fn=complete_fn,
            resolve_model_fn=resolve_model_fn
            or (lambda mid: type("M", (), {"model_id": "default"})()),
        )
    return OperationExecutionContext(
        query=query,
        mode_id="plan",
        host=host,
    )


def test_plan_workflow_returns_structured_target() -> None:
    """``plan_workflow.execute`` calls the model and returns the parsed
    ``target_stage_id`` in ``output_json`` and the rationale in
    ``output_text``."""
    from sdk.packs.core.operations.planning.plan_workflow import execute

    payload = {"target_stage_id": "search", "rationale": "research intent detected"}

    async def fake_complete(
        messages,
        model_def,
        max_tokens,
        temperature,
        response_format=None,
        tools=None,
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
    """When the OperationHost seam is missing, the operation returns a
    no-target fallback instead of raising."""
    from sdk.packs.core.operations.planning.plan_workflow import execute

    ctx = _make_ctx(query="search for the latest news", complete_fn=None)
    result = asyncio.run(execute(ctx))
    assert result.output_json["target_stage_id"] == ""
    assert "OperationHost" in result.output_text or "model" in result.output_text.lower()


def test_plan_workflow_retries_when_first_completion_is_empty() -> None:
    """Regression for the 2026-07-30 plan-mode crash.

    When the provider returns empty content on the first structured-output
    call, ``parse_structured_payload`` raises ``EmptyStructuredOutputError``
    (a ``ValueError`` subclass, NOT a ``ValidationError``). The first
    parse's ``except ValidationError`` missed it, so the exception escaped
    ``execute`` and surfaced as
    ``operation 'plan_workflow' failed: Empty structured-output content``
    — the JSON-only retry never ran. The fix widens the catch to include
    ``EmptyStructuredOutputError`` so the retry fires and recovers.
    """
    from sdk.packs.core.operations.planning.plan_workflow import execute

    payload = {"target_stage_id": "search", "rationale": "research intent detected"}
    call_count = {"n": 0}

    async def fake_complete(
        messages,
        model_def,
        max_tokens,
        temperature,
        response_format=None,
        tools=None,
    ):
        call_count["n"] += 1
        # First call returns empty content (the incident). The retry must
        # then fire and return valid JSON.
        if call_count["n"] == 1:
            return _FakeCompletion("")
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

    assert call_count["n"] == 2, "JSON-only retry must fire after empty first content"
    assert result.output_json["target_stage_id"] == "search"
    assert result.output_text == "research intent detected"


def test_plan_workflow_retry_drops_response_format() -> None:
    """Regression for the 2026-07-30 plan-mode retry failure.

    The JSON-only retry must clear ``response_format`` (pass ``None``),
    not re-pin ``PlanWorkflowOutput``. Ollama reasoning models
    (glm-5.2, kimi-k2.6) return EMPTY content when ``response_format`` is
    pinned on a structured-output call, but emit valid JSON when it is
    cleared and the JSON-only instruction is appended. Re-passing
    ``response_format`` on the retry reproduces the empty-content
    condition and the retry fails identically — every plan_workflow
    retry in the incident log returned "Empty structured-output content"
    because ``response_format`` was still set. The declarative runtime
    (kernel/operations/declarative.py) and synthesize_learning both
    drop ``response_format`` on the JSON-only retry; plan_workflow must
    match.
    """
    from sdk.packs.core.operations.planning.plan_workflow import execute

    payload = {"target_stage_id": "search", "rationale": "research intent detected"}
    calls: list[dict[str, Any]] = []

    async def fake_complete(
        messages,
        model_def,
        max_tokens,
        temperature,
        response_format=None,
        tools=None,
    ):
        calls.append({"response_format": response_format, "n": len(calls) + 1})
        # First call (response_format pinned) returns empty — the incident.
        # Retry (response_format=None) returns valid JSON.
        if len(calls) == 1:
            return _FakeCompletion("")
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

    assert len(calls) == 2, "exactly two model calls (first + retry)"
    # First call pins response_format (the structured-output attempt).
    assert calls[0]["response_format"] is not None, "first call must pin response_format"
    # Retry MUST clear response_format — re-pinning reproduces empty content.
    assert calls[1]["response_format"] is None, (
        "JSON-only retry must clear response_format (None), not re-pin it; "
        "re-pinning causes Ollama reasoning models to return empty content again"
    )
    assert result.output_json["target_stage_id"] == "search"


def test_plan_workflow_retries_when_first_call_raises_unsupported() -> None:
    """Regression for the 2026-07-30 complex-prompt planning failure.

    When the resolved model returns ``400 - The requested operation is
    unsupported`` on the structured-output call (observed with
    ``gpt-5.1-codex-max`` via Azure OpenAI — the model can't honor
    ``response_format``), ``plan_workflow`` previously bailed immediately
    with ``rationale: "model error: ..."`` and no retry. But a 400 on
    ``response_format`` is exactly what the JSON-only retry (which clears
    ``response_format``) is designed to recover from. The fix routes call
    failures into the same retry path as parse failures instead of giving
    up. This test asserts the retry fires and recovers a valid target.
    """
    from sdk.packs.core.operations.planning.plan_workflow import execute

    payload = {"target_stage_id": "search", "rationale": "research intent detected"}
    call_count = {"n": 0}

    async def fake_complete(
        messages,
        model_def,
        max_tokens,
        temperature,
        response_format=None,
        tools=None,
    ):
        call_count["n"] += 1
        # First call (response_format pinned) raises the 400 unsupported
        # error observed with gpt-5.1-codex-max. The retry must fire with
        # response_format cleared and return valid JSON.
        if call_count["n"] == 1:
            raise RuntimeError(
                "Error code: 400 - {'error': {'message': 'The requested operation is unsupported.'}}"
            )
        return _FakeCompletion(json.dumps(payload))

    ctx = _make_ctx(
        query="research the latest rust async runtime then draft a summary",
        complete_fn=fake_complete,
    )
    catalog = _full_catalog()
    with (
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=catalog),
        patch(
            "sdk.packs.core.operations.planning.plan_workflow.get_workflow_catalog",
            return_value=catalog,
        ),
    ):
        result = asyncio.run(execute(ctx))

    assert call_count["n"] == 2, "JSON-only retry must fire after a 400 on the first call"
    assert result.output_json["target_stage_id"] == "search"
    assert result.output_text == "research intent detected"


# ── Phase 1: visible multi-step plan (Option B) ─────────────────────────────


def test_plan_workflow_returns_steps_in_output_json() -> None:
    """When the model returns a ``steps`` array, ``execute`` places it in
    ``output_json`` and normalizes ``target_stage_id`` to the first step's
    ``mode_id`` so existing consumers keep working."""
    from sdk.packs.core.operations.planning.plan_workflow import execute

    payload = {
        "steps": [
            {
                "mode_id": "search",
                "description": "Search for the latest rust async runtime news",
                "rationale": "The search mode provides web_search for current info",
            },
            {
                "mode_id": "markdown",
                "description": "Draft a markdown summary of the findings",
                "rationale": "The markdown mode provides document drafting tools",
            },
        ],
        "target_stage_id": "search",
        "rationale": "Research then write",
    }

    async def fake_complete(
        messages,
        model_def,
        max_tokens,
        temperature,
        response_format=None,
        tools=None,
    ):
        return _FakeCompletion(json.dumps(payload))

    ctx = _make_ctx(
        query="research the latest rust async runtime then draft a summary",
        complete_fn=fake_complete,
    )
    catalog = _full_catalog()
    with (
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=catalog),
        patch(
            "sdk.packs.core.operations.planning.plan_workflow.get_workflow_catalog",
            return_value=catalog,
        ),
    ):
        result = asyncio.run(execute(ctx))

    steps = result.output_json.get("steps")
    assert steps is not None, "steps must be present in output_json"
    assert len(steps) == 2
    assert steps[0]["mode_id"] == "search"
    assert steps[1]["mode_id"] == "markdown"
    # target_stage_id normalized to steps[0].mode_id
    assert result.output_json["target_stage_id"] == "search"


def test_plan_workflow_renders_visible_plan_markdown() -> None:
    """When steps is non-empty, ``output_text`` is a rendered markdown plan
    block (not just the rationale) so the turn runner can surface it to
    the user before the first step executes."""
    from sdk.packs.core.operations.planning.plan_workflow import execute

    payload = {
        "steps": [
            {
                "mode_id": "search",
                "description": "Search for the latest rust async runtime news",
                "rationale": "Provides web_search",
            },
            {
                "mode_id": "markdown",
                "description": "Draft a markdown summary",
                "rationale": "Provides document drafting",
            },
        ],
        "target_stage_id": "search",
        "rationale": "Research then write",
    }

    async def fake_complete(
        messages,
        model_def,
        max_tokens,
        temperature,
        response_format=None,
        tools=None,
    ):
        return _FakeCompletion(json.dumps(payload))

    ctx = _make_ctx(
        query="research the latest rust async runtime then draft a summary",
        complete_fn=fake_complete,
    )
    catalog = _full_catalog()
    with (
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=catalog),
        patch(
            "sdk.packs.core.operations.planning.plan_workflow.get_workflow_catalog",
            return_value=catalog,
        ),
    ):
        result = asyncio.run(execute(ctx))

    assert "## Plan" in result.output_text, "output_text must contain the plan header"
    assert "search" in result.output_text
    assert "markdown" in result.output_text
    assert "Search for the latest" in result.output_text
    assert "Draft a markdown summary" in result.output_text


def test_plan_workflow_no_steps_falls_back_to_rationale_in_output_text() -> None:
    """When the model returns no ``steps`` (weak-model fallback or
    single-target legacy shape), ``output_text`` is the rationale (no plan
    block) — same as the pre-steps behavior. No regression."""
    from sdk.packs.core.operations.planning.plan_workflow import execute

    payload = {"target_stage_id": "search", "rationale": "research intent detected"}

    async def fake_complete(
        messages,
        model_def,
        max_tokens,
        temperature,
        response_format=None,
        tools=None,
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

    assert result.output_json["steps"] == [], "empty steps list when model returns none"
    assert "## Plan" not in result.output_text, "no plan block when steps is empty"
    assert result.output_text == "research intent detected"


def test_plan_workflow_empty_steps_and_empty_target_is_no_match() -> None:
    """When the model returns empty steps AND empty target_stage_id, the
    rationale (missing-capability message) is surfaced in output_text and
    steps is an empty list — the no-match case that tells the framework to
    surface the rationale to the user."""
    from sdk.packs.core.operations.planning.plan_workflow import execute

    payload = {
        "steps": [],
        "target_stage_id": "",
        "rationale": "No installed mode provides a write/markdown capability",
    }

    async def fake_complete(
        messages,
        model_def,
        max_tokens,
        temperature,
        response_format=None,
        tools=None,
    ):
        return _FakeCompletion(json.dumps(payload))

    ctx = _make_ctx(query="draft a document about quantum computing", complete_fn=fake_complete)
    catalog = _full_catalog()
    with (
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=catalog),
        patch(
            "sdk.packs.core.operations.planning.plan_workflow.get_workflow_catalog",
            return_value=catalog,
        ),
    ):
        result = asyncio.run(execute(ctx))

    assert result.output_json["target_stage_id"] == ""
    assert result.output_json["steps"] == []
    assert "write/markdown" in result.output_text
    assert "## Plan" not in result.output_text


# ── Regression: 2026-08-11 ambiguous-intent yield on "review the code" ────


def _persona_catalog() -> dict[str, WorkflowModeConfig]:
    """A persona-shaped catalog mirroring the lisa-pack-personas front
    matter after the 2026-08-11 fix.

    Before the fix, executor's keyword bucket was keyed ``execute:`` while
    its ``routing_intent_tags`` were ``[code, write, implement]``, and
    reviewer's ``review:`` bucket included the bare keyword ``read``. Both
    produced false intents from the query "review the code for
    understanding. The /docs folder and the README.md…":
    ``execute`` (uncoverable — no mode declares it) and ``review`` (also
    matched "README.md" via the bare ``read`` keyword). The gap
    {review, execute} had no single covering target, no fork, no
    producer-first → the ``AMBIGUOUS_INTENT`` sentinel yielded to the user
    at iteration 0. The turn ran zero model calls.

    After the fix: executor's keyword keys are ⊆ its tags, reviewer's
    ``review:`` bucket dropped the bare ``read`` keyword, the ambiguity
    sentinel is gone, and the query detects only ``{review}`` → single
    handoff to ``reviewer``.
    """
    return {
        "default": _mode_config(mode_id="default"),
        "reviewer": _mode_config(
            mode_id="reviewer",
            satisfies_intents=["read", "review", "architect", "analyze"],
            routing_intent_tags=["read", "review", "architect", "analyze"],
            routing_intent_keywords={
                "review": ["review", "analyze", "critique"],
                "architect": ["design", "architect", "structure"],
            },
            description="Analyzes code/architecture and produces design documents.",
        ),
        "executor": _mode_config(
            mode_id="executor",
            satisfies_intents=["code", "write", "implement"],
            routing_intent_tags=["code", "write", "implement"],
            routing_intent_keywords={
                "implement": ["implement", "build", "create", "fix", "refactor", "patch"],
                "code": ["write code", "write the code", "refactor", "patch"],
                "write": ["write code", "write the code"],
            },
            description="Mutates the workspace, writes code.",
        ),
        "researcher": _mode_config(
            mode_id="researcher",
            satisfies_intents=["research", "gather", "learn"],
            routing_intent_tags=["research", "gather", "learn"],
            routing_intent_keywords={"research": ["research", "gather", "learn", "investigate"]},
            description="Gathers context and synthesizes findings.",
        ),
        "planner": _mode_config(
            mode_id="planner",
            satisfies_intents=["plan"],
            routing_intent_tags=["plan"],
            routing_intent_keywords={"plan": ["plan", "workflow", "multi-step", "chain"]},
            description="Orchestrates multi-step tasks.",
        ),
    }


def test_review_code_query_routes_to_reviewer_not_ambiguity() -> None:
    """Regression for the 2026-08-11 turn that yielded ``AMBIGUOUS_INTENT``
    on a simple code-review request and ran zero model calls.

    The query "start by reviewing the code for understanding. The /docs
    folder and the README.md…" must detect only ``{review}`` (not the
    phantom ``execute`` or a false ``read`` from "README.md"), route to
    ``reviewer`` as a single handoff, and produce no ambiguity sentinel
    (the sentinel was removed; the hint returns ``""`` on genuine
    ambiguity so the model self-routes via ``transition_stage``).
    """
    from kernel.routing_vocabulary import reset_plan_reliability
    from kernel.turn_phases._routing_hint import _compute_capability_routing_hint

    reset_plan_reliability()
    catalog = _persona_catalog()
    q = (
        "start by reviewing the code for understanding. The /docs folder "
        "and the README.md are a good entry point"
    )
    with patch("kernel.turn_phases._routing_hint.get_workflow_catalog", return_value=catalog):
        hint = _compute_capability_routing_hint(
            mode_id="default",
            query=q,
            capability_manifest=_empty_manifest(),
            routing_targets=["reviewer", "executor", "researcher", "planner"],
            mode_config=catalog["default"],
        )

    # No ambiguity sentinel — that branch is gone.
    assert not hint.startswith("AMBIGUOUS_INTENT:")
    # Single handoff to reviewer (the only target covering the review intent).
    assert hint != ""
    assert "Routing Hint" in hint
    assert "reviewer" in hint
    assert "transition_stage" in hint
    # The phantom execute intent must not appear — executor's keyword
    # keys are now ⊆ its tags, so "code" maps to the ``code`` tag, which
    # executor covers. But the query says "reviewing the code", not
    # "write code" — so no ``code`` intent fires. Only ``review``.
    assert "executor" not in hint
