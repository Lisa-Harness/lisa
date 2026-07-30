"""Declarative workflow planner for the ``plan`` action mode.

Analyzes a user query against the routable mode catalog's intent keywords,
detects sequential connectors, and returns a visible multi-step plan.

This is the escalation path for complex multi-step queries: the cheap
regex-based ``_detect_query_intents`` in ``kernel.turn_runner`` handles
simple single/double-intent requests with zero model calls; this operation
runs one cheap model call to disambiguate complex or ambiguous requests
and return a structured plan (a list of steps, each naming a target mode).

The plan is surfaced to the user as a markdown block before the first step
executes (auto-proceed — no forced pause). The turn runner persists the plan
as a ``WorkflowInstance`` with ``spawned_from="plan_workflow_op"`` so it
survives across turns and is resumable with ``/proceed``. When the model
returns no ``steps`` (weak model, old shape), the turn runner falls back
to the deterministic ``compile_steps_from_routing_targets`` path.

The sequential-connector vocabulary is shared with the kernel's routing-hint
generator via :mod:`kernel.routing_vocabulary`. The host boot path injects
the session plan's ``routing.sequential_connectors`` list once at startup;
both this planner and ``kernel.turn_phases._routing_hint`` read from the
same registry so they cannot drift.

The mode catalog summary is built in-process from
``kernel.ipsar_fsm.get_workflow_catalog`` so the planner always sees the
live routable mode set. The model never sees stale or hardcoded mode lists.
"""

from __future__ import annotations

import logging
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from kernel.ipsar_fsm import get_workflow_catalog
from kernel.models import OperationExecutionContext, OperationResult
from kernel.operations.structured_output import (
    EmptyStructuredOutputError,
    completion_content,
    json_only_retry_messages,
    parse_structured_payload,
)

logger = logging.getLogger(__name__)

__all__ = ["PlanStep", "PlanWorkflowOutput", "execute"]


def _compact_json_schema(schema: dict[str, Any]) -> None:
    """Strip non-essential schema metadata for structured-output providers."""
    schema.pop("description", None)
    schema.pop("title", None)
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return
    for property_schema in properties.values():
        if not isinstance(property_schema, dict):
            continue
        property_schema.pop("description", None)
        property_schema.pop("title", None)


class PlanStep(BaseModel):
    """One step in a multi-step workflow plan.

    A step names a target mode, a human-readable description of what the
    step does, and a rationale for why that mode handles it. The turn
    runner maps each step to a ``WorkflowStep`` (via ``_step_id_for``) and
    persists the sequence as a ``WorkflowInstance`` so the plan survives
    across turns and is resumable with ``/proceed``.
    """

    model_config = ConfigDict(json_schema_extra=_compact_json_schema)

    mode_id: str = Field(
        description=(
            "The routable mode_id that handles this step. Must be one of "
            "the Available Modes listed above."
        )
    )
    description: str = Field(
        description=(
            "A short human-readable description of what this step does "
            "(e.g. 'Search for the latest rust async runtime news'). "
            "Shown to the user in the visible plan block."
        )
    )
    rationale: str = Field(
        description=(
            "Why this mode handles this step (e.g. 'The search mode "
            "provides web_search for current information'). Shown to the "
            "user in the visible plan block."
        )
    )


class PlanWorkflowOutput(BaseModel):
    """Structured output from the ``plan_workflow`` operation.

    The ``steps`` list is the primary output: one entry per detected step,
    in execution order. The turn runner renders the steps as a visible
    markdown plan block, persists them as a ``WorkflowInstance``, and
    drives the chain from the instance (the first step's ``mode_id``
    becomes the initial handoff target).

    ``target_stage_id`` and ``rationale`` are backward-compatible fields:
    when ``steps`` is non-empty, ``target_stage_id`` is the first step's
    ``mode_id`` and ``rationale`` is a one-line summary. Existing consumers
    that read only ``target_stage_id`` keep working. When the model returns
    no ``steps`` (weak model, single-target case), ``target_stage_id``
    carries the single target as before and the turn runner falls back to
    the deterministic ``compile_steps_from_routing_targets`` path.

    When no installed mode covers the request, return empty ``steps`` AND
    empty ``target_stage_id``; the ``rationale`` MUST then name the missing
    capability (quoted from the user's words) so the framework can surface
    it to the user.
    """

    model_config = ConfigDict(json_schema_extra=_compact_json_schema)

    steps: list[PlanStep] = Field(
        default_factory=list,
        description=(
            "The planned steps in execution order. One entry per detected "
            "step. Each step's mode_id must be a routable mode from the "
            "Available Modes list. For a single-step request, return one "
            "entry. For no match, return an empty list."
        ),
    )
    target_stage_id: str = Field(
        default="",
        description=(
            "Backward-compatible single target: the first step's mode_id "
            "when steps is non-empty, or a single mode_id / "
            "'fork:mode1,mode2,...' for the legacy single-target case. "
            "Empty when no mode matches. Computed from steps[0].mode_id "
            "by the caller when steps is populated; set directly only when "
            "the model returns no steps (weak-model fallback)."
        ),
    )
    rationale: str = Field(
        default="",
        description=(
            "One-line summary of the routing decision, OR the missing-"
            "capability rationale when no mode matches. When steps is "
            "non-empty this is a summary of the plan; the per-step "
            "rationales live on each PlanStep. When target_stage_id is "
            "empty, this MUST name the specific capability the request "
            "needs that no installed mode provides (quote the unmatched "
            "intent or verb from the user's query), NOT a generic 'no "
            "intent keywords' excuse. Example: 'No installed mode provides "
            "a write/markdown capability; the user asked to draft a "
            "document.' This rationale is shown to the user when no "
            "handoff occurs."
        ),
    )


# Static planning instruction. The live mode catalog summary is appended at
# runtime so the model always plans against the currently-installed routable
# modes, not a hardcoded list.
_PLANNING_INSTRUCTION = """\
# Plan Workflow

You are a workflow planner. Your job is to analyze the user's query and
produce a visible multi-step plan that will be shown to the user before
execution.

## Available Modes

The following modes are available as routing targets. Each declares intent
tags and keyword patterns that signal when it should be used.

{catalog_summary}

## Instructions

1. Read the user's query carefully.
2. Identify which intents the query expresses by matching against each
   mode's routing_intent_keywords.
3. Check for sequential connectors. A query containing any of these
   tokens (matched case-insensitively as whole phrases) is a sequential
   chain:
{connectors_block}
   If present, the query is a sequential chain.
4. Build the `steps` array — one entry per detected step, in execution
   order. Each step names a `mode_id` from the Available Modes list, a
   short human-readable `description` of what the step does, and a
   `rationale` for why that mode handles it.

   - **Single step**: If one mode covers all detected intents, return one
     step with that mode's `mode_id`.
   - **Sequential chain**: If the query is sequential, return one step per
     detected intent, in the order the intents appear in the query. The
     framework will chain the steps automatically; you only declare the
     sequence, not the handoff mechanism.
   - **Fork fan-out**: If multiple modes each cover a disjoint subset of
     intents AND the query is NOT sequential, return one step per mode.
     The framework will fan them out in parallel.

5. Set `target_stage_id` to the first step's `mode_id` (the framework
   reads this for the initial handoff). Set `rationale` to a one-line
   summary of the plan. When `steps` is non-empty, `target_stage_id` and
   `rationale` are derived from `steps[0]` — set them consistently.

   - **No match**: If NO installed mode covers the request's main intent
     (the user asks for a capability no available mode's keywords match),
     return an EMPTY `steps` array AND an empty `target_stage_id`. The
     `rationale` MUST then name the specific capability that is missing,
     quoted from the user's own words — e.g. "No installed mode provides
     a write/markdown capability; the user asked to draft a document." Do
     NOT return a generic "no intent keywords detected" excuse; do NOT
     route to `default` for a capability request. The empty target tells
     the framework to surface your rationale to the user so they can
     install the missing pack.

## Output Contract

Return one JSON object matching the declared schema. The `steps` array
is the primary output: one entry per step, in execution order. Each
step's `mode_id` must be a routable mode_id from the Available Modes
list. The `target_stage_id` must be the first step's `mode_id` when
`steps` is non-empty, or an empty string when no installed mode covers
the request. Do not include any other text outside the JSON object.
"""


def _build_mode_catalog_summary() -> str:
    """Render the live routable mode catalog as a compact text block.

    Each routable mode is listed with its mode_id, intent tags, and the
    keyword substrings that signal it. Non-routable modes are omitted —
    the planner only routes to routable targets. The ``default`` mode is
    included so the planner can fall back to it for conversational
    requests that need no specialized mode.
    """

    catalog = get_workflow_catalog()
    lines: list[str] = []
    for mode_id, mode_config in sorted(catalog.items()):
        if not mode_config.routable:
            continue
        tags = ", ".join(mode_config.routing_intent_tags) or "(none)"
        keywords_block: list[str] = []
        for tag, keywords in mode_config.routing_intent_keywords.items():
            if keywords:
                joined = ", ".join(keywords)
                keywords_block.append(f"  - {tag}: {joined}")
        keywords_text = "\n".join(keywords_block) if keywords_block else "  (no keywords)"
        description = (mode_config.description or "").strip().splitlines()[0:1]
        desc_line = description[0] if description else ""
        lines.append(
            f"- **{mode_id}** (intents: {tags})\n{keywords_text}"
            + (f"\n  {desc_line}" if desc_line else "")
        )
    return "\n".join(lines) if lines else "(no routable modes available)"


def _build_sequential_connectors_block() -> str:
    """Render the active sequential-connector vocabulary as a bullet list.

    The list comes from :func:`kernel.routing_vocabulary.get_sequential_connectors`,
    which the host boot path populates from the session plan's
    ``routing.sequential_connectors`` (falling back to the historical
    built-in set when no session vocabulary is wired). Rendering the live
    list here keeps the planner's instruction in sync with the kernel's
    routing-hint generator — the two sites share one vocabulary source.
    """
    from kernel.routing_vocabulary import get_sequential_connectors

    connectors = get_sequential_connectors()
    if not connectors:
        return "   (none — sequential detection disabled)"
    # Strip the surrounding padding the registry normalizes onto each token
    # so the model sees clean phrases ("then", "based on") rather than
    # the padded internal form (" then ").
    items = [f"   - ``{token.strip()}``" for token in connectors]
    return "\n".join(items)


def _render_plan_markdown(parsed: PlanWorkflowOutput) -> str:
    """Render a parsed plan as a human-readable markdown block.

    This is the visible plan artifact surfaced to the user before the
    first step executes (auto-proceed — no forced pause). The block is
    prepended to the turn's ``response_text`` by the turn runner so the
    user sees the plan at the top of the response, followed by the first
    step's output.

    Returns an empty string when ``parsed.steps`` is empty (the no-match
    case or the weak-model single-target fallback) so the caller can skip
    prepending a plan block that has no content.
    """
    if not parsed.steps:
        return ""
    lines: list[str] = ["## Plan", ""]
    for idx, step in enumerate(parsed.steps, start=1):
        lines.append(f"{idx}. **{step.mode_id}** — {step.description}")
        if step.rationale:
            lines.append(f"   _{step.rationale}_")
    lines.append("")
    return "\n".join(lines)


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Run the workflow planner and return the visible plan.

    Builds the live mode catalog summary, assembles the system prompt, and
    calls the wired completion function with ``PlanWorkflowOutput`` as the
    structured response format. The returned ``steps`` list and
    ``target_stage_id`` are placed in ``output_json``; the rendered plan
    markdown block is placed in ``output_text`` so the turn runner can
    surface it to the user before the first step executes.

    When ``steps`` is non-empty, ``target_stage_id`` is normalized to
    ``steps[0].mode_id`` so existing consumers that read only
    ``target_stage_id`` keep working. When the model returns no ``steps``
    (weak-model fallback), ``target_stage_id`` carries the single target
    as before and ``output_text`` is the rationale (no plan block).
    """

    query = (ctx.query or "").strip()
    if not query:
        return OperationResult(
            output_text="plan_workflow received an empty query; nothing to plan.",
            output_json={"target_stage_id": "", "rationale": "empty query"},
        )

    if ctx.complete_fn is None or not callable(ctx.resolve_model_fn):
        return OperationResult(
            output_text=(
                "plan_workflow requires complete_fn and resolve_model_fn to "
                "run; falling back to no routing target."
            ),
            output_json={"target_stage_id": "", "rationale": "no model wiring"},
        )

    catalog_summary = _build_mode_catalog_summary()
    connectors_block = _build_sequential_connectors_block()
    system_prompt = _PLANNING_INSTRUCTION.format(
        catalog_summary=catalog_summary,
        connectors_block=connectors_block,
    )

    preferred_model_id = ctx.preferred_model_id or ctx.model_id or "default"
    model_def = ctx.resolve_model_fn(preferred_model_id)
    if model_def is None:
        model_def = ctx.resolve_model_fn("default")
    if model_def is None:
        return OperationResult(
            output_text="plan_workflow could not resolve a model for planning.",
            output_json={"target_stage_id": "", "rationale": "model unresolved"},
        )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"## User Query\n\n{query}"},
    ]

    # 8192, not 4096: reasoning models routed through Ollama (kimi-k2.6,
    # glm-5.2) spend their token budget on internal reasoning before
    # emitting visible content. 4096 was calibrated on a SIMPLE probe prompt
    # (836 completion_tokens, finish_reason=stop). Complex multi-step prompts
    # need more reasoning room — the 2026-07-30 complex-prompt incident showed
    # plan_workflow returning empty content on the retry at 4096 because the
    # model exhausted its budget on reasoning. The visible JSON answer is
    # tiny (~2 fields); the budget is for the model's thinking room.
    plan_max_tokens = 8192

    first_completion: Any | None = None
    first_call_failed = False
    try:
        first_completion = await ctx.complete_fn(
            messages=messages,
            model_def=model_def,
            max_tokens=plan_max_tokens,
            temperature=0.1,
            response_format=PlanWorkflowOutput,
        )
    except Exception as exc:  # pylint: disable=broad-except
        # A 400 "unsupported" on response_format is EXACTLY what the
        # JSON-only retry (which clears response_format) is designed to
        # recover from. The 2026-07-30 incident: gpt-5.1-codex-max returned
        # `400 - The requested operation is unsupported` on the structured
        # call, and plan_workflow bailed immediately with no retry — even
        # though the retry path drops response_format and would have
        # succeeded. Route call failures into the same retry path as parse
        # failures instead of giving up.
        logger.warning(
            "plan_workflow structured call failed (%s); retrying with JSON-only instruction",
            exc,
        )
        first_call_failed = True

    if not first_call_failed:
        content = completion_content(first_completion)
        parsed: PlanWorkflowOutput | None = None
        try:
            parsed = cast(PlanWorkflowOutput, parse_structured_payload(PlanWorkflowOutput, content))
        except (ValidationError, EmptyStructuredOutputError):
            logger.info(
                "plan_workflow structured parse failed; retrying with JSON-only instruction"
            )
            parsed = await _json_only_retry(
                ctx,
                messages,
                model_def,
                plan_max_tokens,
                first_completion,
            )
            if parsed is None:
                return OperationResult(
                    output_text="plan_workflow could not parse structured output after retry",
                    output_json={"target_stage_id": "", "rationale": "parse failure"},
                )
    else:
        parsed = await _json_only_retry(
            ctx,
            messages,
            model_def,
            plan_max_tokens,
            None,
        )
        if parsed is None:
            return OperationResult(
                output_text="plan_workflow model call failed and JSON-only retry did not recover",
                output_json={"target_stage_id": "", "rationale": "model error"},
            )

    # Normalize: when the model returned steps, target_stage_id is the
    # first step's mode_id (the initial handoff target). When the model
    # returned no steps (weak-model fallback or single-target legacy
    # shape), target_stage_id carries the single target as-is. This keeps
    # existing consumers that read only target_stage_id working while the
    # turn runner reads steps for the visible plan + WorkflowInstance.
    if parsed.steps:
        normalized_target = parsed.steps[0].mode_id
    else:
        normalized_target = parsed.target_stage_id or ""

    plan_block = _render_plan_markdown(parsed)
    # When steps is non-empty, output_text is the rendered plan block
    # (what the user sees). When steps is empty (no-match or fallback),
    # output_text is the rationale (the missing-capability message or
    # the single-target rationale) — same as the pre-steps behavior.
    output_text = plan_block if plan_block else parsed.rationale

    return OperationResult(
        output_text=output_text,
        output_json={
            "target_stage_id": normalized_target,
            "rationale": parsed.rationale,
            "steps": [
                {
                    "mode_id": s.mode_id,
                    "description": s.description,
                    "rationale": s.rationale,
                }
                for s in parsed.steps
            ],
        },
    )


async def _json_only_retry(
    ctx: OperationExecutionContext,
    messages: list[dict[str, Any]],
    model_def: Any,
    plan_max_tokens: int,
    first_completion: Any | None,
) -> PlanWorkflowOutput | None:
    """Run the JSON-only retry and return parsed output, or None on failure.

    Centralizes the retry so both the parse-failure and call-failure paths
    share one implementation. Clears ``response_format`` (the retry is a
    JSON-only *instruction*, not a re-pinned structured call) and logs
    ``finish_reason``/``completion_tokens`` diagnostics so truncation is
    visible instead of the 2026-07-30 silent-empty failure.
    """
    # Log diagnostics from the first attempt when available, so a length
    # truncation on the first call is visible in the host log alongside the
    # retry failure (the 2026-07-30 complex-prompt incident produced empty
    # content with no finish_reason logged, leaving the root cause
    # indistinguishable from a genuine provider-empty response).
    if first_completion is not None:
        logger.info(
            "plan_workflow first attempt diagnostics: finish_reason=%s completion_tokens=%s",
            getattr(first_completion, "finish_reason", None),
            getattr(first_completion, "completion_tokens", None),
        )
    try:
        # Clear response_format on the retry. The first attempt failed to
        # produce parseable structured output — often because the provider
        # (e.g. Ollama reasoning models: glm-5.2, kimi-k2.6) returns EMPTY
        # content when response_format is pinned but emits valid JSON when
        # it is cleared and the JSON-only instruction is appended. A 400
        # "unsupported" on the structured call (gpt-5.1-codex-max) is the
        # same class of failure — the model can't honor response_format, so
        # drop it. Re-passing response_format reproduces the failure.
        # This mirrors the declarative runtime
        # (kernel/operations/declarative.py) and synthesize_learning.
        retry_completion = await ctx.complete_fn(
            messages=json_only_retry_messages(messages, PlanWorkflowOutput),
            model_def=model_def,
            # Same reasoning-model budget as the first attempt. The retry
            # adds one user message (JSON-only instruction), so reasoning
            # models need the same headroom or they truncate again with
            # empty content.
            max_tokens=plan_max_tokens,
            temperature=0.1,
            response_format=None,
        )
        logger.info(
            "plan_workflow retry diagnostics: finish_reason=%s completion_tokens=%s",
            getattr(retry_completion, "finish_reason", None),
            getattr(retry_completion, "completion_tokens", None),
        )
        content = completion_content(retry_completion)
        parsed = cast(PlanWorkflowOutput, parse_structured_payload(PlanWorkflowOutput, content))
    except Exception as retry_exc:  # pylint: disable=broad-except
        logger.warning("plan_workflow JSON-only retry failed: %s", retry_exc)
        return None
    return parsed