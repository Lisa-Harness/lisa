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
by writing it into the thread, where ``kernel.workflow_view`` derives it on
every later turn; it is resumable with ``/resume`` and retirable with
``/workflow conclude``. When the model returns no ``steps`` (weak model, old
shape), the single ``target_stage_id`` handoff is used with no declared
workflow — there is no compiled fallback.

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

from kernel.ipsar_fsm import get_workflow_catalog
from kernel.models import OperationExecutionContext, OperationResult
from kernel.operations.structured_output import (
    EmptyStructuredOutputError,
    completion_content,
    json_only_retry_messages,
    parse_structured_payload,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)

__all__ = ["PlanStep", "PlanWorkflowOutput", "execute"]

# Default for ``resources.packs.configuration.lisa-pack-core.plan_max_tokens``.
# Reasoning models spend this budget thinking before emitting the small JSON
# plan; 8192 truncated glm-5.3 at the length limit (2026-10-01 e2e run 8).
_PLAN_MAX_TOKENS = 32768


def _plan_max_tokens(ctx: OperationExecutionContext) -> int:
    """Return the session's planning budget, rejecting malformed values."""
    raw = (ctx.pack_config or {}).get("plan_max_tokens", _PLAN_MAX_TOKENS)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        raise ValueError(f"lisa-pack-core plan_max_tokens must be a positive integer, got {raw!r}")
    return raw


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
    step does, a rationale for why that mode handles it, and an optional
    ``write_capability`` — the capability id whose write/external_write
    receipt completes the step. The turn runner maps each step to a
    ``WorkflowStep`` (via ``_step_id_for``); the WorkflowView matches the
    step's completion against ``write_capability`` receipts in later
    thread entries, closing the asymmetry with the kernel's turn-level
    write-receipt contract.

    ``write_capability`` is None for read/gather steps (search, review
    prep) — they complete on the next handoff, not a write receipt. For
    write steps (draft, execute, commit), it MUST be a capability in the
    mode's write contract (``mode_completion_effects``); the planner is
    given each mode's write capabilities in the catalog summary.
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
    write_capability: str | None = Field(
        default=None,
        description=(
            "The capability id whose write receipt completes this step "
            "(e.g. 'markdown_forge_document'). MUST be one of the mode's "
            "listed write_capabilities. Set to null for read/gather steps "
            "(search, review prep) that complete on handoff rather than "
            "a write receipt. Binding a deliverable here ensures the "
            "plan cannot self-declare done with prose — the receipt is "
            "required."
        ),
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
    carries the single target as before and no workflow is declared.

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
            "when steps is non-empty, or a single mode_id for the "
            "single-target case. Empty when no mode matches. Computed "
            "from steps[0].mode_id by the caller when steps is populated; "
            "set directly only when the model returns no steps "
            "(weak-model fallback)."
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
2. **If a ``## Recent Conversation`` block is present**, read it first.
   It contains the recent thread excerpt. When the prior turn already
   presented a plan or diagnosis, carry it forward — do not re-synthesize
   a generic plan that loses the detail already presented to the user.
   The step descriptions should reflect the specific work discussed, not
   a generic "execute the previously presented plan" placeholder.
3. Identify which intents the query expresses by matching against each
   mode's routing_intent_keywords.
4. Check for sequential connectors. A query containing any of these
   tokens (matched case-insensitively as whole phrases) is a sequential
   chain:
{connectors_block}
   If present, the query is a sequential chain.
5. Build the `steps` array — one entry per detected step, in execution
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

6. For each step, decide whether it is a **write step** or a **read step**:
   - A **write step** produces a tangible deliverable (a document, a
     commit, a learned concept). Set `write_capability` to the capability
     id from the mode's `write_capabilities` list that this step must
     invoke to produce its deliverable. The framework tracks the step as
     incomplete until a write receipt for that capability appears in the
     thread — no dense-content escape, no false complete. This is how the
     plan guarantees reads-then-writes: a write step without its receipt
     stays incomplete and the workflow cannot finish.
   - A **read step** gathers information (search, review prep) and
     produces no deliverable. Set `write_capability` to null. The step
     completes when the mode hands off to the next step; the reads and
     prose are already in the thread.
   As a rule of thumb: a plan should be **reads then writes** — gather
   first (read steps), then produce (write steps). A plan with a write
   step that produces no receipt stays incomplete; the kernel will route
   back or surface the stall rather than silently accept prose.

7. Set `target_stage_id` to the first step's `mode_id` (the framework
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


def _build_mode_catalog_summary(*, include_tools: bool = False) -> str:
    """Render the live routable mode catalog as a compact text block.

    Each routable mode is listed with its mode_id, intent tags, the
    keyword substrings that signal it, and its write-effect capabilities
    (from ``mode_completion_effects``). The write capabilities tell the
    planner which capability id to bind as a step's ``write_capability`` —
    the deliverable whose receipt completes the step. Non-routable modes
    are omitted; the ``default`` mode is included for conversational
    fallback.

    When ``include_tools`` is True, each mode also lists its full
    ``available_tools`` (the union of ``allowed_tool_ids`` and
    ``allowed_operation_ids``). This is off by default to avoid context
    bloat: most modes only need to know intent coverage + write
    capabilities for planning. A mode that orchestrates routing without
    seeing the workspace (e.g. ``planner``) opts in via the
    ``needs_full_mode_catalog`` front-matter flag so it can see that
    ``reviewer`` has ``coding_find_files`` / ``coding_read_file`` and
    route a "read the codebase" step correctly instead of hallucinating
    that no mode has filesystem tools.
    """

    from kernel.operations.registry import mode_completion_effects

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
        # Available tools: the full allowed_tool_ids + allowed_operation_ids
        # union. Only emitted when the calling mode opts in via
        # ``needs_full_mode_catalog`` — context-bloat guard for the
        # common case where the planner only needs intent coverage + write
        # capabilities.
        tools_line = ""
        if include_tools:
            available_tools = sorted(
                set(
                    list(getattr(mode_config, "allowed_tool_ids", []) or [])
                    + list(getattr(mode_config, "allowed_operation_ids", []) or [])
                )
            )
            if available_tools:
                tools_line = f"\n  available_tools: {', '.join(available_tools)}"
        # Write-effect capabilities: the deliverables the mode can produce.
        # The planner binds one as a write step's ``write_capability``.
        try:
            write_caps = sorted(mode_completion_effects(mode_id))
        except Exception:  # pylint: disable=broad-except
            write_caps = []
        write_caps_line = f"\n  write_capabilities: {', '.join(write_caps)}" if write_caps else ""
        lines.append(
            f"- **{mode_id}** (intents: {tags})\n{keywords_text}"
            + tools_line
            + write_caps_line
            + (f"\n  {desc_line}" if desc_line else "")
        )
    return "\n".join(lines) if lines else "(no routable modes available)"


def _build_sequential_connectors_block() -> str:
    """Render the active sequential-connector vocabulary as a bullet list.

    The list comes from :func:`kernel.routing_vocabulary.get_sequential_connectors`,
    which the host boot path populates from the session plan's
    ``routing.sequential_connectors``. The kernel has no hardcoded
    fallback (the 2026-08-11 cutover): the session defines the list or
    boot fails. Rendering the live list here keeps the planner's
    instruction in sync with the kernel's routing-hint generator — the
    two sites share one vocabulary source.
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

    Each step emits a deliverable line after the description:
      - ``_writes: {capability_id}_`` for write steps (the receipt that
        completes the step)
      - ``_reads_`` for read/gather steps (complete on handoff)
    The WorkflowView parses these to bind step completion to the
    deliverable contract.

    Returns an empty string when ``parsed.steps`` is empty (the no-match
    case or the weak-model single-target fallback) so the caller can skip
    prepending a plan block that has no content.
    """
    if not parsed.steps:
        return ""
    lines: list[str] = ["## Plan", ""]
    for idx, step in enumerate(parsed.steps, start=1):
        lines.append(f"{idx}. **{step.mode_id}** — {step.description}")
        deliverable_line = (
            f"   _writes: {step.write_capability}_" if step.write_capability else "   _reads_"
        )
        lines.append(deliverable_line)
        if step.rationale:
            lines.append(f"   _{step.rationale}_")
    lines.append("")
    return "\n".join(lines)


def _calling_mode_needs_full_catalog(mode_id: str) -> bool:
    """Return True when the calling mode opts in to the full tool catalog.

    Reads the ``needs_full_mode_catalog`` front-matter flag from the
    calling mode's ``WorkflowModeConfig``. Default False: most modes that
    run ``plan_workflow`` only need intent coverage + write capabilities
    (the common case). A mode that orchestrates routing without its own
    workspace tools sets the flag to True so the catalog summary includes
    each target mode's ``available_tools`` — letting it see that
    ``reviewer`` has ``coding_find_files`` and route a "read the codebase"
    step correctly instead of hallucinating that no mode can.
    """

    if not mode_id:
        return False
    mode = get_workflow_catalog().get(mode_id)
    if mode is None:
        return False
    return bool(mode.extra_front_matter.get("needs_full_mode_catalog", False))


async def _structured_plan_call(
    ctx: OperationExecutionContext,
    messages: list[dict[str, Any]],
    model_def: Any,
    plan_max_tokens: int,
) -> PlanWorkflowOutput | None:
    """Run the structured planning call, falling back to the JSON-only retry.

    Returns the parsed plan, or ``None`` when neither the structured call
    nor the JSON-only retry produced a parseable plan. A 400 "unsupported"
    on ``response_format`` is EXACTLY what the JSON-only retry (which
    clears ``response_format``) is designed to recover from — the
    2026-07-30 incident: gpt-5.1-codex-max returned `400 - The requested
    operation is unsupported` on the structured call, and plan_workflow
    bailed immediately with no retry, even though the retry path would
    have succeeded. Call failures route into the same retry path as parse
    failures instead of giving up.
    """
    first_completion: Any | None = None
    try:
        first_completion = await ctx.host.complete(
            messages=messages,
            model_def=model_def,
            max_tokens=plan_max_tokens,
            temperature=0.1,
            response_format=PlanWorkflowOutput,
        )
        content = completion_content(first_completion)
        try:
            return cast(PlanWorkflowOutput, parse_structured_payload(PlanWorkflowOutput, content))
        except (ValidationError, EmptyStructuredOutputError):
            logger.info(
                "plan_workflow structured parse failed; retrying with JSON-only instruction"
            )
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(
            "plan_workflow structured call failed (%s); retrying with JSON-only instruction",
            exc,
        )
        first_completion = None

    return await _json_only_retry(
        ctx,
        messages,
        model_def,
        plan_max_tokens,
        first_completion,
    )


def _validate_step_write_capabilities(parsed: PlanWorkflowOutput) -> None:
    """Validate write_capability assignments against each mode's write contract.

    A write step's capability must be in its mode's write contract
    (``mode_completion_effects``). An invalid assignment is downgraded to
    None (read step) rather than failing the plan — the kernel will block at
    runtime if the model then fails to produce any receipt, which is the
    honest signal.
    """
    from kernel.operations.registry import mode_completion_effects

    for idx, step in enumerate(parsed.steps):
        if not step.write_capability:
            continue
        try:
            allowed = mode_completion_effects(step.mode_id)
        except Exception:  # pylint: disable=broad-except
            allowed = set()
        if step.write_capability not in allowed:
            logger.warning(
                "plan_workflow: step mode=%s write_capability=%s not in "
                "mode's write contract %r; downgrading to read step",
                step.mode_id,
                step.write_capability,
                sorted(allowed),
            )
            parsed.steps[idx] = step.model_copy(update={"write_capability": None})


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

    if ctx.host is None:
        return OperationResult(
            output_text=(
                "plan_workflow requires the OperationHost seam to "
                "run; falling back to no routing target."
            ),
            output_json={"target_stage_id": "", "rationale": "no model wiring"},
        )

    catalog_summary = _build_mode_catalog_summary(
        # The calling mode opts in to the full catalog (with per-mode
        # ``available_tools``) via the ``needs_full_mode_catalog``
        # front-matter flag. Default off to avoid context bloat; ``planner``
        # enables it so it can see that ``reviewer`` has ``coding_find_files``
        # and route a "read the codebase" step correctly.
        include_tools=_calling_mode_needs_full_catalog(ctx.mode_id)
    )
    connectors_block = _build_sequential_connectors_block()
    system_prompt = _PLANNING_INSTRUCTION.format(
        catalog_summary=catalog_summary,
        connectors_block=connectors_block,
    )

    preferred_model_id = ctx.preferred_model_id or ctx.model_id or "default"
    model_def = ctx.host.resolve_model(preferred_model_id)
    if model_def is None:
        model_def = ctx.host.resolve_model("default")
    if model_def is None:
        return OperationResult(
            output_text="plan_workflow could not resolve a model for planning.",
            output_json={"target_stage_id": "", "rationale": "model unresolved"},
        )

    # Build the user message. When the host wired a conversation excerpt
    # (the recent thread from ``assemble_user_context``), include it so
    # the planner can carry forward a plan already presented in the prior
    # turn instead of re-synthesizing a generic one from the raw query
    # alone (the 2026-08-09 plan-amnesia incident: the planner received
    # only ``ctx.query`` and produced a 1-step "execute the previously
    # presented plan" with no visibility into what that plan was).
    _excerpt = (ctx.conversation_excerpt_markdown or "").strip()
    user_content = f"## User Query\n\n{query}"
    if _excerpt:
        user_content = f"## Recent Conversation\n\n{_excerpt}\n\n{user_content}"

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    plan_max_tokens = _plan_max_tokens(ctx)

    parsed = await _structured_plan_call(ctx, messages, model_def, plan_max_tokens)
    if parsed is None:
        return OperationResult(
            output_text="plan_workflow could not parse structured output after retry",
            output_json={"target_stage_id": "", "rationale": "model error"},
        )

    # Normalize: when the model returned steps, target_stage_id is the
    # first step's mode_id (the initial handoff target). When the model
    # returned no steps (weak-model fallback or single-target legacy
    # shape), target_stage_id carries the single target as-is. This keeps
    # existing consumers that read only target_stage_id working while the
    # turn runner reads steps for the visible plan + WorkflowInstance.
    _validate_step_write_capabilities(parsed)
    normalized_target = parsed.steps[0].mode_id if parsed.steps else parsed.target_stage_id or ""

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
                    "write_capability": s.write_capability,
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
        retry_completion = await ctx.host.complete(
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
