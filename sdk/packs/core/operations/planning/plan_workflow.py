"""Declarative workflow planner for the ``plan`` action mode.

Analyzes a user query against the routable mode catalog's intent keywords,
detects sequential connectors, and returns a ``transition_stage`` target
(single handoff, fork fan-out, or the first step of a sequential chain).

This is the escalation path for complex multi-step queries: the cheap
regex-based ``_detect_query_intents`` in ``kernel.turn_runner`` handles
simple single/double-intent requests with zero model calls; this operation
runs one cheap model call to disambiguate complex or ambiguous requests
and return a structured routing target.

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
    json_only_retry_messages,
    parse_structured_payload,
)

logger = logging.getLogger(__name__)

__all__ = ["PlanWorkflowOutput", "execute"]


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


class PlanWorkflowOutput(BaseModel):
    """Structured output from the ``plan_workflow`` operation.

    The ``target_stage_id`` is consumed by the FSM's ``transition_stage``
    interception: the ``plan`` action mode runs this operation as its entry
    action, and the returned target is fed directly to the
    handoff / fork / sequential dispatch by the model's subsequent
    ``transition_stage`` tool call.
    """

    model_config = ConfigDict(json_schema_extra=_compact_json_schema)

    target_stage_id: str = Field(
        description=(
            "The transition_stage target: a mode_id for a single handoff, "
            "'fork:mode1,mode2,...' for a parallel fan-out, or the first "
            "mode_id in a sequential chain."
        )
    )
    rationale: str = Field(
        description="Brief explanation of why this target was chosen.",
    )


# Static planning instruction. The live mode catalog summary is appended at
# runtime so the model always plans against the currently-installed routable
# modes, not a hardcoded list.
_PLANNING_INSTRUCTION = """\
# Plan Workflow

You are a workflow planner. Your job is to analyze the user's query and
determine the best routing target.

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
4. Determine the target:

   - **Single handoff**: If one mode covers all detected intents, return
     its `mode_id` as `target_stage_id`.
   - **Fork fan-out**: If multiple modes each cover a disjoint subset of
     intents AND the query is NOT sequential, return
     `fork:mode1,mode2,...` as `target_stage_id`.
   - **Sequential chain**: If the query is sequential, return the
     `mode_id` of the FIRST step as `target_stage_id`. The framework
     will chain the remaining steps automatically.

5. Provide a brief rationale explaining your choice.

## Output Contract

Return one JSON object matching the declared schema. The
`target_stage_id` must be a single routable mode_id, a `fork:...`
fan-out string, or the first mode_id of a sequential chain. Do not
include any other text outside the JSON object.
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


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Run the workflow planner and return the routing target.

    Builds the live mode catalog summary, assembles the system prompt, and
    calls the wired completion function with ``PlanWorkflowOutput`` as the
    structured response format. The returned ``target_stage_id`` is placed
    in ``output_json`` (and ``rationale`` in ``output_text``) so the
    ``plan`` mode's entry-action result is consumed by the subsequent
    ``transition_stage`` tool call.
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

    try:
        completion = await ctx.complete_fn(
            messages=messages,
            model_def=model_def,
            max_tokens=500,
            temperature=0.1,
            response_format=PlanWorkflowOutput,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("plan_workflow model call failed: %s", exc)
        return OperationResult(
            output_text=f"plan_workflow model call failed: {exc}",
            output_json={"target_stage_id": "", "rationale": f"model error: {exc}"},
        )

    content = getattr(completion, "content", "") or str(completion)
    try:
        parsed = cast(PlanWorkflowOutput, parse_structured_payload(PlanWorkflowOutput, content))
    except ValidationError:
        logger.info("plan_workflow structured parse failed; retrying with JSON-only instruction")
        try:
            completion = await ctx.complete_fn(
                messages=json_only_retry_messages(messages, PlanWorkflowOutput),
                model_def=model_def,
                max_tokens=500,
                temperature=0.1,
                response_format=PlanWorkflowOutput,
            )
            content = getattr(completion, "content", "") or str(completion)
            parsed = cast(PlanWorkflowOutput, parse_structured_payload(PlanWorkflowOutput, content))
        except Exception as retry_exc:  # pylint: disable=broad-except
            logger.warning("plan_workflow JSON-only retry failed: %s", retry_exc)
            return OperationResult(
                output_text=f"plan_workflow could not parse structured output: {retry_exc}",
                output_json={"target_stage_id": "", "rationale": "parse failure"},
            )

    return OperationResult(
        output_text=parsed.rationale,
        output_json={
            "target_stage_id": parsed.target_stage_id,
            "rationale": parsed.rationale,
        },
    )