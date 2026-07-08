"""Workshop action-mode entry operation.

This operation plans a small panel, curates a shared and per-panel context bundle,
delegates each panel into its own workflow session, and synthesizes one answer
back to the caller.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import cast
from uuid import uuid4

from pydantic import BaseModel, Field

from kernel.ipsar_fsm import get_workflow_catalog
from kernel.models import (
    OperationExecutionContext,
    OperationResult,
    SurfaceContext,
    UnifiedState,
    WorkflowModeConfig,
    WorkshopArtifactRefs,
    WorkshopBudgetConfig,
    WorkshopContextBundle,
    WorkshopExecutionSummary,
    WorkshopListStyle,
    WorkshopPanelContext,
    WorkshopPanelModeConfig,
    WorkshopPanelSpec,
    WorkshopPlan,
    WorkshopRenderConfig,
)
from kernel.operations.structured_output import run_structured_completion_with_retry
from lisa_sdk.workshop_execution import (
    WorkshopChildExecutionRequest,
    WorkshopExecutionParentContext,
    WorkshopExecutionService,
)

logger = logging.getLogger(__name__)

# Deterministic fallbacks used only when a mode has no authored workshop_panels.
# The model alias is a manifest designation (resolved by the host's model
# manifest), not a hard-coded model ID. Hosts must register a "reasoning"
# designation in their model manifest for workshop panels to resolve.
_PANEL_MODE_ID = "thinkabout"
_DEFAULT_PANEL_MODEL_ALIAS = "reasoning"
_PLANNER_PROMPT_FALLBACK = (
    "Use your mode's native analytical style to evaluate the user's problem. "
    "Deliver one panel-quality answer that is concrete, evidence-aware, "
    "and explicit about tradeoffs."
)
_SYNTHESIS_PROMPT_FALLBACK = (
    "You are the workshop coordinator. Synthesize panel outputs into one answer. "
    "Resolve disagreements, surface converging signal, and finish with concrete next moves."
)


class WorkshopSynthesis(BaseModel):
    """Compact final synthesis returned after all child panels complete."""

    verdict: str
    agreements: list[str] = Field(default_factory=list, max_length=4)
    tensions: list[str] = Field(default_factory=list, max_length=4)
    next_steps: list[str] = Field(default_factory=list, max_length=4)


def _slugify_panel_id(value: str, fallback: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return normalized or fallback


def _short_title(query: str) -> str:
    trimmed = " ".join(query.split())
    if len(trimmed) <= 72:
        return trimmed
    return f"{trimmed[:69].rstrip()}..."


def _mode_title_prefix(mode: WorkflowModeConfig | None) -> str:
    if mode is None:
        return "Workshop"

    raw_mode_id = str(mode.mode_id or "").strip()
    if not raw_mode_id or raw_mode_id == "workshop":
        return "Workshop"

    return " ".join(part.capitalize() for part in raw_mode_id.replace("_", " ").split())


@lru_cache(maxsize=8)
def _load_prompt_file(path_text: str) -> str:
    path = Path(path_text)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    return text.strip()


def _active_mode_config(ctx: OperationExecutionContext) -> WorkflowModeConfig | None:
    try:
        mode = get_workflow_catalog().get(ctx.mode_id)
    except Exception:  # pylint: disable=broad-except
        return None
    if isinstance(mode, WorkflowModeConfig):
        return mode
    return None


def _resolved_workshop_rendering(mode: WorkflowModeConfig | None) -> WorkshopRenderConfig:
    if mode is None:
        return WorkshopRenderConfig()
    return mode.workshop_rendering.model_copy(deep=True)


def _mode_content_root(mode: WorkflowModeConfig | None) -> Path | None:
    mode_markdown_path = str(getattr(mode, "mode_markdown_path", "") or "").strip()
    if not mode_markdown_path:
        return None
    return Path(mode_markdown_path).resolve().parent


def _load_mode_prompt(
    mode: WorkflowModeConfig | None,
    *,
    configured_file: str,
    default_file: str,
    fallback: str,
) -> str:
    mode_root = _mode_content_root(mode)
    if mode_root is None:
        return fallback

    if configured_file.strip():
        authored = _load_prompt_file(str((mode_root / configured_file.strip()).resolve()))
        return authored or fallback

    authored = _load_prompt_file(str((mode_root / default_file).resolve()))
    return authored or fallback


def _planner_prompt(mode: WorkflowModeConfig | None) -> str:
    return _load_mode_prompt(
        mode,
        configured_file=str(getattr(mode, "workshop_planner_prompt_file", "") or ""),
        default_file="planner.md",
        fallback=_PLANNER_PROMPT_FALLBACK,
    )


def _synthesis_prompt(mode: WorkflowModeConfig | None) -> str:
    return _load_mode_prompt(
        mode,
        configured_file=str(getattr(mode, "workshop_synthesis_prompt_file", "") or ""),
        default_file="synthesis.md",
        fallback=_SYNTHESIS_PROMPT_FALLBACK,
    )


def _deterministic_default_panels() -> list[WorkshopPanelModeConfig]:
    return [
        WorkshopPanelModeConfig(
            panel_id="analysis",
            title="Strongest Case",
            mode_id=_PANEL_MODE_ID,
            model_alias=_DEFAULT_PANEL_MODEL_ALIAS,
            query_prefix="Build the strongest operator-grade case for this proposal.",
        ),
        WorkshopPanelModeConfig(
            panel_id="challenge",
            title="Risk Review",
            mode_id=_PANEL_MODE_ID,
            model_alias=_DEFAULT_PANEL_MODEL_ALIAS,
            query_prefix="Stress-test this proposal and identify the strongest risks.",
        ),
    ]


def _configured_panel_modes(ctx: OperationExecutionContext) -> list[WorkshopPanelModeConfig]:
    params_panels = ctx.params.get("workshop_panels") if isinstance(ctx.params, dict) else None
    if isinstance(params_panels, list):
        parsed: list[WorkshopPanelModeConfig] = []
        for item in params_panels:
            if not isinstance(item, dict):
                continue
            try:
                parsed.append(WorkshopPanelModeConfig.model_validate(item))
            except Exception:  # pylint: disable=broad-except
                continue
        if parsed:
            return parsed

    mode = _active_mode_config(ctx)
    if mode and mode.workshop_panels:
        return [panel.model_copy(deep=True) for panel in mode.workshop_panels]
    return _deterministic_default_panels()


def _resolved_workshop_budget(ctx: OperationExecutionContext) -> WorkshopBudgetConfig:
    params_budget = ctx.params.get("workshop_budget") if isinstance(ctx.params, dict) else None
    if isinstance(params_budget, dict):
        try:
            return WorkshopBudgetConfig.model_validate(params_budget)
        except Exception:  # pylint: disable=broad-except
            pass

    mode = _active_mode_config(ctx)
    if mode is not None:
        return mode.workshop_budget.model_copy(deep=True)
    return WorkshopBudgetConfig()


def _panel_query(
    user_query: str,
    panel: WorkshopPanelModeConfig,
    mode_root: Path | None,
) -> str:
    prompt_file = panel.query_prompt_file.strip()
    prefix = panel.query_prefix.strip()
    if prompt_file and mode_root is not None:
        file_prompt = _load_prompt_file(str((mode_root / prompt_file).resolve()))
        if file_prompt:
            prefix = file_prompt
    if not prefix:
        return user_query
    return f"{prefix}\n\n{user_query}"


def _build_plan_from_roster(ctx: OperationExecutionContext, query: str) -> WorkshopPlan:
    workshop_id = str(uuid4())
    mode = _active_mode_config(ctx)
    mode_root = _mode_content_root(mode)
    rendering = _resolved_workshop_rendering(mode)
    title_prefix = rendering.title_prefix.strip() or _mode_title_prefix(mode)
    panel_specs: list[WorkshopPanelSpec] = []
    used_panel_ids: set[str] = set()
    for index, panel in enumerate(_configured_panel_modes(ctx), start=1):
        panel_id = _slugify_panel_id(panel.panel_id or panel.mode_id, fallback=f"panel-{index}")
        if panel_id in used_panel_ids:
            panel_id = f"{panel_id}-{index}"
        used_panel_ids.add(panel_id)
        panel_specs.append(
            WorkshopPanelSpec(
                panel_id=panel_id,
                title=panel.title.strip() or panel.mode_id,
                mode_id=panel.mode_id,
                query=_panel_query(query, panel, mode_root),
                model_alias=panel.model_alias.strip() or _DEFAULT_PANEL_MODEL_ALIAS,
                panel_kind="analysis",
            )
        )

    if not panel_specs:
        for panel in _deterministic_default_panels():
            panel_specs.append(
                WorkshopPanelSpec(
                    panel_id=panel.panel_id,
                    title=panel.title,
                    mode_id=panel.mode_id,
                    query=_panel_query(query, panel, mode_root),
                    model_alias=panel.model_alias.strip() or _DEFAULT_PANEL_MODEL_ALIAS,
                    panel_kind="analysis",
                )
            )

    return WorkshopPlan(
        workshop_id=workshop_id,
        query=query,
        title=(
            title_prefix
            if not rendering.include_query_in_title
            else f"{title_prefix}: {_short_title(query)}"
        ),
        panel_specs=panel_specs,
    )


def _completion_payload_text(result: object) -> str:
    content = getattr(result, "content", None)
    if isinstance(content, str) and content.strip():
        return content
    text = getattr(result, "text", None)
    if isinstance(text, str):
        return text
    return ""


def _is_reasoning_plan(plan: WorkshopPlan) -> bool:
    """Return whether any panel targets a reasoning-capable model.

    The model alias is a manifest designation (e.g. "reasoning") or a concrete
    model id. A panel is reasoning-capable when its alias is the "reasoning"
    designation or when the resolved model definition reports
    ``reasoning_model=True``. The designation check is deterministic and does
    not require a model resolution round-trip.
    """

    for panel in plan.panel_specs:
        alias = (panel.model_alias or "").strip().lower()
        if alias == "reasoning":
            return True
    return False


def _resolve_state(ctx: OperationExecutionContext) -> UnifiedState | None:
    if ctx.state_provider is None:
        return None
    state = ctx.state_provider()
    if state is None:
        return None
    if isinstance(state, UnifiedState):
        return state
    if hasattr(state, "compiled_at") and hasattr(state, "context_evidence_catalog"):
        return cast(UnifiedState, state)
    return None


def _surface_context(ctx: OperationExecutionContext) -> SurfaceContext:
    return SurfaceContext(
        surface_type=ctx.surface_type,
        surface_id=ctx.surface_id,
        memory_scope=ctx.memory_scope,
    )


def _render_panel_user_block(
    base_user_block: str,
    bundle: WorkshopContextBundle,
    panel: WorkshopPanelContext,
) -> str:
    sections: list[str] = []
    if base_user_block.strip():
        sections.append(base_user_block.strip())
    if bundle.shared_context_markdown.strip():
        sections.append(f"## Workshop Shared Brief\n\n{bundle.shared_context_markdown.strip()}")
    if panel.context_markdown.strip():
        sections.append(
            "## Workshop Panel Brief "
            f"({panel.title or panel.panel_id})\n\n"
            f"{panel.context_markdown.strip()}"
        )
    return "\n\n".join(section for section in sections if section)


def _resolve_child_model(ctx: OperationExecutionContext, panel: WorkshopPanelContext) -> object | None:
    if ctx.resolve_model_fn is None:
        return None
    # Candidates are tried in priority order. The panel's authored model
    # alias wins, then the parent turn's model id, then the pack default
    # designation. ``resolve_model_fn`` may raise ValueError for unknown model
    # ids (e.g. a model id that is not in the host manifest); catch and fall
    # through to the next candidate instead of failing the whole workshop.
    for candidate in (panel.model_alias, ctx.model_id, _DEFAULT_PANEL_MODEL_ALIAS):
        if not candidate:
            continue
        try:
            resolved = ctx.resolve_model_fn(candidate)
        except ValueError:
            continue
        if resolved is not None:
            return cast(object, resolved)
    return None


def _build_child_requests(
    ctx: OperationExecutionContext,
    bundle: WorkshopContextBundle,
    planner_subprompt: str,
) -> list[WorkshopChildExecutionRequest] | None:
    requests: list[WorkshopChildExecutionRequest] = []
    for panel in bundle.panel_contexts:
        model_def = _resolve_child_model(ctx, panel)
        if model_def is None:
            return None
        requests.append(
            WorkshopChildExecutionRequest(
                panel_context=panel,
                model_def=model_def,
                user_block=_render_panel_user_block(ctx.user_block, bundle, panel),
                clean_message=(
                    f"{panel.query}\n\n{planner_subprompt}" if planner_subprompt else panel.query
                ),
                timeout_ms=45_000,
                persist_session_state=False,
                run_post_turn_operations=False,
                include_model_badge=False,
                delegation_depth=1,
            )
        )
    return requests


def _fallback_synthesis_text(
    plan: WorkshopPlan,
    execution_summary: WorkshopExecutionSummary,
    rendering: WorkshopRenderConfig,
) -> str:
    lines = [f"## {plan.title}"]
    if execution_summary.failed_children or execution_summary.timed_out_children:
        lines.append(rendering.incomplete_panels_note.strip())
    for child_result in execution_summary.results:
        header = f"### {child_result.panel_id}"
        if child_result.status != "ok":
            lines.extend(
                [
                    header,
                    f"- Status: {child_result.status}",
                    f"- Detail: {child_result.error_message}",
                ]
            )
            continue
        lines.extend([header, child_result.output_text.strip() or "No output returned."])
    return "\n\n".join(line for line in lines if line)


def _render_list_items(items: list[str], style: WorkshopListStyle) -> list[str]:
    if style == "numbered":
        return [f"1. {item}" for item in items]
    return [f"- {item}" for item in items]


def _append_rendered_section(
    lines: list[str],
    heading: str,
    items: list[str],
    style: WorkshopListStyle,
) -> None:
    if not items:
        return

    normalized_heading = heading.strip()
    if normalized_heading:
        lines.append(f"### {normalized_heading}")
    lines.extend(_render_list_items(items, style))


async def _synthesize_workshop(
    ctx: OperationExecutionContext,
    mode: WorkflowModeConfig | None,
    plan: WorkshopPlan,
    bundle: WorkshopContextBundle,
    execution_summary: WorkshopExecutionSummary,
    workshop_budget: WorkshopBudgetConfig,
) -> tuple[str, dict[str, object]]:
    rendering = _resolved_workshop_rendering(mode)
    successful_results = [result for result in execution_summary.results if result.status == "ok"]
    if not successful_results:
        return _fallback_synthesis_text(plan, execution_summary, rendering), {}

    if ctx.complete_fn is None or ctx.resolve_model_fn is None:
        return _fallback_synthesis_text(plan, execution_summary, rendering), {}

    synthesis_model = ctx.resolve_model_fn(ctx.model_id or "default")
    if synthesis_model is None:
        return _fallback_synthesis_text(plan, execution_summary, rendering), {}

    panel_report_lines: list[str] = []
    for result in execution_summary.results:
        panel_report_lines.append(f"## Panel {result.panel_id}\n")
        panel_report_lines.append(f"status: {result.status}")
        if result.output_text.strip():
            panel_report_lines.append(result.output_text.strip())
        if result.error_message.strip():
            panel_report_lines.append(f"error: {result.error_message.strip()}")
        panel_report_lines.append("")

    synthesis_messages: list[dict[str, object]] = [
        {
            "role": "system",
            "content": _synthesis_prompt(mode),
        },
        {
            "role": "user",
            "content": "\n\n".join(
                [
                    f"User question: {plan.query}",
                    f"Workshop title: {plan.title}",
                    f"Shared brief:\n{bundle.shared_context_markdown}",
                    "\n".join(panel_report_lines),
                ]
            ),
        },
    ]
    try:
        synthesis = cast(
            WorkshopSynthesis,
            await run_structured_completion_with_retry(
                complete_fn=ctx.complete_fn,
                messages=synthesis_messages,
                model_def=synthesis_model,
                response_format_model=WorkshopSynthesis,
                max_tokens=workshop_budget.synthesis_max_tokens,
                temperature=0.2,
                completion_content_fn=_completion_payload_text,
            ),
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(
            "workshop synthesis failed, falling back to deterministic rendering: %s",
            exc,
        )
        return _fallback_synthesis_text(plan, execution_summary, rendering), {}

    lines = [f"## {plan.title}"]
    verdict_heading = rendering.verdict_heading.strip()
    if verdict_heading:
        lines.append(f"### {verdict_heading}")
    lines.append(synthesis.verdict.strip())
    _append_rendered_section(
        lines,
        rendering.agreements_heading,
        synthesis.agreements,
        rendering.agreements_list_style,
    )
    _append_rendered_section(
        lines,
        rendering.tensions_heading,
        synthesis.tensions,
        rendering.tensions_list_style,
    )
    _append_rendered_section(
        lines,
        rendering.next_steps_heading,
        synthesis.next_steps,
        rendering.next_steps_list_style,
    )
    return "\n".join(lines), synthesis.model_dump(mode="json")


def _persist_workshop_bundle(
    ctx: OperationExecutionContext,
    bundle: WorkshopContextBundle,
) -> WorkshopArtifactRefs | None:
    if ctx.blob_io is None:
        return None

    write_fn = getattr(ctx.blob_io, "write_workshop_context_bundle", None)
    if not callable(write_fn):
        return None

    try:
        persisted = write_fn(bundle)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("workshop bundle persistence failed: %s", exc)
        return None

    return WorkshopArtifactRefs.model_validate(persisted)


def _build_context_bundle(
    ctx: OperationExecutionContext,
    *,
    state: UnifiedState,
    plan: WorkshopPlan,
    workshop_budget: WorkshopBudgetConfig,
) -> WorkshopContextBundle:
    if not callable(ctx.build_workshop_context_bundle_fn):
        raise RuntimeError("Workshop context builder not injected into operation context.")

    bundle = ctx.build_workshop_context_bundle_fn(
        state=state,
        plan=plan,
        workspace_id=ctx.workspace_id,
        user_id=ctx.user_id,
        user_slug=ctx.user_slug,
        surface_context=_surface_context(ctx),
        workshop_budget=workshop_budget,
        reasoning_model=_is_reasoning_plan(plan),
    )
    if isinstance(bundle, WorkshopContextBundle):
        return bundle
    return WorkshopContextBundle.model_validate(bundle)


def _result_payload(
    plan: WorkshopPlan,
    bundle: WorkshopContextBundle,
    execution_summary: WorkshopExecutionSummary,
    synthesis_payload: dict[str, object],
    artifact_refs: WorkshopArtifactRefs | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "workshop_id": plan.workshop_id,
        "title": plan.title,
        "query": plan.query,
        "shared_used_tokens": bundle.shared_used_tokens,
        "total_used_tokens": bundle.total_used_tokens,
        "panel_contexts": [
            {
                "panel_id": panel.panel_id,
                "title": panel.title,
                "mode_id": panel.mode_id,
                "panel_kind": panel.panel_kind,
                "allocated_tokens": panel.allocated_tokens,
                "used_tokens": panel.used_tokens,
            }
            for panel in bundle.panel_contexts
        ],
        "execution_summary": execution_summary.model_dump(mode="json"),
        "synthesis": synthesis_payload,
    }
    if artifact_refs is not None:
        payload["artifact_refs"] = artifact_refs.model_dump(mode="json")
    return payload


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Run one workshop panel fan-out and return the synthesized result."""

    query = (ctx.query or ctx.params.get("query", "")).strip()
    if not query:
        return OperationResult(output_text="⛔ Workshop mode needs a problem statement or question.")
    if ctx.workflow_turn_runner is None:
        return OperationResult(
            output_text="⛔ Workshop runner was not injected into the operation context."
        )

    state = _resolve_state(ctx)
    if state is None:
        return OperationResult(output_text="⛔ Workshop mode could not load the cached runtime state.")

    plan = _build_plan_from_roster(ctx, query)
    mode = _active_mode_config(ctx)
    workshop_budget = _resolved_workshop_budget(ctx)
    planner_subprompt = _planner_prompt(mode)
    try:
        bundle = _build_context_bundle(
            ctx,
            state=state,
            plan=plan,
            workshop_budget=workshop_budget,
        )
    except Exception as exc:  # pylint: disable=broad-except
        return OperationResult(
            output_text=f"⛔ Workshop mode could not build panel context: {type(exc).__name__}: {exc}"
        )
    artifact_refs = _persist_workshop_bundle(ctx, bundle)
    child_requests = _build_child_requests(ctx, bundle, planner_subprompt)
    if child_requests is None:
        return OperationResult(
            output_text=(
                "⛔ Workshop mode could not resolve a model for one or more child panels."
            )
        )

    parent_context = WorkshopExecutionParentContext(
        user_id=ctx.user_id,
        user_slug=ctx.user_slug,
        user_name=ctx.user_name or ctx.user_slug,
        principal_key=ctx.principal_key,
        thread_user_id=ctx.thread_user_id or ctx.user_id,
        parent_session_key=ctx.conversation_id or f"workshop::{plan.workshop_id}",
        conversation_id=ctx.conversation_id or f"workshop:{plan.workshop_id}",
        workspace_id=ctx.workspace_id,
        surface_context=_surface_context(ctx),
        event_bus=ctx.event_bus,
    )
    execution_service = WorkshopExecutionService(
        run_workflow_turn=ctx.workflow_turn_runner,
        create_session_fn=ctx.workflow_session_factory,
        max_parallel_children=workshop_budget.max_parallel_panels,
    )
    execution_summary = await execution_service.execute_children(
        workshop_id=plan.workshop_id,
        parent_context=parent_context,
        child_requests=child_requests,
    )
    output_text, synthesis_payload = await _synthesize_workshop(
        ctx,
        mode,
        plan,
        bundle,
        execution_summary,
        workshop_budget,
    )
    return OperationResult(
        output_text=output_text,
        output_json=_result_payload(
            plan,
            bundle,
            execution_summary,
            synthesis_payload,
            artifact_refs,
        ),
        extra={
            "workshop_id": plan.workshop_id,
            "workshop_child_statuses": {
                result.panel_id: result.status for result in execution_summary.results
            },
            "workshop_artifact_refs": (
                artifact_refs.model_dump(mode="json") if artifact_refs is not None else {}
            ),
        },
    )
