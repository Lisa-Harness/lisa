"""Build ``TurnHooks`` from a ``HostRuntime``.

This module wires the ``HostRuntime`` to the kernel's ``TurnHooks`` dataclass,
replacing the closure-based ``_build_turn_hooks`` in ``main.py`` with a
testable, module-level function.

All dependencies are resolved from the runtime — no ``_RUNTIME_STATE`` reads,
no global module-level caches. The returned ``TurnHooks`` is ready to pass to
``kernel.turn_runner.run_turn``.

Public surface
--------------
- :func:`build_turn_hooks`
- :func:`content_loader`
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from kernel.models import OperationExecutionContext, OperationResult, ToolResult
from kernel.operations.registry import execute_operation
from kernel.seams.protocols import ContentRegistry
from kernel.slice import SliceTuning
from kernel.turn_runner import CheckpointTuning, ContextSeams, TurnHooks

from hosts.core.context_assembly import (
    assemble_dynamic_user_context,
    assemble_system_prompt,
)
from hosts.core.model_runtime import StubModelDefinition, load_manifest, resolve_little_lisa_model
from hosts.core.state_provider import FileBackedStateProvider, build_unified_state
from hosts.core.thread import build_recent_thread_excerpt, load_thread_digest

if TYPE_CHECKING:
    from hosts.core.callbacks import TransportCallbacks
    from hosts.core.runtime import HostRuntime

logger = logging.getLogger(__name__)


# ── Content loader ────────────────────────────────────────────────────


def content_loader(runtime: HostRuntime) -> Callable[[str], str]:
    """Return a callable that reads authored text through the kernel's content registry.

    The content registry is set by ``setup_framework(runtime)`` at host startup.
    When it is not set, this raises ``RuntimeError`` so the caller knows the
    framework was not initialized.
    """
    _ = runtime  # runtime is part of the API for future use; registry is module-level
    registry = content_registry()
    if registry is None:
        raise RuntimeError(
            "Content registry not set. Call setup_framework(runtime) first."
        )
    loader: Callable[[str], str] = registry.read_authored_text
    return loader


def content_registry() -> ContentRegistry | None:
    """Return the module-level content registry, or ``None`` when unset.

    The registry is set by ``setup_framework(runtime)`` at host startup.
    Operations that need authored text (e.g. ``synthesize_and_store_learning``,
    ``advise_models``) reach it via ``TurnHooks.context_seams.content_registry``
    so the turn runner's ``_build_minimal_context`` can resolve
    ``read_authored_text_fn``. When ``None``, those operations fail at call time
    with their authored-text error (graceful failure); non-authored operations
    still run normally.
    """
    from kernel.context.context_frames import _content_registry

    return _content_registry


# ── Turn-hooks builder ────────────────────────────────────────────────


def build_turn_hooks(
    runtime: HostRuntime,
    context: dict[str, Any] | None = None,
    transport: TransportCallbacks | None = None,
) -> TurnHooks:
    """Build a ``TurnHooks`` dataclass wired to the host runtime.

    Args:
        runtime: The typed host runtime with config, manifest, MCP pool.
        context: Optional mutable turn-context dict. When provided, the
            ``on_turn_complete`` hook updates it with completion metadata.
            When ``None``, a fresh dict is created internally.
        transport: Optional transport callbacks for streaming/debug events.

    Returns:
        A ``TurnHooks`` ready to pass to ``kernel.turn_runner.run_turn``.
    """
    from hosts.core.providers import complete_live

    ctx_dict = context if context is not None else {}

    # ── Slice tuning ──────────────────────────────────────────────
    slice_config = runtime.config.slice_config
    slice_tuning = SliceTuning.from_mapping(slice_config.to_tuning_dict())

    history_plan = runtime.session_plan.manifest.context.history
    digest_max_lines = history_plan.digest_max_lines
    digest_summary_max_chars = history_plan.digest_summary_max_chars

    # ── State provider (FileBackedStateProvider) ──────────────────
    file_state_provider = FileBackedStateProvider(
        thread_path=runtime.thread_path,
        max_thread_entries=runtime.config.thread_max_entries,
        digest_max_lines=digest_max_lines,
        digest_summary_max_chars=digest_summary_max_chars,
    )

    # ── Model resolution ──────────────────────────────────────────
    def _resolve_model(model_id: str) -> StubModelDefinition:
        if runtime.manifest is None:
            runtime.manifest = load_manifest({}, data_dir=runtime.data_dir)
        return resolve_little_lisa_model(
            model_id,
            runtime_state={"manifest": runtime.manifest},
            data_dir=runtime.data_dir,
        )

    # ── Completion ────────────────────────────────────────────────
    async def _complete(**kwargs: Any) -> Any:
        return await complete_live(
            runtime=runtime,
            transport=transport,
            **kwargs,
        )

    # ── Content loader (injected into context assembly) ───────────
    loader = content_loader(runtime)

    # ── System prompt assembly ────────────────────────────────────
    def _assemble_system_prompt(mode_instruction: str, stage_instruction: str | None) -> str:
        return assemble_system_prompt(
            mode_instruction,
            stage_instruction,
            loader,
            session_instruction=runtime.session_plan.instructions.primary_body,
        )

    # ── Dynamic user context assembly ─────────────────────────────
    def _assemble_user_context(
        mode_id: str,
        query: str,
        pre_turn_briefing: str,
        capability_manifest: Any = None,
    ) -> str:
        # Build the UnifiedState from the runtime snapshot for the broker.
        state = build_unified_state(
            state_snapshot=runtime.state_snapshot,
        )

        def _session_context_loader() -> str:
            # Primary session instructions are system-level and never become
            # a budgeted dynamic-user-context section.
            return ""

        def _thread_digest_loader() -> str:
            return load_thread_digest(thread_path=runtime.thread_path)

        def _recent_thread_excerpt_builder(limit: int) -> str:
            return build_recent_thread_excerpt(
                thread_path=runtime.thread_path,
                limit=limit,
            )

        return assemble_dynamic_user_context(
            mode_id,
            query,
            pre_turn_briefing,
            content_loader=loader,
            user_slug=runtime.config.user_slug,
            timezone=runtime.config.timezone,
            state=state,
            session_context_loader=_session_context_loader,
            thread_digest_loader=_thread_digest_loader,
            recent_thread_excerpt_builder=_recent_thread_excerpt_builder,
            capability_manifest=capability_manifest,
        )

    # ── Operation execution ───────────────────────────────────────
    async def _execute_operation(
        operation_id: str,
        ctx: OperationExecutionContext,
    ) -> OperationResult:
        ctx.memory_recall_default_limit = 8
        return await execute_operation(operation_id, ctx)

    # ── Tool execution ─────────────────────────────────────────────
    async def _execute_tool(tool_id: str, _tool_ctx: Any) -> ToolResult:
        # Delegate to the kernel's MCP tool pool when available.
        pool = runtime.mcp_tool_pool
        if pool is not None:
            try:
                result = await pool.call_tool(tool_id, {})
                # Adapt the MCP result to ToolResult.
                if isinstance(result, dict):
                    return ToolResult(
                        tool_id=tool_id,
                        output_text=str(result.get("content", "")),
                        output_json=result if isinstance(result, dict) else {},
                    )
                return ToolResult(tool_id=tool_id, output_text=str(result))
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("tool dispatch %s failed: %s", tool_id, exc)
                return ToolResult(tool_id=tool_id, error=str(exc))
        return ToolResult(tool_id=tool_id, error="No MCP tool pool available")

    # ── Turn-complete hook ─────────────────────────────────────────
    async def _on_turn_complete(
        mode_id: str,
        query: str,
        response_text: str,
        messages: list[dict[str, Any]],
    ) -> None:
        # Update the mutable turn context dict.
        assistant_message_count = sum(
            1
            for msg in messages
            if str(msg.get("role") or "").strip().lower() == "assistant"
        )
        ctx_dict["completion"] = {
            "mode_id": mode_id,
            "query": query,
            "response_preview": response_text[:240],
            "response_chars": len(response_text),
            "message_count": len(messages),
            "assistant_message_count": assistant_message_count,
        }

    # ── UnifiedState provider callable ─────────────────────────────
    def _state_provider() -> Any:
        return build_unified_state(
            state_snapshot=runtime.state_snapshot,
        )

    # ── Workflow session factory ───────────────────────────────────
    from kernel.ipsar_fsm import create_workflow_session

    def _workflow_session_factory(
        user_id: str,
        user_slug: str,
        mode_id: str,
    ) -> Any:
        return create_workflow_session(user_id, user_slug, mode_id)

    # ── Last slice finished at ─────────────────────────────────────
    persisted_last_slice_finished_at: float | None = None
    if context is not None:
        raw = context.get("last_slice_finished_at")
        if isinstance(raw, (int, float)):
            persisted_last_slice_finished_at = float(raw)
    if runtime.last_slice_finished_at is not None:
        persisted_last_slice_finished_at = runtime.last_slice_finished_at

    # ── Slice callables ────────────────────────────────────────────
    async def _pre_turn_slice(policy: Any) -> Any:
        return await file_state_provider.slice(ctx=None, policy=policy)

    async def _post_turn_slice(policy: Any) -> Any:
        outcome = await file_state_provider.slice(ctx=None, policy=policy)
        runtime.last_slice_finished_at = __import__("time").monotonic()
        return outcome

    async def _explicit_slice(policy: Any) -> Any:
        return await file_state_provider.slice(ctx=None, policy=policy)

    return TurnHooks(
        resolve_model=_resolve_model,
        complete=_complete,
        mcp_tool_pool=runtime.mcp_tool_pool,
        mcp_tool_servers=runtime.mcp_cold_configs,
        execute_operation=_execute_operation,
        assemble_system_prompt=_assemble_system_prompt,
        assemble_user_context=_assemble_user_context,
        on_turn_complete=_on_turn_complete,
        slice_tuning=slice_tuning,
        pre_turn_slice=_pre_turn_slice,
        post_turn_slice=_post_turn_slice,
        explicit_slice=_explicit_slice,
        last_slice_finished_at=persisted_last_slice_finished_at,
        state_provider=_state_provider,
        workflow_session_factory=_workflow_session_factory,
        checkpoint_tuning=CheckpointTuning(
            enabled=True,
            on_stage_transition=True,
            every_n_turns=10,
        ),
        # Wire the content registry into context_seams so the turn runner's
        # _build_minimal_context can resolve read_authored_text_fn for
        # operations that load authored prompts (e.g. synthesize_and_store_learning,
        # advise_models). The registry is the same module-level singleton that
        # content_loader() reads; setup_framework(runtime) populates it at host
        # startup. When unset, content_registry stays None and operations fail at
        # call time with their authored-text error (preserving graceful failure).
        context_seams=ContextSeams(
            content_registry=cast("Any", content_registry()),
        ),
    )
