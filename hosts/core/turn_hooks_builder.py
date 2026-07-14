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
from typing import TYPE_CHECKING, Any

from kernel.models import OperationExecutionContext, OperationResult, ToolResult
from kernel.operations.registry import execute_operation
from kernel.slice import SliceTuning
from kernel.turn_runner import CheckpointTuning, TurnHooks

from hosts.core.context_assembly import (
    assemble_dynamic_user_context,
    assemble_system_prompt,
)
from hosts.core.model_runtime import StubModelDefinition, load_manifest
from hosts.core.model_runtime import resolve_little_lisa_model
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
    from kernel.context.context_frames import _content_registry

    if _content_registry is None:
        raise RuntimeError(
            "Content registry not set. Call setup_framework(runtime) first."
        )
    return _content_registry.read_authored_text


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

    # ── State provider (FileBackedStateProvider) ──────────────────
    file_state_provider = FileBackedStateProvider(
        thread_path=runtime.thread_path,
        max_thread_entries=runtime.config.thread_max_entries,
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
        return assemble_system_prompt(mode_instruction, stage_instruction, loader)

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
            context_routing=runtime.config.context_routing,
        )

        def _session_context_loader() -> str:
            from hosts.core.session_context import (
                load_selected_session_context_text,
                resolve_session_context_auto,
            )

            ref = runtime.config.session_context
            active = bool(runtime.config.session_context)

            # Auto-discover session_context.md or session_context/ in CWD
            # or the data directory when no explicit --session-context was
            # passed. This matches the Little Lisa behavior.
            if not ref:
                _, ref, active = resolve_session_context_auto(
                    data_dir=runtime.config.data_dir,
                )

            if not ref:
                return ""
            return load_selected_session_context_text(
                session_context_ref=ref,
                session_context_active=active,
                current_mode_id=mode_id,
                content_loader=loader,
            )

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
            context_routing=runtime.config.context_routing,
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
    )