"""Checkpoint-pack slash-command handlers.

These handlers live in the pack (not in ``hosts/core``) because they
delegate to checkpoint-pack operations (``checkpoint_create``,
``checkpoint_restore``, ``checkpoint_history_summarize``). The host
discovers them through the ``lisa.extensions.host_commands`` entry-point
group and dispatches ``/checkpoint``, ``/restore``, ``/checkpoints``,
and ``/summarize-history`` to these handlers.
"""

from __future__ import annotations

import json
import logging
from typing import Any, cast

from kernel.host_commands import HostCommandSpec
from kernel.models import (
    ConversationSurface,
    OperationExecutionContext,
    ToolResult,
    WorkflowSessionState,
)
from kernel.operations.registry import execute_operation

# HostRuntime is intentionally typed as ``Any`` here. The checkpoint pack
# command handlers need host-layer services (thread loading, model resolution,
# state provider, content loader) that are only available at runtime via lazy
# imports. A Protocol would not capture the full HostRuntime surface these
# handlers use. This is a known sdk->hosts coupling exception for pack command
# handlers — see ../README.md.

logger = logging.getLogger(__name__)

__all__ = [
    "get_host_commands",
    "handle_checkpoint",
    "handle_list_checkpoints",
    "handle_restore",
    "handle_summarize_history",
]


def _load_turn_history(runtime: Any, limit: int = 50) -> list[dict[str, Any]]:
    """Load recent thread entries and normalize for checkpoint operations."""
    from hosts.core.thread import load_recent_thread_entries

    entries = load_recent_thread_entries(
        thread_path=runtime.thread_path,
        limit=limit,
    )
    return [
        {
            "timestamp_utc": entry.get("timestamp_utc", ""),
            "mode_id": entry.get("mode_id", ""),
            "query": entry.get("query", ""),
            "response_text": entry.get("response_text", ""),
        }
        for entry in entries
    ]


def _resolve_model(runtime: Any, model_id: str) -> Any:
    from hosts.core.model_runtime import resolve_model

    return resolve_model(
        model_id,
        runtime_state={"manifest": runtime.manifest},
        data_dir=runtime.data_dir,
    )


def _content_loader(runtime: Any) -> Any:
    """Return the authored-text reader, or None when the registry is not set."""
    try:
        from hosts.core.turn_hooks_builder import content_loader  # noqa: PLC0415

        return content_loader(runtime)
    except RuntimeError:
        return None


def _build_context(
    runtime: Any,
    message: str,
    *,
    params: dict[str, Any],
    tool_executor: Any = None,
) -> OperationExecutionContext:
    """Build an OperationExecutionContext for a checkpoint operation."""
    return OperationExecutionContext(
        workspace_id="lite-host",
        user_id=str(runtime.config.user_slug or "lite-user"),
        user_slug=str(runtime.config.user_slug or "default"),
        surface_type=cast(ConversationSurface, runtime.config.surface_type),
        surface_id=runtime.config.surface_id,
        memory_scope="personal",
        mode_id=str(runtime.current_mode_id or "default"),
        query=message,
        complete_fn=getattr(runtime, "complete_fn", None),
        resolve_model_fn=lambda model_id: _resolve_model(runtime, model_id),
        read_authored_text_fn=_content_loader(runtime),
        execute_tool_fn=tool_executor,
        state_provider=lambda: _build_unified_state(runtime),
        params=params,
    )


def _build_unified_state(runtime: Any) -> Any:
    from hosts.core.state_provider import build_unified_state

    return build_unified_state(state_snapshot=runtime.state_snapshot)


def _checkpoint_tool_executor(runtime: Any) -> Any:
    """Build a tool executor that routes checkpoint tools to lisa-storage."""
    pool = runtime.mcp_tool_pool

    async def _executor(tool_id: str, _ctx: Any) -> ToolResult:
        params = getattr(_ctx, "parameters", {}) or {}
        mcp_result = await pool.call_tool(tool_id, params, server_id="lisa-storage")
        return ToolResult(
            tool_id=tool_id,
            output_text=mcp_result.output_text,
            error=mcp_result.output_text if mcp_result.is_error else "",
        )

    return _executor


# ── /checkpoint ──────────────────────────────────────────────────────


async def handle_checkpoint(
    runtime: Any,
    stripped: str,
    message: str,
) -> str:
    """Handle ``/checkpoint [label]`` — create an enriched checkpoint."""
    parts = stripped.split(None, 1)
    label = parts[1].strip() if len(parts) > 1 else ""

    if runtime.mcp_tool_pool is None:
        return "No storage pool available to persist checkpoint."

    turn_history = _load_turn_history(runtime)
    ctx = _build_context(
        runtime,
        message,
        params={"label": label, "turn_history": turn_history},
        tool_executor=_checkpoint_tool_executor(runtime),
    )

    try:
        result = await execute_operation("checkpoint_create", ctx)
        return result.output_text or "Checkpoint created."
    except Exception as exc:  # pylint: disable=broad-except
        return f"Checkpoint failed: {exc}"


# ── /restore ─────────────────────────────────────────────────────────


async def handle_restore(
    runtime: Any,
    stripped: str,
    message: str,
) -> str:
    """Handle ``/restore <checkpoint_id>`` — restore state from a checkpoint."""
    parts = stripped.split(None, 1)
    if len(parts) < 2 or not parts[1].strip():
        return "Usage: /restore <checkpoint_id>"

    checkpoint_id = parts[1].strip()
    if runtime.mcp_tool_pool is None:
        return "No storage pool available to restore from."

    ctx = _build_context(
        runtime,
        message,
        params={"checkpoint_id": checkpoint_id},
        tool_executor=_checkpoint_tool_executor(runtime),
    )

    try:
        result = await execute_operation("checkpoint_restore", ctx)

        bundle_data = result.output_json or {}
        if not isinstance(bundle_data, dict):
            bundle_data = {}

        unified_state_dict = bundle_data.get("unified_state") or {}
        if unified_state_dict:
            runtime.state_snapshot = unified_state_dict

        wf_session_dict = bundle_data.get("workflow_session_state") or {}
        if wf_session_dict:
            updater = getattr(runtime, "workflow_session_updater", None)
            if updater is not None:
                try:
                    session = WorkflowSessionState.model_validate(wf_session_dict)
                    updater(session)
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning("restore: workflow session restore failed: %s", exc)
            else:
                logger.debug(
                    "restore: workflow session state present but no updater wired on runtime"
                )

        # ── Durable workflow-instance restore ───────────────────────────
        # Rehydrate the inter-mode chain state (completed_steps,
        # current_step_id) and override ``current_mode_id`` from the
        # instance's current step's mode — the fix for the gap where a
        # restore never changed the active mode, so the next turn ran in
        # whatever mode was active before the restore instead of the
        # workflow's current step's mode.
        wf_instance_dict = bundle_data.get("workflow_instance") or {}
        if wf_instance_dict:
            instance_updater = getattr(runtime, "_set_workflow_instance", None)
            if instance_updater is not None:
                try:
                    from kernel.workflow_instance import WorkflowInstance  # noqa: PLC0415

                    instance = WorkflowInstance.model_validate(wf_instance_dict)
                    instance_updater(instance)
                    # Override current_mode_id from the instance's current step.
                    current_mode = getattr(instance, "current_mode_id", "") or ""
                    if current_mode:
                        runtime.current_mode_id = current_mode
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning("restore: workflow instance restore failed: %s", exc)
            else:
                logger.debug(
                    "restore: workflow instance state present but no updater wired on runtime"
                )

        return result.output_text or f"Restored checkpoint {checkpoint_id}"
    except Exception as exc:  # pylint: disable=broad-except
        return f"Restore failed: {exc}"


# ── /checkpoints ─────────────────────────────────────────────────────


async def handle_list_checkpoints(
    runtime: Any,
    stripped: str,
    message: str,
) -> str:
    """Handle ``/checkpoints`` — list all saved checkpoints."""
    del stripped, message
    pool = runtime.mcp_tool_pool
    if pool is None:
        return "No storage pool available to list checkpoints."

    try:
        result = await pool.call_tool("list_checkpoints", {}, server_id="lisa-storage")
        result_text = getattr(result, "output_text", "") or ""

        data = json.loads(result_text)
        if isinstance(data, dict) and "error" in data:
            return f"List failed: {data['error']}"
        if not isinstance(data, list) or not data:
            return "No checkpoints saved."

        lines = ["Saved checkpoints (newest first):"]
        for entry in data:
            cid = entry.get("checkpoint_id", "")
            label = entry.get("label") or ""
            created = entry.get("created_at", "")[:19]
            stage = entry.get("stage_id") or ""
            turns = entry.get("turn_count") or 0
            lines.append(
                f"  {cid}  {created}  stage={stage}  turns={turns}"
                + (f"  label={label}" if label else "")
            )
        return "\n".join(lines)
    except Exception as exc:  # pylint: disable=broad-except
        return f"List failed: {exc}"


# ── /summarize-history ───────────────────────────────────────────────


def _persist_compacted_history(runtime: Any, new_history: list[dict[str, Any]]) -> bool:
    """Write the compacted turn-history list back to the live thread JSONL.

    Prefers the host's ``write_turn_history_fn`` seam (wired on ``TurnHooks``
    by ``hosts/core/turn_hooks_builder``). Direct pack-level file writes are
    intentionally disallowed so all persistence ownership remains in host
    services / lisa-storage. Returns ``True`` when the write succeeded,
    ``False`` (with a warning logged) when it did not.
    """
    writer = getattr(runtime, "write_turn_history_fn", None)
    if callable(writer):
        try:
            writer(new_history)
            return True
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("summarize-history: write_turn_history_fn failed: %s", exc)
            return False
    logger.warning(
        "summarize-history: write_turn_history_fn is not wired; compacted history not persisted"
    )
    return False


async def handle_summarize_history(
    runtime: Any,
    stripped: str,
    message: str,
) -> str:
    """Handle ``/summarize-history [prompt]`` — condense older turns."""
    parts = stripped.split(None, 1)
    nl_prompt = parts[1].strip() if len(parts) > 1 else ""

    turn_history = _load_turn_history(runtime)
    ctx = _build_context(
        runtime,
        message,
        params={
            "turn_history": turn_history,
            "retention_count": 6,
            "prompt": nl_prompt,
        },
    )

    try:
        result = await execute_operation("checkpoint_history_summarize", ctx)
    except Exception as exc:  # pylint: disable=broad-except
        return f"Summarize-history failed: {exc}"

    output_json = result.output_json or {}
    new_history = output_json.get("turn_history") if isinstance(output_json, dict) else None
    if not isinstance(new_history, list):
        # Operation reported no compaction (e.g. already within retention).
        return result.output_text or "History summarized."

    written = _persist_compacted_history(runtime, new_history)
    suffix = " (persisted)" if written else " (write-back unavailable; result not persisted)"
    return (result.output_text or "History summarized.") + suffix


# ── Entry-point registration ─────────────────────────────────────────


def get_host_commands() -> list[HostCommandSpec]:
    """Return the checkpoint pack's slash-command specs.

    Called by the ``lisa.extensions.host_commands`` entry-point loader.
    """
    return [
        HostCommandSpec(
            prefix="/checkpoint",
            handler=handle_checkpoint,
            help_text="/checkpoint [label] saves a state+workflow checkpoint to lisa-storage.",
        ),
        HostCommandSpec(
            prefix="/restore",
            handler=handle_restore,
            help_text="/restore <checkpoint_id> restores state from a saved checkpoint.",
        ),
        HostCommandSpec(
            prefix="/checkpoints",
            handler=handle_list_checkpoints,
            help_text="/checkpoints lists all saved checkpoints (newest first).",
        ),
        HostCommandSpec(
            prefix="/summarize-history",
            handler=handle_summarize_history,
            help_text="/summarize-history [prompt] condenses older turns into a summary.",
        ),
    ]
