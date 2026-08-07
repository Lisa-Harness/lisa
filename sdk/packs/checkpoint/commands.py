"""Checkpoint-pack slash-command handlers.

These handlers live in the pack (not in ``host``) because they
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
    CHECKPOINT_SCHEMA_VERSION,
    ConversationSurface,
    OperationExecutionContext,
    ToolResult,
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
    "handle_proceed",
    "handle_restore",
    "handle_summarize_history",
]


def _load_turn_history(runtime: Any, limit: int = 50) -> list[dict[str, Any]]:
    """Load recent thread entries and normalize for checkpoint operations."""
    from host.thread import load_recent_thread_entries

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
    from host.model_runtime import resolve_model

    return resolve_model(
        model_id,
        runtime_state={"manifest": runtime.manifest},
        data_dir=runtime.data_dir,
    )


def _content_loader(runtime: Any) -> Any:
    """Return the authored-text reader, or None when the registry is not set."""
    try:
        from host.turn_hooks_builder import content_loader  # noqa: PLC0415

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
    from kernel.seams.host_adapter import build_callable_host

    return OperationExecutionContext(
        workspace_id="lite-host",
        user_id=str(runtime.config.user_slug or "lite-user"),
        user_slug=str(runtime.config.user_slug or "default"),
        surface_type=cast(ConversationSurface, runtime.config.surface_type),
        surface_id=runtime.config.surface_id,
        memory_scope="personal",
        mode_id=str(runtime.current_mode_id or "default"),
        query=message,
        host=build_callable_host(
            complete_fn=getattr(runtime, "complete_fn", None),
            resolve_model_fn=lambda model_id: _resolve_model(runtime, model_id),
            read_authored_text_fn=_content_loader(runtime),
            execute_tool_fn=tool_executor,
        ),
        state_provider=lambda: _build_unified_state(runtime),
        params=params,
    )


def _build_unified_state(runtime: Any) -> Any:
    from host.state_provider import build_unified_state

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


def _check_bundle_version(bundle_data: dict[str, Any]) -> str:
    """Return "" when the bundle's schema version is current, else an error.

    Hard cutover (schema_version 2, 2026-08-07): pre-merge bundles carried
    separate ``workflow_session_state`` / ``workflow_instance`` dicts. They
    are rejected loudly — no compat restore path.
    """
    version = bundle_data.get("schema_version", 1)
    if version != CHECKPOINT_SCHEMA_VERSION:
        return (
            "Checkpoint uses an incompatible pre-merge format "
            f"(schema_version={version}); restore is not supported. "
            "Create a fresh checkpoint."
        )
    return ""


def _restore_position(runtime: Any, bundle_data: dict[str, Any]) -> None:
    """Restore the ``WorkflowPosition`` from a checkpoint bundle.

    Rehydrates the single durable workflow object (inter-mode plan +
    intra-mode stage pointer) and overrides ``current_mode_id`` from the
    position's ``mode_id`` — so the next turn resumes in the mode the
    workflow was parked in, not whatever mode was active before the
    restore.
    """
    position_dict = bundle_data.get("workflow_position") or {}
    if not isinstance(position_dict, dict) or not position_dict:
        return
    updater = getattr(runtime, "_set_workflow_position", None)
    if updater is None:
        logger.debug("restore: workflow position present but no updater wired on runtime")
        return
    try:
        from kernel.workflow_position import WorkflowPosition  # noqa: PLC0415

        position = WorkflowPosition.model_validate(position_dict)
        updater(position)
        if position.mode_id:
            runtime.current_mode_id = position.mode_id
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("restore: workflow position restore failed: %s", exc)


async def handle_proceed(
    runtime: Any,
    stripped: str,
    message: str,
) -> str:
    """Handle ``/proceed`` — resume from the most recent yielded checkpoint.

    The unified turn lifecycle stamps ``CheckpointBundle.lifecycle_state`` on
    every checkpoint. When the harness forced a ``YieldAction`` (iteration cap,
    stuck-in-stage, budget exhaustion, or the model called the ``yield`` tool),
    the checkpoint is saved with ``lifecycle_state="yielded"`` and the question
    in ``awaiting_input_question``.

    ``/proceed`` finds the most recent yielded checkpoint, restores its state
    (unified_state, workflow_session, workflow_instance, current_mode_id), and
    surfaces the original question so the user can answer it. The user's next
    message becomes the turn query; because ``current_mode_id`` was restored to
    the workflow's current step, the turn resumes against the same step.

    Also checks for stalled recovery records (``RecoveryRecord`` with
    ``state="stalled"`` and ``retry_policy="confirm"``) and surfaces them
    alongside the yield question so the user can retry/skip the stalled item.

    Usage:
        /proceed              — resume from the most recent yielded checkpoint
        /proceed <reply>      — resume AND inject <reply> as the next turn query
    """
    pool = runtime.mcp_tool_pool
    if pool is None:
        return "No storage pool available to resume."

    try:
        list_result = await pool.call_tool("list_checkpoints", {}, server_id="lisa-storage")
        result_text = getattr(list_result, "output_text", "") or ""
        data = json.loads(result_text) if result_text.strip() else []
        if not isinstance(data, list) or not data:
            return "No checkpoints available to resume."
    except Exception as exc:  # pylint: disable=broad-except
        return f"Resume failed (list checkpoints): {exc}"

    # Find the most recent yielded checkpoint. The list is newest-first.
    # Each entry is a summary dict (checkpoint_id, label, created_at, ...).
    # The lifecycle_state is on the full bundle, not the summary — so we
    # load the most recent and check its lifecycle_state. To avoid loading
    # every checkpoint, we check the label prefix "yielded:" that the
    # checkpoint_create operation applies when lifecycle_state is "yielded".
    # If the operation doesn't set that label, fall back to loading the
    # most recent and inspecting lifecycle_state directly.
    yielded_checkpoint_id = ""
    for entry in data:
        cid = entry.get("checkpoint_id", "")
        label = entry.get("label", "") or ""
        if label.startswith("yielded"):
            yielded_checkpoint_id = cid
            break

    if not yielded_checkpoint_id:
        # No yielded checkpoint found — check if the most recent one is
        # yielded by loading its full bundle.
        if not data:
            return "No checkpoints available to resume."
        # Load the most recent and inspect lifecycle_state.
        try:
            restore_result = await pool.call_tool(
                "restore_checkpoint",
                {"checkpoint_id": data[0].get("checkpoint_id", "")},
                server_id="lisa-storage",
            )
            rt = getattr(restore_result, "output_text", "") or ""
            bundle_dict = json.loads(rt) if rt.strip() else {}
            if isinstance(bundle_dict, dict):
                ls = bundle_dict.get("lifecycle_state", "")
                if ls == "yielded":
                    yielded_checkpoint_id = data[0].get("checkpoint_id", "")
        except Exception:  # pylint: disable=broad-except
            pass

    if not yielded_checkpoint_id:
        return (
            "No yielded checkpoint to resume. "
            "Use /restore <checkpoint_id> to restore any checkpoint manually."
        )

    # Restore the yielded checkpoint (same flow as handle_restore).
    parts = stripped.split(None, 1)
    user_reply = parts[1].strip() if len(parts) > 1 else ""

    ctx = _build_context(
        runtime,
        message,
        params={"checkpoint_id": yielded_checkpoint_id},
        tool_executor=_checkpoint_tool_executor(runtime),
    )

    try:
        result = await execute_operation("checkpoint_restore", ctx)
    except Exception as exc:  # pylint: disable=broad-except
        return f"Resume failed (restore): {exc}"

    output_json = result.output_json or {}
    if not isinstance(output_json, dict):
        output_json = {}

    # Extract the question from the restored bundle.
    question = output_json.get("awaiting_input_question", "") or ""

    version_error = _check_bundle_version(output_json)
    if version_error:
        return version_error

    # Restore unified_state + workflow_position (mirrors handle_restore).
    unified_state_dict = output_json.get("unified_state") or {}
    if isinstance(unified_state_dict, dict) and unified_state_dict:
        runtime.state_snapshot = unified_state_dict

    _restore_position(runtime, output_json)

    # Check for stalled recovery records that block the workflow.
    stall_summary = ""
    recovery_service = getattr(runtime, "recovery_service", None)
    if recovery_service is not None and hasattr(recovery_service, "list_open"):
        try:
            from kernel.recovery import RecoveryIdentity

            identity = RecoveryIdentity(
                workspace_id="lite-host",
                user_id=str(runtime.config.user_slug or "lite-user"),
                user_slug=str(runtime.config.user_slug or "default"),
                surface_type=cast(ConversationSurface, runtime.config.surface_type),
                surface_id=runtime.config.surface_id,
                memory_scope="personal",
            )
            open_records = await recovery_service.list_open(identity)
            stalled = [
                r
                for r in open_records
                if getattr(r, "state", "") == "stalled"
                and getattr(r, "retry_policy", "") == "confirm"
            ]
            if stalled:
                lines = ["Stalled work items needing your decision:"]
                for r in stalled:
                    lines.append(
                        f"  • {r.target_id} ({r.target_kind}): {r.last_error or 'unknown'}"
                    )
                lines.append("Use /retry, /skip, or /cancel to resolve.")
                stall_summary = "\n\n" + "\n".join(lines)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("proceed: recovery list_open failed: %s", exc)

    # Build the resume prompt. If the user provided a reply via
    # ``/proceed <reply>``, set it as the pending resume message so the host
    # replays it as the next turn query (mirrors the CLI restart-resume seam).
    if user_reply:
        runtime.pending_resume_message = user_reply
        return f"Resumed from checkpoint {yielded_checkpoint_id}. Your reply will be sent next."

    if question:
        return f"Resumed from checkpoint {yielded_checkpoint_id}.\n\nQuestion: {question}{stall_summary}"
    return f"Resumed from checkpoint {yielded_checkpoint_id}.{stall_summary}"


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

        version_error = _check_bundle_version(bundle_data)
        if version_error:
            return f"Restore failed: {version_error}"

        unified_state_dict = bundle_data.get("unified_state") or {}
        if unified_state_dict:
            runtime.state_snapshot = unified_state_dict

        _restore_position(runtime, bundle_data)

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
    by ``host/turn_hooks_builder``). Direct pack-level file writes are
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
            prefix="/proceed",
            handler=handle_proceed,
            help_text=(
                "/proceed [reply] resumes from the most recent yielded checkpoint. "
                "Surfaces the question and optionally injects a reply."
            ),
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
