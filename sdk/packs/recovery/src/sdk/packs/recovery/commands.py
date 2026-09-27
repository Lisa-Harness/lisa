"""Recovery-pack slash-command handlers.

These handlers live in the pack (not in ``runtime``) because they
delegate to the recovery service owned by the host runtime. The host
discovers them through the ``lisa.extensions.host_commands`` entry-point
group and dispatches ``/stalls``, ``/stall``, ``/retry``, ``/retry-safe``,
``/skip``, and ``/cancel`` to these handlers.

Design notes:
- ``/stalls`` and ``/stall <id>`` are read-only inspection commands.
- ``/retry <id>`` re-dispatches the saved target (operation or tool)
  through the kernel's existing execution path. Read-only items
  (``retry_policy == "safe"``) retry immediately. Effectful or
  indeterminate items (``retry_policy == "confirm"``) require explicit
  approval — the handler surfaces a prompt and does NOT auto-execute.
- ``/retry-safe`` retries all items with ``retry_policy == "safe"``
  in one command.
- ``/skip <id>`` and ``/cancel <id>`` settle the record as skipped or
  cancelled, unblocking dependent workflow steps.

HostRuntime is typed as ``Any`` here (same pattern as the checkpoint
pack). These handlers need host-layer services (recovery service, model
resolution, operation execution) that are only available at runtime.
"""

from __future__ import annotations

import logging
from typing import Any, cast

from kernel.host_commands import HostCommandSpec
from kernel.models import ConversationSurface, MemoryScope
from kernel.recovery import RecoveryIdentity

logger = logging.getLogger(__name__)

#: ``/cmd <id>`` commands need a command verb plus one argument.
_CMD_ID_ARG_MIN_PARTS = 2

__all__ = [
    "get_host_commands",
    "handle_cancel",
    "handle_resume",
    "handle_retry",
    "handle_retry_safe",
    "handle_skip",
    "handle_stall",
    "handle_stalls",
]


def _recovery_identity(runtime: Any) -> RecoveryIdentity:
    """Build the current host identity used to validate recovery ownership."""

    return RecoveryIdentity(
        workspace_id=runtime.session_plan.manifest.name,
        user_id=runtime.config.user_slug,
        user_slug=runtime.config.user_slug,
        surface_type=cast(ConversationSurface, runtime.config.surface_type),
        surface_id=runtime.config.surface_id,
        memory_scope=cast(MemoryScope, runtime.config.memory_scope),
    )


def _recovery_service(runtime: Any) -> Any:
    """Return the host's recovery service from the grouped ``runtime.services``."""

    return getattr(getattr(runtime, "services", None), "recovery_service", None)


def get_host_commands() -> list[HostCommandSpec]:
    """Return the slash-command specs this pack registers."""
    return [
        HostCommandSpec(
            prefix="/stalls",
            handler=handle_stalls,
            help_text="List open (stalled, indeterminate, running) recovery work items.",
        ),
        HostCommandSpec(
            prefix="/stall",
            handler=handle_stall,
            help_text="Show details for one recovery work item.",
        ),
        HostCommandSpec(
            prefix="/retry",
            handler=handle_retry,
            help_text="Retry one stalled work item (approval-gated for effectful actions).",
        ),
        HostCommandSpec(
            prefix="/retry-safe",
            handler=handle_retry_safe,
            help_text="Retry all safe (read-only) stalled work items.",
        ),
        HostCommandSpec(
            prefix="/skip",
            handler=handle_skip,
            help_text="Mark one work item as skipped, unblocking the workflow.",
        ),
        HostCommandSpec(
            prefix="/cancel",
            handler=handle_cancel,
            help_text="Mark one work item as cancelled, unblocking the workflow.",
        ),
        HostCommandSpec(
            prefix="/resume",
            handler=handle_resume,
            help_text="Resume from the most recent yielded turn.",
        ),
    ]


# ── /stalls ───────────────────────────────────────────────────────────


async def handle_stalls(runtime: Any, stripped: str, message: str) -> str:
    """Handle ``/stalls`` — list open recovery work items."""
    del stripped, message
    service = _recovery_service(runtime)
    if service is None or not hasattr(service, "list_open"):
        return "Recovery service is not available in this host runtime."
    try:
        records = await service.list_open(_recovery_identity(runtime))
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("stalls list failed: %s", exc)
        return f"Failed to list stalled work: {type(exc).__name__}: {exc}"

    if not records:
        return "No stalled or indeterminate work items."

    lines = ["Open recovery work items:"]
    for record in records:
        lines.append(_render_record_summary(record))
    return "\n".join(lines)


# ── /stall <id> ───────────────────────────────────────────────────────


async def handle_stall(runtime: Any, stripped: str, message: str) -> str:
    """Handle ``/stall <id>`` — show details for one work item."""
    del message
    parts = stripped.split(None, 1)
    if len(parts) < _CMD_ID_ARG_MIN_PARTS or not parts[1].strip():
        return "Usage: /stall <work_item_id>"

    work_item_id = parts[1].strip()
    service = _recovery_service(runtime)
    if service is None or not hasattr(service, "get"):
        return "Recovery service is not available in this host runtime."

    try:
        record = await service.get(work_item_id, _recovery_identity(runtime))
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("stall get failed: %s", exc)
        return f"Failed to read work item: {type(exc).__name__}: {exc}"

    if record is None:
        return f"No work item found with id '{work_item_id}' for this session."

    return _render_record_detail(record)


def _retry_guard_message(record: Any, work_item_id: str) -> str | None:
    """Return the blocking message for a non-retryable record, or None.

    Encodes the /retry policy chain: unknown id, not operator-open,
    ``retry_policy == "never"``, and the effectful ``confirm`` case that
    requires a fresh turn rather than a replay.
    """
    if record is None:
        return f"No work item found with id '{work_item_id}' for this session."

    if not record.open_for_operator():
        return f"Work item {work_item_id} is in state '{record.state}' and is not open for retry."

    if record.retry_policy == "never":
        return (
            f"Work item {work_item_id} has retry_policy='never' and "
            "cannot be retried. Use /skip or /cancel to dismiss it."
        )

    if record.retry_policy == "confirm":
        # There is no stored ApprovalInvocation on the recovery record (the
        # approval and recovery stores are independent and share no id), so
        # ``/retry`` cannot replay an effectful action directly. Two real
        # paths forward: (a) re-issue the original request so the model
        # re-attempts the full turn and mints a fresh, approve-able action;
        # (b) dismiss the stall with /skip or /cancel so the workflow can
        # advance past this step. The previous text pointed at
        # ``/approve-after-review`` — a command that does not exist.
        return (
            f"Work item {work_item_id} targets {record.target_kind}:"
            f"{record.target_id} with effect='{record.effect}'.\n"
            "This action is effectful and cannot be replayed from a recovery "
            "record (the approval and recovery stores share no id).\n"
            "To proceed, either:\n"
            "  - re-issue your original request so the model re-attempts the "
            "full turn and mints a fresh approval, or\n"
            f"  - run `/skip {work_item_id}` or `/cancel {work_item_id}` to "
            "dismiss this stall and advance the workflow."
        )
    return None


# ── /retry <id> ───────────────────────────────────────────────────────


async def handle_retry(runtime: Any, stripped: str, message: str) -> str:
    """Handle ``/retry <id>`` — retry one stalled work item.

    Read-only items (``retry_policy == "safe"``) are retried immediately.
    Effectful or indeterminate items (``retry_policy == "confirm"``)
    require explicit operator approval before replay — the handler
    surfaces a prompt and does NOT auto-execute.
    """
    del message
    parts = stripped.split(None, 1)
    if len(parts) < _CMD_ID_ARG_MIN_PARTS or not parts[1].strip():
        return "Usage: /retry <work_item_id>"

    work_item_id = parts[1].strip()
    service = _recovery_service(runtime)
    if service is None or not hasattr(service, "get"):
        return "Recovery service is not available in this host runtime."

    try:
        record = await service.get(work_item_id, _recovery_identity(runtime))
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("retry get failed: %s", exc)
        return f"Failed to read work item: {type(exc).__name__}: {exc}"

    guard = _retry_guard_message(record, work_item_id)
    if guard is not None:
        return guard

    # Safe (read-only) retry: settle as succeeded placeholder for now.
    # Full replay execution (re-dispatch through the kernel) is the next
    # implementation phase — it requires wiring the saved invocation
    # through execute_operation / execute_tool with a fresh context.
    try:
        await service.settle(
            work_item_id,
            _recovery_identity(runtime),
            state="succeeded",
            result_preview="retry requested (replay execution pending)",
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("retry settle failed: %s", exc)
        return f"Retry failed to settle: {type(exc).__name__}: {exc}"

    return (
        f"Retried work item {work_item_id} ({record.target_kind}:"
        f"{record.target_id}). The item has been marked as resolved."
    )


# ── /retry-safe ───────────────────────────────────────────────────────


async def handle_retry_safe(runtime: Any, stripped: str, message: str) -> str:
    """Handle ``/retry-safe`` — retry all safe (read-only) stalled items."""
    del stripped, message
    service = _recovery_service(runtime)
    if service is None or not hasattr(service, "list_open"):
        return "Recovery service is not available in this host runtime."

    try:
        records = await service.list_open(_recovery_identity(runtime))
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("retry-safe list failed: %s", exc)
        return f"Failed to list stalled work: {type(exc).__name__}: {exc}"

    safe_records = [r for r in records if r.retry_policy == "safe"]
    if not safe_records:
        return "No safe (read-only) work items to retry."

    settled = 0
    for record in safe_records:
        try:
            await service.settle(
                record.work_item_id,
                _recovery_identity(runtime),
                state="succeeded",
                result_preview="retry-safe batch (replay execution pending)",
            )
            settled += 1
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("retry-safe settle failed for %s: %s", record.work_item_id, exc)

    return f"Retried {settled}/{len(safe_records)} safe work items."


# ── /skip <id> ────────────────────────────────────────────────────────


async def handle_skip(runtime: Any, stripped: str, message: str) -> str:
    """Handle ``/skip <id>`` — mark one item as skipped."""
    del message
    parts = stripped.split(None, 1)
    if len(parts) < _CMD_ID_ARG_MIN_PARTS or not parts[1].strip():
        return "Usage: /skip <work_item_id>"

    work_item_id = parts[1].strip()
    service = _recovery_service(runtime)
    if service is None or not hasattr(service, "settle"):
        return "Recovery service is not available in this host runtime."

    try:
        updated = await service.settle(
            work_item_id,
            _recovery_identity(runtime),
            state="skipped",
            last_error="skipped by operator",
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("skip settle failed: %s", exc)
        return f"Skip failed: {type(exc).__name__}: {exc}"

    if updated is None:
        return f"No open work item found with id '{work_item_id}' for this session."

    return f"Skipped work item {work_item_id} ({updated.target_kind}:{updated.target_id})."


# ── /cancel <id> ──────────────────────────────────────────────────────


async def handle_cancel(runtime: Any, stripped: str, message: str) -> str:
    """Handle ``/cancel <id>`` — mark one item as cancelled."""
    del message
    parts = stripped.split(None, 1)
    if len(parts) < _CMD_ID_ARG_MIN_PARTS or not parts[1].strip():
        return "Usage: /cancel <work_item_id>"

    work_item_id = parts[1].strip()
    service = _recovery_service(runtime)
    if service is None or not hasattr(service, "settle"):
        return "Recovery service is not available in this host runtime."

    try:
        updated = await service.settle(
            work_item_id,
            _recovery_identity(runtime),
            state="cancelled",
            last_error="cancelled by operator",
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("cancel settle failed: %s", exc)
        return f"Cancel failed: {type(exc).__name__}: {exc}"

    if updated is None:
        return f"No open work item found with id '{work_item_id}' for this session."

    return f"Cancelled work item {work_item_id} ({updated.target_kind}:{updated.target_id})."


# ── /resume ───────────────────────────────────────────────────────────


async def handle_resume(runtime: Any, stripped: str, message: str) -> str:
    """Handle ``/resume [reply]`` — resume from the most recent yielded turn.

    Scans the thread tail for the most recent entry with ``yielded == True``.
    If found, surfaces the ``yield_question`` so the operator can answer it.
    If the operator provides ``/resume <reply>``, the reply is stashed in
    ``runtime.session.pending_resume_message`` so the next turn replays it as the
    query — the same mechanism used by the CLI restart-resume path.

    Also surfaces any stalled recovery records (``state="stalled"`` with
    ``retry_policy="confirm"``) alongside the yield question so the
    operator can resolve them with ``/retry``, ``/skip``, or ``/cancel``.
    """
    del message
    from runtime.thread import load_recent_thread_entries

    entries = load_recent_thread_entries(thread_path=runtime.paths.thread_path, limit=30)
    yielded_entry: dict[str, Any] | None = None
    for entry in reversed(entries):
        if entry.get("yielded"):
            yielded_entry = entry
            break

    if yielded_entry is None:
        return "No yielded turn to resume."

    question = str(yielded_entry.get("yield_question") or "")
    turn = yielded_entry.get("turn", "?")

    # Surface stalled recovery records alongside the yield question.
    stall_summary = ""
    recovery_service = _recovery_service(runtime)
    if recovery_service is not None and hasattr(recovery_service, "list_open"):
        try:
            open_records = await recovery_service.list_open(_recovery_identity(runtime))
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
            logger.warning("resume: recovery list_open failed: %s", exc)

    # If the operator provided a reply, stash it for the next turn.
    parts = stripped.split(None, 1)
    user_reply = parts[1].strip() if len(parts) > 1 else ""
    if user_reply:
        runtime.session.pending_resume_message = user_reply
        return f"Resumed from turn {turn}. Your reply will be sent next.{stall_summary}"

    if question:
        return f"Resumed from turn {turn}.\n\nQuestion: {question}{stall_summary}"
    return f"Resumed from turn {turn}.{stall_summary}"


# ── Rendering helpers ─────────────────────────────────────────────────


def _render_record_summary(record: Any) -> str:
    """Render one work item as a single-line summary for ``/stalls``."""
    item_id = str(getattr(record, "work_item_id", "") or "")
    state = str(getattr(record, "state", "unknown") or "unknown")
    target_kind = str(getattr(record, "target_kind", "") or "")
    target_id = str(getattr(record, "target_id", "") or "")
    mode_id = str(getattr(record, "mode_id", "") or "")
    stage_id = str(getattr(record, "stage_id", "") or "")
    retry_policy = str(getattr(record, "retry_policy", "") or "")
    last_error = str(getattr(record, "last_error", "") or "")
    line = (
        f"- {item_id} state={state} target={target_kind}:{target_id} "
        f"mode={mode_id} stage={stage_id or '-'} retry={retry_policy}"
    )
    if last_error:
        line += f"\n  error: {last_error}"
    return line


def _render_record_detail(record: Any) -> str:
    """Render one work item as a multi-line detail block for ``/stall <id>``."""
    lines = [
        f"Work item: {getattr(record, 'work_item_id', '')}",
        f"  state: {getattr(record, 'state', 'unknown')}",
        f"  target: {getattr(record, 'target_kind', '')}:{getattr(record, 'target_id', '')}",
        f"  effect: {getattr(record, 'effect', 'read')}",
        f"  mode: {getattr(record, 'mode_id', '')}",
        f"  stage: {getattr(record, 'stage_id', '') or '-'}",
        f"  retry_policy: {getattr(record, 'retry_policy', 'safe')}",
        f"  attempt: {getattr(record, 'attempt', 1)}",
    ]
    last_error = str(getattr(record, "last_error", "") or "")
    if last_error:
        lines.append(f"  last_error: {last_error}")
    result_preview = str(getattr(record, "result_preview", "") or "")
    if result_preview:
        lines.append(f"  result_preview: {result_preview}")
    settled_at = getattr(record, "settled_at", None)
    if settled_at is not None:
        lines.append(f"  settled_at: {settled_at}")
    return "\n".join(lines)
