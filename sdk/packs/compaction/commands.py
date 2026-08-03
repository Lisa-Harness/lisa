"""Compaction-pack slash-command handlers.

These handlers live in the pack (not in ``hosts/core``) because they
either delegate to the ``context_compact`` operation (``/compact-now``)
or read the compaction slice's sidecar state (``/compaction-status``).
The host discovers them through the ``lisa.extensions.host_commands``
entry-point group and dispatches ``/compact-now`` and
``/compaction-status`` to these handlers.
"""

from __future__ import annotations

import json
import logging
from typing import Any, cast

from kernel.host_commands import HostCommandSpec
from kernel.models import ConversationSurface, OperationExecutionContext
from kernel.operations.registry import execute_operation

# HostRuntime is intentionally typed as ``Any`` here. The compaction pack
# command handlers need host-layer services (thread loading, model resolution,
# state provider, content loader) that are only available at runtime via lazy
# imports. A Protocol would not capture the full HostRuntime surface these
# handlers use. This is a known sdk->hosts coupling exception for pack command
# handlers — see ../checkpoint/commands.py for the established pattern.

logger = logging.getLogger(__name__)

__all__ = ["get_host_commands", "handle_compact_now", "handle_compaction_status"]


def _resolve_model(runtime: Any, model_id: str) -> Any:
    from hosts.core.model_runtime import resolve_model  # noqa: PLC0415

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
) -> OperationExecutionContext:
    """Build an OperationExecutionContext for a compaction operation."""
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
        state_provider=lambda: _build_unified_state(runtime),
        params=params,
    )


def _build_unified_state(runtime: Any) -> Any:
    from hosts.core.state_provider import build_unified_state

    return build_unified_state(state_snapshot=runtime.state_snapshot)


def _load_turn_history(runtime: Any, limit: int = 50) -> list[dict[str, Any]]:
    """Load recent thread entries and normalize for compaction operations."""
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


def _write_turn_history(runtime: Any, new_history: list[dict[str, Any]]) -> bool:
    """Persist the compacted turn history via the host write-back seam.

    Prefers the host's ``write_turn_history_fn`` seam (wired on
    ``TurnHooks`` by ``hosts/core/turn_hooks_builder``). Direct pack-level
    file writes are intentionally disallowed so all persistence ownership
    remains in host services / lisa-storage. Returns ``True`` when the
    write succeeded, ``False`` (with a warning logged) when it did not.
    """
    writer = getattr(runtime, "write_turn_history_fn", None)
    if callable(writer):
        try:
            writer(new_history)
            return True
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("compaction /compact-now: write_turn_history_fn failed: %s", exc)
            return False
    logger.warning(
        "compaction /compact-now: write_turn_history_fn is not wired; "
        "compacted history not persisted"
    )
    return False


def _sidecar_path(runtime: Any) -> Any:
    thread_path = runtime.thread_path
    if thread_path is None:
        return None
    return thread_path.parent / "last_compaction.json"


def _read_sidecar(runtime: Any) -> dict[str, Any]:
    path = _sidecar_path(runtime)
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("compaction /compaction-status: sidecar read failed: %s", exc)
        return {}


# ── /compact-now ────────────────────────────────────────────────────


async def handle_compact_now(
    runtime: Any,
    stripped: str,
    message: str,
) -> str:
    """Handle ``/compact-now [prompt]`` — run context compaction immediately."""
    parts = stripped.split(None, 1)
    nl_prompt = parts[1].strip() if len(parts) > 1 else ""

    turn_history = _load_turn_history(runtime)
    if not turn_history:
        return "No turn history to compact."

    utilization = 0.0
    snapshot = getattr(runtime, "last_context_snapshot", None)
    if snapshot is not None:
        window = int(getattr(snapshot, "model_context_window_tokens", 0) or 0)
        if window > 0:
            used = int(getattr(snapshot, "used_context_tokens", 0) or 0)
            conversation = int(getattr(snapshot, "conversation_tokens", 0) or 0)
            utilization = (conversation + used) / window

    ctx = _build_context(
        runtime,
        message,
        params={
            "turn_history": turn_history,
            "retention_count": 6,
            "prompt": nl_prompt,
            "utilization": utilization,
        },
    )

    try:
        result = await execute_operation("context_compact", ctx)
    except Exception as exc:  # pylint: disable=broad-except
        return f"Compact-now failed: {exc}"

    output_json = result.output_json or {}
    new_history = output_json.get("turn_history") if isinstance(output_json, dict) else None
    if not isinstance(new_history, list):
        return result.output_text or "Compaction produced no writable history."

    written = _write_turn_history(runtime, new_history)
    suffix = " (persisted)" if written else " (write-back unavailable; result not persisted)"
    return (result.output_text or "Compacted.") + suffix


# ── /compaction-status ──────────────────────────────────────────────


async def handle_compaction_status(
    runtime: Any,
    stripped: str,
    message: str,
) -> str:
    """Handle ``/compaction-status`` — report utilization and last fire."""
    lines: list[str] = []

    snapshot = getattr(runtime, "last_context_snapshot", None)
    if snapshot is None:
        lines.append("Context snapshot: not available this turn.")
    else:
        window = int(getattr(snapshot, "model_context_window_tokens", 0) or 0)
        used = int(getattr(snapshot, "used_context_tokens", 0) or 0)
        conversation = int(getattr(snapshot, "conversation_tokens", 0) or 0)
        if window > 0:
            utilization = (conversation + used) / window
            lines.append(
                f"Context window: {window} tokens | used+conversation: "
                f"{used + conversation} | utilization: {utilization:.1%}"
            )
        else:
            lines.append("Context window: 0 (utilization unknown).")

    sidecar = _read_sidecar(runtime)
    if not sidecar:
        lines.append("Last compaction: never (sidecar empty).")
    else:
        lines.append(
            "Last compaction: "
            f"turn_count={sidecar.get('turn_count', '?')} "
            f"last_fired_turn={sidecar.get('last_fired_turn', '?')} "
            f"last_fired_at={sidecar.get('last_fired_at', '?')}"
        )

    return "\n".join(lines)


# ── Entry-point registration ─────────────────────────────────────────


def get_host_commands() -> list[HostCommandSpec]:
    """Return the compaction pack's slash-command specs.

    Called by the ``lisa.extensions.host_commands`` entry-point loader.
    """
    return [
        HostCommandSpec(
            prefix="/compact-now",
            handler=handle_compact_now,
            help_text=(
                "/compact-now [prompt] runs context compaction immediately "
                "and persists the result."
            ),
        ),
        HostCommandSpec(
            prefix="/compaction-status",
            handler=handle_compaction_status,
            help_text=(
                "/compaction-status reports context-window utilization "
                "and the last compaction fire."
            ),
        ),
    ]