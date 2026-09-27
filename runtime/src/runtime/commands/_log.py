"""Logging helpers for host command handlers.

These are the low-level helpers that every command cluster reaches for:
structured event emission, thread/turn-context recorders, env-var parsing,
model resolution, and runtime-state accessors. The previous no-op stubs
have been replaced with real emission through
:func:`sdk.logging_setup.emit_structured_event`; the shared
:class:`sdk.logging_setup.JSONFormatter` owns serialization and the
file destination.

Dependency direction: ``_log`` → ``sdk.logging_setup``,
``runtime.runtime``, ``runtime.thread``, ``runtime.model_runtime``.
No imports from sibling ``commands._*`` modules.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from kernel.models import CapabilityManifest
from sdk.logging_setup import emit_structured_event

from runtime.model_runtime import (
    StubModelDefinition,
    resolve_model,
)
from runtime.runtime import HostRuntime, TurnContext
from runtime.thread import append_thread_entry

logger = logging.getLogger(__name__)

_MIN_RESTORE_ARGS = 2


# ── Structured event emission ────────────────────────────────────────


def _append_log_event(
    *,
    level: str,
    event: str,
    payload: dict[str, Any],
    debug_payload: dict[str, Any] | None = None,
) -> None:
    """Emit a structured event through stdlib logging.

    *level* is a stdlib level name (``"info"``, ``"warning"``, ...).
    *event* is the namespaced junction name (e.g. ``"cmd.tokens"``).
    *payload* fields become top-level JSON keys via ``extra=``.
    *debug_payload* is merged into the emission when the runtime log level
    is ``debug`` (callers gate this themselves today; the field is preserved
    for forward compatibility).
    """
    level_name = level.strip().upper()
    if level_name == "WARN":
        level_name = "WARNING"
    level_value = getattr(logging, level_name, logging.INFO)
    message = payload.pop("message", None) or event
    fields = dict(payload)
    if debug_payload:
        fields["debug"] = debug_payload
    emit_structured_event(
        logger,
        level=level_value,
        event=event,
        message=str(message),
        **fields,
    )


def _append_turn_log(*, context: dict[str, Any], response_text: str) -> None:
    """Emit a ``cmd.turn`` structured event for a host-command turn.

    Host commands (``/help``, ``/consolidate``, ``/intake``, ...) do not
    run the kernel turn loop, but they still produce a thread entry and a
    turn-context. This emits the equivalent of the kernel
    ``turn.completed`` event so the JSONL stream stays uniform.
    """
    mode_id = str(context.get("mode_id") or "cmd")
    handoff = str(context.get("handoff") or "")
    emit_structured_event(
        logger,
        level=logging.INFO,
        event="cmd.turn",
        message="cmd.turn: mode=%s handoff=%s",
        mode_id=mode_id,
        handoff=handoff,
        context=context,
        response_chars=len(response_text),
    )


# ── Private helpers ──────────────────────────────────────────────────


def _append_thread_entry(
    runtime: HostRuntime,
    *,
    mode_id: str,
    query: str,
    response_text: str,
    provenance: dict[str, Any] | None = None,
) -> None:
    """Delegate to ``runtime.thread.append_thread_entry`` with runtime paths."""
    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug=runtime.config.user_slug,
        segment_turns=runtime.config.thread_segment_turns,
        mode_id=mode_id,
        query=query,
        response_text=response_text,
        provenance=provenance,
    )


def _set_turn_context(runtime: HostRuntime, **kwargs: Any) -> None:
    """Build a ``TurnContext`` and store it on the runtime."""
    ctx = TurnContext(**kwargs)
    runtime.session.set_last_turn_context(ctx)


def _resolve_int_env(env_var: str, default: int) -> int:
    """Parse a positive integer from an env var with a sensible default."""
    raw = os.getenv(env_var, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _resolve_model(runtime: HostRuntime, model_id: str) -> StubModelDefinition:
    """Resolve one model identifier against the runtime's manifest cache."""
    return resolve_model(
        model_id,
        runtime_state={"manifest": runtime.session.manifest},
        data_dir=runtime.paths.data_dir,
    )


def _last_capability_manifest(runtime: HostRuntime) -> CapabilityManifest | None:
    """Return the last capability manifest stored on the runtime, if any."""
    return getattr(runtime, "last_capability_manifest", None)
