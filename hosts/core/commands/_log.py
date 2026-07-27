"""Logging stubs and runtime-accessor helpers for host command handlers.

These are the low-level helpers that every command cluster reaches for:
no-op structured-logging stubs (wired in a later phase), thread/turn-context
recorders, env-var parsing, model resolution, and runtime-state accessors.

Dependency direction: ``_log`` → ``hosts.core.runtime``, ``hosts.core.thread``,
``hosts.core.model_runtime``. No imports from sibling ``commands._*`` modules.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any

from kernel.models import CapabilityManifest

from hosts.core.model_runtime import (
    StubModelDefinition,
    load_manifest,
    resolve_little_lisa_model,
)
from hosts.core.runtime import HostRuntime, TurnContext
from hosts.core.thread import append_thread_entry

logger = logging.getLogger(__name__)

_MIN_RESTORE_ARGS = 2


# ── Private no-op stubs (logging wired later) ────────────────────────


def _append_log_event(
    *,
    level: str,
    event: str,
    payload: dict[str, Any],
    debug_payload: dict[str, Any] | None = None,
) -> None:
    """No-op stub — structured logging will be wired in a later phase."""
    del level, event, payload, debug_payload


def _append_turn_log(*, context: dict[str, Any], response_text: str) -> None:
    """No-op stub — turn logging will be wired in a later phase."""
    del context, response_text


# ── Private helpers ──────────────────────────────────────────────────


def _append_thread_entry(
    runtime: HostRuntime,
    *,
    mode_id: str,
    query: str,
    response_text: str,
    provenance: dict[str, Any] | None = None,
) -> None:
    """Delegate to ``hosts.core.thread.append_thread_entry`` with runtime paths."""
    append_thread_entry(
        thread_path=runtime.thread_path,
        user_slug=runtime.config.user_slug,
        mode_id=mode_id,
        query=query,
        response_text=response_text,
        provenance=provenance,
    )


def _set_turn_context(runtime: HostRuntime, **kwargs: Any) -> None:
    """Build a ``TurnContext`` and store it on the runtime."""
    ctx = TurnContext(**kwargs)
    runtime.set_last_turn_context(ctx)


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
    if runtime.manifest is None:
        runtime.manifest = load_manifest(
            {}, data_dir=runtime.data_dir, path=runtime.models_path
        )
    return resolve_little_lisa_model(
        model_id,
        runtime_state={"manifest": runtime.manifest},
        data_dir=runtime.data_dir,
    )


def _last_capability_manifest(runtime: HostRuntime) -> CapabilityManifest | None:
    """Return the last capability manifest stored on the runtime, if any."""
    return getattr(runtime, "last_capability_manifest", None)


def _last_turn_context_dict(runtime: HostRuntime) -> dict[str, Any] | None:
    """Return the last turn context as a dict (for debug rendering compatibility)."""
    ctx = runtime.last_turn_context
    if ctx is None:
        return None
    return dataclasses.asdict(ctx)
