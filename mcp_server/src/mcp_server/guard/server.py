"""Lisa Guard MCP server — thin facade over :class:`sdk.guard.GuardBackend`.

Tools:
- ``validate_input`` — validate user input before it reaches the model
- ``validate_output`` — validate model output before it reaches the user
- ``guard_configured`` — report backend configuration

This is a transport-only wrapper. The SDK owns backend selection, the
result shape, and the blocked-audit logger; the server owns ``MCPServer``
wiring, ``--backend`` arg parsing, and the process entrypoint. Hosts or
tools that want in-process guard validation can use a backend directly
without spawning this server.

The default backend is ``passthrough`` (allows everything, sanitizes
nothing). Backend selection via ``--backend`` CLI arg (default from
``LISA_GUARD_BACKEND`` env or ``passthrough``). When a guard blocks, the
SDK backend logs a structured audit record (see
:func:`sdk.guard.log_blocked_guard`).
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.context import Context
from mcp_types import ToolAnnotations
from sdk.guard import GuardBackend, format_guard_result, get_backend, log_blocked_guard

logger = logging.getLogger(__name__)

__all__ = ["GuardDeps", "init_backend", "main", "mcp", "set_deps_for_test"]

# ── Lifespan deps ───────────────────────────────────────────────────────────


@dataclass
class GuardDeps:
    """Server-scoped dependencies owned by the lifespan."""

    backend: GuardBackend
    backend_name: str = "passthrough"


# Test seam: populated by the lifespan in production, or by
# ``set_deps_for_test`` in tests (direct ``call_tool`` does not run the
# lifespan). This is an explicit seam, not a hidden production singleton.
_deps: GuardDeps | None = None


def set_deps_for_test(deps: GuardDeps) -> None:
    """Test-only seam: inject ``GuardDeps`` without spawning stdio.

    The lifespan owns this in production; tests bypass it because the
    SDK's direct ``call_tool`` path does not run the lifespan.
    """
    global _deps  # noqa: PLW0603 — test seam; lifespan owns this in production
    _deps = deps


def _resolve_deps(ctx: Context[Any, Any] | None) -> GuardDeps:
    """Return the active deps, preferring lifespan context when available."""
    if ctx is not None:
        try:
            rc = ctx.request_context
            deps = rc.lifespan_context
            if isinstance(deps, GuardDeps):
                return deps
        except ValueError:
            pass  # direct call_tool path — no request_context
    if _deps is None:
        raise RuntimeError("lisa-guard is not initialized. Call init_backend(name).")
    return _deps


# ── CLI config holder (populated by _init before run()) ────────────────────


@dataclass
class _GuardConfig:
    backend: str = "passthrough"


_cfg = _GuardConfig()


@asynccontextmanager
async def _lifespan(app: MCPServer[Any]) -> AsyncIterator[GuardDeps]:
    """Construct the guard backend from resolved CLI config and yield it."""
    yield GuardDeps(backend=get_backend(_cfg.backend), backend_name=_cfg.backend)


mcp = MCPServer("Lisa Guard Server", lifespan=_lifespan)


def init_backend(backend: str, *, readonly: bool = False) -> None:
    """Initialize the guard backend. Called by ``main()`` after arg parsing.

    Kept as a thin back-compat shim. ``readonly`` is accepted for
    CLI-contract consistency but inert: guards are stateless and read-only
    by design.
    """
    del readonly
    set_deps_for_test(GuardDeps(backend=get_backend(backend), backend_name=backend))


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def validate_input(  # noqa: PLR0917 — one parameter per SurfaceContext field; the signature IS the MCP schema
    query: str,
    surface_type: str = "personal_dm",
    surface_id: str = "",
    memory_scope: str = "personal",
    thread_owner_id: str = "",
    ctx: Context[GuardDeps, Any] | None = None,
) -> str:
    """Validate user input before it reaches the model.

    Returns JSON: ``{"allowed": bool, "blocked_reason": str|null,
    "sanitized_text": str|null, "annotations": {}}``.
    """
    deps = _resolve_deps(ctx)
    result = deps.backend.validate_input(
        query,
        surface_type=surface_type,
        surface_id=surface_id,
        memory_scope=memory_scope,
        thread_owner_id=thread_owner_id,
    )
    if not result.allowed:
        log_blocked_guard(
            direction="input",
            surface_id=surface_id,
            blocked_reason=result.blocked_reason or "",
            text_len=len(query),
        )
    if deps.backend_name != "passthrough":
        result.annotations = {**result.annotations, "backend": deps.backend_name}
    return format_guard_result(result)


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def validate_output(  # noqa: PLR0917 — one parameter per SurfaceContext field; the signature IS the MCP schema
    text: str,
    surface_type: str = "personal_dm",
    surface_id: str = "",
    memory_scope: str = "personal",
    thread_owner_id: str = "",
    ctx: Context[GuardDeps, Any] | None = None,
) -> str:
    """Validate model output before it reaches the user.

    Returns the same JSON shape as ``validate_input``.
    """
    deps = _resolve_deps(ctx)
    result = deps.backend.validate_output(
        text,
        surface_type=surface_type,
        surface_id=surface_id,
        memory_scope=memory_scope,
        thread_owner_id=thread_owner_id,
    )
    if not result.allowed:
        log_blocked_guard(
            direction="output",
            surface_id=surface_id,
            blocked_reason=result.blocked_reason or "",
            text_len=len(text),
        )
    if deps.backend_name != "passthrough":
        result.annotations = {**result.annotations, "backend": deps.backend_name}
    return format_guard_result(result)


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def guard_configured(
    ctx: Context[GuardDeps, Any] | None = None,
) -> str:
    """Report whether the guard is configured and which backend is active."""
    deps = _resolve_deps(ctx)
    return json.dumps({"configured": True, "backend": deps.backend_name})


def _add_guard_args(parser: Any) -> None:
    """Add ``--backend`` to the standard parser."""
    parser.add_argument(
        "--backend",
        type=str,
        default=os.getenv("LISA_GUARD_BACKEND", "passthrough"),
        help="Guard backend selection (default: passthrough).",
    )


def _init(args: Any) -> None:
    """Populate ``_cfg`` from parsed args so the lifespan can build the backend."""
    _cfg.backend = str(getattr(args, "backend", None) or "passthrough")
    # Also populate the test seam so the server is usable immediately after
    # init (the lifespan only runs during a real stdio session).
    set_deps_for_test(GuardDeps(backend=get_backend(_cfg.backend), backend_name=_cfg.backend))


def main() -> None:
    """Entry point for the lisa-guard MCP server (stdio transport).

    Parses the standard CLI contract via the shared ``run()`` helper, plus
    ``--backend`` for backend selection. ``--readonly`` is not supported
    (guards are stateless) but ``--data-dir`` is required for consistency.
    """
    from mcp_server.config.cli import run

    run("lisa-guard", mcp, _init, add_args=_add_guard_args)


if __name__ == "__main__":
    main()
