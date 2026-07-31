"""Lisa Guard MCP server — thin facade over :class:`sdk.guard.GuardBackend`.

Tools:
- ``validate_input`` — validate user input before it reaches the model
- ``validate_output`` — validate model output before it reaches the user
- ``guard_configured`` — report backend configuration

This is a transport-only wrapper. The SDK owns backend selection, the
result shape, and the blocked-audit logger; the server owns ``FastMCP``
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

from mcp.server.fastmcp import FastMCP

from sdk.guard import GuardBackend, format_guard_result, get_backend, log_blocked_guard

logger = logging.getLogger(__name__)

__all__ = ["init_backend", "main", "mcp"]

mcp = FastMCP("Lisa Guard Server")

_backend_name: str = "passthrough"
_backend: GuardBackend | None = None


def init_backend(backend: str, *, readonly: bool = False) -> None:
    """Initialize the guard backend. Called by ``main()`` after arg parsing.

    ``readonly`` is accepted for CLI-contract consistency but inert: guards
    are stateless and read-only by design.
    """
    global _backend_name, _backend
    del readonly
    _backend_name = backend
    _backend = get_backend(backend)


def _backend_instance() -> GuardBackend:
    if _backend is None:
        raise RuntimeError("lisa-guard backend is not initialized. Call init_backend(name).")
    return _backend


@mcp.tool()
async def validate_input(
    query: str,
    surface_type: str = "personal_dm",
    surface_id: str = "",
    memory_scope: str = "personal",
    thread_owner_id: str = "",
) -> str:
    """Validate user input before it reaches the model.

    Returns JSON: ``{"allowed": bool, "blocked_reason": str|null,
    "sanitized_text": str|null, "annotations": {}}``.
    """
    result = _backend_instance().validate_input(
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
    if _backend_name != "passthrough":
        result.annotations = {**result.annotations, "backend": _backend_name}
    return format_guard_result(result)


@mcp.tool()
async def validate_output(
    text: str,
    surface_type: str = "personal_dm",
    surface_id: str = "",
    memory_scope: str = "personal",
    thread_owner_id: str = "",
) -> str:
    """Validate model output before it reaches the user.

    Returns the same JSON shape as ``validate_input``.
    """
    result = _backend_instance().validate_output(
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
    if _backend_name != "passthrough":
        result.annotations = {**result.annotations, "backend": _backend_name}
    return format_guard_result(result)


@mcp.tool()
async def guard_configured() -> str:
    """Report whether the guard is configured and which backend is active."""
    return json.dumps({"configured": True, "backend": _backend_name})


def main() -> None:
    """Entry point for the lisa-guard MCP server (stdio transport).

    Parses the standard CLI contract via the shared argparse helper, plus
    ``--backend`` for backend selection. ``--readonly`` is not supported
    (guards are stateless) but ``--data-dir`` is required for consistency.
    """
    from mcp_server.config.cli import build_parser, configure_logging, resolve_data_dir

    parser = build_parser("lisa-guard")
    parser.add_argument(
        "--backend",
        type=str,
        default=os.getenv("LISA_GUARD_BACKEND", "passthrough"),
        help="Guard backend selection (default: passthrough).",
    )
    args = parser.parse_args()

    log = configure_logging(args.log_level, server_name="lisa-guard")
    data_dir = resolve_data_dir(args.data_dir)
    log.info("lisa-guard starting (data_dir=%s, backend=%s)", data_dir, args.backend)

    init_backend(args.backend)
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
