"""Lisa Guard MCP server — content safety validation.

Tools:
- ``validate_input`` — validate user input before it reaches the model
- ``validate_output`` — validate model output before it reaches the user
- ``guard_configured`` — report backend configuration

The default backend is ``passthrough`` (allows everything, sanitizes nothing).
Backend selection via ``--backend`` CLI arg (default from ``LISA_GUARD_BACKEND``
env or ``passthrough``). When a guard blocks, the server logs a structured
audit record to stderr.

The server owns audit logging — hosts using ``MCPInputGuard``/``MCPOutputGuard``
adapters do not need an event bus for guardrail audits.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

__all__ = ["mcp", "main", "init_backend"]

mcp = FastMCP("Lisa Guard Server")

_backend: str = "passthrough"
_readonly: bool = False


def init_backend(backend: str, *, readonly: bool = False) -> None:
    """Initialize the guard backend. Called by ``main()`` after arg parsing."""
    global _backend, _readonly
    _backend = backend
    _readonly = readonly


def _log_blocked(
    *,
    direction: str,
    surface_id: str,
    blocked_reason: str,
    text_len: int,
) -> None:
    """Log a structured audit record for a blocked guard check."""
    if os.getenv("LISA_GUARD_LOG_BLOCKED", "1") != "0":
        record = {
            "event": "guard.blocked",
            "direction": direction,
            "surface_id": surface_id,
            "blocked_reason": blocked_reason,
            "text_len": text_len,
        }
        print(json.dumps(record), file=sys.stderr, flush=True)


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
    if _backend == "passthrough":
        return json.dumps({
            "allowed": True,
            "blocked_reason": None,
            "sanitized_text": None,
            "annotations": {},
        })
    # Future backends (azure_content_safety, etc.) plug in here.
    return json.dumps({
        "allowed": True,
        "blocked_reason": None,
        "sanitized_text": None,
        "annotations": {"backend": _backend},
    })


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
    if _backend == "passthrough":
        return json.dumps({
            "allowed": True,
            "blocked_reason": None,
            "sanitized_text": None,
            "annotations": {},
        })
    return json.dumps({
        "allowed": True,
        "blocked_reason": None,
        "sanitized_text": None,
        "annotations": {"backend": _backend},
    })


@mcp.tool()
async def guard_configured() -> str:
    """Report whether the guard is configured and which backend is active."""
    return json.dumps({"configured": True, "backend": _backend})


def main() -> None:
    """Entry point for the lisa-guard MCP server (stdio transport).

    Parses the standard CLI contract via the shared argparse helper, plus
    ``--backend`` for backend selection. ``--readonly`` is not supported
    (guards are stateless) but ``--data-dir`` is required for consistency.
    """
    from lisa_mcp_config.cli import build_parser, configure_logging, resolve_data_dir

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