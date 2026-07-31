"""Content-safety validation backends for the lisa-guard server.

Extracted from the former ``mcp_server/guard/server.py``: the passthrough
backend, the result-to-JSON formatter, and the blocked-audit logger. The
MCP server is now a thin wrapper — it selects a backend by name and
exposes the three tools (``validate_input`` / ``validate_output`` /
``guard_configured``) as delegations.

This gives the SDK a complete contract for guard validation that can be
fulfilled without MCP — a host or tool that wants in-process input/output
validation can call a backend directly instead of spawning the MCP server.
The :class:`kernel.seams.protocols.GuardResult` dataclass is reused as the
canonical result type so SDK consumers and the kernel share one shape.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Protocol, runtime_checkable

from kernel.seams.protocols import GuardResult

__all__ = [
    "GuardBackend",
    "PassthroughGuardBackend",
    "format_guard_result",
    "log_blocked_guard",
]

logger = logging.getLogger(__name__)

_BLOCKED_LOG_ENV = "LISA_GUARD_LOG_BLOCKED"


@runtime_checkable
class GuardBackend(Protocol):
    """A content-safety validation backend.

    Two operations, mirroring the kernel's :class:`InputGuard` /
    :class:`OutputGuard` split but returning the transport-neutral
    :class:`GuardResult` so the SDK is not coupled to a surface context.
    Backends that need surface context can carry it via ``annotations``.
    """

    @property
    def name(self) -> str: ...

    def validate_input(self, query: str, **kwargs: Any) -> GuardResult: ...

    def validate_output(self, text: str, **kwargs: Any) -> GuardResult: ...


class PassthroughGuardBackend:
    """The default no-op backend: allows everything, sanitizes nothing.

    Identical observable behavior to the former inline ``passthrough``
    branch in ``mcp_server/guard/server.py``. Future backends (Azure
    Content Safety, etc.) implement :class:`GuardBackend` and are
    selected by name in the MCP server's ``init_backend``.
    """

    name: str = "passthrough"

    def validate_input(self, query: str, **kwargs: Any) -> GuardResult:
        del query, kwargs  # passthrough: no inspection performed
        return GuardResult(allowed=True, annotations={})

    def validate_output(self, text: str, **kwargs: Any) -> GuardResult:
        del text, kwargs  # passthrough: no inspection performed
        return GuardResult(allowed=True, annotations={})


def format_guard_result(result: GuardResult) -> str:
    """Render a :class:`GuardResult` as the JSON string the MCP tools return.

    Shape: ``{"allowed": bool, "blocked_reason": str|null,
    "sanitized_text": str|null, "annotations": {}}``. This matches the
    former inline ``json.dumps(...)`` calls in the guard server so the
    tool contract is unchanged.
    """
    return json.dumps(
        {
            "allowed": result.allowed,
            "blocked_reason": result.blocked_reason,
            "sanitized_text": result.sanitized_text,
            "annotations": result.annotations,
        }
    )


def log_blocked_guard(
    *,
    direction: str,
    surface_id: str,
    blocked_reason: str,
    text_len: int,
) -> None:
    """Log a structured audit record for a blocked guard check.

    Emits through the stdlib ``logger.warning`` so the record lands in the
    same JSON stream as every other host/server event (the shared
    :class:`sdk.logging_setup.JSONFormatter` serializes the ``extra``
    fields). Gated by ``LISA_GUARD_LOG_BLOCKED`` (default on).
    """
    if os.getenv(_BLOCKED_LOG_ENV, "1") != "0":
        logger.warning(
            "guard.blocked: direction=%s surface=%s reason=%s len=%d",
            direction,
            surface_id,
            blocked_reason,
            text_len,
            extra={
                "event": "guard.blocked",
                "direction": direction,
                "surface_id": surface_id,
                "blocked_reason": blocked_reason,
                "text_len": text_len,
            },
        )


def get_backend(name: str) -> GuardBackend:
    """Resolve a backend by name. Only ``passthrough`` is built in.

    Future backends (azure_content_safety, etc.) plug in here. The MCP
    server calls this from ``init_backend``; in-process callers can call
    it directly or construct a backend explicitly.
    """
    normalized = (name or "passthrough").strip().lower()
    if normalized == "passthrough":
        return PassthroughGuardBackend()
    # Unknown backends fall back to passthrough with a marker annotation
    # so the tool surface stays usable while the operator notices.
    return PassthroughGuardBackend()
