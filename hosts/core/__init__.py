"""hosts.core — transport-neutral host runtime for Lisa framework hosts.

This package owns the shared runtime that ``hosts/cli``, ``hosts/web``, and
``hosts/vscode`` all build on. It wires the kernel's ``TurnHooks`` with
manifest-driven model resolution, MCP pool compilation, provider completion,
context assembly, thread persistence, and host-command dispatch — all without
importing Rich, Typer, aiohttp, or any transport-specific code.

Transport hosts inject a ``TransportCallbacks`` implementation to receive
stage updates, streamed text, final responses, debug context, and errors.
The runtime never writes to a console, browser, or editor directly.

Public surface::

    from hosts.core import HostConfig, HostRuntime, TransportCallbacks, handle_message

    config = HostConfig.from_env()
    runtime = HostRuntime(config)
    await runtime.start()           # starts MCP pool in the host's event loop
    result = await handle_message(runtime, "/search test", transport)
    await runtime.aclose()
"""

from __future__ import annotations

from typing import Any

from hosts.core.callbacks import DelegatingTransport, NullTransport, TransportCallbacks
from hosts.core.config import HostConfig, SliceConfig, StreamingConfig, default_data_dir
from hosts.core.runtime import HostRuntime, TokenCounters, TurnContext
from hosts.core.types import (
    CompletionProvider,
    HostCommandResult,
    ModelResolver,
    ToolExecutor,
)

__all__ = [
    "CompletionProvider",
    "DelegatingTransport",
    "HostCommandResult",
    "HostConfig",
    "HostRuntime",
    "ModelResolver",
    "NullTransport",
    "SliceConfig",
    "StreamingConfig",
    "TokenCounters",
    "ToolExecutor",
    "TransportCallbacks",
    "TurnContext",
    "default_data_dir",
    "handle_message",
]


def __getattr__(name: str) -> Any:  # noqa: PLE0604  # PEP 562 lazy module attribute
    """Lazy-import :func:`handle_message` on first attribute access.

    ``handle_message`` is the single heaviest import in the package (it
    pulls in ``openai``, ``mcp``, and ``kernel.ipsar_fsm``). Deferring it
    keeps ``from hosts.core import HostConfig`` / ``HostRuntime`` (used by
    ``--version``, ``modes``, ``models``, ``verify``) from paying the
    turn-execution import cost. Direct ``from hosts.core.handle_message
    import handle_message`` (used by the REPL and ``once``) is unaffected.
    """
    if name == "handle_message":
        from hosts.core.handle_message import handle_message as _handle_message

        globals()["handle_message"] = _handle_message
        return _handle_message
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")