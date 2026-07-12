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

from hosts.core.callbacks import DelegatingTransport, NullTransport, TransportCallbacks
from hosts.core.config import HostConfig, SliceConfig, StreamingConfig
from hosts.core.handle_message import handle_message
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
    "handle_message",
]