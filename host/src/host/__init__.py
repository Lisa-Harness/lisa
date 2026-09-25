"""host — transport-neutral host runtime for Lisa Harness hosts.

This package owns the shared runtime that ``hosts/cli``
builds on. It wires the kernel's ``TurnHooks`` with
manifest-driven model resolution, MCP pool compilation, provider completion,
context assembly, thread persistence, and host-command dispatch — all without
importing Rich, Typer, aiohttp, or any transport-specific code.

Transport hosts inject a ``TransportCallbacks`` implementation to receive
stage updates, streamed text, final responses, debug context, and errors.
The runtime never writes to a console, browser, or editor directly.

Public surface::

    from host import HostConfig, HostRuntime, TransportCallbacks, handle_message

    config = HostConfig.from_env()
    runtime = HostRuntime(config)
    await runtime.start()           # starts MCP pool in the host's event loop
    result = await handle_message(runtime, "/search test", transport)
    await runtime.aclose()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from host.callbacks import DelegatingTransport, NullTransport, TransportCallbacks
from host.config import HostConfig, SliceConfig, StreamingConfig, default_data_dir
from host.types import (
    CompletionProvider,
    HostCommandResult,
    ModelResolver,
    ToolExecutor,
)

if TYPE_CHECKING:
    # ``host.runtime`` is the heaviest submodule (pulls
    # ``kernel.ipsar_fsm`` + ``openai`` via ``session_context``). It is
    # resolved lazily by ``__getattr__`` below so importing
    # ``HostConfig`` / ``callbacks`` / ``types`` from the package does not
    # drag it in. The ``TYPE_CHECKING`` alias keeps these names visible to
    # type checkers for annotation use without an eager runtime import.
    from host.runtime import HostRuntime as HostRuntime
    from host.runtime import TokenCounters as TokenCounters
    from host.runtime import TurnContext as TurnContext

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


def __getattr__(name: str) -> Any:  # PEP 562 lazy module attribute
    """Lazy-import the heavy runtime symbols on first attribute access.

    ``host.runtime`` is the heaviest submodule in the package (it
    pulls in ``openai``, ``mcp``, and ``kernel.ipsar_fsm`` via
    ``session_context``); ``handle_message`` is the heaviest *function*
    (the turn dispatcher). Deferring both keeps ``from host import
    HostConfig`` / ``HostRuntime`` / ``TransportCallbacks`` (used by
    ``--version``, ``modes``, ``models``, ``verify``) from paying the
    ~2s runtime or ~3s turn-execution import cost. Direct submodule
    imports (``from host.runtime import HostRuntime``, ``from
    host.handle_message import handle_message``) are unaffected and
    remain the fast path for code that actually needs the symbol.
    """
    if name == "handle_message":
        from host.handle_message import handle_message as _handle_message

        globals()["handle_message"] = _handle_message
        return _handle_message
    if name in {"HostRuntime", "TokenCounters", "TurnContext"}:
        from host import runtime as _runtime

        value = getattr(_runtime, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
