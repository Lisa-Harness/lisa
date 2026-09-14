"""Type aliases for host-runtime callable seams.

These are the transport-neutral callable shapes that ``HostRuntime`` and
``handle_message`` depend on. They are ``Protocol`` objects (not concrete
implementations) so any host can inject its own provider, model resolver,
or tool executor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ModelResolver(Protocol):
    """Resolve a model alias or designation to a concrete model definition.

    The returned object must have ``model_id: str``, ``provider: str``, and
    ``reasoning_model: bool`` attributes (the ``StubModelDefinition`` shape
    used by the kernel's ``TurnHooks.resolve_model``).
    """

    def __call__(self, model_id: str) -> Any: ...


@runtime_checkable
class CompletionProvider(Protocol):
    """Execute a live model completion against configured provider APIs.

    Must accept keyword arguments matching the kernel's completion call
    shape (``messages``, ``model_def``, ``max_tokens``, ``temperature``,
    ``tools``, ``response_format``) and return an object with ``.content``
    (str), ``.tool_calls`` (list[dict]), and token-count fields.
    """

    async def __call__(self, **kwargs: Any) -> Any: ...


@runtime_checkable
class ToolExecutor(Protocol):
    """Execute a tool by ID and return a ``ToolResult``."""

    async def __call__(self, tool_id: str, ctx: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class HostCommandResult:
    """Structured result from a host-command dispatch.

    ``handle_message`` returns this for every turn. Transport hosts render
    ``text`` to the user; ``data`` is available for richer UI rendering
    (e.g. web SSE payloads).

    ``operator_exit`` carries the pause/stop/terminate discriminator when the
    turn ended via an operator signal or a cap path that degrades to one. The
    transport reads it to decide whether to return the prompt (stop/terminate)
    or hold the turn open (pause). Empty string for normal turns.

    ``yielded_question`` carries the model's question when the turn ended via
    the ``yield`` terminal action (the model asked the user something instead
    of producing a final answer). Non-empty means the turn did NOT complete —
    one-shot hosts use this to fail fast (there is no interactive loop to
    answer the question) while conversational hosts surface it inline.
    """

    text: str
    data: dict[str, Any] | None = None
    is_error: bool = False
    operator_exit: str = ""
    yielded_question: str = ""
