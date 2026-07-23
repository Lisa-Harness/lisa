"""Turn-hook providers for the host, holding shared turn state as instance attributes.

This module flattens the nested closures that previously lived inside
``_build_turn_hooks`` in ``main.py`` into methods on a single
``LittleLisaTurnHooks`` class. Shared state (``state_provider``) is held as
instance attributes rather than captured by closures, so the hooks are
independently testable.

DESIGN — parameterized construction:

    ``_build_memory_and_state_provider`` accepts resolved config values
    rather than reading ``_RUNTIME_STATE`` or ``os.getenv`` directly. The
    caller in ``main.py`` resolves the environment/runtime values and passes
    them in, keeping this module free of global-state coupling.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from kernel.models import ToolResult
from kernel.slice import SlicePolicy
from kernel.turn_runner import TurnHooks

logger = logging.getLogger(__name__)


class LittleLisaTurnHooks:
    """Holds shared turn state and provides ``TurnHooks`` callback methods.

    Each method corresponds to one of the closures previously nested inside
    ``_build_turn_hooks``. Shared dependencies (memory list callable, state
    provider, completion/model/tool callables) are stored as instance
    attributes so the callbacks remain independently testable without closure
    capture.
    """

    def __init__(
        self,
        *,
        state_provider: Any,
        complete_live: Callable[..., Awaitable[Any]],
        resolve_model: Callable[[str], Any],
        execute_tool: Callable[[str, Any], Awaitable[ToolResult]],
        introspect: Callable[[], Awaitable[Any]] | None = None,
    ) -> None:
        self._state_provider = state_provider
        self._complete_live = complete_live
        self._resolve_model = resolve_model
        self._execute_tool = execute_tool
        self._introspect = introspect

    # ── Live introspection ──────────────────────────────────────────────────
    async def introspect(self) -> Any:
        """Gather a live introspection snapshot from the host runtime.

        Delegates to ``kernel.introspect.gather_introspection`` with the
        host's warm MCP pool, turn log path, and token usage path. When
        the introspect callable was not wired at construction time, returns
        ``None``.
        """
        if self._introspect is None:
            return None
        return await self._introspect()

    # ── Slice delegation ──────────────────────────────────────────────────
    async def post_turn_slice(self, policy: SlicePolicy) -> Any:
        """Run the post-turn slice via the state provider."""
        return await self._state_provider.slice(ctx=None, policy=policy)

    async def pre_turn_slice(self, policy: SlicePolicy) -> Any:
        """Run the pre-turn slice via the state provider."""
        return await self._state_provider.slice(ctx=None, policy=policy)

    async def explicit_slice(self, policy: SlicePolicy) -> Any:
        """Run the explicit (user-invoked) slice via the state provider."""
        return await self._state_provider.slice(ctx=None, policy=policy)

    # ── Completion / model / tool delegation ──────────────────────────────
    async def complete(self, **kwargs: Any) -> Any:
        """Delegate to the live completion callable."""
        return await self._complete_live(**kwargs)

    def resolve_model_hook(self, model_id: str) -> Any:
        """Resolve one model identifier via the host's resolver."""
        return self._resolve_model(model_id)

    async def execute_tool_hook(self, tool_id: str, ctx: Any) -> ToolResult:
        """Execute one tool through the host's tool dispatcher."""
        return await self._execute_tool(tool_id, ctx)

    # ── TurnHooks assembly ────────────────────────────────────────────────
    def to_turn_hooks(self, context: dict[str, Any] | None = None) -> TurnHooks:
        """Build a ``TurnHooks`` dataclass wired to this provider's methods.

        Only the hooks this class owns are populated here. The caller in
        ``main.py`` may further populate operation, system-prompt, workshop,
        and MCP-tool-server hooks on the returned dataclass.
        """
        last_slice_finished_at = self._resolve_last_slice_finished_at(context)
        return TurnHooks(
            resolve_model=self.resolve_model_hook,
            complete=self.complete,
            pre_turn_slice=self.pre_turn_slice,
            post_turn_slice=self.post_turn_slice,
            explicit_slice=self.explicit_slice,
            last_slice_finished_at=last_slice_finished_at,
            state_provider=self._state_provider,
        )

    @staticmethod
    def _resolve_last_slice_finished_at(
        context: dict[str, Any] | None,
    ) -> float | None:
        """Extract a persisted ``last_slice_finished_at`` timestamp from context."""
        if context is None:
            return None
        raw = context.get("last_slice_finished_at")
        if isinstance(raw, (int, float)):
            return float(raw)
        return None
