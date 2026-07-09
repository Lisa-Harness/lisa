"""Turn-hook providers for Little Lisa, holding shared turn state as instance attributes.

This module flattens the nine nested closures that previously lived inside
``_build_turn_hooks`` in ``main.py`` into methods on a single
``LittleLisaTurnHooks`` class. Shared state (``memory_store``,
``state_provider``) is held as instance attributes rather than captured by
closures, so the hooks are independently testable.

DESIGN — parameterized construction:

    ``_build_memory_and_state_provider`` accepts resolved config values
    rather than reading ``_RUNTIME_STATE`` or ``os.getenv`` directly. The
    caller in ``main.py`` resolves the environment/runtime values and passes
    them in, keeping this module free of global-state coupling.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kernel.models import (
    MemoryRecallAnswer,
    MemoryRecallEvidence,
    MemoryRecord,
    ToolResult,
)
from kernel.slice import SlicePolicy, SliceTuning
from kernel.turn_runner import TurnHooks

if TYPE_CHECKING:
    from little_lisa.main import LittleLisaStateProvider
    from little_lisa.model_runtime import StubCompletion, StubModelDefinition

logger = logging.getLogger(__name__)

# Minimum token length counted for lexical-overlap ranking. Mirrors the
# ``_RANKED_CONTEXT_MIN_TOKEN_LEN`` constant used in ``main.py`` so the
# ranking algorithm here is byte-for-byte identical to the original closure.
_RANKED_MIN_TOKEN_LEN = 3


def _tokenize(text: str) -> set[str]:
    """Split text into lowercase tokens of at least ``_RANKED_MIN_TOKEN_LEN`` chars."""
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) >= _RANKED_MIN_TOKEN_LEN
    }


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
        memory_list_fn: "Callable[..., Awaitable[list[MemoryRecord]]]",
        state_provider: LittleLisaStateProvider,
        complete_live: Callable[..., Awaitable[StubCompletion]],
        resolve_model: Callable[[str], StubModelDefinition],
        execute_tool: Callable[[str, Any], Awaitable[ToolResult]],
    ) -> None:
        self._memory_list_fn = memory_list_fn
        self._state_provider = state_provider
        self._complete_live = complete_live
        self._resolve_model = resolve_model
        self._execute_tool = execute_tool

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

    # ── Memory recall (ranking logic copied verbatim from main.py) ────────
    async def answer_memory_recall(
        self,
        *,
        query: str,
        user_slug: str,
        workspace_id: str,
        memory_scope: str = "personal",
        surface_id: str = "",
        limit: int = 5,
        **_kwargs: Any,
    ) -> MemoryRecallAnswer:
        """Query the memory store and rank records by lexical-token overlap.

        The ranking algorithm is preserved exactly from the original
        closure in ``main.py``: tokens of at least ``_RANKED_MIN_TOKEN_LEN``
        characters are intersected between the query and each record's
        content + ``content_json``, combined with a small salience bonus,
        and sorted by score then record id.
        """
        records = await self._memory_list_fn(
            user_slug=user_slug,
            workspace_id=workspace_id,
            memory_scope=memory_scope,
            surface_id=surface_id,
            limit=max(1, min(int(limit or 1), 20)),
        )
        if not records:
            return MemoryRecallAnswer(
                query=query,
                found=False,
                answer=f"No memory matches found for: {query}",
            )

        query_tokens = _tokenize(query)
        ranked: list[tuple[float, MemoryRecord]] = []
        for record in records:
            haystack = f"{record.content} {json.dumps(record.content_json, ensure_ascii=True)}"
            overlap = len(query_tokens.intersection(_tokenize(haystack)))
            if overlap <= 0 and query_tokens:
                continue
            lexical_score = overlap / max(len(query_tokens), 1) if query_tokens else 1.0
            combined = min(1.0, lexical_score + max(0.0, record.salience) * 0.2)
            ranked.append((combined, record))

        ranked.sort(key=lambda item: (item[0], item[1].id), reverse=True)
        top = ranked[: max(1, min(int(limit or 1), 8))]
        if not top:
            return MemoryRecallAnswer(
                query=query,
                found=False,
                answer=f"No memory matches found for: {query}",
            )

        evidence: list[MemoryRecallEvidence] = []
        for score, record in top:
            evidence.append(
                MemoryRecallEvidence(
                    memory_id=record.id,
                    created_at=record.created_at,
                    content=record.content,
                    memory_scope=record.memory_scope,
                    surface_id=record.surface_id,
                    source_user_slug=record.source_user_slug,
                    operation_id=record.operation_id,
                    memory_tier=record.memory_tier,
                    salience=record.salience,
                    combined_score=round(score, 3),
                    lexical_score=round(score, 3),
                    recency_score=1.0,
                    match_reasons=["lexical overlap"],
                )
            )

        summary = evidence[0].content
        confidence = evidence[0].combined_score if evidence else 0.0
        return MemoryRecallAnswer(
            query=query,
            found=True,
            answer=summary,
            confidence=round(confidence, 3),
            used_semantic_search=False,
            auto_deep_used=False,
            evidence=evidence,
        )

    # ── Completion / model / tool delegation ──────────────────────────────
    async def complete(self, **kwargs: Any) -> StubCompletion:
        """Delegate to the live completion callable."""
        return await self._complete_live(**kwargs)

    def resolve_model_hook(self, model_id: str) -> StubModelDefinition:
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
