"""Context budget allocation helpers for prompt assembly.

This module keeps per-turn prompt assembly frugal by applying lane-specific
budgets and a total token ceiling across optional context blocks. The
budgets themselves are host-owned (see ``runtime.config``); the kernel provides
only the allocation mechanics plus a shared chars-per-token estimate.
Whole-request sizing lives in ``request_budget`` and uses the active
model's manifest window.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Literal

from kernel.config import settings

__all__ = [
    "ContextBudgetAllocator",
    "ContextBudgetRecord",
    "ContextLane",
    "RequiredContextBudgetError",
    "model_context_window_tokens",
]

ContextLane = Literal["core", "retrieval", "memory", "reflection", "history", "operational"]


def model_context_window_tokens(model_def: Any) -> int:
    """Return the model's declared context window, or the shared default."""

    declared = int(getattr(model_def, "context_window", 0) or 0)
    return declared if declared > 0 else settings.context_default_window_tokens


class RequiredContextBudgetError(RuntimeError):
    """Raised when a required context section cannot fit the token budget.

    Required sections must not be silently dropped — that would degrade the
    model's prompt by omitting mandatory context (e.g. the system prompt or
    a pinned evidence block). Instead of returning ``""`` and proceeding with
    an incomplete prompt, the allocator raises so the caller can fail the turn
    or adjust the budget. Optional sections continue to return ``""`` on drop.
    """


@dataclass
class ContextBudgetRecord:
    """One allocation record for a single context candidate."""

    lane: ContextLane
    requested_tokens: int
    allocated_tokens: int
    budget_tokens: int = 0
    included: bool = False
    required: bool = False
    dropped_reason: str = ""
    truncated: bool = False


class ContextBudgetAllocator:
    """Apply token budgets across context lanes for one turn."""

    def __init__(
        self,
        *,
        enabled: bool,
        total_tokens: int,
        token_char_ratio: int,
        lane_limits: dict[ContextLane, int],
    ) -> None:
        self.enabled = enabled
        self.total_tokens = max(0, total_tokens)
        self.token_char_ratio = max(1, token_char_ratio)
        self.lane_limits = {lane: max(0, limit) for lane, limit in lane_limits.items()}
        self.total_used_tokens = 0
        self.lane_used_tokens: dict[ContextLane, int] = {lane: 0 for lane in self.lane_limits}
        self.records: dict[str, ContextBudgetRecord] = {}

    @classmethod
    def for_turn(
        cls,
        *,
        enabled: bool,
        token_char_ratio: int = 4,
        lane_limits: dict[ContextLane, int] | None = None,
        total_tokens: int | None = None,
    ) -> ContextBudgetAllocator:
        """Build one allocator instance from host-supplied budgets.

        *enabled* is the host's master switch; when False every section
        passes through unbudgeted. *token_char_ratio* is the
        chars-per-token estimate used for sections that lack their own
        token count. *lane_limits* supplies each lane's token ceiling;
        when omitted, every lane falls back to the allocator's total.
        *total_tokens* is the model-aware ceiling for the turn's assembled
        context. When omitted (or non-positive), the allocator falls back
        to the sum of the lane ceilings so the lane limits remain the
        binding constraint — there is no fixed global default to guess at.
        """

        resolved_lane_limits: dict[ContextLane, int] = (
            dict(lane_limits) if lane_limits is not None else {}
        )
        if total_tokens is None or total_tokens <= 0:
            total_tokens = sum(resolved_lane_limits.values()) if resolved_lane_limits else 0
        return cls(
            enabled=enabled,
            total_tokens=total_tokens,
            token_char_ratio=token_char_ratio,
            lane_limits=resolved_lane_limits,
        )

    def _record_unbudgeted(
        self,
        key: str,
        *,
        lane: ContextLane,
        requested_tokens: int,
        required: bool,
    ) -> None:
        """Record an included section on a disabled budget (everything passes)."""
        self.records[key] = ContextBudgetRecord(
            lane=lane,
            requested_tokens=requested_tokens,
            allocated_tokens=requested_tokens,
            budget_tokens=requested_tokens,
            included=True,
            required=required,
        )

    def _record_dropped(
        self,
        key: str,
        *,
        lane: ContextLane,
        requested_tokens: int,
        allocated_budget: int,
        required: bool,
        dropped_reason: str,
        truncated: bool = False,
    ) -> None:
        """Record a dropped section (budget gate / oversized / truncation-empty)."""
        self.records[key] = ContextBudgetRecord(
            lane=lane,
            requested_tokens=requested_tokens,
            allocated_tokens=0,
            budget_tokens=allocated_budget,
            required=required,
            dropped_reason=dropped_reason,
            truncated=truncated,
        )

    def _render_within_budget(
        self,
        key: str,
        *,
        lane: ContextLane,
        normalized: str,
        requested_tokens: int,
        allocated_budget: int,
        required: bool,
        allow_truncate: bool,
    ) -> tuple[str | None, bool]:
        """Render the section within its allocated budget.

        Returns a ``(rendered, truncated)`` tuple; ``rendered`` is ``None``
        when the section was dropped (oversized with truncation disallowed,
        or truncation produced an empty render). Raises
        :class:`RequiredContextBudgetError` for dropped required sections.
        """
        if requested_tokens <= allocated_budget:
            return normalized, False
        if not allow_truncate:
            self._record_dropped(
                key,
                lane=lane,
                requested_tokens=requested_tokens,
                allocated_budget=allocated_budget,
                required=required,
                dropped_reason="required_too_large" if required else "section_too_large",
            )
            if required:
                raise RequiredContextBudgetError(
                    f"Required context section '{key}' (lane={lane}, "
                    f"{requested_tokens} tokens) exceeds the allocated "
                    f"budget ({allocated_budget} tokens) and truncation "
                    "is not allowed. Either increase the lane budget or "
                    "allow truncation for this section."
                )
            return None, False
        rendered, truncated = self._truncate_to_budget(normalized, allocated_budget)
        if not rendered:
            self._record_dropped(
                key,
                lane=lane,
                requested_tokens=requested_tokens,
                allocated_budget=allocated_budget,
                required=required,
                dropped_reason="truncation_empty",
                truncated=True,
            )
            return None, truncated
        return rendered, truncated

    def allocate(
        self,
        *,
        key: str,
        lane: ContextLane,
        text: str,
        required: bool = False,
        allow_truncate: bool = False,
    ) -> str:
        """Return one elected text block or an empty string when dropped.

        Oversized sections are dropped whole by default so semantic prompt
        sections do not arrive half-cut. Callers must opt in to truncation for
        compact generated summaries that are safe to shorten.
        """

        normalized = text.strip()
        if not normalized:
            return ""

        requested_tokens = self._estimate_tokens(normalized)
        if not self.enabled:
            self._record_unbudgeted(
                key,
                lane=lane,
                requested_tokens=requested_tokens,
                required=required,
            )
            return normalized

        lane_limit = self.lane_limits.get(lane, self.total_tokens)
        lane_remaining = max(0, lane_limit - self.lane_used_tokens.get(lane, 0))
        total_remaining = max(0, self.total_tokens - self.total_used_tokens)
        allocated_budget = min(lane_remaining, total_remaining)

        if allocated_budget <= 0:
            self._record_dropped(
                key,
                lane=lane,
                requested_tokens=requested_tokens,
                allocated_budget=allocated_budget,
                required=required,
                dropped_reason="budget_exhausted",
            )
            if required:
                raise RequiredContextBudgetError(
                    f"Required context section '{key}' (lane={lane}, "
                    f"{requested_tokens} tokens) was dropped: budget exhausted "
                    f"(total_remaining={total_remaining}, lane_remaining={lane_remaining})."
                )
            return ""

        rendered, truncated = self._render_within_budget(
            key,
            lane=lane,
            normalized=normalized,
            requested_tokens=requested_tokens,
            allocated_budget=allocated_budget,
            required=required,
            allow_truncate=allow_truncate,
        )
        if rendered is None:
            return ""

        allocated_tokens = self._estimate_tokens(rendered)
        if allocated_tokens <= 0:
            self._record_dropped(
                key,
                lane=lane,
                requested_tokens=requested_tokens,
                allocated_budget=allocated_budget,
                required=required,
                dropped_reason="empty_after_trim",
                truncated=truncated,
            )
            return ""

        self.total_used_tokens += allocated_tokens
        self.lane_used_tokens[lane] = self.lane_used_tokens.get(lane, 0) + allocated_tokens
        self.records[key] = ContextBudgetRecord(
            lane=lane,
            requested_tokens=requested_tokens,
            allocated_tokens=allocated_tokens,
            budget_tokens=allocated_budget,
            included=True,
            required=required,
            truncated=truncated,
        )
        return rendered

    def summary(self) -> dict[str, object]:
        """Return one compact allocation summary for telemetry/debug."""

        dropped = [
            key
            for key, record in self.records.items()
            if record.allocated_tokens <= 0 and record.dropped_reason
        ]
        truncated = [key for key, record in self.records.items() if record.truncated]
        lane_totals = {
            lane: {
                "used_tokens": self.lane_used_tokens.get(lane, 0),
                "max_tokens": self.lane_limits.get(lane, 0),
            }
            for lane in self.lane_limits
        }
        return {
            "enabled": self.enabled,
            "total_tokens": self.total_tokens,
            "used_tokens": self.total_used_tokens,
            "remaining_tokens": max(0, self.total_tokens - self.total_used_tokens),
            "lane_totals": lane_totals,
            "dropped_keys": dropped,
            "truncated_keys": truncated,
            "records": {
                key: {
                    "lane": record.lane,
                    "requested_tokens": record.requested_tokens,
                    "allocated_tokens": record.allocated_tokens,
                    "budget_tokens": record.budget_tokens,
                    "included": record.included,
                    "required": record.required,
                    "drop_reason": record.dropped_reason,
                    "dropped_reason": record.dropped_reason,
                    "truncated": record.truncated,
                }
                for key, record in self.records.items()
            },
        }

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, ceil(len(text) / self.token_char_ratio))

    def _truncate_to_budget(self, text: str, budget_tokens: int) -> tuple[str, bool]:
        max_chars = budget_tokens * self.token_char_ratio
        if max_chars <= 0:
            return "", True
        if len(text) <= max_chars:
            return text, False

        trimmed = text[:max_chars]
        split_markers = ("\n## ", "\n\n", "\n- ", ". ")
        cutoff = int(max_chars * 0.6)
        for marker in split_markers:
            idx = trimmed.rfind(marker)
            if idx >= cutoff:
                trimmed = trimmed[:idx]
                break

        return trimmed.strip(), True
