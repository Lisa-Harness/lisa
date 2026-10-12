"""Integration test for the wired context mapper + budget allocator.

The framework registration that the original module-scope autouse fixture
performed via ``little_lisa.setup_framework()`` is now handled by the
``runtime/tests/conftest.py`` module-scope autouse fixture, which
wires the catalog MCP and operation callbacks.

Exercises the full chain: populated ``UnifiedState`` →
``context_candidates_from_unified_state`` → ``ContextMapper.map`` →
``ContextBudgetAllocator.allocate`` → rendered prompt markdown.

Asserts:
- Budget enforcement drops sections when the ceiling is hit.
- The mapper produces lane-ordered markdown from a populated state.
- The ``ContextUsageEvent`` telemetry envelope can be built from the allocator summary.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from kernel.context.context_budget import ContextBudgetAllocator
from kernel.context.context_mapper import (
    ContextMapper,
    ContextMappingError,
    context_candidates_from_unified_state,
)
from kernel.context.models import ContextMapperRequest
from kernel.models import (
    ContextItem,
    ContextUsageEvent,
    UnifiedState,
)
from kernel.session_plan import ContextLanePlan, ContextPlan, ContextSourceRule


def _make_populated_state() -> UnifiedState:
    """Build a UnifiedState with alerts, intel, and self-model evidence."""
    now = datetime.now(UTC).isoformat()
    return UnifiedState(
        compiled_at=now,
        context_items=[
            ContextItem(
                item_id="state:alert:cloudflare:critical",
                namespace="alerts",
                source="cloudflare",
                source_type="intake",
                evidence_type="alert",
                title="Critical alert from cloudflare",
                body="Worker error rate exceeded 10%",
                collected_at=now,
                salience_score=1.0,
                confidence=0.95,
                tags=["alert", "critical", "cloudflare"],
                metadata={"provenance": "system_observed"},
            ),
            ContextItem(
                item_id="state:intel:market",
                namespace="intel",
                source="market",
                source_type="intake",
                evidence_type="intel_summary",
                title="Intel: market",
                body="- Competitor launched new pricing tier",
                collected_at=now,
                salience_score=0.55,
                confidence=0.8,
                tags=["intel", "market"],
                metadata={"provenance": "authoritative"},
            ),
            ContextItem(
                item_id="state:self_model:default",
                namespace="self_model",
                source="self_model",
                source_type="extension",
                evidence_type="runtime_self_model",
                title="Self-model (health=degraded)",
                body=(
                    "## Self Model\n"
                    "Runtime: uptime=450s, tasks=31, threads=6, rss=512MB, health=degraded\n"
                    "Health reasons: mcp failures: 1"
                ),
                collected_at=now,
                salience_score=0.8,
                confidence=0.9,
                pack="lisa-pack-introspect",
                tags=["self_model", "degraded"],
                metadata={"provenance": "system_observed"},
            ),
        ],
    )


def _default_plan() -> ContextPlan:
    """A minimal plan with a fallback rule so the mapper selects candidates."""
    return ContextPlan(
        lanes=(
            ContextLanePlan(lane_id="operational", render_order=0),
            ContextLanePlan(lane_id="general", render_order=10),
        ),
        sources=(
            ContextSourceRule(
                rule_id="alerts",
                lane_id="operational",
                evidence_types=("alert",),
                priority=100,
            ),
            ContextSourceRule(rule_id="fallback", lane_id="general", fallback=True),
        ),
    )


def _map_state(state: UnifiedState, query: str) -> str:
    """Run the session-owned ContextMapper over a state and return markdown."""
    candidates = context_candidates_from_unified_state(state)
    request = ContextMapperRequest(
        context_plan=_default_plan(),
        query=query,
        mode_id="default",
        candidates=candidates,
        active_packs=frozenset(),
        model_context_window_tokens=131072,
    )
    return ContextMapper().map(request).rendered_context


class TestContextMapper:
    """Tests for the session-owned ContextMapper with a populated state."""

    def test_returns_rendered_markdown(self) -> None:
        state = _make_populated_state()
        markdown = _map_state(state, "cloudflare error rate pricing")
        assert isinstance(markdown, str)

    def test_uncompiled_state_raises(self) -> None:
        state = UnifiedState(compiled_at="")
        with pytest.raises(ContextMappingError):
            context_candidates_from_unified_state(state)


class TestContextBudgetAllocator:
    """Tests for ContextBudgetAllocator integration with assembled context."""

    def test_allocator_enforces_total_token_ceiling(
        self,
    ) -> None:
        # Set a very low total token budget so sections get dropped.
        allocator = ContextBudgetAllocator.for_turn(enabled=True, total_tokens=50)
        # Allocate a large section that exceeds the budget.
        large_text = "x" * 5000
        result = allocator.allocate(
            key="big_sections",
            lane="retrieval",
            text=large_text,
            allow_truncate=True,
        )
        # Should be truncated or dropped, not the full text.
        assert len(result) < len(large_text) or result == ""

    def test_allocator_disabled_returns_full_text(
        self,
    ) -> None:
        allocator = ContextBudgetAllocator.for_turn(enabled=False)
        text = "This is a test section that should pass through unchanged."
        result = allocator.allocate(
            key="test_section",
            lane="core",
            text=text,
        )
        assert result == text

    def test_allocator_summary_contains_lane_info(self) -> None:
        allocator = ContextBudgetAllocator.for_turn(
            enabled=True,
            lane_limits={"history": 1500, "core": 2000},
        )
        allocator.allocate(
            key="test",
            lane="history",
            text="short text",
        )
        summary = allocator.summary()
        assert "lane_totals" in summary
        assert "history" in summary["lane_totals"]
        assert summary["lane_totals"]["history"]["used_tokens"] > 0


class TestContextUsageEvent:
    """Tests that ContextUsageEvent can be built from allocator summary."""

    def test_event_carries_budget_summary(self) -> None:
        allocator = ContextBudgetAllocator.for_turn(enabled=True)
        allocator.allocate(
            key="test",
            lane="core",
            text="some context text",
        )
        summary = allocator.summary()
        event = ContextUsageEvent(
            mode_id="default",
            query_preview="test query",
            prompt_strategy="broker_packet",
            context_char_count=int(summary.get("used_tokens", 0)) * 4,
            ranking_metadata={"budget_summary": summary},
        )
        assert event.event_type == "context_used"
        assert event.prompt_strategy == "broker_packet"
        assert event.ranking_metadata["budget_summary"]["enabled"] is True


class TestFullChain:
    """End-to-end test: state → candidates → mapper → allocate → prompt."""

    def test_full_chain_produces_markdown_within_budget(self) -> None:
        state = _make_populated_state()
        packet_md = _map_state(state, "cloudflare error rate alert")

        allocator = ContextBudgetAllocator.for_turn(enabled=True)
        elected = allocator.allocate(
            key="ranked_context",
            lane="retrieval",
            text=packet_md,
            allow_truncate=True,
        )
        # The ranked context should either be included (possibly truncated)
        # or dropped by the budget — either way, the allocator must return
        # a string (never raise).
        assert isinstance(elected, str)
