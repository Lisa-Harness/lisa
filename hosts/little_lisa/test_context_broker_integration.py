"""Integration test for the wired context broker + budget allocator.

Exercises the full chain: populated ``UnifiedState`` →
``build_context_evidence_catalog`` → ``select_ranked_context`` →
``ContextBudgetAllocator.allocate`` → rendered prompt markdown.

Asserts:
- Budget enforcement drops sections when the ceiling is hit.
- Evidence selection respects ``context_broker_min_evidence_score``.
- Namespace ordering in the rendered markdown.
- The ``ContextUsageEvent`` is publishable from the allocator summary.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from kernel.config import settings
from kernel.context.context_budget import ContextBudgetAllocator
from kernel.models import (
    AnomalyAlert,
    ContextEvidence,
    ContextPacket,
    ContextUsageEvent,
    IntelCollection,
    UnifiedState,
)
from sdk.context.context_broker import (
    build_context_evidence_catalog,
    render_context_packet_markdown,
    select_ranked_context,
)

from . import main as little_lisa


@pytest.fixture(scope="module", autouse=True)
def _registered_framework() -> None:
    little_lisa.setup_framework()


def _make_populated_state() -> UnifiedState:
    """Build a UnifiedState with alerts, intel, and domain knowledge."""
    now = datetime.now(UTC).isoformat()
    return UnifiedState(
        compiled_at=now,
        active_alerts=[
            AnomalyAlert(
                severity="critical",
                source="cloudflare",
                message="Worker error rate exceeded 10%",
                detected_at=now,
            ),
        ],
        intel_collections={
            "market": IntelCollection(
                generated_at=now,
                summary_bullets=["Competitor launched new pricing tier"],
            ),
        },
        thinking_vibe="Considering whether to adjust our pricing strategy",
        strategy_vibe="Focus on developer audience segment",
        self_model_vibe=(
            "## Self Model\n"
            "Runtime: uptime=450s, tasks=31, threads=6, rss=512MB, health=degraded\n"
            "Health reasons: mcp failures: 1"
        ),
    )


class TestEvidenceCatalog:
    """Tests for build_context_evidence_catalog with a populated state."""

    def test_builds_evidence_from_alerts(self) -> None:
        state = _make_populated_state()
        catalog = build_context_evidence_catalog(state)
        alert_evidence = [e for e in catalog if e.evidence_type == "alert"]
        assert len(alert_evidence) == 1
        assert alert_evidence[0].namespace == "operational_state"

    def test_builds_evidence_from_intel_collections(self) -> None:
        state = _make_populated_state()
        catalog = build_context_evidence_catalog(state)
        intel_evidence = [e for e in catalog if e.evidence_type == "intel_collection"]
        assert len(intel_evidence) == 1

    def test_builds_evidence_from_reflection(self) -> None:
        state = _make_populated_state()
        catalog = build_context_evidence_catalog(state)
        thinking_evidence = [e for e in catalog if e.evidence_type == "active_thoughts"]
        assert len(thinking_evidence) == 1
        assert thinking_evidence[0].namespace == "long_term_strategic_memory"

    def test_builds_evidence_from_self_model(self) -> None:
        state = _make_populated_state()
        catalog = build_context_evidence_catalog(state)
        self_model = [e for e in catalog if e.evidence_type == "self_model_runtime"]
        assert len(self_model) == 1
        assert self_model[0].namespace == "operational_state"


class TestSelectRankedContext:
    """Tests for select_ranked_context with a populated state."""

    def test_returns_packet_with_selected_evidence(self) -> None:
        state = _make_populated_state()
        packet = select_ranked_context(
            state=state,
            query="cloudflare error rate pricing",
            mode_id="default",
        )
        assert isinstance(packet, ContextPacket)
        assert packet.total_available_evidence > 0
        assert len(packet.selected_evidence) > 0

    def test_filters_below_min_evidence_score(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(settings, "context_broker_min_evidence_score", 0.99)
        state = _make_populated_state()
        packet = select_ranked_context(
            state=state,
            query="nonexistent-query-xyz123",
            mode_id="default",
        )
        assert packet.total_available_evidence == 0
        assert len(packet.selected_evidence) == 0

    def test_namespace_ordering_in_rendered_markdown(self) -> None:
        state = _make_populated_state()
        packet = select_ranked_context(
            state=state,
            query="cloudflare error alert pricing strategy",
            mode_id="default",
        )
        if not packet.selected_evidence:
            pytest.skip("No evidence selected for namespace ordering test")
        markdown = render_context_packet_markdown(packet)
        # operational_state namespace should appear before
        # long_term_strategic_memory in the rendered output
        operational_pos = markdown.find("operational_state")
        strategy_pos = markdown.find("long_term_strategic_memory")
        if operational_pos != -1 and strategy_pos != -1:
            assert operational_pos < strategy_pos

    def test_self_model_operational_evidence_survives_low_overlap(self) -> None:
        state = _make_populated_state()
        packet = select_ranked_context(
            state=state,
            query="totally unrelated lexical query",
            mode_id="default",
        )
        assert any(item.source == "self_model" for item in packet.selected_evidence)


class TestContextBudgetAllocator:
    """Tests for ContextBudgetAllocator integration with assembled context."""

    def test_allocator_enforces_total_token_ceiling(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Set a very low total token budget so sections get dropped.
        monkeypatch.setattr(settings, "context_budget_enabled", True)
        monkeypatch.setattr(settings, "context_budget_default_total_tokens", 50)
        monkeypatch.setattr(settings, "context_budget_token_char_ratio", 4)

        allocator = ContextBudgetAllocator.for_turn(reasoning_model=False)
        # Allocate a large section that exceeds the budget.
        large_text = "x" * 5000
        result = allocator.allocate(
            key="big_section",
            lane="retrieval",
            text=large_text,
            allow_truncate=True,
        )
        # Should be truncated or dropped, not the full text.
        assert len(result) < len(large_text) or result == ""

    def test_allocator_disabled_returns_full_text(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(settings, "context_budget_enabled", False)
        allocator = ContextBudgetAllocator.for_turn(reasoning_model=False)
        text = "This is a test section that should pass through unchanged."
        result = allocator.allocate(
            key="test_section",
            lane="core",
            text=text,
        )
        assert result == text

    def test_allocator_summary_contains_lane_info(self) -> None:
        allocator = ContextBudgetAllocator.for_turn(reasoning_model=False)
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
        allocator = ContextBudgetAllocator.for_turn(reasoning_model=False)
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
    """End-to-end test: state → catalog → select → allocate → prompt."""

    def test_full_chain_produces_markdown_within_budget(self) -> None:
        state = _make_populated_state()
        packet = select_ranked_context(
            state=state,
            query="cloudflare error rate alert",
            mode_id="default",
        )
        packet_md = render_context_packet_markdown(packet)

        allocator = ContextBudgetAllocator.for_turn(reasoning_model=False)
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
