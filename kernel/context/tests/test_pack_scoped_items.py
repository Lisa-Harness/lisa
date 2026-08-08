"""Tests for pack-scoped evidence boundary filtering in the context mapper.

These tests verify that extension-contributed evidence (tagged with a
``pack`` field) is only included when the mode's allowed pack set includes
that pack, while kernel-core evidence (alerts, self-model, domain knowledge,
intel, reflection) is always included regardless of the pack boundary.
"""

from __future__ import annotations

from datetime import UTC, datetime

from kernel.context.context_mapper import (
    ContextMapper,
    context_candidates_from_unified_state,
)
from kernel.context.models import ContextMapperRequest
from kernel.models import (
    ContextItem,
    UnifiedState,
)
from kernel.session_plan import ContextLanePlan, ContextPlan, ContextSourceRule


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


def _make_state_with_extension_item() -> UnifiedState:
    """Build a state with both kernel-core and extension evidence."""
    now = datetime.now(UTC).isoformat()
    return UnifiedState(
        compiled_at=now,
        context_items=[
            # Kernel-core evidence: no pack field → always allowed
            ContextItem(
                item_id="self_model:runtime_vibe",
                source="self_model",
                evidence_type="self_model_runtime",
                title="Live self-model runtime state",
                body="Runtime health=ok",
                collected_at=now,
            ),
            # Extension evidence from the cloudflare pack
            ContextItem(
                item_id="collector:cloudflare",
                source="cloudflare",
                collector_id="cloudflare",
                evidence_type="connector_status",
                title="Cloudflare health",
                body="Status OK | requests_1h=1000",
                collected_at=now,
                pack="lisa-pack-cloudflare",
            ),
            # Extension evidence from the github pack
            ContextItem(
                item_id="collector:github",
                source="github",
                collector_id="github",
                evidence_type="connector_status",
                title="GitHub delivery state",
                body="Workflow SUCCESS",
                collected_at=now,
                pack="lisa-pack-github",
            ),
        ],
    )


def _map_sources(state: UnifiedState, query: str, active_packs: frozenset[str]) -> set[str]:
    """Run the mapper and return the set of selected candidate source_ids."""
    candidates = context_candidates_from_unified_state(state)
    request = ContextMapperRequest(
        context_plan=_default_plan(),
        query=query,
        mode_id="default",
        candidates=candidates,
        active_packs=active_packs,
        model_context_window_tokens=131072,
        stable_prefix_tokens=0,
        conversation_tokens=0,
        tool_schema_tokens=0,
    )
    result = ContextMapper().map(request)
    return {selection.candidate.source_id for selection in result.selections}


class TestPackScopedFiltering:
    """Integration tests for pack-scoped filtering in the context mapper."""

    def test_default_mode_excludes_all_extension_item(self) -> None:
        """A mode with no active packs gets no extension evidence."""
        state = _make_state_with_extension_item()
        sources = _map_sources(state, "cloudflare status", frozenset())
        assert "self_model" in sources  # kernel-core evidence always included
        assert "cloudflare" not in sources
        assert "github" not in sources

    def test_cloudflare_mode_includes_own_pack_evidence(self) -> None:
        """The cloudflare mode sees its own pack's evidence plus kernel-core."""
        state = _make_state_with_extension_item()
        sources = _map_sources(state, "cloudflare status", frozenset({"lisa-pack-cloudflare"}))
        assert "self_model" in sources
        assert "cloudflare" in sources
        assert "github" not in sources

    def test_mode_with_explicit_cross_pack_allow(self) -> None:
        """A mode can explicitly allow another pack's evidence."""
        state = _make_state_with_extension_item()
        sources = _map_sources(
            state,
            "cloudflare github status",
            frozenset({"lisa-pack-core", "lisa-pack-cloudflare", "lisa-pack-github"}),
        )
        assert "self_model" in sources
        assert "cloudflare" in sources
        assert "github" in sources

    def test_kernel_core_evidence_always_present(self) -> None:
        """Self-model evidence survives even when no packs are allowed."""
        state = _make_state_with_extension_item()
        sources = _map_sources(state, "unrelated query xyz", frozenset())
        assert "self_model" in sources
