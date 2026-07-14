"""Tests for pack-scoped evidence boundary filtering in the context broker.

These tests verify that extension-contributed evidence (tagged with a
``pack`` field) is only included when the mode's allowed pack set includes
that pack, while kernel-core evidence (alerts, self-model, domain knowledge,
intel, reflection) is always included regardless of the pack boundary.
"""

from __future__ import annotations

from datetime import UTC, datetime

from kernel.context.context_broker import (
    _is_pack_allowed,
    select_ranked_context,
)
from kernel.models import (
    ContextEvidence,
    UnifiedState,
)


def _make_state_with_extension_evidence() -> UnifiedState:
    """Build a state with both kernel-core and extension evidence."""
    now = datetime.now(UTC).isoformat()
    return UnifiedState(
        compiled_at=now,
        self_model_vibe="Runtime health=ok",
        context_evidence_catalog=[
            # Kernel-core evidence: no pack field → always allowed
            ContextEvidence(
                evidence_id="self_model:runtime_vibe",
                source="self_model",
                evidence_type="self_model_runtime",
                title="Live self-model runtime state",
                body="Runtime health=ok",
                collected_at=now,
            ),
            # Extension evidence from the cloudflare pack
            ContextEvidence(
                evidence_id="collector:cloudflare",
                source="cloudflare",
                collector_id="cloudflare",
                evidence_type="connector_status",
                title="Cloudflare health",
                body="Status OK | requests_1h=1000",
                collected_at=now,
                pack="lisa-pack-cloudflare",
            ),
            # Extension evidence from the github pack
            ContextEvidence(
                evidence_id="collector:github",
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


class TestIsPackAllowed:
    """Unit tests for the _is_pack_allowed filter predicate."""

    def test_kernel_core_evidence_always_allowed(self) -> None:
        """Evidence with no pack field is always allowed."""
        evidence = ContextEvidence(
            evidence_id="alert:1:cloudflare",
            source="cloudflare",
            evidence_type="alert",
            title="Alert",
            body="Something happened",
            collected_at="",
            pack="",
        )
        assert _is_pack_allowed(evidence, set()) is True
        assert _is_pack_allowed(evidence, {"lisa-pack-other"}) is True

    def test_extension_evidence_allowed_when_pack_in_set(self) -> None:
        evidence = ContextEvidence(
            evidence_id="collector:cloudflare",
            source="cloudflare",
            evidence_type="connector_status",
            title="Cloudflare health",
            body="OK",
            collected_at="",
            pack="lisa-pack-cloudflare",
        )
        assert _is_pack_allowed(evidence, {"lisa-pack-cloudflare"}) is True

    def test_extension_evidence_filtered_when_pack_not_in_set(self) -> None:
        evidence = ContextEvidence(
            evidence_id="collector:cloudflare",
            source="cloudflare",
            evidence_type="connector_status",
            title="Cloudflare health",
            body="OK",
            collected_at="",
            pack="lisa-pack-cloudflare",
        )
        assert _is_pack_allowed(evidence, set()) is False
        assert _is_pack_allowed(evidence, {"lisa-pack-github"}) is False


class TestSelectRankedContextPackFiltering:
    """Integration tests for select_ranked_context with pack-scoped filtering."""

    def test_default_mode_excludes_all_extension_evidence(self) -> None:
        """A mode with no mode_provider_id gets no extension evidence."""
        state = _make_state_with_extension_evidence()
        packet = select_ranked_context(
            state=state,
            query="cloudflare status",
            mode_id="default",
            mode_provider_id="",  # default mode: no provider
            context_evidence_packs=None,
        )
        sources = {e.source for e in packet.selected_evidence}
        assert "self_model" in sources  # kernel-core evidence always included
        assert "cloudflare" not in sources
        assert "github" not in sources

    def test_cloudflare_mode_includes_own_pack_evidence(self) -> None:
        """The cloudflare mode sees its own pack's evidence plus kernel-core."""
        state = _make_state_with_extension_evidence()
        packet = select_ranked_context(
            state=state,
            query="cloudflare status",
            mode_id="cloudflare",
            mode_provider_id="lisa-pack-cloudflare",
            context_evidence_packs=None,
        )
        sources = {e.source for e in packet.selected_evidence}
        assert "self_model" in sources
        assert "cloudflare" in sources
        assert "github" not in sources

    def test_mode_with_explicit_cross_pack_allow(self) -> None:
        """A mode can explicitly allow another pack's evidence."""
        state = _make_state_with_extension_evidence()
        packet = select_ranked_context(
            state=state,
            query="cloudflare github status",
            mode_id="infra",
            mode_provider_id="lisa-pack-core",
            context_evidence_packs=["lisa-pack-cloudflare", "lisa-pack-github"],
        )
        sources = {e.source for e in packet.selected_evidence}
        assert "self_model" in sources
        assert "cloudflare" in sources
        assert "github" in sources

    def test_kernel_core_evidence_always_present(self) -> None:
        """Self-model evidence survives even when no packs are allowed."""
        state = _make_state_with_extension_evidence()
        packet = select_ranked_context(
            state=state,
            query="unrelated query xyz",
            mode_id="default",
            mode_provider_id="",
        )
        assert any(e.source == "self_model" for e in packet.selected_evidence)

    def test_ranking_metadata_includes_allowed_packs(self) -> None:
        """The packet's ranking_metadata records the allowed pack set."""
        state = _make_state_with_extension_evidence()
        packet = select_ranked_context(
            state=state,
            query="cloudflare",
            mode_id="cloudflare",
            mode_provider_id="lisa-pack-cloudflare",
            context_evidence_packs=["lisa-pack-github"],
        )
        allowed = packet.ranking_metadata.get("allowed_evidence_packs", [])
        assert "lisa-pack-cloudflare" in allowed
        assert "lisa-pack-github" in allowed