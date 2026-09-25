"""Focused deterministic coverage for the session-owned context mapper."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from kernel.context.context_mapper import (
    ContextBudgetError,
    ContextMapper,
    ContextMappingError,
    context_candidates_from_items,
    context_candidates_from_unified_state,
)
from kernel.context.models import ContextCandidate, ContextMapperRequest
from kernel.models import ContextItem, UnifiedState
from kernel.session_plan import (
    ContextBudgetPlan,
    ContextLanePlan,
    ContextPlan,
    ContextRankingPlan,
    ContextSourceRule,
)

_NOW = datetime(2026, 7, 14, 12, tzinfo=UTC)
_AVAILABLE_CONTEXT_TOKENS = 20
_SELECTED_CONTEXT_TOKENS = 12
_EVIDENCE_SALIENCE = 0.7


def _plan(
    *,
    lanes: tuple[ContextLanePlan, ...] | None = None,
    ranking: ContextRankingPlan | None = None,
    sources: tuple[ContextSourceRule, ...] | None = None,
) -> ContextPlan:
    active_lanes = lanes or (
        ContextLanePlan(
            lane_id="operational",
            max_tokens=80,
            max_items=3,
            render_order=0,
        ),
        ContextLanePlan(
            lane_id="general",
            max_tokens=80,
            max_items=3,
            render_order=10,
        ),
    )
    default_sources = [
        ContextSourceRule(
            rule_id="fallback",
            lane_id="general",
            fallback=True,
        ),
    ]
    if any(lane.lane_id == "operational" for lane in active_lanes):
        default_sources.insert(
            0,
            ContextSourceRule(
                rule_id="alerts",
                lane_id="operational",
                evidence_types=("alert",),
                priority=100,
            ),
        )
    return ContextPlan(
        lanes=active_lanes,
        sources=sources or tuple(default_sources),
        ranking=ranking or ContextRankingPlan(),
        budget=ContextBudgetPlan(
            output_reserve_tokens=20,
            safety_margin_tokens=10,
            max_materialized_candidates=8,
        ),
    )


def _candidate(candidate_id: str, content: str, **overrides: object) -> ContextCandidate:
    values: dict[str, object] = {
        "candidate_id": candidate_id,
        "source_id": "test-source",
        "source_ref": f"test:{candidate_id}",
        "source_kind": "state",
        "content": content,
        "collected_at": _NOW,
        "expires_at": _NOW + timedelta(hours=1),
        "salience": 0.8,
        "authority": 0.9,
        "confidence": 0.9,
    }
    values.update(overrides)
    return ContextCandidate.model_validate(values)


def _request(
    candidates: tuple[ContextCandidate, ...],
    *,
    context_plan: ContextPlan | None = None,
    query: str = "database deployment",
    model_context_window_tokens: int = 200,
) -> ContextMapperRequest:
    return ContextMapperRequest(
        context_plan=context_plan or _plan(),
        query=query,
        mode_id="default",
        candidates=candidates,
        active_packs=frozenset({"lisa-pack-core"}),
        model_context_window_tokens=model_context_window_tokens,
        stable_prefix_tokens=40,
        conversation_tokens=30,
        tool_schema_tokens=20,
        now=_NOW,
    )


def test_mapper_gates_inactive_packs_maps_lanes_and_renders_provenance() -> None:
    alert = _candidate(
        "alert",
        "Database deployment is failing.",
        evidence_type="alert",
        provenance="authoritative",
    )
    inactive = _candidate(
        "cloudflare",
        "Cloudflare deployment history.",
        pack_id="lisa-pack-cloudflare",
        provenance="recalled_memory",
    )

    result = ContextMapper().map(_request((alert, inactive)))

    assert [selection.candidate.candidate_id for selection in result.selections] == ["alert"]
    assert result.selections[0].lane_id == "operational"
    decision = {entry.candidate_id: entry for entry in result.trace.decisions}["cloudflare"]
    assert decision.outcome == "gated"
    assert decision.drop_reason == "inactive_pack"
    assert '<authoritative source="test:alert">' in result.rendered_context


def test_raw_item_adapter_ignores_broker_scores_and_preserves_identity() -> None:
    item = ContextItem(
        item_id="collector:cloudflare",
        source="cloudflare",
        collector_id="cloudflare",
        source_type="extension",
        evidence_type="connector_status",
        title="Cloudflare health",
        body="Worker error rate is elevated.",
        collected_at="2026-07-14T11:00:00Z",
        freshness_expires_at="2026-07-14T13:00:00Z",
        confidence=0.8,
        salience_score=_EVIDENCE_SALIENCE,
        relevance_score=1.0,
        freshness_score=0.01,
        recency_score=0.01,
        combined_score=1.0,
        pack="lisa-pack-cloudflare",
        tags=["cloudflare", "infra"],
        metadata={"provenance": "system_observed"},
    )

    candidate = context_candidates_from_items((item,))[0]

    assert candidate.candidate_id == "collector:cloudflare"
    assert candidate.source_kind == "extension_item"
    assert candidate.source_ref == "context_item:collector:cloudflare"
    assert candidate.pack_id == "lisa-pack-cloudflare"
    assert candidate.provenance == "system_observed"
    assert candidate.salience == _EVIDENCE_SALIENCE
    assert candidate.collected_at == datetime(2026, 7, 14, 11, tzinfo=UTC)
    assert candidate.expires_at == datetime(2026, 7, 14, 13, tzinfo=UTC)

    request = _request(
        (candidate,),
        query="cloudflare worker error",
    ).model_copy(update={"active_packs": frozenset({"lisa-pack-core", "lisa-pack-cloudflare"})})
    result = ContextMapper().map(request)

    assert result.selections[0].base_score < 1.0
    assert "### Cloudflare health" in result.rendered_context


def test_raw_item_adapter_rejects_duplicate_ids_and_invalid_metadata() -> None:
    base = ContextItem(
        item_id="duplicate",
        source="test",
        title="Test",
        body="Test body",
    )
    invalid_provenance = base.model_copy(
        update={
            "item_id": "invalid",
            "metadata": {"provenance": "not-a-provenance"},
        }
    )

    with pytest.raises(ContextMappingError, match="Duplicate context item ID 'duplicate'"):
        context_candidates_from_items((base, base))
    with pytest.raises(ContextMappingError, match=r"metadata\.provenance"):
        context_candidates_from_items((invalid_provenance,))


def test_unified_state_adapter_normalizes_context_items() -> None:
    state = UnifiedState(
        compiled_at=_NOW.isoformat(),
        context_items=[
            ContextItem(
                item_id="state:alert:cloudflare:critical",
                namespace="alerts",
                source="cloudflare",
                source_type="intake",
                evidence_type="alert",
                title="Critical alert from cloudflare",
                body="Worker error rate is elevated.",
                collected_at=_NOW.isoformat(),
                salience_score=1.0,
                confidence=0.95,
                tags=["alert", "critical", "cloudflare"],
                metadata={"provenance": "system_observed"},
            ),
            ContextItem(
                item_id="state:self_model:default",
                namespace="self_model",
                source="self_model",
                source_type="extension",
                evidence_type="runtime_self_model",
                title="Self-model (health=degraded)",
                body="Runtime health is degraded.",
                collected_at=_NOW.isoformat(),
                salience_score=0.8,
                confidence=0.9,
                pack="lisa-pack-introspect",
                tags=["self_model", "degraded"],
                metadata={"provenance": "system_observed"},
            ),
        ],
    )

    candidates = context_candidates_from_unified_state(state)
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    alert = by_id["state:alert:cloudflare:critical"]
    self_model = by_id["state:self_model:default"]

    assert alert.provenance == "system_observed"
    assert alert.salience == 1.0
    assert self_model.provenance == "system_observed"

    result = ContextMapper().map(_request(candidates, query="cloudflare deployment alert"))

    selected_by_type = {
        selection.candidate.evidence_type: selection.lane_id for selection in result.selections
    }
    assert selected_by_type["alert"] == "operational"


def test_mapper_rejects_required_candidate_that_cannot_fit_budget() -> None:
    plan = _plan(lanes=(ContextLanePlan(lane_id="general", max_tokens=2, render_order=0),))
    required = _candidate(
        "required",
        "This required context is longer than the lane budget.",
        required=True,
        token_estimate=10,
    )

    with pytest.raises(ContextBudgetError, match="Required context candidate 'required'"):
        ContextMapper().map(_request((required,), context_plan=plan))


def test_mapper_does_not_allow_fallback_to_bypass_explicit_or_freshness_gates() -> None:
    plan = _plan(
        sources=(
            ContextSourceRule(
                rule_id="explicit-memory",
                lane_id="general",
                sources=("memory",),
                explicit_invocation=True,
                min_freshness=0.8,
                priority=10,
            ),
            ContextSourceRule(
                rule_id="fallback",
                lane_id="general",
                fallback=True,
            ),
        )
    )
    uninvoked = _candidate("uninvoked", "Stored memory.", source_id="memory")
    stale = _candidate(
        "stale",
        "Old memory.",
        source_id="memory",
        explicitly_invoked=True,
        collected_at=_NOW - timedelta(hours=2),
        expires_at=_NOW - timedelta(hours=1),
    )
    fallback = _candidate("fallback", "Ambient context.", source_id="ambient")

    result = ContextMapper().map(_request((uninvoked, stale, fallback), context_plan=plan))

    decisions = {decision.candidate_id: decision for decision in result.trace.decisions}
    assert decisions["uninvoked"].drop_reason == "source_rule_requires_explicit_invocation"
    assert decisions["stale"].drop_reason == "source_rule_stale"
    assert [selection.candidate.candidate_id for selection in result.selections] == ["fallback"]


def test_mapper_reports_unknown_sources_without_a_fallback_rule() -> None:
    plan = _plan(
        sources=(
            ContextSourceRule(
                rule_id="alerts-only",
                lane_id="operational",
                evidence_types=("alert",),
            ),
        )
    )
    unknown = _candidate("unknown", "Unmapped context.", evidence_type="memory")

    result = ContextMapper().map(_request((unknown,), context_plan=plan))

    decision = result.trace.decisions[0]
    assert not result.selections
    assert decision.outcome == "gated"
    assert decision.drop_reason == "no_matching_source_rule"


def test_mapper_uses_highest_priority_matching_source_rule() -> None:
    plan = _plan(
        sources=(
            ContextSourceRule(
                rule_id="general-status",
                lane_id="general",
                evidence_types=("status",),
                priority=10,
            ),
            ContextSourceRule(
                rule_id="operational-status",
                lane_id="operational",
                evidence_types=("status",),
                priority=20,
            ),
            ContextSourceRule(
                rule_id="fallback",
                lane_id="general",
                fallback=True,
            ),
        )
    )
    status = _candidate("status", "Deployment status is degraded.", evidence_type="status")

    result = ContextMapper().map(_request((status,), context_plan=plan))

    assert result.selections[0].lane_id == "operational"
    assert result.trace.decisions[0].matched_rule_ids == ("operational-status",)


def test_mapper_rejects_equal_priority_conflicting_lane_assignments() -> None:
    plan = _plan(
        sources=(
            ContextSourceRule(
                rule_id="general-status",
                lane_id="general",
                evidence_types=("status",),
                priority=10,
            ),
            ContextSourceRule(
                rule_id="operational-status",
                lane_id="operational",
                evidence_types=("status",),
                priority=10,
            ),
        )
    )
    status = _candidate("status", "Deployment status is degraded.", evidence_type="status")

    with pytest.raises(ContextMappingError, match="matches conflicting context lanes"):
        ContextMapper().map(_request((status,), context_plan=plan))


def test_mapper_satisfies_minimum_lane_coverage_before_mmr_selection() -> None:
    plan = ContextPlan(
        lanes=(
            ContextLanePlan(
                lane_id="operational",
                min_items=1,
                max_items=1,
                max_tokens=80,
                render_order=0,
            ),
            ContextLanePlan(
                lane_id="general",
                max_items=1,
                max_tokens=80,
                render_order=10,
            ),
        ),
        sources=(
            ContextSourceRule(
                rule_id="operational",
                lane_id="operational",
                source_kinds=("operational",),
            ),
            ContextSourceRule(
                rule_id="fallback",
                lane_id="general",
                fallback=True,
            ),
        ),
        budget=ContextBudgetPlan(output_reserve_tokens=20, safety_margin_tokens=10),
    )
    operational = _candidate(
        "operational",
        "A low-priority operation is available.",
        source_kind="operational",
        salience=0.0,
    )
    general = _candidate(
        "general",
        "Database deployment migration is urgent.",
        salience=1.0,
    )

    result = ContextMapper().map(_request((general, operational), context_plan=plan))

    assert {selection.lane_id for selection in result.selections} == {
        "operational",
        "general",
    }


def test_mapper_deduplicates_and_uses_mmr_to_diversify_selection() -> None:
    plan = _plan(
        lanes=(ContextLanePlan(lane_id="general", max_tokens=80, max_items=2, render_order=0),),
        ranking=ContextRankingPlan(diversity_lambda=0.2),
    )
    first = _candidate(
        "a-first",
        "Database deployment incident affects the api service.",
        salience=1.0,
        token_estimate=10,
    )
    duplicate = _candidate(
        "z-duplicate",
        "Database deployment incident affects the api service.",
        salience=0.8,
        token_estimate=10,
    )
    similar = _candidate(
        "similar",
        "Database deployment incident blocks service rollout.",
        salience=0.95,
        token_estimate=10,
    )
    diverse = _candidate(
        "diverse",
        "Database migration safety checklist is available.",
        salience=0.8,
        token_estimate=10,
    )

    result = ContextMapper().map(
        _request(
            (first, duplicate, similar, diverse),
            context_plan=plan,
            query="database deployment migration",
        )
    )

    selected_ids = [selection.candidate.candidate_id for selection in result.selections]
    assert "a-first" in selected_ids
    assert "diverse" in selected_ids
    assert "similar" not in selected_ids
    duplicate_decision = {entry.candidate_id: entry for entry in result.trace.decisions}[
        "z-duplicate"
    ]
    assert duplicate_decision.drop_reason == "duplicate_content"


def test_mapper_breaks_ties_by_candidate_id_and_keeps_required_duplicate() -> None:
    plan = _plan(
        lanes=(ContextLanePlan(lane_id="general", max_tokens=80, max_items=3, render_order=0),)
    )
    alpha = _candidate("alpha", "Alpha context.", token_estimate=10)
    beta = _candidate("beta", "Beta context.", token_estimate=10)
    generic = _candidate("generic", "Required duplicate.", salience=1.0)
    required = _candidate(
        "required",
        "Required duplicate.",
        required=True,
        salience=0.1,
    )

    result = ContextMapper().map(_request((beta, generic, alpha, required), context_plan=plan))

    selected_ids = [selection.candidate.candidate_id for selection in result.selections]
    assert selected_ids[:2] == ["required", "alpha"]
    decisions = {decision.candidate_id: decision for decision in result.trace.decisions}
    assert decisions["generic"].duplicate_of == "required"


def test_mapper_reserves_model_budget_before_context_selection() -> None:
    candidate = _candidate("general", "Useful deployment context.", token_estimate=12)

    result = ContextMapper().map(_request((candidate,), model_context_window_tokens=140))

    budget = result.trace.budget
    assert budget.available_context_tokens == _AVAILABLE_CONTEXT_TOKENS
    assert budget.used_context_tokens == _SELECTED_CONTEXT_TOKENS


def test_mapper_reads_provenance_authority_and_conflict_penalty_from_settings() -> None:
    from kernel.config import settings

    candidate = _candidate(
        "provenance",
        "Deployment context with conflict risk.",
        provenance="speculative",
        conflict_risk="high",
        conflict_key="deploy-conflict",
    )

    original_provenance = dict(settings.context_mapper_provenance_authority)
    original_conflict = dict(settings.context_mapper_conflict_penalty)
    try:
        settings.context_mapper_provenance_authority = {
            **original_provenance,
            "speculative": 0.1,
        }
        settings.context_mapper_conflict_penalty = {
            **original_conflict,
            "high": 0.9,
        }

        result = ContextMapper().map(_request((candidate,)))

        decision = next(
            decision for decision in result.trace.decisions if decision.candidate_id == "provenance"
        )
        assert decision.component_scores["authority"] == pytest.approx(
            candidate.authority * candidate.confidence * 0.1
        )
        assert decision.component_scores["conflict"] == pytest.approx(0.9)
    finally:
        settings.context_mapper_provenance_authority = original_provenance
        settings.context_mapper_conflict_penalty = original_conflict
