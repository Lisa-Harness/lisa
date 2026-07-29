"""Deterministic, session-owned context candidate mapping and selection."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Final, cast

from kernel.config import settings
from kernel.context.models import (
    ContextBudgetSnapshot,
    ContextCandidate,
    ContextDecision,
    ContextDecisionOutcome,
    ContextDecisionTrace,
    ContextMapperRequest,
    ContextMapperResult,
    ContextProvenance,
    ContextSelection,
    ContextSensitivity,
)
from kernel.context.provenance import wrap_section
from kernel.models import ContextEvidence, UnifiedState
from kernel.session_plan import ContextLanePlan, ContextSourceRule

__all__ = [
    "ContextBudgetError",
    "ContextMapper",
    "ContextMappingError",
    "context_candidates_from_evidence",
    "context_candidates_from_unified_state",
]

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[a-z0-9']+")
_STOP_WORDS: Final[frozenset[str]] = frozenset(
    {
        "a",
        "about",
        "an",
        "and",
        "are",
        "at",
        "be",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
)


class ContextMappingError(ValueError):
    """Raised when session source rules cannot map a candidate unambiguously."""


class ContextBudgetError(ContextMappingError):
    """Raised when required context cannot fit the model-aware budget."""


def context_candidates_from_evidence(
    evidence_items: Iterable[ContextEvidence],
    *,
    default_collected_at: datetime | None = None,
) -> tuple[ContextCandidate, ...]:
    """Normalize raw evidence into mapper candidates without broker scores.

    ``ContextEvidence`` remains a compatibility input while packs and state
    providers migrate to direct candidates. Its legacy relevance, freshness,
    recency, and combined-score fields are intentionally ignored: all ranking
    features are calculated by :class:`ContextMapper` for the current turn.
    """

    candidates: list[ContextCandidate] = []
    seen_candidate_ids: set[str] = set()
    for evidence in sorted(evidence_items, key=lambda item: item.evidence_id):
        candidate_id = evidence.evidence_id.strip()
        if not candidate_id:
            raise ContextMappingError("Context evidence requires a non-empty evidence_id.")
        if candidate_id in seen_candidate_ids:
            raise ContextMappingError(
                f"Duplicate context evidence ID '{candidate_id}' cannot be mapped."
            )
        seen_candidate_ids.add(candidate_id)

        content = evidence.body.strip() or evidence.title.strip()
        if not content:
            raise ContextMappingError(
                f"Context evidence '{candidate_id}' requires a title or body."
            )
        metadata = evidence.metadata
        provenance = _metadata_choice(
            metadata,
            key="provenance",
            allowed_values=settings.context_mapper_provenance_authority,
            default="system_observed" if evidence.collector_id else "authoritative",
            evidence_id=candidate_id,
        )
        sensitivity = _metadata_choice(
            metadata,
            key="sensitivity",
            allowed_values={"public", "private", "restricted"},
            default="private",
            evidence_id=candidate_id,
        )
        candidates.append(
            ContextCandidate(
                candidate_id=candidate_id,
                source_id=(
                    evidence.source.strip() or evidence.collector_id.strip() or candidate_id
                ),
                source_ref=f"context_evidence:{candidate_id}",
                source_kind=(
                    "extension_evidence"
                    if evidence.pack or evidence.source_type == "extension"
                    else "state_evidence"
                ),
                evidence_type=evidence.evidence_type,
                title=evidence.title,
                content=content,
                pack_id=evidence.pack,
                provenance=cast(ContextProvenance, provenance),
                sensitivity=cast(ContextSensitivity, sensitivity),
                authority=1.0,
                confidence=_clamp_score(evidence.confidence),
                salience=_clamp_score(evidence.salience_score),
                collected_at=_parse_timestamp(evidence.collected_at, default_collected_at),
                expires_at=_parse_timestamp(evidence.freshness_expires_at),
                tags=tuple(sorted({tag for tag in evidence.tags if tag})),
                conflict_key=_metadata_text(metadata, "conflict_key"),
                conflict_risk=evidence.conflict_risk,
                explicitly_invoked=_metadata_bool(metadata, "explicitly_invoked"),
            )
        )
    return tuple(candidates)


def context_candidates_from_unified_state(
    state: UnifiedState,
) -> tuple[ContextCandidate, ...]:
    """Normalize the currently-supported ``UnifiedState`` fields to candidates.

    This is intentionally a data adapter, not a ranking path. It preserves
    raw evidence and turns host-observed state into immutable descriptors for
    the session-owned mapper. Additional live sources (history, capabilities,
    operations, and memory providers) can join the same candidate input later.
    """

    compiled_at = _parse_timestamp(state.compiled_at)
    if compiled_at is None:
        raise ContextMappingError("UnifiedState requires a compiled_at timestamp.")

    candidates = list(
        context_candidates_from_evidence(
            state.context_evidence_catalog,
            default_collected_at=compiled_at,
        )
    )
    candidates.extend(_candidates_from_alerts(state, compiled_at))
    candidates.extend(_candidates_from_domain_knowledge(state, compiled_at))
    candidates.extend(_candidates_from_state_summaries(state, compiled_at))
    _require_unique_candidate_ids(candidates)
    return tuple(sorted(candidates, key=lambda candidate: candidate.candidate_id))


def _candidates_from_alerts(
    state: UnifiedState,
    compiled_at: datetime,
) -> list[ContextCandidate]:
    candidates: list[ContextCandidate] = []
    severity_salience = {"critical": 1.0, "warn": 0.85, "info": 0.65}
    for alert in state.active_alerts:
        source_id = alert.source.strip() or "system"
        message = alert.message.strip()
        if not message:
            continue
        observed_at = _parse_timestamp(alert.detected_at, compiled_at)
        severity = alert.severity.strip().lower() or "info"
        identifier = _stable_identifier(source_id, alert.detected_at, message)
        candidates.append(
            ContextCandidate(
                candidate_id=f"state:alert:{identifier}",
                source_id=source_id,
                source_ref=f"unified_state:active_alerts:{identifier}",
                source_kind="alert",
                evidence_type="alert",
                title=f"{severity.title()} alert from {source_id}",
                content=message,
                provenance="system_observed",
                authority=1.0,
                confidence=0.95,
                salience=severity_salience.get(severity, 0.5),
                collected_at=observed_at,
                tags=("alert", severity, source_id),
            )
        )
    return candidates


def _candidates_from_domain_knowledge(
    state: UnifiedState,
    compiled_at: datetime,
) -> list[ContextCandidate]:
    candidates: list[ContextCandidate] = []
    confidence_by_status = {"ok": 0.9, "warning": 0.65, "error": 0.3, "unconfigured": 0.2}
    for domain_id, artifact in sorted(state.domain_knowledge.items()):
        body = artifact.prompt_context_markdown.strip()
        if not body:
            body = "\n".join(
                f"- {bullet.strip()}" for bullet in artifact.summary_bullets if bullet.strip()
            )
        if not body:
            continue
        artifact_id = artifact.artifact_id.strip() or domain_id
        collected_at = _parse_timestamp(artifact.generated_at, compiled_at)
        candidates.append(
            ContextCandidate(
                candidate_id=f"state:domain_knowledge:{artifact_id}",
                source_id=domain_id,
                source_ref=f"unified_state:domain_knowledge:{domain_id}",
                source_kind="domain_knowledge",
                evidence_type="domain_knowledge",
                title=artifact.display_name.strip() or artifact_id,
                content=body,
                provenance="authoritative",
                authority=1.0,
                confidence=confidence_by_status.get(artifact.status, 0.5),
                salience=0.65 if artifact.status == "error" else 0.45,
                collected_at=collected_at,
                tags=("domain_knowledge", artifact.status, domain_id),
            )
        )
    return candidates


def _candidates_from_state_summaries(
    state: UnifiedState,
    compiled_at: datetime,
) -> list[ContextCandidate]:
    summaries = (
        (
            "self_model_vibe",
            "self_model",
            "runtime_self_model",
            "Runtime self-model",
            "system_observed",
            1.0,
            0.8,
        ),
        (
            "capability_manifest_summary",
            "capabilities",
            "live_capabilities",
            "Live capabilities",
            "authoritative",
            1.0,
            0.8,
        ),
        (
            "architecture_vibe",
            "architecture",
            "architecture_summary",
            "Architecture summary",
            "authoritative",
            0.9,
            0.55,
        ),
        (
            "thinking_vibe",
            "thinking",
            "reflection",
            "Reasoning reflection",
            "inferred",
            0.6,
            0.45,
        ),
        (
            "strategy_vibe",
            "strategy",
            "strategy_reflection",
            "Strategy reflection",
            "inferred",
            0.6,
            0.45,
        ),
        (
            "reflection_vibe",
            "reflection",
            "reflection",
            "Reflection",
            "inferred",
            0.6,
            0.45,
        ),
    )
    candidates: list[ContextCandidate] = []
    for (
        field_name,
        source_id,
        evidence_type,
        title,
        provenance,
        confidence,
        salience,
    ) in summaries:
        body = getattr(state, field_name).strip()
        if not body:
            continue
        candidates.append(
            ContextCandidate(
                candidate_id=f"state:{field_name}",
                source_id=source_id,
                source_ref=f"unified_state:{field_name}",
                source_kind="state_summary",
                evidence_type=evidence_type,
                title=title,
                content=body,
                provenance=cast(ContextProvenance, provenance),
                authority=1.0,
                confidence=confidence,
                salience=salience,
                collected_at=compiled_at,
                tags=(field_name,),
            )
        )
    return candidates


def _require_unique_candidate_ids(candidates: list[ContextCandidate]) -> None:
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    duplicates = sorted(
        candidate_id for candidate_id in set(candidate_ids) if candidate_ids.count(candidate_id) > 1
    )
    if duplicates:
        values = ", ".join(duplicates)
        raise ContextMappingError(f"UnifiedState produced duplicate candidate IDs: {values}")


def _stable_identifier(*parts: str) -> str:
    digest_input = "\x00".join(parts).encode("utf-8")
    return hashlib.sha256(digest_input).hexdigest()[:16]


@dataclass(slots=True)
class _DecisionState:
    candidate: ContextCandidate
    matched_rule_ids: tuple[str, ...] = ()
    lane_id: str = ""
    gate_passed: bool = False
    component_scores: dict[str, float] = field(default_factory=dict)
    base_score: float = 0.0
    mmr_score: float | None = None
    token_estimate: int = 0
    allocated_tokens: int = 0
    selection_order: int | None = None
    outcome: ContextDecisionOutcome = "gated"
    drop_reason: str = ""
    duplicate_of: str = ""

    def as_model(self) -> ContextDecision:
        return ContextDecision(
            candidate_id=self.candidate.candidate_id,
            source_ref=self.candidate.source_ref,
            outcome=self.outcome,
            matched_rule_ids=self.matched_rule_ids,
            lane_id=self.lane_id,
            gate_passed=self.gate_passed,
            component_scores=self.component_scores,
            base_score=self.base_score,
            mmr_score=self.mmr_score,
            token_estimate=self.token_estimate,
            allocated_tokens=self.allocated_tokens,
            selection_order=self.selection_order,
            drop_reason=self.drop_reason,
            duplicate_of=self.duplicate_of,
        )


@dataclass(frozen=True, slots=True)
class _MappedCandidate:
    candidate: ContextCandidate
    lane: ContextLanePlan
    required: bool
    pinned: bool
    token_count: int
    base_score: float
    component_scores: dict[str, float]


@dataclass(slots=True)
class _BudgetState:
    available_tokens: int
    lane_used_tokens: dict[str, int]
    lane_item_counts: dict[str, int]
    used_tokens: int = 0

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.available_tokens - self.used_tokens)


class ContextMapper:
    """Map, gate, rank, diversify, budget, and render context candidates."""

    def map(self, request: ContextMapperRequest) -> ContextMapperResult:
        """Select context deterministically from one static plan and live request."""

        now = _as_utc(request.now) if request.now is not None else datetime.now(UTC)
        budget = _build_budget_state(request)
        decisions = {
            candidate.candidate_id: _DecisionState(candidate=candidate)
            for candidate in sorted(request.candidates, key=lambda item: item.candidate_id)
        }
        mapped = self._map_candidates(request, decisions, now)
        deduped = self._deduplicate(mapped, decisions)
        selections = self._select(deduped, request, budget, decisions)
        rendered = _render_selections(selections, request)
        budget_snapshot = _budget_snapshot(request, budget)
        trace = ContextDecisionTrace(
            decisions=tuple(
                decisions[candidate_id].as_model() for candidate_id in sorted(decisions)
            ),
            budget=budget_snapshot,
            selected_candidate_ids=tuple(
                selection.candidate.candidate_id for selection in selections
            ),
        )
        return ContextMapperResult(
            selections=tuple(selections),
            rendered_context=rendered,
            trace=trace,
        )

    def _map_candidates(
        self,
        request: ContextMapperRequest,
        decisions: dict[str, _DecisionState],
        now: datetime,
    ) -> list[_MappedCandidate]:
        mapped: list[_MappedCandidate] = []
        lane_by_id = {lane.lane_id: lane for lane in request.context_plan.lanes}
        for candidate in sorted(request.candidates, key=lambda item: item.candidate_id):
            state = decisions[candidate.candidate_id]
            gate_reason = _candidate_gate_reason(candidate, request)
            if gate_reason:
                state.drop_reason = gate_reason
                continue

            matching_rules, source_rule_gate_reason = _resolve_source_rules(
                candidate,
                request,
                now,
            )
            if not matching_rules:
                state.drop_reason = source_rule_gate_reason
                continue
            top_priority = max(rule.priority for rule in matching_rules)
            top_rules = [rule for rule in matching_rules if rule.priority == top_priority]
            lane_ids = {rule.lane_id for rule in top_rules}
            if len(lane_ids) > 1:
                values = ", ".join(sorted(lane_ids))
                raise ContextMappingError(
                    f"Candidate '{candidate.candidate_id}' matches conflicting "
                    f"context lanes at priority {top_priority}: {values}"
                )

            selected_rule = min(top_rules, key=lambda rule: rule.rule_id)
            lane = lane_by_id[selected_rule.lane_id]
            token_count = _estimate_tokens(candidate)
            components = _score_components(candidate, request, now, token_count)
            base_score = _base_score(components, request)
            state.matched_rule_ids = tuple(rule.rule_id for rule in top_rules)
            state.lane_id = lane.lane_id
            state.gate_passed = True
            state.outcome = "dropped"
            state.component_scores = components
            state.base_score = base_score
            state.token_estimate = token_count
            mapped.append(
                _MappedCandidate(
                    candidate=candidate,
                    lane=lane,
                    required=candidate.required or selected_rule.required,
                    pinned=candidate.pinned or selected_rule.pinned,
                    token_count=token_count,
                    base_score=base_score,
                    component_scores=components,
                )
            )
        return mapped

    def _deduplicate(
        self,
        mapped: list[_MappedCandidate],
        decisions: dict[str, _DecisionState],
    ) -> list[_MappedCandidate]:
        grouped: dict[str, list[_MappedCandidate]] = {}
        for item in mapped:
            key = item.candidate.content_hash or _content_hash(item.candidate.content)
            grouped.setdefault(key, []).append(item)

        deduped: list[_MappedCandidate] = []
        for group in grouped.values():
            ranked = sorted(
                group,
                key=lambda item: (
                    not item.required,
                    not item.pinned,
                    -item.base_score,
                    item.candidate.candidate_id,
                ),
            )
            winner = ranked[0]
            deduped.append(winner)
            for duplicate in ranked[1:]:
                state = decisions[duplicate.candidate.candidate_id]
                state.drop_reason = "duplicate_content"
                state.duplicate_of = winner.candidate.candidate_id
        return deduped

    def _select(
        self,
        mapped: list[_MappedCandidate],
        request: ContextMapperRequest,
        budget: _BudgetState,
        decisions: dict[str, _DecisionState],
    ) -> list[ContextSelection]:
        selections: list[ContextSelection] = []
        selected_ids: set[str] = set()
        forced = sorted(
            (item for item in mapped if item.required or item.pinned),
            key=lambda item: (
                not item.required,
                not item.pinned,
                item.lane.render_order,
                -item.base_score,
                item.candidate.candidate_id,
            ),
        )
        for item in forced:
            if _fits_budget(item, budget):
                self._commit(
                    item,
                    score=item.base_score,
                    budget=budget,
                    decisions=decisions,
                    selections=selections,
                )
                selected_ids.add(item.candidate.candidate_id)
                continue
            if item.required:
                raise ContextBudgetError(
                    f"Required context candidate '{item.candidate.candidate_id}' "
                    "cannot fit the model-aware context budget."
                )
            decisions[item.candidate.candidate_id].drop_reason = "pinned_budget_exhausted"

        self._satisfy_lane_minimums(
            mapped,
            selected_ids,
            request,
            budget,
            decisions,
            selections,
        )
        self._select_with_mmr(
            mapped,
            selected_ids,
            request,
            budget,
            decisions,
            selections,
        )
        self._mark_budget_drops(mapped, selected_ids, budget, decisions)
        return sorted(
            selections,
            key=lambda selection: (
                _lane_render_order(selection.lane_id, request),
                selection.selection_order,
            ),
        )

    def _satisfy_lane_minimums(
        self,
        mapped: list[_MappedCandidate],
        selected_ids: set[str],
        request: ContextMapperRequest,
        budget: _BudgetState,
        decisions: dict[str, _DecisionState],
        selections: list[ContextSelection],
    ) -> None:
        for lane in sorted(request.context_plan.lanes, key=lambda item: item.render_order):
            while _lane_needs_coverage(lane, budget):
                candidates = [
                    item
                    for item in mapped
                    if item.lane.lane_id == lane.lane_id
                    and item.candidate.candidate_id not in selected_ids
                    and not decisions[item.candidate.candidate_id].drop_reason
                    and _fits_budget(item, budget)
                ]
                if not candidates:
                    break
                item = min(
                    candidates,
                    key=lambda candidate: (
                        -candidate.base_score,
                        candidate.candidate.candidate_id,
                    ),
                )
                self._commit(
                    item,
                    score=item.base_score,
                    budget=budget,
                    decisions=decisions,
                    selections=selections,
                )
                selected_ids.add(item.candidate.candidate_id)

    def _select_with_mmr(
        self,
        mapped: list[_MappedCandidate],
        selected_ids: set[str],
        request: ContextMapperRequest,
        budget: _BudgetState,
        decisions: dict[str, _DecisionState],
        selections: list[ContextSelection],
    ) -> None:
        diversity_lambda = request.context_plan.ranking.diversity_lambda
        while True:
            eligible = [
                item
                for item in mapped
                if item.candidate.candidate_id not in selected_ids
                and not decisions[item.candidate.candidate_id].drop_reason
                and _fits_budget(item, budget)
            ]
            if not eligible:
                return
            scored = [
                (
                    item,
                    _mmr_score(item, selections, diversity_lambda),
                )
                for item in eligible
            ]
            item, score = min(
                scored,
                key=lambda pair: (
                    -pair[1],
                    -pair[0].base_score,
                    pair[0].candidate.candidate_id,
                ),
            )
            self._commit(
                item,
                score=score,
                budget=budget,
                decisions=decisions,
                selections=selections,
            )
            selected_ids.add(item.candidate.candidate_id)

    @staticmethod
    def _commit(
        item: _MappedCandidate,
        *,
        score: float,
        budget: _BudgetState,
        decisions: dict[str, _DecisionState],
        selections: list[ContextSelection],
    ) -> None:
        selection_order = len(selections)
        budget.used_tokens += item.token_count
        budget.lane_used_tokens[item.lane.lane_id] = (
            budget.lane_used_tokens.get(item.lane.lane_id, 0) + item.token_count
        )
        budget.lane_item_counts[item.lane.lane_id] = (
            budget.lane_item_counts.get(item.lane.lane_id, 0) + 1
        )
        state = decisions[item.candidate.candidate_id]
        state.outcome = "selected"
        state.mmr_score = score
        state.allocated_tokens = item.token_count
        state.selection_order = selection_order
        selections.append(
            ContextSelection(
                candidate=item.candidate,
                lane_id=item.lane.lane_id,
                token_count=item.token_count,
                base_score=item.base_score,
                mmr_score=score,
                selection_order=selection_order,
            )
        )

    @staticmethod
    def _mark_budget_drops(
        mapped: list[_MappedCandidate],
        selected_ids: set[str],
        budget: _BudgetState,
        decisions: dict[str, _DecisionState],
    ) -> None:
        for item in mapped:
            candidate_id = item.candidate.candidate_id
            state = decisions[candidate_id]
            if candidate_id in selected_ids or state.drop_reason:
                continue
            state.drop_reason = _budget_drop_reason(item, budget)


def _candidate_gate_reason(
    candidate: ContextCandidate,
    request: ContextMapperRequest,
) -> str:
    if candidate.pack_id and candidate.pack_id not in request.active_packs:
        return "inactive_pack"
    if candidate.sensitivity not in request.allowed_sensitivities:
        return "surface_privacy_policy"
    if candidate.mode_ids and request.mode_id not in candidate.mode_ids:
        return "candidate_mode_mismatch"
    if candidate.stage_ids and request.stage_id not in candidate.stage_ids:
        return "candidate_stage_mismatch"
    return ""


def _resolve_source_rules(
    candidate: ContextCandidate,
    request: ContextMapperRequest,
    now: datetime,
) -> tuple[list[ContextSourceRule], str]:
    normal_rules = [
        rule
        for rule in request.context_plan.sources
        if not rule.fallback and _rule_source_selectors_match(rule, candidate)
    ]
    normal_matches = [
        rule for rule in normal_rules if _rule_conditions_match(rule, candidate, request, now)
    ]
    if normal_matches:
        return normal_matches, ""
    if normal_rules:
        return [], _source_rule_gate_reason(normal_rules, candidate, request, now)

    fallback_matches = [
        rule
        for rule in request.context_plan.sources
        if rule.fallback
        and _rule_source_selectors_match(rule, candidate)
        and _rule_conditions_match(rule, candidate, request, now)
    ]
    if fallback_matches:
        return fallback_matches, ""
    return [], "no_matching_source_rule"


def _rule_source_selectors_match(
    rule: ContextSourceRule,
    candidate: ContextCandidate,
) -> bool:
    matchers = (
        not rule.source_kinds or candidate.source_kind in rule.source_kinds,
        not rule.sources or candidate.source_id in rule.sources,
        not rule.packs or candidate.pack_id in rule.packs,
        not rule.evidence_types or candidate.evidence_type in rule.evidence_types,
        not rule.tags or bool(set(rule.tags).intersection(candidate.tags)),
    )
    return all(matchers)


def _rule_conditions_match(
    rule: ContextSourceRule,
    candidate: ContextCandidate,
    request: ContextMapperRequest,
    now: datetime,
) -> bool:
    conditions = (
        not rule.modes or request.mode_id in rule.modes,
        not rule.stages or request.stage_id in rule.stages,
        not rule.explicit_invocation or candidate.explicitly_invoked,
        _freshness(candidate, now) >= rule.min_freshness,
    )
    return all(conditions)


def _source_rule_gate_reason(
    rules: list[ContextSourceRule],
    candidate: ContextCandidate,
    request: ContextMapperRequest,
    now: datetime,
) -> str:
    ordered_rules = sorted(rules, key=lambda rule: (-rule.priority, rule.rule_id))
    for rule in ordered_rules:
        if rule.modes and request.mode_id not in rule.modes:
            return "source_rule_mode_mismatch"
        if rule.stages and request.stage_id not in rule.stages:
            return "source_rule_stage_mismatch"
        if rule.explicit_invocation and not candidate.explicitly_invoked:
            return "source_rule_requires_explicit_invocation"
        if _freshness(candidate, now) < rule.min_freshness:
            return "source_rule_stale"
    return "source_rule_gate_failed"


def _score_components(
    candidate: ContextCandidate,
    request: ContextMapperRequest,
    now: datetime,
    token_count: int,
) -> dict[str, float]:
    available_tokens = _available_context_tokens(request)
    provenance_authority = settings.context_mapper_provenance_authority
    conflict_penalty = settings.context_mapper_conflict_penalty
    return {
        "relevance": _lexical_relevance(request.query, candidate),
        "mode_affinity": 1.0 if request.mode_id in candidate.mode_ids else 0.6,
        "salience": candidate.salience,
        "freshness": _freshness(candidate, now),
        "authority": candidate.authority
        * candidate.confidence
        * provenance_authority[candidate.provenance],
        "continuity": candidate.continuity,
        "user_priority": candidate.user_priority,
        "recency": _recency(candidate, now),
        "conflict": conflict_penalty[candidate.conflict_risk],
        "token_cost": min(token_count / max(available_tokens, 1), 1.0),
    }


def _base_score(
    components: dict[str, float],
    request: ContextMapperRequest,
) -> float:
    ranking = request.context_plan.ranking
    positive = ranking.normalized_positive_weights
    weighted = sum(positive[name] * components[name] for name in positive)
    penalties = (
        ranking.conflict_penalty * components["conflict"]
        + ranking.token_cost_penalty * components["token_cost"]
    )
    return round(weighted - penalties, 6)


def _estimate_tokens(candidate: ContextCandidate) -> int:
    if candidate.token_estimate:
        return candidate.token_estimate
    return max(1, math.ceil(len(candidate.content) / 4))


def _build_budget_state(request: ContextMapperRequest) -> _BudgetState:
    return _BudgetState(
        available_tokens=_available_context_tokens(request),
        lane_used_tokens={lane.lane_id: 0 for lane in request.context_plan.lanes},
        lane_item_counts={lane.lane_id: 0 for lane in request.context_plan.lanes},
    )


def _available_context_tokens(request: ContextMapperRequest) -> int:
    budget = request.context_plan.budget
    return max(
        0,
        request.model_context_window_tokens
        - budget.output_reserve_tokens
        - budget.safety_margin_tokens
        - request.stable_prefix_tokens
        - request.conversation_tokens
        - request.tool_schema_tokens,
    )


def _budget_snapshot(
    request: ContextMapperRequest,
    budget: _BudgetState,
) -> ContextBudgetSnapshot:
    plan_budget = request.context_plan.budget
    return ContextBudgetSnapshot(
        model_context_window_tokens=request.model_context_window_tokens,
        output_reserve_tokens=plan_budget.output_reserve_tokens,
        safety_margin_tokens=plan_budget.safety_margin_tokens,
        stable_prefix_tokens=request.stable_prefix_tokens,
        conversation_tokens=request.conversation_tokens,
        tool_schema_tokens=request.tool_schema_tokens,
        available_context_tokens=budget.available_tokens,
        used_context_tokens=budget.used_tokens,
        lane_used_tokens=budget.lane_used_tokens,
    )


def _fits_budget(item: _MappedCandidate, budget: _BudgetState) -> bool:
    if item.token_count > budget.remaining_tokens:
        return False
    lane = item.lane
    lane_used = budget.lane_used_tokens.get(lane.lane_id, 0)
    lane_items = budget.lane_item_counts.get(lane.lane_id, 0)
    if lane.max_tokens is not None and lane_used + item.token_count > lane.max_tokens:
        return False
    return lane.max_items is None or lane_items < lane.max_items


def _budget_drop_reason(item: _MappedCandidate, budget: _BudgetState) -> str:
    if item.token_count > budget.remaining_tokens:
        return "global_budget_exhausted"
    lane = item.lane
    lane_used = budget.lane_used_tokens.get(lane.lane_id, 0)
    if lane.max_tokens is not None and lane_used + item.token_count > lane.max_tokens:
        return "lane_token_budget_exhausted"
    if (
        lane.max_items is not None
        and budget.lane_item_counts.get(lane.lane_id, 0) >= lane.max_items
    ):
        return "lane_item_limit_reached"
    return "not_selected"


def _lane_needs_coverage(lane: ContextLanePlan, budget: _BudgetState) -> bool:
    return (
        budget.lane_item_counts.get(lane.lane_id, 0) < lane.min_items
        or budget.lane_used_tokens.get(lane.lane_id, 0) < lane.min_tokens
    )


def _mmr_score(
    item: _MappedCandidate,
    selections: list[ContextSelection],
    diversity_lambda: float,
) -> float:
    if not selections:
        return item.base_score
    similarity = max(
        _candidate_similarity(item.candidate, selection.candidate) for selection in selections
    )
    return round(
        (diversity_lambda * item.base_score) - ((1.0 - diversity_lambda) * similarity),
        6,
    )


def _candidate_similarity(left: ContextCandidate, right: ContextCandidate) -> float:
    if left.conflict_key and left.conflict_key == right.conflict_key:
        return 1.0
    left_terms = _candidate_terms(left)
    right_terms = _candidate_terms(right)
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms | right_terms)


def _lexical_relevance(query: str, candidate: ContextCandidate) -> float:
    query_terms = _terms(query)
    if not query_terms:
        return 0.0
    candidate_terms = _candidate_terms(candidate)
    if not candidate_terms:
        return 0.0
    return len(query_terms & candidate_terms) / len(query_terms)


def _candidate_terms(candidate: ContextCandidate) -> set[str]:
    return _terms(" ".join((candidate.title, candidate.content, " ".join(candidate.tags))))


def _terms(text: str) -> set[str]:
    return {
        match.group(0)
        for match in _TOKEN_RE.finditer(text.lower())
        if match.group(0) not in _STOP_WORDS
    }


def _freshness(candidate: ContextCandidate, now: datetime) -> float:
    if candidate.expires_at is None:
        return 1.0
    expires_at = _as_utc(candidate.expires_at)
    if now <= expires_at:
        return 1.0
    if candidate.collected_at is None:
        return 0.0
    collected_at = _as_utc(candidate.collected_at)
    ttl_seconds = max((expires_at - collected_at).total_seconds(), 1.0)
    stale_seconds = (now - expires_at).total_seconds()
    return max(0.0, min(1.0, math.exp(-stale_seconds / ttl_seconds)))


def _recency(candidate: ContextCandidate, now: datetime) -> float:
    if candidate.collected_at is None:
        return 0.5
    age_hours = max((now - _as_utc(candidate.collected_at)).total_seconds() / 3600.0, 0.0)
    return math.exp(-0.69314718056 * age_hours / 24.0)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _parse_timestamp(value: str, default: datetime | None = None) -> datetime | None:
    normalized = value.strip()
    if not normalized:
        return _as_utc(default) if default is not None else None
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContextMappingError(f"Invalid context evidence timestamp '{value}'.") from exc
    return _as_utc(parsed)


def _metadata_choice(
    metadata: dict[str, object],
    *,
    key: str,
    allowed_values: dict[str, float] | set[str],
    default: str,
    evidence_id: str,
) -> str:
    value = metadata.get(key, default)
    if not isinstance(value, str) or value not in allowed_values:
        allowed = ", ".join(sorted(allowed_values))
        raise ContextMappingError(
            f"Context evidence '{evidence_id}' metadata.{key} must be one of: {allowed}."
        )
    return value


def _metadata_text(metadata: dict[str, object], key: str) -> str:
    value = metadata.get(key, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ContextMappingError(f"Context evidence metadata.{key} must be a string.")
    return value.strip()


def _metadata_bool(metadata: dict[str, object], key: str) -> bool:
    value = metadata.get(key, False)
    if not isinstance(value, bool):
        raise ContextMappingError(f"Context evidence metadata.{key} must be a boolean.")
    return value


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()


def _lane_render_order(lane_id: str, request: ContextMapperRequest) -> int:
    for lane in request.context_plan.lanes:
        if lane.lane_id == lane_id:
            return lane.render_order
    return 0


def _render_selections(
    selections: list[ContextSelection],
    request: ContextMapperRequest,
) -> str:
    if not selections:
        return ""
    parts: list[str] = []
    lane_ids = sorted(
        {selection.lane_id for selection in selections},
        key=lambda lane_id: _lane_render_order(lane_id, request),
    )
    for lane_id in lane_ids:
        lane_parts = [
            wrap_section(
                selection.candidate.provenance,
                _render_candidate_content(selection.candidate),
                source=selection.candidate.source_ref,
            )
            for selection in selections
            if selection.lane_id == lane_id
        ]
        rendered = "\n\n".join(part for part in lane_parts if part)
        if rendered:
            parts.append(f"## {lane_id}\n\n{rendered}")
    return "\n\n".join(parts)


def _render_candidate_content(candidate: ContextCandidate) -> str:
    if not candidate.title or candidate.title == candidate.content:
        return candidate.content
    return f"### {candidate.title}\n\n{candidate.content}"
