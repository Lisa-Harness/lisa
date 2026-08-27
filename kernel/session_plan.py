"""Strict session-manifest contracts and deterministic plan compilation.

The session manifest is the declarative control plane for one Lisa runtime.
This module deliberately contains no host, MCP, pack-discovery, or filesystem
logic: hosts resolve a source and catalog first, then pass its text here for
strict validation and stable plan construction.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from kernel.ipsar_fsm import parse_front_matter

__all__ = [
    "ContextBudgetPlan",
    "ContextHistoryPlan",
    "ContextLanePlan",
    "ContextPlan",
    "ContextRankingPlan",
    "ContextSourceRule",
    "InstructionFragment",
    "ModelManifestRef",
    "PackInstallSpec",
    "PackSelection",
    "ProviderSelections",
    "ResourceSelection",
    "SessionCapabilityExecutionRequest",
    "SessionExecutionRequest",
    "SessionInstructionPlan",
    "SessionInstructions",
    "SessionManifestV2",
    "SessionPlan",
    "SessionPlanError",
    "SessionResources",
    "SessionRouting",
    "SessionSurface",
    "compile_session_plan",
]


class _StrictSessionModel(BaseModel):
    """Base model for public session-manifest contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class SessionPlanError(ValueError):
    """Raised when a session manifest cannot be parsed into a plan."""


class SessionSurface(_StrictSessionModel):
    """The privacy and interaction surface of a session."""

    type: str = Field(min_length=1)
    id: str = Field(min_length=1)
    memory_scope: str = Field(min_length=1)


class ResourceSelection(_StrictSessionModel):
    """A deterministic include/exclude/required selector for one resource kind."""

    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    required: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_sets(self) -> ResourceSelection:
        include = set(self.include)
        exclude = set(self.exclude)
        required = set(self.required)
        overlap = include & exclude
        if overlap:
            overlap_ids = ", ".join(sorted(overlap))
            raise ValueError(f"include and exclude overlap: {overlap_ids}")
        overlap = required & exclude
        if overlap:
            overlap_ids = ", ".join(sorted(overlap))
            raise ValueError(f"required and exclude overlap: {overlap_ids}")
        for field_name, resource_ids in (
            ("include", self.include),
            ("exclude", self.exclude),
            ("required", self.required),
        ):
            if len(resource_ids) != len(set(resource_ids)):
                raise ValueError(f"{field_name} contains duplicate resource IDs")
        return self


class PackInstallSpec(_StrictSessionModel):
    """A declared install source for one required pack.

    The compiler validates shape only — it never reads the filesystem or
    network. The host boot path resolves ``source`` against the trust policy
    (``LISA_PACK_ALLOWED_PREFIXES``, ``LISA_PACK_GIT_ALLOW_HOSTS``) and
    performs the actual install. Git refs must pin ``@tag/@branch/@sha``;
    the host treats pinning as mandatory for auto-install regardless of
    ``LISA_PACK_REQUIRE_GIT_REF``.

    When ``source`` is a local directory and ``editable`` is ``True`` (the
    default), the host installs it with ``uv pip install -e`` so edits to the
    pack source are picked up without reinstall. Set ``editable: false`` to
    force a non-editable install (rarely needed — use it only when the pack
    source is a throwaway snapshot).
    """

    source: str = Field(min_length=1)
    pin: str = ""
    editable: bool = True


class PackSelection(ResourceSelection):
    """Pack selection plus pack-owned configuration validated after discovery.

    ``auto_install`` declares install sources for packs listed in ``required``.
    The compiler enforces that every ``auto_install`` key is also in
    ``required``; the host boot path performs the actual install.
    """

    configuration: dict[str, dict[str, Any]] = Field(default_factory=dict)
    auto_install: dict[str, PackInstallSpec] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_auto_install_subset_of_required(self) -> PackSelection:
        required = set(self.required)
        extras = set(self.auto_install) - required
        if extras:
            extras_sorted = ", ".join(sorted(extras))
            raise ValueError(f"auto_install keys must also appear in required: {extras_sorted}")
        return self


class SessionResources(_StrictSessionModel):
    """Session-selected resources across every discovery surface."""

    packs: PackSelection = Field(default_factory=PackSelection)
    modes: ResourceSelection = Field(default_factory=ResourceSelection)
    operations: ResourceSelection = Field(default_factory=ResourceSelection)
    tools: ResourceSelection = Field(default_factory=ResourceSelection)
    servers: ResourceSelection = Field(default_factory=ResourceSelection)
    signals: ResourceSelection = Field(default_factory=ResourceSelection)
    intake: ResourceSelection = Field(default_factory=ResourceSelection)
    models: ResourceSelection = Field(default_factory=ResourceSelection)


class ProviderSelections(_StrictSessionModel):
    """Explicit ownership decisions for duplicate catalog resources."""

    modes: dict[str, str] = Field(default_factory=dict)
    operations: dict[str, str] = Field(default_factory=dict)
    tools: dict[str, str] = Field(default_factory=dict)
    capabilities: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_assignments(self) -> ProviderSelections:
        for kind, assignments in (
            ("modes", self.modes),
            ("operations", self.operations),
            ("tools", self.tools),
            ("capabilities", self.capabilities),
        ):
            for resource_id, provider_id in assignments.items():
                if not resource_id or not provider_id:
                    raise ValueError(
                        f"providers.{kind} requires non-empty resource and provider IDs"
                    )
        return self


class SessionRouting(_StrictSessionModel):
    """Static route defaults and author-curated routing overlays."""

    start_mode: str = "default"
    aliases: dict[str, str] = Field(default_factory=dict)
    priority_adjustments: dict[str, int] = Field(default_factory=dict)
    hints: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    sequential_connectors: tuple[str, ...] = ()
    """Whitespace-padded substrings that mark a query as a sequential chain
    of dependent steps (e.g. ``"then"``, ``"based on"``, ``"once"``). The
    kernel's proactive routing-hint generator and the ``planner`` action
    mode's planner consult this list to choose a single-step handoff over
    a parallel ``fork:`` fan-out: a fork runs children concurrently with
    the whole query, so no child can consume another's output. When a
    connector matches, the chain runs step by step instead.

    The list is **session-authored vocabulary** — opaque to the kernel,
    which only does case-insensitive substring matching. The session
    defines the list; the kernel has no hardcoded fallback. Add
    conservative entries here (e.g. ``"based on"``, ``"once"``) for
    surfaces that need richer sequential detection; avoid common words
    like ``"after"`` or ``"from"`` which produce false positives on
    concurrent requests."""

    resume_phrases: tuple[str, ...] = ()
    """Lowercased imperatives that mark a short user message as a
    workflow/approval resume rather than a fresh query (e.g.
    ``"proceed"``, ``"continue"``, ``"go ahead"``). The host's
    ``handle_message`` affirmative-resume path and the kernel's
    ``_looks_like_continuation_query`` detector both read this list via
    :func:`kernel.routing_vocabulary.get_resume_phrases` so the vocabulary
    stays in one place. The detectors apply entries as **prefix** matches
    on the lowered message, so ``"proceed with the current step in the
    plan"`` matches ``"proceed"``.

    The list is **session-authored vocabulary**. The session defines the
    list; the kernel has no hardcoded fallback. Keep entries to
    unambiguous imperatives; avoid common words like ``"ok"`` or
    ``"yes"`` that would false-positive on normal conversation."""

    producer_intents: tuple[str, ...] = ()
    """Intent tags whose satisfaction produces evidence a downstream
    consumer intent needs as input (e.g. ``"research"``, ``"review"``,
    ``"analyze"``). The kernel's fork guard
    (:func:`kernel.routing_vocabulary.get_producer_intents`) refuses a
    parallel fork when the gap mixes a producer intent with a consumer
    intent — the consumer can't see the producer's output — and routes
    sequential instead (single handoff to the first producer, then chain
    the consumer via post-handoff hints). A tag may appear in both
    ``producer_intents`` and ``consumer_intents`` (``review`` is
    dual-role: self-sufficient alone, but a producer when paired with
    ``code``).

    The set is **session-authored vocabulary** — opaque to the kernel.
    The session defines the set; the kernel has no hardcoded fallback.
    Override when a session adds modes with new producer-role intent
    tags."""

    consumer_intents: tuple[str, ...] = ()
    """Intent tags whose satisfaction consumes a producer's output to
    create an artifact (e.g. ``"code"``, ``"write"``, ``"implement"``).
    The counterpart to ``producer_intents``; see that field's docstring.
    The session defines the set; the kernel has no hardcoded fallback."""

    topic_abandonment_markers: tuple[str, ...] = ()
    """Lowercased substrings that signal the user is *leaving* a topic,
    not requesting it (e.g. ``"side quest"``, ``"rewind"``, ``"never
    mind"``, ``"forget"``, ``"different topic"``, ``"not about"``). When a
    marker is present in the query, the routing-hint generator's
    :func:`kernel.turn_phases._routing_hint._detect_query_intents`
    suppresses intents for any routing target whose keywords/patterns also
    appear in the query — the user is *mentioning* the topic to abandon
    it, not requesting its tools.

    Without this guard a bare-substring keyword match fires on any
    mention of a topic. "TBH the whole cloudservers thing was a side
    quest, so lets rewind" matched the ``cloudservers`` mode's ``cloud``
    keyword, routed the user INTO ``cloudservers``, and the write-contract
    backstop synthesized a destructive ``cs_apply_cloud_init`` call with
    empty args (2026-08-27 incident). This is the same class of bug as the
    ``read``/``coding`` keyword-pollution loop
    (``read-keyword-pollution-traps-coding-2026-08-03``): the detector has
    no notion of intent polarity (requesting vs. referencing vs.
    rejecting a topic).

    The list is **session-authored vocabulary** — opaque to the kernel,
    which only does case-insensitive substring matching. The session
    defines the list; the kernel has no hardcoded fallback. Keep entries
    to unambiguous discourse markers; avoid words that appear in genuine
    requests (e.g. ``"back"`` would suppress a real "go back to coding"
    handoff)."""


class InstructionFragment(_StrictSessionModel):
    """An explicit authored instruction fragment resolved at plan compilation."""

    ref: str = Field(min_length=1)
    modes: tuple[str, ...] = ()
    stages: tuple[str, ...] = ()
    position: Literal["before_mode", "after_mode", "after_stage"] = "after_mode"
    required: bool = True


class SessionInstructions(_StrictSessionModel):
    """Instruction fragment declarations; the root Markdown body is primary."""

    fragments: tuple[InstructionFragment, ...] = ()


class SessionCapabilityExecutionRequest(_StrictSessionModel):
    """Optional tighten-only limits for one capability.

    ``None`` means "no request" (host default stays in effect).
    """

    max_invocations_per_turn: int | None = Field(default=None, ge=1)
    max_internal_tool_calls: int | None = Field(default=None, ge=1)


class SessionExecutionRequest(_StrictSessionModel):
    """Optional session-authored execution-policy requests.

    Hosts treat these as tighten-only requests relative to their resolved
    runtime defaults.
    """

    max_model_calls: int | None = Field(default=None, ge=1)
    max_capability_calls: int | None = Field(default=None, ge=1)
    max_total_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    max_wall_ms: int | None = Field(default=None, ge=1)
    max_session_tokens: int | None = Field(default=None, ge=1)
    force_final_on_exhausted: bool | None = None
    capabilities: dict[str, SessionCapabilityExecutionRequest] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_capability_keys(self) -> SessionExecutionRequest:
        for capability_id in self.capabilities:
            if not capability_id.strip():
                raise ValueError("execution.capabilities keys must be non-empty")
        return self


class ContextLanePlan(_StrictSessionModel):
    """Budget and rendering constraints for one named context lane."""

    lane_id: str = Field(min_length=1)
    min_tokens: int = Field(default=0, ge=0)
    max_tokens: int | None = Field(default=None, ge=1)
    min_items: int = Field(default=0, ge=0)
    max_items: int | None = Field(default=None, ge=1)
    render_order: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_bounds(self) -> ContextLanePlan:
        if self.max_tokens is not None and self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens cannot exceed max_tokens")
        if self.max_items is not None and self.min_items > self.max_items:
            raise ValueError("min_items cannot exceed max_items")
        return self


class ContextSourceRule(_StrictSessionModel):
    """Session-owned admission and lane assignment criteria for candidates."""

    rule_id: str = Field(min_length=1)
    lane_id: str = Field(min_length=1)
    source_kinds: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    packs: tuple[str, ...] = ()
    evidence_types: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    modes: tuple[str, ...] = ()
    stages: tuple[str, ...] = ()
    priority: int = 0
    required: bool = False
    pinned: bool = False
    min_freshness: float = Field(default=0.0, ge=0.0, le=1.0)
    explicit_invocation: bool = False
    fallback: bool = False

    @model_validator(mode="after")
    def _validate_matcher(self) -> ContextSourceRule:
        has_selector = any(
            (
                self.source_kinds,
                self.sources,
                self.packs,
                self.evidence_types,
                self.tags,
                self.modes,
                self.stages,
            )
        )
        if not has_selector and not self.fallback:
            raise ValueError("a context source rule needs a selector or fallback=true")
        return self


class ContextRankingPlan(_StrictSessionModel):
    """Deterministic scoring and diversity policy for context selection."""

    relevance: float = Field(default=0.35, ge=0.0)
    mode_affinity: float = Field(default=0.10, ge=0.0)
    salience: float = Field(default=0.15, ge=0.0)
    freshness: float = Field(default=0.10, ge=0.0)
    authority: float = Field(default=0.10, ge=0.0)
    continuity: float = Field(default=0.05, ge=0.0)
    user_priority: float = Field(default=0.10, ge=0.0)
    recency: float = Field(default=0.05, ge=0.0)
    conflict_penalty: float = Field(default=0.10, ge=0.0)
    token_cost_penalty: float = Field(default=0.05, ge=0.0)
    diversity_lambda: float = Field(default=0.75, ge=0.0, le=1.0)
    relevance_provider: Literal["lexical", "semantic", "hybrid"] = "lexical"

    @property
    def normalized_positive_weights(self) -> dict[str, float]:
        """Return positive score weights normalized to one for selection."""

        raw = {
            "relevance": self.relevance,
            "mode_affinity": self.mode_affinity,
            "salience": self.salience,
            "freshness": self.freshness,
            "authority": self.authority,
            "continuity": self.continuity,
            "user_priority": self.user_priority,
            "recency": self.recency,
        }
        total = sum(raw.values())
        if total <= 0:
            raise SessionPlanError("context ranking needs at least one positive weight")
        return {name: value / total for name, value in raw.items()}


class ContextBudgetPlan(_StrictSessionModel):
    """Model-aware limits applied after fixed per-turn reservations."""

    output_reserve_tokens: int = Field(default=1024, ge=1)
    safety_margin_tokens: int = Field(default=512, ge=0)
    max_materialized_candidates: int = Field(default=32, ge=1)


class ContextHistoryPlan(_StrictSessionModel):
    """Conversation sources and deterministic digest constraints."""

    include_digest: bool = True
    include_recent_conversation: bool = True
    include_reflection: bool = False
    recent_message_limit: int = Field(default=6, ge=0)
    digest_max_lines: int = Field(default=200, ge=1)
    digest_summary_max_chars: int = Field(default=240, ge=60)


def _default_context_lanes() -> tuple[ContextLanePlan, ...]:
    return (
        ContextLanePlan(lane_id="session", min_tokens=0, render_order=0),
        ContextLanePlan(lane_id="operational", min_tokens=0, render_order=10),
        ContextLanePlan(lane_id="history", min_tokens=0, render_order=20),
        ContextLanePlan(lane_id="retrieval", min_tokens=0, render_order=30),
    )


class ModelManifestRef(_StrictSessionModel):
    """A reference to a model manifest file (``models.yaml``).

    The path may be session-relative (resolved against the resolved session
    context file's directory by the host boot path) or absolute. The compiler
    validates non-emptiness only; the host boot path resolves the path and
    checks existence. When absent, the host falls back to
    ``data_dir / models.yaml``.
    """

    path: str = Field(min_length=1)


class MemoryConsolidationTuning(_StrictSessionModel):
    """Episodic-pressure-triggered, slice-driven memory consolidation knobs.

    When ``enabled`` is ``True`` (the default), a host that wires the
    memory-consolidation slice provider counts active episodic records each
    turn and fires the configured ``consolidate_operation`` (default:
    ``consolidate_semantic``) when the episodic-count pressure (count /
    ``max_records``) crosses a threshold, paid for out of the post-turn
    slice pressure budget rather than the user's turn latency. Set
    ``enabled`` to ``False`` to disable automatic consolidation and let
    episodic records accumulate until TTL expiry.

    Thresholds are fractions of ``max_records`` (``0.0``-``1.0``). The
    pressure curve (``kernel.slice``) decides whether a between-thresholds
    turn actually runs the consolidation; the ``hard_threshold`` overrides
    the curve when episodic pressure is critical.
    """

    enabled: bool = True
    trigger: Literal["post_turn", "pre_turn", "explicit", "never"] = "post_turn"
    pressure_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    hard_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    min_interval_turns: int = Field(default=8, ge=0)
    batch_size: int = Field(default=10, ge=1)
    min_episodic_count: int = Field(default=5, ge=1)
    max_records: int = Field(default=200, ge=1)
    preferred_model_id: str = "default"
    consolidate_operation: str = "consolidate_semantic"

    @model_validator(mode="after")
    def _validate_thresholds(self) -> MemoryConsolidationTuning:
        if self.hard_threshold < self.pressure_threshold:
            raise ValueError(
                "memory.consolidation.hard_threshold must be >= "
                "memory.consolidation.pressure_threshold"
            )
        return self


class MemoryContextPlan(_StrictSessionModel):
    """Memory-store policy on the session context plan."""

    consolidation: MemoryConsolidationTuning = Field(default_factory=MemoryConsolidationTuning)


class ContextPlan(_StrictSessionModel):
    """The session-owned context mapper policy."""

    lanes: tuple[ContextLanePlan, ...] = Field(default_factory=_default_context_lanes)
    sources: tuple[ContextSourceRule, ...] = ()
    ranking: ContextRankingPlan = Field(default_factory=ContextRankingPlan)
    budget: ContextBudgetPlan = Field(default_factory=ContextBudgetPlan)
    history: ContextHistoryPlan = Field(default_factory=ContextHistoryPlan)
    memory: MemoryContextPlan = Field(default_factory=MemoryContextPlan)
    models: ModelManifestRef | None = None
    conflict_policy: Literal["prefer_authority", "surface_conflict", "drop_lower_authority"] = (
        "prefer_authority"
    )

    @model_validator(mode="after")
    def _validate_lanes_and_sources(self) -> ContextPlan:
        lane_ids = [lane.lane_id for lane in self.lanes]
        if len(lane_ids) != len(set(lane_ids)):
            raise ValueError("context lanes must have unique lane_id values")
        known_lanes = set(lane_ids)
        rule_ids = [rule.rule_id for rule in self.sources]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("context source rules must have unique rule_id values")
        fallback_count = sum(rule.fallback for rule in self.sources)
        if fallback_count > 1:
            raise ValueError("context sources allow at most one fallback rule")
        unknown_lanes = sorted({rule.lane_id for rule in self.sources} - known_lanes)
        if unknown_lanes:
            values = ", ".join(unknown_lanes)
            raise ValueError(f"context source rules reference unknown lanes: {values}")
        return self


class SessionManifestV2(_StrictSessionModel):
    """Strict front-matter contract for a session-owned Lisa harness."""

    schema_version: Literal[2]
    name: str = Field(min_length=1)
    surface: SessionSurface
    resources: SessionResources = Field(default_factory=SessionResources)
    providers: ProviderSelections = Field(default_factory=ProviderSelections)
    routing: SessionRouting = Field(default_factory=SessionRouting)
    execution: SessionExecutionRequest = Field(default_factory=SessionExecutionRequest)
    instructions: SessionInstructions = Field(default_factory=SessionInstructions)
    context: ContextPlan = Field(default_factory=ContextPlan)


class SessionInstructionPlan(_StrictSessionModel):
    """Compiled stable instruction content and provenance."""

    primary_body: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    source_sha256: str = Field(min_length=64, max_length=64)
    fragments: tuple[InstructionFragment, ...] = ()


class SessionPlan(_StrictSessionModel):
    """Immutable, validated policy compiled from one authored session source."""

    manifest: SessionManifestV2
    instructions: SessionInstructionPlan
    source_uri: str = Field(min_length=1)
    plan_digest: str = Field(min_length=64, max_length=64)
    source_dir: str | None = None
    """Directory of the resolved session-context file, as an absolute path
    string. Set by the host boot path (``compile_selected_session_plan``)
    so session-relative references like ``context.models.path`` can be
    resolved. ``None`` when the plan was compiled without a source
    directory (e.g. tests passing raw text)."""


def compile_session_plan(*, text: str, source_uri: str) -> SessionPlan:
    """Compile strict v2 front matter and its Markdown body into a plan.

    This compiler intentionally does not resolve packs or fragments. It proves
    the author-owned policy is internally valid; host bootstrap resolves it
    against installed resources and host security ceilings in a later phase.
    """

    normalized_source_uri = source_uri.strip()
    if not normalized_source_uri:
        raise SessionPlanError("session source URI is required")

    try:
        front_matter, body = parse_front_matter(text)
    except Exception as exc:  # pragma: no cover - parser-specific exception type
        raise SessionPlanError(
            f"invalid session front matter in {normalized_source_uri}: {exc}"
        ) from exc

    if not front_matter:
        raise SessionPlanError(
            f"session manifest in {normalized_source_uri} requires schema_version: 2"
        )
    try:
        manifest = SessionManifestV2.model_validate(front_matter)
    except ValidationError as exc:
        raise SessionPlanError(
            f"invalid session manifest in {normalized_source_uri}: {exc}"
        ) from exc

    primary_body = body.strip()
    if not primary_body:
        raise SessionPlanError(
            f"session instruction body in {normalized_source_uri} must not be empty"
        )

    source_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    instructions = SessionInstructionPlan(
        primary_body=primary_body,
        source_uri=normalized_source_uri,
        source_sha256=source_sha256,
        fragments=manifest.instructions.fragments,
    )
    digest_payload = {
        "manifest": manifest.model_dump(mode="json"),
        "instructions": instructions.model_dump(mode="json"),
    }
    plan_digest = hashlib.sha256(
        json.dumps(digest_payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return SessionPlan(
        manifest=manifest,
        instructions=instructions,
        source_uri=normalized_source_uri,
        plan_digest=plan_digest,
    )
