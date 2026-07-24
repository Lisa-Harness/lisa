"""Runtime-neutral Pydantic contracts for Lisa framework hosts and packs."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self, runtime_checkable

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from kernel.approval import EffectKind

if TYPE_CHECKING:
    pass  # forward-ref imports only


ConversationSurface = Literal["personal_dm", "group_chat", "channel"]
MemoryScope = Literal["personal", "shared", "channel"]
AttachmentKind = Literal["adaptive_card", "image"]

_MAX_THREAD_TURNS = 500
_MAX_RECENT_HANDOFFS = 5

CapabilityKind = Literal[
    "command", "mode", "operation", "tool", "model", "connector", "setting", "memory"
]
CapabilityStatus = Literal["available", "unavailable", "restricted", "disabled", "unknown"]
CapabilityConstraintKind = Literal[
    "config",
    "connector",
    "feature_flag",
    "model",
    "mode",
    "privacy",
    "surface",
    "user",
]
ModelRole = Literal[
    "fast_chat",
    "default_chat",
    "deep_reasoning",
    "pro_reasoning",
    "code_reasoning",
    "critic",
    "distiller",
]
ModelCapability = Literal[
    "text",
    "structured_output",
    "tools",
    "reasoning",
    "code",
    "async_research",
    "image",
    "video",
]
ModelCostTier = Literal["budget", "standard", "premium", "frontier"]
ModelLatencyTier = Literal["low", "medium", "high", "batch"]
OperationExecutionKind = Literal["runtime", "prompt"]


class InferenceProfileDefinition(BaseModel):
    """Runtime-neutral model policy for one named inference pipeline."""

    profile_id: str
    description: str = ""
    owner_package: str = "app"
    preferred_role: ModelRole | None = None
    setting_name: str = ""
    fallback_aliases: list[str] = Field(default_factory=list)
    operator_only: bool = False


class SurfaceContext(BaseModel):
    """Normalized conversation-surface metadata for one inbound or proactive turn.

    Hosts adapt their transport payloads into this shape before calling the
    framework. The context is intentionally explicit so privacy boundaries do
    not depend on prompt text or a model guessing whether a turn is private or
    shared.
    """

    surface_type: ConversationSurface = "personal_dm"
    surface_id: str = ""
    display_name: str = ""
    memory_scope: MemoryScope = "personal"
    thread_owner_id: str = ""


@runtime_checkable
class OperationHooks(Protocol):
    """Optional plugin contract for one internal operation implementation module.

    Operations are reusable internal behaviors discovered from
    host or pack operation roots and invoked by modes. They may orchestrate tools,
    query first-class data providers, or persist memory, but they are never
    directly user-routable.
    """

    async def execute(self, ctx: OperationExecutionContext) -> OperationResult: ...


class ToolExecutionContext(BaseModel):
    """Normalized request payload passed to one low-level internal tool."""

    tool_id: str = ""
    workspace_id: str = "default"
    user_id: str = ""
    user_slug: str = "default"
    surface_type: ConversationSurface = "personal_dm"
    surface_id: str = ""
    memory_scope: MemoryScope = "personal"
    mode_id: str = ""
    query: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    pack_config: Any = Field(default_factory=dict, exclude=True)
    pack_configuration_resolver: Any = Field(default=None, exclude=True)
    approval_service: Any = Field(default=None, exclude=True)
    approval_request_id: str = Field(default="", exclude=True)
    session_plan_digest: str = Field(default="", exclude=True)
    code_change_artifact_store: Any = Field(default=None, exclude=True)
    event_bus: Any = Field(default=None, exclude=True)
    state_provider: Any = Field(default=None, exclude=True)


class ToolResult(BaseModel):
    """Structured result returned by one low-level internal tool."""

    tool_id: str = ""
    output_text: str = ""
    output_json: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)
    error: str = ""


class ToolDefinition(BaseModel):
    """Catalog metadata for one internal tool discovered from content packs.

    Tools are the capability layer (distinct from operations, which are the
    behavior layer). Like operations, tools are discovered from ``tools/``
    subdirectories in content packs via the ``lisa-content`` MCP server and
    dispatched in-process through the imported ``implementation_module``.
    """

    tool_id: str
    description: str = ""
    pack_id: str = ""
    # The full distribution-package name that owns this tool (e.g.
    # ``lisa-pack-search``), inferred from the content-root path during SDK
    # catalog build. Distinct from ``pack_id`` (the raw pack slug authored in
    # YAML). Used by ``validate_export_boundaries`` to compare against
    # ``WorkflowModeConfig.mode_provider_id`` so same-pack references are not
    # misflagged as cross-pack.
    provider_id: str = ""
    effect: EffectKind = "read"
    approval_required: bool = False
    approval_preview: str = ""
    parameters_schema: type[BaseModel] | None = Field(default=None, exclude=True)
    mode_allowlist: list[str] | None = None
    max_calls_per_turn: int = Field(default=3)
    timeout_seconds: int = Field(default=30)
    module_path: str = ""
    implementation_module: Any = Field(default=None, exclude=True)
    # When ``False`` (default), this tool is private to its owning pack: a
    # mode in another pack may not reference it in ``allowed_tool_ids``. When
    # ``True``, the owning pack explicitly exports it for cross-pack use.
    exportable: bool = False
    # When ``True``, the SDK catalog loader auto-merges this tool id into
    # every mode's effective ``allowed_tool_ids`` (unless the mode opts out
    # via ``accept_default_capabilities: false`` or lists it in
    # ``excluded_tool_ids``). Lets a pack ship ambient tools (e.g. recall)
    # without forcing every mode author to re-list them. The kernel never
    # names tool ids; it only filters on the already-merged allowlist.
    default_available: bool = False

    def effective_provider_id(self) -> str:
        """Return the canonical owning-provider id for this tool.

        The single accessor for all provider-id comparisons (export
        boundaries, pack-config lookups, telemetry). Prefers the
        fully-qualified ``provider_id`` (inferred from the content root
        during SDK catalog build) and falls back to the raw authored
        ``pack_id`` slug. After SDK load both are populated, so the
        fallback is belt-and-suspenders; callers should never read
        ``provider_id``/``pack_id`` directly for comparison.
        """
        return self.provider_id or self.pack_id


class OperationExecutionContext(BaseModel):
    """Execution context passed to every internal operation."""

    workspace_id: str = "default"
    operation_id: str = ""
    user_id: str = ""
    user_slug: str = "default"
    user_name: str = ""
    principal_key: str = ""
    thread_user_id: str = ""
    conversation_id: str = ""
    surface_type: ConversationSurface = "personal_dm"
    surface_id: str = ""
    memory_scope: MemoryScope = "personal"
    mode_id: str = ""
    query: str = ""
    model_id: str = ""
    response: str = ""
    user_block: str = ""
    conversation_excerpt_markdown: str = ""
    stage_id: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    pack_config: Any = Field(default_factory=dict, exclude=True)
    pack_configuration_resolver: Any = Field(default=None, exclude=True)
    approval_service: Any = Field(default=None, exclude=True)
    approval_request_id: str = Field(default="", exclude=True)
    session_plan_digest: str = Field(default="", exclude=True)
    code_change_artifact_store: Any = Field(default=None, exclude=True)
    extra: dict[str, Any] = Field(default_factory=dict)
    tool_schemas: list[dict[str, Any]] = Field(default_factory=list, exclude=True)
    event_bus: Any = Field(default=None, exclude=True)
    complete_fn: Any = Field(default=None, exclude=True)
    resolve_model_fn: Any = Field(default=None, exclude=True)
    read_authored_text_fn: Any = Field(default=None, exclude=True)
    generate_image_fn: Any = Field(default=None, exclude=True)
    execute_tool_fn: Any = Field(default=None, exclude=True)
    execute_operation_fn: Any = Field(default=None, exclude=True)
    memory_recall_default_limit: int | None = Field(default=None, exclude=True)
    state_provider: Any = Field(default=None, exclude=True)
    introspect_fn: Any = Field(default=None, exclude=True)
    preferred_model_id: str | None = Field(default=None, exclude=True)


class OperationResult(BaseModel):
    """Structured result returned by one internal operation."""

    output_text: str = ""
    output_json: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)


class OperationDefinition(BaseModel):
    """Catalog metadata for one internal operation discovered from colocated YAML."""

    operation_id: str
    description: str = ""
    pack_id: str = ""
    # The full distribution-package name that owns this operation (e.g.
    # ``lisa-pack-memory``), inferred from the content-root path during SDK
    # catalog build. Distinct from ``pack_id`` (the raw, often short pack
    # slug authored in YAML). Used by ``validate_export_boundaries`` to
    # compare against ``WorkflowModeConfig.mode_provider_id`` so same-pack
    # references are not misflagged as cross-pack.
    provider_id: str = ""
    effect: EffectKind = "read"
    approval_required: bool = False
    approval_preview: str = ""
    execution_kind: OperationExecutionKind = "runtime"
    parameters_schema: type[BaseModel] | None = Field(default=None, exclude=True)
    response_format_model: type[BaseModel] | None = Field(default=None, exclude=True)
    response_format_model_path: str = ""
    response_output_text_field: str = "output_text"
    response_output_json_passthrough: dict[str, str] = Field(default_factory=dict)
    preferred_model_id: str | None = Field(default=None)
    tool_ids: list[str] = Field(default_factory=list)
    mode_allowlist: list[str] = Field(default_factory=list)
    max_calls_per_turn: int = Field(default=3)
    timeout_seconds: int = Field(default=60)
    completion_max_tokens: int = Field(default=1500)
    completion_temperature: float = Field(default=0.2)
    instruction_markdown_path: str = ""
    # When ``False`` (default), this operation is private to its owning pack:
    # a mode in another pack may not reference it in ``allowed_operation_ids``
    # or lifecycle bindings. When ``True``, the owning pack explicitly
    # exports it for cross-pack use. Validated generically in
    # ``validate_operation_catalog`` against ``operation_provider_id`` vs.
    # ``mode_provider_id`` — the kernel never names operation ids.
    exportable: bool = False
    # When ``True``, the SDK catalog loader auto-merges this operation id
    # into every mode's effective ``allowed_operation_ids`` (unless the mode
    # opts out via ``accept_default_capabilities: false`` or lists it in
    # ``excluded_operation_ids``). Lets a pack ship ambient operations
    # (e.g. ``recall_memory``) without forcing every mode author to re-list
    # them. The kernel never names operation ids; it only filters on the
    # already-merged allowlist.
    default_available: bool = False
    instruction_body: str = Field(default="", exclude=True)
    implementation_module: Any = Field(default=None, exclude=True)

    def effective_provider_id(self) -> str:
        """Return the canonical owning-provider id for this operation.

        The single accessor for all provider-id comparisons (export
        boundaries, pack-config lookups, telemetry). Prefers the
        fully-qualified ``provider_id`` (inferred from the content root
        during SDK catalog build) and falls back to the raw authored
        ``pack_id`` slug. After SDK load both are populated, so the
        fallback is belt-and-suspenders; callers should never read
        ``provider_id``/``pack_id`` directly for comparison.
        """
        return self.provider_id or self.pack_id


class CapabilityConstraint(BaseModel):
    """One machine-readable condition that affects a runtime capability.

    Capability manifests are compiled from existing registries, settings, and
    the hot state cache. Constraints keep help text and self-knowledge honest
    by making availability, privacy, surface, and dependency limits explicit.
    """

    constraint_type: CapabilityConstraintKind
    description: str = ""
    value: str = ""
    active: bool = True


class CapabilityEntry(BaseModel):
    """One capability Lisa can describe, route to, or decide not to use."""

    capability_id: str
    kind: CapabilityKind
    display_name: str = ""
    description: str = ""
    status: CapabilityStatus = "unknown"
    status_reason: str = ""
    invocation: str = ""
    examples: list[str] = Field(default_factory=list)
    surfaces: list[ConversationSurface] = Field(default_factory=list)
    memory_scopes: list[MemoryScope] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    constraints: list[CapabilityConstraint] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    source: str = ""
    confidence: float = 1.0
    last_checked_at: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class CapabilityManifest(BaseModel):
    """Deterministic runtime manifest of Lisa's current capability surface."""

    generated_at: str
    version: str = "1"
    entries: list[CapabilityEntry] = Field(default_factory=list)
    summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class CapabilityManifestView(BaseModel):
    """Surface-filtered view of the capability manifest for help rendering."""

    generated_at: str
    user_slug: str = "default"
    surface_type: ConversationSurface = "personal_dm"
    surface_id: str = ""
    memory_scope: MemoryScope = "personal"
    entries: list[CapabilityEntry] = Field(default_factory=list)
    hidden_count: int = 0
    filters: dict[str, Any] = Field(default_factory=dict)


class ArchitectureKnowledgeEntry(BaseModel):
    """One deterministic fact about Lisa's design, ownership, or runtime topology."""

    entry_id: str
    category: Literal[
        "layer",
        "ownership",
        "flow",
        "extension_point",
        "constraint",
        "gap",
    ]
    title: str = ""
    body: str = ""
    confidence: float = 0.0
    source_refs: list[str] = Field(default_factory=list)
    applies_to_capabilities: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    updated_at: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class LisaArchitectureAtlasArtifact(BaseModel):
    """Persisted, bounded architecture atlas generated from deterministic evidence."""

    generated_at: str
    version: str = "1"
    system_summary: str = ""
    layer_model: list[ArchitectureKnowledgeEntry] = Field(default_factory=list)
    module_ownership: list[ArchitectureKnowledgeEntry] = Field(default_factory=list)
    runtime_flows: list[ArchitectureKnowledgeEntry] = Field(default_factory=list)
    extension_points: list[ArchitectureKnowledgeEntry] = Field(default_factory=list)
    constraints: list[ArchitectureKnowledgeEntry] = Field(default_factory=list)
    known_gaps: list[ArchitectureKnowledgeEntry] = Field(default_factory=list)
    source_manifest_generated_at: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


SelfKnowledgeCategory = Literal[
    "identity",
    "capability",
    "limitation",
    "architecture",
    "memory",
    "grounding",
    "known_unknown",
]


class SelfKnowledgeEntry(BaseModel):
    """One deterministic fact in Lisa's scoped operational self-model."""

    entry_id: str
    category: SelfKnowledgeCategory
    title: str = ""
    body: str = ""
    confidence: float = 0.0
    source_refs: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    updated_at: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Live introspection (Python-level runtime snapshot)
#
# These helpers are defined BEFORE the artifacts that reference them so the
# module resolves top-down. With `from __future__ import annotations` already
# in effect, no forward-reference quotes are needed or wanted.
# ---------------------------------------------------------------------------

McpConnectionStatus = Literal["warm", "cold", "failed", "unknown"]


class McpServerHealth(BaseModel):
    """Health snapshot for one MCP server in the host's tool pool."""

    server_id: str
    status: McpConnectionStatus = "unknown"
    warm: bool = False
    last_error: str = ""


class TokenUsageSummary(BaseModel):
    """Aggregated token-usage summary derived from the telemetry JSONL tail."""

    total_cost_estimate: float = 0.0
    total_tokens: int = 0
    top_model: str = ""
    top_model_tokens: int = 0
    spike_at: str = ""  # ISO timestamp of the highest-cost event, or empty
    event_count: int = 0


class LogErrorEntry(BaseModel):
    """One error or warning entry extracted from the turn log tail."""

    ts: str = ""
    level: str = ""
    event: str = ""
    message: str = ""
    server: str = ""


OperationalHealth = Literal["healthy", "degraded", "critical", "unknown"]


class LisaSelfModelArtifact(BaseModel):
    """Persisted, surface-scoped operational self-model for Lisa."""

    generated_at: str
    version: str = "1"
    workspace_id: str = "default"
    user_slug: str = "default"
    surface_type: ConversationSurface = "personal_dm"
    surface_id: str = ""
    memory_scope: MemoryScope = "personal"
    identity_summary: str = ""
    grounding_summary: str = ""
    identity: list[SelfKnowledgeEntry] = Field(default_factory=list)
    capabilities: list[SelfKnowledgeEntry] = Field(default_factory=list)
    limitations: list[SelfKnowledgeEntry] = Field(default_factory=list)
    architecture: list[SelfKnowledgeEntry] = Field(default_factory=list)
    memory_awareness: list[SelfKnowledgeEntry] = Field(default_factory=list)
    grounding: list[SelfKnowledgeEntry] = Field(default_factory=list)
    known_unknowns: list[SelfKnowledgeEntry] = Field(default_factory=list)
    source_manifest_generated_at: str = ""
    # ── Live runtime telemetry (populated by introspection, not catalog walk) ──
    uptime_seconds: float = 0.0
    asyncio_tasks: int = 0
    python_threads: int = 0
    process_rss_mb: float = 0.0
    mcp_pool_health: dict[str, McpServerHealth] = Field(default_factory=dict)
    token_usage_24h: TokenUsageSummary | None = None
    recent_log_errors: list[LogErrorEntry] = Field(default_factory=list)
    operational_health: str = "unknown"  # healthy|degraded|critical|unknown
    operational_health_reasons: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IntrospectSnapshot(BaseModel):
    """Live Python-level introspection snapshot for the running host process."""

    captured_at: str
    uptime_seconds: float = 0.0
    asyncio_tasks: int = 0
    python_threads: int = 0
    python_modules: int = 0
    process_rss_mb: float = 0.0
    process_cpu_percent: float = 0.0
    mcp_pool_health: dict[str, McpServerHealth] = Field(default_factory=dict)
    token_usage_24h: TokenUsageSummary | None = None
    recent_log_errors: list[LogErrorEntry] = Field(default_factory=list)
    logfile_tail_lines: int = 0
    failed_mcp_servers: int = 0
    recent_error_count: int = 0
    operational_health: str = "unknown"  # healthy|degraded|critical|unknown
    operational_health_reasons: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Reflective self-assessment (LLM-generated from inventory + introspection)
# ---------------------------------------------------------------------------

ReflectionHealth = Literal["healthy", "degraded", "struggling", "unknown"]


class LisaReflectionArtifact(BaseModel):
    """LLM-generated holistic assessment of Lisa's current operational state."""

    generated_at: str
    overall_health: ReflectionHealth = "unknown"
    health_reasoning: str = ""
    attention_items: list[str] = Field(default_factory=list)
    behavioral_patterns: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    self_model_snapshot_generated_at: str = ""


# ---------------------------------------------------------------------------
# Dev-state outbox (pushed from VS Code extension)
# ---------------------------------------------------------------------------


class OutboxEntry(BaseModel):
    ts: str  # ISO 8601
    type: str  # canonical: "accomplishment" | "doc-save" | "memory-save" | "note"
    # legacy aliases still accepted at parse boundary: "doc_save", "memory_update"
    summary: str
    files: list[str] = Field(default_factory=list)
    commits: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Generic intelligence pipeline (collection → analysis → synthesis → memory)
# ---------------------------------------------------------------------------


class IntelItem(BaseModel):
    """One raw item from an arbitrary external source.

    Domain-agnostic base class for any intelligence collection pipeline.
    SDK/app layers subclass this to add source-specific fields.
    """

    id: str = ""
    source_type: str = ""  # e.g. "hackernews", "reddit", "clarity", "stripe_webhook"
    source_id: str = ""  # specific subreddit, site ID, feed URL
    title: str = ""
    url: str = ""
    body_text: str = ""
    excerpt: str = ""
    author: str = ""
    score: float = 0.0  # generic relevance/popularity signal
    created_at: str = ""
    collected_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    raw_metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    language: str = ""


class IntelCollection(BaseModel):
    """A batch of collected items from one or more sources.

    Produced by data-source connectors; consumed by the analysis pipeline.
    The kernel treats this as an opaque bag of scored items.
    """

    generated_at: str
    window_hours: int = 24
    item_count: int = 0
    source_counts: dict[str, int] = Field(default_factory=dict)
    collection_warnings: list[str] = Field(default_factory=list)
    items: list[IntelItem] = Field(default_factory=list)
    summary_bullets: list[str] = Field(default_factory=list)
    collection_metadata: dict[str, Any] = Field(default_factory=dict)


class IntelScoredItem(IntelItem):
    """An item after local deterministic scoring.

    The generic pipeline adds velocity and clustering scores.
    SDK layers may add domain-specific scores in ``custom_scores``.
    """

    velocity_score: float = 0.0
    theme_cluster: str = ""
    custom_scores: dict[str, float] = Field(default_factory=dict)


class IntelLocalAnalysis(BaseModel):
    """Stage A: deterministic analysis (no model calls).

    Produced by the kernel's generic scoring pipeline from an IntelCollection.
    SDK layers subclass to add domain-specific analysis fields.
    """

    generated_at: str
    window_hours: int = 24
    item_count: int = 0
    directional_query: str = ""
    top_themes: list[str] = Field(default_factory=list)
    search_seeds: list[str] = Field(default_factory=list)
    scored_items: list[IntelScoredItem] = Field(default_factory=list)
    analysis_metadata: dict[str, Any] = Field(default_factory=dict)


class IntelSynthesis(BaseModel):
    """Stage B+C: model-generated synthesis.

    Produced by the kernel's synthesis pipeline from an IntelLocalAnalysis.
    The ``vibe_markdown`` is a short summary; ``detailed_markdown`` is the
    full synthesis.  SDK layers subclass to add domain-specific fields.
    """

    generated_at: str
    directional_query: str = ""
    vibe_markdown: str = ""
    detailed_markdown: str = ""
    search_queries_used: list[str] = Field(default_factory=list)
    synthesis_metadata: dict[str, Any] = Field(default_factory=dict)


class IntelTheme(BaseModel):
    """One durable theme distilled from intelligence analysis.

    Generic enough for any domain: audience themes, competitor themes,
    operational themes, etc.  SDK layers subclass to add domain fields.
    """

    theme_id: str
    label: str
    summary: str = ""
    confidence: float = 0.0
    evidence_item_ids: list[str] = Field(default_factory=list)
    custom_attributes: dict[str, Any] = Field(default_factory=dict)


class IntelMemoryArtifact(BaseModel):
    """Compact system-scoped memory from intelligence analysis.

    The kernel stores these per-scope ("audience", "competitor", "ops", etc.)
    so each intelligence domain has its own durable memory.
    """

    generated_at: str
    summary_markdown: str = ""
    emerging_themes: list[IntelTheme] = Field(default_factory=list)
    declining_themes: list[IntelTheme] = Field(default_factory=list)
    notable_entities: list[str] = Field(default_factory=list)
    recommended_angles: list[str] = Field(default_factory=list)
    memory_metadata: dict[str, Any] = Field(default_factory=dict)


class SignalCollectorError(BaseModel):
    kind: str
    message: str
    status_code: int | None = None
    retryable: bool = False


class SignalCollectorRun(BaseModel):
    collector_id: str
    run_mode: str = "scheduled"  # "scheduled" | "on_demand" | "live_test"
    status: str = "ok"  # "ok" | "degraded" | "unconfigured" | "error"
    started_at: str
    finished_at: str
    duration_ms: int = 0
    attempts: int = 0
    error: SignalCollectorError | None = None


class SignalCollectorRefreshResult(BaseModel):
    run: SignalCollectorRun
    snapshot: BaseModel | None = None


class ModelVerificationError(BaseModel):
    kind: str
    message: str


class ModelVerificationResult(BaseModel):
    model_alias: str
    provider: str
    model_id: str
    display_name: str
    checked_at: str
    status: str = "ok"  # "ok" | "unconfigured" | "error"
    duration_ms: int = 0
    request_id: str = ""
    output_preview: str = ""
    error: ModelVerificationError | None = None


# ---------------------------------------------------------------------------
# Developer mind-state (from outbox)
# ---------------------------------------------------------------------------


class DevState(BaseModel):
    last_updated: str = ""
    recent_accomplishments: list[OutboxEntry] = Field(default_factory=list)
    current_branch: str = ""
    active_files: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


RepoLookupStatus = Literal["pending", "answered", "not_found", "failed", "expired"]


class RepoLookupRequest(BaseModel):
    """Request for exact code or docs evidence from a configured Git repository."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    schema_version: str = "1"
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    repo: str = ""
    requester: str = "lisa"
    query: str
    context: str = ""
    max_results: int = 8


class RepoLookupEvidence(BaseModel):
    """One grounded file/snippet match returned by a repo lookup responder."""

    path: str
    summary: str = ""
    snippet: str = ""
    line_start: int | None = None
    line_end: int | None = None


class RepoLookupResponse(BaseModel):
    """Response to a repo lookup request with commit-bound evidence."""

    request_id: str
    schema_version: str = "1"
    status: RepoLookupStatus = "answered"
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    repo: str = ""
    git_sha: str = ""
    answer_summary: str = ""
    confidence: float = 0.0
    evidence: list[RepoLookupEvidence] = Field(default_factory=list)
    error: str = ""


# ---------------------------------------------------------------------------
# Unified state (hot context cache for host queries)
# ---------------------------------------------------------------------------


class AnomalyAlert(BaseModel):
    severity: str  # "info" | "warn" | "critical"
    source: str
    message: str
    detected_at: str


ContextEvidenceNamespace = str
ContextEvidenceSourceType = str


class ContextRoutingNamespaceConfig(BaseModel):
    """Routing rules for mapping evidence to a specific namespace."""

    match_sources: list[str] = Field(default_factory=list)
    match_types: list[str] = Field(default_factory=list)
    match_tags: list[str] = Field(default_factory=list)
    is_fallback: bool = False


class ContextRoutingConfig(BaseModel):
    """User-facing configuration for how evidence is categorized and ordered."""

    render_order: list[str] = Field(default_factory=list)
    namespaces: dict[str, ContextRoutingNamespaceConfig] = Field(default_factory=dict)


ContextConflictRisk = Literal["low", "medium", "high"]


class ContextEvidence(BaseModel):
    """One structured evidence item available for per-turn context selection."""

    evidence_id: str
    namespace: ContextEvidenceNamespace = "general_context"
    source: str = ""
    source_type: ContextEvidenceSourceType = "general"
    collector_id: str = ""
    evidence_type: str = ""
    title: str = ""
    body: str = ""
    collected_at: str = ""
    freshness_expires_at: str = ""
    confidence: float = 0.0
    salience_score: float = 0.0
    relevance_score: float = 0.0
    freshness_score: float = 0.0
    recency_score: float = 0.0
    combined_score: float = 0.0
    conflict_risk: ContextConflictRisk = "low"
    # Normalized distribution package name that owns this evidence (e.g.
    # ``lisa-pack-cloudflare``). Populated by the EntryPointExtensionEvidenceBuilder
    # for extension-contributed evidence; empty for kernel-core evidence
    # (alerts, self-model, domain knowledge, etc.). Used by the context broker
    # to enforce pack-scoped evidence boundaries — a mode only sees extension
    # evidence from its own pack plus any packs it explicitly allows via
    # ``context_evidence_packs`` in its ``_mode.md`` front-matter.
    pack: str = ""
    tags: list[str] = Field(default_factory=list)
    match_reasons: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextPacket(BaseModel):
    """One ranked, prompt-safe evidence packet for a single turn."""

    compiled_at: str
    query: str = ""
    mode_id: str = "default"
    stage_id: str = ""
    total_available_evidence: int = 0
    selected_evidence: list[ContextEvidence] = Field(default_factory=list)
    dropped_evidence_count: int = 0
    truncation_reason: str = ""
    ranking_metadata: dict[str, Any] = Field(default_factory=dict)


class ContextBrokerRequest(BaseModel):
    """Request payload for the MCP context broker."""

    query: str
    mode_id: str = "default"
    stage_id: str = ""
    surface: SurfaceContext = Field(default_factory=SurfaceContext)
    user_slug: str = "default"
    workspace_id: str = "default"
    connectors: list[str] = Field(default_factory=list)
    context_budget: dict[str, Any] = Field(default_factory=dict)
    # Pack-scoped evidence boundary: the mode's own provider pack and any
    # explicitly allowed packs whose extension evidence should be included.
    mode_provider_id: str = ""
    context_evidence_packs: list[str] = Field(default_factory=list)


class ContextBrokerResponse(BaseModel):
    """Response payload from the MCP context broker."""

    packet: ContextPacket | None = None
    packet_markdown: str = ""
    context_tag: str = ""
    warnings: list[str] = Field(default_factory=list)


ContextPromptStrategy = Literal["broker_packet",]


class LoadedContextPart(BaseModel):
    """One dynamic prompt fragment recorded for previous-turn replay.

    The agent stores these parts in the exact order they were loaded into the
    model input so `/debug context` can replay the prior turn without
    reconstructing from telemetry.
    """

    key: str
    role: Literal["system", "user", "assistant"] = "user"
    phase: Literal["system_addition", "history", "user_context"] = "user_context"
    lane: str = ""
    hidden: bool = False
    content: str = ""
    char_count: int = 0


class LoadedContextSnapshot(BaseModel):
    """Exact dynamic prompt context loaded for one assistant turn.

    This snapshot intentionally excludes the static base system prompt so thread
    state stays compact while still preserving the dynamic context that changed
    this turn: planner additions, history injections, and the assembled user
    context payload.
    """

    recorded_at: str = ""
    prompt_context_strategy: ContextPromptStrategy = "broker_packet"
    context_tag: str = ""
    active_context_sources: list[str] = Field(default_factory=list)
    parts: list[LoadedContextPart] = Field(default_factory=list)
    final_user_content: str = ""
    final_user_content_char_count: int = 0
    context_budget: dict[str, Any] = Field(default_factory=dict)


DomainKnowledgeStatus = Literal["ok", "warning", "error", "unconfigured"]


class DomainKnowledgeArtifact(BaseModel):
    """One pack-owned domain-knowledge artifact surfaced through a generic seam."""

    artifact_id: str = ""
    display_name: str = ""
    generated_at: str = ""
    status: DomainKnowledgeStatus = "ok"
    summary_bullets: list[str] = Field(default_factory=list)
    prompt_context_markdown: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class UnifiedState(BaseModel):
    compiled_at: str
    context_routing: ContextRoutingConfig | None = None
    signal_snapshots: dict[str, dict[str, Any]] = Field(default_factory=dict)
    intel_collections: dict[str, IntelCollection] = Field(default_factory=dict)
    intel_analysis: dict[str, IntelSynthesis] = Field(default_factory=dict)
    intel_memory: dict[str, IntelMemoryArtifact] = Field(default_factory=dict)
    domain_knowledge: dict[str, DomainKnowledgeArtifact] = Field(default_factory=dict)
    dev_state: DevState | None = None
    active_alerts: list[AnomalyAlert] = Field(default_factory=list)
    collector_runs: dict[str, SignalCollectorRun] = Field(default_factory=dict)
    context_evidence_catalog: list[ContextEvidence] = Field(default_factory=list)
    context_markdown: str = ""  # pre-rendered markdown context for prompt injection
    thinking_vibe: str = ""
    strategy_vibe: str = ""
    # Compact runtime capability and architecture pointers for help/routing context.
    capability_manifest_summary: str = ""
    architecture_vibe: str = ""
    self_model_vibe: str = ""
    reflection_vibe: str = ""


# ---------------------------------------------------------------------------
# IPSAR memory
# ---------------------------------------------------------------------------

MemoryTier = Literal["daily", "weekly", "long_term", "permanent"]


class MemoryRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # External backends (Mem0, Hindsight, …) use UUID strings; first-party
    # lisa_memory uses integer ids. Accept both so model_validate doesn't
    # reject external-backend records at the alias boundary.
    id: int | str
    workspace_id: str = "default"
    user_id: str = ""
    user_slug: str = "default"
    memory_scope: MemoryScope = "personal"
    surface_id: str = ""
    source_user_id: str = ""
    source_user_slug: str = ""
    agent_id: str = "core"
    operation_id: str = ""
    instruction_id: str = ""
    memory_tier: MemoryTier = "daily"
    # Accept both "content" (first-party lisa_memory) and "memory" (Mem0)
    # as source keys. This is a secondary safety net — the primary
    # normalization happens at the alias boundary via result_maps.
    content: str = Field(
        default="",
        validation_alias=AliasChoices("content", "memory"),
    )
    content_json: dict[str, Any] = Field(default_factory=dict)
    salience: float = 0.0
    created_at: str
    memory_kind: Literal["episodic", "semantic", "procedural", "reflection", "preference"] = (
        "episodic"
    )
    confidence: float = 1.0
    evidence_ids: list[int] = Field(default_factory=list)
    supersedes_id: int | None = None
    valid_until: str | None = None
    last_accessed_at: str | None = None
    access_count: int = 0
    status: Literal["active", "stale", "disputed", "retired"] = "active"


class MemoryWriteRequest(BaseModel):
    workspace_id: str = "default"
    user_id: str = ""
    user_slug: str = "default"
    memory_scope: MemoryScope = "personal"
    surface_id: str = ""
    source_user_id: str = ""
    source_user_slug: str = ""
    agent_id: str = "core"
    operation_id: str = ""
    instruction_id: str = ""
    memory_tier: MemoryTier = "daily"
    content: str = ""
    content_json: dict[str, Any] = Field(default_factory=dict)
    salience: float = 0.0
    memory_kind: Literal["episodic", "semantic", "procedural", "reflection", "preference"] = (
        "episodic"
    )
    confidence: float = 1.0
    evidence_ids: list[int] = Field(default_factory=list)
    supersedes_id: int | None = None
    valid_until: str | None = None


class MemoryEmbeddingRecord(BaseModel):
    """One persisted embedding row tied to a single memory record."""

    memory_id: int
    workspace_id: str = "default"
    user_slug: str = "default"
    provider: str = "azure_openai"
    model_id: str = ""
    content_hash: str = ""
    dimensions: int = 0
    vector: list[float] = Field(default_factory=list)
    created_at: str
    updated_at: str


class MemoryRecallEvidence(BaseModel):
    """One ranked memory candidate returned by the recall engine."""

    memory_id: int | str
    created_at: str
    content: str
    memory_scope: MemoryScope = "personal"
    surface_id: str = ""
    source_user_slug: str = ""
    operation_id: str = ""
    memory_tier: MemoryTier = "daily"
    salience: float = 0.0
    combined_score: float = 0.0
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    recency_score: float = 0.0
    match_reasons: list[str] = Field(default_factory=list)


class MemoryRecallContext(BaseModel):
    """Structured retrieval context used before recall answer synthesis."""

    query: str
    memory_scope: MemoryScope = "personal"
    surface_id: str = ""
    confidence: float = 0.0
    deep_hint: bool = False
    used_semantic_search: bool = False
    semantic_backfill_count: int = 0
    candidate_pool_size: int = 0
    candidates: list[MemoryRecallEvidence] = Field(default_factory=list)


class MemoryRecallAnswer(BaseModel):
    """Final synthesized answer returned from the supercharged recall flow."""

    query: str
    found: bool = False
    answer: str = ""
    confidence: float = 0.0
    used_semantic_search: bool = False
    auto_deep_used: bool = False
    uncertainty_note: str = ""
    evidence: list[MemoryRecallEvidence] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# IPSAR workflow FSM
# ---------------------------------------------------------------------------

# WorkflowModeId and WorkflowConnectorId are plain str so that new modes and
# connectors can be added by dropping a folder into a discovered modes root without
# any code change. FSM invariants are enforced at startup by validate_workflow_catalog().
WorkflowModeId = str
WorkflowConnectorId = str
ModeKind = Literal["workflow", "chat", "action"]


class WorkflowStageConfig(BaseModel):
    """Defines one stage in a mode-specific workflow finite state machine.

    Loaded at startup via the ``lisa-content`` MCP server from YAML front-matter
    blocks in discovered mode-root `{mode}/{stage_id}.md` files.  The instruction_body
    field holds the body text below the front-matter delimiter so that runtime
    prompt injection never needs a disk read.
    """

    stage_id: str
    # Relative path under the loaded mode root kept for reference / debugging.
    # The body text is pre-loaded into instruction_body at startup.
    instruction_markdown_path: str = ""
    # Parsed body text below the front-matter --- delimiter.
    instruction_body: str = ""
    description: str = ""
    default_model: str | None = None
    required_connectors: list[WorkflowConnectorId] = Field(default_factory=list)
    allowed_next_stages: list[str] = Field(default_factory=list)
    allow_user_questions: bool = True
    terminal: bool = False


class DefaultIntentBridge(BaseModel):
    """Generic configuration for bridging natural-language intents from default chat to modes.

    Content pack modes can declare deterministic natural-language patterns that should
    automatically route to that mode from default chat, without requiring explicit slash
    commands. This allows SDK/pack layers to define product-specific intents (e.g., repo
    investigation or claim review) while keeping app routing mechanics
    generic and reusable.

    Bridges run before freeform completion so they stay cheap and deterministic. They are
    invoked only while the active route is default chat, but the declarations themselves
    can live on any loaded mode so SDK packs keep ownership of product-specific intents.
    Slash-command routes always take precedence.

    The matcher patterns and extraction logic are authored by the SDK pack and declared
    in the mode's _mode.md front-matter under `default_intent_bridges`.
    """

    target_mode_id: str  # Which mode to route to (e.g. "claims", "repo")
    matcher_patterns: list[str] = Field(default_factory=list)  # Regex patterns to match intent
    query_extraction_strategy: str = (
        "passthrough"  # "passthrough" (use full message) or "pattern" (extract from pattern group)
    )
    prerequisites: list[str] = Field(
        default_factory=list
    )  # Optional checks: "require_recent_assistant_list", "require_recent_handoff"
    recent_handoff_source_modes: list[str] = Field(default_factory=list)
    recent_handoff_required_artifact_keys: list[str] = Field(default_factory=list)
    recent_handoff_context_key: str = ""
    response_appendix: str = ""  # Optional text to append to mode response (e.g. grounding hint)


class TurnBudget(BaseModel):
    """Whole-turn resource budget spanning inference + compute.

    This is the missing "ratio" knob. The host expresses the mix of model
    work and tool work as one budget; the turn loop enforces it across all
    iterations regardless of which dispatch branch (operation / tool / MCP)
    fired.

    A budget field is "unlimited" when set to 0. An exhausted budget forces a
    final no-tools completion so the turn still returns a coherent response.
    """

    max_tool_calls: int = Field(default=0, ge=0, description="0 = unlimited.")
    max_total_tokens: int = Field(default=0, ge=0, description="0 = unlimited.")
    max_wall_ms: int = Field(default=0, ge=0, description="0 = unlimited.")


class CallLedger(BaseModel):
    """Per-turn call counter shared across all dispatch branches.

    Enforces ``max_calls_per_turn`` uniformly for operations, pack tools, and
    MCP tools. Without this, only declarative operations were bounded; the
    main turn loop enforced only a global iteration count, not a per-target
    cap.
    """

    counts: dict[str, int] = Field(default_factory=dict)

    def remaining(self, target_id: str, max_calls: int) -> int:
        """Return how many more calls to ``target_id`` are permitted.

        ``max_calls <= 0`` means unlimited. The ledger never blocks unlimited
        targets; it still records the count so callers can observe it.
        """
        if max_calls <= 0:
            return max_calls
        return max(0, max_calls - self.counts.get(target_id, 0))

    def record(self, target_id: str) -> None:
        """Record one call to ``target_id``."""
        self.counts[target_id] = self.counts.get(target_id, 0) + 1

    def is_exhausted(self, target_id: str, max_calls: int) -> bool:
        """Return True when ``target_id`` has hit its per-turn cap."""
        if max_calls <= 0:
            return False
        return self.counts.get(target_id, 0) >= max_calls


class WorkflowModeConfig(BaseModel):
    """Defines all stage metadata for a single workflow mode.

    Loaded at startup via the ``lisa-content`` MCP server from a _mode.md file
    at the root of each discovered mode folder. Adding a new mode
    requires only a new folder — no code changes.

    Modes are markdown-first user-facing entry points. Workflow modes own an
    FSM stage graph; chat modes run through the normal conversational turn
    flow; action modes are zero-stage one-shot entry points. Internal reusable
    behavior is bound via operation identifiers declared in the mode front
    matter.
    """

    mode_id: WorkflowModeId
    start_stage_id: str
    stages: dict[str, WorkflowStageConfig] = Field(default_factory=dict)
    mode_kind: ModeKind = "workflow"
    artifact_type: str = "workflow_summary"
    description: str = ""
    instruction_body: str = ""
    # Absolute path to the mode's _mode.md source file populated during catalog load.
    mode_markdown_path: str = ""
    default_model: str | None = None
    # Routing metadata for natural-language mode selection.
    routable: bool = False
    priority: int = 0
    # Optional regex patterns for high-precision fast-path candidate detection.
    # Patterns are authored in mode front matter and evaluated case-insensitively.
    fast_path_patterns: list[str] = Field(default_factory=list)
    # Intent tags this mode declares it can satisfy (e.g. ["research", "reasoning"]).
    # Used by the proactive routing hint generator to detect when a single mode
    # covers all the intents in the user's query (research-then-reason), vs.
    # when a fork fan-out is needed (learn + write). Authored in _mode.md front
    # matter. The kernel never hardcodes mode-specific intent semantics.
    routing_intent_tags: list[str] = Field(default_factory=list)
    # Per-tag substring keywords for the proactive routing hint generator.
    # Maps each intent tag to the broad substring keywords that signal that
    # specific intent in a user query (e.g. {"research": ["latest", "news"],
    # "reasoning": ["swot", "analyze"]}). Case-insensitive substring match.
    # Authored in _mode.md front matter under ``routing_intent_keywords``.
    # Each keyword only attributes its associated tag, not all of the mode's
    # tags, so a "latest" match does not falsely trigger a "reasoning" hint.
    routing_intent_keywords: dict[str, list[str]] = Field(default_factory=dict)
    # Intents this mode satisfies in-place when it is the *current* mode, so
    # the proactive routing hint generator knows not to suggest a handoff for
    # them. The symmetric counterpart to ``routing_intent_tags``: that field
    # declares what intents a mode *covers* when routing *to* it; this field
    # declares what intents a mode already *handles* when routing *from* it.
    # Both share the same intent-tag vocabulary (research, reasoning, learn,
    # write, ...) authored by packs; the kernel never hardcodes mode-specific
    # intent semantics or tool names. Authored in _mode.md front matter under
    # ``satisfies_intents``. Default empty means every detected intent is a
    # gap → a handoff hint is offered (the safe direction for modes that have
    # not opted in, e.g. ``default``).
    satisfies_intents: list[str] = Field(default_factory=list)
    # ``ModelCapability`` strings (``tools``, ``code``, ``reasoning``, ...) the
    # resolved model must declare for this mode to run safely. Authored in
    # ``_mode.md`` front matter under ``required_capabilities``. The turn
    # runner validates the resolved model's ``ManifestModelEntry.capabilities``
    # against this list before the tool loop starts; when a router is wired and
    # the model is missing a required capability, the turn returns a clear
    # user-facing block instead of letting a weak model reach for write-effect
    # tools it cannot use reliably. Default empty means no capability gate
    # (backward compatible — every model is admitted). The kernel never
    # hardcodes specific capability names; they are pack-authored vocabulary
    # matching ``ModelCapability``.
    required_capabilities: list[str] = Field(default_factory=list)
    # Python regex string used by parse_route() to detect activation.
    # Must include a named group (?P<query>...) for the trailing user text.
    command_pattern: str = ""
    require_query: bool = False
    # Lifecycle bindings accept BOTH operation IDs and tool IDs. The dispatcher
    # resolves each ID against the operation catalog first, then the tool catalog.
    # This dissolves the historical asymmetry where only operations could run in
    # pre/post-turn slots — a deterministic compute step no longer needs an
    # operation wrapper to participate in the lifecycle.
    entry_operation_ids: list[str] = Field(default_factory=list)
    pre_turn_operation_ids: list[str] = Field(default_factory=list)
    post_turn_operation_ids: list[str] = Field(default_factory=list)
    allowed_operation_ids: list[str] = Field(default_factory=list)
    allowed_tool_ids: list[str] = Field(default_factory=list)
    # The explicit ``allowed_operation_ids`` / ``allowed_tool_ids`` as authored
    # in ``_mode.md`` front matter, *before* the SDK catalog loader merges in
    # any ``default_available`` capabilities. The merged (effective) sets live
    # in ``allowed_operation_ids`` / ``allowed_tool_ids`` above; these authored
    # sets let the kernel distinguish "the mode declared action tools" from
    # "the mode inherited ambient capabilities" — e.g. ``_fork_mode_requires_tools``
    # checks the authored set so a chat-only mode that inherits ``recall_memory``
    # is not misflagged as requiring tools to complete successfully.
    authored_allowed_operation_ids: list[str] = Field(default_factory=list)
    authored_allowed_tool_ids: list[str] = Field(default_factory=list)
    # Failure policy for entry (action-mode) bindings. ``abort`` (default) is
    # the historical behavior: a failing entry target kills the mode transition.
    # ``skip`` logs the failure and continues to the next binding, mirroring the
    # failure isolation pre/post-turn hooks already enjoy.
    entry_failure_policy: Literal["abort", "skip"] = "abort"
    # When True (default), the mode's allowed_tool_ids and allowed_operation_ids
    # are authoritative: empty means no tools/operations are advertised to the
    # LLM. When False, empty means all pass through.
    tools_authoritative: bool = True
    # When True (default), the SDK catalog loader auto-merges any operation
    # or tool flagged ``default_available: true`` into this mode's effective
    # ``allowed_operation_ids`` / ``allowed_tool_ids``. Lets a pack ship
    # ambient capabilities (e.g. ``recall_memory``) without forcing every
    # mode author to re-list them. A mode that genuinely wants zero tools
    # sets this to False. The merge happens entirely in the SDK catalog
    # layer; the kernel only ever sees the already-merged allowlist.
    accept_default_capabilities: bool = True
    # Operation ids to subtract from the default-available merge, letting a
    # mode opt out of specific ambient operations while keeping the rest.
    # Only consulted when ``accept_default_capabilities`` is True. Does not
    # affect capabilities the mode lists explicitly in
    # ``allowed_operation_ids`` — those survive regardless.
    excluded_operation_ids: list[str] = Field(default_factory=list)
    # Tool ids to subtract from the default-available merge. Symmetric to
    # ``excluded_operation_ids``.
    excluded_tool_ids: list[str] = Field(default_factory=list)
    # Natural-language intent bridges: declarative routing from default chat to modes (SDK-owned).
    default_intent_bridges: list[DefaultIntentBridge] = Field(default_factory=list)
    # The distribution package name that owns this mode (e.g.
    # ``lisa-pack-cloudflare``). Inferred from the content-root path during
    # catalog build unless explicitly set via ``mode_provider_id`` in the
    # mode's ``_mode.md`` front-matter. Used to enforce pack-scoped evidence
    # boundaries: the context broker auto-includes extension evidence from
    # the mode's own provider.
    mode_provider_id: str = ""
    # Extension packs whose context evidence this mode wants injected in
    # addition to its own provider's evidence. Each entry is a normalized
    # pack name (e.g. ``lisa-pack-cloudflare``). When empty (default), only
    # the mode's own pack contributes extension evidence. When non-empty,
    # the union of the mode's own pack and these packs is admitted. Kernel-
    # core evidence (alerts, self-model, domain knowledge, intel) is never
    # gated by this field — it is always available to every mode.
    context_evidence_packs: list[str] = Field(default_factory=list)
    # Opaque extension front matter the kernel does not interpret. Carries
    # app/pack-owned keys (e.g. ``lenses``) from ``_mode.md`` and ``_overrides.yaml``
    # so the kernel never has to name or allowlist app-owned concepts. Hosts and
    # packs read this dict directly; the kernel only persists and passes it through.
    extra_front_matter: dict[str, Any] = Field(default_factory=dict)


class WorkflowSessionState(BaseModel):
    """Tracks active workflow state for a user while a mode session is running."""

    user_id: str
    user_slug: str = "default"
    mode_id: WorkflowModeId
    stage_id: str
    entered_at: str
    turn_count: int = 0
    stage_data: dict[str, Any] = Field(default_factory=dict)


class CheckpointBundle(BaseModel):
    """Serialized snapshot of kernel + host state at a named point.

    The kernel builds this from ``UnifiedState`` (via the host's
    ``state_provider``) and ``WorkflowSessionState`` (via the host's
    ``workflow_session`` hook) at checkpoint time. The host's
    ``on_checkpoint`` hook persists it — typically by calling
    ``save_checkpoint`` on the ``lisa-storage`` MCP server through the
    tool pool. On restore, the host deserializes the bundle and feeds
    the ``unified_state`` and ``workflow_session_state`` dicts back into
    the turn runner's next invocation.

    ``turn_history`` is host-owned: the kernel leaves it empty; the host
    appends its own conversation thread before persisting.

    The enrichment fields (``overview``, ``work_done``,
    ``important_files``, ``next_steps``) are populated by a
    checkpoint-pack operation that calls the model to summarize the
    session at checkpoint time. They default to empty so the kernel's
    bare bundle still works without enrichment.
    """

    checkpoint_id: str
    created_at: str
    label: str = ""
    mode_id: str = ""
    stage_id: str = ""
    turn_count: int = 0
    unified_state: dict[str, Any] = Field(default_factory=dict)
    workflow_session_state: dict[str, Any] = Field(default_factory=dict)
    turn_history: list[dict[str, Any]] = Field(default_factory=list)
    # ── Enrichment fields (populated by checkpoint-pack operations) ──
    overview: str = ""
    work_done: str = ""
    important_files: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)


class WorkflowTransitionRequest(BaseModel):
    """Structured transition request returned by model tool-calling."""

    target_stage_id: str
    rationale: str = ""
    required_missing_inputs: list[str] = Field(default_factory=list)


class WorkflowDelegationRequest(BaseModel):
    """Structured request to delegate a subtask to another workflow mode.

    Used by agent-level orchestration when a workflow turn decides another mode
    is better suited to produce an intermediate answer or artifact. Delegation
    is intentionally lightweight: target mode, input text, and an optional
    model override for that delegated run.
    """

    target_mode_id: str
    input_text: str
    model_alias: str = ""


# ---------------------------------------------------------------------------
# Unified FSM routing: fork fan-out (replaces workshop execution)
# ---------------------------------------------------------------------------


class ForkTarget(BaseModel):
    """One fork worker spec for a parallel fan-out transition.

    A fork transition fans out to N heterogeneous modes concurrently. Each
    target carries its own mode_id (determines system prompt + tool surface),
    an optional stage_id (defaults to the mode's start_stage_id), and a scoped
    query for that worker.
    """

    mode_id: str
    stage_id: str = ""
    query: str = ""


ForkChildStatus = Literal["ok", "timeout", "cancelled", "error", "no_tools"]


class ForkChildResult(BaseModel):
    """Result of one fork child run in a parallel fan-out transition."""

    target: ForkTarget
    status: ForkChildStatus = "ok"
    output_text: str = ""
    error_message: str = ""
    duration_ms: int = 0


# ---------------------------------------------------------------------------
# Shared conversation thread state
# ---------------------------------------------------------------------------


class ThreadMessage(BaseModel):
    """One message in a durable conversation thread.

    This shape mirrors OpenAI-style chat messages so agent.py can inject
    persisted turns into a new completion call without custom transformation.
    """

    role: Literal["system", "user", "assistant"]
    content: str
    provenance: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, object] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ConversationDigest(BaseModel):
    """Rolling per-thread summary used between raw turns and durable memory.

    The digest is stored on ThreadState so it inherits the same personal-DM,
    group-chat, and channel isolation keys as the raw conversation turns.
    """

    version: str = "1"
    updated_at: str = ""
    source_turn_count: int = 0
    last_message_at: str = ""
    summary: str = ""
    open_loops: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    commitments: list[str] = Field(default_factory=list)
    unresolved_references: list[str] = Field(default_factory=list)


ThreadArtifactType = Literal[
    "web_page",
    "search_results",
    "stats_snapshot",
    "workflow_summary",
    "markdown_document",
]


class HandoffArtifact(BaseModel):
    """Structured handoff from a mode into the next default-chat turn.

    Chat, action, and workflow modes emit one of these artifacts when they
    complete so the next direct chat turn can reference prior mode output
    without requiring the user to restate context.
    """

    source_mode: str
    artifact_type: ThreadArtifactType
    canonical_subject: dict[str, Any] = Field(default_factory=dict)
    summary_for_next_turn: str
    created_at: str
    expires_after_turns: int = 5
    artifact_data: dict[str, Any] = Field(default_factory=dict)


class ThreadState(BaseModel):
    """Shared base conversation state for one host conversation/user pair.

    This state is local-first with async blob synchronization. It intentionally
    stores only the latest turns and handoffs to keep prompt injection bounded.
    """

    workspace_id: str = "default"
    conversation_id: str
    user_id: str
    user_slug: str = "default"
    surface_type: ConversationSurface = "personal_dm"
    surface_id: str = ""
    turns: list[ThreadMessage] = Field(default_factory=list)
    recent_handoffs: list[HandoffArtifact] = Field(default_factory=list)
    conversation_digest: ConversationDigest = Field(default_factory=ConversationDigest)
    created_at: str
    updated_at: str

    @model_validator(mode="after")
    def _enforce_bounded_state(self) -> Self:
        """Enforce bounded state size after model initialization."""
        if len(self.turns) > _MAX_THREAD_TURNS:
            self.turns = self.turns[-_MAX_THREAD_TURNS:]
        if len(self.recent_handoffs) > _MAX_RECENT_HANDOFFS:
            self.recent_handoffs = self.recent_handoffs[-_MAX_RECENT_HANDOFFS:]
        return self


# ---------------------------------------------------------------------------
# Reflection / thinking memory
# ---------------------------------------------------------------------------


class ReflectiveTheme(BaseModel):
    """One recurring theme detected by a reflection loop."""

    label: str
    """Short human-readable theme label derived from top co-occurring content tokens."""
    evidence_count: int = 0
    """Number of distinct memory/turn records that mention this theme."""
    recency_score: float = 0.0
    """Decay-weighted recency of evidence in [0.0, 1.0]."""
    salience_avg: float = 0.0
    """Mean salience of contributing memory records."""
    aspiration_alignment: float = 0.0
    """Overlap score with goal/aspiration keywords from authored user context."""
    tension_score: float = 0.0
    """Problem-keyword proximity score; high means the theme is a live blocker or risk."""
    cooldown_expires_at: str = ""
    """ISO-8601 UTC timestamp until which this topic is suppressed."""
    source_memory_ids: list[int] = Field(default_factory=list)
    """IDs of memory records that contributed to this theme."""


class ProactiveTopic(BaseModel):
    """One candidate topic that a reflection loop considers surfacing proactively."""

    topic_id: str
    """Deterministic topic key used for cooldown state."""
    theme: ReflectiveTheme
    candidate_text: str = ""
    """Draft proactive message text; populated by synthesis when available."""
    confidence: float = 0.0
    """Combined score from recurrence, recency, salience, and alignment signals."""
    rationale: str = ""
    """One-sentence explanation of why this topic is surfacing now."""
    evidence_snippets: list[str] = Field(default_factory=list)
    """Bounded excerpts from contributing records."""
    topic_kind: Literal["assertion", "open_question", "reminder"] = "assertion"
    """Classification for assertions, open questions, and pending-action reminders."""
    source_memory_ids: list[int] = Field(default_factory=list)
    """IDs of memory records that contributed to this topic."""


class ThinkingMemorySnapshot(BaseModel):
    """Raw deterministic output of one reflection-memory scan."""

    generated_at: str
    memory_count: int = 0
    """Number of memory records included in this cycle's candidate pool."""
    turn_count: int = 0
    """Number of thread-turn messages included in this cycle's scan."""
    themes: list[ReflectiveTheme] = Field(default_factory=list)
    top_candidates: list[ProactiveTopic] = Field(default_factory=list)
    status: str = "ok"
    """'ok' | 'skipped' | 'error'"""


class ThinkingAnalysisLocal(BaseModel):
    """Deterministic local analysis artifact for a reflection pipeline."""

    generated_at: str
    aspiration_labels: list[str] = Field(default_factory=list)
    """Goal/aspiration keywords extracted from authored user context."""
    open_loops: list[str] = Field(default_factory=list)
    """Theme labels that appear unresolved."""
    top_candidates: list[ProactiveTopic] = Field(default_factory=list)
    scoring_metadata: dict[str, Any] = Field(default_factory=dict)


class ThinkingAnalysisSynth(BaseModel):
    """Synthesis artifact produced from top reflection candidates."""

    generated_at: str
    thinking_vibe: str = ""
    """Short summary of what the assistant is actively considering."""
    proactive_suggestions: list[ProactiveTopic] = Field(default_factory=list)
    """Model-enriched candidate topics with response text and rationale populated."""
    model_used: str = ""


DreamHypothesisStatus = Literal["tentative", "validated", "discarded", "promoted"]


class DreamHypothesis(BaseModel):
    """One bounded, testable hypothesis produced by an offline reflection loop."""

    hypothesis_id: str
    title: str = ""
    hypothesis: str
    why_now: str = ""
    suggested_next_move: str = ""
    counterevidence_to_seek: list[str] = Field(default_factory=list)
    evidence_snippets: list[str] = Field(default_factory=list)
    source_topic_ids: list[str] = Field(default_factory=list)
    source_memory_ids: list[int] = Field(default_factory=list)
    workspace_id: str = "default"
    user_slug: str = "default"
    memory_scope: MemoryScope = "personal"
    surface_id: str = ""
    confidence: float = 0.0
    status: DreamHypothesisStatus = "tentative"
    created_at: str = ""
    expires_at: str = ""


class DreamHypothesisArtifact(BaseModel):
    """Persisted output of one bounded hypothesis synthesis cycle."""

    generated_at: str
    workspace_id: str = "default"
    user_slug: str = "default"
    memory_scope: MemoryScope = "personal"
    surface_id: str = ""
    thinking_vibe: str = ""
    hypotheses: list[DreamHypothesis] = Field(default_factory=list)
    model_used: str = ""


class DreamVerificationResult(BaseModel):
    """Verification result for one hypothesis before promotion."""

    hypothesis_id: str
    status: DreamHypothesisStatus = "tentative"
    support_score: float = 0.0
    conflict_score: float = 0.0
    verification_notes: list[str] = Field(default_factory=list)
    source_topic_ids: list[str] = Field(default_factory=list)


class DreamVerificationArtifact(BaseModel):
    """Persisted verification output for one hypothesis cycle."""

    generated_at: str
    workspace_id: str = "default"
    user_slug: str = "default"
    memory_scope: MemoryScope = "personal"
    surface_id: str = ""
    results: list[DreamVerificationResult] = Field(default_factory=list)


class StrategyMemoryEntry(BaseModel):
    """One promoted strategy distilled from validated hypotheses."""

    strategy_id: str
    title: str = ""
    summary: str
    applies_when: str = ""
    recommended_action: str = ""
    confidence: float = 0.0
    source_hypothesis_ids: list[str] = Field(default_factory=list)
    source_topic_ids: list[str] = Field(default_factory=list)
    workspace_id: str = "default"
    user_slug: str = "default"
    memory_scope: MemoryScope = "personal"
    surface_id: str = ""
    promoted_at: str = ""
    expires_at: str = ""


class StrategyMemoryArtifact(BaseModel):
    """Persisted promoted strategy memory kept separate from raw episodic memory."""

    generated_at: str
    workspace_id: str = "default"
    user_slug: str = "default"
    memory_scope: MemoryScope = "personal"
    surface_id: str = ""
    strategy_vibe: str = ""
    strategies: list[StrategyMemoryEntry] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Passive memory insights
# ---------------------------------------------------------------------------


PassiveIntelligenceKind = Literal["theme", "entity", "preference", "open_loop"]


class PassiveIntelligenceItem(BaseModel):
    """One medium-granularity passive insight derived from scoped memories.

    This layer sits between raw episodic memories and heavier host-owned
    synthesis artifacts. Items are deterministic and cheap to compute so they
    can be refreshed frequently without model calls.
    """

    insight_id: str
    label: str
    kind: PassiveIntelligenceKind = "theme"
    summary: str = ""
    evidence_count: int = 0
    recency_score: float = 0.0
    salience_score: float = 0.0
    last_seen_at: str = ""
    source_memory_ids: list[int] = Field(default_factory=list)
    evidence_snippets: list[str] = Field(default_factory=list)
    lexical_tokens: list[str] = Field(default_factory=list)


class PassiveIntelligenceSnapshot(BaseModel):
    """Deterministic passive-memory snapshot for one privacy-scoped surface.

    Default chat can inject this artifact directly for the active user / surface
    without promoting private context into workspace-global live state.
    """

    generated_at: str
    workspace_id: str = "default"
    user_slug: str = "default"
    memory_scope: MemoryScope = "personal"
    surface_id: str = ""
    source_memory_count: int = 0
    item_count: int = 0
    summary_markdown: str = ""
    insights: list[PassiveIntelligenceItem] = Field(default_factory=list)
    status: str = "ok"
    """'ok' | 'skipped' (feature disabled) | 'error'"""


# ---------------------------------------------------------------------------
# Internal event bus contracts
#
# All events published to the in-process bus (and in future to an external
# broker) must subclass BusEvent.  The event_type literal acts as the routing
# key; consumers declare which types they handle via the BusConsumer.handles
# frozenset.  The idempotency_key lets consumers suppress duplicate side effects
# under at-least-once delivery semantics.
#
# workspace_id scopes events to one logical workspace. Hosts map their own
# tenant/account/surface identifiers into this field before publishing.
# ---------------------------------------------------------------------------

BusEventType = Literal[
    "anomaly_detected",
    "state_refreshed",
    "digest_due",
    "reminder_due",
    "memory_salience",
    "context_used",
    "workflow_runtime_used",
    "token_used",
    "tool_telemetry",
    "market_intel_requested",
    "market_analysis_refreshed",
    "intel_collection_requested",
    "intel_analysis_refreshed",
    "thinking_requested",
    "thinking_refreshed",
]


def _new_event_id() -> str:
    """Generate a new UUID4 string used as the default event_id."""
    return str(uuid.uuid4())


def _now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(UTC).isoformat()


class BusEvent(BaseModel):
    """Base envelope for all events on the internal bus.

    Every concrete event subclasses this and sets a fixed event_type literal.
    Consumers match events by comparing event.event_type against their handles
    frozenset (defined by the host event bus consumer protocol).

    Idempotency:
        The idempotency_key field is used by consumers to deduplicate redeliveries
        under at-least-once delivery semantics.  Publishers set it to a
        deterministic string (e.g. alert fingerprint, state hash, slot_key).
        If left empty the consumer treats the event as always-novel.

    Workspace scoping:
        workspace_id is "default" until multi-workspace Phase 4 lands.
        Consumers that scope side effects per workspace read this field.
    """

    event_id: str = Field(default_factory=_new_event_id)
    event_type: BusEventType
    emitted_at: str = Field(default_factory=_now_iso)
    idempotency_key: str = ""
    workspace_id: str = "default"


class AnomalyDetectedEvent(BusEvent):
    """Emitted by a host scheduler when anomaly checks produce alerts.

    Consumed by host alert handlers which:
    - Reads alert dedupe state from blob.
    - Skips the event if idempotency_key already appears in active_alerts.
    - Dispatches to the host's configured alert target if novel.
    - Writes updated dispatch state back to blob.

    The idempotency_key is the sha256 fingerprint of severity+source+message,
    identical to the host's alert fingerprinting function.
    """

    event_type: BusEventType = "anomaly_detected"
    alert: AnomalyAlert


class StateRefreshedEvent(BusEvent):
    """Emitted after each successful UnifiedState compilation.

    Consumed by host state handlers which persist the state to object storage.
    Hosts can also update their in-memory state before publishing so immediate
    reads never hit a stale cache.

    The idempotency_key is the compiled_at timestamp of the state.  Two events
    with the same compiled_at (possible if heartbeat tick is retried) will
    result in an idempotent re-write (safe, same payload).
    """

    event_type: BusEventType = "state_refreshed"
    state: UnifiedState


class DigestDueEvent(BusEvent):
    """Emitted by a host scheduler when a scheduled digest slot is reached.

    Consumed by host proactive digest handlers which:
    - Reads current UnifiedState from the in-memory cache (get_current_state()).
    - Iterates registered ProactiveRecipients.
    - Sends a formatted digest message per recipient via the host adapter.
    - Updates last_digest_slot on each recipient to prevent repeat sends.

    The idempotency_key is the slot_key string (e.g. "2026-04-28T09:00-0400"),
    so duplicate DigestDueEvents for the same slot are safely dropped.
    """

    event_type: BusEventType = "digest_due"
    # ISO-8601 local timestamp string used to format the digest header.
    local_timestamp: str
    # Slot key used for idempotency: YYYY-MM-DDTHH:MM<tz> e.g. 2026-04-28T09:00-0400
    slot_key: str


class ReminderDueEvent(BusEvent):
    """Emitted by heartbeat when one or more pending reminders become due."""

    event_type: BusEventType = "reminder_due"
    reminder_ids: list[str] = Field(default_factory=list)


class MemorySalienceEvent(BusEvent):
    """Emitted by memory_store.record_memory() when a memory exceeds the salience threshold.

    Currently has no registered consumer (reserved for future use).  A future
    consumer could trigger a proactive nudge to the user when a critical
    operational insight is recorded (e.g. during a chase or coach session).

    The idempotency_key is the sqlite memory_id as a string, since each
    write produces a unique row.
    """

    event_type: BusEventType = "memory_salience"
    memory_id: int
    user_slug: str
    memory_scope: MemoryScope = "personal"
    surface_id: str = ""
    operation_id: str
    salience_score: float


class ContextUsageEvent(BusEvent):
    """Telemetry envelope for context packet selection and prompt injection."""

    event_type: BusEventType = "context_used"
    conversation_id: str = ""
    user_slug: str = "default"
    surface_type: ConversationSurface = "personal_dm"
    surface_id: str = ""
    memory_scope: MemoryScope = "personal"
    mode_id: str = "default"
    query_preview: str = ""
    requested_model_id: str = ""
    response_model_id: str = ""
    prompt_strategy: ContextPromptStrategy = "broker_packet"
    total_available_evidence: int = 0
    selected_evidence_ids: list[str] = Field(default_factory=list)
    used_evidence_ids: list[str] = Field(default_factory=list)
    selected_sources: list[str] = Field(default_factory=list)
    context_char_count: int = 0
    ranking_metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowRuntimeUsageEvent(BusEvent):
    """Telemetry envelope for workflow runtime-state prefetch and fallback usage."""

    event_type: BusEventType = "workflow_runtime_used"
    conversation_id: str = ""
    user_slug: str = "default"
    surface_type: ConversationSurface = "personal_dm"
    surface_id: str = ""
    memory_scope: MemoryScope = "personal"
    mode_id: str = "default"
    stage_id: str = ""
    query_preview: str = ""
    usage_phase: Literal["prefetch", "fallback"] = "prefetch"
    requested_views: list[str] = Field(default_factory=list)
    returned_views: list[str] = Field(default_factory=list)
    blocked_views: list[str] = Field(default_factory=list)
    stop_reason: str = ""


class TokenUsageEvent(BusEvent):
    """Telemetry envelope for one LLM provider call's token usage and cost.

    Consumed by the host's `TokenUsageTelemetryHandler` which appends every
    payload to the durable `token-usage.jsonl` blob under the host's data
    directory. The handler uses a timestamp + nanosecond suffix as the
    idempotency key so retries and stream-final events always append without
    collapsing.
    """

    event_type: BusEventType = "token_used"
    conversation_id: str = ""
    user_slug: str = "default"
    surface_type: ConversationSurface = "personal_dm"
    surface_id: str = ""
    memory_scope: MemoryScope = "personal"
    mode_id: str = "default"
    stage_id: str = ""
    operation_id: str = ""
    call_site: str = ""
    provider: str = ""
    model_id: str = ""
    request_id: str = ""
    kind: Literal[
        "chat_completion",
        "embedding",
        "image_generation",
        "stream_chunk",
        "time_billed",
    ] = "chat_completion"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: float = 0.0
    time_cost_usd: float = 0.0
    latency_ms: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)
    extra_context_appended: bool = False


class ToolExecutionTelemetry(BusEvent):
    """Telemetry envelope for tracking tool and operation execution via EventBus."""

    event_type: BusEventType = "tool_telemetry"
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    target_id: str  # The `tool_id` or `operation_id`
    target_type: Literal["tool", "operation"]
    status: Literal["success", "error", "timeout"]
    duration_ms: int
    error_message: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class IntelCollectionRequestedEvent(BusEvent):
    """Emitted when a host requests a fresh intelligence collection."""

    event_type: BusEventType = "intel_collection_requested"
    collection_scope: str = "market"
    trigger_source: str
    directional_query: str = ""
    trigger_metadata: dict[str, Any] = Field(default_factory=dict)


class IntelAnalysisRefreshedEvent(BusEvent):
    """Emitted after an intelligence analysis pipeline successfully completes."""

    event_type: BusEventType = "intel_analysis_refreshed"
    collection_scope: str = "market"
    vibe_markdown: str = ""


class ThinkingRequestedEvent(BusEvent):
    """Emitted when a host requests a reflection cycle."""

    event_type: BusEventType = "thinking_requested"
    trigger_source: str = "scheduled"
    trigger_metadata: dict[str, Any] = Field(default_factory=dict)


class ThinkingRefreshedEvent(BusEvent):
    """Emitted after a successful reflection synthesis cycle."""

    event_type: BusEventType = "thinking_refreshed"
    thinking_vibe: str = ""
    candidate_count: int = 0
