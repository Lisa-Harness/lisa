"""Kernel-owned runtime configuration.

This module intentionally contains only settings consumed by `kernel/**`.
Public packs and extensions use `sdk.config`, while the root app owns its
broader host and product settings in `app.config`.
"""

from __future__ import annotations

from typing import Any

from pydantic import field_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

_REMOVED_MODEL_ENV_ALIASES = frozenset(
    {
        "deep_model",
        "DEEP_MODEL",
        "azure_openai_deployment_deep",
        "AZURE_OPENAI_DEPLOYMENT_DEEP",
        "self_model_enabled",
        "SELF_MODEL_ENABLED",
        "self_model_max_entries",
        "SELF_MODEL_MAX_ENTRIES",
        "self_model_max_markdown_chars",
        "SELF_MODEL_MAX_MARKDOWN_CHARS",
        "self_model_include_help_hints",
        "SELF_MODEL_INCLUDE_HELP_HINTS",
        "self_learning_enabled",
        "SELF_LEARNING_ENABLED",
    }
)


def _normalized_settings_keys(settings_cls: type[BaseSettings]) -> set[str]:
    """Return normalized field and alias keys accepted by the settings model."""

    keys: set[str] = set()
    for field_name, field_info in settings_cls.model_fields.items():
        keys.add(field_name.lower())
        alias = field_info.alias
        if isinstance(alias, str) and alias:
            keys.add(alias.lower())
    return keys


class _KnownSettingFilteringSource(PydanticBaseSettingsSource):
    """Drop unrelated env vars before Pydantic validates kernel settings."""

    def __init__(self, source: PydanticBaseSettingsSource) -> None:
        super().__init__(source.settings_cls)
        self._source = source

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return self._source.get_field_value(field, field_name)

    def __call__(self) -> dict[str, Any]:
        data = dict(self._source())
        for alias in _REMOVED_MODEL_ENV_ALIASES:
            data.pop(alias, None)
        allowed_keys = _normalized_settings_keys(self.settings_cls)
        return {key: value for key, value in data.items() if key.lower() in allowed_keys}


class Settings(BaseSettings):
    """Settings needed by `kernel/**` only."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        del settings_cls
        return (
            init_settings,
            _KnownSettingFilteringSource(env_settings),
            _KnownSettingFilteringSource(dotenv_settings),
            file_secret_settings,
        )

    sdk_content_roots: str = ""
    workspace_id_fallback: str = "default"

    # ── Declarative operations ──────────────────────────────────────────
    declarative_completion_max_tokens: int = 1500
    declarative_completion_temperature: float = 0.2

    # ── Turn runner ─────────────────────────────────────────────────────
    # Maximum model-completion iterations per turn. Shared across all
    # handoffs within a single turn (the counter is not reset on
    # ``transition_stage``), so chained producer→consumer pipelines
    # (research→write) consume the same budget. Override via
    # ``MAX_TOOL_ITERATIONS``.
    max_tool_iterations: int = 15
    # Per-child iteration cap for fork fan-out. Fork children run in
    # parallel and get a smaller slice than a full sequential turn.
    # Override via ``FORK_CHILD_MAX_TOOL_ITERATIONS``.
    fork_child_max_tool_iterations: int = 5
    # Per-step iteration sub-budget. Each workflow step (between two
    # ``transition_stage`` handoffs) gets its own iteration budget so a
    # chained producer→consumer pipeline does not starve its final step
    # on the shared ``max_tool_iterations`` budget. When a step exhausts
    # its per-step cap but the global cap has not fired, the kernel
    # synthesizes a handoff to the next remaining step instead of the
    # generic empty-content final completion. The global cap remains the
    # hard ceiling: a 6-step chain with per_step=8 cannot exceed 15 total
    # iterations — the per-step cap only redistributes *when* the budget
    # runs out. Override via ``PER_STEP_MAX_ITERATIONS``.
    per_step_max_iterations: int = 8
    # Per-step tool-call sub-budget. Secondary guard against a model that
    # burns many tool calls within one step without converging. Same reset
    # and handoff-synth behavior as ``per_step_max_iterations``.
    # Override via ``PER_STEP_MAX_TOOL_CALLS``.
    per_step_max_tool_calls: int = 12

    # ── Context broker ──────────────────────────────────────────────────
    heartbeat_interval_seconds: int = 60
    repo_context_stale_after_hours: int = 6
    reflection_cadence_minutes: int = 30
    dream_hypothesis_ttl_hours: int = 24
    context_broker_prefer_recent: bool = True
    context_broker_min_evidence_score: float = 0.15
    context_broker_max_selected_items: int = 8
    context_broker_max_markdown_chars: int = 4000

    # ── Context budget ──────────────────────────────────────────────────
    context_budget_enabled: bool = True
    context_budget_default_total_tokens: int = 6000
    context_budget_deep_total_tokens: int = 12000
    context_budget_core_max_tokens: int = 2000
    context_budget_retrieval_max_tokens: int = 1500
    context_budget_memory_max_tokens: int = 1200
    context_budget_reflection_max_tokens: int = 1000
    context_budget_history_max_tokens: int = 1500
    context_budget_operational_max_tokens: int = 800
    context_budget_token_char_ratio: int = 4

    # ── Conversation digest ─────────────────────────────────────────────
    conversation_digest_enabled: bool = True
    conversation_digest_model: str = ""
    conversation_digest_max_prompt_chars: int = 12000
    conversation_digest_max_summary_chars: int = 800
    conversation_digest_source_turns: int = 8
    conversation_digest_recent_turns: int = 4

    # ── Memory ──────────────────────────────────────────────────────────
    memory_enabled: bool = True
    memory_sqlite_path: str = ""
    memory_sync_enabled: bool = False
    memory_eager_sync_on_write: bool = False
    memory_sync_batch_size: int = 50
    memory_final_sync_timeout_seconds: float = 5.0
    memory_flush_interval_seconds: int = 60
    memory_snapshot_interval_seconds: int = 300
    memory_salience_event_threshold: float = 0.6
    memory_embedding_enabled: bool = False
    memory_embedding_provider: str = "azure_openai"
    memory_embedding_model: str = ""
    memory_embedding_dimensions: int = 1536
    memory_ann_enabled: bool = False
    memory_decay_enabled: bool = True
    memory_decay_half_life_days: float = 14.0
    memory_decay_floor: float = 0.05
    memory_recall_candidate_pool_size: int = 50
    memory_recall_recent_limit: int = 5
    memory_recall_lexical_limit: int = 20
    memory_recall_semantic_limit: int = 10
    memory_recall_final_limit: int = 5
    memory_recall_auto_deep_threshold: float = 0.4
    memory_recall_cheap_model: str = ""
    memory_promotion_threshold: int = 3
    memory_consolidation_enabled: bool = True
    memory_eviction_enabled: bool = True
    memory_eviction_min_age_days: int = 30
    memory_eviction_default_ttl_days: int = 90
    memory_eviction_min_salience: float = 0.1
    memory_eviction_min_access_count: int = 0
    memory_eviction_max_active_per_scope: int = 500
    memory_eviction_max_kind_per_user: dict[str, int] = {}

    # ── Memory recall (pre-turn injection) ─────────────────────────────
    memory_recall_enabled: bool = True
    memory_recall_pre_turn_limit: int = 3
    memory_recall_min_query_chars: int = 3

    # ── Memory auto-capture (post-turn learning) ───────────────────────
    memory_auto_capture_enabled: bool = True
    memory_auto_capture_min_salience: float = 0.5

    # ── Passive intelligence ────────────────────────────────────────────
    passive_intelligence_enabled: bool = True
    passive_intelligence_default_chat_enabled: bool = True
    passive_intelligence_min_recurrence: int = 2
    passive_intelligence_min_salience: float = 0.3
    passive_intelligence_min_score: float = 0.2
    passive_intelligence_evidence_limit: int = 3
    passive_intelligence_max_items: int = 5
    passive_intelligence_window_turns: int = 12
    passive_intelligence_window_memories: int = 20
    passive_intelligence_stale_after_minutes: int = 30
    passive_intelligence_default_chat_min_overlap: float = 0.25

    @staticmethod
    def _strip_inline_comment(value: object) -> object:
        if not isinstance(value, str):
            return value
        return value.split("#", 1)[0].strip()

    @field_validator(
        "declarative_completion_max_tokens",
        "max_tool_iterations",
        "fork_child_max_tool_iterations",
        "heartbeat_interval_seconds",
        "repo_context_stale_after_hours",
        "reflection_cadence_minutes",
        "dream_hypothesis_ttl_hours",
        "context_budget_default_total_tokens",
        "context_budget_deep_total_tokens",
        "context_budget_core_max_tokens",
        "context_budget_retrieval_max_tokens",
        "context_budget_memory_max_tokens",
        "context_budget_reflection_max_tokens",
        "context_budget_history_max_tokens",
        "context_budget_operational_max_tokens",
        "context_budget_token_char_ratio",
        "conversation_digest_max_prompt_chars",
        "conversation_digest_max_summary_chars",
        "conversation_digest_source_turns",
        "conversation_digest_recent_turns",
        "memory_sync_batch_size",
        "memory_final_sync_timeout_seconds",
        "memory_flush_interval_seconds",
        "memory_snapshot_interval_seconds",
        "memory_embedding_dimensions",
        "memory_recall_candidate_pool_size",
        "memory_recall_recent_limit",
        "memory_recall_lexical_limit",
        "memory_recall_semantic_limit",
        "memory_recall_final_limit",
        "memory_promotion_threshold",
        "memory_eviction_min_age_days",
        "memory_eviction_default_ttl_days",
        "memory_eviction_min_access_count",
        "memory_eviction_max_active_per_scope",
        "passive_intelligence_min_recurrence",
        "passive_intelligence_evidence_limit",
        "passive_intelligence_max_items",
        "passive_intelligence_window_turns",
        "passive_intelligence_window_memories",
        "passive_intelligence_stale_after_minutes",
        mode="before",
    )
    @classmethod
    def _coerce_numeric_with_inline_comments(cls, value: object) -> object:
        stripped = cls._strip_inline_comment(value)
        if stripped == "":
            return None
        return stripped


settings = Settings()


def reload_settings() -> Settings:
    """Reinitialize the module-level settings singleton from the environment."""

    new_settings = Settings()
    current_settings = globals().get("settings")
    if isinstance(current_settings, Settings):
        for field_name in new_settings.model_dump():
            setattr(current_settings, field_name, getattr(new_settings, field_name))
        return current_settings

    globals()["settings"] = new_settings
    return new_settings
