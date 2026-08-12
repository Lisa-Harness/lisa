"""Kernel-owned runtime configuration.

This module intentionally contains only settings consumed by `kernel/**`.
Public packs and extensions use `sdk.config`, while the root app owns its
broader host and product settings in `app.config`.
"""

from __future__ import annotations

from typing import Any

from pydantic import field_validator, model_validator
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

    # ── Identity defaults ───────────────────────────────────────────────
    identity_workspace_id: str = "lisa-host"
    identity_user_id: str = "lisa-user"
    identity_user_slug: str = "lisa-user"
    identity_user_name: str = "Lisa User"
    identity_principal_key: str = "lisa-user"
    identity_conversation_id: str = "lisa-host-console"
    identity_surface_type: str = "personal_dm"
    identity_surface_id: str = "lisa-host-console"
    identity_memory_scope: str = "personal"

    # ── Declarative operations ──────────────────────────────────────────
    declarative_completion_max_tokens: int = 1500
    declarative_completion_temperature: float = 0.2

    # ── Turn runner ─────────────────────────────────────────────────────
    # Maximum model-completion iterations per turn. Shared across all
    # handoffs within a single turn (the counter is not reset on
    # ``transition_stage``), so chained producer→consumer pipelines
    # (research→write) consume the same budget. The default is the
    # per-turn runaway governor value (see docs/TOKEN_BUDGET.md); lower it
    # only for sandboxed test runs. Override via
    # ``MAX_TOOL_ITERATIONS``.
    max_tool_iterations: int = 300
    # Per-child iteration cap for fork fan-out. Fork children run in
    # parallel and get a smaller slice than a full sequential turn, but
    # still need enough room to inspect → act → verify. A coding child
    # (inspect → isolate → patch → check → publish) cannot fit a
    # non-trivial change in fewer than ~30 iterations; 50 leaves headroom
    # for the publish step. Override via
    # ``FORK_CHILD_MAX_TOOL_ITERATIONS``.
    fork_child_max_tool_iterations: int = 50
    # Per-child iteration cap for delegate sub-turns. Delegate children
    # run sequentially (one at a time, parent waits) and get the same
    # inspect → act → verify room as fork children. Defaults to the fork
    # child cap so a delegate child has identical headroom unless a host
    # explicitly tunes it. Override via
    # ``DELEGATE_CHILD_MAX_TOOL_ITERATIONS``.
    delegate_child_max_tool_iterations: int = 50
    # Maximum recursion depth for delegate sub-turns. A delegate child
    # may itself delegate, forming an agent tree of arbitrary depth. This
    # cap bounds the tree so a misbehaving mode cannot recurse
    # indefinitely. The root parent turn is depth 0; a direct delegate
    # child is depth 1. Override via ``DELEGATE_MAX_DEPTH``.
    delegate_max_depth: int = 4
    # Per-step iteration sub-budget. Each workflow step (between two
    # ``transition_stage`` handoffs) gets its own iteration budget so a
    # chained producer→consumer pipeline does not starve its final step
    # on the shared ``max_tool_iterations`` budget. When a step exhausts
    # its per-step cap but the global cap has not fired, the kernel
    # synthesizes a handoff to the next remaining step instead of the
    # generic empty-content final completion. The global cap remains the
    # hard ceiling: a 6-step chain with per_step=75 cannot exceed 300
    # total iterations — the per-step cap only redistributes *when* the
    # budget runs out. Override via ``PER_STEP_MAX_ITERATIONS``.
    per_step_max_iterations: int = 75
    # Per-step tool-call sub-budget. Secondary guard against a model that
    # burns many tool calls within one step without converging. Same reset
    # and handoff-synth behavior as ``per_step_max_iterations``.
    # Override via ``PER_STEP_MAX_TOOL_CALLS``.
    per_step_max_tool_calls: int = 150

    # ── Turn-runner guard caps ──────────────────────────────────────────
    # These bound the various correction-nudge and loop-backstop guards
    # in ``kernel.turn_runner``. They were previously module-level
    # constants; they now live here so hosts can tune them via env vars
    # without editing kernel source. Each defaults to its historical
    # value (see the git-blame comments in ``turn_runner.py`` for the
    # incidents that motivated each cap).
    # Forced-action enforcer (Phase 5): when a routing hint encodes a
    # deterministic routing decision — a ``fork:`` fan-out or any
    # single-handoff target (``plan``, ``search``, ``remember`` …) — and
    # the model refuses to call the expected tool (``fork`` /
    # ``transition_stage``), instead burning iterations on read-only
    # tools or returning empty completions (the 2026-07-27 fork incident;
    # the 2026-07-30 400k-token pre-plan burn; the 2026-08-07
    # ``plan → remember → search`` stall), the kernel synthesizes the
    # call itself after this many non-compliant tool dispatches. This
    # bounds the pre-handoff token burn to roughly
    # ``cap x per-iteration-tokens`` instead of the full
    # ``max_tool_iterations`` budget. Override via
    # ``FORCED_ACTION_NON_COMPLIANT_CAP``.
    forced_action_non_compliant_cap: int = 2
    # Stuck-session guard: after this many turns in the same stage
    # without a transition, the loop injects a nudge. Override via
    # ``STAGE_NUDGE_THRESHOLD``.
    stage_nudge_threshold: int = 4
    # After this many turns in the same stage, the loop forces a final
    # no-tools completion to stop burning tokens. Override via
    # ``STAGE_NUDGE_CAP``.
    stage_nudge_cap: int = 6
    # Iterations *before* the stuck-in-stage cap at which the parent-turn
    # write-nudge fires for a mode with unmet ``action_tool_ids``. ``1``
    # means the nudge fires one iteration before the cap would pause the
    # turn, giving the model a final chance to call a write tool (e.g.
    # ``markdown_forge_document``) rather than ending blocked as
    # ``no_tools``. ``0`` disables the pre-cap nudge. Override via
    # ``WRITE_DEADLINE_MARGIN``.
    write_deadline_margin: int = 1
    # Write-rescue last-resort mode: when a write-contract mode is one
    # iteration away from the stuck-in-stage cap with no verified write,
    # the lifecycle resolver forces a transition to this mode so the
    # turn ends productively instead of pausing on a hallucinated-success
    # block. Defaults to ``expounder`` (a terminal capture persona whose
    # only write is ``markdown_forge_document``). Set to ``""`` to disable
    # the write-rescue forced transition. Override via
    # ``WRITE_RESCUE_MODE``.
    write_rescue_mode: str = "expounder"
    # Hard backstop against routing-hint loops: caps the total mode
    # handoffs per turn and forces a final completion. Loosened from the
    # historical 6 to accommodate deep producer→consumer pipelines (the
    # 2026-08-02 fork fan-out work hit 6 handoffs on a routine coding
    # turn). Override via ``MAX_HANDOFFS_PER_TURN``.
    max_handoffs_per_turn: int = 50
    # Consecutive identical routing hints before the duplicate-hint dedup
    # suppresses injection. Override via ``DUPLICATE_HINT_LIMIT``.
    duplicate_hint_limit: int = 2
    # Minimum hint-target history length that forms a detectable 2-cycle
    # (A, B, A). Override via ``HINT_CYCLE_MIN_LEN``.
    hint_cycle_min_len: int = 3
    # Consecutive routing-hint overrides (model hands off to a target
    # that differs from the hint's suggested target) before the hint
    # layer stops injecting further hints for the rest of the turn. The
    # historical 2 silenced hints far too early on multi-intent queries
    # where the model legitimately reorders the suggested sequence; 50
    # let the model redirect but bounded nothing — the override path
    # suppressed the hint yet kept cycling until the 300-iteration
    # ceiling (the 2026-08-05 markdown burn: 16 round-trips at ~48k
    # prompt tokens each after hint suppression). Override via
    # ``HINT_OVERRIDE_LIMIT``.
    hint_override_limit: int = 5
    # Budget briefing: inject a compact, per-iteration budget status
    # block into the system prompt so the model can self-scope its
    # work within the remaining iteration / tool-call / per-target
    # envelope. Only discrete, actionable budgets are surfaced
    # (iterations, tool calls, per-target call limits) — never
    # continuous budgets (tokens, wall time) which the model cannot
    # reliably convert into action and which risk anchoring the model
    # to "use" the budget it sees. The full contract is injected at
    # turn start; each subsequent iteration replaces it with a
    # super-brief delta. Override via ``BUDGET_BRIEFING_ENABLED``.
    budget_briefing_enabled: bool = True

    # ── Context budget ──────────────────────────────────────────────────
    context_budget_enabled: bool = True
    context_budget_core_max_tokens: int = 2000
    context_budget_retrieval_max_tokens: int = 1500
    context_budget_memory_max_tokens: int = 1200
    context_budget_reflection_max_tokens: int = 1000
    context_budget_history_max_tokens: int = 1500
    context_budget_operational_max_tokens: int = 800
    context_budget_token_char_ratio: int = 4

    # ── Context mapper scoring ─────────────────────────────────────────
    # Per-provenance authority multiplier used when scoring context
    # candidates. Keys must exactly match the ``ContextProvenance``
    # literals in ``kernel/context/models.py``. Override via
    # ``CONTEXT_MAPPER_PROVENANCE_AUTHORITY`` as a JSON object.
    context_mapper_provenance_authority: dict[str, float] = {
        "authoritative": 1.0,
        "system_observed": 0.98,
        "user_asserted": 0.75,
        "recalled_memory": 0.70,
        "inferred": 0.55,
        "diagnostic": 0.45,
        "speculative": 0.30,
    }
    # Per-conflict-risk penalty applied to the conflict component. Keys
    # must exactly match the ``ContextConflictRisk`` literals in
    # ``kernel/context/models.py``. Override via
    # ``CONTEXT_MAPPER_CONFLICT_PENALTY`` as a JSON object.
    context_mapper_conflict_penalty: dict[str, float] = {
        "low": 0.0,
        "medium": 0.5,
        "high": 1.0,
    }

    # ── Conversation digest ─────────────────────────────────────────────
    conversation_digest_enabled: bool = True
    conversation_digest_model: str = ""
    conversation_digest_max_prompt_chars: int = 12000
    conversation_digest_max_summary_chars: int = 800
    conversation_digest_source_turns: int = 8
    conversation_digest_recent_turns: int = 4

    # ── Memory recall (pre-turn injection) ─────────────────────────────
    memory_recall_enabled: bool = True
    memory_recall_pre_turn_limit: int = 3
    memory_recall_min_query_chars: int = 3

    # ── Memory auto-capture (post-turn learning) ───────────────────────
    memory_auto_capture_enabled: bool = True
    memory_auto_capture_min_salience: float = 0.5
    # Salience → tier cut points for auto-captured learnings (see
    # ``host.turn_hooks.classify_memory_tier``). Below ``daily_threshold``
    # is ``daily``; between the two is ``weekly``; at/above
    # ``weekly_threshold`` is ``long_term``.
    memory_auto_capture_daily_threshold: float = 0.65
    memory_auto_capture_weekly_threshold: float = 0.8

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

    @model_validator(mode="after")
    def _validate_mapper_scoring_keys(self) -> Settings:
        """Ensure context-mapper scoring dicts cover exactly the Literal keys."""
        required_provenance = {
            "authoritative",
            "inferred",
            "speculative",
            "user_asserted",
            "system_observed",
            "recalled_memory",
            "diagnostic",
        }
        provenance_keys = set(self.context_mapper_provenance_authority)
        missing_provenance = required_provenance - provenance_keys
        extra_provenance = provenance_keys - required_provenance
        if missing_provenance or extra_provenance:
            details: list[str] = []
            if missing_provenance:
                details.append("missing: " + ", ".join(sorted(missing_provenance)))
            if extra_provenance:
                details.append("extra: " + ", ".join(sorted(extra_provenance)))
            raise ValueError(
                "context_mapper_provenance_authority keys must match the "
                "ContextProvenance literals exactly (" + "; ".join(details) + ")"
            )
        required_conflict = {"low", "medium", "high"}
        conflict_keys = set(self.context_mapper_conflict_penalty)
        missing_conflict = required_conflict - conflict_keys
        extra_conflict = conflict_keys - required_conflict
        if missing_conflict or extra_conflict:
            details = []
            if missing_conflict:
                details.append("missing: " + ", ".join(sorted(missing_conflict)))
            if extra_conflict:
                details.append("extra: " + ", ".join(sorted(extra_conflict)))
            raise ValueError(
                "context_mapper_conflict_penalty keys must match the "
                "ContextConflictRisk literals exactly (" + "; ".join(details) + ")"
            )
        return self

    @staticmethod
    def _strip_inline_comment(value: object) -> object:
        if not isinstance(value, str):
            return value
        return value.split("#", 1)[0].strip()

    @field_validator(
        "declarative_completion_max_tokens",
        "max_tool_iterations",
        "fork_child_max_tool_iterations",
        "delegate_child_max_tool_iterations",
        "delegate_max_depth",
        "forced_action_non_compliant_cap",
        "stage_nudge_threshold",
        "stage_nudge_cap",
        "write_deadline_margin",
        "max_handoffs_per_turn",
        "duplicate_hint_limit",
        "hint_cycle_min_len",
        "hint_override_limit",
        "budget_briefing_enabled",
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
