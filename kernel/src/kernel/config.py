"""Kernel-owned runtime configuration.

This module intentionally contains only settings consumed by `kernel/**`.
Public packs and extensions use `sdk.config`; hosts own their wiring and
product settings (surface identity, per-lane context budgets) in
`runtime.config`.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

# Shell (global) env vars for ``kernel.config.Settings`` accept an optional
# ``LISA_`` prefix. ``LISA_STAGE_NUDGE_CAP`` and ``STAGE_NUDGE_CAP`` are both
# honored; ``LISA_`` is never *required*. Settings load from the live
# environment only — ``.env`` file handling is owned by the host/CLI
# transport layer, which exports values into ``os.environ`` before config
# resolution. See ``docs/python-BCP.md`` (env-var conventions) for rationale.
_LISA_ENV_PREFIX = "LISA_"


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
        allowed_keys = _normalized_settings_keys(self.settings_cls)
        return {key: value for key, value in data.items() if key.lower() in allowed_keys}


class Settings(BaseSettings):
    """Settings needed by `kernel/**` only."""

    # Live environment only: no ``env_file`` here. ``.env`` loading is a
    # host/CLI responsibility (the transport exports values into
    # ``os.environ`` before config resolution — see ``sdk.layout``), so a
    # CWD-relative second reader here would disagree with the host's
    # data-dir-anchored search order.
    model_config = SettingsConfigDict(populate_by_name=True)

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
        # Shell env: honor both bare (``STAGE_NUDGE_CAP``) and
        # ``LISA_``-prefixed (``LISA_STAGE_NUDGE_CAP``) names. The bare
        # source wins so an explicit unprefixed value is never shadowed by
        # a stray prefixed one. ``dotenv_settings`` is deliberately unused:
        # the kernel reads no ``.env`` file of its own (see model_config).
        del dotenv_settings
        _lisa_env = EnvSettingsSource(cls, env_prefix=_LISA_ENV_PREFIX)
        return (
            init_settings,
            _KnownSettingFilteringSource(env_settings),
            _KnownSettingFilteringSource(_lisa_env),
            file_secret_settings,
        )

    # ── Content roots ──────────────────────────────────────────────────────
    # Semicolon-separated list of extra pack directories to scan for modes,
    # operations, and tools. Resolved via ``os.pathsep`` so Windows
    # drive-letter paths survive; entries are stripped and skipped with a
    # warning when they do not exist. Override via ``SDK_CONTENT_ROOTS``.
    sdk_content_roots: str = ""

    # ── Declarative operations ────────────────────────────────────────────
    # Completion defaults for prompt-authored declarative operations. A
    # declarative operation is markdown front matter plus instructions; the
    # kernel executes it with one chat completion and no Python logic of its
    # own. These are the completion budget and sampling temperature used
    # when the operation's own front matter does not override them. 1500
    # tokens fits a structured extraction/transformation answer; 0.2 keeps
    # output near-deterministic so retries and structured parsing are
    # stable. Authoring docs (``sdk/AUTHORING.md``) and the content catalog
    # surface the same two values for front-matter documentation, so hosts
    # override in exactly one place. Override via
    # ``DECLARATIVE_COMPLETION_MAX_TOKENS`` and
    # ``DECLARATIVE_COMPLETION_TEMPERATURE``.
    declarative_completion_max_tokens: int = 1500
    declarative_completion_temperature: float = 0.2

    # ── Turn runner: iteration budgets ────────────────────────────────────
    # Maximum model-completion iterations (one model round-trip plus its
    # tool dispatches) in a single top-level turn. The counter is shared
    # across every stage handoff inside the turn — it never resets on
    # ``transition_stage`` — so a chained producer→consumer pipeline
    # (research→write) draws down the same pool. 300 is the per-turn
    # runaway governor: far above any legitimate workflow (the deepest
    # observed chained pipeline used ~90) but tight enough that a looping
    # model cannot burn unbounded tokens. The per-step sub-budgets below
    # only redistribute *when* this pool is spent. Hosts lower this only
    # for sandboxed test runs. Override via ``MAX_TOOL_ITERATIONS``.
    max_tool_iterations: int = 300
    max_auto_approved_actions: int = 10
    # Per-child iteration cap for fork fan-out. When a mode calls
    # ``fork``, each child runs as an independent sub-turn in parallel
    # with this ceiling before its turn is concluded. Children get a
    # smaller share than the parent's 300 because their job is narrower
    # (inspect → act → verify) and N children multiply N× the cost at the
    # same cap. 50 fits a non-trivial code change (~30 iterations for
    # inspect→isolate→patch→check, plus headroom for the publish step).
    # Override via ``FORK_CHILD_MAX_TOOL_ITERATIONS``.
    fork_child_max_tool_iterations: int = 50
    # Per-child iteration cap for delegate sub-turns. When a mode calls
    # ``delegate``, exactly one child runs at a time and the parent waits
    # for it; this is the child's iteration ceiling. Defaults to the fork
    # child cap so a delegated child and a forked child have identical
    # headroom unless a host explicitly tunes them apart. Override via
    # ``DELEGATE_CHILD_MAX_TOOL_ITERATIONS``.
    delegate_child_max_tool_iterations: int = 50
    # Delegate recursion depth bound. A delegate child may itself
    # delegate, forming a tree of sub-turns; this caps the tree's depth.
    # The root turn is depth 0, its delegate children depth 1, and so on.
    # 4 admits parent → child → grandchild → great-grandchild chains (the
    # legitimate orchestration pattern) while bounding a misbehaving mode
    # that would otherwise recurse indefinitely. Override via
    # ``DELEGATE_MAX_DEPTH``.
    delegate_max_depth: int = 4
    # Grace window subtracted from a delegation/fork operation's
    # ``timeout_seconds`` when arming the child sub-turn's wall budget.
    # The child's ``budget_forces_final`` fires at
    # ``timeout_seconds - grace`` so the child completes gracefully
    # (returning partial output, classified ``interrupted``) before the
    # operation registry's hard ``asyncio.wait_for`` cancel at
    # ``timeout_seconds`` discards all child work — the 2026-10-19
    # delegate timeouts. 30s covers one final forced no-tools completion
    # on a slow cloud model. Override via
    # ``DELEGATION_GRACE_SECONDS``.
    delegation_grace_seconds: int = 30

    # ── Turn runner: per-step sub-budgets ─────────────────────────────────
    # Per-step iteration sub-budget. Each workflow step (the span between
    # two ``transition_stage`` handoffs) gets its own iteration allowance
    # so a chained producer→consumer pipeline cannot starve its final step
    # on the shared ``max_tool_iterations`` pool: when a step exhausts its
    # allowance but the global cap has not fired, the kernel synthesizes a
    # handoff to the next remaining step instead of ending the turn with
    # empty content. The global cap remains the hard ceiling — a 6-step
    # chain at per_step=75 cannot exceed 300 total iterations; the
    # per-step cap only controls *when* the remaining budget moves on.
    # Override via ``PER_STEP_MAX_ITERATIONS``.
    per_step_max_iterations: int = 75
    # Per-step tool-call sub-budget. Companion to
    # ``per_step_max_iterations`` but counted in dispatched tool calls
    # rather than model iterations — a model that burns many tool calls
    # inside one step without converging is cut over to the next step (or
    # a synthesized handoff) instead of consuming the step's whole
    # iteration allowance. Resets on ``transition_stage`` exactly like the
    # per-step iteration budget. Override via ``PER_STEP_MAX_TOOL_CALLS``.
    per_step_max_tool_calls: int = 150

    # ── Turn runner: completion budgets ──────────────────────────────
    # Seconds without a streamed chunk before ``_collect_stream`` aborts
    # the generation and the turn falls back to an error completion. The
    # 2026-10-06 glm-5.3:cloud marathon froze mid-generation (socket byte
    # counters frozen 8+ minutes; the turn only recovered because the
    # upstream happened to resume) and the bare ``async for`` blocked
    # indefinitely. 0 disables the watchdog. Long *quiet-but-alive*
    # generations are unaffected: the window resets on every chunk.
    # Override via ``STREAM_IDLE_TIMEOUT_S``.
    stream_idle_timeout_s: float = 300.0
    # Per-iteration completion token budget for turn-loop model calls.
    # Reasoning-capable models (``reasoning_model``/``thinking_model``
    # flags, or a ``reasoning`` capability) bill their chain-of-thought
    # against this budget on some providers (Ollama), so a small value
    # starves the visible answer or the tool-call payload:
    # ``finish_reason=length`` with empty content, or reasoning leaking
    # into ``content`` (the 2026-09-02 empty-turn and 2026-09-17
    # meta-analysis incidents). The 2026-09-19 narration-loop incident
    # showed 8192 still truncates a single completion that must hold
    # deliberation PLUS an artifact-sized tool-call payload (an expanded
    # config and a multi-KB report script): every turn died mid-write
    # across three different models. 16384 fits deliberation (~1-3K) +
    # payload (~5-10K) with headroom, and completion spend is ~3% of
    # baseline cost (docs/TOKEN_BUDGET.md: 97% is prompt re-send), so
    # the raise is cost-negligible. Both the streaming and non-streaming
    # turn-loop paths rescue a ``finish_reason=length`` truncation with
    # one retry at 2x, context-window-capped, bounding the pathological
    # case at 32768. Override via ``TURN_COMPLETION_MAX_TOKENS``.
    turn_completion_max_tokens: int = 16384

    # ── Turn runner: guard caps ───────────────────────────────────────────
    # Caps for the correction-nudge and loop-backstop guards in
    # ``kernel.turn_runner`` and the lifecycle resolver. Each was a
    # module-level constant until hosts needed per-env tuning; the values
    # below are the production defaults (git-blame ``turn_runner.py`` for
    # the incident that motivated each). All are read by the turn loop and
    # threaded into the lifecycle resolver.
    #
    # Forced-action enforcer cap. When the routing hint encodes a
    # deterministic decision — a ``fork:`` fan-out or a single handoff
    # target (``plan``, ``search``, ``remember``, …) — and the model
    # refuses to call the expected tool (``fork`` / ``transition_stage``),
    # burning iterations instead on read-only tools or empty completions,
    # this bounds how many non-compliant dispatches pass before the
    # kernel synthesizes the required call itself. 2 bounds the
    # pre-handoff token burn to roughly ``2 × per-iteration tokens``
    # instead of the full ``max_tool_iterations`` budget (the 2026-07-27
    # fork incident; the 2026-07-30 400k-token pre-plan burn). Override
    # via ``FORCED_ACTION_NON_COMPLIANT_CAP``.
    forced_action_non_compliant_cap: int = 2
    # Stuck-in-stage nudge threshold. After this many consecutive
    # iterations in the same stage without a ``transition_stage``, the
    # loop injects a nudge into the next prompt telling the model it is
    # stalled and what to do about it. 7 gives a normal step (2–3
    # productive iterations) plus room for longer steps (4–6 productive
    # iterations) before the kernel starts pushing.
    # Override via ``STAGE_NUDGE_THRESHOLD``.
    stage_nudge_threshold: int = 7
    # Stuck-in-stage hard cap. At this many consecutive non-transitioning
    # iterations in the same stage the resolver pauses the turn (or
    # terminates for action modes with no yield escape) so the operator can
    # resume with a fresh budget — caps never silently truncate the turn.
    # Must be greater than ``stage_nudge_threshold`` — nudges fire between
    # the threshold and this cap. Override via ``STAGE_NUDGE_CAP``.
    stage_nudge_cap: int = 10
    # Write-nudge lead time, in iterations before the stuck-stage cap.
    # For a mode that carries write-contract obligations
    # (``action_tool_ids``), the parent turn fires one last "land your
    # write" nudge this many iterations before the stage cap would pause
    # the turn — the model's final chance to complete the write (e.g.
    # ``markdown_forge_document``) rather than end blocked as
    # ``no_tools``. ``0`` disables the pre-cap nudge. Override via
    # ``WRITE_DEADLINE_MARGIN``.
    write_deadline_margin: int = 1
    # Fire the one-shot read-loop nudge when this many whole-turn tool calls
    # remain (``TurnBudget.max_tool_calls``). ``0`` disables. Override via
    # ``BUDGET_NUDGE_REMAINING_CALLS``.
    budget_nudge_remaining_calls: int = 3
    # Write-rescue forced-transition target. When a write-contract mode
    # is within ``write_deadline_margin`` of the stuck-stage cap with no
    # verified write, the lifecycle resolver transitions the turn to this
    # mode so the turn ends with a productively written artifact instead
    # of pausing on a hallucinated-success block. Must name a terminal
    # capture persona whose only write is ``markdown_forge_document``;
    # set to ``""`` to disable the rescue transition entirely. Override
    # via ``WRITE_RESCUE_MODE``.
    write_rescue_mode: str = "explainer"
    # Total mode-handoff ceiling per turn. Counts every
    # ``transition_stage`` call, routed handoff, and delegate return in
    # one turn; when reached, the kernel forces a final completion. This
    # is the backstop against routing-hint ping-pong loops. It bounds
    # *handoffs*, not model iterations (``max_tool_iterations`` does
    # that). 50 accommodates deep producer→consumer pipelines; the
    # historical 6 was exceeded routinely by the 2026-08-02 fork fan-out
    # work. Override via ``MAX_HANDOFFS_PER_TURN``.
    max_handoffs_per_turn: int = 50
    # Duplicate-hint suppression threshold. The hint layer re-injects the
    # routing hint each iteration; once this many consecutive hints carry
    # the same target, further identical injection stops — the model has
    # demonstrably seen the instruction and repetition is now prompt
    # noise. 2 suppresses the third identical consecutive hint. Override
    # via ``DUPLICATE_HINT_LIMIT``.
    duplicate_hint_limit: int = 2
    # Minimum routing-hint history length for 2-cycle detection. Cycle
    # detection scans the hint-target history for A,B,A… alternation;
    # with fewer than this many consecutive targets the history cannot
    # distinguish a genuine alternation loop from legitimate reordering,
    # so suppression stays off. 3 makes A,B,A the shortest detectable
    # cycle. Override via ``HINT_CYCLE_MIN_LEN``.
    hint_cycle_min_len: int = 3
    # Hint-override allowance before suppression. An override is a
    # handoff to a target that differs from the hint's suggestion —
    # legitimate routing flexibility. But a long run of overrides while
    # hints keep flowing means the hints have stopped helping and are just
    # prompt overhead; after this many consecutive overrides the hint
    # layer stops injecting for the rest of the turn. 5 balances two
    # failure modes: 2 silenced hints too early on multi-intent queries
    # (the model legitimately reorders the suggested sequence), while 50
    # let the model redirect but bounded nothing (the 2026-08-05 markdown
    # burn: 16 round-trips at ~48k prompt tokens each after hint
    # suppression). Override via ``HINT_OVERRIDE_LIMIT``.
    hint_override_limit: int = 5
    # Identical tool-batch repetition ceiling. The model may re-issue the
    # exact same tool-call batch (same tools, same arguments) this many
    # times in a row; the next repeat is refused and the model is told to
    # answer from the results it already holds, and one further repeat
    # forces a final completion. 0 disables the guard. Read-only tools
    # return the same output for the same arguments, so a repeat buys
    # nothing — the 2026-10-28 researcher loop re-ran one
    # thread_read_recent + thread_search pair 33 times.
    # Override via ``MAX_IDENTICAL_TOOL_BATCH_REPEATS``.
    max_identical_tool_batch_repeats: int = 3

    # ── Derived workflow view ─────────────────────────────────────────────
    # A ``## Plan`` stays active only while it makes progress. After this
    # many consecutive later turns that neither complete a step nor run in
    # any open step's mode, the plan is retired so it stops decorating and
    # re-routing unrelated turns (the 2026-10-28 zombie plan re-attached
    # its progress footer and ``WORKFLOW CHAIN IN FLIGHT`` handoff to every
    # turn after it). ``/workflow conclude`` still retires a plan at once.
    # Override via ``WORKFLOW_PLAN_IDLE_TURN_LIMIT``.
    workflow_plan_idle_turn_limit: int = 2
    # ── Budget briefing ───────────────────────────────────────────────────
    # When True, inject a compact per-iteration budget status block into
    # the system prompt — remaining iterations, remaining tool calls, and
    # remaining per-target call allowance — so the model can self-scope
    # its work. Only discrete, actionable budgets are surfaced; never
    # continuous budgets (tokens, wall time), which the model cannot
    # convert into action and which anchor it to "use up" whatever it
    # sees. The full contract is injected at turn start and replaced with
    # a one-line delta each later iteration so it never accumulates in
    # the prompt. Override via ``BUDGET_BRIEFING_ENABLED``.
    budget_briefing_enabled: bool = True

    # ── Effect receipts: chat-kind softening ─────────────────────────────
    # Minimum response length (characters) for the chat-kind softening in
    # the effect-receipt verifier. A write-contract mode that ends with
    # no verified write normally blocks as ``no_tools``; the exception is
    # a chat-kind mode that dispatched at least one tool and produced
    # prose at least this long — read-only work plus a genuine
    # synthesized answer is accepted as a complete turn without a write.
    # The old hard-coded 150 let "I'll start the research once you
    # confirm" announcement prose pass as success (the 2026-08-14
    # proceed-loop); 1000 requires an actual synthesized answer. Override
    # via ``CHAT_MODE_DENSE_CONTENT_MIN_CHARS``.
    chat_mode_dense_content_min_chars: int = 1000

    # ── Context candidate token estimation ───────────────────────────────
    # Characters-per-token ratio used to estimate context candidate sizes
    # (and budget-allocated prompt sections) that do not carry their own
    # token count. 4 is the conventional English-prose middle ground:
    # 1 token ≈ 4 characters. The estimate is deliberately cheap — real
    # tokenizers are too slow per candidate in the mapper's hot path, and
    # a consistent ±20% estimate is sufficient for ranking and lane
    # budgeting. Override via ``CONTEXT_BUDGET_TOKEN_CHAR_RATIO``.
    context_budget_token_char_ratio: int = 4

    # ── Model context window budgets ─────────────────────────────────────
    # Window assumed for a model whose manifest entry does not declare
    # ``context_window`` (tokens). Shared by the context mapper and the
    # tool-result projection so both size against the same window.
    context_default_window_tokens: int = 131072
    # Provider framing/token-count uncertainty reserve. Tool results use
    # the remainder after complete request input and output reservations.
    context_request_margin_share: float = Field(default=0.05, ge=0, lt=1)
    # Low-water mark for a pressure reduction, as a share of the room tool
    # results may occupy. Fitting just under the limit forced another
    # reduction (rewriting early messages and missing the provider prefix
    # cache) on nearly every later call; the slack lets the next calls
    # append without rewriting anything.
    context_reduction_target_share: float = Field(default=0.75, gt=0, le=1)

    # ── Context mapper: scoring weights ────────────────────────────────
    # Per-provenance authority multiplier, applied in the mapper's
    # authority component as ``authority × confidence × multiplier``.
    # Keys must exactly match the ``ContextProvenance`` literals in
    # ``kernel/context/models.py``. The ordering encodes trust: curated
    # and system-observed evidence outranks user assertion, which
    # outranks recalled memory, then inference, diagnostics, and pure
    # speculation. Override via ``CONTEXT_MAPPER_PROVENANCE_AUTHORITY``
    # as a JSON object (all seven keys required).
    context_mapper_provenance_authority: dict[str, float] = {
        "authoritative": 1.0,
        "system_observed": 0.98,
        "user_asserted": 0.75,
        "recalled_memory": 0.70,
        "inferred": 0.55,
        "diagnostic": 0.45,
        "speculative": 0.30,
    }
    # Per-conflict-risk penalty, weighted by the session plan's
    # ``conflict_penalty`` weight and subtracted from a candidate's base
    # score. Keys must exactly match the ``ContextConflictRisk`` literals
    # in ``kernel/context/models.py``. Contradicting evidence stays
    # selectable but ranks below clean evidence proportionally to how
    # risky the contradiction is. Override via
    # ``CONTEXT_MAPPER_CONFLICT_PENALTY`` as a JSON object.
    context_mapper_conflict_penalty: dict[str, float] = {
        "low": 0.0,
        "medium": 0.5,
        "high": 1.0,
    }

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
        "stream_idle_timeout_s",
        "max_tool_iterations",
        "max_auto_approved_actions",
        "fork_child_max_tool_iterations",
        "delegate_child_max_tool_iterations",
        "delegate_max_depth",
        "delegation_grace_seconds",
        "per_step_max_iterations",
        "per_step_max_tool_calls",
        "turn_completion_max_tokens",
        "forced_action_non_compliant_cap",
        "stage_nudge_threshold",
        "stage_nudge_cap",
        "write_deadline_margin",
        "budget_nudge_remaining_calls",
        "max_handoffs_per_turn",
        "duplicate_hint_limit",
        "hint_cycle_min_len",
        "hint_override_limit",
        "max_identical_tool_batch_repeats",
        "workflow_plan_idle_turn_limit",
        "chat_mode_dense_content_min_chars",
        "context_budget_token_char_ratio",
        "context_default_window_tokens",
        "context_request_margin_share",
        "context_reduction_target_share",
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
