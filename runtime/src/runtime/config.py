"""HostConfig — typed, frozen configuration for a Lisa Harness host.

This replaces the config-reading parts of the old ``_RUNTIME_STATE`` dict.
It is constructed once at host startup (from CLI args, env vars, or direct
kwargs) and passed to ``HostRuntime``. All fields are resolved at
construction time — no deferred env-var reads.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from kernel.execution_policy import ExecutionPolicy
from sdk.layout import CONFIG_DIR, DATA_DIR, LOGS_DIR, STATE_DIR

from runtime.execution_policy import default_interactive_execution_policy

logger = logging.getLogger(__name__)

# ── Environment-variable name constants ──────────────────────────────
# Centralised so HostConfig and any host that reads env directly share
# a single source of truth.

LISA_TIMEZONE_ENV = "LISA_TIMEZONE"
LISA_LOG_LEVEL_ENV = "LISA_LOG_LEVEL"
LISA_LOG_DEBUG_ENV = "LISA_LOG_DEBUG"
LISA_DATA_DIR_ENV = "LISA_DATA_DIR"
LISA_PROFILE_ENV = "LISA_PROFILE"
LISA_THREAD_DIR_ENV = "LISA_THREAD_DIR"
LISA_SLICE_PRE_TURN_ENV = "LISA_SLICE_PRE_TURN"
LISA_SLICE_POST_TURN_ENV = "LISA_SLICE_POST_TURN"
LISA_SLICE_EXPLICIT_ENV = "LISA_SLICE_EXPLICIT"
LISA_SLICE_BUDGET_MS_ENV = "LISA_SLICE_BUDGET_MS"
LISA_THREAD_SEGMENT_TURNS_ENV = "LISA_THREAD_SEGMENT_TURNS"
LISA_THREAD_REDACTION_ENABLED_ENV = "LISA_THREAD_REDACTION_ENABLED"
LISA_THREAD_REDACTION_PATTERNS_ENV = "LISA_THREAD_REDACTION_PATTERNS"
LISA_CONTEXT_BUDGET_ENABLED_ENV = "LISA_CONTEXT_BUDGET_ENABLED"
LISA_CONTEXT_BUDGET_CORE_MAX_TOKENS_ENV = "LISA_CONTEXT_BUDGET_CORE_MAX_TOKENS"
LISA_CONTEXT_BUDGET_RETRIEVAL_MAX_TOKENS_ENV = "LISA_CONTEXT_BUDGET_RETRIEVAL_MAX_TOKENS"
LISA_CONTEXT_BUDGET_MEMORY_MAX_TOKENS_ENV = "LISA_CONTEXT_BUDGET_MEMORY_MAX_TOKENS"
LISA_CONTEXT_BUDGET_REFLECTION_MAX_TOKENS_ENV = "LISA_CONTEXT_BUDGET_REFLECTION_MAX_TOKENS"
LISA_CONTEXT_BUDGET_HISTORY_MAX_TOKENS_ENV = "LISA_CONTEXT_BUDGET_HISTORY_MAX_TOKENS"
LISA_CONTEXT_BUDGET_OPERATIONAL_MAX_TOKENS_ENV = "LISA_CONTEXT_BUDGET_OPERATIONAL_MAX_TOKENS"
LISA_CONTEXT_BUDGET_TOKEN_CHAR_RATIO_ENV = "LISA_CONTEXT_BUDGET_TOKEN_CHAR_RATIO"
LISA_CONTEXT_BUDGET_REPLAY_MAX_BYTES_ENV = "LISA_CONTEXT_BUDGET_REPLAY_MAX_BYTES"
LISA_PROMPT_CACHE_ENABLED_ENV = "LISA_PROMPT_CACHE_ENABLED"

# ── Default values ────────────────────────────────────────────────────

_DEFAULT_THREAD_SEGMENT_TURNS = 256
#: Host ceiling for the turn-start replay working set (bytes). Matches the
#: projection module's per-turn tool-result ceiling so default turns
#: behave as they did before the budget existed; sessions tighten via
#: ``context.history.replay_max_bytes`` (never raise past this).
_DEFAULT_REPLAY_BUDGET_BYTES = 65536
_DEFAULT_SLICE_EXPLICIT_BUDGET_MS = 2000
_DEFAULT_DATA_DIR_NAME = ".lisa"
_PROFILES_SUBDIR_NAME = "profiles"


def default_data_dir() -> Path:
    """Return the platform-appropriate default Lisa data directory.

    On Windows the home directory is derived from ``%USERPROFILE%`` (via
    ``Path.home()``), and on Linux/macOS from ``$HOME``. ``Path.home()``
    handles both correctly and expands ``~`` on POSIX. The resolved path is
    ``~/.lisa`` (i.e. ``$HOME/.lisa`` on POSIX, ``%USERPROFILE%\\.lisa`` on
    Windows).
    """
    return Path.home() / _DEFAULT_DATA_DIR_NAME


def profile_data_dir(profile: str, *, base_dir: Path | None = None) -> Path:
    """Return the data directory for a named profile.

    A profile is a named sub-workspace rooted at
    ``<base_dir>/profiles/<name>`` (default base: ``~/.lisa``). It isolates
    everything derived from ``data_dir`` — threads, memory, models.yaml,
    policy, approvals, session context, files — while the packs venv stays
    shared with the base data dir (see :func:`runtime.paths.packs_venv_for`).

    ``LISA_DATA_DIR`` (explicit) still wins over ``LISA_PROFILE``; use the
    profile when the user did not pin an explicit data dir. The name must
    be a single path-safe segment (letters, digits, ``-``, ``_``, ``.``) —
    ``..`` and separators are rejected so a profile name can never escape
    the profiles/ root.

    The directory is not created here; hosts create it lazily like any
    other data dir (``ensure_data_dir`` MCP specs, ``resolve_data_dir``).
    """
    name = profile.strip()
    if not name:
        raise ValueError("LISA_PROFILE must not be empty")
    if not _PROFILE_NAME_RE.fullmatch(name):
        raise ValueError(
            f"Invalid profile name {name!r}: use letters, digits, '-', '_', '.' "
            "(single path segment, no separators)"
        )
    base = base_dir if base_dir is not None else default_data_dir()
    return (base / _PROFILES_SUBDIR_NAME / name).resolve()


_PROFILE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _resolve_env(name: str, default: str = "") -> str:
    """Read an env var and return the stripped value or default."""
    value = os.getenv(name, "").strip()
    return value if value else default


def _resolve_env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = _resolve_env(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        logger.warning("invalid %s value %r; using default %d", name, raw, default)
        return default


def _resolve_env_bool(name: str, default: bool) -> bool:
    """Read an env var as a boolean. Accepts 1/true/yes/on (case-insensitive)."""
    raw = _resolve_env(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    logger.warning("invalid %s value %r; using default %s", name, raw, default)
    return default


def _normalize_slice_flag(value: str) -> bool | None:
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    if cleaned in {"1", "true", "yes", "on"}:
        return True
    if cleaned in {"0", "false", "no", "off"}:
        return False
    return None


def _context_budget_overrides() -> dict[str, int | bool]:
    """Resolve context-budget values that have explicit env overrides.

    Returns only the keys actually present in the environment; the
    dataclass defaults are the source of truth for everything else.
    Invalid integers log a warning and fall back to the default.
    """

    overrides: dict[str, int | bool] = {}
    bool_raw = _resolve_env(LISA_CONTEXT_BUDGET_ENABLED_ENV)
    if bool_raw:
        enabled = _resolve_env_bool(LISA_CONTEXT_BUDGET_ENABLED_ENV, True)
        overrides["context_budget_enabled"] = enabled
    int_specs = (
        (LISA_CONTEXT_BUDGET_CORE_MAX_TOKENS_ENV, "context_budget_core_max_tokens", 2000),
        (
            LISA_CONTEXT_BUDGET_RETRIEVAL_MAX_TOKENS_ENV,
            "context_budget_retrieval_max_tokens",
            1500,
        ),
        (LISA_CONTEXT_BUDGET_MEMORY_MAX_TOKENS_ENV, "context_budget_memory_max_tokens", 1200),
        (
            LISA_CONTEXT_BUDGET_REFLECTION_MAX_TOKENS_ENV,
            "context_budget_reflection_max_tokens",
            1000,
        ),
        (LISA_CONTEXT_BUDGET_HISTORY_MAX_TOKENS_ENV, "context_budget_history_max_tokens", 1500),
        (
            LISA_CONTEXT_BUDGET_OPERATIONAL_MAX_TOKENS_ENV,
            "context_budget_operational_max_tokens",
            800,
        ),
        (
            LISA_CONTEXT_BUDGET_TOKEN_CHAR_RATIO_ENV,
            "context_budget_token_char_ratio",
            4,
        ),
        (
            LISA_CONTEXT_BUDGET_REPLAY_MAX_BYTES_ENV,
            "context_budget_replay_max_bytes",
            _DEFAULT_REPLAY_BUDGET_BYTES,
        ),
    )
    for env_name, field_name, default_value in int_specs:
        raw = _resolve_env(env_name)
        if not raw:
            continue
        try:
            overrides[field_name] = max(0, int(raw))
        except ValueError:
            logger.warning("invalid %s value %r; using default %d", env_name, raw, default_value)
    return overrides


@dataclass(frozen=True, slots=True)
class SliceConfig:
    """Slice (opportunistic house-keeping) configuration."""

    pre_turn_enabled: bool | None = None
    post_turn_enabled: bool | None = None
    explicit_enabled: bool | None = None
    explicit_budget_ms: int = _DEFAULT_SLICE_EXPLICIT_BUDGET_MS

    def to_tuning_dict(self) -> dict[str, object]:
        tuning: dict[str, object] = {}
        if self.pre_turn_enabled is not None:
            tuning["pre_turn_enabled"] = self.pre_turn_enabled
        if self.post_turn_enabled is not None:
            tuning["post_turn_enabled"] = self.post_turn_enabled
        if self.explicit_enabled is not None:
            tuning["explicit_enabled"] = self.explicit_enabled
        return tuning


@dataclass(frozen=True, slots=True)
class _SliceEnvOverrides:
    """Slice flags and explicit budget as resolved from the environment."""

    pre_turn_enabled: bool | None
    post_turn_enabled: bool | None
    explicit_enabled: bool | None
    explicit_budget_ms: int


def _slice_overrides() -> _SliceEnvOverrides:
    """Resolve slice flags and explicit-budget from the environment."""

    return _SliceEnvOverrides(
        pre_turn_enabled=_normalize_slice_flag(_resolve_env(LISA_SLICE_PRE_TURN_ENV)),
        post_turn_enabled=_normalize_slice_flag(_resolve_env(LISA_SLICE_POST_TURN_ENV)),
        explicit_enabled=_normalize_slice_flag(_resolve_env(LISA_SLICE_EXPLICIT_ENV)),
        explicit_budget_ms=_slice_explicit_budget_env(),
    )


def _slice_explicit_budget_env() -> int:
    """Read LISA_SLICE_BUDGET_MS with clamping; default when unset/invalid."""

    raw_budget = _resolve_env(LISA_SLICE_BUDGET_MS_ENV, "").strip()
    if not raw_budget:
        return _DEFAULT_SLICE_EXPLICIT_BUDGET_MS
    with contextlib.suppress(ValueError):
        return max(50, min(int(raw_budget), 30_000))
    return _DEFAULT_SLICE_EXPLICIT_BUDGET_MS


def _log_level_override(explicit_log_level: str) -> str:
    """Resolve log level: explicit CLI arg wins, then env, else empty."""

    raw_log_level = explicit_log_level.strip()
    if raw_log_level:
        return raw_log_level
    env_log = _resolve_env(LISA_LOG_LEVEL_ENV)
    return env_log if env_log else ""


def _thread_redaction_env() -> tuple[bool, tuple[str, ...]]:
    """Resolve thread-redaction enabled flag and patterns from the environment.

    Returns the effective ``(enabled, patterns)`` pair. A ``none`` pattern
    value disables redaction entirely; explicit patterns replace defaults;
    unset falls back to the compiled-in defaults with redaction enabled.
    """

    enabled = _resolve_env_bool(LISA_THREAD_REDACTION_ENABLED_ENV, True)
    raw_patterns = _resolve_env(LISA_THREAD_REDACTION_PATTERNS_ENV, "").strip()
    if raw_patterns.lower() == "none":
        return False, ()
    if raw_patterns:
        return enabled, tuple(p for p in raw_patterns.splitlines() if p.strip())
    return enabled, ()


def _resolved_path_or_none(raw_value: str, env_name: str) -> Path | None:
    """Resolve one optional path field (CLI arg first, then env) to absolute."""

    raw = raw_value.strip() or _resolve_env(env_name)
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


@dataclass(frozen=True, slots=True)
class StreamingConfig:
    """Streaming display configuration (transport-hint, not transport-owned)."""

    mode: str = "none"  # none | stage | text | hybrid
    char_delay_ms: int = 0

    @property
    def enabled(self) -> bool:
        return self.mode != "none"


@dataclass(frozen=True, slots=True)
class HostConfig:
    """Typed, frozen configuration for a Lisa Harness host.

    Constructed once at startup from CLI args, env vars, or direct kwargs.
    Passed to ``HostRuntime`` which uses it to wire the kernel turn loop.

    All path fields are resolved to absolute ``Path`` objects at
    construction time. No deferred env-var reads.
    """

    # ── Paths ────────────────────────────────────────────────────────────
    data_dir: Path = field(default_factory=default_data_dir)
    overlays_dir: Path | None = None
    thread_dir: Path | None = None
    # User-editable MCP server manifest path. When ``None``, defaults to
    # ``\u003cdata_dir\u003e/mcp_servers.yaml`` (absent = no user servers).
    # The framework-owned manifest (``host/mcp_servers.yaml`` in-repo) is
    # always loaded as a fixed base layer and cannot be redirected here.
    mcp_manifest_path: Path | None = None
    state_snapshot_path: Path | None = None

    # ── Runtime ─────────────────────────────────────────────────────────
    debug: bool = False
    log_level: str = "info"
    log_path: Path | None = None
    host_log_path: Path | None = None
    # ``"cli"``, ``"web"``, or ``None`` (MCP servers). Controls the
    # stderr-log default — see ``sdk.logging_setup.configure_logging``.
    host_kind: str | None = None
    # Logger names forced to DEBUG via ``LISA_LOG_DEBUG`` (comma-separated).
    debug_log_modules: tuple[str, ...] = ()
    timezone: str = "UTC"
    timezone_source: str = "default"
    user_slug: str = ""
    session_context: str = ""
    no_thread: bool = False

    # ── Streaming (transport hint) ─────────────────────────────────────
    streaming: StreamingConfig = field(default_factory=StreamingConfig)

    # ── Slice ──────────────────────────────────────────────────────────
    slice_config: SliceConfig = field(default_factory=SliceConfig)

    # ── Limits ─────────────────────────────────────────────────────────
    # Turn threshold at which the active thread segment is sealed to a
    # compressed sidecar. Honoured by every ``append_thread_entry`` call
    # site with a runtime in scope. See ``host/thread.py`` and
    # ``docs/SPEC/THREAD_STORE.md``.
    thread_segment_turns: int = _DEFAULT_THREAD_SEGMENT_TURNS
    # ── Thread redaction ─────────────────────────────────────────────
    # When True, thread-read methods (build_recent_excerpt,
    # build_manifest, search) apply a :class:`ThreadRedactionPolicy`
    # that strips regex-matched substrings from ``response_text`` and
    # ``query`` before rendering into model context. The default policy
    # removes prior cap-fire / budget-exhaustion self-reports that
    # contaminate subsequent turns. Override the patterns via
    # ``LISA_THREAD_REDACTION_PATTERNS`` (newline-separated regex strings
    # — empty means defaults only; ``none`` disables all patterns).
    thread_redaction_enabled: bool = True
    thread_redaction_patterns: tuple[str, ...] = ()  # empty = defaults only

    # ── Memory recall (pre-turn injection) ─────────────────────────────
    memory_recall_enabled: bool = True
    memory_recall_pre_turn_limit: int = 3
    memory_recall_min_query_chars: int = 3

    # ── Memory auto-capture (post-turn learning) ───────────────────────
    memory_auto_capture_enabled: bool = True
    memory_auto_capture_min_salience: float = 0.5
    # Salience → tier cut points for auto-captured learnings (see
    # ``classify_memory_tier``). Below ``daily_threshold`` is ``daily``;
    # between the two is ``weekly``; at/above ``weekly_threshold`` is
    # ``long_term``. Workflow-state captures (write-effect tools, soft
    # commitments) thus expire quickly rather than polluting long-term
    # recall for a year.
    memory_auto_capture_daily_threshold: float = 0.65
    memory_auto_capture_weekly_threshold: float = 0.8

    # ── Context budget (per-lane token ceilings for prompt assembly) ───
    # The dynamic user-context block is assembled from optional prompt
    # sections grouped into six lanes. Each lane has a token ceiling; a
    # section that exceeds its lane's remaining budget is dropped (or
    # truncated when allowed) rather than pushing other sections out.
    # ``ContextBudgetAllocator`` (kernel) consults these values via the
    # ``for_turn`` factory; the allocator's total ceiling is normally the
    # model-aware window computed by the context mapper, and the lane
    # ceilings are the per-lane split of that window.
    # Master switch. When False, sections bypass budgeting entirely and
    # pass through at full size — assembly is unbounded. Override via
    # ``LISA_CONTEXT_BUDGET_ENABLED``.
    context_budget_enabled: bool = True
    # Ceiling for the ``core`` lane: always-on turn framing — ambient
    # facts (time, host OS, host name) and the per-turn dynamic-context
    # header (mode id/kind, query length, host/user slug). This lane is
    # allocated first and small by design; if the core header ever
    # exceeds 2000 tokens something upstream is broken, so treat a breach
    # as a bug signal rather than raising the cap. Override via
    # ``LISA_CONTEXT_BUDGET_CORE_MAX_TOKENS``.
    context_budget_core_max_tokens: int = 2000
    # Ceiling for the ``retrieval`` lane: ranked context evidence the
    # session-owned ContextMapper selected from UnifiedState (pack items,
    # operational state, recalled evidence). Note the mapper already fits
    # its output to the model-aware window before this ceiling applies,
    # so this lane budget is a secondary bound, not the primary trim.
    # Override via ``LISA_CONTEXT_BUDGET_RETRIEVAL_MAX_TOKENS``.
    context_budget_retrieval_max_tokens: int = 1500
    # Ceiling for the ``memory`` lane: authored session context the user
    # pinned for the session (``--session-context`` markdown, normalized
    # to provenance tags). Pinned instructions outrank derived evidence,
    # so this lane is generous; it exists so a runaway authored file
    # cannot crowd out live conversation. Override via
    # ``LISA_CONTEXT_BUDGET_MEMORY_MAX_TOKENS``.
    context_budget_memory_max_tokens: int = 1200
    # Ceiling for the ``reflection`` lane: the diagnostic-reflection
    # block injected when replay controls enable it, summarizing what the
    # previous turn did and what remains. Session front matter may
    # tighten this cap but never raise it. Override via
    # ``LISA_CONTEXT_BUDGET_REFLECTION_MAX_TOKENS``.
    context_budget_reflection_max_tokens: int = 1000
    # Ceiling for the ``history`` lane: the recent-conversation excerpt
    # and the thread manifest. Both sections share this lane; the recent
    # excerpt is allocated first (it carries the assistant's last answer
    # the user is replying to) and the manifest takes the remainder, so
    # an oversized manifest is what gets dropped. Session front matter
    # may tighten this cap but never raise it. Override via
    # ``LISA_CONTEXT_BUDGET_HISTORY_MAX_TOKENS``.
    context_budget_history_max_tokens: int = 1500
    # Ceiling for the ``operational`` lane: the live capability manifest
    # (modes/operations/tools/models the model may call this turn).
    # Counts scale with installed packs; keep the ceiling generous, but
    # not so generous that a pathological pack catalog silently floods
    # the prompt. Override via
    # ``LISA_CONTEXT_BUDGET_OPERATIONAL_MAX_TOKENS``.
    context_budget_operational_max_tokens: int = 800
    # Ceiling for the turn-start replay working set (bytes): the total
    # tool-result bytes replayed verbatim for the open task before the
    # rest collapse to recoverable ref-stubs (residency by allocation,
    # R2). Default matches the sdk projection ceiling so ordinary turns
    # are unchanged; session front matter may tighten via
    # ``context.history.replay_max_bytes`` but never raise past this.
    # Override via ``LISA_CONTEXT_BUDGET_REPLAY_MAX_BYTES``.
    context_budget_replay_max_bytes: int = _DEFAULT_REPLAY_BUDGET_BYTES

    # ── Prompt caching (provider wire-level) ────────────────────────
    # When True (default), providers that support explicit prompt-cache
    # breakpoints (Anthropic's ``cache_control``) mark the stable system
    # prefix as cacheable while the dynamic tail (routing hints, lifecycle
    # briefing) stays uncached. When False, requests fall back to legacy
    # single-string system prompts — the immediate rollback switch.
    # Override via ``LISA_PROMPT_CACHE_ENABLED`` (0/false/no/off).
    prompt_cache_enabled: bool = True

    # ── Surface identity defaults ─────────────────────────────────────
    # Default scope values stamped onto every OperationExecutionContext
    # and TurnHooks when the host does not supply a more specific value.
    # The CLI host derives most of these from ``user_slug`` and the
    # session plan; these defaults exist so bare/test hosts produce
    # coherent scope addresses without wiring. Keep them aligned with the
    # framework's convention for single-user personal surfaces; a host
    # with real multi-tenancy must set its own identity per turn.
    # Workspace partitions durable state (memories, approvals, caches).
    identity_workspace_id: str = "lisa-host"
    # Stable user identifiers for memory scoping and privacy boundaries.
    identity_user_id: str = "lisa-user"
    identity_user_slug: str = "lisa-user"
    identity_user_name: str = "Lisa User"
    # Principal key for approval/audit attribution.
    identity_principal_key: str = "lisa-user"
    # Conversation and surface identity for the default single-user
    # personal-DM console surface.
    identity_conversation_id: str = "lisa-host-console"
    identity_surface_type: str = "personal_dm"
    identity_surface_id: str = "lisa-host-console"
    # Memory privacy scope: which memory tier this surface may read.
    identity_memory_scope: str = "personal"

    # ── Execution policy ───────────────────────────────────────────────
    execution_policy: ExecutionPolicy = field(default_factory=default_interactive_execution_policy)

    # ── Surface identity ───────────────────────────────────────────────
    surface_type: str = ""
    surface_id: str = ""
    memory_scope: str = ""

    # ── Pack policy ────────────────────────────────────────────────────
    always_on_packs: tuple[str, ...] = (
        "lisa-pack-core",
        "lisa-pack-introspect",
    )

    @property
    def resolved_overlays_dir(self) -> Path:
        """Return the resolved overlays directory, defaulting to ``<data_dir>/config/overlays``."""
        return self.overlays_dir or (self.data_dir / CONFIG_DIR / "overlays")

    @property
    def resolved_thread_dir(self) -> Path:
        return self.thread_dir or (self.data_dir / DATA_DIR / "threads")

    @property
    def resolved_log_path(self) -> Path | None:
        return self.log_path

    @property
    def resolved_host_log_path(self) -> Path:
        return self.host_log_path or (self.data_dir / LOGS_DIR / "runtime.jsonl")

    @property
    def resolved_models_path(self) -> Path:
        return self.data_dir / CONFIG_DIR / "models.yaml"

    @property
    def resolved_models_proposal_path(self) -> Path:
        return self.data_dir / STATE_DIR / "models.proposed.yaml"

    @property
    def effective_log_level(self) -> str:
        """Return the log level, falling back to debug when debug is on.

        If ``log_level`` is explicitly set to a valid value, that wins.
        Otherwise, ``debug=True`` implies ``"debug"``; ``debug=False``
        defaults to ``"info"``.
        """
        from runtime.config_resolver import normalize_log_level

        normalized = normalize_log_level(self.log_level)
        if normalized is not None:
            return normalized
        return "debug" if self.debug else "info"

    @classmethod
    def from_env(
        cls,
        *,
        debug: bool = False,
        log_level: str = "",
        log_path: str = "",
        host_log_path: str = "",
        data_dir: str = "",
        overlays_dir: str = "",
        thread_dir: str = "",
        no_thread: bool = False,
        timezone: str = "",
        session_context: str = "",
        execution_policy: ExecutionPolicy | None = None,
        user_slug: str = "",
        state_snapshot: str = "",
        mcp_manifest_path: str = "",
        streaming: str = "hybrid",
        surface_type: str = "",
        surface_id: str = "",
    ) -> HostConfig:
        """Build a ``HostConfig`` from CLI args + env vars.

        CLI args take precedence over env vars; env vars take precedence
        over defaults. This is the primary constructor for all hosts.
        """
        # ── Timezone ──────────────────────────────────────────────────
        from runtime.config_resolver import resolve_timezone

        # CLI arg takes precedence; only pass it to resolve_timezone when
        # the caller explicitly provided one. Otherwise let resolve_timezone
        # read the env var itself so the source label is "env:..." not "cli".
        cli_tz = timezone.strip()
        resolved_tz, tz_source = resolve_timezone(cli_tz)

        resolved_log_level = _log_level_override(log_level)

        # ── Data dir ──────────────────────────────────────────────────
        # Precedence: explicit CLI arg > LISA_DATA_DIR > LISA_PROFILE >
        # platform default (~/.lisa). A profile resolves to
        # <default_data_dir>/profiles/<name> and isolates all data-dir
        # derived state while sharing the base packs venv.
        raw_data_dir = data_dir.strip()
        if not raw_data_dir:
            raw_data_dir = _resolve_env(LISA_DATA_DIR_ENV)
        if not raw_data_dir:
            raw_profile = _resolve_env(LISA_PROFILE_ENV)
            if raw_profile:
                raw_data_dir = str(profile_data_dir(raw_profile))
        resolved_data_dir = Path(raw_data_dir or str(default_data_dir())).expanduser().resolve()

        # ── Overlays dir ──────────────────────────────────────────────
        resolved_overlays_dir: Path | None = None
        raw_overlays = overlays_dir.strip()
        if raw_overlays:
            resolved_overlays_dir = Path(raw_overlays).expanduser().resolve()

        # ── Paths ──────────────────────────────────────────────────────
        resolved_thread_dir = _resolved_path_or_none(thread_dir, LISA_THREAD_DIR_ENV)
        resolved_log_path = _resolved_path_or_none(log_path, "")
        resolved_host_log_path = _resolved_path_or_none(host_log_path, "")

        # ── Debug-module overrides (LISA_LOG_DEBUG) ────────────────────
        from sdk.logging_setup import _parse_debug_modules

        resolved_debug_modules = tuple(_parse_debug_modules(_resolve_env(LISA_LOG_DEBUG_ENV)))

        resolved_mcp_manifest = _resolved_path_or_none(mcp_manifest_path, "")
        resolved_snapshot = _resolved_path_or_none(state_snapshot, "")

        # ── Streaming ──────────────────────────────────────────────────
        from runtime.config_resolver import parse_streaming_option

        stream_mode, stream_delay = parse_streaming_option(streaming)

        # ── Slice ──────────────────────────────────────────────────────
        slice_values = _slice_overrides()

        # ── Limits ─────────────────────────────────────────────────────
        thread_segment_turns = _resolve_env_int(
            LISA_THREAD_SEGMENT_TURNS_ENV, _DEFAULT_THREAD_SEGMENT_TURNS
        )
        # ── Thread redaction config ────────────────
        thread_redaction_enabled, thread_redaction_patterns = _thread_redaction_env()

        context_budget = _context_budget_overrides()
        prompt_cache_enabled = _resolve_env_bool(LISA_PROMPT_CACHE_ENABLED_ENV, True)

        # ── User / session ─────────────────────────────────────────────
        # Session source compilation is intentionally deferred to HostRuntime
        # so every host has exactly one strict SessionPlan startup path.
        resolved_session_context = session_context.strip()
        raw_user_slug = user_slug.strip()
        resolved_user_slug = raw_user_slug
        resolved_surface_type = surface_type.strip()
        resolved_surface_id = surface_id.strip()
        resolved_memory_scope = ""
        resolved_always_on_packs: tuple[str, ...] = (
            "lisa-pack-core",
            "lisa-pack-introspect",
        )

        return cls(
            data_dir=resolved_data_dir,
            overlays_dir=resolved_overlays_dir,
            thread_dir=resolved_thread_dir,
            mcp_manifest_path=resolved_mcp_manifest,
            state_snapshot_path=resolved_snapshot,
            debug=debug,
            log_level=resolved_log_level,
            log_path=resolved_log_path,
            host_log_path=resolved_host_log_path,
            debug_log_modules=resolved_debug_modules,
            timezone=resolved_tz,
            timezone_source=tz_source,
            user_slug=resolved_user_slug,
            session_context=resolved_session_context,
            no_thread=no_thread,
            streaming=StreamingConfig(mode=stream_mode, char_delay_ms=stream_delay),
            slice_config=SliceConfig(
                pre_turn_enabled=slice_values.pre_turn_enabled,
                post_turn_enabled=slice_values.post_turn_enabled,
                explicit_enabled=slice_values.explicit_enabled,
                explicit_budget_ms=slice_values.explicit_budget_ms,
            ),
            thread_segment_turns=thread_segment_turns,
            thread_redaction_enabled=thread_redaction_enabled,
            thread_redaction_patterns=thread_redaction_patterns,
            prompt_cache_enabled=prompt_cache_enabled,
            execution_policy=execution_policy or default_interactive_execution_policy(),
            surface_type=resolved_surface_type,
            surface_id=resolved_surface_id,
            memory_scope=resolved_memory_scope,
            always_on_packs=resolved_always_on_packs,
            **context_budget,  # type: ignore[arg-type]
        )
