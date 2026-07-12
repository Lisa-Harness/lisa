"""HostConfig — typed, frozen configuration for a Lisa framework host.

This replaces the config-reading parts of the old ``_RUNTIME_STATE`` dict.
It is constructed once at host startup (from CLI args, env vars, or direct
kwargs) and passed to ``HostRuntime``. All fields are resolved at
construction time — no deferred env-var reads.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Environment-variable name constants ──────────────────────────────
# Centralised so HostConfig and any host that reads env directly share
# a single source of truth.

LISA_TIMEZONE_ENV = "LISA_TIMEZONE"
LISA_LOG_LEVEL_ENV = "LISA_LOG_LEVEL"
LISA_DATA_DIR_ENV = "LISA_DATA_DIR"
LISA_MEMORY_DB_ENV = "LISA_MEMORY_DB"
LISA_THREAD_DIR_ENV = "LISA_THREAD_DIR"
LISA_MODELS_FILE_ENV = "LISA_MODELS_FILE"
LISA_SLICE_PRE_TURN_ENV = "LISA_SLICE_PRE_TURN"
LISA_SLICE_POST_TURN_ENV = "LISA_SLICE_POST_TURN"
LISA_SLICE_EXPLICIT_ENV = "LISA_SLICE_EXPLICIT"
LISA_SLICE_BUDGET_MS_ENV = "LISA_SLICE_BUDGET_MS"
LISA_THREAD_MAX_ENTRIES_ENV = "LISA_THREAD_MAX_ENTRIES"
LISA_MEMORY_MAX_RECORDS_ENV = "LISA_MEMORY_MAX_RECORDS"

# Legacy env var aliases (Little Lisa compatibility during migration).
_LITTLE_LISA_PREFIX = "LITTLE_LISA_"
_LITTLE_LISA_ALIASES: dict[str, str] = {
    "LITTLE_LISA_TIMEZONE": LISA_TIMEZONE_ENV,
    "LITTLE_LISA_LOG_LEVEL": LISA_LOG_LEVEL_ENV,
    "LITTLE_LISA_DATA_DIR": LISA_DATA_DIR_ENV,
    "LITTLE_LISA_MEMORY_DB": LISA_MEMORY_DB_ENV,
    "LITTLE_LISA_THREAD_DIR": LISA_THREAD_DIR_ENV,
    "LITTLE_LISA_MODELS_FILE": LISA_MODELS_FILE_ENV,
    "LITTLE_LISA_SLICE_PRE_TURN": LISA_SLICE_PRE_TURN_ENV,
    "LITTLE_LISA_SLICE_POST_TURN": LISA_SLICE_POST_TURN_ENV,
    "LITTLE_LISA_SLICE_EXPLICIT": LISA_SLICE_EXPLICIT_ENV,
    "LITTLE_LISA_SLICE_BUDGET_MS": LISA_SLICE_BUDGET_MS_ENV,
    "LITTLE_LISA_THREAD_MAX_ENTRIES": LISA_THREAD_MAX_ENTRIES_ENV,
    "LITTLE_LISA_MEMORY_MAX_RECORDS": LISA_MEMORY_MAX_RECORDS_ENV,
}

# ── Default values ────────────────────────────────────────────────────

_DEFAULT_THREAD_MAX_ENTRIES = 50
_DEFAULT_MEMORY_MAX_RECORDS = 200
_DEFAULT_SLICE_EXPLICIT_BUDGET_MS = 2000


def _resolve_env(name: str, default: str = "") -> str:
    """Read an env var, falling back to legacy LITTLE_LISA_* aliases."""
    value = os.getenv(name, "").strip()
    if value:
        return value
    legacy = _LITTLE_LISA_ALIASES.get(name)
    if legacy is None:
        # Try the reverse: if the caller passed a LITTLE_LISA_* name, use it.
        return os.getenv(name, default).strip()
    legacy_value = os.getenv(legacy, "").strip()
    return legacy_value if legacy_value else default


def _resolve_env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = _resolve_env(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        logger.warning("invalid %s value %r; using default %d", name, raw, default)
        return default


def _normalize_log_level(value: str) -> str | None:
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    if cleaned in {"debug", "info", "warning", "error", "critical"}:
        return cleaned
    return None


def _normalize_slice_flag(value: str) -> bool | None:
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    if cleaned in {"1", "true", "yes", "on"}:
        return True
    if cleaned in {"0", "false", "no", "off"}:
        return False
    return None


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
class StreamingConfig:
    """Streaming display configuration (transport-hint, not transport-owned)."""

    mode: str = "none"  # none | stage | text | hybrid
    char_delay_ms: int = 0

    @property
    def enabled(self) -> bool:
        return self.mode != "none"


@dataclass(frozen=True, slots=True)
class HostConfig:
    """Typed, frozen configuration for a Lisa framework host.

    Constructed once at startup from CLI args, env vars, or direct kwargs.
    Passed to ``HostRuntime`` which uses it to wire the kernel turn loop.

    All path fields are resolved to absolute ``Path`` objects at
    construction time. No deferred env-var reads.
    """

    # ── Paths ────────────────────────────────────────────────────────────
    data_dir: Path = field(default_factory=lambda: Path.cwd() / "artifacts")
    memory_db_path: Path | None = None
    thread_dir: Path | None = None
    models_file: Path | None = None
    mcp_manifest_path: Path | None = None
    state_snapshot_path: Path | None = None

    # ── Runtime ─────────────────────────────────────────────────────────
    debug: bool = False
    log_level: str = "info"
    log_path: Path | None = None
    host_log_path: Path | None = None
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
    thread_max_entries: int = _DEFAULT_THREAD_MAX_ENTRIES
    memory_max_records: int = _DEFAULT_MEMORY_MAX_RECORDS

    # ── Surface identity ───────────────────────────────────────────────
    surface_type: str = "personal_dm"
    surface_id: str = "host-console"
    memory_scope: str = "personal"

    # ── Pack policy ────────────────────────────────────────────────────
    always_on_packs: tuple[str, ...] = (
        "lisa-pack-core",
        "lisa-pack-learning",
        "lisa-pack-search",
    )

    @property
    def resolved_memory_db_path(self) -> Path:
        return self.memory_db_path or (self.data_dir / "memory.json")

    @property
    def resolved_thread_dir(self) -> Path:
        return self.thread_dir or (self.data_dir / "threads")

    @property
    def resolved_log_path(self) -> Path | None:
        return self.log_path

    @property
    def resolved_host_log_path(self) -> Path:
        return self.host_log_path or (self.data_dir / "host.log")

    @property
    def effective_log_level(self) -> str:
        """Return the log level, falling back to debug when debug is on.

        If ``log_level`` is explicitly set to a valid value, that wins.
        Otherwise, ``debug=True`` implies ``"debug"``; ``debug=False``
        defaults to ``"info"``.
        """
        from hosts.core.config_resolver import normalize_log_level

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
        memory_db: str = "",
        thread_dir: str = "",
        no_thread: bool = False,
        timezone: str = "",
        session_context: str = "",
        user_slug: str = "",
        state_snapshot: str = "",
        models_file: str = "",
        mcp_manifest_path: str = "",
        streaming: str = "none",
        surface_type: str = "",
        surface_id: str = "",
    ) -> HostConfig:
        """Build a ``HostConfig`` from CLI args + env vars.

        CLI args take precedence over env vars; env vars take precedence
        over defaults. This is the primary constructor for all hosts.
        """
        # ── Timezone ──────────────────────────────────────────────────
        from hosts.core.config_resolver import resolve_timezone

        resolved_tz, tz_source = resolve_timezone(timezone or _resolve_env(LISA_TIMEZONE_ENV))

        # ── Log level ─────────────────────────────────────────────────
        # When debug=True and no explicit log_level is passed, leave it
        # empty so effective_log_level falls back to "debug".
        raw_log_level = log_level.strip()
        if raw_log_level:
            resolved_log_level = raw_log_level
        else:
            env_log = _resolve_env(LISA_LOG_LEVEL_ENV)
            resolved_log_level = env_log if env_log else ""

        # ── Data dir ──────────────────────────────────────────────────
        resolved_data_dir = Path(
            data_dir.strip() or _resolve_env(LISA_DATA_DIR_ENV) or str(Path.cwd() / "artifacts")
        ).expanduser().resolve()

        # ── Paths ──────────────────────────────────────────────────────
        resolved_memory_db: Path | None = None
        raw_memory_db = memory_db.strip() or _resolve_env(LISA_MEMORY_DB_ENV)
        if raw_memory_db:
            resolved_memory_db = Path(raw_memory_db).expanduser().resolve()

        resolved_thread_dir: Path | None = None
        raw_thread_dir = thread_dir.strip() or _resolve_env(LISA_THREAD_DIR_ENV)
        if raw_thread_dir:
            resolved_thread_dir = Path(raw_thread_dir).expanduser().resolve()

        resolved_log_path: Path | None = None
        raw_log_path = log_path.strip()
        if raw_log_path:
            resolved_log_path = Path(raw_log_path).expanduser().resolve()

        resolved_host_log_path: Path | None = None
        raw_host_log = host_log_path.strip()
        if raw_host_log:
            resolved_host_log_path = Path(raw_host_log).expanduser().resolve()

        resolved_models_file: Path | None = None
        raw_models = models_file.strip() or _resolve_env(LISA_MODELS_FILE_ENV)
        if raw_models:
            resolved_models_file = Path(raw_models).expanduser().resolve()

        resolved_mcp_manifest: Path | None = None
        raw_mcp = mcp_manifest_path.strip()
        if raw_mcp:
            resolved_mcp_manifest = Path(raw_mcp).expanduser().resolve()

        resolved_snapshot: Path | None = None
        raw_snapshot = state_snapshot.strip()
        if raw_snapshot:
            resolved_snapshot = Path(raw_snapshot).expanduser().resolve()

        # ── Streaming ──────────────────────────────────────────────────
        from hosts.core.config_resolver import parse_streaming_option

        stream_mode, stream_delay = parse_streaming_option(streaming)

        # ── Slice ──────────────────────────────────────────────────────
        pre = _normalize_slice_flag(_resolve_env(LISA_SLICE_PRE_TURN_ENV))
        post = _normalize_slice_flag(_resolve_env(LISA_SLICE_POST_TURN_ENV))
        explicit = _normalize_slice_flag(_resolve_env(LISA_SLICE_EXPLICIT_ENV))
        raw_budget = _resolve_env(LISA_SLICE_BUDGET_MS_ENV, "").strip()
        explicit_budget = _DEFAULT_SLICE_EXPLICIT_BUDGET_MS
        if raw_budget:
            try:
                explicit_budget = max(50, min(int(raw_budget), 30_000))
            except ValueError:
                pass

        # ── Limits ─────────────────────────────────────────────────────
        thread_max = _resolve_env_int(
            LISA_THREAD_MAX_ENTRIES_ENV, _DEFAULT_THREAD_MAX_ENTRIES
        )
        memory_max = _resolve_env_int(
            LISA_MEMORY_MAX_RECORDS_ENV, _DEFAULT_MEMORY_MAX_RECORDS
        )

        # ── User / session ─────────────────────────────────────────────
        resolved_user_slug = user_slug.strip()
        resolved_session_context = session_context.strip()

        # ── Surface ────────────────────────────────────────────────────
        resolved_surface_type = surface_type.strip() or "personal_dm"
        resolved_surface_id = surface_id.strip() or "host-console"

        return cls(
            data_dir=resolved_data_dir,
            memory_db_path=resolved_memory_db,
            thread_dir=resolved_thread_dir,
            models_file=resolved_models_file,
            mcp_manifest_path=resolved_mcp_manifest,
            state_snapshot_path=resolved_snapshot,
            debug=debug,
            log_level=resolved_log_level,
            log_path=resolved_log_path,
            host_log_path=resolved_host_log_path,
            timezone=resolved_tz,
            timezone_source=tz_source,
            user_slug=resolved_user_slug,
            session_context=resolved_session_context,
            no_thread=no_thread,
            streaming=StreamingConfig(mode=stream_mode, char_delay_ms=stream_delay),
            slice_config=SliceConfig(
                pre_turn_enabled=pre,
                post_turn_enabled=post,
                explicit_enabled=explicit,
                explicit_budget_ms=explicit_budget,
            ),
            thread_max_entries=thread_max,
            memory_max_records=memory_max,
            surface_type=resolved_surface_type,
            surface_id=resolved_surface_id,
        )