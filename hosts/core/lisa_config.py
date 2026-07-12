"""Configuration resolution for the host.

This module centralises all *pure* configuration resolution logic: timezone
normalisation, log-level resolution, streaming-mode
parsing, and slice-tuning env-var reading.  Each function is a deterministic
mapping from user input (CLI flags / env vars) to a normalised value with no
side effects on the global ``_RUNTIME_STATE`` dict.

Keeping these resolvers separate from ``main.py`` makes the precedence rules
(CLI > env > default) explicit and individually testable, and it shrinks the
main orchestration file by ~300 lines.

Dependency direction
--------------------
``lisa_config.py`` ← ``main.py``  (main imports from config; config never imports main)

Public surface (re-exported by ``main.py`` for backward compatibility)
----------------------------------------------------------------------
- :func:`parse_streaming_option`
- :func:`resolve_stream_enabled`
- :func:`context_slug_to_thread_name`
"""

from __future__ import annotations

import logging
import os
import platform
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

logger = logging.getLogger(__name__)

# ── Environment-variable name constants ──────────────────────────────
# Centralised here so config functions and main.py share a single source
# of truth.  Main.py re-imports the names it needs for _initialize_runtime.

LISA_TIMEZONE_ENV = "LISA_TIMEZONE"
LISA_LOG_LEVEL_ENV = "LISA_LOG_LEVEL"
LISA_SLICE_PRE_TURN_ENV = "LISA_SLICE_PRE_TURN"
LISA_SLICE_POST_TURN_ENV = "LISA_SLICE_POST_TURN"
LISA_SLICE_EXPLICIT_ENV = "LISA_SLICE_EXPLICIT"
LISA_SLICE_BUDGET_MS_ENV = "LISA_SLICE_BUDGET_MS"
LISA_THREAD_MAX_ENTRIES_ENV = "LISA_THREAD_MAX_ENTRIES"
LISA_MEMORY_MAX_RECORDS_ENV = "LISA_MEMORY_MAX_RECORDS"

# ── Default values ────────────────────────────────────────────────────

_DEFAULT_THREAD_MAX_ENTRIES = 50
_DEFAULT_MEMORY_MAX_RECORDS = 200
_THREAD_DIGEST_FILE_NAME = "thread_digest.md"

# ── Lookup tables ─────────────────────────────────────────────────────

#: Numeric ordering used by ``_runtime_should_log`` to filter events.
_LOG_LEVEL_ORDER: dict[str, int] = {
    "debug": 10,
    "info": 20,
    "warning": 30,
    "error": 40,
}

#: POSIX-style zoneinfo keys for common US abbreviations.  These correctly
#: handle both standard and daylight time without a hard-coded alias table.
_POSIX_ALIAS_MAP: dict[str, str] = {
    "EST": "EST5EDT",
    "EDT": "EST5EDT",
    "CST": "CST6CDT",
    "CDT": "CST6CDT",
    "MST": "MST7MDT",
    "MDT": "MST7MDT",
    "PST": "PST8PDT",
    "PDT": "PST8PDT",
}

#: Per-OS env-var candidates checked when resolving the system-local
#: timezone.  The first non-empty, resolvable value wins.
_OS_TIMEZONE_ENV_CANDIDATES: dict[str, tuple[str, ...]] = {
    "Windows": (
        "TZ",
        "TIMEZONE",
        "LOCAL_TIMEZONE",
        "APPSETTING_WEBSITE_TIME_ZONE",
    ),
    "Linux": (
        "TZ",
        "TIMEZONE",
        "LOCAL_TIMEZONE",
    ),
    "Darwin": (
        "TZ",
        "TIMEZONE",
        "LOCAL_TIMEZONE",
        "SYSTEM_TIMEZONE",
    ),
}

#: Windows reports local tzinfo display names (e.g. "Eastern Daylight Time")
#: that are not IANA keys and are rejected by ``_normalize_timezone_name``.
#: Map the common Windows zone display names to canonical IANA keys so the
#: system-local fallback resolves correctly instead of silently defaulting
#: to UTC.
_WINDOWS_TZ_DISPLAY_TO_IANA: dict[str, str] = {
    "Dateline Standard Time": "Etc/GMT+12",
    "UTC-11": "Etc/GMT+11",
    "Aleutian Standard Time": "America/Adak",
    "Hawaiian Standard Time": "Pacific/Honolulu",
    "Alaskan Standard Time": "America/Anchorage",
    "Pacific Standard Time": "America/Los_Angeles",
    "Pacific Standard Time (Mexico)": "America/Tijuana",
    "Mountain Standard Time": "America/Denver",
    "Mountain Standard Time (Mexico)": "America/Mazatlan",
    "US Mountain Standard Time": "America/Phoenix",
    "Central Standard Time": "America/Chicago",
    "Central Standard Time (Mexico)": "America/Mexico_City",
    "Canada Central Standard Time": "America/Regina",
    "Central America Standard Time": "America/Guatemala",
    "Eastern Standard Time": "America/New_York",
    "Eastern Standard Time (Mexico)": "America/Cancun",
    "US Eastern Standard Time": "America/Indiana/Indianapolis",
    "Paraguay Standard Time": "America/Asuncion",
    "Atlantic Standard Time": "America/Halifax",
    "Venezuela Standard Time": "America/Caracas",
    "SA Pacific Standard Time": "America/Bogota",
    "Greenland Standard Time": "America/Godthab",
    "Montevideo Standard Time": "America/Montevideo",
    "Bahia Standard Time": "America/Bahia",
    "Argentina Standard Time": "America/Argentina/Buenos_Aires",
    "SA Eastern Standard Time": "America/Cayenne",
    "Magallanes Standard Time": "America/Punta_Arenas",
    "Saint Pierre Standard Time": "America/Miquelon",
    "Newfoundland Standard Time": "America/St_Johns",
    "Azores Standard Time": "Atlantic/Azores",
    "Cape Verde Standard Time": "Atlantic/Cape_Verde",
    "GMT Standard Time": "Europe/London",
    "Greenwich Standard Time": "Africa/Monrovia",
    "W. Europe Standard Time": "Europe/Berlin",
    "Central Europe Standard Time": "Europe/Budapest",
    "Romance Standard Time": "Europe/Paris",
    "Morocco Standard Time": "Africa/Casablanca",
    "Sao Tome Standard Time": "Africa/Sao_Tome",
    "Central European Standard Time": "Europe/Warsaw",
    "GTB Standard Time": "Europe/Bucharest",
    "Syria Standard Time": "Asia/Damascus",
    "E. Europe Standard Time": "Europe/Chisinau",
    "Turkey Standard Time": "Europe/Istanbul",
    "Israel Standard Time": "Asia/Jerusalem",
    "Jordan Standard Time": "Asia/Amman",
    "Arabic Standard Time": "Asia/Baghdad",
    "Arab Standard Time": "Asia/Riyadh",
    "Russian Standard Time": "Europe/Kaliningrad",
    "E. Africa Standard Time": "Africa/Nairobi",
    "Iran Standard Time": "Asia/Tehran",
    "Arabian Standard Time": "Asia/Dubai",
    "Azerbaijan Standard Time": "Asia/Baku",
    "Mauritius Standard Time": "Indian/Mauritius",
    "Georgian Standard Time": "Asia/Tbilisi",
    "Caucasus Standard Time": "Asia/Yerevan",
    "Afghanistan Standard Time": "Asia/Kabul",
    "West Asia Standard Time": "Asia/Tashkent",
    "Pakistan Standard Time": "Asia/Karachi",
    "India Standard Time": "Asia/Calcutta",
    "Sri Lanka Standard Time": "Asia/Colombo",
    "Nepal Standard Time": "Asia/Kathmandu",
    "Central Asia Standard Time": "Asia/Almaty",
    "Bangladesh Standard Time": "Asia/Dhaka",
    "N. Central Asia Standard Time": "Asia/Novosibirsk",
    "Myanmar Standard Time": "Asia/Yangon",
    "SE Asia Standard Time": "Asia/Bangkok",
    "North Asia Standard Time": "Asia/Krasnoyarsk",
    "China Standard Time": "Asia/Shanghai",
    "North Asia East Standard Time": "Asia/Irkutsk",
    "Singapore Standard Time": "Asia/Singapore",
    "W. Australia Standard Time": "Australia/Perth",
    "Taipei Standard Time": "Asia/Taipei",
    "Ulaanbaatar Standard Time": "Asia/Ulaanbaatar",
    "Tokyo Standard Time": "Asia/Tokyo",
    "Korea Standard Time": "Asia/Seoul",
    "Yakutsk Standard Time": "Asia/Yakutsk",
    "Cen. Australia Standard Time": "Australia/Adelaide",
    "AUS Central Standard Time": "Australia/Darwin",
    "E. Australia Standard Time": "Australia/Brisbane",
    "AUS Eastern Standard Time": "Australia/Sydney",
    "West Pacific Standard Time": "Pacific/Port_Moresby",
    "Tasmania Standard Time": "Australia/Hobart",
    "Vladivostok Standard Time": "Asia/Vladivostok",
    "Magadan Standard Time": "Asia/Magadan",
    "Central Pacific Standard Time": "Pacific/Guadalcanal",
    "New Zealand Standard Time": "Pacific/Auckland",
    "Fiji Standard Time": "Pacific/Fiji",
    "Kamchatka Standard Time": "Asia/Kamchatka",
    "Chatham Islands Standard Time": "Pacific/Chatham",
    "Tonga Standard Time": "Pacific/Tongatapu",
    "Samoa Standard Time": "Pacific/Apia",
    "Line Islands Standard Time": "Pacific/Kiritimati",
    "UTC": "UTC",
    "Coordinated Universal Time": "UTC",
}


# ── Timezone resolution ──────────────────────────────────────────────


def _resolve_timezone_alias(value: str) -> str | None:
    """Resolve a timezone alias to a canonical IANA key using stdlib zoneinfo.

    Common US abbreviations (EST, EDT, CST, CDT, MST, MDT, PST, PDT) are mapped
    to POSIX-style zoneinfo keys (EST5EDT, CST6CDT, etc.) that correctly handle
    both standard and daylight time without maintaining a hard-coded alias
    table.
    """
    posix_key = _POSIX_ALIAS_MAP.get(value.upper())
    if posix_key is None:
        return None
    try:
        zi = ZoneInfo(posix_key)
        return str(zi)
    except ZoneInfoNotFoundError:
        # Windows/Python envs may not ship tzdata for POSIX keys; keep alias stable.
        logger.debug("POSIX timezone alias %r not found in zoneinfo; using key directly", posix_key)
        return posix_key


def _normalize_timezone_name(value: str) -> str | None:
    """Validate and normalise a timezone string to an IANA key.

    Returns ``None`` if *value* is not a recognised timezone.  Accepts:
    - IANA keys containing ``/`` (e.g. ``America/New_York``)
    - The canonical ``UTC`` constant
    - POSIX aliases resolved via :func:`_resolve_timezone_alias`
    """
    configured = value.strip()
    if not configured:
        return None
    resolved_alias = _resolve_timezone_alias(configured)
    if resolved_alias is not None:
        return resolved_alias
    if configured == "UTC":
        return configured
    if "/" not in configured:
        return None
    if any(char.isspace() for char in configured):
        return None
    return configured


def _resolve_timezone(cli_timezone: str) -> tuple[str, str]:
    """Resolve the runtime timezone with explicit precedence.

    Precedence (first wins):
    1. ``--timezone`` CLI flag
    2. ``LISA_TIMEZONE`` env var
    3. OS-specific env-var candidates (``TZ``, ``TIMEZONE``, etc.)
    4. System-local tzinfo (with Windows display-name mapping)
    5. ``UTC`` fallback

    Returns ``(iana_key, source_label)`` where *source_label* identifies
    which precedence tier supplied the value (e.g. ``"cli"``, ``"env:..."``,
    ``"system-local"``, ``"default"``).

    Raises :class:`ValueError` if a CLI or env value is present but not
    resolvable to a valid IANA key.
    """
    cli_result = _resolve_cli_timezone(cli_timezone)
    if cli_result is not None:
        return cli_result

    env_result = _resolve_explicit_env_timezone()
    if env_result is not None:
        return env_result

    for env_name in _OS_TIMEZONE_ENV_CANDIDATES.get(platform.system(), ()):
        os_result = _resolve_env_timezone(env_name)
        if os_result is not None:
            return os_result

    return _resolve_system_local_timezone()


def _resolve_cli_timezone(cli_timezone: str) -> tuple[str, str] | None:
    """Resolve timezone from CLI flag; return (iana, 'cli') or None."""
    cli_raw = cli_timezone.strip()
    if not cli_raw:
        return None
    cli_value = _normalize_timezone_name(cli_raw)
    if cli_value is None:
        logger.warning("unresolvable --timezone value %r", cli_raw)
        raise ValueError(
            f"Timezone '{cli_raw}' from --timezone is not resolvable. "
            "Use an IANA timezone such as 'UTC' or 'America/New_York'."
        )
    logger.debug("timezone resolved from CLI: %s", cli_value)
    return cli_value, "cli"


def _resolve_explicit_env_timezone() -> tuple[str, str] | None:
    """Resolve timezone from LISA_TIMEZONE env var."""
    explicit_env_raw = os.getenv(LISA_TIMEZONE_ENV, "").strip()
    if not explicit_env_raw:
        return None
    explicit_env = _normalize_timezone_name(explicit_env_raw)
    if explicit_env is None:
        logger.warning("unresolvable %s value %r", LISA_TIMEZONE_ENV, explicit_env_raw)
        raise ValueError(
            f"Timezone '{explicit_env_raw}' from "
            f"{LISA_TIMEZONE_ENV} is not resolvable. "
            "Use an IANA timezone such as 'UTC' or 'America/New_York'."
        )
    logger.debug("timezone resolved from %s: %s", LISA_TIMEZONE_ENV, explicit_env)
    return explicit_env, f"env:{LISA_TIMEZONE_ENV}"


def _resolve_env_timezone(env_name: str) -> tuple[str, str] | None:
    """Resolve a timezone from a single OS env var; return (iana, source) or None."""
    env_raw = os.getenv(env_name, "").strip()
    if not env_raw:
        return None
    env_value = _normalize_timezone_name(env_raw)
    if env_value is None:
        logger.warning("unresolvable OS env timezone %s=%r", env_name, env_raw)
        raise ValueError(
            f"Timezone '{env_raw}' from {env_name} is not resolvable. "
            "Use an IANA timezone such as 'UTC' or 'America/New_York'."
        )
    logger.debug("timezone resolved from OS env %s: %s", env_name, env_value)
    return env_value, f"os-env:{env_name}"


def _resolve_system_local_timezone() -> tuple[str, str]:
    """Resolve timezone from system-local tzinfo, falling back to UTC."""
    local_tz = datetime.now().astimezone().tzinfo
    if local_tz is not None:
        local_name = str(local_tz).strip()
        normalized_local = _normalize_timezone_name(local_name)
        if normalized_local is None:
            normalized_local = _resolve_windows_tz_display_name(local_name)
        if normalized_local is not None:
            logger.debug("timezone resolved from system-local: %s", normalized_local)
            return normalized_local, "system-local"
        logger.debug("system-local timezone unresolvable; falling back to UTC")
    return "UTC", "default"


def _resolve_windows_tz_display_name(local_name: str) -> str | None:
    """Map a Windows display-name timezone to an IANA key."""
    windows_iana = _WINDOWS_TZ_DISPLAY_TO_IANA.get(local_name)
    if windows_iana is None and local_name.endswith(" Daylight Time"):
        windows_iana = _WINDOWS_TZ_DISPLAY_TO_IANA.get(
            local_name.replace(" Daylight Time", " Standard Time")
        )
    return windows_iana


# ── Log-level resolution ──────────────────────────────────────────────


def _normalize_log_level(value: str) -> str | None:
    """Normalise a log-level string to a canonical name.

    Returns one of ``"debug"``, ``"info"``, ``"warning"``, ``"error"`` or
    ``None`` if *value* is not recognised.  ``"warn"`` is accepted as an
    alias for ``"warning"``.
    """
    normalized = value.strip().lower()
    if normalized == "warn":
        normalized = "warning"
    if normalized in _LOG_LEVEL_ORDER:
        return normalized
    logger.debug("unrecognised log-level value %r", value)
    return None


def _resolve_log_level(cli_log_level: str, *, debug_enabled: bool) -> str:
    """Resolve the effective log level with CLI > env > default precedence.

    If ``--debug`` is set and no explicit level is chosen, defaults to
    ``"debug"``; otherwise defaults to ``"info"``.
    """
    cli_value = _normalize_log_level(cli_log_level)
    if cli_value is not None:
        logger.debug("log level resolved from CLI: %s", cli_value)
        return cli_value

    env_value = _normalize_log_level(os.getenv(LISA_LOG_LEVEL_ENV, ""))
    if env_value is not None:
        logger.debug("log level resolved from %s: %s", LISA_LOG_LEVEL_ENV, env_value)
        return env_value

    if debug_enabled:
        logger.debug("log level defaulting to debug because --debug is set")
        return "debug"
    return "info"


# ── Streaming-mode resolution ──────────────────────────────────────────


def resolve_stream_enabled(*, force_stream: bool, force_no_stream: bool, is_tty: bool) -> bool:
    """Resolve whether terminal streaming should be active for this runtime.

    When neither flag forces a decision, streaming auto-enables on TTYs.
    """
    if force_stream:
        return True
    if force_no_stream:
        return False
    return is_tty


def parse_streaming_option(raw_value: str) -> tuple[str, int]:
    """Parse one ``--streaming`` value into ``(mode, char_delay_ms)``.

    Accepted forms:
    - ``"none"`` / ``"stage"`` / ``"text"`` / ``"hybrid"``  → delay 0
    - ``"hybrid:50"`` → mode ``"hybrid"``, delay ``50`` ms

    Raises :class:`ValueError` for unknown modes or negative / non-integer
    delay suffixes.
    """
    value = raw_value.strip().lower()
    if not value:
        return "none", 0

    if ":" in value:
        mode, delay_raw = value.split(":", 1)
        mode = mode.strip()
        delay_raw = delay_raw.strip()
        if mode not in {"none", "stage", "text", "hybrid"}:
            raise ValueError("--streaming mode must be one of: none, stage, text, hybrid")
        if not delay_raw:
            raise ValueError("--streaming delay suffix must be a non-negative integer")
        try:
            delay_ms = int(delay_raw)
        except ValueError as exc:
            raise ValueError("--streaming delay suffix must be a non-negative integer") from exc
        if delay_ms < 0:
            raise ValueError("--streaming delay suffix must be a non-negative integer")
        return mode, delay_ms

    if value not in {"none", "stage", "text", "hybrid"}:
        raise ValueError("--streaming mode must be one of: none, stage, text, hybrid")
    return value, 0


# ── Slice-tuning resolution ────────────────────────────────────────────


def _normalize_slice_flag(value: str) -> bool | None:
    """Parse a boolean-ish env var value for the slice trigger toggles.

    Accepts ``1/true/yes/on`` (case-insensitive) as truthy, ``0/false/no/off``
    as falsy, and empty string as ``None`` (i.e. let the default win).
    """
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    if cleaned in {"1", "true", "yes", "on"}:
        return True
    if cleaned in {"0", "false", "no", "off"}:
        return False
    logger.debug("unrecognised slice flag value %r", value)
    return None


def _resolve_slice_tuning() -> tuple[dict[str, object], int]:
    """Resolve the host slice tuning from env vars.

    Returns a tuple of ``(tuning_dict, explicit_budget_ms)``. The tuning dict
    feeds directly into :meth:`SliceTuning.from_mapping`. The explicit
    budget is the wall-clock ceiling for the explicit-slice escape hatch
    (e.g. the ``/consolidate`` command); pre/post slices use the pressure
    curve and ignore this number.
    """
    tuning: dict[str, object] = {}
    pre = _normalize_slice_flag(os.getenv(LISA_SLICE_PRE_TURN_ENV, ""))
    if pre is not None:
        tuning["pre_turn_enabled"] = pre
    post = _normalize_slice_flag(os.getenv(LISA_SLICE_POST_TURN_ENV, ""))
    if post is not None:
        tuning["post_turn_enabled"] = post
    explicit = _normalize_slice_flag(os.getenv(LISA_SLICE_EXPLICIT_ENV, ""))
    if explicit is not None:
        tuning["explicit_enabled"] = explicit
    raw_budget = os.getenv(LISA_SLICE_BUDGET_MS_ENV, "").strip()
    explicit_budget_ms = 2000  # safe default for an explicit user-initiated slice
    if raw_budget:
        try:
            explicit_budget_ms = max(50, min(int(raw_budget), 30_000))
        except ValueError:
            logger.warning(
                "invalid %s value %r; using default %d ms",
                LISA_SLICE_BUDGET_MS_ENV,
                raw_budget,
                explicit_budget_ms,
            )
    return tuning, explicit_budget_ms


# ── Session-context helpers ───────────────────────────────────────────


def context_slug_to_thread_name(slug: str) -> str:
    """Convert a path or slug to a decent thread name if no front matter is present."""
    from pathlib import Path  # noqa: PLC0415 — local import to keep module lightweight

    p = Path(slug)
    name = p.stem
    if name.lower() == "session_context":
        return "session_context"
    return name