"""Pure configuration-resolution helpers for ``hosts.core``.

These functions are deterministic mappings from user input (CLI flags / env
vars) to normalised values with no side effects. They are extracted from
``hosts/little_lisa/config.py`` and generalised to use the new
``LISA_*`` env var names (with legacy ``LITTLE_LISA_*`` fallback handled
in ``HostConfig.from_env``).
"""

from __future__ import annotations

import logging
import os
import platform
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

logger = logging.getLogger(__name__)

# ── Lookup tables ─────────────────────────────────────────────────────

_LOG_LEVEL_ORDER: dict[str, int] = {
    "debug": 10,
    "info": 20,
    "warning": 30,
    "error": 40,
}

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
    "India Standard Time": "Asia/Kolkata",
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
    "Magadan Standard Time": "Asia/Magadan",
    "Vladivostok Standard Time": "Asia/Vladivostok",
    "Russia Time Zone 11": "Asia/Kamchatka",
    "Central Pacific Standard Time": "Pacific/Guadalcanal",
    "New Zealand Standard Time": "Pacific/Auckland",
    "Fiji Standard Time": "Pacific/Fiji",
    "Kamchatka Standard Time": "Asia/Kamchatka",
    "Chatham Islands Standard Time": "Pacific/Chatham",
    "Tonga Standard Time": "Pacific/Tongatapu",
    "Samoa Standard Time": "Pacific/Samoa",
    "Line Islands Standard Time": "Pacific/Kiritimati",
}

# ── Timezone resolution ──────────────────────────────────────────────


def _resolve_timezone_alias(value: str) -> str | None:
    """Resolve a timezone alias (EST, PST, etc.) to a canonical IANA key."""
    posix_key = _POSIX_ALIAS_MAP.get(value.upper())
    if posix_key is None:
        return None
    try:
        zi = ZoneInfo(posix_key)
        return str(zi)
    except ZoneInfoNotFoundError:
        return posix_key


def _normalize_timezone_name(value: str) -> str | None:
    """Validate and normalise a timezone string to an IANA key.

    Returns ``None`` if *value* is not a recognised timezone. Accepts:
    - IANA keys containing ``/`` (e.g. ``America/New_York``)
    - The canonical ``UTC`` constant
    - POSIX aliases resolved via :func:`_resolve_timezone_alias`

    Note: ``/``-containing values are accepted without a ZoneInfo lookup
    because the target runtime may have different tzdata availability.
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


def resolve_timezone(cli_timezone: str) -> tuple[str, str]:
    """Resolve the runtime timezone with explicit precedence.

    Precedence (first wins):
    1. ``--timezone`` CLI flag
    2. ``LISA_TIMEZONE`` env var
    3. OS-specific env-var candidates (``TZ``, ``TIMEZONE``, etc.)
    4. System-local tzinfo (with Windows display-name mapping)
    5. ``UTC`` fallback

    Returns ``(iana_key, source_label)``.
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
    cli_raw = cli_timezone.strip()
    if not cli_raw:
        return None
    cli_value = _normalize_timezone_name(cli_raw)
    if cli_value is None:
        raise ValueError(
            f"Timezone '{cli_raw}' from --timezone is not resolvable. "
            "Use an IANA timezone such as 'UTC' or 'America/New_York'."
        )
    return cli_value, "cli"


def _resolve_explicit_env_timezone() -> tuple[str, str] | None:
    raw = os.getenv("LISA_TIMEZONE", "").strip()
    if not raw:
        return None
    normalized = _normalize_timezone_name(raw)
    if normalized is None:
        raise ValueError(
            f"Timezone '{raw}' from LISA_TIMEZONE is not resolvable. "
            "Use an IANA timezone such as 'UTC' or 'America/New_York'."
        )
    return normalized, "env:LISA_TIMEZONE"


def _resolve_env_timezone(env_name: str) -> tuple[str, str] | None:
    env_raw = os.getenv(env_name, "").strip()
    if not env_raw:
        return None
    env_value = _normalize_timezone_name(env_raw)
    if env_value is None:
        raise ValueError(
            f"Timezone '{env_raw}' from {env_name} is not resolvable. "
            "Use an IANA timezone such as 'UTC' or 'America/New_York'."
        )
    return env_value, f"os-env:{env_name}"


def _resolve_system_local_timezone() -> tuple[str, str]:
    local_tz = datetime.now().astimezone().tzinfo
    if local_tz is not None:
        local_name = str(local_tz).strip()
        normalized_local = _normalize_timezone_name(local_name)
        if normalized_local is None:
            normalized_local = _resolve_windows_tz_display_name(local_name)
        if normalized_local is not None:
            return normalized_local, "system-local"
    return "UTC", "default"


def _resolve_windows_tz_display_name(local_name: str) -> str | None:
    windows_iana = _WINDOWS_TZ_DISPLAY_TO_IANA.get(local_name)
    if windows_iana is None and local_name.endswith(" Daylight Time"):
        windows_iana = _WINDOWS_TZ_DISPLAY_TO_IANA.get(
            local_name.replace(" Daylight Time", " Standard Time")
        )
    return windows_iana


# ── Log-level resolution ──────────────────────────────────────────────


def normalize_log_level(value: str) -> str | None:
    """Normalise a log-level string to a canonical name.

    Returns one of ``"debug"``, ``"info"``, ``"warning"``, ``"error"`` or
    ``None`` if *value* is not recognised. ``"warn"`` is accepted as an
    alias for ``"warning"``.
    """
    normalized = value.strip().lower()
    if normalized == "warn":
        normalized = "warning"
    if normalized in _LOG_LEVEL_ORDER:
        return normalized
    return None


def resolve_log_level(cli_log_level: str, *, debug_enabled: bool) -> str:
    """Resolve the effective log level with CLI > env > default precedence."""
    from hosts.core.config import LISA_LOG_LEVEL_ENV, _resolve_env

    cli_value = normalize_log_level(cli_log_level)
    if cli_value is not None:
        return cli_value

    env_value = normalize_log_level(_resolve_env(LISA_LOG_LEVEL_ENV))
    if env_value is not None:
        return env_value

    return "debug" if debug_enabled else "info"


# ── Streaming-mode resolution ──────────────────────────────────────────


def resolve_stream_enabled(*, force_stream: bool, force_no_stream: bool, is_tty: bool) -> bool:
    """Resolve whether streaming should be active for this runtime."""
    if force_stream:
        return True
    if force_no_stream:
        return False
    return is_tty


def parse_streaming_option(raw_value: str) -> tuple[str, int]:
    """Parse a ``--streaming`` option value into ``(mode, char_delay_ms)``.

    Accepts ``none``, ``stage``, ``text``, ``hybrid``, optionally suffixed
    with ``:N`` for per-character delay in milliseconds (e.g. ``hybrid:50``).
    Invalid values default to ``("none", 0)``. An empty value defaults to
    ``("hybrid", 0)`` — the CLI host's default is hybrid streaming (the
    thinking/streaming pane).
    """
    cleaned = (raw_value or "").strip().lower()
    if not cleaned:
        return "hybrid", 0

    parts = cleaned.split(":", 1)
    mode = parts[0]
    delay = 0

    if len(parts) == 2:
        try:
            delay = max(0, int(parts[1]))
        except ValueError:
            pass

    if mode not in {"none", "stage", "text", "hybrid"}:
        return "none", 0

    return mode, delay
