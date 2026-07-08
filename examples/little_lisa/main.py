"""Little Lisa a local terminal host for exercising Lisa framework packs without the private app.

Little Lisa wires public content and operation catalogs, uses local tool callbacks
and live model completion hooks, and stores generated Markdown artifacts on disk.
It is useful for pack authors who want to verify routing and operation shape
without Teams, Azure, or production Lisa services.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import re
import sys
import time
from datetime import UTC, datetime
from functools import lru_cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version
from pathlib import Path
from typing import Annotated, Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
import typer
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AsyncOpenAI
from pydantic import ValidationError
from rich.cells import cell_len
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.history import FileHistory
except ImportError:  # pragma: no cover - optional dependency for richer REPL UX
    PromptSession = None
    WordCompleter = None
    FileHistory = None

FRAMEWORK_ROOT = Path(__file__).resolve().parents[2]
if str(FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_ROOT))

_LITTLE_LISA_ENV_FILE = Path(__file__).resolve().parent / ".env"


def _load_local_env_file() -> None:
    """Load the Little Lisa local .env without overriding explicit process env vars."""
    load_dotenv(dotenv_path=_LITTLE_LISA_ENV_FILE, override=False)


# Load environment variables before kernel imports so Settings() sees them.
_load_local_env_file()

from kernel import frontdoor as shared_frontdoor  # noqa: E402
from kernel.content_roots import (  # noqa: E402
    discover_configured_content_roots,
    resolve_content_subdir_roots,
)
from kernel.ipsar import parse_route  # noqa: E402
from kernel.ipsar_fsm import (  # noqa: E402
    create_workflow_session,
    get_workflow_catalog,
    parse_front_matter,
    register_content_roots,
    register_operation_callbacks,
)
from kernel.mcp_client import MCPMemoryClient  # noqa: E402
from kernel.model_manifest import ModelManifest  # noqa: E402
from kernel.models import (  # noqa: E402
    MemoryRecallAnswer,
    MemoryRecallEvidence,
    MemoryRecord,
    MemoryWriteRequest,
    OperationExecutionContext,
    OperationResult,
    RouteDecision,
    SurfaceContext,
    ToolDefinition,
    ToolResult,
    UnifiedState,
    WorkflowModeConfig,
)
from kernel.operations.registry import (  # noqa: E402
    OperationExecutionError,
    execute_operation,
    get_operation_catalog,
    register_operation_roots,
    register_tool_catalog_callback,
)
from kernel.slice import SliceOutcome  # noqa: E402
from kernel.token_usage import (  # noqa: E402
    TokenUsageTelemetryHandler,
    build_token_usage_event,
)
from kernel.turn_runner import TurnHooks, TurnResult, run_turn  # noqa: E402
from lisa_sdk.lisa_search.web_search import is_configured as web_search_is_configured  # noqa: E402
from lisa_sdk.lisa_search.web_search import search as web_search  # noqa: E402

try:  # noqa: E402
    from .model_manifest import load_model_manifest
    from .model_runtime import (
        StubCompletion,
        StubModelDefinition,
    )
    from .model_runtime import (
        load_manifest as _load_manifest_runtime,
    )
    from .model_runtime import (
        manifest_path_from_env as _manifest_path_from_env_runtime,
    )
    from .model_runtime import (
        reload_manifest as _reload_manifest_runtime,
    )
    from .model_runtime import (
        resolve_little_lisa_model as _resolve_little_lisa_model_runtime,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from model_manifest import load_model_manifest  # type: ignore[no-redef]
    from model_runtime import (  # type: ignore[no-redef]
        StubCompletion,
        StubModelDefinition,
    )
    from model_runtime import (
        load_manifest as _load_manifest_runtime,
    )
    from model_runtime import (
        manifest_path_from_env as _manifest_path_from_env_runtime,
    )
    from model_runtime import (
        reload_manifest as _reload_manifest_runtime,
    )
    from model_runtime import (
        resolve_little_lisa_model as _resolve_little_lisa_model_runtime,
    )

try:  # noqa: E402
    from .desktop import (
        DEFAULT_DESKTOP_HOST,
        DEFAULT_DESKTOP_PORT,
        DesktopCallbacks,
        generate_desktop_token,
        run_desktop_server,
    )
    from .tray import TrayConfig, run_tray
except ImportError:  # pragma: no cover - direct script execution fallback
    from desktop import (  # type: ignore[no-redef]
        DEFAULT_DESKTOP_HOST,
        DEFAULT_DESKTOP_PORT,
        DesktopCallbacks,
        generate_desktop_token,
        run_desktop_server,
    )
    from tray import TrayConfig, run_tray  # type: ignore[no-redef]

CORE_PACK_ROOT = FRAMEWORK_ROOT / "sdk_packs" / "core"

_DEFAULT_DATA_DIR = FRAMEWORK_ROOT / "examples" / "little_lisa" / "artifacts"
_DEFAULT_GOOGLE_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_DEFAULT_TURN_LOG_NAME = "turns.jsonl"

_LITTLE_LISA_MANIFEST_ENV = "LITTLE_LISA_MODELS_FILE"
_LITTLE_LISA_DATA_DIR_ENV = "LITTLE_LISA_DATA_DIR"
_LITTLE_LISA_MEMORY_DB_ENV = "LITTLE_LISA_MEMORY_DB"
_LITTLE_LISA_THREAD_DIR_ENV = "LITTLE_LISA_THREAD_DIR"
_LITTLE_LISA_AZURE_API_VERSION_ENV = "LITTLE_LISA_AZURE_OPENAI_API_VERSION"
_LITTLE_LISA_OPENAI_API_KEY_ENV = "LITTLE_LISA_OPENAI_API_KEY"
_LITTLE_LISA_AZURE_API_KEY_ENV = "LITTLE_LISA_AZURE_OPENAI_API_KEY"
_LITTLE_LISA_AZURE_ENDPOINT_ENV = "LITTLE_LISA_AZURE_OPENAI_ENDPOINT"
_LITTLE_LISA_ANTHROPIC_API_KEY_ENV = "LITTLE_LISA_ANTHROPIC_API_KEY"
_LITTLE_LISA_ANTHROPIC_BASE_URL_ENV = "LITTLE_LISA_ANTHROPIC_BASE_URL"
_LITTLE_LISA_ANTHROPIC_VERSION_ENV = "LITTLE_LISA_ANTHROPIC_VERSION"
_LITTLE_LISA_OLLAMA_HOST_ENV = "LITTLE_LISA_OLLAMA_HOST"
_LITTLE_LISA_OLLAMA_API_KEY_ENV = "LITTLE_LISA_OLLAMA_API_KEY"
_LITTLE_LISA_GOOGLE_API_KEY_ENV = "LITTLE_LISA_GOOGLE_API_KEY"
_LITTLE_LISA_GOOGLE_API_BASE_URL_ENV = "LITTLE_LISA_GOOGLE_API_BASE_URL"
_LITTLE_LISA_TIMEOUT_PROFILE_ENV = "LITTLE_LISA_TIMEOUT_PROFILE"
_LITTLE_LISA_TIMEZONE_ENV = "LITTLE_LISA_TIMEZONE"
_LITTLE_LISA_LOG_LEVEL_ENV = "LITTLE_LISA_LOG_LEVEL"
_LITTLE_LISA_SLICE_PRE_TURN_ENV = "LITTLE_LISA_SLICE_PRE_TURN"
_LITTLE_LISA_SLICE_POST_TURN_ENV = "LITTLE_LISA_SLICE_POST_TURN"
_LITTLE_LISA_SLICE_EXPLICIT_ENV = "LITTLE_LISA_SLICE_EXPLICIT"
_LITTLE_LISA_SLICE_BUDGET_MS_ENV = "LITTLE_LISA_SLICE_BUDGET_MS"
_LITTLE_LISA_THREAD_MAX_ENTRIES_ENV = "LITTLE_LISA_THREAD_MAX_ENTRIES"
_LITTLE_LISA_MEMORY_MAX_RECORDS_ENV = "LITTLE_LISA_MEMORY_MAX_RECORDS"
_DEFAULT_THREAD_MAX_ENTRIES = 50
_DEFAULT_MEMORY_MAX_RECORDS = 200
_THREAD_DIGEST_FILE_NAME = "thread_digest.md"
_LOG_LEVEL_ORDER = {
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

# Windows reports local tzinfo display names (e.g. "Eastern Daylight Time")
# that are not IANA keys and are rejected by _normalize_timezone_name. Map the
# common Windows zone display names to canonical IANA keys so the system-local
# fallback resolves correctly instead of silently defaulting to UTC.
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


def _resolve_timezone_alias(value: str) -> str | None:
    """Resolve a timezone alias to a canonical IANA key using stdlib zoneinfo.

    Common US abbreviations (EST, EDT, CST, CDT, MST, MDT, PST, PDT) are mapped
    to POSIX-style zoneinfo keys (EST5EDT, CST6CDT, etc.) that correctly handle
    both standard and daylight time without maintaining a hard-coded alias table.
    """
    posix_key = _POSIX_ALIAS_MAP.get(value.upper())
    if posix_key is None:
        return None
    try:
        zi = ZoneInfo(posix_key)
        return str(zi)
    except ZoneInfoNotFoundError:
        return None


_RUNTIME_STATE: dict[str, Any] = {
    "manifest": None,
    "azure_client": None,
    "openai_client": None,
    "anthropic_http_client": None,
    "ollama_client": None,
    "google_client": None,
    "debug": False,
    "log_level": "info",
    "log_path": None,
    "last_turn_context": None,
    "last_frontdoor_debug": None,
    "user_slug": "",
    "session_context_ref": "",
    "session_context_active": False,
    "timeout_profile": "balanced",
    "timezone": "UTC",
    "timezone_source": "default",
    "current_mode_id": "default",
    "current_stage_id": "-",
    "next_model_id": "-",
    "next_model_provider": "-",
    "turn_prompt_tokens": 0,
    "turn_completion_tokens": 0,
    "turn_total_tokens": 0,
    "turn_cached_tokens": 0,
    "turn_reasoning_tokens": 0,
    "turn_duration_ms": 0.0,
    "turn_latency_ms": 0,
    "pending_turn_prompt_tokens": 0,
    "pending_turn_completion_tokens": 0,
    "pending_turn_total_tokens": 0,
    "session_prompt_tokens": 0,
    "session_completion_tokens": 0,
    "session_total_tokens": 0,
    "session_cached_tokens": 0,
    "session_reasoning_tokens": 0,
    "session_duration_ms": 0.0,
    "session_latency_ms": 0,
    "session_token_ledger": [],
    "model_token_totals": {},
    "mode_token_totals": {},
    "stage_token_totals": {},
    "session_started_utc": None,
    "session_turn_count": 0,
    "session_success_count": 0,
    "session_error_count": 0,
    "session_total_latency_ms": 0.0,
    "session_mode_counts": {},
    "thread_path": None,
    "state_snapshot_path": None,
    "state_snapshot": None,
    "streaming_mode": "none",
    "streaming_char_delay_ms": 0,
    "stream_enabled": False,
    "stream_mode": "hybrid",
    "stream_char_delay_ms": 0,
}

console = Console()
app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    help="Run Little Lisa.",
)


def _get_version() -> str:
    try:
        return _package_version("little-lisa")
    except PackageNotFoundError:
        return "0.1.0"


_VERSION_OPTION = typer.Option(
    False,
    "--version",
    help="Print the Little Lisa version and exit.",
    is_eager=True,
)

# ---------------------------------------------------------------------------
# Shared Typer option aliases
# ---------------------------------------------------------------------------
# These Annotated type aliases define each runtime option exactly once so every
# subcommand shares identical flag names, defaults, and help text. Add a new
# runtime knob here and it propagates to all commands automatically.

DebugOption = Annotated[
    bool,
    typer.Option(
        "--debug",
        help="Print turn context above each response.",
    ),
]
LogLevelOption = Annotated[
    str,
    typer.Option(
        "--log-level",
        help=("Log threshold: debug, info, warning, or error. Overrides LITTLE_LISA_LOG_LEVEL."),
    ),
]
LogOption = Annotated[
    str,
    typer.Option(
        "--log",
        help="Append JSONL request/response turn logs to this file.",
    ),
]
NoLogOption = Annotated[
    bool,
    typer.Option(
        "--no-log",
        help="Disable turn logging entirely.",
    ),
]
DataDirOption = Annotated[
    str,
    typer.Option(
        "--data-dir",
        help=(
            "Data root for artifacts, memory, and logs. "
            "Overrides LITTLE_LISA_DATA_DIR. "
            "Default: examples/little_lisa/artifacts/"
        ),
    ),
]
MemoryDbOption = Annotated[
    str,
    typer.Option(
        "--memory-db",
        help=(
            "Memory database JSON file path. "
            "Overrides LITTLE_LISA_MEMORY_DB. "
            "Default: <data-dir>/memory.json"
        ),
    ),
]
ThreadDirOption = Annotated[
    str,
    typer.Option(
        "--thread-dir",
        help=(
            "Thread log directory. Overrides LITTLE_LISA_THREAD_DIR. "
            "Defaults to <data-dir>/threads when a session context is active."
        ),
    ),
]
NoThreadOption = Annotated[
    bool,
    typer.Option(
        "--no-thread",
        help="Disable local turn history thread logging.",
    ),
]
SessionContextOption = Annotated[
    str,
    typer.Option(
        "--session-context",
        help=(
            "User context source: a slug name (content/user_contexts/<name>.md), "
            "a full .md file path, or a directory (all files concatenated). "
            "Omit to skip session context."
        ),
    ),
]
TimeoutProfileOption = Annotated[
    str,
    typer.Option(
        "--timeout-profile",
        help=(
            "Operation timeout profile: fast, balanced, or deep. "
            "Overrides LITTLE_LISA_TIMEOUT_PROFILE."
        ),
    ),
]
TimezoneOption = Annotated[
    str,
    typer.Option(
        "--timezone",
        help=(
            "IANA timezone (e.g. UTC, America/New_York). "
            "Overrides LITTLE_LISA_TIMEZONE, then OS timezone env vars."
        ),
    ),
]
StateSnapshotOption = Annotated[
    str,
    typer.Option(
        "--state-snapshot",
        help="Local JSON snapshot file to inject compact context from.",
    ),
]
StreamingOption = Annotated[
    str,
    typer.Option(
        "--streaming",
        help=(
            "Streaming mode: none, stage, text, hybrid. "
            "Optionally append ':<ms>' for per-character delay (e.g. hybrid:50)."
        ),
    ),
]
STYLE_TEXT_PRIMARY = "bright_white"
STYLE_TEXT_MUTED = "grey58"
STYLE_BORDER_SESSION = "steel_blue3"
STYLE_BORDER_RESPONSE = "dark_sea_green4"
STYLE_BORDER_CONTEXT = "gold3"
STYLE_RULE = "grey42"
STYLE_PROMPT = "bold steel_blue3"
STYLE_STATUS_INFO = "bright_white on steel_blue3"
STYLE_STATUS_METRIC = "bright_white on grey27"
STYLE_STATUS_DEBUG_ON = "black on dark_khaki"
STYLE_STATUS_DEBUG_OFF = "bright_white on grey35"
STYLE_STATUS_SEPARATOR = "grey42"
_LEGACY_ONCE_ARG_COUNT = 2
_RANKED_CONTEXT_MIN_TOKEN_LEN = 3
_STATUS_COMPACT_FULL = 0
_STATUS_COMPACT_ABBREVIATED = 1
_STATUS_COMPACT_TRIMMED = 2
_STATUS_COMPACT_MINIMAL = 3
_TIMEZONE_INITIALISM_MIN_LEN = 2
_STATUS_TIME_PADDING_WIDTH = 2
_REPL_HISTORY_FILE_NAME = "repl_history.txt"
_DEFAULT_DESKTOP_OPEN_ON_START = True
_DEFAULT_DESKTOP_NO_ARGS_MODE = "tray"
_REPL_COMMAND_HINTS: tuple[str, ...] = (
    "/help",
    "/help capabilities",
    "/settings",
    "/tokens",
    "/debug context",
    "/debug frontdoor",
    "/search ",
    "/markdown ",
    "/deep ",
    "/discuss ",
    "/quit",
)
_MODE_MODEL_ALIAS_MAP = {
    "default": "default",
    "discuss": "default",
    "search": "default",
    "markdown": "default",
    "deep": "deep",
}


def _unique_paths(paths: list[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _sdk_content_roots() -> list[Path]:
    roots = discover_configured_content_roots()
    return _unique_paths([root for root in roots if root.is_dir()])


def _authored_content_roots() -> list[Path]:
    roots = [CORE_PACK_ROOT, *_sdk_content_roots()]
    return _unique_paths([root / "content" for root in roots if (root / "content").is_dir()])


def _sdk_mode_roots() -> list[Path]:
    return _unique_paths(
        resolve_content_subdir_roots(_sdk_content_roots(), "modes")
    )


def _sdk_operation_roots() -> list[Path]:
    return _unique_paths(
        resolve_content_subdir_roots(_sdk_content_roots(), "operations")
    )


FrontDoorRouteDecision = RouteDecision


class LocalBlobIO:
    """File-backed subset of the host blob protocol used by simple examples."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def upload_markdown_doc(self, markdown_text: str, *, blob_name: str) -> str:
        """Persist one generated Markdown document and return a local path."""

        target_dir = self.root / "markdown"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / blob_name
        target.write_text(markdown_text, encoding="utf-8")
        return str(target)

    def read_unified_state(self) -> UnifiedState | None:
        return None

    def write_unified_state(self, state: UnifiedState) -> bool:
        payload = state.model_dump_json(indent=2)
        self.write_blob_text("state", "unified-state.json", payload)
        return True

    def read_blob_bytes(self, container: str, blob_path: str) -> bytes | None:
        path = self._resolve(container, blob_path)
        return path.read_bytes() if path.exists() else None

    def write_blob_bytes(
        self,
        container: str,
        blob_path: str,
        payload: bytes,
        content_type: str = "application/octet-stream",
    ) -> bool:
        _ = content_type
        path = self._resolve(container, blob_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return True

    def read_blob_text(self, container: str, blob_path: str) -> str | None:
        path = self._resolve(container, blob_path)
        return path.read_text(encoding="utf-8") if path.exists() else None

    def write_blob_text(
        self,
        container: str,
        blob_path: str,
        payload: str,
        content_type: str = "text/plain; charset=utf-8",
    ) -> bool:
        _ = content_type
        path = self._resolve(container, blob_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
        return True

    def _resolve(self, container: str, blob_path: str) -> Path:
        safe_parts = [part for part in Path(blob_path).parts if part not in {"..", ""}]
        return self.root / container / Path(*safe_parts)


def _tool_catalog() -> dict[str, ToolDefinition]:
    """Return stub tool definitions required to validate installed public operations."""

    return {
        "web_search": ToolDefinition(
            tool_id="web_search",
            description="Stub web search tool for Little Lisa.",
        ),
        "surf_browser": ToolDefinition(
            tool_id="surf_browser",
            description="Stub browser automation tool for Little Lisa.",
        ),
        "surf_session_flush": ToolDefinition(
            tool_id="surf_session_flush",
            description="Stub surf-session flush tool for Little Lisa.",
        ),
    }


def get_little_lisa_tool_catalog() -> dict[str, ToolDefinition]:
    """Return the Little Lisa tool catalog for diagnostics and harnesses."""

    return _tool_catalog()


def setup_framework() -> None:
    """Wire public content roots, operation roots, and stub tools."""

    register_content_roots(
        core_root=CORE_PACK_ROOT / "modes",
        sdk_roots=_sdk_mode_roots(),
    )
    register_operation_roots(
        primary_root=CORE_PACK_ROOT / "operations",
        sdk_roots=_sdk_operation_roots(),
    )
    register_tool_catalog_callback(_tool_catalog)
    register_operation_callbacks(get_operation_catalog, execute_operation)


def configure_runtime() -> None:
    """Configure runtime and reset provider clients."""

    _RUNTIME_STATE["azure_client"] = None
    _RUNTIME_STATE["openai_client"] = None
    _RUNTIME_STATE["anthropic_http_client"] = None
    _RUNTIME_STATE["ollama_client"] = None
    _RUNTIME_STATE["google_client"] = None


def _runtime_debug_enabled() -> bool:
    return bool(_RUNTIME_STATE.get("debug", False))


def _normalize_log_level(value: str) -> str | None:
    normalized = value.strip().lower()
    if normalized == "warn":
        normalized = "warning"
    if normalized in _LOG_LEVEL_ORDER:
        return normalized
    return None


def _runtime_log_level() -> str:
    configured = _normalize_log_level(str(_RUNTIME_STATE.get("log_level") or ""))
    if configured is not None:
        return configured
    return "debug" if _runtime_debug_enabled() else "info"


def _runtime_should_log(level: str) -> bool:
    normalized = _normalize_log_level(level) or "info"
    runtime_level = _runtime_log_level()
    return _LOG_LEVEL_ORDER[normalized] >= _LOG_LEVEL_ORDER[runtime_level]


def _runtime_log_path() -> Path | None:
    configured = _RUNTIME_STATE.get("log_path")
    return configured if isinstance(configured, Path) else None


def _runtime_token_usage_path() -> Path:
    """Return the durable ``token-usage.jsonl`` path under the data dir.

    The file lives at ``<data_dir>/token-usage.jsonl`` (alongside
    ``memory.json``) so it survives process restarts. The
    ``TokenUsageTelemetryHandler`` creates parent directories lazily
    on first write.
    """
    return _runtime_data_dir() / "token-usage.jsonl"


def _token_usage_handler() -> TokenUsageTelemetryHandler:
    """Return (and cache) the singleton ``TokenUsageTelemetryHandler``."""
    cached = _RUNTIME_STATE.get("token_usage_handler")
    if isinstance(cached, TokenUsageTelemetryHandler):
        return cached
    handler = TokenUsageTelemetryHandler(_runtime_token_usage_path())
    _RUNTIME_STATE["token_usage_handler"] = handler
    return handler


async def _publish_token_usage(
    *,
    provider: str,
    model_id: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    cached_tokens: int = 0,
    reasoning_tokens: int = 0,
    duration_ms: float = 0.0,
    latency_ms: int = 0,
    kind: str = "chat_completion",
    request_id: str = "",
    mode_id: str = "",
    stage_id: str = "",
    call_site: str = "",
) -> None:
    """Publish one ``TokenUsageEvent`` to the durable JSONL handler.

    Called from every LLM call site so each provider call produces
    exactly one durable record. Failures are swallowed inside the
    handler; this function never raises into the turn loop.
    """
    event = build_token_usage_event(
        provider=provider,
        model_id=model_id,
        kind=kind,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
        duration_ms=duration_ms,
        latency_ms=latency_ms,
        request_id=request_id,
        mode_id=mode_id,
        stage_id=stage_id,
        call_site=call_site,
        user_slug=_runtime_user_slug() or "default",
        surface_id="little-lisa-console",
    )
    try:
        await _token_usage_handler().publish(event)
    except Exception as exc:  # pylint: disable=broad-except
        # Telemetry must never break the turn loop.
        _append_log_event(
            level="warning",
            event="token_usage.publish_failed",
            payload={
                "provider": provider,
                "model_id": model_id,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )


def _runtime_user_slug() -> str:
    configured = str(_RUNTIME_STATE.get("user_slug") or "").strip()
    return configured


def _runtime_session_context_ref() -> str:
    configured = str(_RUNTIME_STATE.get("session_context_ref") or "").strip()
    return configured


def _runtime_next_model_provider() -> str:
    configured = str(_RUNTIME_STATE.get("next_model_provider") or "").strip()
    return configured or "-"


def _runtime_timeout_profile() -> str:
    configured = str(_RUNTIME_STATE.get("timeout_profile") or "").strip().lower()
    return configured if configured in {"fast", "balanced", "deep"} else "balanced"


def _runtime_timezone() -> str:
    configured = str(_RUNTIME_STATE.get("timezone") or "").strip()
    return configured or "UTC"


def _runtime_timezone_source() -> str:
    configured = str(_RUNTIME_STATE.get("timezone_source") or "").strip()
    return configured or "default"


def _runtime_current_mode_id() -> str:
    configured = str(_RUNTIME_STATE.get("current_mode_id") or "").strip()
    return configured or "default"


def _runtime_current_stage_id() -> str:
    configured = str(_RUNTIME_STATE.get("current_stage_id") or "").strip()
    return configured or "-"


def _runtime_next_model_id() -> str:
    configured = str(_RUNTIME_STATE.get("next_model_id") or "").strip()
    return configured or "-"


def resolve_stream_enabled(*, force_stream: bool, force_no_stream: bool, is_tty: bool) -> bool:
    """Resolve whether terminal streaming should be active for this runtime."""

    if force_stream:
        return True
    if force_no_stream:
        return False
    return is_tty


def _runtime_stream_enabled() -> bool:
    return _runtime_streaming_mode() != "none"


def _runtime_stream_mode() -> str:
    configured = _runtime_streaming_mode()
    return configured if configured in {"stage", "text", "hybrid"} else "hybrid"


def _runtime_stream_char_delay_ms() -> int:
    return _runtime_streaming_char_delay_ms()


def _runtime_streaming_mode() -> str:
    configured = str(_RUNTIME_STATE.get("streaming_mode") or "").strip().lower()
    return configured if configured in {"none", "stage", "text", "hybrid"} else "none"


def _runtime_streaming_char_delay_ms() -> int:
    return max(0, int(_RUNTIME_STATE.get("streaming_char_delay_ms", 0) or 0))


def parse_streaming_option(raw_value: str) -> tuple[str, int]:
    """Parse one --streaming value into mode and optional char delay."""

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


def _runtime_session_started_utc() -> datetime:
    configured = _RUNTIME_STATE.get("session_started_utc")
    return configured if isinstance(configured, datetime) else datetime.now(UTC)


def _record_session_turn(*, succeeded: bool, latency_ms: float, mode_id: str | None = None) -> None:
    _RUNTIME_STATE["session_turn_count"] = int(_RUNTIME_STATE.get("session_turn_count", 0)) + 1
    if succeeded:
        _RUNTIME_STATE["session_success_count"] = (
            int(_RUNTIME_STATE.get("session_success_count", 0)) + 1
        )
    else:
        _RUNTIME_STATE["session_error_count"] = (
            int(_RUNTIME_STATE.get("session_error_count", 0)) + 1
        )
    total_latency = float(_RUNTIME_STATE.get("session_total_latency_ms", 0.0))
    _RUNTIME_STATE["session_total_latency_ms"] = total_latency + max(0.0, latency_ms)

    selected_mode = (mode_id or "").strip()
    if selected_mode:
        mode_counts = dict(_RUNTIME_STATE.get("session_mode_counts") or {})
        mode_counts[selected_mode] = int(mode_counts.get(selected_mode, 0)) + 1
        _RUNTIME_STATE["session_mode_counts"] = mode_counts


def _apply_turn_usage(
    result: TurnResult,
    *,
    model_id: str = "",
    mode_id: str = "",
    stage_id: str = "",
    frontdoor_routing: dict[str, Any] | None = None,
) -> None:
    """Persist turn and session token counters from one completed turn result.

    Records a per-turn ledger entry with model/mode/stage attribution and
    accumulates breakdown totals keyed by resolved model id (never designation),
    mode id, and stage id. Sub-counters (``cached_tokens``,
    ``reasoning_tokens``), ``duration_ms``, and ``latency_ms`` flow through
    so time-based providers (e.g. Ollama local GPU time) can be billed
    alongside token-based providers.
    """

    # ── Input extraction ───────────────────────────────────────────────
    main_prompt = _coerce_non_negative_int(result.prompt_tokens)
    main_completion = _coerce_non_negative_int(result.completion_tokens)
    main_total = _coerce_non_negative_int(result.total_tokens)
    if main_total <= 0:
        main_total = main_prompt + main_completion
    main_cached = _coerce_non_negative_int(getattr(result, "cached_tokens", 0))
    main_reasoning = _coerce_non_negative_int(getattr(result, "reasoning_tokens", 0))
    main_duration_ms = _coerce_non_negative_float(getattr(result, "duration_ms", 0.0))
    main_latency_ms = _coerce_non_negative_int(getattr(result, "latency_ms", 0))

    fd_prompt = 0
    fd_completion = 0
    fd_total = 0
    fd_cached = 0
    fd_reasoning = 0
    fd_duration_ms = 0.0
    fd_latency_ms = 0
    fd_model_id = ""
    if isinstance(frontdoor_routing, dict):
        fd_prompt = _coerce_non_negative_int(frontdoor_routing.get("planner_prompt_tokens"))
        fd_completion = _coerce_non_negative_int(frontdoor_routing.get("planner_completion_tokens"))
        fd_total = _coerce_non_negative_int(frontdoor_routing.get("planner_total_tokens"))
        if fd_total <= 0:
            fd_total = fd_prompt + fd_completion
        fd_cached = _coerce_non_negative_int(frontdoor_routing.get("planner_cached_tokens"))
        fd_reasoning = _coerce_non_negative_int(frontdoor_routing.get("planner_reasoning_tokens"))
        fd_duration_ms = _coerce_non_negative_float(
            frontdoor_routing.get("planner_duration_ms")
        )
        fd_latency_ms = _coerce_non_negative_int(
            frontdoor_routing.get("planner_latency_ms")
        )
        fd_model_id = str(frontdoor_routing.get("resolved_model_id") or "").strip()

    # ── Turn totals ────────────────────────────────────────────────────
    turn_prompt = main_prompt + fd_prompt
    turn_completion = main_completion + fd_completion
    turn_total = main_total + fd_total
    turn_cached = main_cached + fd_cached
    turn_reasoning = main_reasoning + fd_reasoning
    turn_duration_ms = main_duration_ms + fd_duration_ms
    turn_latency_ms = main_latency_ms + fd_latency_ms

    _RUNTIME_STATE["turn_prompt_tokens"] = turn_prompt
    _RUNTIME_STATE["turn_completion_tokens"] = turn_completion
    _RUNTIME_STATE["turn_total_tokens"] = turn_total
    _RUNTIME_STATE["turn_cached_tokens"] = turn_cached
    _RUNTIME_STATE["turn_reasoning_tokens"] = turn_reasoning
    _RUNTIME_STATE["turn_duration_ms"] = turn_duration_ms
    _RUNTIME_STATE["turn_latency_ms"] = turn_latency_ms

    _RUNTIME_STATE["session_prompt_tokens"] = (
        _coerce_non_negative_int(_RUNTIME_STATE.get("session_prompt_tokens", 0)) + turn_prompt
    )
    _RUNTIME_STATE["session_completion_tokens"] = (
        _coerce_non_negative_int(_RUNTIME_STATE.get("session_completion_tokens", 0))
        + turn_completion
    )
    _RUNTIME_STATE["session_total_tokens"] = (
        _coerce_non_negative_int(_RUNTIME_STATE.get("session_total_tokens", 0)) + turn_total
    )
    _RUNTIME_STATE["session_cached_tokens"] = (
        _coerce_non_negative_int(_RUNTIME_STATE.get("session_cached_tokens", 0)) + turn_cached
    )
    _RUNTIME_STATE["session_reasoning_tokens"] = (
        _coerce_non_negative_int(_RUNTIME_STATE.get("session_reasoning_tokens", 0))
        + turn_reasoning
    )
    _RUNTIME_STATE["session_duration_ms"] = (
        _coerce_non_negative_float(_RUNTIME_STATE.get("session_duration_ms", 0.0))
        + turn_duration_ms
    )
    _RUNTIME_STATE["session_latency_ms"] = (
        _coerce_non_negative_int(_RUNTIME_STATE.get("session_latency_ms", 0)) + turn_latency_ms
    )

    ledger: list[dict[str, Any]] = _RUNTIME_STATE.setdefault("session_token_ledger", [])
    turn_index = len(ledger) + 1

    # ── Front-door ledger and breakdowns (if applicable) ───────────────
    if fd_total > 0 or fd_duration_ms > 0:
        # Record front-door call first in the ledger
        ledger.append(
            {
                "turn": turn_index,
                "model_id": fd_model_id or "front-door",
                "mode_id": "front-door",
                "stage_id": "routing",
                "prompt_tokens": fd_prompt,
                "completion_tokens": fd_completion,
                "total_tokens": fd_total,
                "cached_tokens": fd_cached,
                "reasoning_tokens": fd_reasoning,
                "duration_ms": fd_duration_ms,
                "latency_ms": fd_latency_ms,
            }
        )

        model_key = fd_model_id or "front-door"
        if model_key != "-":
            model_totals: dict[str, dict[str, Any]] = _RUNTIME_STATE.setdefault(
                "model_token_totals", {}
            )
            model_entry = _ensure_usage_entry(
                model_totals.setdefault(
                    model_key,
                    {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "cached_tokens": 0,
                        "reasoning_tokens": 0,
                        "duration_ms": 0.0,
                        "latency_ms": 0,
                    },
                )
            )
            model_entry["prompt_tokens"] += fd_prompt
            model_entry["completion_tokens"] += fd_completion
            model_entry["total_tokens"] += fd_total
            model_entry["cached_tokens"] += fd_cached
            model_entry["reasoning_tokens"] += fd_reasoning
            model_entry["duration_ms"] += fd_duration_ms
            model_entry["latency_ms"] += fd_latency_ms

        mode_totals: dict[str, dict[str, Any]] = _RUNTIME_STATE.setdefault("mode_token_totals", {})
        mode_entry = _ensure_usage_entry(
            mode_totals.setdefault(
                "front-door",
                {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cached_tokens": 0,
                    "reasoning_tokens": 0,
                    "duration_ms": 0.0,
                    "latency_ms": 0,
                },
            )
        )
        mode_entry["prompt_tokens"] += fd_prompt
        mode_entry["completion_tokens"] += fd_completion
        mode_entry["total_tokens"] += fd_total
        mode_entry["cached_tokens"] += fd_cached
        mode_entry["reasoning_tokens"] += fd_reasoning
        mode_entry["duration_ms"] += fd_duration_ms
        mode_entry["latency_ms"] += fd_latency_ms

    # ── Main execution ledger entry ────────────────────────────────────
    ledger.append(
        {
            "turn": turn_index,
            "model_id": (model_id or "-").strip(),
            "mode_id": (mode_id or "-").strip(),
            "stage_id": (stage_id or "-").strip(),
            "prompt_tokens": main_prompt,
            "completion_tokens": main_completion,
            "total_tokens": main_total,
            "cached_tokens": main_cached,
            "reasoning_tokens": main_reasoning,
            "duration_ms": main_duration_ms,
            "latency_ms": main_latency_ms,
        }
    )

    # ── Per-model breakdown ────────────────────────────────────────────
    model_key = (model_id or "-").strip()
    model_totals = _RUNTIME_STATE.setdefault("model_token_totals", {})
    model_entry = _ensure_usage_entry(
        model_totals.setdefault(
            model_key,
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cached_tokens": 0,
                "reasoning_tokens": 0,
                "duration_ms": 0.0,
                "latency_ms": 0,
            },
        )
    )
    model_entry["prompt_tokens"] += main_prompt
    model_entry["completion_tokens"] += main_completion
    model_entry["total_tokens"] += main_total
    model_entry["cached_tokens"] += main_cached
    model_entry["reasoning_tokens"] += main_reasoning
    model_entry["duration_ms"] += main_duration_ms
    model_entry["latency_ms"] += main_latency_ms

    # ── Per-mode breakdown ─────────────────────────────────────────────
    mode_key = (mode_id or "-").strip()
    mode_totals = _RUNTIME_STATE.setdefault("mode_token_totals", {})
    mode_entry = _ensure_usage_entry(
        mode_totals.setdefault(
            mode_key,
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cached_tokens": 0,
                "reasoning_tokens": 0,
                "duration_ms": 0.0,
                "latency_ms": 0,
            },
        )
    )
    mode_entry["prompt_tokens"] += main_prompt
    mode_entry["completion_tokens"] += main_completion
    mode_entry["total_tokens"] += main_total
    mode_entry["cached_tokens"] += main_cached
    mode_entry["reasoning_tokens"] += main_reasoning
    mode_entry["duration_ms"] += main_duration_ms
    mode_entry["latency_ms"] += main_latency_ms

    # ── Per-stage breakdown ────────────────────────────────────────────
    stage_key = (stage_id or "-").strip()
    stage_totals: dict[str, dict[str, Any]] = _RUNTIME_STATE.setdefault("stage_token_totals", {})
    stage_entry = _ensure_usage_entry(
        stage_totals.setdefault(
            stage_key,
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cached_tokens": 0,
                "reasoning_tokens": 0,
                "duration_ms": 0.0,
                "latency_ms": 0,
            },
        )
    )
    stage_entry["prompt_tokens"] += main_prompt
    stage_entry["completion_tokens"] += main_completion
    stage_entry["total_tokens"] += main_total
    stage_entry["cached_tokens"] += main_cached
    stage_entry["reasoning_tokens"] += main_reasoning
    stage_entry["duration_ms"] += main_duration_ms
    stage_entry["latency_ms"] += main_latency_ms


def _top_session_mode() -> tuple[str, int] | None:
    mode_counts = _RUNTIME_STATE.get("session_mode_counts")
    if not isinstance(mode_counts, dict) or not mode_counts:
        return None
    best_mode = ""
    best_count = 0
    for mode, count in mode_counts.items():
        mode_name = str(mode).strip()
        if not mode_name:
            continue
        count_value = int(count)
        if count_value > best_count:
            best_mode = mode_name
            best_count = count_value
    if not best_mode:
        return None
    return best_mode, best_count


def _format_session_exit_summary_lines() -> list[str]:
    now_utc = datetime.now(UTC)
    started_utc = _runtime_session_started_utc()
    duration_seconds = max(0.0, (now_utc - started_utc).total_seconds())

    turn_count = int(_RUNTIME_STATE.get("session_turn_count", 0))
    success_count = int(_RUNTIME_STATE.get("session_success_count", 0))
    error_count = int(_RUNTIME_STATE.get("session_error_count", 0))
    total_latency_ms = float(_RUNTIME_STATE.get("session_total_latency_ms", 0.0))
    avg_latency_ms = (total_latency_ms / turn_count) if turn_count else 0.0

    session_prompt = int(_RUNTIME_STATE.get("session_prompt_tokens", 0) or 0)
    session_completion = int(_RUNTIME_STATE.get("session_completion_tokens", 0) or 0)
    session_total = int(_RUNTIME_STATE.get("session_total_tokens", 0) or 0)
    session_cached = int(_RUNTIME_STATE.get("session_cached_tokens", 0) or 0)
    session_reasoning = int(_RUNTIME_STATE.get("session_reasoning_tokens", 0) or 0)
    session_duration_ms = float(_RUNTIME_STATE.get("session_duration_ms", 0.0) or 0.0)
    turn_prompt = int(_RUNTIME_STATE.get("turn_prompt_tokens", 0) or 0)
    turn_completion = int(_RUNTIME_STATE.get("turn_completion_tokens", 0) or 0)
    turn_total = int(_RUNTIME_STATE.get("turn_total_tokens", 0) or 0)
    turn_cached = int(_RUNTIME_STATE.get("turn_cached_tokens", 0) or 0)
    turn_reasoning = int(_RUNTIME_STATE.get("turn_reasoning_tokens", 0) or 0)
    turn_duration_ms = float(_RUNTIME_STATE.get("turn_duration_ms", 0.0) or 0.0)

    avg_prompt = (session_prompt / turn_count) if turn_count else 0.0
    avg_completion = (session_completion / turn_count) if turn_count else 0.0
    avg_total_tokens = (session_total / turn_count) if turn_count else 0.0
    avg_duration_ms = (session_duration_ms / turn_count) if turn_count else 0.0

    lines = [
        "## Session Summary",
        f"- Duration: {duration_seconds:.1f}s",
        f"- Turns: {turn_count} (ok: {success_count}, errors: {error_count})",
        f"- Avg turn latency: {avg_latency_ms:.0f} ms",
        f"- Provider time: {session_duration_ms:.0f} ms (avg {avg_duration_ms:.0f} ms/turn)",
        "",
        "### Token Totals",
        "| Category | Prompt | Completion | Total | Cached | Reasoning |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| Session | {session_prompt} | {session_completion} | {session_total} | "
        f"{session_cached} | {session_reasoning} |",
        f"| Last turn | {turn_prompt} | {turn_completion} | {turn_total} | "
        f"{turn_cached} | {turn_reasoning} |",
        f"| Avg/turn | {avg_prompt:.0f} | {avg_completion:.0f} | {avg_total_tokens:.0f} | "
        f"{(session_cached / turn_count) if turn_count else 0:.0f} | "
        f"{(session_reasoning / turn_count) if turn_count else 0:.0f} |",
    ]

    # ── Per-model breakdown ────────────────────────────────────────────
    model_totals: dict[str, dict[str, Any]] = _RUNTIME_STATE.get("model_token_totals") or {}
    if model_totals:
        lines.extend(
            [
                "",
                "### By Model",
                "| Model | Prompt | Completion | Total | Cached | Reasoning | Duration ms |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for model_key in sorted(model_totals):
            entry = model_totals[model_key]
            lines.append(
                f"| {model_key} | {entry.get('prompt_tokens', 0)} | "
                f"{entry.get('completion_tokens', 0)} | {entry.get('total_tokens', 0)} | "
                f"{entry.get('cached_tokens', 0)} | {entry.get('reasoning_tokens', 0)} | "
                f"{entry.get('duration_ms', 0.0):.0f} |"
            )

    # ── Per-mode breakdown ─────────────────────────────────────────────
    mode_totals: dict[str, dict[str, Any]] = _RUNTIME_STATE.get("mode_token_totals") or {}
    if mode_totals:
        lines.extend(
            [
                "",
                "### By Mode",
                "| Mode | Prompt | Completion | Total | Cached | Reasoning | Duration ms |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for mode_key in sorted(mode_totals):
            entry = mode_totals[mode_key]
            lines.append(
                f"| {mode_key} | {entry.get('prompt_tokens', 0)} | "
                f"{entry.get('completion_tokens', 0)} | {entry.get('total_tokens', 0)} | "
                f"{entry.get('cached_tokens', 0)} | {entry.get('reasoning_tokens', 0)} | "
                f"{entry.get('duration_ms', 0.0):.0f} |"
            )

    # ── Per-stage breakdown ────────────────────────────────────────────
    stage_totals: dict[str, dict[str, Any]] = _RUNTIME_STATE.get("stage_token_totals") or {}
    if stage_totals:
        lines.extend(
            [
                "",
                "### By Stage",
                "| Stage | Prompt | Completion | Total | Cached | Reasoning | Duration ms |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for stage_key in sorted(stage_totals):
            entry = stage_totals[stage_key]
            lines.append(
                f"| {stage_key} | {entry.get('prompt_tokens', 0)} | "
                f"{entry.get('completion_tokens', 0)} | {entry.get('total_tokens', 0)} | "
                f"{entry.get('cached_tokens', 0)} | {entry.get('reasoning_tokens', 0)} | "
                f"{entry.get('duration_ms', 0.0):.0f} |"
            )

    top_mode = _top_session_mode()
    if top_mode is not None:
        lines.append(f"- Most used mode: {top_mode[0]} ({top_mode[1]})")
    return lines


def _render_session_exit_summary() -> None:
    summary = "\n".join(_format_session_exit_summary_lines())
    console.print(
        Panel(
            Markdown(summary),
            title="Session",
            border_style=STYLE_BORDER_SESSION,
        )
    )


def _format_token_stats() -> str:
    """Return on-demand detailed token statistics as Markdown."""
    return "\n".join(_format_session_exit_summary_lines())


def _predict_model(preferred_model_id: str) -> tuple[str, str]:
    try:
        model = _resolve_model(preferred_model_id)
        return model.model_id, model.provider
    except Exception:  # pylint: disable=broad-except
        return preferred_model_id, "unknown"


def _set_runtime_route_preview(*, mode_id: str, preferred_model_id: str) -> None:
    catalog = get_workflow_catalog()
    mode = catalog.get(mode_id)
    stage_id = "-"
    if mode is not None and mode.mode_kind == "workflow":
        stage_id = str(mode.start_stage_id).strip() or "-"
    model_id, provider = _predict_model(preferred_model_id)
    _RUNTIME_STATE["current_mode_id"] = mode_id
    _RUNTIME_STATE["current_stage_id"] = stage_id or "-"
    _RUNTIME_STATE["next_model_id"] = model_id
    _RUNTIME_STATE["next_model_provider"] = provider


def _resolve_session_context(session_context: str) -> tuple[str, str, bool]:
    """
    Resolve the session context source and return (thread_name, source_path, is_active).

    Resolution order:
    1. If empty: check for 'session_context.md' or 'session_context/' in current dir.
    2. Literal path: if it's a file or directory.
    3. artifacts/ suffix: if 'artifacts/{session_context}.md' or 'artifacts/{session_context}/' exists.
    4. user_contexts/ slug: if 'user_contexts/{session_context}.md' exists.
    """
    s = session_context.strip()

    # 1. auto-discovery
    if not s:
        # Check current directory
        p = Path("session_context.md")
        if p.is_file():
            tn = _extract_front_matter_thread_name(p) or "session_context"
            return tn, str(p.absolute()), True
        p = Path("session_context")
        if p.is_dir():
            return "session_context", str(p.absolute()), True

        # Check data directory
        dd = _runtime_data_dir()
        p = dd / "session_context.md"
        if p.is_file():
            tn = _extract_front_matter_thread_name(p) or "session_context"
            return tn, str(p.absolute()), True
        p = dd / "session_context"
        if p.is_dir():
            return "session_context", str(p.absolute()), True

        return "", "", False

    # 2. literal path
    p = Path(s).expanduser()
    if p.exists():
        if p.is_dir():
            return "session_context", str(p.absolute()), True
        tn = _extract_front_matter_thread_name(p) or context_slug_to_thread_name(s)
        return tn, str(p.absolute()), True

    # 3. artifacts/ relative
    dd = _runtime_data_dir()
    p = dd / f"{s}.md"
    if p.is_file():
        tn = _extract_front_matter_thread_name(p) or context_slug_to_thread_name(s)
        return tn, str(p.absolute()), True
    p = dd / s
    if p.is_dir():
        return "session_context", str(p.absolute()), True

    # special case for session_context/ or .md specifically in artifacts
    if s == "session_context":
        p = dd / "session_context.md"
        if p.is_file():
            tn = _extract_front_matter_thread_name(p) or "session_context"
            return tn, str(p.absolute()), True
        p = dd / "session_context"
        if p.is_dir():
            return "session_context", str(p.absolute()), True

    # 4. user_contexts/ slug (legacy)
    context_name = s
    if s.lower().endswith(".md"):
        context_name = s[:-3]

    relative_path = f"user_contexts/{context_name}.md"
    try:
        # Check if the text is authorable (we don't want to use absolute paths here)
        _read_authored_text(relative_path)
        return context_name, relative_path, False
    except FileNotFoundError:
        pass

    raise ValueError(f"Session context '{s}' was not found.")


def context_slug_to_thread_name(slug: str) -> str:
    """Convert a path or slug to a decent thread name if no front matter is present."""
    p = Path(slug)
    name = p.stem
    if name.lower() == "session_context":
        return "session_context"
    return name


def _normalize_timeout_profile(value: str) -> str | None:
    normalized = value.strip().lower()
    if normalized in {"fast", "balanced", "deep"}:
        return normalized
    return None


def _resolve_timeout_profile(cli_timeout_profile: str) -> str:
    cli_value = _normalize_timeout_profile(cli_timeout_profile)
    if cli_value is not None:
        return cli_value
    env_value = _normalize_timeout_profile(os.getenv(_LITTLE_LISA_TIMEOUT_PROFILE_ENV, ""))
    if env_value is not None:
        return env_value
    return "balanced"


def _resolve_log_level(cli_log_level: str, *, debug_enabled: bool) -> str:
    cli_value = _normalize_log_level(cli_log_level)
    if cli_value is not None:
        return cli_value

    env_value = _normalize_log_level(os.getenv(_LITTLE_LISA_LOG_LEVEL_ENV, ""))
    if env_value is not None:
        return env_value

    if debug_enabled:
        return "debug"
    return "info"


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
    pre = _normalize_slice_flag(os.getenv(_LITTLE_LISA_SLICE_PRE_TURN_ENV, ""))
    if pre is not None:
        tuning["pre_turn_enabled"] = pre
    post = _normalize_slice_flag(os.getenv(_LITTLE_LISA_SLICE_POST_TURN_ENV, ""))
    if post is not None:
        tuning["post_turn_enabled"] = post
    explicit = _normalize_slice_flag(os.getenv(_LITTLE_LISA_SLICE_EXPLICIT_ENV, ""))
    if explicit is not None:
        tuning["explicit_enabled"] = explicit
    raw_budget = os.getenv(_LITTLE_LISA_SLICE_BUDGET_MS_ENV, "").strip()
    explicit_budget_ms = 2000  # safe default for an explicit user-initiated slice
    if raw_budget:
        try:
            explicit_budget_ms = max(50, min(int(raw_budget), 30_000))
        except ValueError:
            pass
    return tuning, explicit_budget_ms


def _normalize_timezone_name(value: str) -> str | None:
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
    cli_raw = cli_timezone.strip()
    if cli_raw:
        cli_value = _normalize_timezone_name(cli_raw)
        if cli_value is None:
            raise ValueError(
                f"Timezone '{cli_raw}' from --timezone is not resolvable. "
                "Use an IANA timezone such as 'UTC' or 'America/New_York'."
            )
        return cli_value, "cli"

    explicit_env_raw = os.getenv(_LITTLE_LISA_TIMEZONE_ENV, "").strip()
    if explicit_env_raw:
        explicit_env = _normalize_timezone_name(explicit_env_raw)
        if explicit_env is None:
            raise ValueError(
                f"Timezone '{explicit_env_raw}' from "
                f"{_LITTLE_LISA_TIMEZONE_ENV} is not resolvable. "
                "Use an IANA timezone such as 'UTC' or 'America/New_York'."
            )
        return explicit_env, f"env:{_LITTLE_LISA_TIMEZONE_ENV}"

    os_candidates = _OS_TIMEZONE_ENV_CANDIDATES.get(platform.system(), ())
    for env_name in os_candidates:
        env_raw = os.getenv(env_name, "").strip()
        if not env_raw:
            continue
        env_value = _normalize_timezone_name(env_raw)
        if env_value is None:
            raise ValueError(
                f"Timezone '{env_raw}' from {env_name} is not resolvable. "
                "Use an IANA timezone such as 'UTC' or 'America/New_York'."
            )
        return env_value, f"os-env:{env_name}"

    local_tz = datetime.now().astimezone().tzinfo
    if local_tz is not None:
        local_name = str(local_tz).strip()
        normalized_local = _normalize_timezone_name(local_name)
        if normalized_local is None:
            # Windows reports tzinfo display names (e.g. "Eastern Daylight Time")
            # that are not IANA keys. The registry uses "Standard Time" suffixes,
            # but the runtime string swaps in "Daylight Time" during DST, so try
            # both the raw name and the standard-time form before falling back.
            windows_iana = _WINDOWS_TZ_DISPLAY_TO_IANA.get(local_name)
            if windows_iana is None and local_name.endswith(" Daylight Time"):
                windows_iana = _WINDOWS_TZ_DISPLAY_TO_IANA.get(
                    local_name.replace(" Daylight Time", " Standard Time")
                )
            if windows_iana is not None:
                normalized_local = windows_iana
        if normalized_local is not None:
            return normalized_local, "system-local"

    return "UTC", "default"


def _load_selected_session_context_text() -> str:
    context_ref = _runtime_session_context_ref()
    if not context_ref:
        return ""
    current_mode = _runtime_current_mode_id()
    if bool(_RUNTIME_STATE.get("session_context_active", False)):
        path = Path(context_ref)
        if path.is_dir():
            parts: list[str] = []
            for child in sorted(path.iterdir()):
                if child.is_file():
                    try:
                        text = child.read_text(encoding="utf-8")
                    except Exception:
                        continue
                    front_matter, body = parse_front_matter(text)
                    applies_to = front_matter.get("applies_to")
                    if applies_to is not None:
                        if not isinstance(applies_to, list):
                            applies_to = [applies_to]
                        if current_mode not in applies_to:
                            continue
                    parts.append(body)
            return "\n\n".join(parts)
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            return ""
        front_matter, body = parse_front_matter(text)
        applies_to = front_matter.get("applies_to")
        if applies_to is not None:
            if not isinstance(applies_to, list):
                applies_to = [applies_to]
            if current_mode not in applies_to:
                return ""
        return body
    # For authored (packaged) context we do not filter by front-matter;
    # pack authors control scope via pack selection instead.
    return _read_authored_text(context_ref)


def _runtime_data_dir() -> Path:
    configured = _RUNTIME_STATE.get("data_dir")
    return configured if isinstance(configured, Path) else _DEFAULT_DATA_DIR


def _runtime_memory_db_path() -> Path:
    configured = _RUNTIME_STATE.get("memory_db_path")
    return configured if isinstance(configured, Path) else _runtime_data_dir() / "memory.json"


def _mcp_server_extra_args() -> list[str]:
    """Build the extra CLI args passed to the MCP Context Server subprocess.

    The context server derives ``memory.db`` from ``--data-dir``; no env-var
    path fallback. We pass Little Lisa's runtime data directory so the MCP
    server's SQLite store lives alongside the host's other artifacts.
    """
    return ["--data-dir", str(_runtime_data_dir())]


def _runtime_thread_dir() -> Path | None:
    configured = _RUNTIME_STATE.get("thread_dir")
    return configured if isinstance(configured, Path) else None


def _runtime_thread_path() -> Path | None:
    configured = _RUNTIME_STATE.get("thread_path")
    return configured if isinstance(configured, Path) else None


def _runtime_state_snapshot() -> dict[str, Any] | None:
    configured = _RUNTIME_STATE.get("state_snapshot")
    return configured if isinstance(configured, dict) else None


def _set_last_turn_context(context: dict[str, Any]) -> None:
    _RUNTIME_STATE["last_turn_context"] = context


def _last_turn_context() -> dict[str, Any] | None:
    context = _RUNTIME_STATE.get("last_turn_context")
    return context if isinstance(context, dict) else None


def _set_last_frontdoor_debug(payload: dict[str, Any]) -> None:
    _RUNTIME_STATE["last_frontdoor_debug"] = payload


def _last_frontdoor_debug() -> dict[str, Any] | None:
    payload = _RUNTIME_STATE.get("last_frontdoor_debug")
    return payload if isinstance(payload, dict) else None


def _preview_text(value: str, limit: int = 600) -> str:
    normalized = " ".join(str(value or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)] + "..."


def _handoff_note(
    *,
    raw_message: str,
    mode_id: str,
    deep_route: bool,
    preferred_model_id: str,
) -> str:
    if deep_route:
        return "explicit /deep route handoff to deep-designated model"
    if mode_id == "default" and not raw_message.lstrip().startswith("/"):
        return "natural-language intent parsed by front-door, executing on default model"
    if raw_message.lstrip().startswith("/"):
        return f"explicit command route handoff to {preferred_model_id} model selection"
    return "default routing handoff"


def _build_turn_context(
    *,
    raw_message: str,
    mode_id: str,
    query: str,
    deep_route: bool,
    preferred_model_id: str,
    resolved_model: StubModelDefinition,
    mode: WorkflowModeConfig,
) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "raw_message": raw_message,
        "mode_id": mode_id,
        "mode_kind": mode.mode_kind,
        "query": query,
        "deep_route": deep_route,
        "preferred_model_id": preferred_model_id,
        "resolved_model": {
            "model_id": resolved_model.model_id,
            "provider": resolved_model.provider,
            "reasoning_model": resolved_model.reasoning_model,
        },
        "handoff": _handoff_note(
            raw_message=raw_message,
            mode_id=mode_id,
            deep_route=deep_route,
            preferred_model_id=preferred_model_id,
        ),
    }


@lru_cache(maxsize=1)
def _load_system_prompt() -> str:
    try:
        return _read_authored_text("user_contexts/system.md").strip()
    except FileNotFoundError:
        return ""


@lru_cache(maxsize=1)
def _load_turn_input_contract() -> str:
    try:
        return (
            _read_authored_text("prompts/agent/turn_input_contract.md")
            .strip()
            .format(user_query_sentinel="## User Query")
        )
    except FileNotFoundError:
        return ""


@lru_cache(maxsize=1)
def _load_frontdoor_route_prompt() -> str:
    try:
        return _read_authored_text("frontdoor/route.md").strip()
    except FileNotFoundError:
        return ""


@lru_cache(maxsize=1)
def _load_frontdoor_fast_path_verifier_prompt() -> str:
    try:
        return _read_authored_text("frontdoor/fast_path_verifier.md").strip()
    except FileNotFoundError:
        return ""


def _assemble_system_prompt(mode_instruction: str, stage_instruction: str | None) -> str:
    parts = [
        _load_system_prompt(),
        mode_instruction,
        _load_turn_input_contract(),
    ]
    if stage_instruction:
        parts.append(stage_instruction)
    return "\n\n".join(part for part in parts if part.strip())


def _frontdoor_model_alias() -> str:
    return "front-door"


def _model_alias_for_mode(mode_id: str) -> str:
    return _MODE_MODEL_ALIAS_MAP.get(mode_id, "default")


def _build_frontdoor_route_catalog_markdown() -> str:
    return shared_frontdoor.build_route_catalog_markdown(
        "rich",
        catalog=get_workflow_catalog(),
        mode_supported=_frontdoor_mode_supported,
    )


def _frontdoor_mode_supported(_mode_id: str, mode: WorkflowModeConfig) -> bool:
    return _mode_support_gap(mode) is None


def _build_frontdoor_user_prompt(
    *,
    normalized_message: str,
    surface_context: SurfaceContext,
    preferred_mode_id: str,
    conversation_excerpt_markdown: str,
    fast_path_candidate_mode_id: str = "",
    is_default_executor: bool = False,
) -> str:
    _ = surface_context
    _ = preferred_mode_id
    parts = [
        "## Front Door Input",
        f"- user_slug: {_runtime_user_slug() or 'none'}",
        f"- raw_message: {normalized_message}",
        f"- is_default_executor: {str(is_default_executor).lower()}",
    ]
    if is_default_executor:
        parts.append(
            "Optimization Hint: You are ALSO the default executor for this turn. "
            "If you route to 'default', you may provide the final user-facing answer "
            "directly in the `direct_response` field to save one LLM call."
        )
    if fast_path_candidate_mode_id:
        parts.append(f"- fast_path_candidate_mode: {fast_path_candidate_mode_id}")
    if conversation_excerpt_markdown.strip():
        parts.extend(["", "## Recent Conversation", conversation_excerpt_markdown.strip()])
    parts.extend(["", "## User Query", normalized_message.strip()])
    return "\n".join(parts)


async def _retry_frontdoor_completion(
    *,
    system_prompt: str,
    user_prompt: str,
    model_def: StubModelDefinition,
    frontdoor_debug: dict[str, Any],
    fallback: FrontDoorRouteDecision,
) -> tuple[StubCompletion, bool] | FrontDoorRouteDecision:
    """Retry front-door completion without structured output format.

    Returns (completion, True) on success, or a fallback FrontDoorRouteDecision
    when the retry also fails.
    """
    retry_prompt = (
        "Return only one valid JSON object matching FrontDoorRouteDecision. "
        "Do not include Markdown fences, prose, or comments."
    )
    retry_messages = [
        {
            "role": "system",
            "content": f"{system_prompt}\n\n{retry_prompt}",
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]
    try:
        completion = await _complete_live(
            messages=retry_messages,
            model_def=model_def,
            max_tokens=512,
            temperature=0.0,
            response_format=None,
        )
    except Exception as retry_exc:  # pylint: disable=broad-except
        frontdoor_debug["planner_retry_error"] = f"{type(retry_exc).__name__}: {retry_exc}"
        frontdoor_debug["structured_output_retry_used"] = True
        frontdoor_debug["structured_output_retry_succeeded"] = False
        frontdoor_debug["raw_completion_preview"] = ""
        frontdoor_debug.update(
            {
                "fallback_used": True,
                "fallback_reason": "planner_completion_error",
            }
        )
        _set_last_frontdoor_debug(frontdoor_debug)
        return fallback
    return (completion, True)


async def _retry_validation_failure(
    *,
    system_prompt: str,
    user_prompt: str,
    model_def: StubModelDefinition,
    frontdoor_debug: dict[str, Any],
    fallback: FrontDoorRouteDecision,
    structured_retry_used: bool,
) -> tuple[StubCompletion, bool, FrontDoorRouteDecision] | FrontDoorRouteDecision:
    """Retry front-door completion when structured output validation fails.

    Returns (completion, True, decision) on successful retry, or a fallback
    FrontDoorRouteDecision when the retry also fails or was already attempted.
    """
    if structured_retry_used:
        frontdoor_debug.update(
            {
                "fallback_used": True,
                "fallback_reason": "planner_output_validation_error",
            }
        )
        _set_last_frontdoor_debug(frontdoor_debug)
        return fallback

    retry_result = await _retry_frontdoor_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_def=model_def,
        frontdoor_debug=frontdoor_debug,
        fallback=fallback,
    )
    if isinstance(retry_result, FrontDoorRouteDecision):
        return retry_result

    retry_completion, _ = retry_result
    retry_text = _extract_json_object(retry_completion.content)
    frontdoor_debug["structured_output_retry_used"] = True
    frontdoor_debug["structured_output_retry_succeeded"] = True
    try:
        retry_decision = FrontDoorRouteDecision.model_validate_json(retry_text)
    except ValidationError as retry_exc:
        frontdoor_debug["planner_retry_validation_error"] = (
            f"{type(retry_exc).__name__}: {retry_exc}"
        )
        frontdoor_debug["raw_completion_preview"] = _preview_text(retry_completion.content)
        frontdoor_debug.update(
            {
                "fallback_used": True,
                "fallback_reason": "planner_output_validation_error",
            }
        )
        _set_last_frontdoor_debug(frontdoor_debug)
        return fallback

    return (retry_completion, True, retry_decision)


async def _plan_frontdoor_route(
    *,
    raw_message: str,
    conversation_excerpt_markdown: str,
) -> FrontDoorRouteDecision:
    frontdoor_alias = _frontdoor_model_alias()
    fallback = FrontDoorRouteDecision(
        selected_mode_id="default",
        normalized_query=raw_message.strip(),
        reasoning="Front-door planner fallback to default route.",
        confidence_score=0.0,
    )
    frontdoor_debug: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "model_alias": frontdoor_alias,
        "resolved_model": {},
        "raw_message": raw_message,
        "system_prompt_chars": 0,
        "user_prompt_chars": 0,
        "conversation_excerpt_chars": len(conversation_excerpt_markdown),
        "user_prompt_preview": "",
        "planner_prompt_tokens": 0,
        "planner_completion_tokens": 0,
        "planner_total_tokens": 0,
        "planner_completion_attempts": [],
        "structured_output_retry_used": False,
        "structured_output_retry_succeeded": False,
    }

    try:
        model_def = _resolve_model(frontdoor_alias)
    except Exception as exc:  # pylint: disable=broad-except
        frontdoor_debug["planner_model_resolution_error"] = f"{type(exc).__name__}: {exc}"
        frontdoor_debug.update(
            {
                "fallback_used": True,
                "fallback_reason": "planner_model_resolution_error",
                "planner_reasoning": fallback.reasoning,
                "selected_mode_id": fallback.selected_mode_id,
                "normalized_query": fallback.normalized_query,
                "confidence_score": fallback.confidence_score,
            }
        )
        _set_last_frontdoor_debug(frontdoor_debug)
        return fallback

    frontdoor_debug["resolved_model"] = {
        "model_id": model_def.model_id,
        "provider": model_def.provider,
        "reasoning_model": model_def.reasoning_model,
    }

    is_default_executor = False
    try:
        default_alias = _model_alias_for_mode("default")
        default_model_def = _resolve_model(default_alias)
        is_default_executor = model_def.model_id == default_model_def.model_id
    except Exception:  # pylint: disable=broad-except
        pass

    def _record_frontdoor_completion(
        completion_result: StubCompletion,
        *,
        attempt: str,
    ) -> None:
        prompt_tokens = _coerce_non_negative_int(completion_result.prompt_tokens)
        completion_tokens = _coerce_non_negative_int(completion_result.completion_tokens)
        total_tokens = _coerce_non_negative_int(completion_result.total_tokens)
        if total_tokens <= 0:
            total_tokens = prompt_tokens + completion_tokens

        frontdoor_debug["planner_prompt_tokens"] = (
            _coerce_non_negative_int(frontdoor_debug.get("planner_prompt_tokens")) + prompt_tokens
        )
        frontdoor_debug["planner_completion_tokens"] = (
            _coerce_non_negative_int(frontdoor_debug.get("planner_completion_tokens"))
            + completion_tokens
        )
        frontdoor_debug["planner_total_tokens"] = (
            _coerce_non_negative_int(frontdoor_debug.get("planner_total_tokens")) + total_tokens
        )

        attempts = frontdoor_debug.get("planner_completion_attempts")
        if not isinstance(attempts, list):
            attempts = []
        attempts.append(
            {
                "attempt": attempt,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "content_chars": len(completion_result.content),
                "tool_call_count": len(completion_result.tool_calls),
            }
        )
        frontdoor_debug["planner_completion_attempts"] = attempts

    def _capture_prompt_shape(messages: list[dict[str, Any]]) -> None:
        if not messages:
            return
        system_text = str(messages[0].get("content") or "")
        user_text = str(messages[1].get("content") or "") if len(messages) > 1 else ""
        frontdoor_debug["system_prompt_chars"] = len(system_text)
        frontdoor_debug["user_prompt_chars"] = len(user_text)
        frontdoor_debug["user_prompt_preview"] = _preview_text(user_text, limit=240)

    async def _complete_frontdoor_planner(**kwargs: Any) -> StubCompletion:
        response_format = kwargs.get("response_format")
        messages = kwargs.get("messages")
        if isinstance(messages, list):
            _capture_prompt_shape(messages)

        async def _retry_without_response_format(*, attempt: str) -> StubCompletion:
            assert response_format is not None
            retry_messages = list(messages or [])
            if retry_messages:
                retry_system = dict(retry_messages[0])
                retry_system["content"] = (
                    f"{str(retry_system.get('content') or '')}\n\n"
                    f"Return only one valid JSON object matching {response_format.__name__}. "
                    "Do not include Markdown fences, prose, or comments."
                )
                retry_messages[0] = retry_system
            retry_completion = await _complete_live(
                messages=retry_messages,
                model_def=model_def,
                max_tokens=int(kwargs.get("max_tokens") or 0),
                temperature=float(kwargs.get("temperature") or 0.0),
                response_format=None,
            )
            _record_frontdoor_completion(retry_completion, attempt=attempt)
            frontdoor_debug["structured_output_retry_used"] = True
            frontdoor_debug["structured_output_retry_succeeded"] = True
            retry_text = _extract_json_object(retry_completion.content)
            parsed = response_format.model_validate_json(retry_text)
            frontdoor_debug["raw_completion_preview"] = _preview_text(retry_completion.content)
            return StubCompletion(
                content=parsed.model_dump_json(),
                tool_calls=retry_completion.tool_calls,
                prompt_tokens=retry_completion.prompt_tokens,
                completion_tokens=retry_completion.completion_tokens,
                total_tokens=retry_completion.total_tokens,
            )

        try:
            completion = await _complete_live(**kwargs)
            attempt = (
                "fast_path_structured"
                if response_format is not FrontDoorRouteDecision
                else "primary_structured"
            )
            _record_frontdoor_completion(completion, attempt=attempt)
        except Exception as exc:  # pylint: disable=broad-except
            if response_format is None:
                raise
            frontdoor_debug["planner_completion_error"] = f"{type(exc).__name__}: {exc}"
            try:
                return await _retry_without_response_format(attempt="retry_after_completion_error")
            except ValidationError as retry_exc:
                frontdoor_debug["planner_retry_validation_error"] = (
                    f"{type(retry_exc).__name__}: {retry_exc}"
                )
            except Exception as retry_exc:  # pylint: disable=broad-except
                frontdoor_debug["planner_retry_error"] = f"{type(retry_exc).__name__}: {retry_exc}"
            frontdoor_debug["structured_output_retry_succeeded"] = False
            frontdoor_debug["fallback_reason"] = "planner_completion_error"
            raise RuntimeError("frontdoor planner completion failed") from exc

        if response_format is None:
            return completion

        try:
            parsed = response_format.model_validate_json(_extract_json_object(completion.content))
        except ValidationError as exc:
            frontdoor_debug["planner_output_validation_error"] = f"{type(exc).__name__}: {exc}"
            frontdoor_debug["raw_completion_preview"] = _preview_text(completion.content)
            try:
                return await _retry_without_response_format(attempt="retry_after_validation_error")
            except ValidationError as retry_exc:
                frontdoor_debug["planner_retry_validation_error"] = (
                    f"{type(retry_exc).__name__}: {retry_exc}"
                )
            except Exception as retry_exc:  # pylint: disable=broad-except
                frontdoor_debug["planner_retry_error"] = f"{type(retry_exc).__name__}: {retry_exc}"
            frontdoor_debug["structured_output_retry_succeeded"] = False
            frontdoor_debug["fallback_reason"] = "planner_output_validation_error"
            raise RuntimeError("frontdoor planner validation failed") from exc

        frontdoor_debug["raw_completion_preview"] = _preview_text(completion.content)
        return StubCompletion(
            content=parsed.model_dump_json(),
            tool_calls=completion.tool_calls,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            total_tokens=completion.total_tokens,
        )

    decision = await shared_frontdoor.plan_frontdoor_route(
        user_message=raw_message,
        current_route_mode="default",
        surface_context=SurfaceContext(
            surface_type="personal_dm",
            surface_id="little-lisa-console",
            memory_scope="personal",
        ),
        preferred_mode_id="default",
        conversation_excerpt_markdown=conversation_excerpt_markdown,
        catalog=get_workflow_catalog(),
        load_route_prompt=lambda: "\n\n".join(
            part
            for part in (
                _load_frontdoor_route_prompt(),
                _load_turn_input_contract(),
            )
            if part.strip()
        ),
        load_fast_path_verifier_prompt=_load_frontdoor_fast_path_verifier_prompt,
        resolve_planner_model=lambda: model_def,
        complete=_complete_frontdoor_planner,
        mode_supported=_frontdoor_mode_supported,
        build_user_prompt=_build_frontdoor_user_prompt,
        model_bounded_max_tokens=512,
        is_default_executor=is_default_executor,
    )

    fallback_reason = str(frontdoor_debug.get("fallback_reason") or "")
    if (
        not fallback_reason
        and decision.planner_source == "deterministic"
        and decision.reasoning == fallback.reasoning
    ):
        fallback_reason = "planner_fallback"

    frontdoor_debug.update(
        {
            "fallback_used": bool(fallback_reason),
            "fallback_reason": fallback_reason,
            "selected_mode_id": decision.selected_mode_id,
            "normalized_query": decision.normalized_query,
            "planner_reasoning": decision.reasoning.strip(),
            "confidence_score": decision.confidence_score,
            "clarification_needed": decision.clarification_needed,
            "coerced_to_default_reason": "",
        }
    )
    _set_last_frontdoor_debug(frontdoor_debug)
    return decision


def _extract_markdown_section(content: str, heading: str) -> str:
    marker = re.escape(f"## {heading}")
    pattern = re.compile(rf"(?ms)^{marker}\n\n(.*?)(?=\n\n## |\Z)")
    match = pattern.search(content)
    if not match:
        return ""
    return match.group(1).strip()


def _extract_front_matter_thread_name(path: Path) -> str | None:
    """Return the display name from YAML front matter if present.

    Checks ``display_name`` first, then falls back to ``thread_name``.
    Delegates to the canonical kernel ``parse_front_matter`` so the parsing
    logic stays in one place.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None
    front_matter, _ = parse_front_matter(text)
    display_name = front_matter.get("display_name")
    if isinstance(display_name, str):
        return display_name.strip() or None
    thread_name = front_matter.get("thread_name")
    if isinstance(thread_name, str):
        return thread_name.strip() or None
    return None


def _summarize_turn_messages(messages: list[dict[str, Any]]) -> dict[str, Any]:
    system_prompt = ""
    user_payload = ""
    assistant_tool_call_count = 0
    tool_result_count = 0

    for message in messages:
        role = str(message.get("role") or "").strip().lower()
        content = message.get("content")
        if role == "system" and isinstance(content, str) and not system_prompt:
            system_prompt = content
        if role == "user" and isinstance(content, str) and not user_payload:
            user_payload = content
        if role == "assistant":
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                assistant_tool_call_count += len(tool_calls)
        if role == "tool":
            tool_result_count += 1

    max_preview_chars = 240
    system_prompt_preview = system_prompt[:max_preview_chars]
    return {
        "system_prompt": {
            "chars": len(system_prompt),
            "preview": system_prompt_preview,
            "omitted_for_brevity": len(system_prompt) > len(system_prompt_preview),
        },
        "assembled_user_payload": {
            "pre_turn_briefing": _extract_markdown_section(user_payload, "Pre-turn Briefing"),
            "memory_recall": _extract_markdown_section(user_payload, "Memory Recall"),
            "dynamic_user_context": _extract_markdown_section(user_payload, "Dynamic User Context"),
            "selected_user_context": _extract_markdown_section(
                user_payload, "Selected User Context"
            ),
            "user_query": _extract_markdown_section(user_payload, "User Query"),
            "chars": len(user_payload),
        },
        "message_count": len(messages),
        "assistant_tool_call_count": assistant_tool_call_count,
        "tool_result_count": tool_result_count,
    }


def _assemble_dynamic_user_context(mode_id: str, query: str, pre_turn_briefing: str) -> str:
    mode = get_workflow_catalog().get(mode_id)
    mode_kind = mode.mode_kind if mode is not None else "unknown"
    mode_description = (mode.description or "").strip() if mode is not None else ""
    command_pattern = (mode.command_pattern or "").strip() if mode is not None else ""
    lines = [
        f"- mode_id: {mode_id}",
        f"- mode_kind: {mode_kind}",
        f"- query_chars: {len(query)}",
        f"- pre_turn_briefing_present: {'true' if bool(pre_turn_briefing.strip()) else 'false'}",
        f"- timeout_profile: {_runtime_timeout_profile()}",
        "- host: little_lisa",
    ]
    user_slug = _runtime_user_slug()
    lines.insert(2, f"- user_slug: {user_slug or 'none'}")
    if mode_description:
        lines.append(f"- mode_description: {mode_description}")
    if command_pattern:
        lines.append(f"- command_pattern: {command_pattern}")

    sections = ["## Dynamic User Context\n\n" + "\n".join(lines)]

    user_slug = _runtime_user_slug()
    user_context = ""
    if user_slug:
        user_context = _load_selected_session_context_text().strip()
    if user_context:
        sections.append("## Selected User Context\n\n" + user_context[:1200])

    thread_digest = _load_thread_digest()
    if thread_digest:
        sections.append(f"## Prior Conversation Digest\n\n{thread_digest[:2000]}")

    conversation_excerpt = _build_recent_thread_excerpt(limit=6)
    if conversation_excerpt:
        sections.append(f"## Recent Conversation\n\n{conversation_excerpt}")

    ranked = _build_ranked_context(query)
    if ranked:
        ranked_lines = "\n".join(f"- {item}" for item in ranked)
        sections.append("## Ranked Context\n\n" + ranked_lines)

    snapshot_section = _build_state_snapshot_context()
    if snapshot_section:
        sections.append(snapshot_section)

    query_lower = query.lower()
    capability_keywords = (
        "capability",
        "capabilities",
        "what can you do",
        "tool",
        "mode",
    )
    if any(keyword in query_lower for keyword in capability_keywords):
        capability = _build_capability_snapshot()
        sections.append(
            "## Capability Context\n\n"
            f"- supported_modes: {len(capability['supported_modes'])}\n"
            f"- operations: {len(capability['operations'])}\n"
            f"- tools: {len(capability['tools'])}\n"
            f"- models: {len(capability['models'])}"
        )

    return "\n\n".join(sections)


def _runtime_log_metadata() -> dict[str, Any]:
    return {
        "runtime": {
            "log_level": _runtime_log_level(),
            "debug": _runtime_debug_enabled(),
            "timeout_profile": _runtime_timeout_profile(),
            "timezone": _runtime_timezone(),
            "timezone_source": _runtime_timezone_source(),
            "session_context": _runtime_user_slug() or "none",
            "mode_id": _runtime_current_mode_id(),
            "stage_id": _runtime_current_stage_id(),
            "next_model": {
                "provider": _runtime_next_model_provider(),
                "model_id": _runtime_next_model_id(),
            },
            "streaming": {
                "enabled": _runtime_stream_enabled(),
                "mode": _runtime_streaming_mode(),
                "char_delay_ms": _runtime_streaming_char_delay_ms(),
            },
        }
    }


def _write_log_record(record: dict[str, Any]) -> None:
    log_path = _runtime_log_path()
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _append_log_event(
    *,
    level: str,
    event: str,
    payload: dict[str, Any],
    debug_payload: dict[str, Any] | None = None,
) -> None:
    normalized_level = _normalize_log_level(level) or "info"
    if not _runtime_should_log(normalized_level):
        return

    record: dict[str, Any] = {
        "logged_at_utc": datetime.now(UTC).isoformat(),
        "level": normalized_level,
        "event": event,
        **payload,
    }
    record.update(_runtime_log_metadata())

    if _runtime_log_level() == "debug":
        debug_data = dict(debug_payload or {})
        if "last_frontdoor_debug" not in debug_data:
            latest_frontdoor_debug = _last_frontdoor_debug()
            if isinstance(latest_frontdoor_debug, dict):
                debug_data["last_frontdoor_debug"] = latest_frontdoor_debug
        if debug_data:
            record["debug"] = debug_data

    _write_log_record(record)


def _append_turn_log(*, context: dict[str, Any], response_text: str) -> None:
    response_chars = len(response_text)
    completion = context.get("completion") if isinstance(context, dict) else None
    mode_model_payload = context.get("resolved_model") if isinstance(context, dict) else None
    frontdoor_debug = context.get("frontdoor_debug") if isinstance(context, dict) else None

    mode_model_id = ""
    mode_model_provider = ""
    mode_reasoning_model = False
    if isinstance(mode_model_payload, dict):
        mode_model_id = str(mode_model_payload.get("model_id") or "")
        mode_model_provider = str(mode_model_payload.get("provider") or "")
        mode_reasoning_model = bool(mode_model_payload.get("reasoning_model", False))

    frontdoor_model_payload = {}
    if isinstance(frontdoor_debug, dict):
        candidate_payload = frontdoor_debug.get("resolved_model")
        if isinstance(candidate_payload, dict):
            frontdoor_model_payload = candidate_payload

    frontdoor_model_id = str(frontdoor_model_payload.get("model_id") or "")
    frontdoor_model_provider = str(frontdoor_model_payload.get("provider") or "")
    frontdoor_reasoning_model = bool(frontdoor_model_payload.get("reasoning_model", False))

    completion_tool_calls = 0
    completion_iterations = 0
    mode_prompt_tokens = 0
    mode_completion_tokens = 0
    mode_total_tokens = 0
    if isinstance(completion, dict):
        completion_tool_calls = _coerce_non_negative_int(completion.get("tool_call_count"))
        completion_iterations = _coerce_non_negative_int(completion.get("iteration_count"))
        mode_prompt_tokens = _coerce_non_negative_int(completion.get("prompt_tokens"))
        mode_completion_tokens = _coerce_non_negative_int(completion.get("completion_tokens"))
        mode_total_tokens = _coerce_non_negative_int(completion.get("total_tokens"))
        if mode_total_tokens <= 0:
            mode_total_tokens = mode_prompt_tokens + mode_completion_tokens

    frontdoor_prompt_tokens = 0
    frontdoor_completion_tokens = 0
    frontdoor_total_tokens = 0
    frontdoor_attempt_count = 0
    if isinstance(frontdoor_debug, dict):
        frontdoor_prompt_tokens = _coerce_non_negative_int(
            frontdoor_debug.get("planner_prompt_tokens")
        )
        frontdoor_completion_tokens = _coerce_non_negative_int(
            frontdoor_debug.get("planner_completion_tokens")
        )
        frontdoor_total_tokens = _coerce_non_negative_int(
            frontdoor_debug.get("planner_total_tokens")
        )
        planner_attempts = frontdoor_debug.get("planner_completion_attempts")
        if isinstance(planner_attempts, list):
            frontdoor_attempt_count = len(planner_attempts)

    mode_model = ""
    if mode_model_provider and mode_model_id:
        mode_model = f"{mode_model_provider}:{mode_model_id}"
    elif mode_model_id:
        mode_model = mode_model_id

    frontdoor_model = ""
    if frontdoor_model_provider and frontdoor_model_id:
        frontdoor_model = f"{frontdoor_model_provider}:{frontdoor_model_id}"
    elif frontdoor_model_id:
        frontdoor_model = frontdoor_model_id

    completion_tokens = {
        "prompt_tokens": mode_prompt_tokens,
        "completion_tokens": mode_completion_tokens,
        "total_tokens": mode_total_tokens,
    }

    record = {
        "type": "little_lisa_turn",
        "context": context,
        "mode_model": mode_model,
        "mode_model_id": mode_model_id,
        "mode_model_provider": mode_model_provider,
        "mode_reasoning_model": mode_reasoning_model,
        "frontdoor_model": frontdoor_model,
        "frontdoor_model_id": frontdoor_model_id,
        "frontdoor_model_provider": frontdoor_model_provider,
        "frontdoor_reasoning_model": frontdoor_reasoning_model,
        "mode_prompt_tokens": mode_prompt_tokens,
        "mode_completion_tokens": mode_completion_tokens,
        "mode_total_tokens": mode_total_tokens,
        "frontdoor_prompt_tokens": frontdoor_prompt_tokens,
        "frontdoor_completion_tokens": frontdoor_completion_tokens,
        "frontdoor_total_tokens": frontdoor_total_tokens,
        "frontdoor_completion_attempt_count": frontdoor_attempt_count,
        "response": {
            "text": response_text,
            "chars": response_chars,
        },
        "route": {
            "mode_id": str(context.get("mode_id") or ""),
            "mode_kind": str(context.get("mode_kind") or ""),
            "query_chars": len(str(context.get("query") or "")),
            "handoff": str(context.get("handoff") or ""),
            "deep_route": bool(context.get("deep_route", False)),
        },
        "completion_summary": {
            "tool_call_count": completion_tool_calls,
            "iteration_count": completion_iterations,
            "tokens": completion_tokens,
        },
        "response_preview": _preview_text(response_text, limit=320),
    }
    debug_payload = {
        "context_keys": sorted(context.keys()),
        "turn_input": context.get("turn_input"),
        "completion": completion,
        "response_full_text": response_text,
        "thread_dir": str(_runtime_thread_dir() or ""),
        "state_snapshot_loaded": isinstance(_runtime_state_snapshot(), dict),
        "session_tokens": {
            "turn_prompt_tokens": _coerce_non_negative_int(
                _RUNTIME_STATE.get("turn_prompt_tokens")
            ),
            "turn_completion_tokens": _coerce_non_negative_int(
                _RUNTIME_STATE.get("turn_completion_tokens")
            ),
            "turn_total_tokens": _coerce_non_negative_int(_RUNTIME_STATE.get("turn_total_tokens")),
            "session_prompt_tokens": _coerce_non_negative_int(
                _RUNTIME_STATE.get("session_prompt_tokens")
            ),
            "session_completion_tokens": _coerce_non_negative_int(
                _RUNTIME_STATE.get("session_completion_tokens")
            ),
            "session_total_tokens": _coerce_non_negative_int(
                _RUNTIME_STATE.get("session_total_tokens")
            ),
        },
    }
    _append_log_event(
        level="info",
        event="turn.completed",
        payload=record,
        debug_payload=debug_payload,
    )


def _render_turn_context(context: dict[str, Any]) -> None:
    console.print(
        Panel(
            json.dumps(context, indent=2, ensure_ascii=True),
            title="Turn Context",
            border_style=STYLE_BORDER_CONTEXT,
        )
    )


def _format_operation_error(exc: OperationExecutionError) -> str:
    message = str(exc).strip()
    lowered = message.lower()
    if "timed out" in lowered:
        return (
            f"{message}\n\n"
            "Little Lisa kept running, but this operation exceeded its local time budget. "
            "Try a narrower query, a faster model, or a larger operation timeout."
        )
    return message


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) >= _RANKED_CONTEXT_MIN_TOKEN_LEN
    }


def _load_recent_thread_entries(limit: int = 30) -> list[dict[str, Any]]:
    path = _runtime_thread_path()
    if path is None or not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict):
            entries.append(decoded)
    return entries


def _append_thread_entry(*, mode_id: str, query: str, response_text: str) -> None:
    path = _runtime_thread_path()
    user_slug = _runtime_user_slug()
    if path is None or not user_slug:
        return
    entry = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "user_slug": user_slug,
        "mode_id": mode_id,
        "query": query,
        "response_text": response_text,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=True) + "\n")


def _build_ranked_context(query: str) -> list[str]:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []
    scored: list[tuple[int, str]] = []
    for entry in _load_recent_thread_entries(limit=40):
        entry_query = str(entry.get("query") or "")
        entry_response = str(entry.get("response_text") or "")
        candidate_text = f"{entry_query} {entry_response}".strip()
        if not candidate_text:
            continue
        overlap = len(query_tokens.intersection(_tokenize(candidate_text)))
        if overlap <= 0:
            continue
        preview = entry_response.strip()[:180] or entry_query.strip()[:180]
        scored.append((overlap, preview))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [text for _, text in scored[:3]]


def _load_thread_digest() -> str:
    """Load the thread digest sidecar if it exists."""
    thread_path = _runtime_thread_path()
    if thread_path is None:
        return ""
    digest_path = thread_path.parent / _THREAD_DIGEST_FILE_NAME
    if not digest_path.is_file():
        return ""
    try:
        return digest_path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _build_recent_thread_excerpt(limit: int = 3) -> str:
    entries = _load_recent_thread_entries(limit=limit)
    if not entries:
        return ""
    lines = ["Recent thread context:"]
    for entry in entries[-limit:]:
        query = str(entry.get("query") or "").strip()
        response = str(entry.get("response_text") or "").strip()
        if not query and not response:
            continue
        preview = response[:160] if response else query[:160]
        lines.append(f"- {query or 'unknown'}: {preview}")
    return "\n".join(lines)


def _build_state_snapshot_context() -> str:
    snapshot = _runtime_state_snapshot()
    if not snapshot:
        return ""
    top_keys = sorted(snapshot.keys())[:8]
    compiled_at = str(snapshot.get("compiled_at") or snapshot.get("generated_at") or "")
    lines = ["## Local State Snapshot", f"- keys: {', '.join(top_keys) or 'none'}"]
    if compiled_at:
        lines.append(f"- compiled_at: {compiled_at}")
    return "\n".join(lines)


def _build_capability_snapshot() -> dict[str, Any]:
    catalog = get_workflow_catalog()
    manifest = _load_manifest()
    operations = get_operation_catalog()
    tools = _tool_catalog()
    supported_modes = [
        mode_id for mode_id, mode in sorted(catalog.items()) if _mode_support_gap(mode) is None
    ]
    unsupported_modes = {
        mode_id: (_mode_support_gap(mode) or "additional host wiring")
        for mode_id, mode in sorted(catalog.items())
        if _mode_support_gap(mode) is not None
    }
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "user_slug": _runtime_user_slug(),
        "timeout_profile": _runtime_timeout_profile(),
        "timezone": _runtime_timezone(),
        "timezone_source": _runtime_timezone_source(),
        "supported_modes": supported_modes,
        "unsupported_modes": unsupported_modes,
        "operations": sorted(operations.keys()),
        "tools": sorted(tools.keys()),
        "models": [
            {
                "model_id": model.model_id,
                "provider": model.provider,
                "designations": sorted(model.designations),
                "reasoning_model": model.reasoning_model,
            }
            for model in manifest.models
        ],
    }


def _render_capability_help() -> str:
    snapshot = _build_capability_snapshot()
    mode_preview = ", ".join(snapshot["supported_modes"][:8])
    model_preview = ", ".join(model["model_id"] for model in snapshot["models"][:4]) or "none"
    unsupported_count = len(snapshot["unsupported_modes"])
    return "\n".join(
        [
            "## Capability Snapshot",
            f"- User context: {snapshot['user_slug']}",
            f"- Timeout profile: {snapshot['timeout_profile']}",
            f"- Timezone: {snapshot['timezone']} ({snapshot['timezone_source']})",
            f"- Supported modes ({len(snapshot['supported_modes'])}): {mode_preview}",
            f"- Operations: {len(snapshot['operations'])}",
            f"- Tools: {len(snapshot['tools'])}",
            f"- Models: {model_preview}",
            f"- Unsupported installed modes: {unsupported_count} (run /debug capabilities)",
        ]
    )


def _render_capability_debug() -> str:
    snapshot = _build_capability_snapshot()
    return "## Capability Debug\n\n```json\n" + json.dumps(snapshot, indent=2) + "\n```"


def _render_debug_context() -> str:
    context = _last_turn_context() or {}
    if not context:
        return "No completed turn context is available yet."
    turn_input = context.get("turn_input") if isinstance(context, dict) else None
    if not isinstance(turn_input, dict):
        return "No assembled turn-input summary is available yet."
    payload = turn_input.get("assembled_user_payload")
    if not isinstance(payload, dict):
        return "No assembled user payload is available yet."
    lines = ["## Last Turn Context"]

    # System prompt summary
    system_prompt_info = turn_input.get("system_prompt")
    if isinstance(system_prompt_info, dict):
        lines.append(f"\n### system_prompt")
        lines.append(f"- chars: {system_prompt_info.get('chars', 0)}")
        lines.append(f"- preview: {system_prompt_info.get('preview', '(empty)')}")
        if system_prompt_info.get("omitted_for_brevity"):
            lines.append("- (truncated for brevity)")
    else:
        lines.append("\n### system_prompt\n(not available)")

    # Session context
    session_ref = _runtime_session_context_ref()
    session_active = bool(_RUNTIME_STATE.get("session_context_active", False))
    lines.append(f"\n### session_context")
    lines.append(f"- ref: {session_ref or '(none)'}")
    lines.append(f"- active: {'true' if session_active else 'false'}")

    for key in (
        "pre_turn_briefing",
        "memory_recall",
        "dynamic_user_context",
        "selected_user_context",
        "user_query",
    ):
        value = str(payload.get(key) or "").strip()
        lines.append(f"\n### {key}\n{value or '(empty)'}")
    return "\n".join(lines)


def _render_debug_self() -> str:
    snapshot = _build_capability_snapshot()
    thread_entries = _load_recent_thread_entries(limit=10)
    state_snapshot = _runtime_state_snapshot()
    return "\n".join(
        [
            "## Little Lisa Self Model",
            f"- user_slug: {snapshot['user_slug']}",
            f"- timeout_profile: {snapshot['timeout_profile']}",
            f"- timezone: {snapshot['timezone']} ({snapshot['timezone_source']})",
            f"- supported_modes: {len(snapshot['supported_modes'])}",
            f"- operations: {len(snapshot['operations'])}",
            f"- tools: {len(snapshot['tools'])}",
            f"- models: {len(snapshot['models'])}",
            f"- local_thread_entries: {len(thread_entries)}",
            f"- local_state_snapshot_loaded: {'yes' if isinstance(state_snapshot, dict) else 'no'}",
        ]
    )


def _render_debug_frontdoor() -> str:
    payload = _last_frontdoor_debug()
    if payload is None:
        return "No front-door routing diagnostics are available yet."
    return "## Front-Door Debug\n\n```json\n" + json.dumps(payload, indent=2) + "\n```"


def _render_runtime_settings() -> str:
    return "\n".join(
        [
            "## Runtime Settings",
            f"- timeout_profile: {_runtime_timeout_profile()}",
            f"- timezone: {_runtime_timezone()}",
            f"- timezone_source: {_runtime_timezone_source()}",
            f"- session_context: {_runtime_user_slug() or 'none'}",
            f"- debug: {'on' if _runtime_debug_enabled() else 'off'}",
            f"- log_level: {_runtime_log_level()}",
        ]
    )


def _load_state_snapshot(path: str) -> dict[str, Any] | None:
    configured = path.strip()
    if not configured:
        return None
    file_path = Path(configured).expanduser()
    if not file_path.exists():
        raise ValueError(f"State snapshot file not found: {file_path}")
    decoded = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("State snapshot file must contain a JSON object.")
    return decoded


def _mode_timeout_seconds(mode_id: str) -> int | None:
    profile = _runtime_timeout_profile()
    profile_map = {
        "fast": {"default": 30, "discuss": 30, "search": 45, "thinkabout": 60},
        "balanced": {"default": 60, "discuss": 60, "search": 120, "thinkabout": 120},
        "deep": {"default": 120, "discuss": 120, "search": 240, "thinkabout": 240},
    }
    return profile_map.get(profile, {}).get(mode_id)


def _apply_mode_timeout_profile(mode_id: str) -> list[tuple[str, int]]:
    timeout_seconds = _mode_timeout_seconds(mode_id)
    if timeout_seconds is None:
        return []
    updated: list[tuple[str, int]] = []
    for operation_id, definition in get_operation_catalog().items():
        if definition.mode_allowlist and mode_id not in definition.mode_allowlist:
            continue
        previous = int(definition.timeout_seconds)
        definition.timeout_seconds = timeout_seconds
        updated.append((operation_id, previous))
    return updated


def _restore_mode_timeouts(applied: list[tuple[str, int]]) -> None:
    if not applied:
        return
    catalog = get_operation_catalog()
    for operation_id, previous in applied:
        definition = catalog.get(operation_id)
        if definition is not None:
            definition.timeout_seconds = previous


def _load_manifest() -> ModelManifest:
    return _load_manifest_runtime(_RUNTIME_STATE, env_var=_LITTLE_LISA_MANIFEST_ENV)


def reload_model_manifest() -> ModelManifest:
    """Reload manifest from disk and return the new snapshot."""

    return _reload_manifest_runtime(_RUNTIME_STATE, env_var=_LITTLE_LISA_MANIFEST_ENV)


def _require_models_present() -> None:
    manifest = _load_manifest()
    if not manifest.models:
        raise ValueError("Little Lisa requires at least one model in models.yaml.")


def resolve_little_lisa_model(model_id: str) -> StubModelDefinition:
    """Resolve one model id or designation against the Little Lisa manifest."""

    return _resolve_little_lisa_model_runtime(
        model_id,
        runtime_state=_RUNTIME_STATE,
        env_var=_LITTLE_LISA_MANIFEST_ENV,
    )


def _resolve_model(model_id: str) -> StubModelDefinition:
    """Adapter used by TurnHooks to resolve one requested model identifier."""

    return resolve_little_lisa_model(model_id)


def _explicit_deep_query(message: str) -> str | None:
    stripped = message.lstrip()
    match = re.match(r"^/deep(?:[\s,:;]+(?P<query>.+))?$", stripped, flags=re.IGNORECASE)
    if not match:
        return None
    return str(match.group("query") or "").strip()


def _preferred_model_id_for_route(*, raw_message: str, mode_id: str, deep_route: bool) -> str:
    if deep_route:
        return "deep"
    return _model_alias_for_mode(mode_id)


def _azure_client_instance() -> AsyncAzureOpenAI:
    existing_client = _RUNTIME_STATE.get("azure_client")
    if isinstance(existing_client, AsyncAzureOpenAI):
        return existing_client

    endpoint = (
        os.getenv(_LITTLE_LISA_AZURE_ENDPOINT_ENV, "").strip()
        or os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    )
    api_key = (
        os.getenv(_LITTLE_LISA_AZURE_API_KEY_ENV, "").strip()
        or os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    )
    if not endpoint or not api_key:
        raise ValueError(
            "Azure OpenAI is not configured for Little Lisa. Set "
            "LITTLE_LISA_AZURE_OPENAI_ENDPOINT and LITTLE_LISA_AZURE_OPENAI_API_KEY "
            "(or AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY)."
        )
    api_version = os.getenv(_LITTLE_LISA_AZURE_API_VERSION_ENV, "").strip() or "2024-10-21"
    client = AsyncAzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key,
    )
    _RUNTIME_STATE["azure_client"] = client
    return client


def _openai_client_instance() -> AsyncOpenAI:
    existing_client = _RUNTIME_STATE.get("openai_client")
    if isinstance(existing_client, AsyncOpenAI):
        return existing_client

    api_key = (
        os.getenv(_LITTLE_LISA_OPENAI_API_KEY_ENV, "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
    )
    if not api_key:
        raise ValueError(
            "OpenAI is not configured for Little Lisa. Set LITTLE_LISA_OPENAI_API_KEY "
            "(or OPENAI_API_KEY)."
        )
    client = AsyncOpenAI(api_key=api_key)
    _RUNTIME_STATE["openai_client"] = client
    return client


def _google_client_instance() -> AsyncOpenAI:
    existing_client = _RUNTIME_STATE.get("google_client")
    if isinstance(existing_client, AsyncOpenAI):
        return existing_client

    api_key = (
        os.getenv(_LITTLE_LISA_GOOGLE_API_KEY_ENV, "").strip()
        or os.getenv("GOOGLE_API_KEY", "").strip()
    )
    endpoint = _normalize_google_base_url(
        os.getenv(_LITTLE_LISA_GOOGLE_API_BASE_URL_ENV, "").strip()
        or os.getenv("GOOGLE_API_BASE_URL", "").strip()
    )
    if not api_key:
        raise ValueError(
            "Google Gemini is not configured for Little Lisa. Set LITTLE_LISA_GOOGLE_API_KEY "
            "(or GOOGLE_API_KEY)."
        )
    client = AsyncOpenAI(api_key=api_key, base_url=endpoint)
    _RUNTIME_STATE["google_client"] = client
    return client


def _normalize_google_base_url(base_url: str) -> str:
    """Return a supported Google OpenAI-compatible base URL."""

    normalized = base_url.strip()
    if not normalized:
        return _DEFAULT_GOOGLE_OPENAI_BASE_URL

    lowered = normalized.lower()
    if "generativelanguage.googleapis.com" in lowered and "/openai" not in lowered:
        return _DEFAULT_GOOGLE_OPENAI_BASE_URL

    if not normalized.endswith("/"):
        normalized = f"{normalized}/"
    return normalized


def _normalize_ollama_host_to_base_url(host: str) -> str:
    normalized = host.strip().rstrip("/")
    if not normalized:
        return ""
    if not normalized.startswith(("http://", "https://")):
        normalized = f"http://{normalized}"
    if normalized.endswith("/v1"):
        return f"{normalized}/"
    return f"{normalized}/v1/"


def _ollama_client_instance() -> AsyncOpenAI:
    existing_client = _RUNTIME_STATE.get("ollama_client")
    if isinstance(existing_client, AsyncOpenAI):
        return existing_client

    configured_host = (
        os.getenv(_LITTLE_LISA_OLLAMA_HOST_ENV, "").strip() or os.getenv("OLLAMA_HOST", "").strip()
    )
    base_url = _normalize_ollama_host_to_base_url(configured_host) or "http://localhost:11434/v1/"

    api_key = (
        os.getenv(_LITTLE_LISA_OLLAMA_API_KEY_ENV, "").strip()
        or os.getenv("OLLAMA_API_KEY", "").strip()
    )
    client = AsyncOpenAI(api_key=api_key or "ollama", base_url=base_url)
    _RUNTIME_STATE["ollama_client"] = client
    return client


def _anthropic_http_client_instance() -> httpx.AsyncClient:
    existing_client = _RUNTIME_STATE.get("anthropic_http_client")
    if isinstance(existing_client, httpx.AsyncClient):
        return existing_client

    configured_base_url = os.getenv(_LITTLE_LISA_ANTHROPIC_BASE_URL_ENV, "").strip()
    base_url = (
        configured_base_url.rstrip("/") if configured_base_url else "https://api.anthropic.com"
    )
    client = httpx.AsyncClient(base_url=base_url, timeout=60.0)
    _RUNTIME_STATE["anthropic_http_client"] = client
    return client


def _extract_anthropic_error_detail(response: httpx.Response) -> str:
    """Extract a concise detail string from an Anthropic error response."""

    try:
        payload = response.json()
    except ValueError:
        response_text = response.text.strip()
        return response_text or "no error payload provided by Anthropic"

    if not isinstance(payload, dict):
        return "unexpected non-object error payload from Anthropic"

    error_payload = payload.get("error")
    if isinstance(error_payload, dict):
        error_type = str(error_payload.get("type") or "").strip()
        error_message = str(error_payload.get("message") or "").strip()
        if error_type and error_message:
            return f"{error_type}: {error_message}"
        if error_message:
            return error_message
        if error_type:
            return error_type

    top_level_message = str(payload.get("message") or "").strip()
    return top_level_message or "no error payload provided by Anthropic"


def _normalize_response_format(response_format: type[Any] | None) -> dict[str, str] | None:
    if response_format is None:
        return None
    return {"type": "json_object"}


def _extract_completion_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(tool_calls, list):
        return []
    normalized: list[dict[str, Any]] = []
    for call in tool_calls:
        call_id = str(getattr(call, "id", "") or "")
        call_type = str(getattr(call, "type", "function") or "function")
        function = getattr(call, "function", None)
        function_name = str(getattr(function, "name", "") or "")
        function_arguments = str(getattr(function, "arguments", "") or "")
        entry: dict[str, Any] = {
            "id": call_id,
            "type": call_type,
            "function": {
                "name": function_name,
                "arguments": function_arguments,
            },
        }
        # Gemini thinking models attach thought_signature to each function call;
        # it must be echoed verbatim in the corresponding tool-result message.
        # Google requires it inside the function object ("functionCall parts").
        thought_sig = getattr(call, "thought_signature", None) or getattr(
            function, "thought_signature", None
        )
        if thought_sig:
            entry["function"]["thought_signature"] = thought_sig
        normalized.append(entry)
    return normalized


def _coerce_non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _coerce_non_negative_float(value: Any) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _ensure_usage_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Ensure one breakdown dict has every token/time field.

    Pre-existing entries (from earlier sessions or tests with the old
    3-key schema) are upgraded in place so the accumulator math never
    raises ``KeyError`` on a missing ``cached_tokens`` etc.
    """
    entry.setdefault("prompt_tokens", 0)
    entry.setdefault("completion_tokens", 0)
    entry.setdefault("total_tokens", 0)
    entry.setdefault("cached_tokens", 0)
    entry.setdefault("reasoning_tokens", 0)
    entry.setdefault("duration_ms", 0.0)
    entry.setdefault("latency_ms", 0)
    return entry


def _extract_usage_counts(
    usage_payload: Any,
) -> tuple[int, int, int, int, int]:
    """Extract token and sub-counter usage across provider payload shapes.

    Returns a 5-tuple ``(prompt_tokens, completion_tokens, total_tokens,
    cached_tokens, reasoning_tokens)``. Sub-counters default to 0 when
    the provider did not surface them.
    """

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    cached_tokens = 0
    reasoning_tokens = 0
    if isinstance(usage_payload, dict):
        prompt_tokens = _coerce_non_negative_int(
            usage_payload.get("prompt_tokens") or usage_payload.get("input_tokens")
        )
        completion_tokens = _coerce_non_negative_int(
            usage_payload.get("completion_tokens") or usage_payload.get("output_tokens")
        )
        total_tokens = _coerce_non_negative_int(usage_payload.get("total_tokens"))
        cached_tokens = _coerce_non_negative_int(
            usage_payload.get("cached_tokens")
            or usage_payload.get("cache_read_input_tokens")
        )
        reasoning_tokens = _coerce_non_negative_int(
            usage_payload.get("reasoning_tokens")
            or usage_payload.get("cache_creation_input_tokens")
        )
    elif usage_payload is not None:
        prompt_tokens = _coerce_non_negative_int(
            getattr(usage_payload, "prompt_tokens", None)
            or getattr(usage_payload, "input_tokens", None)
        )
        completion_tokens = _coerce_non_negative_int(
            getattr(usage_payload, "completion_tokens", None)
            or getattr(usage_payload, "output_tokens", None)
        )
        total_tokens = _coerce_non_negative_int(getattr(usage_payload, "total_tokens", None))
        cached_tokens = _coerce_non_negative_int(
            getattr(usage_payload, "cached_tokens", None)
            or getattr(usage_payload, "cache_read_input_tokens", None)
        )
        reasoning_tokens = _coerce_non_negative_int(
            getattr(usage_payload, "reasoning_tokens", None)
            or getattr(usage_payload, "cache_creation_input_tokens", None)
        )

    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens
    return prompt_tokens, completion_tokens, total_tokens, cached_tokens, reasoning_tokens


def _parse_tool_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        return {}
    text = arguments.strip()
    if not text:
        return {}
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}
    if isinstance(decoded, dict):
        return decoded
    return {"value": decoded}


def _to_anthropic_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not tools:
        return []
    translated: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function_payload = tool.get("function")
        if not isinstance(function_payload, dict):
            continue
        name = str(function_payload.get("name") or "").strip()
        if not name:
            continue
        description = str(function_payload.get("description") or "").strip()
        parameters = function_payload.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {"type": "object", "properties": {}}
        translated.append(
            {
                "name": name,
                "description": description,
                "input_schema": parameters,
            }
        )
    return translated


def _to_anthropic_payload_messages(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:

    def _translate_tool_result_message(message: dict[str, Any]) -> dict[str, Any]:
        tool_call_id = str(message.get("tool_call_id") or "").strip() or "tool_call"
        tool_result_content = str(message.get("content") or "")
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": tool_result_content,
                }
            ],
        }

    def _translate_assistant_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
        translated_calls: list[dict[str, Any]] = []
        raw_tool_calls = message.get("tool_calls")
        if not isinstance(raw_tool_calls, list):
            return translated_calls
        for index, raw_tool_call in enumerate(raw_tool_calls, start=1):
            if not isinstance(raw_tool_call, dict):
                continue
            function_payload = raw_tool_call.get("function")
            if isinstance(function_payload, dict):
                tool_name = str(function_payload.get("name") or "").strip()
                tool_arguments = function_payload.get("arguments")
            else:
                tool_name = str(raw_tool_call.get("name") or "").strip()
                tool_arguments = raw_tool_call.get("arguments")
            if not tool_name:
                continue
            tool_id = str(raw_tool_call.get("id") or "").strip() or f"tool_call_{index}"
            translated_calls.append(
                {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": tool_name,
                    "input": _parse_tool_arguments(tool_arguments),
                }
            )
        return translated_calls

    def _translate_chat_message(message: dict[str, Any], role: str) -> dict[str, Any]:
        blocks: list[dict[str, Any]] = []
        text_content = str(message.get("content") or "")
        if text_content:
            blocks.append({"type": "text", "text": text_content})
        if role == "assistant":
            blocks.extend(_translate_assistant_tool_calls(message))
        if not blocks:
            blocks.append({"type": "text", "text": ""})
        return {"role": role, "content": blocks}

    system_parts: list[str] = []
    translated_messages: list[dict[str, Any]] = []

    for message in messages:
        role = str(message.get("role") or "").strip().lower()
        if role == "system":
            text = str(message.get("content") or "").strip()
            if text:
                system_parts.append(text)
            continue

        if role == "tool":
            translated_messages.append(_translate_tool_result_message(message))
            continue

        if role not in {"user", "assistant"}:
            continue

        translated_messages.append(_translate_chat_message(message, role))

    if not translated_messages:
        translated_messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": ""}],
            }
        )

    system_prompt = "\n\n".join(system_parts).strip()
    return (system_prompt or None, translated_messages)


def _extract_anthropic_text(content_blocks: Any) -> str:
    if not isinstance(content_blocks, list):
        return ""
    text_parts: list[str] = []
    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        if str(block.get("type") or "") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str) and text:
            text_parts.append(text)
    return "\n".join(text_parts).strip()


def _extract_anthropic_tool_calls(content_blocks: Any) -> list[dict[str, Any]]:
    if not isinstance(content_blocks, list):
        return []
    calls: list[dict[str, Any]] = []
    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        if str(block.get("type") or "") != "tool_use":
            continue
        calls.append(
            {
                "id": str(block.get("id") or ""),
                "type": "function",
                "function": {
                    "name": str(block.get("name") or ""),
                    "arguments": json.dumps(block.get("input") or {}),
                },
            }
        )
    return calls


async def _complete_anthropic(
    *,
    messages: list[dict[str, Any]],
    model_def: StubModelDefinition,
    max_tokens: int,
    temperature: float,
    tools: list[dict[str, Any]] | None = None,
    response_format: type[Any] | None = None,
) -> StubCompletion:
    api_key = (
        os.getenv(_LITTLE_LISA_ANTHROPIC_API_KEY_ENV, "").strip()
        or os.getenv("ANTHROPIC_API_KEY", "").strip()
    )
    if not api_key:
        raise ValueError(
            "Anthropic is not configured for Little Lisa. Set LITTLE_LISA_ANTHROPIC_API_KEY "
            "(or ANTHROPIC_API_KEY)."
        )

    system_prompt, anthropic_messages = _to_anthropic_payload_messages(messages)
    payload: dict[str, Any] = {
        "model": model_def.model_id,
        "max_tokens": max_tokens,
        "messages": anthropic_messages,
    }
    if temperature >= 0:
        payload["temperature"] = temperature
    if system_prompt:
        payload["system"] = system_prompt

    anthropic_tools = _to_anthropic_tools(tools)
    if anthropic_tools:
        payload["tools"] = anthropic_tools

    if response_format is not None:
        format_prompt = "Return only a valid JSON object with no Markdown wrapper."
        payload["system"] = f"{payload.get('system', '').strip()}\n\n{format_prompt}".strip()

    headers = {
        "x-api-key": api_key,
        "anthropic-version": os.getenv(_LITTLE_LISA_ANTHROPIC_VERSION_ENV, "").strip()
        or "2023-06-01",
    }
    response = await _anthropic_http_client_instance().post(
        "/v1/messages",
        headers=headers,
        json=payload,
    )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = _extract_anthropic_error_detail(exc.response)
        message = (
            f"Anthropic request failed ({exc.response.status_code}) for model "
            f"'{model_def.model_id}': {detail}"
        )
        if exc.response.status_code == httpx.codes.NOT_FOUND:
            message += (
                " Check the model_id in models.yaml and confirm the model is available "
                "for your Anthropic account."
            )
        raise httpx.HTTPStatusError(
            message,
            request=exc.request,
            response=exc.response,
        ) from exc

    response_json = response.json()
    content_blocks = response_json.get("content")
    content = _extract_anthropic_text(content_blocks)
    tool_calls = _extract_anthropic_tool_calls(content_blocks)
    (
        prompt_tokens,
        completion_tokens,
        total_tokens,
        cached_tokens,
        reasoning_tokens,
    ) = _extract_usage_counts(response_json.get("usage"))

    content = _normalize_structured_content(content, response_format)

    return StubCompletion(
        content=content,
        tool_calls=tool_calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
    )


def _first_message_content(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            item_text = getattr(item, "text", "")
            if item_text:
                text_parts.append(str(item_text))
        return "\n".join(text_parts).strip()
    return str(content or "")


def _extract_json_object(text: str) -> str:
    """Extract the first likely JSON object from a model response."""

    normalized = str(text or "").strip()
    if normalized.startswith("```json"):
        normalized = normalized[7:]
    elif normalized.startswith("```"):
        normalized = normalized[3:]
    if normalized.endswith("```"):
        normalized = normalized[:-3]
    normalized = normalized.strip()

    if normalized.startswith("{") and normalized.endswith("}"):
        return normalized

    start = normalized.find("{")
    end = normalized.rfind("}")
    if start >= 0 and end > start:
        return normalized[start : end + 1]
    return normalized


def _normalize_structured_content(
    content: str,
    response_format: type[Any] | None,
) -> str:
    """Return normalized structured JSON when possible, otherwise original content."""

    if (
        response_format is None
        or not content
        or not hasattr(response_format, "model_validate_json")
    ):
        return content
    try:
        parsed = response_format.model_validate_json(_extract_json_object(content))
    except ValidationError:
        return content
    normalized = parsed.model_dump_json()
    if not isinstance(normalized, str):
        return content
    return normalized


async def _complete_live(
    *,
    messages: list[dict[str, Any]],
    model_def: StubModelDefinition,
    max_tokens: int,
    temperature: float,
    tools: list[dict[str, Any]] | None = None,
    response_format: type[Any] | None = None,
) -> StubCompletion:
    """Execute a live provider completion against configured provider APIs."""

    stream_stage = _runtime_stream_enabled() and _runtime_stream_mode() in {"stage", "hybrid"}
    stage_label = f"{model_def.provider}:{model_def.model_id}"
    started_at = time.perf_counter()
    response_format_name = ""
    if response_format is not None:
        response_format_name = str(getattr(response_format, "__name__", "") or "")

    kwargs: dict[str, Any] = {
        "model": model_def.model_id,
        "messages": messages,
    }
    if model_def.provider == "azure_openai" or model_def.reasoning_model:
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens
        kwargs["temperature"] = temperature
    if tools:
        kwargs["tools"] = tools
    normalized_format = _normalize_response_format(response_format)
    if normalized_format is not None:
        kwargs["response_format"] = normalized_format

    _append_log_event(
        level="debug",
        event="completion.request",
        payload={
            "provider": model_def.provider,
            "model_id": model_def.model_id,
            "reasoning_model": model_def.reasoning_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "response_format": response_format_name,
            "message_count": len(messages),
            "tool_schema_count": len(tools or []),
            "message_summary": _summarize_turn_messages(messages),
        },
        debug_payload={
            "messages": messages,
            "tool_schemas": tools or [],
        },
    )

    if model_def.provider == "anthropic":
        try:
            if stream_stage:
                console.print(f"[dim][stage] contacting {stage_label}...[/]")
                result = await _complete_anthropic(
                    messages=messages,
                    model_def=model_def,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    tools=tools,
                    response_format=response_format,
                )
                console.print(f"[dim][stage] {stage_label} completed.[/]")
            else:
                with console.status("[bold yellow]Thinking...", spinner="dots"):
                    result = await _complete_anthropic(
                        messages=messages,
                        model_def=model_def,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        tools=tools,
                        response_format=response_format,
                    )
        except Exception as exc:  # pylint: disable=broad-except
            if stream_stage:
                console.print(f"[dim][stage] {stage_label} failed.[/]")
            elapsed_ms = max(0.0, (time.perf_counter() - started_at) * 1000.0)
            _append_log_event(
                level="error",
                event="completion.error",
                payload={
                    "provider": model_def.provider,
                    "model_id": model_def.model_id,
                    "reasoning_model": model_def.reasoning_model,
                    "response_format": response_format_name,
                    "latency_ms": round(elapsed_ms, 2),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
                debug_payload={
                    "messages": messages,
                    "tool_schemas": tools or [],
                },
            )
            raise

        elapsed_ms = max(0.0, (time.perf_counter() - started_at) * 1000.0)
        _append_log_event(
            level="debug",
            event="completion.response",
            payload={
                "provider": model_def.provider,
                "model_id": model_def.model_id,
                "reasoning_model": model_def.reasoning_model,
                "response_format": response_format_name,
                "latency_ms": round(elapsed_ms, 2),
                "content_chars": len(result.content),
                "tool_call_count": len(result.tool_calls),
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
                "cached_tokens": result.cached_tokens,
                "reasoning_tokens": result.reasoning_tokens,
                "response_preview": _preview_text(result.content, limit=320),
            },
            debug_payload={
                "response_text": result.content,
                "tool_calls": result.tool_calls,
            },
        )
        # StubCompletion is frozen; rebuild with the host-measured
        # duration so the ledger can bill time-based providers.
        final_result = StubCompletion(
            content=result.content,
            tool_calls=result.tool_calls,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            cached_tokens=result.cached_tokens,
            reasoning_tokens=result.reasoning_tokens,
            duration_ms=elapsed_ms,
        )
        await _publish_token_usage(
            provider=model_def.provider,
            model_id=model_def.model_id,
            prompt_tokens=final_result.prompt_tokens,
            completion_tokens=final_result.completion_tokens,
            total_tokens=final_result.total_tokens,
            cached_tokens=final_result.cached_tokens,
            reasoning_tokens=final_result.reasoning_tokens,
            duration_ms=final_result.duration_ms,
            latency_ms=int(elapsed_ms),
            call_site="complete_live.anthropic",
        )
        return final_result

    if stream_stage:
        console.print(f"[dim][stage] contacting {stage_label}...[/]")
    try:
        if stream_stage:
            if model_def.provider == "azure_openai":
                response = await _azure_client_instance().chat.completions.create(**kwargs)
            elif model_def.provider == "openai":
                response = await _openai_client_instance().chat.completions.create(**kwargs)
            elif model_def.provider == "ollama":
                response = await _ollama_client_instance().chat.completions.create(**kwargs)
            elif model_def.provider == "google":
                response = await _google_client_instance().chat.completions.create(**kwargs)
            else:
                raise ValueError(
                    f"Unsupported provider '{model_def.provider}' in Little Lisa manifest."
                )
            console.print(f"[dim][stage] {stage_label} completed.[/]")
        else:
            with console.status("[bold yellow]Thinking...", spinner="dots"):
                if model_def.provider == "azure_openai":
                    response = await _azure_client_instance().chat.completions.create(**kwargs)
                elif model_def.provider == "openai":
                    response = await _openai_client_instance().chat.completions.create(**kwargs)
                elif model_def.provider == "ollama":
                    response = await _ollama_client_instance().chat.completions.create(**kwargs)
                elif model_def.provider == "google":
                    response = await _google_client_instance().chat.completions.create(**kwargs)
                else:
                    raise ValueError(
                        f"Unsupported provider '{model_def.provider}' in Little Lisa manifest."
                    )
    except Exception as exc:  # pylint: disable=broad-except
        if stream_stage:
            console.print(f"[dim][stage] {stage_label} failed.[/]")
        elapsed_ms = max(0.0, (time.perf_counter() - started_at) * 1000.0)
        _append_log_event(
            level="error",
            event="completion.error",
            payload={
                "provider": model_def.provider,
                "model_id": model_def.model_id,
                "reasoning_model": model_def.reasoning_model,
                "response_format": response_format_name,
                "latency_ms": round(elapsed_ms, 2),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
            debug_payload={
                "messages": messages,
                "tool_schemas": tools or [],
            },
        )
        raise

    choice = response.choices[0]
    message = choice.message
    content = _first_message_content(message)
    tool_calls = _extract_completion_tool_calls(getattr(message, "tool_calls", None))
    (
        prompt_tokens,
        completion_tokens,
        total_tokens,
        cached_tokens,
        reasoning_tokens,
    ) = _extract_usage_counts(getattr(response, "usage", None))

    content = _normalize_structured_content(content, response_format)

    elapsed_ms = max(0.0, (time.perf_counter() - started_at) * 1000.0)
    _append_log_event(
        level="debug",
        event="completion.response",
        payload={
            "provider": model_def.provider,
            "model_id": model_def.model_id,
            "reasoning_model": model_def.reasoning_model,
            "response_format": response_format_name,
            "latency_ms": round(elapsed_ms, 2),
            "content_chars": len(content),
            "tool_call_count": len(tool_calls),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cached_tokens": cached_tokens,
            "reasoning_tokens": reasoning_tokens,
            "response_preview": _preview_text(content, limit=320),
        },
        debug_payload={
            "response_text": content,
            "tool_calls": tool_calls,
        },
    )

    final_result = StubCompletion(
        content=content,
        tool_calls=tool_calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
        duration_ms=elapsed_ms,
    )
    await _publish_token_usage(
        provider=model_def.provider,
        model_id=model_def.model_id,
        prompt_tokens=final_result.prompt_tokens,
        completion_tokens=final_result.completion_tokens,
        total_tokens=final_result.total_tokens,
        cached_tokens=final_result.cached_tokens,
        reasoning_tokens=final_result.reasoning_tokens,
        duration_ms=final_result.duration_ms,
        latency_ms=int(elapsed_ms),
        call_site="complete_live.openai_compatible",
    )
    return final_result


async def _execute_tool(tool_id: str, ctx: Any) -> ToolResult:
    """Return deterministic local tool results without external API calls."""

    if tool_id == "web_search":
        query = str(
            getattr(ctx, "parameters", {}).get("query") or getattr(ctx, "query", "")
        ).strip()
        if not web_search_is_configured():
            return ToolResult(
                tool_id=tool_id,
                error="Web search is not configured. Set WEB_SEARCH_API_KEY to enable it.",
            )
        if not query:
            return ToolResult(tool_id=tool_id, error="web_search requires a query.")
        try:
            response = await web_search(query)
        except Exception as exc:  # pylint: disable=broad-except
            return ToolResult(
                tool_id=tool_id,
                error=f"Web search failed: {type(exc).__name__}: {exc}",
            )
        return ToolResult(
            tool_id=tool_id,
            output_json={"response": response.model_dump(mode="json")},
        )
    return ToolResult(tool_id=tool_id, output_text=f"{tool_id} is stubbed in Little Lisa.")


async def execute_little_lisa_tool(tool_id: str, ctx: Any) -> ToolResult:
    """Execute one Little Lisa tool with the same behavior as the turn loop."""

    return await _execute_tool(tool_id, ctx)


def _read_authored_text(relative_path: str) -> str:
    """Load authored pack content by relative path from public content roots."""

    for root in reversed(_authored_content_roots()):
        candidate = root / relative_path
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    raise FileNotFoundError(relative_path)


def _mode_support_gap(_mode: WorkflowModeConfig) -> str | None:
    """Return the missing host capability for modes Little Lisa cannot run.

    Little Lisa only loads the core pack. Modes that require
    app-level wiring (workshop, vibecheck) live in the cloudservers pack
    and are not visible here.
    """

    return None


def _state_provider() -> UnifiedState:
    """Return an empty state cache for operations that request host state."""

    return UnifiedState(compiled_at=datetime.now(UTC).isoformat())


def format_modes() -> str:
    """Render supported Little Lisa modes separately from installed pack modes."""

    catalog = get_workflow_catalog()
    lines = ["Supported Little Lisa modes:"]
    for mode_id, mode in sorted(catalog.items()):
        if _mode_support_gap(mode) is not None:
            continue
        command = mode.command_pattern or "default route"
        lines.append(f"- {mode_id} ({mode.mode_kind}): {mode.description or command}")
    unsupported = [mode_id for mode_id, mode in sorted(catalog.items()) if _mode_support_gap(mode)]
    if unsupported:
        lines.extend(["", "Installed public modes that need a fuller host:"])
        for mode_id in unsupported:
            requirement = _mode_support_gap(catalog[mode_id]) or "additional host wiring"
            lines.append(f"- {mode_id}: requires {requirement}.")
    return "\n".join(lines)


def format_models() -> Table:
    """Render a table of configured models with designations and aliases."""

    manifest = _load_manifest()
    table = Table(title="Little Lisa Models")
    table.add_column("Model", style="bold")
    table.add_column("Provider")
    table.add_column("Designations")
    table.add_column("Aliases")
    table.add_column("Reasoning")

    for model in manifest.models:
        designations = ", ".join(sorted(model.designations)) or "—"
        aliases = ", ".join(sorted(model.aliases)) or "—"
        reasoning = "yes" if model.reasoning_model else "no"
        table.add_row(model.model_id, model.provider, designations, aliases, reasoning)

    return table


def _format_help() -> str:
    """Return a capability-accurate help message for Little Lisa."""

    return "\n".join(
        [
            "Little Lisa",
            "",
            "Supported commands:",
            "- /search <query> runs the prompt-authored search operation.",
            "- /markdown <request> writes a local Markdown artifact.",
            "- /deep <request> routes through default mode using the deep-designated model.",
            "- /discuss <topic> parses discuss mode and runs its optional pre-turn briefing.",
            "- Natural-language turns pass through the front-door route planner first.",
            "- /help capabilities shows the runtime capability snapshot.",
            "- /settings shows effective profile/timezone/debug/log-level runtime settings.",
            "- /tokens shows detailed session token breakdown by model, mode, and stage.",
            "- /debug context|capabilities|self|frontdoor shows local diagnostics.",
            "- /quit exits the interactive console loop.",
            "- Action and chat modes added through SDK_CONTENT_ROOTS run when they only need "
            "Little Lisa callbacks.",
            "- Any other text uses default chat parsing.",
            "",
            "Model manifest: configure lisa_framework/examples/little_lisa/models.yaml with",
            "designations 'front-door', 'default', and optional 'deep'.",
            "Set LITTLE_LISA_MODELS_FILE to override the manifest path.",
            "Set LITTLE_LISA_TIMEOUT_PROFILE or --timeout-profile to tune budgets.",
            "Set LITTLE_LISA_TIMEZONE or --timezone to pin local status time.",
            "Use --session-context <name> to select content/user_contexts/<name>.md; "
            "or pass a full .md file path. Omit it to skip session context.",
            "Use --timeout-profile fast|balanced|deep for per-mode operation budgets.",
            "Use --state-snapshot <file.json> to inject compact local state context.",
            "Run `little-lisa cli` for terminal REPL.",
            "Run `little-lisa desktop` for localhost web UI.",
            "Run `little-lisa` with no args for tray-first desktop launch.",
            "Live web search is available when WEB_SEARCH_API_KEY is configured.",
            "Not included: reminders, image generation, surf, "
            "proactive delivery, Teams, Azure, or production Lisa services.",
            "Memory recall is auto-injected before each turn from the local "
            "memory store; the recall_memory operation is also available.",
        ]
    )


def _format_unsupported_mode(mode: WorkflowModeConfig) -> str:
    """Return an explicit capability gap for installed modes the example does not wire."""

    requirement = _mode_support_gap(mode) or "additional host wiring"
    return "\n".join(
        [
            f"Mode '{mode.mode_id}' is installed in the public packs, "
            "but Little Lisa does not wire it.",
            f"It requires {requirement}.",
            "Run --list-modes to see what this example supports.",
        ]
    )


def _resolve_int_env(env_var: str, default: int) -> int:
    """Parse a positive integer from an env var with a sensible default."""
    raw = os.getenv(env_var, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


class LittleLisaStateProvider:
    """File-only digest + consolidation provider for Little Lisa.

    Replaces BlankStateProvider with two deterministic, no-LLM operations
    that run inside the existing slice budget:

    1. Thread digest: when the thread JSONL exceeds max_thread_entries,
       compact the oldest overflow entries into a thread_digest.md
       sidecar and truncate the JSONL to the most recent entries.
    2. Memory consolidation: when the memory store exceeds
       max_memory_records, evict the oldest daily episodic records with
       the lowest salience.

    Both operations are pure file I/O — no model calls, no background
    scheduler, no new dependencies. They run inside the existing
    /consolidate command and the optional post-turn slice.
    """

    def __init__(
        self,
        *,
        thread_path: Path | None,
        memory_store: MCPMemoryClient,
        max_thread_entries: int = _DEFAULT_THREAD_MAX_ENTRIES,
        max_memory_records: int = _DEFAULT_MEMORY_MAX_RECORDS,
    ) -> None:
        self._thread_path = thread_path
        self._memory_store = memory_store
        self._max_thread_entries = max(1, max_thread_entries)
        self._max_memory_records = max(1, max_memory_records)

    def __call__(self) -> Any:
        return None

    async def slice(self, ctx: Any, policy: Any) -> SliceOutcome:
        """Run bounded thread compaction + memory eviction."""
        del ctx
        started = time.monotonic()
        work_items = 0
        notes: list[str] = []
        last_item = ""

        thread_compacted = self._compact_thread(policy, started)
        if thread_compacted[0] > 0:
            work_items += thread_compacted[0]
            last_item = thread_compacted[1]
            notes.append(thread_compacted[2])

        memory_evicted = self._consolidate_memory(policy, started)
        if memory_evicted[0] > 0:
            work_items += memory_evicted[0]
            last_item = memory_evicted[1]
            notes.append(memory_evicted[2])

        elapsed_ms = (time.monotonic() - started) * 1000.0
        return SliceOutcome(
            work_items_completed=work_items,
            elapsed_ms=elapsed_ms,
            last_item=last_item,
            notes=tuple(notes),
        )

    def _compact_thread(
        self,
        policy: Any,
        started: float,
    ) -> tuple[int, str, str]:
        """Compact overflowed thread entries into a digest sidecar.

        Returns ``(entries_compacted, last_item, note)``.
        """
        if self._thread_path is None or not self._thread_path.exists():
            return 0, "", ""
        if _slice_budget_expired(policy, started):
            return 0, "", ""

        lines = self._thread_path.read_text(encoding="utf-8").splitlines()
        if len(lines) <= self._max_thread_entries:
            return 0, "", ""

        overflow_count = len(lines) - self._max_thread_entries
        overflow_lines = lines[:overflow_count]
        digest_parts: list[str] = []
        for raw_line in overflow_lines:
            try:
                entry = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if not isinstance(entry, dict):
                continue
            query = str(entry.get("query") or "").strip()
            response = str(entry.get("response_text") or "").strip()
            if not query and not response:
                continue
            preview = response[:200] if response else query[:200]
            digest_parts.append(f"- {query or 'unknown'}: {preview}")

        digest_path = self._thread_path.parent / _THREAD_DIGEST_FILE_NAME
        existing = ""
        if digest_path.is_file():
            existing = digest_path.read_text(encoding="utf-8").strip()
        new_block = "\n".join(digest_parts)
        if existing:
            digest_text = f"{existing}\n\n{new_block}"
        else:
            digest_text = new_block
        digest_path.write_text(digest_text, encoding="utf-8")

        remaining_lines = lines[overflow_count:]
        self._thread_path.write_text(
            "\n".join(remaining_lines) + "\n" if remaining_lines else "",
            encoding="utf-8",
        )

        note = f"compacted {overflow_count} thread entries into digest"
        return overflow_count, f"thread:{overflow_count} entries", note

    def _consolidate_memory(
        self,
        policy: Any,
        started: float,
    ) -> tuple[int, str, str]:
        """Evict low-salience episodic daily memories under the cap.

        Returns ``(records_evicted, last_item, note)``.
        """
        if _slice_budget_expired(policy, started):
            return 0, "", ""
        if not hasattr(self._memory_store, "_records"):
            return 0, "", "eviction skipped: store is an external MCP client"
            
        records = getattr(self._memory_store, "_records", [])  # noqa: SLF001
        if len(records) <= self._max_memory_records:
            return 0, "", ""

        evict_count = len(records) - self._max_memory_records
        candidates = [
            record
            for record in records
            if record.memory_tier == "daily" and record.memory_kind == "episodic"
        ]
        candidates.sort(key=lambda r: (r.salience, r.id))
        to_evict = candidates[:evict_count]
        if not to_evict:
            return 0, "", ""
        evict_ids = {record.id for record in to_evict}
        self._memory_store._records = [  # noqa: SLF001
            record for record in records if record.id not in evict_ids
        ]
        self._memory_store._save()  # noqa: SLF001

        note = f"evicted {len(to_evict)} low-salience episodic memories"
        return len(to_evict), f"memory:{len(to_evict)} evicted", note


def _slice_budget_expired(policy: Any, started: float) -> bool:
    """Return True if the slice budget has been consumed."""
    budget_ms = getattr(policy, "budget_ms", 0) or 0
    if budget_ms <= 0:
        return False
    return (time.monotonic() - started) * 1000.0 >= budget_ms


async def _run_workflow_turn(
    *,
    session: Any,
    slug: str,
    clean_message: str,
    model_def: object,
    user_block: str,
    user_id: str,
    user_name: str,
    principal_key: str,
    thread_user_id: str,
    session_key: str,
    conversation_id: str,
    workspace_id: str,
    surface_context: Any,
    event_bus: Any = None,
    persist_session_state: bool = True,
    run_post_turn_operations: bool = True,
    include_model_badge: bool = True,
    delegation_depth: int = 0,
) -> Any:
    """Adapter from WorkflowTurnRunner protocol to run_turn for workshop children.

    Each workshop child runs a full turn via run_turn with the same hooks
    the parent uses, but with a child-scoped session key and mode_id.
    """
    child_mode_id = getattr(session, "mode_id", "default")
    child_result = await run_turn(
        mode_id=child_mode_id,
        query=clean_message,
        hooks=_build_turn_hooks(),
        preferred_model_id=None,
    )
    return TurnResultAdapter(child_result)


class TurnResultAdapter:
    """Minimal adapter so TurnResult satisfies the TextResponse protocol."""

    __slots__ = ("_result",)

    def __init__(self, result: Any) -> None:
        self._result = result

    @property
    def text(self) -> str:
        return getattr(self._result, "response_text", "") or ""


def _build_mcp_tool_servers() -> list[Any]:
    """Build the approved MCP server registry and compile it to configs.

    Registers all first-party MCP servers (context, operations, search, surf,
    guard, storage, content) with the launch policy registry, then compiles
    them into ``MCPClientConfig`` entries for ``MCPToolPool``.

    v1 keeps ``MCPMemoryClient`` for the memory seam; the servers compiled
    here go through ``MCPToolPool`` for tool discovery and dispatch.

    All first-party servers use ``VENV_BIN`` runtime — they run as native
    console_scripts entry points (``.venv/Scripts/lisa-*.exe``) with zero
    subprocess overhead. No ``uv`` resolution needed.
    """
    from kernel.mcp_launch.compiler import compile_registry  # noqa: PLC0415
    from kernel.mcp_launch.policy import (  # noqa: PLC0415
        EnvPolicy,
        MCPServerRegistry,
        MCPServerSpec,
        RuntimeKind,
        ToolPolicy,
    )

    registry = MCPServerRegistry()
    data_dir = str(_runtime_data_dir())
    workspace_root = str(FRAMEWORK_ROOT)

    # All first-party servers run as console_scripts entry points from the
    # project venv. The entry_point defaults to server_id (e.g. "lisa-context"),
    # which matches the [project.scripts] name in mcp_server/pyproject.toml.
    venv_dir = str(FRAMEWORK_ROOT / ".venv")

    first_party_servers = [
        ("lisa-context", "lisa_context.server", True),
        ("lisa-operations", "lisa_operations.server", False),
        ("lisa-search", "lisa_search.server", False),
        ("lisa-surf", "lisa_surf.server", False),
        ("lisa-guard", "lisa_guard.server", False),
        ("lisa-storage", "lisa_storage.server", True),
        ("lisa-content", "lisa_content.server", False),
    ]

    for server_id, module, supports_readonly in first_party_servers:
        # lisa-operations exposes every operation as a tool, including
        # web_search. lisa-search already owns web_search, so deny it
        # from lisa-operations to avoid a tool-name collision.
        tool_policy = ToolPolicy()
        if server_id == "lisa-operations":
            tool_policy = ToolPolicy(denied_tools=["web_search"])
        registry.register(
            MCPServerSpec(
                server_id=server_id,
                runtime_kind=RuntimeKind.VENV_BIN,
                venv_dir=venv_dir,
                env_policy=EnvPolicy(inherit_parent=True),
                tool_policy=tool_policy,
            )
        )

    return compile_registry(registry, data_dir=data_dir, workspace_root=workspace_root)


def _build_workshop_context_bundle(
    *,
    state: Any,
    plan: Any,
    workspace_id: str,
    user_id: str,
    user_slug: str,
    surface_context: Any,
    workshop_budget: Any,
    reasoning_model: bool = False,
) -> Any:
    """Build a minimal WorkshopContextBundle for Little Lisa workshop runs.

    Little Lisa has no context broker, so each panel gets an empty evidence
    packet. The shared context packet is also empty. This is sufficient for
    workshop operations that rely on the model's own knowledge rather than
    curated evidence.
    """
    from datetime import UTC, datetime  # noqa: PLC0415
    from kernel.models import (  # noqa: PLC0415
        ContextEvidence,
        ContextPacket,
        WorkshopContextBundle,
        WorkshopPanelContext,
    )

    compiled_at = datetime.now(UTC).isoformat()
    empty_packet = ContextPacket(
        compiled_at=compiled_at,
        query=getattr(plan, "query", ""),
        selected_evidence=[],
    )
    panel_contexts: list[WorkshopPanelContext] = []
    for spec in getattr(plan, "panel_specs", []) or []:
        panel_contexts.append(
            WorkshopPanelContext(
                panel_id=spec.panel_id,
                title=spec.title or spec.panel_id,
                mode_id=spec.mode_id,
                query=spec.query,
                model_alias=spec.model_alias or "",
                panel_kind=spec.panel_kind,
                connectors=list(spec.connectors or []),
                runtime_prefetch_views=list(spec.runtime_prefetch_views or []),
                setup_instructions=spec.setup_instructions or "",
                surface_context=surface_context,
                compiled_at=compiled_at,
                context_packet=empty_packet.model_copy(deep=True),
                context_markdown="",
            )
        )

    return WorkshopContextBundle(
        workshop_id=getattr(plan, "workshop_id", ""),
        title=getattr(plan, "title", ""),
        query=getattr(plan, "query", ""),
        workspace_id=workspace_id,
        user_id=user_id,
        user_slug=user_slug,
        surface_context=surface_context,
        compiled_at=compiled_at,
        shared_context_packet=empty_packet.model_copy(deep=True),
        shared_context_markdown="",
        panel_contexts=panel_contexts,
    )


def _build_turn_hooks(context: dict[str, Any] | None = None) -> TurnHooks:
    """Build TurnHooks wired to Little Lisa callbacks and memory operations."""

    from kernel.slice import SlicePolicy, SliceTuning  # noqa: PLC0415

    # MCP Client wiring — memory stays on MCPMemoryClient (v1);
    # tool servers go through MCPToolPool via the compiled registry.
    mcp_command = os.getenv("LISA_MCP_COMMAND", str(FRAMEWORK_ROOT / ".venv" / "Scripts" / "lisa-context.exe"))
    mcp_args = os.getenv("LISA_MCP_ARGS", "").split()
    mcp_args = [*mcp_args, *_mcp_server_extra_args()]
    memory_store = MCPMemoryClient(
        mcp_command,
        mcp_args,
    )
    slice_tuning_dict, explicit_budget_ms = _resolve_slice_tuning()
    slice_tuning = SliceTuning.from_mapping(slice_tuning_dict)
    state_provider = LittleLisaStateProvider(
        thread_path=_runtime_thread_path(),
        memory_store=memory_store,
        max_thread_entries=_resolve_int_env(
            _LITTLE_LISA_THREAD_MAX_ENTRIES_ENV, _DEFAULT_THREAD_MAX_ENTRIES
        ),
        max_memory_records=_resolve_int_env(
            _LITTLE_LISA_MEMORY_MAX_RECORDS_ENV, _DEFAULT_MEMORY_MAX_RECORDS
        ),
    )

    persisted_last_slice_finished_at: float | None = None
    if context is not None:
        raw = context.get("last_slice_finished_at")
        if isinstance(raw, (int, float)):
            persisted_last_slice_finished_at = float(raw)

    async def _post_turn_slice(
        policy: SlicePolicy,
    ) -> Any:
        return await state_provider.slice(ctx=None, policy=policy)

    async def _pre_turn_slice(policy: SlicePolicy) -> Any:
        return await state_provider.slice(ctx=None, policy=policy)

    async def _explicit_slice(policy: SlicePolicy) -> Any:
        return await state_provider.slice(ctx=None, policy=policy)


    async def _answer_memory_recall(
        *,
        query: str,
        user_slug: str,
        workspace_id: str,
        memory_scope: str = "personal",
        surface_id: str = "",
        limit: int = 5,
        **_kwargs: Any,
    ) -> MemoryRecallAnswer:
        records = await memory_store.list_memories(
            user_slug=user_slug,
            workspace_id=workspace_id,
            memory_scope=memory_scope,
            surface_id=surface_id,
            limit=max(1, min(int(limit or 1), 20)),
        )
        if not records:
            return MemoryRecallAnswer(
                query=query,
                found=False,
                answer=f"No memory matches found for: {query}",
            )

        query_tokens = _tokenize(query)
        ranked: list[tuple[float, MemoryRecord]] = []
        for record in records:
            haystack = f"{record.content} {json.dumps(record.content_json, ensure_ascii=True)}"
            overlap = len(query_tokens.intersection(_tokenize(haystack)))
            if overlap <= 0 and query_tokens:
                continue
            lexical_score = overlap / max(len(query_tokens), 1) if query_tokens else 1.0
            combined = min(1.0, lexical_score + max(0.0, record.salience) * 0.2)
            ranked.append((combined, record))

        ranked.sort(key=lambda item: (item[0], item[1].id), reverse=True)
        top = ranked[: max(1, min(int(limit or 1), 8))]
        if not top:
            return MemoryRecallAnswer(
                query=query,
                found=False,
                answer=f"No memory matches found for: {query}",
            )

        evidence: list[MemoryRecallEvidence] = []
        for score, record in top:
            evidence.append(
                MemoryRecallEvidence(
                    memory_id=record.id,
                    created_at=record.created_at,
                    content=record.content,
                    memory_scope=record.memory_scope,
                    surface_id=record.surface_id,
                    source_user_slug=record.source_user_slug,
                    operation_id=record.operation_id,
                    memory_tier=record.memory_tier,
                    salience=record.salience,
                    combined_score=round(score, 3),
                    lexical_score=round(score, 3),
                    recency_score=1.0,
                    match_reasons=["lexical overlap"],
                )
            )

        summary = evidence[0].content
        confidence = evidence[0].combined_score if evidence else 0.0
        return MemoryRecallAnswer(
            query=query,
            found=True,
            answer=summary,
            confidence=round(confidence, 3),
            used_semantic_search=False,
            auto_deep_used=False,
            evidence=evidence,
        )

    def _base_operation_context(
        *,
        query: str,
        mode_id: str,
        memory_scope: str,
    ) -> OperationExecutionContext:
        user_slug = _runtime_user_slug() or "system"
        return OperationExecutionContext(
            workspace_id="little-lisa",
            user_id=user_slug,
            user_slug=user_slug,
            user_name=user_slug,
            conversation_id="little-lisa-console",
            surface_type="personal_dm",
            surface_id="little-lisa-console",
            memory_scope=memory_scope,
            mode_id=mode_id,
            query=query,
            model_id="default",
            complete_fn=_complete_live,
            resolve_model_fn=_resolve_model,
            execute_tool_fn=_execute_tool,
            execute_operation_fn=_execute_operation,
            memory_store=memory_store,
            answer_memory_recall_fn=_answer_memory_recall,
            memory_recall_default_limit=8,
        )

    async def _execute_operation(
        operation_id: str,
        ctx: OperationExecutionContext,
    ) -> OperationResult:
        ctx.memory_store = memory_store
        ctx.answer_memory_recall_fn = _answer_memory_recall
        ctx.memory_recall_default_limit = 8
        return await execute_operation(operation_id, ctx)

    async def _on_turn_complete(
        mode_id: str,
        query: str,
        response_text: str,
        messages: list[dict[str, Any]],
    ) -> None:
        if context is None:
            return
        turn_input_summary = _summarize_turn_messages(messages)
        assistant_message_count = 0
        for message in messages:
            role = str(message.get("role") or "").strip().lower()
            if role == "assistant":
                assistant_message_count += 1

        context["turn_input"] = turn_input_summary
        context["completion"] = {
            "mode_id": mode_id,
            "query": query,
            "response_preview": response_text[:240],
            "response_chars": len(response_text),
            "message_count": turn_input_summary.get("message_count", len(messages)),
            "assistant_message_count": assistant_message_count,
            "tool_call_count": turn_input_summary.get("assistant_tool_call_count", 0),
            "tool_result_count": turn_input_summary.get("tool_result_count", 0),
        }

    return TurnHooks(
        resolve_model=_resolve_model,
        complete=_complete_live,
        mcp_tool_servers=_build_mcp_tool_servers(),
        execute_operation=_execute_operation,
        assemble_system_prompt=_assemble_system_prompt,
        assemble_user_context=_assemble_dynamic_user_context,
        on_turn_complete=_on_turn_complete,
        slice_tuning=slice_tuning,
        pre_turn_slice=_pre_turn_slice,
        post_turn_slice=_post_turn_slice,
        explicit_slice=_explicit_slice,
        last_slice_finished_at=persisted_last_slice_finished_at,
        workflow_turn_runner=_run_workflow_turn,
        workflow_session_factory=create_workflow_session,
        state_provider=_state_provider,
        build_workshop_context_bundle_fn=_build_workshop_context_bundle,
    )


async def _run_explicit_slice_command(*, message: str) -> str:
    """Run the explicit-mode slice as a user-invoked command.

    This is the escape hatch for hosts whose opportunistic pre/post
    slices are disabled (or whose state provider has no work to do).
    Calling ``/consolidate`` (or ``/slice``) invokes the same slice
    machinery the kernel uses, but with a one-shot user-controlled
    budget read from the ``LITTLE_LISA_SLICE_BUDGET_MS`` env var.
    """
    import time as _time  # noqa: PLC0415

    from kernel.slice import SliceMode, SlicePolicy  # noqa: PLC0415

    _, explicit_budget_ms = _resolve_slice_tuning()
    provider = LittleLisaStateProvider(
        thread_path=_runtime_thread_path(),
        memory_store=MCPMemoryClient(
            os.getenv("LISA_MCP_COMMAND", str(FRAMEWORK_ROOT / ".venv" / "Scripts" / "lisa-context.exe")),
            [
                *os.getenv("LISA_MCP_ARGS", "").split(),
                *_mcp_server_extra_args(),
            ],
        ),
        max_thread_entries=_resolve_int_env(
            _LITTLE_LISA_THREAD_MAX_ENTRIES_ENV, _DEFAULT_THREAD_MAX_ENTRIES
        ),
        max_memory_records=_resolve_int_env(
            _LITTLE_LISA_MEMORY_MAX_RECORDS_ENV, _DEFAULT_MEMORY_MAX_RECORDS
        ),
    )
    policy = SlicePolicy(
        mode=SliceMode.EXPLICIT,
        budget_ms=explicit_budget_ms,
    )
    started = _time.monotonic()
    outcome = await provider.slice(ctx=None, policy=policy)
    elapsed_ms = (_time.monotonic() - started) * 1000.0
    persisted = _RUNTIME_STATE.setdefault("last_slice_finished_at", None)
    if isinstance(persisted, (int, float)):
        _RUNTIME_STATE["last_slice_finished_at"] = _time.monotonic()
    timestamp_utc = datetime.now(UTC).isoformat()
    context = {
        "timestamp_utc": timestamp_utc,
        "raw_message": message,
        "mode_id": "consolidate",
        "handoff": "explicit slice command",
        "explicit_slice": {
            "budget_ms": explicit_budget_ms,
            "provider": type(provider).__name__,
            "work_items_completed": outcome.work_items_completed,
            "last_item": outcome.last_item,
            "notes": list(outcome.notes),
            "elapsed_ms": round(elapsed_ms, 3),
            "policy_budget_ms": policy.budget_ms,
        },
    }
    _set_last_turn_context(context)
    _append_turn_log(context=context, response_text="")
    _append_thread_entry(mode_id="consolidate", query=message, response_text="")
    _RUNTIME_STATE["current_mode_id"] = "consolidate"
    _RUNTIME_STATE["current_stage_id"] = "-"
    _RUNTIME_STATE["next_model_id"] = "-"
    if outcome.work_items_completed == 0:
        return (
            f"Slice ran in {elapsed_ms:.1f} ms with no work items. "
            "Thread and memory are within their caps; nothing to compact."
        )
    note_text = "; ".join(outcome.notes) if outcome.notes else "ok"
    return (
        f"Slice drained {outcome.work_items_completed} work item(s) in "
        f"{elapsed_ms:.1f} ms (budget {policy.budget_ms} ms). "
        f"Last item: {outcome.last_item or '<none>'}. Notes: {note_text}."
    )


async def handle_message(message: str) -> str:  # noqa: PLR0911, PLR0912, PLR0915
    """Route one console message through the kernel turn runner.

    For action modes, entry operations run through `run_turn` which delegates
    to the operation. For chat and workflow modes, the full model+tool loop
    executes via `run_turn` with live provider-backed completion.
    """

    stripped = message.strip()
    lowered = stripped.lower()
    _append_log_event(
        level="debug",
        event="turn.received",
        payload={
            "raw_message": message,
            "message_chars": len(message),
            "is_command": bool(message.lstrip().startswith("/")),
        },
    )

    if lowered.startswith("/help") or lowered == "help":
        help_topic = ""
        if lowered.startswith("/help"):
            help_topic = stripped[5:].strip().lower()
        help_context = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "raw_message": message,
            "mode_id": "help",
            "handoff": "help shortcut",
        }
        _set_last_turn_context(help_context)
        if help_topic in {"capabilities", "capability"}:
            response_text = _render_capability_help()
        else:
            response_text = _format_help()
        _append_turn_log(context=help_context, response_text=response_text)
        _append_thread_entry(mode_id="help", query=message, response_text=response_text)
        _RUNTIME_STATE["current_mode_id"] = "help"
        _RUNTIME_STATE["current_stage_id"] = "-"
        _RUNTIME_STATE["next_model_id"] = "-"
        return response_text

    if lowered in {"/tokens", "tokens"}:
        tokens_context = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "raw_message": message,
            "mode_id": "tokens",
            "handoff": "tokens shortcut",
        }
        response_text = _format_token_stats()
        _set_last_turn_context(tokens_context)
        _append_turn_log(context=tokens_context, response_text=response_text)
        _append_thread_entry(mode_id="tokens", query=message, response_text=response_text)
        _RUNTIME_STATE["current_mode_id"] = "tokens"
        _RUNTIME_STATE["current_stage_id"] = "-"
        _RUNTIME_STATE["next_model_id"] = "-"
        return response_text

    if lowered in {"/settings", "settings"}:
        settings_context = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "raw_message": message,
            "mode_id": "settings",
            "handoff": "settings shortcut",
        }
        response_text = _render_runtime_settings()
        _set_last_turn_context(settings_context)
        _append_turn_log(context=settings_context, response_text=response_text)
        _append_thread_entry(mode_id="settings", query=message, response_text=response_text)
        _RUNTIME_STATE["current_mode_id"] = "settings"
        _RUNTIME_STATE["current_stage_id"] = "-"
        _RUNTIME_STATE["next_model_id"] = "-"
        return response_text

    if lowered in {"/consolidate", "consolidate", "/slice", "slice"}:
        response_text = await _run_explicit_slice_command(message=message)
        return response_text

    if lowered.startswith("/debug"):
        subcommand = stripped[6:].strip().lower()
        if subcommand in {"", "help"}:
            response_text = (
                "Supported debug commands: "
                "/debug context, /debug capabilities, /debug self, /debug frontdoor"
            )
        elif subcommand == "context":
            response_text = _render_debug_context()
        elif subcommand == "capabilities":
            response_text = _render_capability_debug()
        elif subcommand == "self":
            response_text = _render_debug_self()
        elif subcommand == "frontdoor":
            response_text = _render_debug_frontdoor()
        else:
            response_text = (
                f"Unknown debug command '{subcommand}'. "
                "Supported: context, capabilities, self, frontdoor."
            )
        debug_context = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "raw_message": message,
            "mode_id": "debug",
            "handoff": "debug shortcut",
        }
        _set_last_turn_context(debug_context)
        _append_turn_log(context=debug_context, response_text=response_text)
        _append_thread_entry(mode_id="debug", query=message, response_text=response_text)
        _RUNTIME_STATE["current_mode_id"] = "debug"
        _RUNTIME_STATE["current_stage_id"] = "-"
        _RUNTIME_STATE["next_model_id"] = "-"
        return response_text

    _require_models_present()

    deep_query = _explicit_deep_query(message)
    deep_route = deep_query is not None
    if deep_route:
        mode_id = "default"
        query = deep_query or ""
    elif message.lstrip().startswith("/"):
        mode_id, query = parse_route(message)
    else:
        route_decision = await _plan_frontdoor_route(
            raw_message=message,
            conversation_excerpt_markdown=_build_recent_thread_excerpt(),
        )
        if (
            not route_decision.clarification_needed
            and route_decision.direct_response
            and route_decision.selected_mode_id == "default"
        ):
            response_text = route_decision.direct_response
            planner_debug = _last_frontdoor_debug() or {}
            direct_context = {
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "raw_message": message,
                "mode_id": "default",
                "handoff": "front-door optimization (direct response)",
                "frontdoor_route": route_decision.model_dump(mode="json"),
            }
            if planner_debug:
                direct_context["frontdoor_debug"] = planner_debug
            _set_last_turn_context(direct_context)
            _append_turn_log(context=direct_context, response_text=response_text)
            _append_thread_entry(
                mode_id="default",
                query=route_decision.normalized_query or message,
                response_text=response_text,
            )

            # Record usage as if it was a default mode turn
            default_alias = _model_alias_for_mode("default")
            resolved_default = _resolve_model(default_alias)
            _apply_turn_usage(
                TurnResult(
                    content=response_text,
                    prompt_tokens=0,  # Included in front-door usage
                    completion_tokens=0,
                ),
                model_id=resolved_default.model_id,
                mode_id="default",
                stage_id="-",
                frontdoor_routing=planner_debug,
            )
            return response_text

        if route_decision.clarification_needed and route_decision.clarification_question:
            response_text = route_decision.clarification_question
            planner_debug = _last_frontdoor_debug() or {}
            clarification_context = {
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "raw_message": message,
                "mode_id": "front-door",
                "handoff": "front-door clarification",
                "frontdoor_route": route_decision.model_dump(mode="json"),
            }
            if planner_debug:
                clarification_context["frontdoor_debug"] = planner_debug
            _set_last_turn_context(clarification_context)
            _append_turn_log(context=clarification_context, response_text=response_text)
            _append_thread_entry(
                mode_id="front-door",
                query=message,
                response_text=response_text,
            )
            return response_text
        mode_id = route_decision.selected_mode_id
        query = route_decision.normalized_query or message

    catalog = get_workflow_catalog()
    mode = catalog.get(mode_id) or catalog["default"]
    preview_model_id = _preferred_model_id_for_route(
        raw_message=message,
        mode_id=mode.mode_id,
        deep_route=deep_route,
    )
    _set_runtime_route_preview(mode_id=mode.mode_id, preferred_model_id=preview_model_id)
    if _mode_support_gap(mode) is not None:
        unsupported_context = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "raw_message": message,
            "mode_id": mode.mode_id,
            "mode_kind": mode.mode_kind,
            "query": query,
            "deep_route": deep_route,
            "handoff": "unsupported mode in Little Lisa",
            "unsupported_requirement": _mode_support_gap(mode),
        }
        _set_last_turn_context(unsupported_context)
        response_text = _format_unsupported_mode(mode)
        _append_turn_log(context=unsupported_context, response_text=response_text)
        return response_text

    preferred_model_id = preview_model_id
    resolved_model = _resolve_model(preferred_model_id)
    run_context = _build_turn_context(
        raw_message=message,
        mode_id=mode.mode_id,
        query=query or message,
        deep_route=deep_route,
        preferred_model_id=preferred_model_id,
        resolved_model=resolved_model,
        mode=mode,
    )
    if not message.lstrip().startswith("/"):
        planner_debug = _last_frontdoor_debug() or {}
        planner_model_payload = planner_debug.get("resolved_model")
        planner_model_id = ""
        planner_model_provider = ""
        if isinstance(planner_model_payload, dict):
            planner_model_id = str(planner_model_payload.get("model_id") or "")
            planner_model_provider = str(planner_model_payload.get("provider") or "")

        run_context["frontdoor_route"] = {
            "selected_mode_id": mode.mode_id,
            "normalized_query": query or message,
            "model_alias": _frontdoor_model_alias(),
            "resolved_model_id": planner_model_id,
            "resolved_model_provider": planner_model_provider,
            "reasoning": planner_debug.get("planner_reasoning", ""),
            "confidence_score": planner_debug.get("confidence_score", 0.0),
            "fallback_used": planner_debug.get("fallback_used", False),
            "fallback_reason": planner_debug.get("fallback_reason", ""),
            "prompt_tokens": _coerce_non_negative_int(planner_debug.get("planner_prompt_tokens")),
            "completion_tokens": _coerce_non_negative_int(
                planner_debug.get("planner_completion_tokens")
            ),
            "total_tokens": _coerce_non_negative_int(planner_debug.get("planner_total_tokens")),
        }
        if planner_debug:
            run_context["frontdoor_debug"] = planner_debug
    _set_last_turn_context(run_context)

    hooks = _build_turn_hooks(run_context)
    applied_timeouts = _apply_mode_timeout_profile(mode.mode_id)
    run_context["applied_operation_timeouts"] = {
        "profile": _runtime_timeout_profile(),
        "count": len(applied_timeouts),
    }
    try:
        result: TurnResult = await run_turn(
            mode_id=mode.mode_id,
            query=query or message,
            hooks=hooks,
            preferred_model_id=preferred_model_id,
        )
    except OperationExecutionError as exc:
        run_context["operation_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        _append_log_event(
            level="warning",
            event="turn.operation_error",
            payload={
                "mode_id": mode.mode_id,
                "query": query or message,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
            debug_payload={
                "turn_context": run_context,
            },
        )
        response_text = _format_operation_error(exc)
        _append_turn_log(context=run_context, response_text=response_text)
        _append_thread_entry(
            mode_id=mode.mode_id,
            query=query or message,
            response_text=response_text,
        )
        return response_text
    finally:
        _restore_mode_timeouts(applied_timeouts)

    _apply_turn_usage(
        result,
        model_id=resolved_model.model_id,
        mode_id=mode.mode_id,
        stage_id=_runtime_current_stage_id(),
        frontdoor_routing=_last_frontdoor_debug(),
    )

    completion_context = run_context.get("completion")
    if not isinstance(completion_context, dict):
        completion_context = {}
    completion_context.update(
        {
            "mode_id": mode.mode_id,
            "query": query or message,
            "response_preview": result.response_text[:240],
            "response_chars": len(result.response_text),
            "tool_call_count": _coerce_non_negative_int(result.tool_call_count),
            "iteration_count": _coerce_non_negative_int(result.iteration_count),
            "prompt_tokens": _coerce_non_negative_int(result.prompt_tokens),
            "completion_tokens": _coerce_non_negative_int(result.completion_tokens),
            "total_tokens": _coerce_non_negative_int(result.total_tokens),
            "resolved_model": {
                "model_id": resolved_model.model_id,
                "provider": resolved_model.provider,
                "reasoning_model": resolved_model.reasoning_model,
            },
        }
    )
    run_context["completion"] = completion_context
    _set_last_turn_context(run_context)

    _append_turn_log(context=run_context, response_text=result.response_text)
    _append_thread_entry(
        mode_id=mode.mode_id,
        query=query or message,
        response_text=result.response_text,
    )
    return result.response_text


async def _run_once(message: str) -> None:
    try:
        response = await handle_message(message)
    except Exception as exc:  # pylint: disable=broad-except
        _render_runtime_error(exc, title="Turn Error")
        return
    if _runtime_debug_enabled():
        context = _last_turn_context()
        if context is not None:
            _render_turn_context(context)
    await _render_response(response)


def _resolved_data_dir_path(data_dir: str) -> Path:
    if data_dir.strip():
        return Path(data_dir).expanduser()
    env_value = os.getenv(_LITTLE_LISA_DATA_DIR_ENV, "").strip()
    if env_value:
        return Path(env_value).expanduser()
    return _DEFAULT_DATA_DIR


def _resolve_desktop_token(raw_token: str) -> str:
    configured = raw_token.strip()
    return configured or generate_desktop_token()


def _build_desktop_subprocess_args(  # noqa: PLR0912
    *,
    host: str,
    port: int,
    token: str,
    debug: bool,
    log_level: str,
    log: str,
    no_log: bool,
    data_dir: str,
    memory_db: str,
    thread_dir: str,
    no_thread: bool,
    session_context: str,
    timeout_profile: str,
    timezone: str,
    state_snapshot: str,
    streaming: str,
) -> list[str]:
    args = [
        "desktop",
        "--host",
        host,
        "--port",
        str(port),
        "--token",
        token,
        "--no-open-browser",
    ]
    if debug:
        args.append("--debug")
    if log_level.strip():
        args.extend(["--log-level", log_level])
    if log.strip():
        args.extend(["--log", log])
    if no_log:
        args.append("--no-log")
    if data_dir.strip():
        args.extend(["--data-dir", data_dir])
    if memory_db.strip():
        args.extend(["--memory-db", memory_db])
    if thread_dir.strip():
        args.extend(["--thread-dir", thread_dir])
    if no_thread:
        args.append("--no-thread")
    if session_context.strip():
        args.extend(["--session-context", session_context])
    if timeout_profile.strip():
        args.extend(["--timeout-profile", timeout_profile])
    if timezone.strip():
        args.extend(["--timezone", timezone])
    if state_snapshot.strip():
        args.extend(["--state-snapshot", state_snapshot])
    if streaming.strip():
        args.extend(["--streaming", streaming])
    return args


def _run_desktop_host(
    *,
    host: str,
    port: int,
    token: str,
    open_browser: bool,
) -> None:
    callbacks = DesktopCallbacks(handle_turn=handle_message, last_turn_context=_last_turn_context)
    run_desktop_server(
        callbacks=callbacks,
        host=host,
        port=port,
        token=token,
        open_browser=open_browser,
    )


def _build_repl_prompt_session() -> PromptSession[str] | None:
    if PromptSession is None or FileHistory is None or WordCompleter is None:
        return None
    history_path = _runtime_data_dir() / _REPL_HISTORY_FILE_NAME
    history_path.parent.mkdir(parents=True, exist_ok=True)
    return PromptSession(
        history=FileHistory(str(history_path)),
        completer=WordCompleter(list(_REPL_COMMAND_HINTS), ignore_case=True),
        complete_while_typing=True,
        multiline=False,
    )


def _run_default_no_args_mode(
    *,
    debug: bool,
    log_level: str,
    log: str,
    no_log: bool,
    data_dir: str,
    memory_db: str,
    thread_dir: str,
    no_thread: bool,
    session_context: str,
    timeout_profile: str,
    timezone: str,
    state_snapshot: str,
    streaming: str,
    desktop_host: str,
    desktop_port: int,
    desktop_token: str,
) -> None:
    resolved_token = _resolve_desktop_token(desktop_token)
    if _DEFAULT_DESKTOP_NO_ARGS_MODE == "tray":
        tray_args = _build_desktop_subprocess_args(
            host=desktop_host,
            port=desktop_port,
            token=resolved_token,
            debug=debug,
            log_level=log_level,
            log=log,
            no_log=no_log,
            data_dir=data_dir,
            memory_db=memory_db,
            thread_dir=thread_dir,
            no_thread=no_thread,
            session_context=session_context,
            timeout_profile=timeout_profile,
            timezone=timezone,
            state_snapshot=state_snapshot,
            streaming=streaming,
        )
        logs_dir = str(_resolved_data_dir_path(data_dir))
        try:
            run_tray(
                TrayConfig(
                    host=desktop_host,
                    port=desktop_port,
                    token=resolved_token,
                    desktop_args=tray_args,
                    open_on_start=_DEFAULT_DESKTOP_OPEN_ON_START,
                    logs_dir=logs_dir,
                )
            )
            return
        except Exception as exc:  # pylint: disable=broad-except
            console.print(
                f"[{STYLE_TEXT_MUTED}]Tray unavailable ({type(exc).__name__}: {exc}). "
                "Launching desktop web host directly.[/]"
            )

    _initialize_runtime_or_exit(
        debug=debug,
        log_level=log_level,
        log_path=log,
        no_log=no_log,
        data_dir=data_dir,
        memory_db=memory_db,
        thread_dir=thread_dir,
        no_thread=no_thread,
        user_slug=session_context,
        timeout_profile=timeout_profile,
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
    )
    _run_desktop_host(
        host=desktop_host,
        port=desktop_port,
        token=resolved_token,
        open_browser=True,
    )


async def _run_repl() -> None:
    _print_startup_banner()
    console.print(
        Panel(
            "Type /help, /help capabilities, /debug context, "
            "/debug frontdoor, /search <query>, /deep <query>, or /quit.",
            title="Session",
            border_style=STYLE_BORDER_SESSION,
        )
    )
    prompt_session = _build_repl_prompt_session()
    if prompt_session is None:
        console.print(f"[{STYLE_TEXT_MUTED}]prompt_toolkit unavailable; using basic input mode.[/]")
    while True:
        try:
            _render_status_line()
            if prompt_session is not None:
                message = (await prompt_session.prompt_async("> ")).strip()
            else:
                message = console.input(f"[{STYLE_PROMPT}]> [/]").strip()
        except KeyboardInterrupt:
            console.print()
            continue
        except EOFError:
            console.print(Rule(style=STYLE_RULE))
            _render_session_exit_summary()
            return
        if not message:
            continue
        if message.lower() in {"exit", "quit", "/exit", "/quit"}:
            console.print(Rule(style=STYLE_RULE))
            _render_session_exit_summary()
            console.print(f"[{STYLE_TEXT_MUTED}]Session closed.[/]")
            return
        turn_start = time.perf_counter()
        try:
            response = await handle_message(message)
        except Exception as exc:  # pylint: disable=broad-except
            elapsed_ms = (time.perf_counter() - turn_start) * 1000.0
            _record_session_turn(succeeded=False, latency_ms=elapsed_ms)
            _render_runtime_error(exc, title="Turn Error")
            console.print(Rule(style=STYLE_RULE))
            continue
        elapsed_ms = (time.perf_counter() - turn_start) * 1000.0
        _record_session_turn(
            succeeded=True,
            latency_ms=elapsed_ms,
            mode_id=_runtime_current_mode_id(),
        )
        if _runtime_debug_enabled():
            context = _last_turn_context()
            if context is not None:
                _render_turn_context(context)
        await _render_response(response)
        console.print(Rule(style=STYLE_RULE))


def _print_startup_banner() -> None:
    """Print a compact startup banner."""

    title = Text()
    title.append("Ready", style="bold steel_blue3")
    title.append("  ")
    title.append("interactive terminal host", style=STYLE_TEXT_MUTED)
    console.print(title)
    console.print()


def _status_segments(compact_level: int = _STATUS_COMPACT_FULL) -> list[tuple[str, str]]:
    profile = _runtime_timeout_profile()
    profile_emoji = {"fast": "⚡", "balanced": "⚖", "deep": "🧠"}.get(profile, "⏱")
    profile_badge = f"{profile_emoji} {profile}"
    user = _runtime_user_slug().strip()
    user_segment = (f"👤 {user}", STYLE_STATUS_INFO) if user else None
    mode_id = _runtime_current_mode_id()
    stage_id = _runtime_current_stage_id()
    mode_badge = f"mode {mode_id}{'' if stage_id == '-' else '/' + stage_id}"
    mode_badge_short = f"m {mode_id}{'' if stage_id == '-' else '/' + stage_id}"
    model_badge = f"{_runtime_next_model_provider()}:{_runtime_next_model_id()}"
    turn_token_total = int(_RUNTIME_STATE.get("turn_total_tokens", 0) or 0)
    session_token_total = int(_RUNTIME_STATE.get("session_total_tokens", 0) or 0)
    token_badge = f"tokens {turn_token_total}/{session_token_total}"
    token_badge_short = f"tc {turn_token_total}/{session_token_total}"
    debug_enabled = _runtime_debug_enabled()

    if compact_level <= _STATUS_COMPACT_FULL:
        return _finalize_status_segments(
            segments=[
                (profile_badge, STYLE_STATUS_INFO),
                (mode_badge, STYLE_STATUS_METRIC),
                (model_badge, STYLE_STATUS_METRIC),
                (token_badge, STYLE_STATUS_METRIC),
            ],
            user_segment=user_segment,
            debug_enabled=debug_enabled,
            abbreviated_debug=False,
        )
    if compact_level == _STATUS_COMPACT_ABBREVIATED:
        return _finalize_status_segments(
            segments=[
                (profile_badge, STYLE_STATUS_INFO),
                (mode_badge_short, STYLE_STATUS_METRIC),
                (model_badge, STYLE_STATUS_METRIC),
                (token_badge_short, STYLE_STATUS_METRIC),
            ],
            user_segment=user_segment,
            debug_enabled=debug_enabled,
            abbreviated_debug=True,
        )
    if compact_level == _STATUS_COMPACT_TRIMMED:
        return _finalize_status_segments(
            segments=[
                (profile_badge, STYLE_STATUS_INFO),
                (mode_badge_short, STYLE_STATUS_METRIC),
                (model_badge, STYLE_STATUS_METRIC),
                (token_badge_short, STYLE_STATUS_METRIC),
            ],
            user_segment=None,
            debug_enabled=debug_enabled,
            abbreviated_debug=True,
        )
    return _finalize_status_segments(
        segments=[
            (mode_badge_short, STYLE_STATUS_METRIC),
            (model_badge, STYLE_STATUS_METRIC),
            (token_badge_short, STYLE_STATUS_METRIC),
        ],
        user_segment=None,
        debug_enabled=debug_enabled,
        abbreviated_debug=True,
    )


def _finalize_status_segments(
    *,
    segments: list[tuple[str, str]],
    user_segment: tuple[str, str] | None,
    debug_enabled: bool,
    abbreviated_debug: bool,
) -> list[tuple[str, str]]:
    if user_segment is not None:
        segments.insert(1, user_segment)
    if debug_enabled:
        debug_label = "dbg on" if abbreviated_debug else "debug on"
        segments.append((debug_label, STYLE_STATUS_DEBUG_ON))
    return segments


def _status_time_badge() -> tuple[str, str]:
    timezone_name = _runtime_timezone()
    try:
        status_dt = datetime.now(ZoneInfo(timezone_name))
    except ZoneInfoNotFoundError:
        status_dt = datetime.now().astimezone()
    timezone_label = timezone_name
    if "/" in timezone_name:
        tz_abbrev_raw = str(status_dt.tzname() or "").strip()
        if tz_abbrev_raw:
            if " " in tz_abbrev_raw:
                initials = "".join(
                    token[0] for token in tz_abbrev_raw.split() if token and token[0].isalpha()
                ).upper()
                timezone_label = (
                    initials
                    if len(initials) >= _TIMEZONE_INITIALISM_MIN_LEN
                    else tz_abbrev_raw.upper()
                )
            else:
                timezone_label = tz_abbrev_raw.upper()

        # Windows fallback: if tzname resolves to a long phrase, prefer canonical US abbreviations.
        if " " in timezone_label:
            is_daylight = bool(status_dt.dst())
            us_zone_abbrevs = {
                "America/New_York": ("EST", "EDT"),
                "America/Chicago": ("CST", "CDT"),
                "America/Denver": ("MST", "MDT"),
                "America/Los_Angeles": ("PST", "PDT"),
            }
            pair = us_zone_abbrevs.get(timezone_name)
            if pair is not None:
                timezone_label = pair[1] if is_daylight else pair[0]
    now_local = status_dt.strftime("%H:%M:%S")
    return f"🕒 {now_local} {timezone_label}", STYLE_STATUS_INFO


def _estimated_status_line_width(segments: list[tuple[str, str]]) -> int:
    """Estimate rendered width for status segments and separators."""

    if not segments:
        return 0
    label_width = sum(cell_len(label) + 2 for label, _ in segments)
    separator_width = max(0, len(segments) - 1)
    return label_width + separator_width


def _fitted_status_segments() -> list[tuple[str, str]]:
    """Return the richest status segments that fit the current terminal width."""

    time_label, _ = _status_time_badge()
    time_width = cell_len(time_label) + _STATUS_TIME_PADDING_WIDTH
    # Reserve 1 char for the inter-column gap and fit left badges against true cell widths.
    max_width = max(0, int(console.size.width) - time_width - 1)
    for compact_level in (
        _STATUS_COMPACT_FULL,
        _STATUS_COMPACT_ABBREVIATED,
        _STATUS_COMPACT_TRIMMED,
        _STATUS_COMPACT_MINIMAL,
    ):
        segments = _status_segments(compact_level)
        if _estimated_status_line_width(segments) <= max_width:
            return segments
    return _status_segments(_STATUS_COMPACT_MINIMAL)


def _render_status_line() -> None:
    """Render a powerline-style status row above the input prompt."""

    left = Text()
    segments = _fitted_status_segments()
    for index, (label, style) in enumerate(segments):
        left.append(f" {label} ", style=style)
        if index < len(segments) - 1:
            left.append("", style=STYLE_STATUS_SEPARATOR)

    time_label, time_style = _status_time_badge()
    right = Text(f" {time_label} ", style=time_style)

    left_width = cell_len(left.plain)
    right_width = cell_len(right.plain)
    padding = max(1, int(console.size.width) - left_width - right_width)
    line = Text()
    line.append(left)
    line.append(" " * padding)
    line.append(right)
    # Keep the time badge atomic on a single row; compacting handles width pressure first.
    console.print(line, no_wrap=True)


async def _render_response(response_text: str) -> None:
    """Render one response, optionally streaming text for terminal UX."""

    stream_text = _runtime_stream_enabled() and _runtime_stream_mode() in {"text", "hybrid"}
    if not stream_text:
        console.print(
            Panel(
                Markdown(response_text),
                title="Response",
                border_style=STYLE_BORDER_RESPONSE,
            )
        )
        return

    console.print(
        Panel(
            "Streaming response...",
            title="Response",
            border_style=STYLE_BORDER_RESPONSE,
        )
    )
    delay_seconds = _runtime_stream_char_delay_ms() / 1000.0
    for char in response_text:
        console.file.write(char)
        console.file.flush()
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
    if response_text:
        console.file.write("\n")
        console.file.flush()


def _render_runtime_error(exc: Exception, *, title: str = "Error") -> None:
    error_text = f"{type(exc).__name__}: {exc}"
    _append_log_event(
        level="error",
        event="runtime.error",
        payload={
            "title": title,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        },
    )
    console.print(
        Panel(
            error_text,
            title=title,
            border_style="indian_red",
        )
    )


def _resolve_runtime_thread_strategy(
    *,
    no_thread: bool,
    context_active: bool,
    context_ref: str,
    normalized_user: str,
    thread_from_cli: Path | None,
    thread_from_env: Path | None,
    data_dir: Path,
) -> tuple[Path | None, Path | None]:
    """Determine thread directory and path based on context and flags.

    Threads are only enabled when an explicit --thread-dir is provided, when
    the data directory resolves to a non-default location, or when a session
    context is active. Otherwise thread logging is disabled.
    """
    if no_thread:
        return None, None

    base_dir: Path | None = None
    if thread_from_cli:
        base_dir = thread_from_cli
    elif thread_from_env:
        base_dir = thread_from_env
    elif data_dir != _DEFAULT_DATA_DIR:
        base_dir = data_dir / "threads"

    if base_dir is None:
        if not context_active:
            return None, None
        base_dir = data_dir / "threads"

    path: Path | None = None
    if context_active:
        p = Path(context_ref)
        if p.is_dir():
            target_dir = base_dir / p.name
            return target_dir, target_dir / f"{p.name}.jsonl"
        path = base_dir / f"{normalized_user}.jsonl"
    elif normalized_user:
        path = base_dir / f"{normalized_user}.jsonl"

    return base_dir, path


def _initialize_runtime(
    *,
    debug: bool,
    log_level: str = "",
    log_path: str = "",
    no_log: bool = False,
    data_dir: str = "",
    memory_db: str = "",
    thread_dir: str = "",
    no_thread: bool = False,
    user_slug: str,
    timeout_profile: str,
    timezone: str = "",
    state_snapshot_path: str,
    streaming: str = "",
) -> None:
    """Initialize Little Lisa runtime before each command entry point.

    The model manifest is always reloaded from disk at startup so the runtime
    reflects the latest models.yaml without requiring a manual flag.
    """

    _load_local_env_file()
    setup_framework()
    configure_runtime()
    _RUNTIME_STATE["debug"] = debug
    _RUNTIME_STATE["log_level"] = _resolve_log_level(log_level, debug_enabled=debug)

    # Configure artifacts, memory, and thread log directories
    data_from_cli = Path(data_dir).expanduser() if data_dir.strip() else None
    data_from_env = (
        Path(os.getenv(_LITTLE_LISA_DATA_DIR_ENV, "")).expanduser()
        if os.getenv(_LITTLE_LISA_DATA_DIR_ENV, "").strip()
        else None
    )
    _RUNTIME_STATE["data_dir"] = data_from_cli or data_from_env or _DEFAULT_DATA_DIR

    # Resolve turn log path: --no-log disables; --log overrides; default is
    # turns.jsonl inside the resolved artifacts directory.
    if no_log:
        _RUNTIME_STATE["log_path"] = None
    elif log_path.strip():
        _RUNTIME_STATE["log_path"] = Path(log_path).expanduser()
    else:
        _RUNTIME_STATE["log_path"] = _runtime_data_dir() / _DEFAULT_TURN_LOG_NAME

    memory_from_cli = Path(memory_db).expanduser() if memory_db.strip() else None
    memory_from_env = (
        Path(os.getenv(_LITTLE_LISA_MEMORY_DB_ENV, "")).expanduser()
        if os.getenv(_LITTLE_LISA_MEMORY_DB_ENV, "").strip()
        else None
    )
    _RUNTIME_STATE["memory_db_path"] = (
        memory_from_cli or memory_from_env or _runtime_data_dir() / "memory.json"
    )

    thread_from_cli = Path(thread_dir).expanduser() if thread_dir.strip() else None
    thread_from_env = (
        Path(os.getenv(_LITTLE_LISA_THREAD_DIR_ENV, "")).expanduser()
        if os.getenv(_LITTLE_LISA_THREAD_DIR_ENV, "").strip()
        else None
    )

    # Resolve session context: prefer explicit CLI `--session-context` when
    # provided. Otherwise, check for session_context metadata in artifacts.
    if str(user_slug).strip():
        normalized_user, context_ref, context_active = _resolve_session_context(user_slug)
    else:
        # Check artifacts for 'session_context' file or folder
        normalized_user, context_ref, context_active = _resolve_session_context("")

    # Determine thread strategy
    thread_dir, thread_path = _resolve_runtime_thread_strategy(
        no_thread=no_thread,
        context_active=context_active,
        context_ref=context_ref,
        normalized_user=normalized_user,
        thread_from_cli=thread_from_cli,
        thread_from_env=thread_from_env,
        data_dir=_runtime_data_dir(),
    )
    _RUNTIME_STATE["thread_dir"] = thread_dir
    _RUNTIME_STATE["thread_path"] = thread_path

    _RUNTIME_STATE["user_slug"] = normalized_user
    _RUNTIME_STATE["session_context_ref"] = context_ref
    _RUNTIME_STATE["session_context_active"] = context_active

    _RUNTIME_STATE["timeout_profile"] = _resolve_timeout_profile(timeout_profile)

    _RUNTIME_STATE["session_started_utc"] = datetime.now(UTC)
    _RUNTIME_STATE["session_turn_count"] = 0
    _RUNTIME_STATE["session_success_count"] = 0
    _RUNTIME_STATE["session_error_count"] = 0
    _RUNTIME_STATE["session_total_latency_ms"] = 0.0
    _RUNTIME_STATE["session_mode_counts"] = {}
    resolved_timezone, timezone_source = _resolve_timezone(timezone)
    _RUNTIME_STATE["timezone"] = resolved_timezone
    _RUNTIME_STATE["timezone_source"] = timezone_source
    default_mode_id = "default"
    if default_mode_id not in get_workflow_catalog():
        default_mode_id = next(iter(get_workflow_catalog().keys()), "default")
    _set_runtime_route_preview(
        mode_id=default_mode_id,
        preferred_model_id=_model_alias_for_mode(default_mode_id),
    )
    _RUNTIME_STATE["state_snapshot_path"] = (
        Path(state_snapshot_path).expanduser() if state_snapshot_path.strip() else None
    )
    _RUNTIME_STATE["state_snapshot"] = _load_state_snapshot(state_snapshot_path)
    stream_mode = "none"
    stream_char_delay_ms = 0
    if streaming.strip():
        stream_mode, stream_char_delay_ms = parse_streaming_option(streaming)
    force_stream = stream_mode != "none"
    force_no_stream = stream_mode == "none"
    _RUNTIME_STATE["stream_enabled"] = resolve_stream_enabled(
        force_stream=force_stream,
        force_no_stream=force_no_stream,
        is_tty=bool(sys.stdout.isatty()),
    )
    normalized_stream_mode = stream_mode.strip().lower()
    _RUNTIME_STATE["stream_mode"] = (
        normalized_stream_mode
        if normalized_stream_mode in {"stage", "text", "hybrid"}
        else "hybrid"
    )
    _RUNTIME_STATE["stream_char_delay_ms"] = max(0, int(stream_char_delay_ms))
    _RUNTIME_STATE["streaming_mode"] = (
        _RUNTIME_STATE["stream_mode"] if _RUNTIME_STATE["stream_enabled"] else "none"
    )
    _RUNTIME_STATE["streaming_char_delay_ms"] = _RUNTIME_STATE["stream_char_delay_ms"]

    _append_log_event(
        level="info",
        event="runtime.initialized",
        payload={
            "session_context": normalized_user or "none",
            "session_context_ref": context_ref,
            "session_context_active": context_active,
            "timezone": resolved_timezone,
            "timezone_source": timezone_source,
            "timeout_profile": _runtime_timeout_profile(),
            "log_path": str(_runtime_log_path() or ""),
            "streaming_mode": _runtime_streaming_mode(),
            "streaming_enabled": _runtime_stream_enabled(),
            "streaming_char_delay_ms": _runtime_streaming_char_delay_ms(),
        },
        debug_payload={
            "state_snapshot_path": str(_RUNTIME_STATE.get("state_snapshot_path") or ""),
            "state_snapshot_loaded": isinstance(_RUNTIME_STATE.get("state_snapshot"), dict),
        },
    )

    reload_model_manifest()


def _normalize_legacy_cli_args(raw_args: list[str]) -> list[str]:
    """Map legacy argparse flags to Typer subcommands for compatibility."""

    if not raw_args:
        return raw_args
    if raw_args[0] == "--list-modes":
        return ["modes", *raw_args[1:]]
    if len(raw_args) >= _LEGACY_ONCE_ARG_COUNT and raw_args[0] == "--once":
        return ["once", raw_args[1], *raw_args[2:]]

    known_commands = {"cli", "desktop", "verify", "modes", "models", "once", "help"}

    # If the user passed options before the subcommand, typer will consume them
    # in the callback and drop them for the subcommand. Move the subcommand to the front
    # to ensure the subcommand receives all its intended options.
    cmd_index = -1
    i = 0
    while i < len(raw_args):
        arg = raw_args[i]
        if arg in known_commands:
            cmd_index = i
            break
        if arg.startswith("-") and "=" not in arg:
            # Skip the next token if it's the value for an option, unless it's a known boolean flag.
            if arg in {"--version", "--debug", "--no-log", "--no-thread"}:
                pass
            else:
                i += 1
        i += 1

    if cmd_index > 0:
        # Move the known subcommand to position 0
        cmd = raw_args.pop(cmd_index)
        raw_args.insert(0, cmd)

    return raw_args


def _initialize_runtime_or_exit(**kwargs: Any) -> None:
    try:
        _initialize_runtime(**kwargs)
    except (ValueError, FileNotFoundError) as exc:
        _render_runtime_error(exc, title="Startup Error")
        raise typer.Exit(code=2) from exc


@app.callback(invoke_without_command=True)
def _cli_callback(
    ctx: typer.Context,
    version: bool = _VERSION_OPTION,
    debug: DebugOption = False,
    log_level: LogLevelOption = "",
    log: LogOption = "",
    no_log: NoLogOption = False,
    data_dir: DataDirOption = "",
    memory_db: MemoryDbOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    timeout_profile: TimeoutProfileOption = "",
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    desktop_host: str = typer.Option(
        DEFAULT_DESKTOP_HOST,
        "--desktop-host",
        help="Local desktop host bind (default: 127.0.0.1).",
    ),
    desktop_port: int = typer.Option(
        DEFAULT_DESKTOP_PORT,
        "--desktop-port",
        min=1,
        max=65535,
        help="Local desktop port for browser and tray launch.",
    ),
    desktop_token: str = typer.Option(
        "",
        "--desktop-token",
        help="Optional fixed localhost token for desktop API auth.",
    ),
    streaming: StreamingOption = "none",
) -> None:
    if version:
        console.print(f"little-lisa v{_get_version()}")
        raise typer.Exit()
    if ctx.invoked_subcommand is not None:
        return
    _run_default_no_args_mode(
        debug=debug,
        log_level=log_level,
        log=log,
        no_log=no_log,
        data_dir=data_dir,
        memory_db=memory_db,
        thread_dir=thread_dir,
        no_thread=no_thread,
        session_context=session_context,
        timeout_profile=timeout_profile,
        timezone=timezone,
        state_snapshot=state_snapshot,
        streaming=streaming,
        desktop_host=desktop_host,
        desktop_port=desktop_port,
        desktop_token=desktop_token,
    )


@app.command("cli", help="Run the interactive Little Lisa terminal REPL.")
def _cli_command(
    debug: DebugOption = False,
    log_level: LogLevelOption = "",
    log: LogOption = "",
    no_log: NoLogOption = False,
    data_dir: DataDirOption = "",
    memory_db: MemoryDbOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    timeout_profile: TimeoutProfileOption = "",
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "none",
) -> None:
    _initialize_runtime_or_exit(
        debug=debug,
        log_level=log_level,
        log_path=log,
        no_log=no_log,
        data_dir=data_dir,
        memory_db=memory_db,
        thread_dir=thread_dir,
        no_thread=no_thread,
        user_slug=session_context,
        timeout_profile=timeout_profile,
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
    )
    asyncio.run(_run_repl())


@app.command("desktop", help="Run the local Little Lisa desktop web host.")
def _desktop_command(
    host: str = typer.Option(
        DEFAULT_DESKTOP_HOST,
        "--host",
        help="Local desktop host bind (default: 127.0.0.1).",
    ),
    port: int = typer.Option(
        DEFAULT_DESKTOP_PORT,
        "--port",
        min=1,
        max=65535,
        help="Local desktop port for browser and tray launch.",
    ),
    token: str = typer.Option(
        "",
        "--token",
        help="Optional fixed localhost token for desktop API auth.",
    ),
    open_browser: bool = typer.Option(
        True,
        "--open-browser/--no-open-browser",
        help="Open the local desktop URL in the default browser at startup.",
    ),
    debug: DebugOption = False,
    log_level: LogLevelOption = "",
    log: LogOption = "",
    no_log: NoLogOption = False,
    data_dir: DataDirOption = "",
    memory_db: MemoryDbOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    timeout_profile: TimeoutProfileOption = "",
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "none",
) -> None:
    _initialize_runtime_or_exit(
        debug=debug,
        log_level=log_level,
        log_path=log,
        no_log=no_log,
        data_dir=data_dir,
        memory_db=memory_db,
        thread_dir=thread_dir,
        no_thread=no_thread,
        user_slug=session_context,
        timeout_profile=timeout_profile,
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
    )
    _run_desktop_host(
        host=host,
        port=port,
        token=_resolve_desktop_token(token),
        open_browser=open_browser,
    )


# ---------------------------------------------------------------------------
# verify command
# ---------------------------------------------------------------------------

_VERIFY_PASS = "[green]PASS[/green]"
_VERIFY_WARN = "[yellow]WARN[/yellow]"
_VERIFY_FAIL = "[red]FAIL[/red]"
_KEY_REDACT_VISIBLE_CHARS = 4
_MIN_PYTHON_MAJOR = 3
_MIN_PYTHON_MINOR = 11
_HTTP_OK = 200
_OLLAMA_MODEL_PREVIEW_LIMIT = 5


def _redact_key(value: str) -> str:
    """Redact an API key, showing only the last 4 characters."""
    if not value:
        return "(not set)"
    if len(value) <= _KEY_REDACT_VISIBLE_CHARS:
        return "*" * len(value)
    return "*" * (len(value) - _KEY_REDACT_VISIBLE_CHARS) + value[-_KEY_REDACT_VISIBLE_CHARS:]


def _verify_check_python() -> tuple[str, str]:
    """Check Python version meets the minimum requirement."""
    version = platform.python_version()
    major, minor, *_ = version.split(".")
    if int(major) >= _MIN_PYTHON_MAJOR and int(minor) >= _MIN_PYTHON_MINOR:
        return _VERIFY_PASS, f"Python {version}"
    return _VERIFY_FAIL, f"Python {version} (need >= 3.11)"


def _verify_check_env() -> tuple[str, str]:
    """Check .env file exists in the little_lisa directory."""
    env_path = _LITTLE_LISA_ENV_FILE
    if env_path.exists():
        return _VERIFY_PASS, f".env found ({env_path})"
    return _VERIFY_WARN, f".env not found at {env_path} — copy .env.example to .env"


def _verify_check_manifest() -> tuple[str, str]:
    """Check the model manifest loads without errors."""
    try:
        manifest = load_model_manifest(
            _manifest_path_from_env_runtime(env_var=_LITTLE_LISA_MANIFEST_ENV)
        )
        model_count = len(manifest.models)
        if model_count == 0:
            return _VERIFY_FAIL, "Model manifest loaded but contains zero models"
        model_ids = ", ".join(m.model_id for m in manifest.models)
        return _VERIFY_PASS, f"{model_count} model(s): {model_ids}"
    except Exception as exc:
        return _VERIFY_FAIL, f"Model manifest failed to load: {exc}"


def _verify_check_pack_catalog() -> tuple[str, str]:
    """Check the pack catalog loads without errors."""
    try:
        setup_framework()
        catalog = get_workflow_catalog()
        mode_count = len(catalog)
        if mode_count == 0:
            return _VERIFY_WARN, "Pack catalog loaded but contains zero modes"
        mode_ids = ", ".join(sorted(catalog.keys()))
        return _VERIFY_PASS, f"{mode_count} mode(s): {mode_ids}"
    except Exception as exc:
        return _VERIFY_FAIL, f"Pack catalog failed to load: {exc}"


def _verify_check_ollama() -> tuple[str, str]:
    """Check Ollama is reachable and has the configured models pulled."""
    configured_host = (
        os.getenv(_LITTLE_LISA_OLLAMA_HOST_ENV, "").strip()
        or os.getenv("OLLAMA_HOST", "").strip()
        or "127.0.0.1:11434"
    )
    base_url = _normalize_ollama_host_to_base_url(configured_host)
    if not base_url:
        return _VERIFY_WARN, "Ollama host not configured"

    async def _ping() -> tuple[str, str]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                ollama_api = base_url.rstrip("/").removesuffix("/v1") + "/api/tags"
                resp = await client.get(ollama_api)
                if resp.status_code == _HTTP_OK:
                    data = resp.json()
                    models = [m.get("name", "?") for m in data.get("models", [])]
                    if models:
                        preview = ", ".join(models[:_OLLAMA_MODEL_PREVIEW_LIMIT])
                        suffix = "..." if len(models) > _OLLAMA_MODEL_PREVIEW_LIMIT else ""
                        return _VERIFY_PASS, (
                            f"Ollama reachable, {len(models)} model(s) pulled: {preview}{suffix}"
                        )
                    return _VERIFY_WARN, "Ollama reachable but no models pulled"
                return _VERIFY_WARN, f"Ollama returned HTTP {resp.status_code}"
        except Exception as exc:
            return _VERIFY_WARN, f"Ollama not reachable at {base_url}: {exc}"

    try:
        return asyncio.run(_ping())
    except Exception as exc:
        return _VERIFY_WARN, f"Ollama check failed: {exc}"


def _verify_check_web_search() -> tuple[str, str]:
    """Check web search is configured (optional)."""
    if web_search_is_configured():
        return _VERIFY_PASS, "Web search configured"
    return _VERIFY_WARN, "Web search not configured (optional — set WEB_SEARCH_API_KEY)"


def _verify_check_provider_keys() -> tuple[str, str]:
    """Check which provider API keys are configured (redacted)."""
    parts: list[str] = []

    azure_key = (
        os.getenv(_LITTLE_LISA_AZURE_API_KEY_ENV, "").strip()
        or os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    )
    azure_endpoint = (
        os.getenv(_LITTLE_LISA_AZURE_ENDPOINT_ENV, "").strip()
        or os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    )
    if azure_key and azure_endpoint:
        parts.append(f"Azure OpenAI: {_redact_key(azure_key)} @ {azure_endpoint}")
    elif azure_key or azure_endpoint:
        parts.append("Azure OpenAI: partially configured")

    openai_key = (
        os.getenv(_LITTLE_LISA_OPENAI_API_KEY_ENV, "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
    )
    if openai_key:
        parts.append(f"OpenAI: {_redact_key(openai_key)}")

    anthropic_key = (
        os.getenv(_LITTLE_LISA_ANTHROPIC_API_KEY_ENV, "").strip()
        or os.getenv("ANTHROPIC_API_KEY", "").strip()
    )
    if anthropic_key:
        parts.append(f"Anthropic: {_redact_key(anthropic_key)}")

    google_key = (
        os.getenv(_LITTLE_LISA_GOOGLE_API_KEY_ENV, "").strip()
        or os.getenv("GOOGLE_API_KEY", "").strip()
    )
    if google_key:
        parts.append(f"Google Gemini: {_redact_key(google_key)}")

    ollama_key = (
        os.getenv(_LITTLE_LISA_OLLAMA_API_KEY_ENV, "").strip()
        or os.getenv("OLLAMA_API_KEY", "").strip()
    )
    if ollama_key:
        parts.append(f"Ollama API key: {_redact_key(ollama_key)}")
    else:
        parts.append("Ollama: no auth (default)")

    if not parts:
        return _VERIFY_WARN, "No provider keys configured"
    return _VERIFY_PASS, "; ".join(parts)


def _verify_check_paths() -> tuple[str, str]:
    """Show resolved runtime paths."""
    parts: list[str] = []
    parts.append(f"Data dir: {_runtime_data_dir()}")
    parts.append(f"Memory DB: {_runtime_memory_db_path()}")
    tld = _runtime_thread_dir()
    if tld:
        parts.append(f"Thread logs: {tld}")
    tp = _runtime_thread_path()
    if tp:
        parts.append(f"Thread path: {tp}")

    if not tld and not tp:
        parts.append("Thread logs: disabled")
    return _VERIFY_PASS, "; ".join(parts)


def _verify_check_session_context() -> tuple[str, str]:
    """Check if session context is active and what it is."""
    active = _RUNTIME_STATE.get("session_context_active")
    ref = _RUNTIME_STATE.get("session_context_ref")
    metadata = _RUNTIME_STATE.get("session_context_metadata") or {}

    if not active:
        return "[yellow]NONE[/yellow]", "No session context active"

    details = f"Context: {ref}"
    if metadata.get("thread_name"):
        details += f" (thread_name: {metadata['thread_name']})"

    return _VERIFY_PASS, details


@app.command("verify", help="Run verification checks of the Little Lisa environment.")
def _verify_command(
    debug: DebugOption = False,
    log_level: LogLevelOption = "",
    log: LogOption = "",
    no_log: NoLogOption = False,
    data_dir: DataDirOption = "",
    memory_db: MemoryDbOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    timeout_profile: TimeoutProfileOption = "",
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "none",
) -> None:
    """Run verification checks of the Little Lisa environment.

    Accepts the same runtime options as ``cli`` and ``desktop`` so it
    validates the exact startup shape you will actually run with.
    """
    _initialize_runtime_or_exit(
        debug=debug,
        log_level=log_level,
        log_path=log,
        no_log=no_log,
        data_dir=data_dir,
        memory_db=memory_db,
        thread_dir=thread_dir,
        no_thread=no_thread,
        user_slug=session_context,
        timeout_profile=timeout_profile,
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
    )
    checks: list[tuple[str, str, str]] = [
        ("Python", *_verify_check_python()),
        (".env", *_verify_check_env()),
        ("Session context", *_verify_check_session_context()),
        ("Model manifest", *_verify_check_manifest()),
        ("Pack catalog", *_verify_check_pack_catalog()),
        ("Provider keys", *_verify_check_provider_keys()),
        ("Ollama", *_verify_check_ollama()),
        ("Web search", *_verify_check_web_search()),
        ("Paths", *_verify_check_paths()),
    ]

    table = Table(title="little-lisa verify")
    table.add_column("Check", style="bold")
    table.add_column("Status")
    table.add_column("Detail")

    failures = 0
    for name, status, detail in checks:
        table.add_row(name, status, detail)
        if "FAIL" in status:
            failures += 1

    console.print(table)

    if failures:
        console.print(f"\n[red]{failures} check(s) failed.[/red]")
        raise typer.Exit(code=1)
    console.print("\n[green]All checks passed.[/green]")


# ---------------------------------------------------------------------------
# --setup wizard command
# ---------------------------------------------------------------------------

_SETUP_PROVIDERS = {
    "1": ("ollama", "Ollama (local, no API key needed)"),
    "2": ("openai", "OpenAI (requires API key)"),
    "3": ("azure_openai", "Azure OpenAI (requires endpoint + API key)"),
    "4": ("anthropic", "Anthropic (requires API key)"),
    "5": ("google", "Google Gemini (requires API key)"),
}


def _setup_write_env(env_path: Path, updates: dict[str, str]) -> None:
    """Write or update .env with the given key-value pairs, preserving existing content."""
    existing: dict[str, str] = {}
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            existing[key.strip()] = value.strip()

    existing.update(updates)

    lines = [
        "# Little Lisa environment configuration",
        "# Generated by: little-lisa --setup",
        "",
    ]
    for key, value in sorted(existing.items()):
        lines.append(f"{key}={value}")
    lines.append("")
    env_path.write_text("\n".join(lines), encoding="utf-8")


async def _setup_validate_ollama() -> tuple[bool, str]:
    """Validate Ollama is reachable."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("http://127.0.0.1:11434/api/tags")
            if resp.status_code == _HTTP_OK:
                return True, "Ollama is reachable"
            return False, f"Ollama returned HTTP {resp.status_code}"
    except Exception as exc:
        return False, f"Cannot reach Ollama: {exc}"


async def _setup_validate_openai(api_key: str) -> tuple[bool, str]:
    """Validate OpenAI API key with a cheap models list call."""
    try:
        client = AsyncOpenAI(api_key=api_key)
        await client.models.list()
        return True, "OpenAI API key is valid"
    except Exception as exc:
        return False, f"OpenAI validation failed: {exc}"


async def _setup_validate_google(api_key: str) -> tuple[bool, str]:
    """Validate Google Gemini API key with one cheap models list call."""
    try:
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        await client.models.list()
        return True, "Google Gemini API key is valid"
    except Exception as exc:
        return False, f"Google Gemini validation failed: {exc}"


async def _setup_validate_azure(endpoint: str, api_key: str) -> tuple[bool, str]:
    """Validate Azure OpenAI endpoint and key."""
    try:
        client = AsyncAzureOpenAI(
            api_version="2024-10-21",
            azure_endpoint=endpoint,
            api_key=api_key,
        )
        await client.models.list()
        return True, "Azure OpenAI endpoint is reachable"
    except Exception as exc:
        return False, f"Azure OpenAI validation failed: {exc}"


async def _setup_validate_anthropic(api_key: str) -> tuple[bool, str]:
    """Validate Anthropic API key."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://api.anthropic.com/v1/models",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
            )
            if resp.status_code == _HTTP_OK:
                return True, "Anthropic API key is valid"
            return False, f"Anthropic returned HTTP {resp.status_code}"
    except Exception as exc:
        return False, f"Anthropic validation failed: {exc}"


@app.command("setup", help="Interactive wizard to configure Little Lisa providers.")
def _setup_command() -> None:  # noqa: PLR0912, PLR0915
    """Interactive wizard to configure Little Lisa providers."""
    env_path = Path(__file__).resolve().parent / ".env"

    console.print(
        Panel.fit(
            "Little Lisa Setup Wizard\n\n"
            "This wizard will help you configure a model provider.\n"
            "Ollama is the easiest path no API key needed, fully local.",
            title="Setup",
            border_style="steel_blue3",
        )
    )

    console.print("\nChoose a provider:")
    for key, (_, label) in _SETUP_PROVIDERS.items():
        console.print(f"  [{key}] {label}")

    choice = console.input("\nEnter choice [1-5] (default: 1): ").strip() or "1"
    if choice not in _SETUP_PROVIDERS:
        console.print(f"[red]Invalid choice: {choice}[/red]")
        raise typer.Exit(code=1)

    provider, _ = _SETUP_PROVIDERS[choice]
    updates: dict[str, str] = {}

    if provider == "ollama":
        console.print("\n[bold]Ollama[/bold] no API key required.")
        console.print("Make sure Ollama is installed and running.")
        console.print("Install: https://ollama.com")
        console.print("\nPull the default models:")
        console.print("  ollama pull llama3.2:latest")
        console.print("  ollama pull qwen2.5:7b")
        console.print()
        ok, msg = asyncio.run(_setup_validate_ollama())
        if ok:
            console.print(f"[green]{msg}[/green]")
        else:
            console.print(f"[yellow]{msg}[/yellow]")
            console.print("You can still proceed just start Ollama before running Little Lisa.")

    elif provider == "openai":
        console.print("\n[bold]OpenAI[/bold]")
        api_key = console.input("Enter your OpenAI API key: ").strip()
        if not api_key:
            console.print("[red]API key is required.[/red]")
            raise typer.Exit(code=1)
        console.print("Validating...")
        ok, msg = asyncio.run(_setup_validate_openai(api_key))
        if ok:
            console.print(f"[green]{msg}[/green]")
        else:
            console.print(f"[red]{msg}[/red]")
            raise typer.Exit(code=1)
        updates["LITTLE_LISA_OPENAI_API_KEY"] = api_key

    elif provider == "azure_openai":
        console.print("\n[bold]Azure OpenAI[/bold]")
        endpoint = console.input("Enter your Azure OpenAI endpoint URL: ").strip()
        if not endpoint:
            console.print("[red]Endpoint URL is required.[/red]")
            raise typer.Exit(code=1)
        api_key = console.input("Enter your Azure OpenAI API key: ").strip()
        if not api_key:
            console.print("[red]API key is required.[/red]")
            raise typer.Exit(code=1)
        console.print("Validating...")
        ok, msg = asyncio.run(_setup_validate_azure(endpoint, api_key))
        if ok:
            console.print(f"[green]{msg}[/green]")
        else:
            console.print(f"[red]{msg}[/red]")
            raise typer.Exit(code=1)
        updates["LITTLE_LISA_AZURE_OPENAI_ENDPOINT"] = endpoint
        updates["LITTLE_LISA_AZURE_OPENAI_API_KEY"] = api_key

    elif provider == "anthropic":
        console.print("\n[bold]Anthropic[/bold]")
        api_key = console.input("Enter your Anthropic API key: ").strip()
        if not api_key:
            console.print("[red]API key is required.[/red]")
            raise typer.Exit(code=1)
        console.print("Validating...")
        ok, msg = asyncio.run(_setup_validate_anthropic(api_key))
        if ok:
            console.print(f"[green]{msg}[/green]")
        else:
            console.print(f"[red]{msg}[/red]")
            raise typer.Exit(code=1)
        updates["LITTLE_LISA_ANTHROPIC_API_KEY"] = api_key

    elif provider == "google":
        console.print("\n[bold]Google Gemini[/bold]")
        api_key = console.input("Enter your Google API key: ").strip()
        if not api_key:
            console.print("[red]API key is required.[/red]")
            raise typer.Exit(code=1)
        console.print("Validating...")
        ok, msg = asyncio.run(_setup_validate_google(api_key))
        if ok:
            console.print(f"[green]{msg}[/green]")
        else:
            console.print(f"[red]{msg}[/red]")
            raise typer.Exit(code=1)
        updates["LITTLE_LISA_GOOGLE_API_KEY"] = api_key

    if updates:
        _setup_write_env(env_path, updates)
        console.print(f"\n[green]Configuration written to {env_path}[/green]")

    console.print("\n[bold]Next steps:[/bold]")
    console.print("  1. Run [bold]little-lisa verify[/bold] to verify everything is working")
    console.print(
        '  2. Run [bold]little-lisa once "/markdown Hello, Lisa!"[/bold] for your first turn'
    )
    console.print("  3. Run [bold]little-lisa cli[/bold] for the interactive REPL")
    console.print("  4. Run [bold]little-lisa[/bold] for tray-first desktop launch")


@app.command("modes", help="List supported Little Lisa modes.")
def _modes_command(
    debug: DebugOption = False,
    log_level: LogLevelOption = "",
    log: LogOption = "",
    no_log: NoLogOption = False,
    data_dir: DataDirOption = "",
    memory_db: MemoryDbOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    timeout_profile: TimeoutProfileOption = "",
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "none",
) -> None:
    _initialize_runtime_or_exit(
        debug=debug,
        log_level=log_level,
        log_path=log,
        no_log=no_log,
        data_dir=data_dir,
        memory_db=memory_db,
        thread_dir=thread_dir,
        no_thread=no_thread,
        user_slug=session_context,
        timeout_profile=timeout_profile,
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
    )
    console.print(format_modes())


@app.command("models", help="List configured models with designations and aliases.")
def _models_command(
    debug: DebugOption = False,
    log_level: LogLevelOption = "",
    log: LogOption = "",
    no_log: NoLogOption = False,
    data_dir: DataDirOption = "",
    memory_db: MemoryDbOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    timeout_profile: TimeoutProfileOption = "",
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "none",
) -> None:
    _initialize_runtime_or_exit(
        debug=debug,
        log_level=log_level,
        log_path=log,
        no_log=no_log,
        data_dir=data_dir,
        memory_db=memory_db,
        thread_dir=thread_dir,
        no_thread=no_thread,
        user_slug=session_context,
        timeout_profile=timeout_profile,
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
    )
    console.print(format_models())


@app.command("once", help="Run a single turn and print the response.")
def _once_command(
    message: str = typer.Argument(..., help="Message to route through the host."),
    debug: DebugOption = False,
    log_level: LogLevelOption = "",
    log: LogOption = "",
    no_log: NoLogOption = False,
    data_dir: DataDirOption = "",
    memory_db: MemoryDbOption = "",
    thread_dir: ThreadDirOption = "",
    no_thread: NoThreadOption = False,
    session_context: SessionContextOption = "",
    timeout_profile: TimeoutProfileOption = "",
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "none",
) -> None:
    _initialize_runtime_or_exit(
        debug=debug,
        log_level=log_level,
        log_path=log,
        no_log=no_log,
        data_dir=data_dir,
        memory_db=memory_db,
        thread_dir=thread_dir,
        no_thread=no_thread,
        user_slug=session_context,
        timeout_profile=timeout_profile,
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
    )
    asyncio.run(_run_once(message))


def main() -> None:
    """CLI entry point for Little Lisa."""

    app(args=_normalize_legacy_cli_args(sys.argv[1:]))


if __name__ == "__main__":
    main()
