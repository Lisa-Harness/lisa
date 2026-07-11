"""Little Lisa — a local terminal host for exercising Lisa framework packs.

Little Lisa wires public content and operation catalogs, uses local tool callbacks
and live model completion hooks, and stores generated Markdown artifacts on disk.
It is useful for pack authors who want to verify routing and operation shape
without Teams, Azure, or production Lisa services.

Module Organisation
-------------------
This file is the **orchestration shell**: it owns the CLI entry points, the REPL
loop, the front-door planner, provider routing (``_complete_live``), and the
``handle_message`` command dispatcher.

Pure, stateless logic has been extracted into focused submodules:

- ``config.py`` — timezone / timeout / log-level / streaming / slice resolution
  (all deterministic, no global state, no side effects).
- ``token_accounting.py`` — token ledger accumulation and session-summary
  Markdown generation (accepts the runtime state dict as a parameter).
- ``turn_logging.py`` — JSONL log-record assembly and turn-log payload
  construction (accepts a resolved context dict, pure and testable).
- ``model_runtime.py`` — model manifest loading and model resolution by
  designation / alias / concrete ID.
- ``model_manifest.py`` — manifest file discovery and loading from ``models.yaml``.
- ``desktop.py`` — localhost web-host callbacks for the desktop UI.
- ``tray.py`` — system-tray launcher managing the desktop host subprocess.

Test Compatibility
------------------
Tests use ``monkeypatch.setattr(little_lisa, "_resolve_model", fake)`` and
``monkeypatch.setitem(little_lisa._RUNTIME_STATE, key, value)`` to exercise
individual functions and state.  To preserve these seams:

- All test-visible function names remain on this module's surface (re-imported
  from submodules or defined here as thin wrappers).
- ``_RUNTIME_STATE`` stays a plain dict (not a typed dataclass) so tests can
  mutate it directly.
- Submodule imports use ``try: from .module import ... / except ImportError``
  so this file also works as a standalone script (``python main.py --list-modes``).

See ``docs/LITTLE_LISA_GUIDE.md`` → *Architecture* for the full module map and
data-flow diagram.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
import os
import platform
import re
import sys
import time
import tomllib
from collections.abc import Callable
from datetime import UTC, datetime
from functools import lru_cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version
from pathlib import Path
from typing import Annotated, Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
import typer
import yaml
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

logger = logging.getLogger(__name__)

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.history import FileHistory, History
except ImportError:  # pragma: no cover - optional dependency for richer REPL UX
    PromptSession = None
    WordCompleter = None
    FileHistory = None
    History = None


class _SurrogateSafeFileHistory(History):
    """FileHistory wrapper that sanitises unpaired surrogates before persisting.

    On Windows, the console can emit unpaired surrogate characters for emoji
    (e.g. ??).  Python's utf-8 codec rejects those, which crashes
    prompt_toolkit's FileHistory.store_string.  This wrapper strips them.
    """

    def __init__(self, filename: str) -> None:
        super().__init__()
        self._inner = FileHistory(filename)

    def load_history_strings(self) -> list[str]:
        return self._inner.load_history_strings()

    def store_string(self, string: str) -> None:
        # Replace unpaired surrogates with the Unicode replacement character
        # so the string is always valid UTF-8.
        safe = string.encode("utf-8", errors="surrogateescape").decode(
            "utf-8", errors="replace"
        )
        self._inner.store_string(safe)

    def append_string(self, string: str) -> None:
        safe = string.encode("utf-8", errors="surrogateescape").decode(
            "utf-8", errors="replace"
        )
        self._inner.append_string(safe)

FRAMEWORK_ROOT = Path(__file__).resolve().parents[2]
if str(FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_ROOT))

_LITTLE_LISA_ENV_FILE = Path(__file__).resolve().parent / ".env"


def _load_local_env_file() -> None:
    """Load the Little Lisa local .env without overriding explicit process env vars."""
    if _LITTLE_LISA_ENV_FILE.exists():
        logger.debug("loading .env from %s", _LITTLE_LISA_ENV_FILE)
    load_dotenv(dotenv_path=_LITTLE_LISA_ENV_FILE, override=False)


# Load environment variables before kernel imports so Settings() sees them.
_load_local_env_file()

from kernel import frontdoor as shared_frontdoor  # noqa: E402
from kernel.content_roots import (  # noqa: E402
    discover_configured_content_roots,
    discover_installed_content_roots,
    resolve_content_subdir_roots,
)
from kernel.context.context_budget import ContextBudgetAllocator, ContextLane  # noqa: E402
from kernel.ipsar import parse_route  # noqa: E402
from kernel.ipsar_fsm import (  # noqa: E402
    create_workflow_session,
    get_workflow_catalog,
    parse_front_matter,
    register_content_mcp,
    register_operation_callbacks,
)
from kernel.model_manifest import ModelManifest  # noqa: E402
from kernel.models import (  # noqa: E402
    AnomalyAlert,
    DomainKnowledgeArtifact,
    IntelCollection,
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
    register_operation_mcp,
)
from kernel.operations.structured_output import parse_structured_payload  # noqa: E402
from kernel.pack_activation import load_disabled_pack_names  # noqa: E402
from kernel.slice import SliceOutcome  # noqa: E402
from kernel.token_usage import (  # noqa: E402
    TokenUsageTelemetryHandler,
    build_token_usage_event,
)
from kernel.tools.mcp_client import (  # noqa: E402
    MCPClientConfig,
    MCPConnectionError,
    MCPToolPool,
)
from kernel.tools.registry import (  # noqa: E402
    get_tool_catalog,
    register_tool_mcp,
)
from kernel.turn_runner import TurnHooks, TurnResult, run_turn  # noqa: E402
from sdk.lisa_context.context_broker import (  # noqa: E402
    build_ambient_facts,
    render_context_packet_markdown,
    select_ranked_context,
)

try:
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

try:
    from . import frontdoor_routing
    from . import turn_hooks as turn_hooks_module
    from .desktop import (
        DEFAULT_DESKTOP_HOST,
        DEFAULT_DESKTOP_PORT,
        DesktopCallbacks,
        generate_desktop_token,
        run_desktop_server,
    )
    from .provider_completions import (
        _extract_anthropic_error_detail,
        _extract_anthropic_text,
        _extract_anthropic_tool_calls,
        _extract_completion_tool_calls,
        _extract_json_object,
        _first_message_content,
        _normalize_ollama_host_to_base_url,
        _normalize_response_format,
        _normalize_structured_content,
        _to_anthropic_payload_messages,
        _to_anthropic_tools,
    )
    from .provider_completions import (
        _extract_usage_counts as _extract_usage_counts_helper,
    )
    from .provider_completions import (
        _normalize_google_base_url as _normalize_google_base_url_helper,
    )
    from .tray import TrayConfig, run_tray
except ImportError:  # pragma: no cover - direct script execution fallback
    import frontdoor_routing  # type: ignore[no-redef]
    import turn_hooks as turn_hooks_module  # type: ignore[no-redef]
    from desktop import (  # type: ignore[no-redef]
        DEFAULT_DESKTOP_HOST,
        DEFAULT_DESKTOP_PORT,
        DesktopCallbacks,
        generate_desktop_token,
        run_desktop_server,
    )
    from provider_completions import (  # type: ignore[no-redef]
        _extract_anthropic_error_detail,
        _extract_anthropic_text,
        _extract_anthropic_tool_calls,
        _extract_completion_tool_calls,
        _extract_json_object,
        _first_message_content,
        _normalize_ollama_host_to_base_url,
        _normalize_response_format,
        _normalize_structured_content,
        _to_anthropic_payload_messages,
        _to_anthropic_tools,
    )
    from provider_completions import (
        _extract_usage_counts as _extract_usage_counts_helper,
    )
    from provider_completions import (
        _normalize_google_base_url as _normalize_google_base_url_helper,
    )
    from tray import TrayConfig, run_tray  # type: ignore[no-redef]

_DEFAULT_DATA_DIR = FRAMEWORK_ROOT / "hosts" / "little_lisa" / "artifacts"
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
_LITTLE_LISA_TIMEZONE_ENV = "LITTLE_LISA_TIMEZONE"
_LITTLE_LISA_LOG_LEVEL_ENV = "LITTLE_LISA_LOG_LEVEL"
_LITTLE_LISA_SLICE_PRE_TURN_ENV = "LITTLE_LISA_SLICE_PRE_TURN"
_LITTLE_LISA_SLICE_POST_TURN_ENV = "LITTLE_LISA_SLICE_POST_TURN"
_LITTLE_LISA_SLICE_EXPLICIT_ENV = "LITTLE_LISA_SLICE_EXPLICIT"
_LITTLE_LISA_SLICE_BUDGET_MS_ENV = "LITTLE_LISA_SLICE_BUDGET_MS"
_LITTLE_LISA_THREAD_MAX_ENTRIES_ENV = "LITTLE_LISA_THREAD_MAX_ENTRIES"
_LITTLE_LISA_MEMORY_MAX_RECORDS_ENV = "LITTLE_LISA_MEMORY_MAX_RECORDS"
_LITTLE_LISA_PACK_ALLOWED_PREFIXES_ENV = "LITTLE_LISA_PACK_ALLOWED_PREFIXES"
_LITTLE_LISA_PACK_GIT_ALLOW_HOSTS_ENV = "LITTLE_LISA_PACK_GIT_ALLOW_HOSTS"
_LITTLE_LISA_PACK_REQUIRE_GIT_REF_ENV = "LITTLE_LISA_PACK_REQUIRE_GIT_REF"
_LITTLE_LISA_AUTO_RESTART_ENV = "LITTLE_LISA_AUTO_RESTART"
_LITTLE_LISA_RESTART_RESUME_MESSAGE_ENV = "LITTLE_LISA_RESTART_RESUME_MESSAGE"
_LISA_MODE_PROVIDER_PREFERENCES_ENV = "LISA_MODE_PROVIDER_PREFERENCES"
_LISA_CONTENT_PROVIDER_PREFERENCES_ENV = "LISA_CONTENT_PROVIDER_PREFERENCES"
_LISA_DISABLED_PACKS_ENV = "LISA_DISABLED_PACKS"
_PACK_LOCK_FILE_NAME = "installed-packs.json"
_DEFAULT_ALWAYS_ON_PACKAGES: tuple[str, ...] = (
    "lisa-pack-core",
    "lisa-pack-learning",
    "lisa-pack-search",
)
_RUNTIME_INSTALL_SOURCE_KINDS = {"package", "git", "local"}

# Config-resolution logic (timezone, timeout, log-level, streaming, slice)
# lives in the dedicated ``config`` module.  We import the names here so
# that ``monkeypatch.setattr(little_lisa, "_resolve_timezone", ...)`` and
# similar test seams continue to resolve on the ``main`` module surface.
try:
    from .config import (
        _DEFAULT_MEMORY_MAX_RECORDS,
        _DEFAULT_THREAD_MAX_ENTRIES,
        _LOG_LEVEL_ORDER,
        _THREAD_DIGEST_FILE_NAME,
        _normalize_log_level,
        _resolve_log_level,
        _resolve_slice_tuning,
        _resolve_timezone,
        context_slug_to_thread_name,
        parse_streaming_option,
        resolve_stream_enabled,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from config import (  # type: ignore[no-redef]
        _DEFAULT_MEMORY_MAX_RECORDS,
        _DEFAULT_THREAD_MAX_ENTRIES,
        _LOG_LEVEL_ORDER,
        _THREAD_DIGEST_FILE_NAME,
        _normalize_log_level,
        _resolve_log_level,
        _resolve_slice_tuning,
        _resolve_timezone,
        context_slug_to_thread_name,
        parse_streaming_option,
        resolve_stream_enabled,
    )


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
    "memory_config": None,
    "memory_config_pack_slug": None,
    "restart_required": False,
    "restart_reason": "",
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
HostLogOption = Annotated[
    str,
    typer.Option(
        "--host-log",
        help=(
            "Stdlib host log file (kernel/sdk startup and runtime logs). "
            "Default: <data-dir>/host.log"
        ),
    ),
]
DataDirOption = Annotated[
    str,
    typer.Option(
        "--data-dir",
        help=(
            "Data root for artifacts, memory, and logs. "
            "Overrides LITTLE_LISA_DATA_DIR. "
            "Default: hosts/little_lisa/artifacts/"
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
_STATUS_COMPACT_FULL = 0
_STATUS_COMPACT_ABBREVIATED = 1
_STATUS_COMPACT_TRIMMED = 2
_STATUS_COMPACT_MINIMAL = 3
_TIMEZONE_INITIALISM_MIN_LEN = 2
_STATUS_TIME_PADDING_WIDTH = 2
_REPL_HISTORY_FILE_NAME = "repl_history.txt"
_DEFAULT_DESKTOP_OPEN_ON_START = True
_DEFAULT_DESKTOP_NO_ARGS_MODE = "tray"
# Host-only REPL shortcuts that are not catalog modes. These are the only
# hardcoded command surface; everything else is derived from the mode catalog
# at help-render time. Keep this list minimal and host-specific.
_HOST_REPL_SHORTCUTS: tuple[str, ...] = (
    "/help",
    "/help capabilities",
    "/pack list",
    "/pack enable ",
    "/pack disable ",
    "/pack install ",
    "/pack uninstall ",
    "/settings",
    "/tokens",
    "/debug context",
    "/debug frontdoor",
    "/consolidate",
    "/deep ",
    "/quit",
)


def _repl_command_hints() -> tuple[str, ...]:
    """Build REPL completion hints from host shortcuts plus catalog mode commands."""
    catalog = get_workflow_catalog()
    mode_hints: list[str] = []
    for mode_id, mode in sorted(catalog.items()):
        if _mode_support_gap(mode) is not None:
            continue
        if mode.command_pattern and mode_id != "default":
            mode_hints.append(f"/{mode_id} ")
    return (*_HOST_REPL_SHORTCUTS, *mode_hints)


_MODE_MODEL_ALIAS_MAP = {
    "default": "default",
    "discuss": "default",
    "search": "default",
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


def _host_mcp_manifest_path() -> Path:
    return FRAMEWORK_ROOT / "hosts" / "little_lisa" / "mcp_servers.yaml"


def _normalize_pack_name(raw: str) -> str:
    value = raw.strip().lower().replace("_", "-")
    value = re.sub(r"[^a-z0-9-]+", "-", value).strip("-")
    return value


def _infer_pack_package_name(root: Path) -> str:
    """Infer the pack package name for one discovered content root."""
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        try:
            payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            payload = {}
        project = payload.get("project") if isinstance(payload, dict) else None
        if isinstance(project, dict):
            name = project.get("name")
            if isinstance(name, str) and name.strip():
                return _normalize_pack_name(name)

    root_name = _normalize_pack_name(root.name)
    if root.parent.name == "sdk_packs" and root_name:
        return f"lisa-pack-{root_name}"
    if root_name.startswith("lisa-pack-"):
        return root_name
    return root_name


def _pack_aliases(package_name: str, *, root: Path | None = None) -> set[str]:
    aliases: set[str] = set()
    normalized = _normalize_pack_name(package_name)
    if normalized:
        aliases.add(normalized)
        if normalized.startswith("lisa-pack-"):
            aliases.add(normalized[len("lisa-pack-"):])
    if root is not None:
        root_alias = _normalize_pack_name(root.name)
        if root_alias:
            aliases.add(root_alias)
    return aliases


def _always_on_pack_names(host_manifest_path: Path | None = None) -> set[str]:
    always_on = {_normalize_pack_name(name) for name in _DEFAULT_ALWAYS_ON_PACKAGES}
    manifest_path = host_manifest_path or _host_mcp_manifest_path()
    policy = _load_pack_policy_dict(manifest_path, warning_subject="pack activation policy")

    configured = policy.get("always_on_packs") or []
    if not isinstance(configured, list):
        return always_on

    for item in configured:
        if not isinstance(item, str):
            continue
        normalized = _normalize_pack_name(item)
        if normalized:
            always_on.add(normalized)
    return always_on


def _runtime_installed_pack_names(lock: dict[str, Any]) -> set[str]:
    entries = lock.get("packs") if isinstance(lock, dict) else None
    if not isinstance(entries, list):
        return set()

    runtime_installed: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        source_kind = str(entry.get("source_kind") or "").strip().lower()
        if source_kind not in _RUNTIME_INSTALL_SOURCE_KINDS:
            continue
        package_name = _normalize_pack_name(str(entry.get("package_name") or ""))
        if package_name:
            runtime_installed.add(package_name)
    return runtime_installed


def _runtime_enabled_pack_map(lock: dict[str, Any]) -> dict[str, bool]:
    entries = lock.get("packs") if isinstance(lock, dict) else None
    if not isinstance(entries, list):
        return {}

    enabled_map: dict[str, bool] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        package_name = _normalize_pack_name(str(entry.get("package_name") or ""))
        if not package_name:
            continue
        enabled_map[package_name] = bool(entry.get("enabled", True))
    return enabled_map


def _pack_is_active(
    package_name: str,
    *,
    always_on_packs: set[str],
    runtime_installed: set[str],
    enabled_map: dict[str, bool],
) -> bool:
    if package_name in always_on_packs:
        return True
    if package_name in runtime_installed:
        return enabled_map.get(package_name, True)
    return enabled_map.get(package_name, False)


def _active_installed_pack_roots() -> list[Path]:
    lock = _load_installed_packs_lock()
    always_on_packs = _always_on_pack_names()
    runtime_installed = _runtime_installed_pack_names(lock)
    enabled_map = _runtime_enabled_pack_map(lock)

    active: list[Path] = []
    for root in discover_installed_content_roots():
        package_name = _infer_pack_package_name(root)
        if not package_name:
            continue
        if _pack_is_active(
            package_name,
            always_on_packs=always_on_packs,
            runtime_installed=runtime_installed,
            enabled_map=enabled_map,
        ):
            active.append(root)
    return active


def _sdk_content_roots() -> list[Path]:
    roots = [*_active_installed_pack_roots(), *discover_configured_content_roots()]
    return _unique_paths([root for root in roots if root.is_dir()])


def _authored_content_roots() -> list[Path]:
    roots = _sdk_content_roots()
    return _unique_paths([root / "content" for root in roots if (root / "content").is_dir()])


def _sdk_mode_roots() -> list[Path]:
    return _unique_paths(resolve_content_subdir_roots(_sdk_content_roots(), "modes"))


def _sdk_operation_roots() -> list[Path]:
    return _unique_paths(resolve_content_subdir_roots(_sdk_content_roots(), "operations"))


FrontDoorRouteDecision = RouteDecision


def get_little_lisa_tool_catalog() -> dict[str, ToolDefinition]:
    """Return the Little Lisa tool catalog for diagnostics and harnesses.

    Tools are discovered from content packs via ``register_tool_mcp`` and
    exposed through the kernel tool registry. Little Lisa does not maintain
    a separate host-side override catalog, so this wrapper returns the live
    pack-contributed catalog (which is empty before ``setup_framework`` runs).
    """

    return get_tool_catalog()


def _build_content_mcp_config() -> MCPClientConfig:
    """Build an MCPClientConfig that spawns lisa-content for discovered pack roots.

    The kernel calls ``list_modes``, ``list_operations``, and
    ``read_authored_text`` on this server to discover and load all content
    packs. The server does all filesystem access; the kernel never touches
    the filesystem directly.

    ``SDK_CONTENT_ROOTS`` includes both the pack root (for modes/operations
    discovery) and the ``content/`` subdirectory (so ``read_authored_text``
    can resolve authored text assets like ``frontdoor/route.md`` without a
    ``content/`` prefix).
    """
    host_manifest_path = _host_mcp_manifest_path()
    content_provider_preferences = _load_content_provider_preferences(host_manifest_path)
    mode_preferences = content_provider_preferences.get("modes", {})
    roots_for_child = _sdk_content_roots()
    content_roots = [root / "content" for root in roots_for_child if (root / "content").is_dir()]
    configured_roots = _unique_paths([*roots_for_child, *content_roots])
    env = {"SDK_CONTENT_ROOTS": os.pathsep.join(str(root) for root in configured_roots)}

    # Pass disabled pack names to the lisa-content child process so that
    # entry-point discovery in the child also excludes disabled packs.
    disabled_packs = load_disabled_pack_names(_runtime_data_dir())
    if disabled_packs:
        env[_LISA_DISABLED_PACKS_ENV] = os.pathsep.join(sorted(disabled_packs))

    if any(content_provider_preferences.get(kind) for kind in ("modes", "operations", "tools")):
        env[_LISA_CONTENT_PROVIDER_PREFERENCES_ENV] = json.dumps(
            content_provider_preferences,
            ensure_ascii=True,
        )
    if mode_preferences:
        env[_LISA_MODE_PROVIDER_PREFERENCES_ENV] = json.dumps(mode_preferences, ensure_ascii=True)
    return MCPClientConfig(
        server_id="lisa-content",
        command=sys.executable,
        args=[
            "-m",
            "sdk.lisa_content.server",
            "--data-dir",
            str(_DEFAULT_DATA_DIR),
        ],
        env=env,
    )


def setup_framework() -> None:
    """Wire the lisa-content MCP server for content discovery and stub tools.

    All content (modes, operations, authored text) is loaded through the
    ``lisa-content`` MCP server. The kernel never touches the filesystem
    directly — it calls ``list_modes``, ``list_operations``, and
    ``read_authored_text`` on the server, which does all scanning and
    front-matter parsing.
    """
    content_config = _build_content_mcp_config()
    # Order matters: operation MCP first, then operation callbacks, then
    # content MCP. validate_workflow_catalog calls _validate_mode_operation_bindings
    # which needs the operation callbacks registered before it can validate.
    register_operation_mcp(content_config)
    register_tool_mcp(content_config)
    register_operation_callbacks(get_operation_catalog, execute_operation)
    register_content_mcp(content_config)
    # Wire the MCPContentRegistry as the content registry for context frames.
    from kernel.context.context_frames import set_content_registry  # noqa: PLC0415
    from kernel.context.mcp_content_registry import MCPContentRegistry  # noqa: PLC0415

    set_content_registry(MCPContentRegistry(content_config))

    # Wire the production extension-evidence builder so the context broker's
    # catalog builder picks up evidence from installed SDK extensions
    # (Cloudflare, GitHub, Stripe) via entry-point discovery.
    from sdk.lisa_context.extension_evidence_builder import (  # noqa: PLC0415
        set_production_extension_evidence_builder,
    )

    set_production_extension_evidence_builder(
        excluded_packs=load_disabled_pack_names(_runtime_data_dir()),
    )


def configure_runtime() -> None:
    """Configure runtime and reset provider clients."""

    _RUNTIME_STATE["azure_client"] = None
    _RUNTIME_STATE["openai_client"] = None
    _RUNTIME_STATE["anthropic_http_client"] = None
    _RUNTIME_STATE["ollama_client"] = None
    _RUNTIME_STATE["google_client"] = None


def _runtime_debug_enabled() -> bool:
    """Return whether --debug mode is active."""
    return bool(_RUNTIME_STATE.get("debug", False))


# _normalize_log_level is imported from .config (see import block above).
# _runtime_log_level and _runtime_should_log below are thin state-reading
# wrappers that use the imported normaliser.


def _runtime_log_level() -> str:
    """Return the effective log level, falling back to debug when --debug is on."""
    configured = _normalize_log_level(str(_RUNTIME_STATE.get("log_level") or ""))
    if configured is not None:
        return configured
    return "debug" if _runtime_debug_enabled() else "info"


def _runtime_should_log(level: str) -> bool:
    """Return True when *level* is at or above the runtime log-level threshold."""
    normalized = _normalize_log_level(level) or "info"
    runtime_level = _runtime_log_level()
    return _LOG_LEVEL_ORDER[normalized] >= _LOG_LEVEL_ORDER[runtime_level]


def _runtime_log_path() -> Path | None:
    configured = _RUNTIME_STATE.get("log_path")
    return configured if isinstance(configured, Path) else None


def _runtime_host_log_path() -> Path:
    """Return the durable stdlib host-log path under the data dir.

    Distinct from ``_runtime_log_path()`` (the JSONL turn log). The stdlib
    ``logging`` stream from ``kernel.*``, ``hosts.little_lisa.*``, and any
    third-party library lands here in plain-text form. Honors an explicit
    ``--host-log`` when set, otherwise defaults to ``<data_dir>/host.log``.
    """
    configured = _RUNTIME_STATE.get("host_log_path")
    if isinstance(configured, Path):
        return configured
    return _runtime_data_dir() / "host.log"


def _configure_root_logging() -> None:
    """Own the stdlib root logger so FastMCP's RichHandler stops spamming stderr.

    ``FastMCP.__init__`` (imported eagerly via ``sdk.lisa_content`` etc.)
    calls ``configure_logging``, which installs a ``RichHandler`` on the
    root logger at INFO with a stderr console. That makes every
    ``logger.info(...)`` in the host process render to stderr with Rich's
    ``[time] INFO  message  file:line`` format — including ``registry.py``
    and ``model_manifest.py`` startup messages.

    We strip every existing root handler (the RichHandler included) and
    install one ``FileHandler`` pointed at ``_runtime_host_log_path()``
    at the resolved runtime log level. Must run after ``_RUNTIME_STATE``
    has ``log_level`` and the data-dir / ``host_log_path`` resolved.
    """
    import logging as _logging  # noqa: PLC0415

    target = _runtime_host_log_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    root = _logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        with contextlib.suppress(Exception):
            handler.close()

    file_handler = _logging.FileHandler(target, encoding="utf-8")
    file_handler.setFormatter(
        _logging.Formatter(
            fmt="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(file_handler)
    root.setLevel(_runtime_log_level().upper())


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
    return str(_RUNTIME_STATE.get("user_slug") or "").strip()


def _runtime_session_context_ref() -> str:
    return str(_RUNTIME_STATE.get("session_context_ref") or "").strip()


def _runtime_next_model_provider() -> str:
    configured = str(_RUNTIME_STATE.get("next_model_provider") or "").strip()
    return configured or "-"


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


# resolve_stream_enabled and parse_streaming_option are imported from .config
# and re-exported here so tests using ``little_lisa.parse_streaming_option``
# and ``little_lisa.resolve_stream_enabled`` continue to resolve.


def _runtime_stream_enabled() -> bool:
    """Streaming is active when the resolved streaming mode is not 'none'."""
    return _runtime_streaming_mode() != "none"


def _runtime_stream_mode() -> str:
    """Return the configured stream mode, defaulting to 'hybrid' when unset."""
    configured = _runtime_streaming_mode()
    return configured if configured in {"stage", "text", "hybrid"} else "hybrid"


def _runtime_stream_char_delay_ms() -> int:
    """Return the configured per-character streaming delay in milliseconds."""
    return _runtime_streaming_char_delay_ms()


def _runtime_streaming_mode() -> str:
    """Return the raw streaming mode from state ('none' when unset or invalid)."""
    configured = str(_RUNTIME_STATE.get("streaming_mode") or "").strip().lower()
    return configured if configured in {"none", "stage", "text", "hybrid"} else "none"


def _runtime_streaming_char_delay_ms() -> int:
    """Return the raw streaming char delay from state (clamped to >= 0)."""
    return max(0, int(_RUNTIME_STATE.get("streaming_char_delay_ms", 0) or 0))


def _runtime_session_started_utc() -> datetime:
    configured = _RUNTIME_STATE.get("session_started_utc")
    return configured if isinstance(configured, datetime) else datetime.now(UTC)


# Token-accounting and session-summary logic lives in the dedicated
# ``token_accounting`` module.  We import the pure helpers and wrap them
# with thin state-reading adapters below so tests using
# ``little_lisa._apply_turn_usage(...)`` and
# ``little_lisa._format_session_exit_summary_lines()`` keep working.
try:
    from .token_accounting import (
        apply_turn_usage as _ta_apply_turn_usage,
    )
    from .token_accounting import (
        coerce_non_negative_int as _coerce_non_negative_int,
    )
    from .token_accounting import (
        format_session_exit_summary_lines as _ta_format_summary_lines,
    )
    from .token_accounting import (
        record_session_turn as _ta_record_session_turn,
    )
    from .token_accounting import (
        top_session_mode as _ta_top_session_mode,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from token_accounting import (  # type: ignore[no-redef]
        apply_turn_usage as _ta_apply_turn_usage,
    )
    from token_accounting import (
        coerce_non_negative_int as _coerce_non_negative_int,
    )
    from token_accounting import (
        format_session_exit_summary_lines as _ta_format_summary_lines,
    )
    from token_accounting import (
        record_session_turn as _ta_record_session_turn,
    )
    from token_accounting import (
        top_session_mode as _ta_top_session_mode,
    )

# Structured-logging record assembly lives in the dedicated ``turn_logging``
# module.  We import the assemblers here so the thin wrappers below
# (_append_log_event, _append_turn_log, etc.) can delegate to them.
try:
    from .turn_logging import (
        build_log_record as _log_build_log_record,
    )
    from .turn_logging import (
        build_runtime_log_metadata as _log_build_runtime_log_metadata,
    )
    from .turn_logging import (
        build_turn_log_debug_payload as _log_build_turn_debug_payload,
    )
    from .turn_logging import (
        build_turn_log_record as _log_build_turn_log_record,
    )
    from .turn_logging import (
        write_log_record as _log_write_log_record,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from turn_logging import (  # type: ignore[no-redef]
        build_log_record as _log_build_log_record,
    )
    from turn_logging import (
        build_runtime_log_metadata as _log_build_runtime_log_metadata,
    )
    from turn_logging import (
        build_turn_log_debug_payload as _log_build_turn_debug_payload,
    )
    from turn_logging import (
        build_turn_log_record as _log_build_turn_log_record,
    )
    from turn_logging import (
        write_log_record as _log_write_log_record,
    )


def _record_session_turn(
    *,
    succeeded: bool,
    latency_ms: float,
    mode_id: str | None = None,
) -> None:
    """Increment session turn counters (thin wrapper over token_accounting)."""
    _ta_record_session_turn(
        _RUNTIME_STATE,
        succeeded=succeeded,
        latency_ms=latency_ms,
        mode_id=mode_id,
    )


def _apply_turn_usage(
    result: TurnResult,
    *,
    model_id: str = "",
    mode_id: str = "",
    stage_id: str = "",
    frontdoor_routing: dict[str, Any] | None = None,
) -> None:
    """Persist turn and session token counters (thin wrapper over token_accounting).

    See :func:`token_accounting.apply_turn_usage` for the full semantics.
    """
    _ta_apply_turn_usage(
        _RUNTIME_STATE,
        result,
        model_id=model_id,
        mode_id=mode_id,
        stage_id=stage_id,
        frontdoor_routing=frontdoor_routing,
    )


def _top_session_mode() -> tuple[str, int] | None:
    """Return ``(mode_id, count)`` for the most-used mode (thin wrapper)."""
    return _ta_top_session_mode(_RUNTIME_STATE)


def _format_session_exit_summary_lines() -> list[str]:
    """Build the Markdown session-summary lines (thin wrapper)."""
    return _ta_format_summary_lines(
        _RUNTIME_STATE,
        session_started_utc=_runtime_session_started_utc(),
    )


# _apply_turn_usage is defined above as a thin wrapper over token_accounting.


def _top_session_mode() -> tuple[str, int] | None:
    """Return ``(mode_id, count)`` for the most-used mode (thin wrapper)."""
    return _ta_top_session_mode(_RUNTIME_STATE)


# _format_session_exit_summary_lines is defined above as a thin wrapper
# over token_accounting.format_session_exit_summary_lines.


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
    logger.debug(
        "route preview set: mode=%s stage=%s model=%s provider=%s",
        mode_id,
        stage_id or "-",
        model_id,
        provider,
    )


def _resolve_session_context(session_context: str) -> tuple[str, str, bool]:
    """Resolve the session context source and return (thread_name, source_path, is_active).

    Resolution order:
    1. If empty: check for 'session_context.md' or 'session_context/' in current dir.
    2. Literal path: if it's a file or directory.
    3. artifacts/ suffix: if 'artifacts/{session_context}.md'
       or 'artifacts/{session_context}/' exists.
    4. user_contexts/ slug: if 'user_contexts/{session_context}.md' exists.
    """
    s = session_context.strip()

    if not s:
        return _resolve_session_context_auto()
    if (result := _resolve_session_context_literal_path(s)) is not None:
        return result
    if (result := _resolve_session_context_artifacts(s)) is not None:
        return result
    if (result := _resolve_session_context_user_contexts(s)) is not None:
        return result
    raise ValueError(f"Session context '{s}' was not found.")


def _resolve_session_context_auto() -> tuple[str, str, bool]:
    """Auto-discover session_context.md or session_context/ in cwd and data dir."""
    for base in (Path("."), _runtime_data_dir()):
        md_path = base / "session_context.md"
        if md_path.is_file():
            tn = _extract_front_matter_thread_name(md_path) or "session_context"
            return tn, str(md_path.absolute()), True
        dir_path = base / "session_context"
        if dir_path.is_dir():
            return "session_context", str(dir_path.absolute()), True
    return "", "", False


def _resolve_session_context_literal_path(s: str) -> tuple[str, str, bool] | None:
    """Resolve a literal file or directory path."""
    p = Path(s).expanduser()
    if not p.exists():
        return None
    if p.is_dir():
        return "session_context", str(p.absolute()), True
    if p.suffix.lower() != ".md":
        raise ValueError("Session context file path must end with '.md'.")
    tn = _extract_front_matter_thread_name(p) or context_slug_to_thread_name(s)
    return tn, str(p.absolute()), True


def _resolve_session_context_artifacts(s: str) -> tuple[str, str, bool] | None:
    """Resolve a session context relative to the data directory (artifacts/)."""
    dd = _runtime_data_dir()
    md_path = dd / f"{s}.md"
    if md_path.is_file():
        tn = _extract_front_matter_thread_name(md_path) or context_slug_to_thread_name(s)
        return tn, str(md_path.absolute()), True
    dir_path = dd / s
    if dir_path.is_dir():
        return "session_context", str(dir_path.absolute()), True
    if s == "session_context":
        return _resolve_session_context_artifacts_session_context(dd)
    return None


def _resolve_session_context_artifacts_session_context(
    dd: Path,
) -> tuple[str, str, bool] | None:
    """Special-case resolution for 'session_context' slug in the data dir."""
    md_path = dd / "session_context.md"
    if md_path.is_file():
        tn = _extract_front_matter_thread_name(md_path) or "session_context"
        return tn, str(md_path.absolute()), True
    dir_path = dd / "session_context"
    if dir_path.is_dir():
        return "session_context", str(dir_path.absolute()), True
    return None


def _resolve_session_context_user_contexts(s: str) -> tuple[str, str, bool] | None:
    """Resolve a user_contexts/ slug (legacy path)."""
    context_name = s[:-3] if s.lower().endswith(".md") else s
    relative_path = f"user_contexts/{context_name}.md"
    try:
        content = _read_authored_text(relative_path)
    except FileNotFoundError:
        return None
    if not str(content).strip():
        return None
    return context_name, relative_path, False


# context_slug_to_thread_name, _resolve_log_level, _normalize_slice_flag,
# and _resolve_slice_tuning are
# imported from .config and re-exported here for test compatibility.


# _normalize_timezone_name and _resolve_timezone are imported from .config
# and re-exported here for test compatibility.


def _load_selected_session_context_text() -> str:
    """Load the active session context text, filtering by applies_to front matter."""
    context_ref = _runtime_session_context_ref()
    if not context_ref:
        return ""
    if bool(_RUNTIME_STATE.get("session_context_active", False)):
        return _load_active_session_context_text(context_ref)
    return _read_authored_text(context_ref)


def _load_active_session_context_text(context_ref: str) -> str:
    """Load session context from a file path or directory, filtering by applies_to."""
    path = Path(context_ref)
    current_mode = _runtime_current_mode_id()
    if path.is_dir():
        return _load_session_context_directory(path, current_mode)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    front_matter, body = parse_front_matter(text)
    if not _applies_to_mode(front_matter, current_mode):
        return ""
    return body


def _load_session_context_directory(path: Path, current_mode: str) -> str:
    """Load and concatenate session context files from a directory."""
    parts: list[str] = []
    for child in sorted(path.iterdir()):
        if not child.is_file():
            continue
        try:
            text = child.read_text(encoding="utf-8")
        except OSError:
            continue
        front_matter, body = parse_front_matter(text)
        if not _applies_to_mode(front_matter, current_mode):
            continue
        parts.append(body)
    return "\n\n".join(parts)


def _applies_to_mode(front_matter: dict[str, Any], current_mode: str) -> bool:
    """Return True if the front matter applies_to includes the current mode."""
    applies_to = front_matter.get("applies_to")
    if applies_to is None:
        return True
    if not isinstance(applies_to, list):
        applies_to = [applies_to]
    return current_mode in applies_to


def _runtime_data_dir() -> Path:
    configured = _RUNTIME_STATE.get("data_dir")
    return configured if isinstance(configured, Path) else _DEFAULT_DATA_DIR


def _runtime_pack_lock_path() -> Path:
    """Return the path to the installed pack provenance lock file."""
    return _runtime_data_dir() / _PACK_LOCK_FILE_NAME


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
        retry_decision = parse_structured_payload(FrontDoorRouteDecision, retry_text)
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


def _record_frontdoor_completion(
    completion_result: StubCompletion,
    *,
    attempt: str,
    frontdoor_debug: dict[str, Any],
    coerce_int: Any,
) -> None:
    """Record one front-door planner completion attempt into the debug dict.

    Accumulates token counts across attempts and appends a summary entry to
    ``planner_completion_attempts``.

    Parameters
    ----------
    completion_result
        The completion returned by the provider call.
    attempt
        A label identifying which pass produced this completion
        (e.g. ``"primary_structured"``, ``"retry_after_validation_error"``).
    frontdoor_debug
        The mutable debug-dict that accumulates planner diagnostics.
    coerce_int
        A callable that coerces values to non-negative ints (typically
        ``token_accounting.coerce_non_negative_int``).
    """
    prompt_tokens = coerce_int(completion_result.prompt_tokens)
    completion_tokens = coerce_int(completion_result.completion_tokens)
    total_tokens = coerce_int(completion_result.total_tokens)
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens

    frontdoor_debug["planner_prompt_tokens"] = (
        coerce_int(frontdoor_debug.get("planner_prompt_tokens")) + prompt_tokens
    )
    frontdoor_debug["planner_completion_tokens"] = (
        coerce_int(frontdoor_debug.get("planner_completion_tokens")) + completion_tokens
    )
    frontdoor_debug["planner_total_tokens"] = (
        coerce_int(frontdoor_debug.get("planner_total_tokens")) + total_tokens
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


def _capture_prompt_shape(
    messages: list[dict[str, Any]],
    *,
    frontdoor_debug: dict[str, Any],
    preview_fn: Any,
) -> None:
    """Capture system/user prompt shape into the front-door debug dict."""
    if not messages:
        return
    system_text = str(messages[0].get("content") or "")
    user_text = str(messages[1].get("content") or "") if len(messages) > 1 else ""
    frontdoor_debug["system_prompt_chars"] = len(system_text)
    frontdoor_debug["user_prompt_chars"] = len(user_text)
    frontdoor_debug["user_prompt_preview"] = preview_fn(user_text, limit=240)


async def _plan_frontdoor_route(
    *,
    raw_message: str,
    conversation_excerpt_markdown: str,
) -> FrontDoorRouteDecision:
    """Thin wrapper — delegates to frontdoor_routing.plan_frontdoor_route."""
    return await frontdoor_routing.plan_frontdoor_route(
        raw_message=raw_message,
        conversation_excerpt_markdown=conversation_excerpt_markdown,
        frontdoor_alias=_frontdoor_model_alias(),
        resolve_model=_resolve_model,
        complete_live=_complete_live,
        model_alias_for_mode=_model_alias_for_mode,
        coerce_int=_coerce_non_negative_int,
        preview_fn=_preview_text,
        load_route_prompt=lambda: "\n\n".join(
            part
            for part in (
                _load_frontdoor_route_prompt(),
                _load_turn_input_contract(),
            )
            if part.strip()
        ),
        load_fast_path_verifier_prompt=_load_frontdoor_fast_path_verifier_prompt,
        mode_supported=_frontdoor_mode_supported,
        build_user_prompt=_build_frontdoor_user_prompt,
        get_workflow_catalog=get_workflow_catalog,
        set_last_frontdoor_debug=_set_last_frontdoor_debug,
    )


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
        "- host: little_lisa",
    ]
    user_slug = _runtime_user_slug()
    lines.insert(2, f"- user_slug: {user_slug or 'none'}")
    if mode_description:
        lines.append(f"- mode_description: {mode_description}")
    if command_pattern:
        lines.append(f"- command_pattern: {command_pattern}")
    ambient_facts = build_ambient_facts(
        timezone=_runtime_timezone(),
        host_os=platform.system(),
        host_name="little_lisa",
    )

    # Build the raw sections dict, tagged by budget lane.
    raw_sections: list[tuple[str, ContextLane, str]] = [
        ("ambient_facts", "core", "## Ambient Facts\n\n" + ambient_facts),
        ("dynamic_context", "core", "## Dynamic User Context\n\n" + "\n".join(lines)),
    ]

    user_context = ""
    if user_slug:
        user_context = _load_selected_session_context_text().strip()
    if user_context:
        raw_sections.append(("selected_user_context", "memory", user_context))

    thread_digest = _load_thread_digest()
    if thread_digest:
        raw_sections.append(("prior_conversation_digest", "history", thread_digest))

    conversation_excerpt = _build_recent_thread_excerpt(limit=6)
    if conversation_excerpt:
        raw_sections.append(("recent_conversation", "history", conversation_excerpt))

    # Ranked context evidence from the broker (no MCP round-trip — the
    # broker is a pure ranking function over the in-process UnifiedState).
    packet_markdown = _build_brokered_context_markdown(mode_id=mode_id, query=query)
    if packet_markdown:
        raw_sections.append(("ranked_context", "retrieval", packet_markdown))

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
        raw_sections.append(
            (
                "capability_context",
                "operational",
                "## Capability Context\n\n"
                f"- supported_modes: {len(capability['supported_modes'])}\n"
                f"- operations: {len(capability['operations'])}\n"
                f"- tools: {len(capability['tools'])}\n"
                f"- models: {len(capability['models'])}",
            )
        )

    # Route every section through the budget allocator. The allocator enforces
    # per-lane token limits and a total ceiling, dropping or truncating as
    # needed. Surviving sections are joined into the final user context block.
    allocator = ContextBudgetAllocator.for_turn(reasoning_model=False)
    rendered_sections: list[str] = []
    for key, lane, text in raw_sections:
        elected = allocator.allocate(
            key=key,
            lane=lane,
            text=text,
            required=(key == "dynamic_context"),
            allow_truncate=True,
        )
        if elected:
            rendered_sections.append(elected)

    _publish_context_usage_event(allocator=allocator, mode_id=mode_id, query=query)
    return "\n\n".join(rendered_sections)


def _build_brokered_context_markdown(*, mode_id: str, query: str) -> str:
    """Select ranked context evidence from the in-process state and render it.

    Calls ``select_ranked_context`` directly (pure function, no MCP round-trip)
    using the compiled ``UnifiedState`` from ``_state_provider()``. Returns an
    empty string when no evidence is selected.
    """
    state = _state_provider()
    if not state.compiled_at:
        return ""
    packet = select_ranked_context(
        state=state,
        query=query,
        mode_id=mode_id,
    )
    if not packet.selected_evidence:
        return ""
    return render_context_packet_markdown(packet)


def _publish_context_usage_event(
    *,
    allocator: ContextBudgetAllocator,
    mode_id: str,
    query: str,
) -> None:
    """Publish a ContextUsageEvent with the allocator summary to the event bus.

    The event bus may be absent in test contexts; when it is, this is a no-op.
    """
    pool = _runtime_warm_pool()
    event_bus = getattr(pool, "event_bus", None) if pool is not None else None
    if event_bus is None:
        return
    summary = allocator.summary()
    import asyncio  # noqa: PLC0415

    from kernel.models import ContextUsageEvent  # noqa: PLC0415

    event = ContextUsageEvent(
        mode_id=mode_id,
        query_preview=query[:200],
        prompt_strategy="broker_packet",
        context_char_count=int(summary.get("used_tokens", 0)) * 4,
        ranking_metadata={
            "budget_summary": summary,
        },
    )
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(event_bus.publish(event.model_dump()))
        else:
            loop.run_until_complete(event_bus.publish(event.model_dump()))
    except RuntimeError:
        pass


def _runtime_log_metadata() -> dict[str, Any]:
    """Build the nested runtime-metadata block for log records.

    Thin wrapper over :func:`logging.build_runtime_log_metadata` that
    assembles the context dict from runtime-state accessors.
    """
    return _log_build_runtime_log_metadata(
        {
            "log_level": _runtime_log_level(),
            "debug": _runtime_debug_enabled(),
            "timezone": _runtime_timezone(),
            "timezone_source": _runtime_timezone_source(),
            "session_context": _runtime_user_slug() or "none",
            "mode_id": _runtime_current_mode_id(),
            "stage_id": _runtime_current_stage_id(),
            "next_model_provider": _runtime_next_model_provider(),
            "next_model_id": _runtime_next_model_id(),
            "stream_enabled": _runtime_stream_enabled(),
            "streaming_mode": _runtime_streaming_mode(),
            "streaming_char_delay_ms": _runtime_streaming_char_delay_ms(),
        }
    )


def _write_log_record(record: dict[str, Any]) -> None:
    """Write one JSONL record to the configured log path."""
    _log_write_log_record(record, _runtime_log_path())


def _append_log_event(
    *,
    level: str,
    event: str,
    payload: dict[str, Any],
    debug_payload: dict[str, Any] | None = None,
) -> None:
    """Append one structured JSONL log record if the level passes the threshold."""
    normalized_level = _normalize_log_level(level) or "info"
    if not _runtime_should_log(normalized_level):
        return

    ctx = {
        "log_level": _runtime_log_level(),
        "debug": _runtime_debug_enabled(),
        "timezone": _runtime_timezone(),
        "timezone_source": _runtime_timezone_source(),
        "session_context": _runtime_user_slug() or "none",
        "mode_id": _runtime_current_mode_id(),
        "stage_id": _runtime_current_stage_id(),
        "next_model_provider": _runtime_next_model_provider(),
        "next_model_id": _runtime_next_model_id(),
        "stream_enabled": _runtime_stream_enabled(),
        "streaming_mode": _runtime_streaming_mode(),
        "streaming_char_delay_ms": _runtime_streaming_char_delay_ms(),
    }
    record = _log_build_log_record(
        level=normalized_level,
        event=event,
        payload=payload,
        ctx=ctx,
        debug_payload=debug_payload,
        last_frontdoor_debug=_last_frontdoor_debug(),
    )
    if record is not None:
        _write_log_record(record)


def _append_turn_log(*, context: dict[str, Any], response_text: str) -> None:
    """Append a ``turn.completed`` JSONL record with flattened model/token fields."""
    record = _log_build_turn_log_record(
        context=context,
        response_text=response_text,
        coerce_int=_coerce_non_negative_int,
        preview_fn=_preview_text,
    )
    debug_payload = _log_build_turn_debug_payload(
        context=context,
        response_text=response_text,
        state=_RUNTIME_STATE,
        coerce_int=_coerce_non_negative_int,
        thread_dir=str(_runtime_thread_dir() or ""),
        state_snapshot_loaded=isinstance(_runtime_state_snapshot(), dict),
    )
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
        body = response or query
        lines.append(f"- {query or 'unknown'}: {body}")
    return "\n".join(lines)


def _build_capability_snapshot() -> dict[str, Any]:
    catalog = get_workflow_catalog()
    manifest = _load_manifest()
    operations = get_operation_catalog()
    tools = get_little_lisa_tool_catalog()
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
        lines.append("\n### system_prompt")
        lines.append(f"- chars: {system_prompt_info.get('chars', 0)}")
        lines.append(f"- preview: {system_prompt_info.get('preview', '(empty)')}")
        if system_prompt_info.get("omitted_for_brevity"):
            lines.append("- (truncated for brevity)")
    else:
        lines.append("\n### system_prompt\n(not available)")

    # Session context
    session_ref = _runtime_session_context_ref()
    session_active = bool(_RUNTIME_STATE.get("session_context_active", False))
    lines.append("\n### session_context")
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
        logger.error("state snapshot file not found: %s", file_path)
        raise ValueError(f"State snapshot file not found: {file_path}")
    decoded = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        logger.error("state snapshot file must contain a JSON object: %s", file_path)
        raise ValueError("State snapshot file must contain a JSON object.")
    logger.info("loaded state snapshot from %s (%d keys)", file_path, len(decoded))
    return decoded


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
    return _normalize_google_base_url_helper(base_url, _DEFAULT_GOOGLE_OPENAI_BASE_URL)


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


# _normalize_response_format, _extract_completion_tool_calls, _extract_usage_counts,
# _parse_tool_arguments, _to_anthropic_tools, _to_anthropic_payload_messages,
# _extract_anthropic_text, _extract_anthropic_tool_calls, _first_message_content,
# _extract_json_object, _normalize_structured_content, _extract_anthropic_error_detail,
# _normalize_ollama_host_to_base_url are imported from .provider_completions
# and re-exported here for test compatibility.


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


def _extract_usage_counts(
    usage_payload: Any,
) -> tuple[int, int, int, int, int]:
    """Extract token usage — delegates to provider_completions with the shared coercer."""
    return _extract_usage_counts_helper(usage_payload, _coerce_non_negative_int)


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
    response_format_name = _completion_response_format_name(response_format)
    kwargs = _build_completion_kwargs(
        model_def=model_def,
        max_tokens=max_tokens,
        temperature=temperature,
        tools=tools,
        response_format=response_format,
    )

    _log_completion_request(
        messages=messages,
        model_def=model_def,
        max_tokens=max_tokens,
        temperature=temperature,
        response_format_name=response_format_name,
        tools=tools,
    )

    if model_def.provider == "anthropic":
        return await _complete_live_anthropic(
            messages=messages,
            model_def=model_def,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            response_format=response_format,
            stream_stage=stream_stage,
            stage_label=stage_label,
            started_at=started_at,
            response_format_name=response_format_name,
        )

    return await _complete_live_openai_compatible(
        messages=messages,
        model_def=model_def,
        response_format=response_format,
        stream_stage=stream_stage,
        stage_label=stage_label,
        started_at=started_at,
        response_format_name=response_format_name,
        kwargs=kwargs,
        tools=tools,
    )


def _completion_response_format_name(response_format: type[Any] | None) -> str:
    """Return the __name__ of the response_format type, or empty string."""
    if response_format is None:
        return ""
    return str(getattr(response_format, "__name__", "") or "")


def _build_completion_kwargs(
    *,
    model_def: StubModelDefinition,
    max_tokens: int,
    temperature: float,
    tools: list[dict[str, Any]] | None,
    response_format: type[Any] | None,
) -> dict[str, Any]:
    """Build the kwargs dict for an OpenAI-compatible chat completion call."""
    kwargs: dict[str, Any] = {
        "model": model_def.model_id,
        "messages": [],  # placeholder — caller sets messages via kwargs.update
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
    return kwargs


def _log_completion_request(
    *,
    messages: list[dict[str, Any]],
    model_def: StubModelDefinition,
    max_tokens: int,
    temperature: float,
    response_format_name: str,
    tools: list[dict[str, Any]] | None,
) -> None:
    """Log the completion.request debug event."""
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


def _log_completion_error(
    *,
    exc: Exception,
    model_def: StubModelDefinition,
    response_format_name: str,
    elapsed_ms: float,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> None:
    """Log the completion.error event for a failed provider call."""
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


def _log_completion_response(
    *,
    model_def: StubModelDefinition,
    response_format_name: str,
    elapsed_ms: float,
    result: StubCompletion,
) -> None:
    """Log the completion.response debug event for a successful provider call."""
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


async def _publish_completion_usage(
    *,
    model_def: StubModelDefinition,
    result: StubCompletion,
    elapsed_ms: float,
    call_site: str,
) -> None:
    """Publish token usage for a completed provider call."""
    await _publish_token_usage(
        provider=model_def.provider,
        model_id=model_def.model_id,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=result.total_tokens,
        cached_tokens=result.cached_tokens,
        reasoning_tokens=result.reasoning_tokens,
        duration_ms=result.duration_ms,
        latency_ms=int(elapsed_ms),
        call_site=call_site,
    )


def _with_duration(result: StubCompletion, elapsed_ms: float) -> StubCompletion:
    """Rebuild a frozen StubCompletion with the host-measured duration."""
    return StubCompletion(
        content=result.content,
        tool_calls=result.tool_calls,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=result.total_tokens,
        cached_tokens=result.cached_tokens,
        reasoning_tokens=result.reasoning_tokens,
        duration_ms=elapsed_ms,
    )


def _stage_status_context(stream_stage: bool, stage_label: str) -> Any:
    """Return a context manager for the provider call spinner."""
    if stream_stage:
        console.print(f"[dim][stage] contacting {stage_label}...[/]")
        return contextlib.nullcontext()
    return console.status("[bold yellow]Thinking...", spinner="dots")


async def _complete_live_anthropic(
    *,
    messages: list[dict[str, Any]],
    model_def: StubModelDefinition,
    max_tokens: int,
    temperature: float,
    tools: list[dict[str, Any]] | None,
    response_format: type[Any] | None,
    stream_stage: bool,
    stage_label: str,
    started_at: float,
    response_format_name: str,
) -> StubCompletion:
    """Execute an Anthropic provider completion with logging and token publishing."""
    try:
        with _stage_status_context(stream_stage, stage_label):
            result = await _complete_anthropic(
                messages=messages,
                model_def=model_def,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools,
                response_format=response_format,
            )
        if stream_stage:
            console.print(f"[dim][stage] {stage_label} completed.[/]")
    except Exception as exc:  # pylint: disable=broad-except
        if stream_stage:
            console.print(f"[dim][stage] {stage_label} failed.[/]")
        elapsed_ms = max(0.0, (time.perf_counter() - started_at) * 1000.0)
        _log_completion_error(
            exc=exc,
            model_def=model_def,
            response_format_name=response_format_name,
            elapsed_ms=elapsed_ms,
            messages=messages,
            tools=tools,
        )
        raise

    elapsed_ms = max(0.0, (time.perf_counter() - started_at) * 1000.0)
    _log_completion_response(
        model_def=model_def,
        response_format_name=response_format_name,
        elapsed_ms=elapsed_ms,
        result=result,
    )
    final_result = _with_duration(result, elapsed_ms)
    await _publish_completion_usage(
        model_def=model_def,
        result=final_result,
        elapsed_ms=elapsed_ms,
        call_site="complete_live.anthropic",
    )
    return final_result


async def _dispatch_openai_compatible_provider(
    *,
    model_def: StubModelDefinition,
    kwargs: dict[str, Any],
) -> Any:
    """Dispatch to the correct OpenAI-compatible client based on provider name."""
    provider = model_def.provider
    if provider == "azure_openai":
        return await _azure_client_instance().chat.completions.create(**kwargs)
    if provider == "openai":
        return await _openai_client_instance().chat.completions.create(**kwargs)
    if provider == "ollama":
        return await _ollama_client_instance().chat.completions.create(**kwargs)
    if provider == "google":
        return await _google_client_instance().chat.completions.create(**kwargs)
    raise ValueError(f"Unsupported provider '{provider}' in Little Lisa manifest.")


async def _complete_live_openai_compatible(
    *,
    messages: list[dict[str, Any]],
    model_def: StubModelDefinition,
    response_format: type[Any] | None,
    stream_stage: bool,
    stage_label: str,
    started_at: float,
    response_format_name: str,
    kwargs: dict[str, Any],
    tools: list[dict[str, Any]] | None,
) -> StubCompletion:
    """Execute an OpenAI-compatible provider completion with logging and token publishing."""
    kwargs["messages"] = messages
    if stream_stage:
        console.print(f"[dim][stage] contacting {stage_label}...[/]")
    try:
        with _stage_status_context(stream_stage, stage_label):
            response = await _dispatch_openai_compatible_provider(
                model_def=model_def,
                kwargs=kwargs,
            )
        if stream_stage:
            console.print(f"[dim][stage] {stage_label} completed.[/]")
    except Exception as exc:  # pylint: disable=broad-except
        if stream_stage:
            console.print(f"[dim][stage] {stage_label} failed.[/]")
        elapsed_ms = max(0.0, (time.perf_counter() - started_at) * 1000.0)
        _log_completion_error(
            exc=exc,
            model_def=model_def,
            response_format_name=response_format_name,
            elapsed_ms=elapsed_ms,
            messages=messages,
            tools=tools,
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
    result = StubCompletion(
        content=content,
        tool_calls=tool_calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
        duration_ms=elapsed_ms,
    )
    _log_completion_response(
        model_def=model_def,
        response_format_name=response_format_name,
        elapsed_ms=elapsed_ms,
        result=result,
    )
    await _publish_completion_usage(
        model_def=model_def,
        result=result,
        elapsed_ms=elapsed_ms,
        call_site="complete_live.openai_compatible",
    )
    return result


async def _execute_tool(tool_id: str, ctx: Any) -> ToolResult:
    """Fallback tool executor for host-owned tools.

    Pack-contributed tools are dispatched in-process via the tool registry
    (``kernel.tools.registry.execute_tool``) — the turn loop and operation
    ``execute_tool_fn`` both go catalog-first. This stub is only reached
    for host-owned tools that are not in any pack catalog.
    """

    logger.debug("tool %s is stubbed in Little Lisa", tool_id)
    return ToolResult(tool_id=tool_id, output_text=f"{tool_id} is stubbed in Little Lisa.")


async def execute_little_lisa_tool(tool_id: str, ctx: Any) -> ToolResult:
    """Execute one Little Lisa tool with the same behavior as the turn loop."""

    return await _execute_tool(tool_id, ctx)


def _read_authored_text(relative_path: str) -> str:
    """Load authored pack content by relative path via the lisa-content MCP server.

    All filesystem access is mediated through the ``lisa-content`` MCP server;
    the host never touches the filesystem directly for content loading.
    """
    from kernel.context.context_frames import _content_registry  # noqa: PLC0415

    if _content_registry is None:
        raise RuntimeError(
            "Content registry not set. Call setup_framework() before reading authored text."
        )
    return _content_registry.read_authored_text(relative_path)


def _mode_support_gap(_mode: WorkflowModeConfig) -> str | None:
    """Return the missing host capability for modes Little Lisa cannot run.

    Little Lisa only loads the core pack. Modes that require
    app-level wiring (workshop, vibecheck) live in the cloudservers pack
    and are not visible here.
    """

    return None


def _state_provider() -> UnifiedState:
    """Return a compiled UnifiedState for operations and the context broker.

    Builds the state from the dict-based ``_runtime_state_snapshot()`` (loaded
    via ``--state-snapshot``), populating ``signal_snapshots``,
    ``intel_collections``, ``active_alerts``, and ``domain_knowledge`` from
    the snapshot's keys. The broker's evidence-catalog builder then walks
    these fields plus extension-contributed evidence to produce a ranked
    ContextPacket without a live MCP round-trip.
    """
    compiled_at = datetime.now(UTC).isoformat()
    snapshot = _runtime_state_snapshot()
    if not snapshot:
        return UnifiedState(compiled_at=compiled_at)

    state = UnifiedState(
        compiled_at=str(snapshot.get("compiled_at") or snapshot.get("generated_at") or compiled_at),
    )

    raw_signals = snapshot.get("signal_snapshots")
    if isinstance(raw_signals, dict):
        state.signal_snapshots = raw_signals

    raw_alerts = snapshot.get("active_alerts")
    if isinstance(raw_alerts, list):
        try:
            state.active_alerts = [
                a if isinstance(a, AnomalyAlert) else AnomalyAlert.model_validate(a)
                for a in raw_alerts
            ]
        except (ValidationError, TypeError):
            state.active_alerts = []

    raw_intel = snapshot.get("intel_collections")
    if isinstance(raw_intel, dict):
        try:
            state.intel_collections = {
                k: v if isinstance(v, IntelCollection) else IntelCollection.model_validate(v)
                for k, v in raw_intel.items()
            }
        except (ValidationError, TypeError):
            state.intel_collections = {}

    raw_domain = snapshot.get("domain_knowledge")
    if isinstance(raw_domain, dict):
        try:
            state.domain_knowledge = {
                k: (
                    v
                    if isinstance(v, DomainKnowledgeArtifact)
                    else DomainKnowledgeArtifact.model_validate(v)
                )
                for k, v in raw_domain.items()
            }
        except (ValidationError, TypeError):
            state.domain_knowledge = {}

    thinking = snapshot.get("thinking_vibe")
    if isinstance(thinking, str) and thinking.strip():
        state.thinking_vibe = thinking.strip()

    strategy = snapshot.get("strategy_vibe")
    if isinstance(strategy, str) and strategy.strip():
        state.strategy_vibe = strategy.strip()

    return state


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
    """Compose a capability-accurate help message for Little Lisa.

    Mode commands are derived from the live workflow catalog: every supported
    mode with a ``command_pattern`` gets a ``/<mode>`` line built from its
    authored ``description``. Only host-only shortcuts (``/help``, ``/settings``,
    ``/tokens``, ``/debug``, ``/consolidate``, ``/deep``, ``/quit``) and the
    runtime entry-point summary are hardcoded — everything else reflects the
    packs actually installed at startup.
    """

    catalog = get_workflow_catalog()
    lines: list[str] = ["Little Lisa", "", "Supported commands:"]

    # Mode commands — derived from the catalog, not hardcoded.
    mode_lines: list[str] = []
    default_mode = catalog.get("default")
    for mode_id, mode in sorted(catalog.items()):
        if _mode_support_gap(mode) is not None or mode_id == "default":
            continue
        if not mode.command_pattern:
            continue
        summary = (mode.description or "").strip()
        if not summary:
            summary = f"{mode_id} mode"
        mode_lines.append(f"- /{mode_id} <request> — {summary}")
    if mode_lines:
        lines.extend(mode_lines)

    # Host-only shortcuts — the minimal hardcoded surface.
    lines.extend(
        [
            "- /deep <request> routes through default mode on the deep-designated model.",
            "- /pack list|enable|disable|install|uninstall manages pack activation and runtime installs (restart required after changes).",
            "- Natural-language turns pass through the front-door route planner first.",
            "- /help capabilities shows the runtime capability snapshot.",
            "- /settings shows effective profile/timezone/debug/log-level runtime settings.",
            "- /tokens shows detailed session token breakdown by model, mode, and stage.",
            "- /debug context|capabilities|self|frontdoor shows local diagnostics.",
            "- /consolidate (or /slice) runs the explicit thread/memory compaction slice.",
            "- /quit exits the interactive console loop.",
        ]
    )

    default_description = (
        (default_mode.description or "general conversation").strip()
        if default_mode is not None
        else "general conversation"
    )
    lines.append(f"- Any other text uses default chat ({default_description}).")

    lines.extend(
        [
            "",
            "Modes and operations added through SDK_CONTENT_ROOTS run when they only "
            "need Little Lisa callbacks.",
            "",
            "Run `little-lisa cli` for terminal REPL, `little-lisa desktop` for the "
            "localhost web UI, or `little-lisa` with no args for tray-first launch.",
            "Run `little-lisa --help` for the full CLI flag reference.",
            "Run `/help capabilities` for the live mode/operation/tool/model inventory.",
        ]
    )
    return "\n".join(lines)


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
    """File-only digest provider for Little Lisa.

    Runs one deterministic, no-LLM operation inside the existing slice
    budget:

    1. Thread digest: when the thread JSONL exceeds max_thread_entries,
       compact the oldest overflow entries into a thread_digest.md
       sidecar and truncate the JSONL to the most recent entries.

    Memory eviction is no longer host-side: ``lisa-memory``'s SQLite store
    enforces TTL via its ``expires_at`` column (daily/weekly/long_term),
    so there is nothing for the slice to do. The former ``_consolidate_memory``
    reached into the in-process MCPMemoryClient's private ``_records`` /
    ``_save`` — impossible across the MCP boundary, and redundant with TTL.
    """

    def __init__(
        self,
        *,
        thread_path: Path | None,
        max_thread_entries: int = _DEFAULT_THREAD_MAX_ENTRIES,
    ) -> None:
        self._thread_path = thread_path
        self._max_thread_entries = max(1, max_thread_entries)

    def __call__(self) -> Any:
        return None

    async def slice(self, ctx: Any, policy: Any) -> SliceOutcome:
        """Run bounded thread compaction."""
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
        digest_text = f"{existing}\n\n{new_block}" if existing else new_block
        digest_path.write_text(digest_text, encoding="utf-8")

        remaining_lines = lines[overflow_count:]
        self._thread_path.write_text(
            "\n".join(remaining_lines) + "\n" if remaining_lines else "",
            encoding="utf-8",
        )

        note = f"compacted {overflow_count} thread entries into digest"
        return overflow_count, f"thread:{overflow_count} entries", note


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

    Registers all MCP servers (first-party and external) from
    ``mcp_servers.yaml`` with the launch policy registry, then compiles
    them into ``MCPClientConfig`` entries for ``MCPToolPool``. The
    ``warm`` / ``required`` / ``disabled`` launch policy is declared in
    YAML — no hardcoded server list or warm_ids set remains.

    Memory is an MCP feature: ``lisa-memory`` serves the three core
    memory calls, Mem0 (or any other HTTP/stdio memory server) serves the
    same three plus its full surface, and the generic ``memory.*`` synthetic
    aliases route to the default backend via the pool. No in-process memory
    seam remains.

    All first-party servers use ``VENV_BIN`` runtime — they run as native
    console_scripts entry points (``.venv/Scripts/lisa-*.exe``) with zero
    subprocess overhead. No ``uv`` resolution needed.

    Warm vs cold: servers marked ``warm: true`` in the manifest are
    spawned eagerly at pool ``start()`` and kept alive across turns (with
    bounded retry and event-driven restart). Servers with ``warm: false``
    (default) are cold — they are **not** connected at start; they are
    lazily connected on first ``list_tools`` or ``call_tool`` that needs
    them.
    """
    warm_configs, cold_configs = _compile_mcp_configs()
    return [*warm_configs, *cold_configs]


def _compile_mcp_configs() -> tuple[list[Any], list[Any]]:
    """Compile all MCP servers from mcp_servers.yaml and split into (warm, cold).

    All servers (first-party and external) are declared in
    ``mcp_servers.yaml`` with their ``warm`` / ``required`` / ``disabled``
    launch policy. The manifest loader produces ``MCPServerSpec`` entries
    that carry these fields; the compiler passes them through to
    ``MCPClientConfig``. No hardcoded server list or ``warm_ids`` set
    remains — adding, removing, disabling, or changing a server's launch
    policy is a YAML edit.

    The manifest's memory-backend designation (which server_id backs the
    generic ``memory.*`` aliases) is stashed on runtime state for
    ``_build_turn_hooks`` to consume.
    """
    from kernel.mcp_launch.compiler import compile_registry  # noqa: PLC0415
    from kernel.mcp_launch.policy import MCPServerRegistry  # noqa: PLC0415
    from kernel.mcp_launch.server_manifest import (  # noqa: PLC0415
        MCPServerManifestError,
        load_mcp_server_manifest,
    )

    # Make FRAMEWORK_ROOT available for ${FRAMEWORK_ROOT} expansion in
    # the manifest's path-like fields (venv_dir, command_path, etc.).
    os.environ.setdefault("FRAMEWORK_ROOT", str(FRAMEWORK_ROOT))

    registry = MCPServerRegistry()
    data_dir = str(_runtime_data_dir())
    workspace_root = str(FRAMEWORK_ROOT)
    host_manifest_path = _host_mcp_manifest_path()
    capability_preferences = _load_capability_provider_preferences(host_manifest_path)
    lock = _load_installed_packs_lock()
    runtime_installed_packages = _runtime_installed_pack_names(lock)
    selected_capability_providers_raw = lock.get("selected_capability_providers")
    selected_capability_providers = (
        selected_capability_providers_raw
        if isinstance(selected_capability_providers_raw, dict)
        else {}
    )

    # All host-owned servers (first-party) from mcp_servers.yaml. The
    # manifest loader honors ``disabled: true`` by skipping entries, and
    # parses ``warm`` / ``required`` onto the spec. The compiler passes
    # them through to MCPClientConfig. The host manifest no longer carries
    # a ``memory:`` section or memory server entries; memory backends +
    # memory tool servers are contributed by the learning pack (see
    # sdk_packs/learning/mcp_servers.yaml) and merged below.
    specs, memory_config = load_mcp_server_manifest(host_manifest_path)
    for spec in specs:
        registry.register(spec)

    excluded_pack_servers = _load_pack_server_exclusions(host_manifest_path)
    used_pack_slugs: set[str] = set()
    memory_owner_slug: str | None = None
    memory_candidates: list[dict[str, Any]] = []
    for root in _sdk_content_roots():
        pack_manifest_path = root / "mcp_servers.yaml"
        if not pack_manifest_path.exists():
            continue
        try:
            pack_specs, pack_memory = load_mcp_server_manifest(pack_manifest_path)
        except MCPServerManifestError:
            logger.warning(
                "Skipping invalid pack MCP manifest at %s",
                pack_manifest_path,
                exc_info=True,
            )
            continue

        pack_slug = _unique_pack_slug_for_root(root, used_pack_slugs)
        pack_package_name = _infer_pack_package_name(root)

        # A pack may contribute memory backends + memory tool servers. Ownership
        # is selected deterministically from host policy + lock state.
        if pack_memory.default_backend or pack_memory.backends:
            memory_candidates.append(
                {
                    "pack_slug": pack_slug,
                    "package_name": pack_package_name,
                    "manifest_path": pack_manifest_path,
                    "memory_config": pack_memory,
                }
            )

        for spec in pack_specs:
            if _pack_server_excluded(pack_slug, spec.server_id, excluded_pack_servers):
                logger.info(
                    "Pack MCP server excluded by policy: %s.%s",
                    pack_slug,
                    spec.server_id,
                )
                continue

            # Compile-time namespacing allows multiple packs to declare the same
            # server_id without collision in the global registry.
            namespaced = dataclasses.replace(spec, server_id=f"{pack_slug}.{spec.server_id}")
            registry.register(namespaced)

    selected_memory_provider = _normalize_pack_name(
        str(selected_capability_providers.get("memory") or "")
    )
    selected_memory_owner = _select_capability_provider_owner(
        candidates=memory_candidates,
        selected_provider=selected_memory_provider,
        preferred_providers=capability_preferences.get("memory", []),
        runtime_installed_packages=runtime_installed_packages,
    )
    if selected_memory_provider and selected_memory_owner is None:
        logger.warning(
            "Selected memory provider %r is not active; falling back to preference order.",
            selected_memory_provider,
        )

    if selected_memory_owner is not None:
        memory_config = selected_memory_owner["memory_config"]
        memory_owner_slug = str(selected_memory_owner.get("pack_slug") or "") or None
        selected_manifest = str(selected_memory_owner.get("manifest_path") or "")
        selected_package = str(selected_memory_owner.get("package_name") or "")
        logger.info(
            "Selected memory provider owner: package=%s slug=%s manifest=%s",
            selected_package,
            memory_owner_slug or "(none)",
            selected_manifest,
        )
        for candidate in memory_candidates:
            if candidate is selected_memory_owner:
                continue
            logger.info(
                "Ignoring memory manifest from package=%s slug=%s at %s; selected owner is %s.",
                candidate.get("package_name"),
                candidate.get("pack_slug"),
                candidate.get("manifest_path"),
                selected_package,
            )

    _RUNTIME_STATE["memory_config"] = memory_config
    _RUNTIME_STATE["memory_config_pack_slug"] = memory_owner_slug

    compiled = compile_registry(registry, data_dir=data_dir, workspace_root=workspace_root)
    warm = [c for c in compiled if c.warm]
    cold = [c for c in compiled if not c.warm]
    return warm, cold


def _unique_pack_slug_for_root(root: Path, used_slugs: set[str]) -> str:
    """Return a stable, filesystem-safe slug for one content-pack root."""
    raw = root.name.strip().lower()
    slug = re.sub(r"[^a-z0-9_-]+", "-", raw).strip("-") or "pack"
    if slug not in used_slugs:
        used_slugs.add(slug)
        return slug

    suffix = 2
    while f"{slug}-{suffix}" in used_slugs:
        suffix += 1
    unique = f"{slug}-{suffix}"
    used_slugs.add(unique)
    return unique


def _load_pack_server_exclusions(host_manifest_path: Path) -> set[str]:
    """Load host-side pack server exclusions from ``pack_policy.exclude_servers``.

    Exclusion entries are normalized lowercase strings and may be expressed as:
    - ``pack_slug.server_id`` (preferred)
    - ``pack_slug:server_id``
    - ``pack_slug/server_id``
    - ``pack_slug.*`` (exclude all servers from one pack)
    """
    policy = _load_pack_policy_dict(host_manifest_path, warning_subject="pack exclusion policy")
    entries = policy.get("exclude_servers") or []
    if not isinstance(entries, list):
        logger.warning("pack_policy.exclude_servers must be a list in %s", host_manifest_path)
        return set()

    exclusions: set[str] = set()
    for item in entries:
        if not isinstance(item, str):
            continue
        normalized = item.strip().lower()
        if normalized:
            exclusions.add(normalized)
    return exclusions


def _load_pack_policy_dict(host_manifest_path: Path, *, warning_subject: str) -> dict[str, Any]:
    """Load and return ``pack_policy`` from the host manifest."""
    if not host_manifest_path.exists():
        return {}

    try:
        raw = yaml.safe_load(host_manifest_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        logger.warning(
            "Failed to read %s from %s",
            warning_subject,
            host_manifest_path,
            exc_info=True,
        )
        return {}

    if not isinstance(raw, dict):
        return {}
    policy = raw.get("pack_policy") or {}
    if not isinstance(policy, dict):
        return {}
    return policy


def _parse_provider_priority_mapping(raw: Any) -> dict[str, list[str]]:
    """Parse a mapping of id -> ordered provider preference list."""
    if not isinstance(raw, dict):
        return {}

    preferences: dict[str, list[str]] = {}
    for key_raw, provider_values in raw.items():
        if not isinstance(key_raw, str):
            continue
        key = key_raw.strip().lower()
        if not key:
            continue

        providers: list[str]
        if isinstance(provider_values, str):
            normalized = _normalize_pack_name(provider_values)
            providers = [normalized] if normalized else []
        elif isinstance(provider_values, list):
            providers = [
                _normalize_pack_name(item)
                for item in provider_values
                if isinstance(item, str) and _normalize_pack_name(item)
            ]
        else:
            providers = []

        if providers:
            preferences[key] = providers

    return preferences


def _load_content_provider_preferences(host_manifest_path: Path) -> dict[str, dict[str, list[str]]]:
    """Load host provider preferences for content collisions.

    Preferred shape:

    ``pack_policy.preferred_providers.{modes|operations|tools}``

    Legacy compatibility:

    ``pack_policy.preferred_mode_providers`` (modes only)
    """
    policy = _load_pack_policy_dict(host_manifest_path, warning_subject="content provider preferences")

    structured = policy.get("preferred_providers") or {}
    if not isinstance(structured, dict):
        structured = {}

    mode_preferences = _parse_provider_priority_mapping(structured.get("modes") or {})
    operation_preferences = _parse_provider_priority_mapping(structured.get("operations") or {})
    tool_preferences = _parse_provider_priority_mapping(structured.get("tools") or {})

    if not mode_preferences:
        mode_preferences = _parse_provider_priority_mapping(policy.get("preferred_mode_providers") or {})

    return {
        "modes": mode_preferences,
        "operations": operation_preferences,
        "tools": tool_preferences,
    }


def _load_mode_provider_preferences(host_manifest_path: Path) -> dict[str, list[str]]:
    """Backward-compatible mode-only provider preferences view."""
    return _load_content_provider_preferences(host_manifest_path).get("modes", {})


def _load_capability_provider_preferences(host_manifest_path: Path) -> dict[str, list[str]]:
    """Load host capability-provider ownership preferences.

    Shape: ``pack_policy.preferred_capability_providers.<capability_id>``
    where each value is an ordered provider list.
    """
    policy = _load_pack_policy_dict(host_manifest_path, warning_subject="capability provider preferences")
    return _parse_provider_priority_mapping(policy.get("preferred_capability_providers") or {})


def _provider_identifiers_for_pack(*, package_name: str, pack_slug: str) -> set[str]:
    identifiers = {
        _normalize_pack_name(package_name),
        _normalize_pack_name(pack_slug),
    }
    normalized_package = _normalize_pack_name(package_name)
    if normalized_package.startswith("lisa-pack-"):
        identifiers.add(normalized_package[len("lisa-pack-"):])
    return {identifier for identifier in identifiers if identifier}


def _provider_preference_rank(
    provider_identifiers: set[str],
    preferred_providers: list[str],
) -> int:
    for index, provider in enumerate(preferred_providers):
        if provider in provider_identifiers:
            return index
    return len(preferred_providers) + 1


def _select_capability_provider_owner(
    *,
    candidates: list[dict[str, Any]],
    selected_provider: str,
    preferred_providers: list[str],
    runtime_installed_packages: set[str],
) -> dict[str, Any] | None:
    if not candidates:
        return None

    selected = _normalize_pack_name(selected_provider)
    if selected:
        for candidate in candidates:
            identifiers = _provider_identifiers_for_pack(
                package_name=str(candidate.get("package_name") or ""),
                pack_slug=str(candidate.get("pack_slug") or ""),
            )
            if selected in identifiers:
                return candidate

    def _sort_key(candidate: dict[str, Any]) -> tuple[int, int, str, str]:
        package_name = _normalize_pack_name(str(candidate.get("package_name") or ""))
        pack_slug = _normalize_pack_name(str(candidate.get("pack_slug") or ""))
        identifiers = _provider_identifiers_for_pack(package_name=package_name, pack_slug=pack_slug)
        preference_rank = _provider_preference_rank(identifiers, preferred_providers)
        runtime_rank = 0 if package_name in runtime_installed_packages else 1
        return (preference_rank, runtime_rank, package_name, pack_slug)

    return min(candidates, key=_sort_key)


def _pack_server_excluded(pack_slug: str, declared_server_id: str, exclusions: set[str]) -> bool:
    """Return True when host policy excludes a pack-contributed server."""
    slug = pack_slug.strip().lower()
    server = declared_server_id.strip().lower()
    if not slug or not server or not exclusions:
        return False

    dotted = f"{slug}.{server}"
    return (
        dotted in exclusions
        or f"{slug}:{server}" in exclusions
        or f"{slug}/{server}" in exclusions
        or f"{slug}.*" in exclusions
    )


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
    """Build TurnHooks wired to Little Lisa callbacks and memory operations.

    Delegates to ``LittleLisaTurnHooks`` in ``turn_hooks.py`` which holds
    shared state as instance attributes instead of capturing it via closures.

    Memory is now an MCP feature, not a kernel seam. The host owns the
    pool-backed ``memory.add`` / ``memory.search`` / ``memory.list`` aliases
    and generic operation callbacks; the learning pack owns memory recall and
    synthesis behavior by calling those aliases through ``ctx.execute_tool_fn``.
    """
    from kernel.slice import SliceTuning  # noqa: PLC0415

    slice_tuning_dict, _explicit_budget_ms = _resolve_slice_tuning()
    slice_tuning = SliceTuning.from_mapping(slice_tuning_dict)
    state_provider = LittleLisaStateProvider(
        thread_path=_runtime_thread_path(),
        max_thread_entries=_resolve_int_env(
            _LITTLE_LISA_THREAD_MAX_ENTRIES_ENV, _DEFAULT_THREAD_MAX_ENTRIES
        ),
    )

    persisted_last_slice_finished_at: float | None = None
    if context is not None:
        raw = context.get("last_slice_finished_at")
        if isinstance(raw, (int, float)):
            persisted_last_slice_finished_at = float(raw)

    provider = turn_hooks_module.LittleLisaTurnHooks(
        state_provider=state_provider,
        complete_live=_complete_live,
        resolve_model=_resolve_model,
        execute_tool=_execute_tool,
    )

    async def _execute_operation(
        operation_id: str,
        ctx: OperationExecutionContext,
    ) -> OperationResult:
        """Execute an operation with host callbacks wired.

        Learning-pack memory operations reach ``memory.add`` / ``memory.list``
        via ``ctx.execute_tool_fn`` like any other tool. The kernel turn
        runner injects that executor; this host hook adds the remaining
        runtime defaults such as recall limits and live introspection.
        """
        ctx.memory_recall_default_limit = 8
        ctx.introspect_fn = provider.introspect
        return await execute_operation(operation_id, ctx)

    async def _on_turn_complete(
        mode_id: str,
        query: str,
        response_text: str,
        messages: list[dict[str, Any]],
    ) -> None:
        """Update the context dict with turn completion metadata."""
        if context is None:
            return
        turn_input_summary = _summarize_turn_messages(messages)
        assistant_message_count = sum(
            1 for msg in messages if str(msg.get("role") or "").strip().lower() == "assistant"
        )
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
        resolve_model=provider.resolve_model_hook,
        complete=provider.complete,
        mcp_tool_pool=_runtime_warm_pool(),
        mcp_tool_servers=_runtime_cold_mcp_configs(),
        execute_operation=_execute_operation,
        assemble_system_prompt=_assemble_system_prompt,
        assemble_user_context=_assemble_dynamic_user_context,
        on_turn_complete=_on_turn_complete,
        slice_tuning=slice_tuning,
        pre_turn_slice=provider.pre_turn_slice,
        post_turn_slice=provider.post_turn_slice,
        explicit_slice=provider.explicit_slice,
        last_slice_finished_at=persisted_last_slice_finished_at,
        workflow_turn_runner=_run_workflow_turn,
        workflow_session_factory=create_workflow_session,
        state_provider=_state_provider,
        build_workshop_context_bundle_fn=_build_workshop_context_bundle,
        introspect=provider.introspect,
    )


def _runtime_warm_pool() -> Any:
    """Return the host-owned warm MCP pool, or None when not initialized.

    When None (e.g. tests calling ``_build_turn_hooks`` without
    ``_initialize_runtime``), ``run_turn`` falls back to building a
    per-turn pool from ``mcp_tool_servers``. To preserve that fallback
    shape in the no-pool case, ``_runtime_cold_mcp_configs`` returns
    the full server list so existing tests keep working.
    """
    return _RUNTIME_STATE.get("mcp_tool_pool")


def _runtime_cold_mcp_configs() -> list[Any]:
    """Return cold MCP configs for the per-turn fallback pool.

    When a warm pool is active, only cold servers are returned (warm
    servers live in the pool). When no warm pool is active, the full
    server list is returned so ``run_turn``'s per-turn fallback can
    spawn everything — preserving the pre-warm-pool behavior for tests
    and minimal hosts.
    """
    pool = _RUNTIME_STATE.get("mcp_tool_pool")
    cold = _RUNTIME_STATE.get("mcp_cold_configs")
    if pool is not None and cold is not None:
        return list(cold)
    return _build_mcp_tool_servers()


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
        max_thread_entries=_resolve_int_env(
            _LITTLE_LISA_THREAD_MAX_ENTRIES_ENV, _DEFAULT_THREAD_MAX_ENTRIES
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


async def handle_message(message: str) -> str:
    """Route one console message through the kernel turn runner.

    Host shortcut commands are dispatched first, then model-backed routing
    (front-door + run_turn) handles everything else.
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

    result = await _dispatch_host_command(stripped, lowered, message)
    if result is not None:
        return result

    return await _route_model_turn(message, stripped, lowered)


async def _dispatch_host_command(
    stripped: str,
    lowered: str,
    message: str,
) -> str | None:
    """Try each host shortcut command; return response or None to continue."""
    if lowered.startswith("/help") or lowered == "help":
        return await _handle_help_command(stripped, lowered, message)
    if lowered in {"/tokens", "tokens"}:
        return _handle_simple_command(message, "tokens", _format_token_stats())
    if lowered in {"/settings", "settings"}:
        return _handle_simple_command(message, "settings", _render_runtime_settings())
    if lowered in {"/consolidate", "consolidate", "/slice", "slice"}:
        return await _run_explicit_slice_command(message=message)
    if lowered.startswith("/pack"):
        return await _handle_pack_command(stripped, message)
    if lowered.startswith("/debug"):
        return _handle_debug_command(stripped, message)
    return None


def _load_installed_packs_lock() -> dict[str, Any]:
    """Load installed pack provenance metadata from disk."""
    path = _runtime_pack_lock_path()
    default_payload: dict[str, Any] = {
        "packs": [],
        "selected_capability_providers": {},
    }
    if not path.exists():
        return default_payload
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default_payload
    if not isinstance(decoded, dict):
        return default_payload

    packs = decoded.get("packs")
    if not isinstance(packs, list):
        packs = []

    normalized_packs: list[dict[str, Any]] = []
    for item in packs:
        if not isinstance(item, dict):
            continue
        package_name = _normalize_pack_name(str(item.get("package_name") or ""))
        if not package_name:
            continue
        source_kind = str(item.get("source_kind") or "package").strip().lower() or "package"
        normalized_packs.append(
            {
                **item,
                "package_name": package_name,
                "source_kind": source_kind,
                "enabled": bool(item.get("enabled", True)),
            }
        )

    selected = decoded.get("selected_capability_providers")
    selected_map: dict[str, str] = {}
    if isinstance(selected, dict):
        for capability_raw, provider_raw in selected.items():
            if not isinstance(capability_raw, str) or not isinstance(provider_raw, str):
                continue
            capability_id = capability_raw.strip().lower()
            provider_id = _normalize_pack_name(provider_raw)
            if capability_id and provider_id:
                selected_map[capability_id] = provider_id

    return {
        "packs": normalized_packs,
        "selected_capability_providers": selected_map,
    }


def _write_installed_packs_lock(payload: dict[str, Any]) -> None:
    """Persist installed pack provenance metadata to disk."""
    path = _runtime_pack_lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _installed_pack_roots_by_name() -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for root in discover_installed_content_roots():
        package_name = _infer_pack_package_name(root)
        if not package_name or package_name in roots:
            continue
        roots[package_name] = root
    return roots


def _pack_alias_map(lock: dict[str, Any]) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    installed_roots = _installed_pack_roots_by_name()

    for package_name, root in installed_roots.items():
        for alias in _pack_aliases(package_name, root=root):
            alias_map.setdefault(alias, package_name)

    for package_name in _always_on_pack_names():
        for alias in _pack_aliases(package_name):
            alias_map.setdefault(alias, package_name)

    entries = lock.get("packs") if isinstance(lock, dict) else None
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        package_name = _normalize_pack_name(str(entry.get("package_name") or ""))
        if not package_name:
            continue
        for alias in _pack_aliases(package_name):
            alias_map.setdefault(alias, package_name)

    return alias_map


def _resolve_pack_name(raw: str, lock: dict[str, Any] | None = None) -> str:
    normalized = _normalize_pack_name(raw)
    if not normalized:
        return ""

    lock_payload = lock if isinstance(lock, dict) else _load_installed_packs_lock()
    aliases = _pack_alias_map(lock_payload)
    if normalized in aliases:
        return aliases[normalized]

    prefixed = normalized if normalized.startswith("lisa-pack-") else f"lisa-pack-{normalized}"
    return aliases.get(prefixed, "")


def _clear_selected_provider_for_package(lock: dict[str, Any], package_name: str) -> list[str]:
    selected = lock.get("selected_capability_providers")
    if not isinstance(selected, dict):
        lock["selected_capability_providers"] = {}
        return []

    cleared: list[str] = []
    for capability_id, provider_id in list(selected.items()):
        if not isinstance(capability_id, str) or not isinstance(provider_id, str):
            continue
        if _normalize_pack_name(provider_id) != package_name:
            continue
        selected.pop(capability_id, None)
        cleared.append(capability_id)
    return cleared


def _pack_contributes_memory_capability(package_name: str) -> bool:
    normalized = _normalize_pack_name(package_name)
    if not normalized:
        return False

    from kernel.mcp_launch.server_manifest import (  # noqa: PLC0415
        MCPServerManifestError,
        load_mcp_server_manifest,
    )

    for root in discover_installed_content_roots():
        if _infer_pack_package_name(root) != normalized:
            continue
        manifest_path = root / "mcp_servers.yaml"
        if not manifest_path.is_file():
            continue
        try:
            _, memory = load_mcp_server_manifest(manifest_path)
        except MCPServerManifestError:
            continue
        if memory.default_backend or memory.backends:
            return True

    return normalized.startswith("lisa-pack-mem")


def _auto_select_capability_provider_on_install(
    lock: dict[str, Any],
    package_name: str,
) -> None:
    selected = lock.get("selected_capability_providers")
    if not isinstance(selected, dict):
        selected = {}
        lock["selected_capability_providers"] = selected

    capability_id = "memory"
    if selected.get(capability_id):
        return
    if package_name in _always_on_pack_names():
        return
    if not _pack_contributes_memory_capability(package_name):
        return

    selected[capability_id] = package_name


def _set_pack_enabled(package_name: str, *, enabled: bool) -> tuple[bool, str]:
    """Persist explicit activation for one pack and request host restart."""
    lock = _load_installed_packs_lock()
    resolved = _resolve_pack_name(package_name, lock)
    if not resolved:
        return False, (
            f"Unknown pack '{package_name}'. Run /pack list to see available packs "
            "and their canonical names."
        )

    always_on = _always_on_pack_names()
    if not enabled and resolved in always_on:
        return False, f"Pack '{resolved}' is always on and cannot be disabled."

    entries = [entry for entry in lock.get("packs", []) if isinstance(entry, dict)]
    index = next(
        (
            i
            for i, entry in enumerate(entries)
            if _normalize_pack_name(str(entry.get("package_name") or "")) == resolved
        ),
        -1,
    )
    now_iso = datetime.now(UTC).isoformat()

    if index >= 0:
        entry = dict(entries[index])
        entry["package_name"] = resolved
        entry["enabled"] = enabled
        entry["updated_at"] = now_iso
        entry.setdefault("source_kind", "bundled")
        if entry.get("source_kind") == "bundled":
            entry.setdefault("source_ref", "bundled")
        entries[index] = entry
    else:
        entries.append(
            {
                "package_name": resolved,
                "enabled": enabled,
                "source_kind": "bundled",
                "source_ref": "bundled",
                "updated_at": now_iso,
            }
        )

    lock["packs"] = entries

    cleared = []
    if not enabled:
        cleared = _clear_selected_provider_for_package(lock, resolved)

    _write_installed_packs_lock(lock)
    action = "enable" if enabled else "disable"
    _request_runtime_restart(reason=f"pack {action}: {resolved}")

    if enabled:
        return (
            True,
            f"Enabled '{resolved}'. Restart required to reload content roots and MCP manifests.",
        )

    if cleared:
        return (
            True,
            f"Disabled '{resolved}' and cleared provider selections for: {', '.join(sorted(cleared))}. "
            "Restart required to reload content roots and MCP manifests.",
        )
    return (
        True,
        f"Disabled '{resolved}'. Restart required to reload content roots and MCP manifests.",
    )


def _pack_allowed_prefixes() -> list[str]:
    """Return allowed package-name prefixes for runtime pack installs."""
    configured = os.getenv(_LITTLE_LISA_PACK_ALLOWED_PREFIXES_ENV, "").strip()
    if not configured:
        return ["lisa-pack-"]
    values = [value.strip().lower() for value in configured.split(",") if value.strip()]
    return values or ["lisa-pack-"]


def _pack_allowed_git_hosts() -> set[str]:
    """Return allowed hosts for git-based pack installation refs."""
    configured = os.getenv(_LITTLE_LISA_PACK_GIT_ALLOW_HOSTS_ENV, "").strip()
    if not configured:
        return {"github.com"}
    values = {value.strip().lower() for value in configured.split(",") if value.strip()}
    return values or {"github.com"}


def _pack_require_git_ref() -> bool:
    """Return whether git install refs must pin @tag/@branch/@sha."""
    configured = os.getenv(_LITTLE_LISA_PACK_REQUIRE_GIT_REF_ENV, "true").strip().lower()
    return configured not in {"0", "false", "no", "off"}


def _runtime_restart_auto_enabled() -> bool:
    """Return whether the host should auto-restart after pack lifecycle changes."""
    configured = os.getenv(_LITTLE_LISA_AUTO_RESTART_ENV, "true").strip().lower()
    return configured not in {"0", "false", "no", "off"}


def _is_git_pack_spec(spec: str) -> bool:
    return spec.strip().lower().startswith("git+")


def _is_local_pack_spec(spec: str) -> bool:
    raw = spec.strip().strip('"')
    if not raw:
        return False
    return Path(raw).expanduser().is_dir()


def _read_local_pack_project_name(path: Path) -> str:
    """Return ``project.name`` from a local pack's pyproject.toml, or empty."""
    pyproject = path / "pyproject.toml"
    if not pyproject.is_file():
        return ""
    try:
        payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return ""
    project = payload.get("project")
    if not isinstance(project, dict):
        return ""
    name = project.get("name")
    return str(name).strip() if isinstance(name, str) else ""


def _extract_package_name_from_spec(spec: str) -> str:
    """Infer package name from package or git install spec."""
    raw = spec.strip()
    if not raw:
        return ""
    if _is_local_pack_spec(raw):
        path = Path(raw.strip('"')).expanduser()
        project_name = _read_local_pack_project_name(path)
        if project_name:
            return project_name
        return path.name.strip()
    if _is_git_pack_spec(raw):
        egg_marker = "#egg="
        lowered = raw.lower()
        marker_index = lowered.find(egg_marker)
        if marker_index >= 0:
            return raw[marker_index + len(egg_marker):].strip()
        base = raw.split("@", 1)[0].rstrip("/")
        tail = base.rsplit("/", 1)[-1]
        return tail[:-4] if tail.lower().endswith(".git") else tail

    first = raw.split()[0]
    name = re.split(r"[<>=!~\[]", first, maxsplit=1)[0]
    return name.strip()


def _extract_git_host(spec: str) -> str:
    """Best-effort host extraction for git+https refs."""
    raw = spec.strip()
    if not raw.lower().startswith("git+https://"):
        return ""
    remainder = raw[len("git+https://"):]
    if not remainder:
        return ""
    return remainder.split("/", 1)[0].strip().lower()


def _validate_pack_spec_for_install(spec: str) -> tuple[bool, str]:
    """Validate a runtime pack install reference against host trust policy."""
    normalized = spec.strip()
    if not normalized:
        return False, "Usage: /pack install <package-ref-or-git-ref>"

    if _is_local_pack_spec(normalized):
        pack_path = Path(normalized.strip('"')).expanduser()
        project_name = _read_local_pack_project_name(pack_path)
        if not project_name:
            return False, (
                f"Local pack '{pack_path}' is missing a valid pyproject.toml project.name."
            )
        package_name = project_name.lower()
        if not any(package_name.startswith(prefix) for prefix in _pack_allowed_prefixes()):
            return False, (
                f"Pack '{package_name}' rejected: allowed package prefixes are "
                f"{', '.join(_pack_allowed_prefixes())}."
            )
        return True, ""

    if _is_git_pack_spec(normalized):
        host = _extract_git_host(normalized)
        if not host:
            return False, "Only git+https refs are supported for runtime git installs."
        if host not in _pack_allowed_git_hosts():
            return False, f"Git host '{host}' is not in the allowlist."
        if _pack_require_git_ref() and "@" not in normalized:
            return False, "Git refs must pin @tag/@branch/@sha (policy LITTLE_LISA_PACK_REQUIRE_GIT_REF)."
        package_name = _extract_package_name_from_spec(normalized).lower()
        if not any(package_name.startswith(prefix) for prefix in _pack_allowed_prefixes()):
            return False, (
                f"Pack '{package_name}' rejected: allowed package prefixes are "
                f"{', '.join(_pack_allowed_prefixes())}."
            )
        return True, ""

    package_name = _extract_package_name_from_spec(normalized).lower()
    if not package_name:
        return False, "Invalid package ref."
    if not any(package_name.startswith(prefix) for prefix in _pack_allowed_prefixes()):
        return False, (
            f"Pack '{package_name}' rejected: allowed package prefixes are "
            f"{', '.join(_pack_allowed_prefixes())}."
        )
    return True, ""


async def _run_pack_install(spec: str) -> tuple[bool, str]:
    """Install one pack ref using uv pip and persist provenance metadata."""
    is_allowed, reason = _validate_pack_spec_for_install(spec)
    if not is_allowed:
        return False, reason

    install_target = spec.strip().strip('"') if _is_local_pack_spec(spec) else spec
    process = await asyncio.create_subprocess_exec(
        "uv",
        "pip",
        "install",
        install_target,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(FRAMEWORK_ROOT),
    )
    output_bytes, _ = await process.communicate()
    output = output_bytes.decode("utf-8", errors="replace") if output_bytes else ""
    if process.returncode != 0:
        preview = "\n".join(output.splitlines()[-12:]).strip()
        return False, f"Install failed for '{spec}'.\n{preview}"

    package_name = _normalize_pack_name(_extract_package_name_from_spec(spec))
    if not package_name:
        return False, f"Install succeeded for '{spec}' but package name could not be inferred."
    source_kind = (
        "local"
        if _is_local_pack_spec(spec)
        else ("git" if _is_git_pack_spec(spec) else "package")
    )
    lock = _load_installed_packs_lock()
    entries = [entry for entry in lock.get("packs", []) if isinstance(entry, dict)]
    now_iso = datetime.now(UTC).isoformat()
    filtered = [
        entry
        for entry in entries
        if _normalize_pack_name(str(entry.get("package_name") or "")) != package_name
    ]
    filtered.append(
        {
            "package_name": package_name,
            "installed_at": now_iso,
            "source_kind": source_kind,
            "source_ref": install_target,
            "enabled": True,
            "updated_at": now_iso,
        }
    )
    lock["packs"] = filtered
    _auto_select_capability_provider_on_install(lock, package_name)
    _write_installed_packs_lock(lock)

    _request_runtime_restart(reason=f"pack install: {package_name}")
    return True, (
        f"Installed '{package_name}' from {source_kind} source. "
        "Restart required to reload entry points, pack catalogs, and MCP manifests."
    )


async def _run_pack_uninstall(package_name: str) -> tuple[bool, str]:
    """Uninstall one pack and update provenance metadata."""
    requested = package_name.strip()
    if not requested:
        return False, "Usage: /pack uninstall <package-name>"

    lock = _load_installed_packs_lock()
    normalized = _resolve_pack_name(requested, lock) or _normalize_pack_name(requested)
    if not normalized:
        return False, "Usage: /pack uninstall <package-name>"

    process = await asyncio.create_subprocess_exec(
        "uv",
        "pip",
        "uninstall",
        "-y",
        normalized,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(FRAMEWORK_ROOT),
    )
    output_bytes, _ = await process.communicate()
    output = output_bytes.decode("utf-8", errors="replace") if output_bytes else ""
    if process.returncode != 0:
        preview = "\n".join(output.splitlines()[-12:]).strip()
        return False, f"Uninstall failed for '{normalized}'.\n{preview}"

    entries = [entry for entry in lock.get("packs", []) if isinstance(entry, dict)]
    filtered = [
        entry
        for entry in entries
        if _normalize_pack_name(str(entry.get("package_name") or "")) != normalized
    ]
    lock["packs"] = filtered
    cleared = _clear_selected_provider_for_package(lock, normalized)
    _write_installed_packs_lock(lock)

    _request_runtime_restart(reason=f"pack uninstall: {normalized}")
    if cleared:
        cleared_text = ", ".join(sorted(cleared))
        return True, (
            f"Uninstalled '{normalized}' and cleared provider selections for: {cleared_text}. "
            "Restart required to unload entry points, pack catalogs, and MCP manifests."
        )
    return True, (
        f"Uninstalled '{normalized}'. "
        "Restart required to unload entry points, pack catalogs, and MCP manifests."
    )


def _request_runtime_restart(*, reason: str, resume_message: str = "") -> None:
    """Mark runtime as requiring restart and optionally persist a resume message."""
    _RUNTIME_STATE["restart_required"] = True
    _RUNTIME_STATE["restart_reason"] = reason
    if resume_message.strip():
        os.environ[_LITTLE_LISA_RESTART_RESUME_MESSAGE_ENV] = resume_message.strip()


def _consume_restart_resume_message() -> str:
    """Read and clear the pending resume message passed across process restart."""
    message = str(os.getenv(_LITTLE_LISA_RESTART_RESUME_MESSAGE_ENV, "") or "").strip()
    if message:
        os.environ.pop(_LITTLE_LISA_RESTART_RESUME_MESSAGE_ENV, None)
    return message


def _restart_current_process() -> None:
    """Exec the current process in-place to reload entry points and manifests."""
    try:
        argv = list(getattr(sys, "orig_argv", []))
    except Exception:  # pylint: disable=broad-except
        argv = []

    if argv:
        os.execv(sys.executable, argv)

    fallback = [sys.executable, *sys.argv]
    os.execv(sys.executable, fallback)


def _escape_markdown_cell(value: str) -> str:
    """Escape a string for safe use inside a Markdown pipe-table cell."""
    return value.replace("|", "\\|").replace("\n", " ")


def _resolve_pack_source_kind(
    name: str,
    entry: dict[str, Any],
    always_on: set[str],
    runtime_installed: set[str],
) -> str:
    """Resolve the source-kind label for a pack entry."""
    source_kind = str(entry.get("source_kind") or "").strip().lower()
    if source_kind:
        return source_kind
    if name in always_on:
        return "always-on"
    if name in runtime_installed:
        return "package"
    return "bundled"


def _render_pack_list() -> str:
    """Render discovered packs, activation state, and capability-provider selections."""
    lock = _load_installed_packs_lock()
    entries = [entry for entry in lock.get("packs", []) if isinstance(entry, dict)]
    installed_roots = _installed_pack_roots_by_name()
    always_on = _always_on_pack_names()
    runtime_installed = _runtime_installed_pack_names(lock)
    enabled_map = _runtime_enabled_pack_map(lock)
    entry_map = {
        _normalize_pack_name(str(entry.get("package_name") or "")): entry
        for entry in entries
        if _normalize_pack_name(str(entry.get("package_name") or ""))
    }

    all_packages = set(installed_roots) | set(entry_map) | always_on
    if not all_packages:
        return "No packs discovered yet."

    lines = ["## Pack Activation", ""]
    lines.append("| Pack | State | Source | Root |")
    lines.append("| --- | --- | --- | --- |")
    for package_name in sorted(all_packages):
        entry = entry_map.get(package_name) or {}
        source_kind = _resolve_pack_source_kind(
            package_name, entry, always_on, runtime_installed
        )
        active = _pack_is_active(
            package_name,
            always_on_packs=always_on,
            runtime_installed=runtime_installed,
            enabled_map=enabled_map,
        )
        if package_name in always_on:
            activation = "always-on"
        elif active:
            activation = "enabled"
        else:
            activation = "disabled"
        root = installed_roots.get(package_name)
        root_text = str(root) if root is not None else "(not discovered in current environment)"
        lines.append(
            f"| {_escape_markdown_cell(package_name)} | {_escape_markdown_cell(activation)} | "
            f"{_escape_markdown_cell(source_kind)} | {_escape_markdown_cell(root_text)} |"
        )

    selected = lock.get("selected_capability_providers")
    if isinstance(selected, dict) and selected:
        lines.append("")
        lines.append("## Selected Capability Providers")
        lines.append("")
        lines.append("| Capability | Provider |")
        lines.append("| --- | --- |")
        for capability_id, provider_id in sorted(selected.items()):
            cap = _escape_markdown_cell(str(capability_id))
            prov = _escape_markdown_cell(str(provider_id))
            lines.append(f"| {cap} | {prov} |")

    lines.append("")
    lines.append("Tip: use /pack enable <name> or /pack disable <name> for bundled/runtime packs.")
    return "\n".join(lines)


async def _handle_pack_command(stripped: str, message: str) -> str:
    """Handle /pack commands for runtime pack management."""
    command_text = stripped[5:].strip()
    if not command_text:
        return _handle_simple_command(
            message,
            "pack",
            "Usage: /pack list | /pack enable <pack> | /pack disable <pack> | /pack install <spec> | /pack uninstall <package>",
        )

    verb, _, remainder = command_text.partition(" ")
    action = verb.strip().lower()
    arg = remainder.strip()

    resume_message = ""
    if action in {"install", "uninstall"} and " --resume " in arg:
        arg, _, maybe_resume = arg.partition(" --resume ")
        resume_message = maybe_resume.strip()

    if action == "list":
        return _handle_simple_command(message, "pack", _render_pack_list())

    if action == "enable":
        if not arg:
            return _handle_simple_command(message, "pack", "Usage: /pack enable <pack>")
        ok, details = _set_pack_enabled(arg, enabled=True)
        if not ok:
            return _handle_simple_command(message, "pack", details)
        return _handle_simple_command(message, "pack", details)

    if action == "disable":
        if not arg:
            return _handle_simple_command(message, "pack", "Usage: /pack disable <pack>")
        ok, details = _set_pack_enabled(arg, enabled=False)
        if not ok:
            return _handle_simple_command(message, "pack", details)
        return _handle_simple_command(message, "pack", details)

    if action == "install":
        ok, details = await _run_pack_install(arg)
        if not ok:
            return _handle_simple_command(message, "pack", details)
        if resume_message:
            _request_runtime_restart(
                reason=str(_RUNTIME_STATE.get("restart_reason") or "pack install"),
                resume_message=resume_message,
            )
        return _handle_simple_command(message, "pack", details)

    if action == "uninstall":
        ok, details = await _run_pack_uninstall(arg)
        if not ok:
            return _handle_simple_command(message, "pack", details)
        if resume_message:
            _request_runtime_restart(
                reason=str(_RUNTIME_STATE.get("restart_reason") or "pack uninstall"),
                resume_message=resume_message,
            )
        return _handle_simple_command(message, "pack", details)

    return _handle_simple_command(
        message,
        "pack",
        "Unknown /pack action. Supported: list, enable, disable, install, uninstall.",
    )


def _handle_simple_command(message: str, mode_id: str, response_text: str) -> str:
    """Handle a simple shortcut command that just renders static output."""
    context = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "raw_message": message,
        "mode_id": mode_id,
        "handoff": f"{mode_id} shortcut",
    }
    _set_last_turn_context(context)
    _append_turn_log(context=context, response_text=response_text)
    _append_thread_entry(mode_id=mode_id, query=message, response_text=response_text)
    _RUNTIME_STATE["current_mode_id"] = mode_id
    _RUNTIME_STATE["current_stage_id"] = "-"
    _RUNTIME_STATE["next_model_id"] = "-"
    return response_text


async def _handle_help_command(stripped: str, lowered: str, message: str) -> str:
    """Handle /help and /help capabilities."""
    help_topic = ""
    if lowered.startswith("/help"):
        help_topic = stripped[5:].strip().lower()
    if help_topic in {"capabilities", "capability"}:
        response_text = _render_capability_help()
    else:
        response_text = _format_help()
    return _handle_simple_command(message, "help", response_text)


def _handle_debug_command(stripped: str, message: str) -> str:
    """Handle /debug subcommands."""
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
    return _handle_simple_command(message, "debug", response_text)


async def _route_model_turn(message: str, stripped: str, lowered: str) -> str:
    """Handle model-backed routing: front-door planning and run_turn dispatch."""
    _require_models_present()

    deep_query = _explicit_deep_query(message)
    deep_route = deep_query is not None
    if deep_route:
        mode_id = "default"
        query = deep_query or ""
    elif message.lstrip().startswith("/"):
        mode_id, query = parse_route(message)
    else:
        direct = await _try_frontdoor_shortcut(message)
        if direct is not None:
            return direct
        route_decision = await _plan_frontdoor_route(
            raw_message=message,
            conversation_excerpt_markdown=_build_recent_thread_excerpt(),
        )
        clarification = _try_frontdoor_clarification(message, route_decision)
        if clarification is not None:
            return clarification
        mode_id = route_decision.selected_mode_id
        query = route_decision.normalized_query or message

    return await _execute_mode_turn(message, mode_id, query, deep_route)


async def _try_frontdoor_shortcut(message: str) -> str | None:
    """Try the front-door direct-response optimization; return response or None."""
    route_decision = await _plan_frontdoor_route(
        raw_message=message,
        conversation_excerpt_markdown=_build_recent_thread_excerpt(),
    )
    if (
        route_decision.clarification_needed
        or not route_decision.direct_response
        or route_decision.selected_mode_id != "default"
    ):
        return None

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

    default_alias = _model_alias_for_mode("default")
    resolved_default = _resolve_model(default_alias)
    _apply_turn_usage(
        TurnResult(
            content=response_text,
            prompt_tokens=0,
            completion_tokens=0,
        ),
        model_id=resolved_default.model_id,
        mode_id="default",
        stage_id="-",
        frontdoor_routing=planner_debug,
    )
    return response_text


def _try_frontdoor_clarification(
    message: str,
    route_decision: FrontDoorRouteDecision,
) -> str | None:
    """Return a clarification question if the front-door requests one."""
    if not (route_decision.clarification_needed and route_decision.clarification_question):
        return None

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


def _handle_unsupported_mode(
    message: str,
    mode: WorkflowModeConfig,
    query: str,
    deep_route: bool,
) -> str:
    """Handle a mode that Little Lisa cannot run."""
    logger.info(
        "unsupported mode %s in Little Lisa (gap=%s)",
        mode.mode_id,
        _mode_support_gap(mode),
    )
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


async def _execute_mode_turn(
    message: str,
    mode_id: str,
    query: str,
    deep_route: bool,
) -> str:
    """Resolve mode, build context, and run the turn through the kernel."""
    catalog = get_workflow_catalog()
    mode = catalog.get(mode_id) or catalog["default"]
    preview_model_id = _preferred_model_id_for_route(
        raw_message=message,
        mode_id=mode.mode_id,
        deep_route=deep_route,
    )
    _set_runtime_route_preview(mode_id=mode.mode_id, preferred_model_id=preview_model_id)
    if _mode_support_gap(mode) is not None:
        return _handle_unsupported_mode(message, mode, query, deep_route)

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
        logger.error("_run_once turn failed: %s: %s", type(exc).__name__, exc)
        _render_runtime_error(exc, title="Turn Error")
        return
    if _runtime_debug_enabled():
        context = _last_turn_context()
        if context is not None:
            _render_turn_context(context)
    await _render_response(response)
    _render_restart_required_notice_if_needed()


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


def _build_desktop_subprocess_args(
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
    timezone: str,
    state_snapshot: str,
    streaming: str,
    host_log: str,
) -> list[str]:
    """Build the CLI args for the desktop host subprocess."""
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
    _append_flag_args(args, log_level, "--log-level")
    _append_flag_args(args, log, "--log")
    if no_log:
        args.append("--no-log")
    _append_flag_args(args, data_dir, "--data-dir")
    _append_flag_args(args, memory_db, "--memory-db")
    _append_flag_args(args, thread_dir, "--thread-dir")
    if no_thread:
        args.append("--no-thread")
    _append_flag_args(args, session_context, "--session-context")
    _append_flag_args(args, timezone, "--timezone")
    _append_flag_args(args, state_snapshot, "--state-snapshot")
    _append_flag_args(args, streaming, "--streaming")
    _append_flag_args(args, host_log, "--host-log")
    return args


def _append_flag_args(args: list[str], value: str, flag: str) -> None:
    """Append a flag+value pair to args if value is non-empty."""
    if value.strip():
        args.extend([flag, value])


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
        on_startup=astart_warm_mcp_pool,
        on_cleanup=aclose_warm_mcp_pool,
    )


def _build_repl_prompt_session() -> PromptSession[str] | None:
    if PromptSession is None or FileHistory is None or WordCompleter is None:
        return None
    history_path = _runtime_data_dir() / _REPL_HISTORY_FILE_NAME
    history_path.parent.mkdir(parents=True, exist_ok=True)
    return PromptSession(
        history=_SurrogateSafeFileHistory(str(history_path)),
        completer=WordCompleter(list(_repl_command_hints()), ignore_case=True),
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
    timezone: str,
    state_snapshot: str,
    streaming: str,
    host_log: str,
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
            timezone=timezone,
            state_snapshot=state_snapshot,
            streaming=streaming,
            host_log=host_log,
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
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
        host_log_path=host_log,
    )
    _run_desktop_host(
        host=desktop_host,
        port=desktop_port,
        token=resolved_token,
        open_browser=True,
    )


async def _run_repl() -> None:
    """Run the interactive REPL loop."""
    await astart_warm_mcp_pool()
    _print_startup_banner()
    console.print(
        Panel(
            "Type /help, /help capabilities, /debug context, "
            "/debug frontdoor, /pack list, /search <query>, /deep <query>, or /quit.",
            title="Session",
            border_style=STYLE_BORDER_SESSION,
        )
    )

    resume_message = _consume_restart_resume_message()
    if resume_message:
        console.print(
            Panel(
                f"Resuming request after restart:\n{resume_message}",
                title="Resume",
                border_style="steel_blue3",
            )
        )

    prompt_session = _build_repl_prompt_session()
    if prompt_session is None:
        console.print(f"[{STYLE_TEXT_MUTED}]prompt_toolkit unavailable; using basic input mode.[/]")

    restart_requested = False
    try:
        if resume_message:
            await _process_repl_turn(resume_message)
        restart_requested = await _run_repl_loop(prompt_session)
    finally:
        await aclose_warm_mcp_pool()

    if restart_requested and _runtime_restart_auto_enabled():
        _restart_current_process()


async def _run_repl_loop(prompt_session: PromptSession[str] | None) -> bool:
    """Process REPL turns until the user quits or sends EOF."""
    while True:
        message = await _read_repl_input(prompt_session)
        if message is None:
            _render_session_exit_summary()
            return False
        if not message:
            continue
        if _is_repl_quit_command(message):
            _render_session_exit_summary()
            console.print(f"[{STYLE_TEXT_MUTED}]Session closed.[/]")
            return False
        await _process_repl_turn(message)
        if _runtime_restart_required():
            _render_restart_required_notice_if_needed()
            _render_session_exit_summary()
            if _runtime_restart_auto_enabled():
                console.print(f"[{STYLE_TEXT_MUTED}]Applying restart now...[/]")
            else:
                console.print(
                    f"[{STYLE_TEXT_MUTED}]Restarting is required before continuing this session.[/]"
                )
            return True


async def _read_repl_input(prompt_session: PromptSession[str] | None) -> str | None:
    """Read one line of REPL input, returning None on EOF."""
    try:
        _render_status_line()
        if prompt_session is not None:
            return (await prompt_session.prompt_async("> ")).strip()
        return console.input(f"[{STYLE_PROMPT}]> [/]").strip()
    except KeyboardInterrupt:
        console.print()
        return ""
    except EOFError:
        console.print(Rule(style=STYLE_RULE))
        return None


def _is_repl_quit_command(message: str) -> bool:
    """Return True if the message is a quit/exit command."""
    return message.lower() in {"exit", "quit", "/exit", "/quit"}


async def _process_repl_turn(message: str) -> None:
    """Execute one REPL turn: call handle_message, render response, record metrics."""
    turn_start = time.perf_counter()
    try:
        response = await handle_message(message)
    except Exception as exc:  # pylint: disable=broad-except
        elapsed_ms = (time.perf_counter() - turn_start) * 1000.0
        _record_session_turn(succeeded=False, latency_ms=elapsed_ms)
        logger.error("REPL turn failed: %s: %s", type(exc).__name__, exc)
        _render_runtime_error(exc, title="Turn Error")
        console.print(Rule(style=STYLE_RULE))
        return
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


def _runtime_restart_required() -> bool:
    """Return whether a host restart is required to apply runtime changes."""
    return bool(_RUNTIME_STATE.get("restart_required", False))


def _render_restart_required_notice_if_needed() -> None:
    """Render a one-shot restart-required notice after pack lifecycle changes."""
    if not _runtime_restart_required():
        return
    reason = str(_RUNTIME_STATE.get("restart_reason") or "runtime change").strip()
    console.print(
        Panel(
            "Restart required to reload entry points, pack catalogs, and MCP manifests.\n"
            f"Reason: {reason}",
            title="Restart Required",
            border_style="gold3",
        )
    )


def _print_startup_banner() -> None:
    """Print a compact startup banner."""

    title = Text()
    title.append("Ready", style="bold steel_blue3")
    title.append("  ")
    title.append("interactive terminal host", style=STYLE_TEXT_MUTED)
    console.print(title)
    console.print()


def _status_segments(compact_level: int = _STATUS_COMPACT_FULL) -> list[tuple[str, str]]:
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
        status_dt = datetime.now()
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
        elif timezone_label == timezone_name:
            # ZoneInfo may be unavailable (no tzdata); keep label compact anyway.
            tail = timezone_name.rsplit("/", 1)[-1]
            timezone_label = "".join(part[:1] for part in tail.split("_") if part).upper() or tail

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
    logger.error("%s: %s: %s", title, type(exc).__name__, exc)
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
    elif normalized_user:
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
    timezone: str = "",
    state_snapshot_path: str,
    streaming: str = "",
    host_log_path: str = "",
) -> None:
    """Initialize Little Lisa runtime before each command entry point.

    Order matters: log-level, data-dir, and host-log-path are resolved and
    the stdlib root logger is reconfigured **before** ``setup_framework()``
    runs, so the ``registry.py`` "Loaded operation module" INFO lines land
    in ``host.log`` instead of FastMCP's stderr RichHandler.
    """
    _init_core_state(debug=debug, log_level=log_level)
    _init_paths(
        data_dir=data_dir,
        log_path=log_path,
        no_log=no_log,
        memory_db=memory_db,
    )
    _init_host_log(host_log_path=host_log_path)
    _configure_root_logging()
    _init_core_framework()
    normalized_user, context_ref, context_active = _init_session_context_and_threads(
        user_slug=user_slug,
        thread_dir=thread_dir,
        no_thread=no_thread,
    )
    _init_timezone(timezone=timezone)
    _init_state_snapshot(state_snapshot_path=state_snapshot_path)
    _init_streaming(streaming=streaming)
    _build_warm_mcp_pool()
    _log_runtime_initialized(
        normalized_user=normalized_user,
        context_ref=context_ref,
        context_active=context_active,
    )
    reload_model_manifest()


def _build_warm_mcp_pool() -> None:
    """Build (but do NOT start) the long-lived warm MCP server pool.

    Servers marked ``warm: true`` in ``mcp_servers.yaml`` are compiled as
    warm configs and collected into an :class:`MCPToolPool` here. The pool
    is **started** later, inside the host's long-lived event loop, by
    :func:`astart_warm_mcp_pool`.

    Why split build and start: ``stdio_client`` (an anyio async context
    manager) binds its cancel scope to the task/loop that entered it. If
    ``start()`` ran here inside a throwaway ``asyncio.run`` loop, the
    pool's sessions would belong to a dead loop, and ``aclose()`` in the
    REPL's loop would raise
    ``RuntimeError: Attempted to exit cancel scope in a different task
    than it was entered in``. Starting and closing in the same loop is
    required.

    Cold server configs are also compiled here and stashed on runtime
    state for the per-turn fallback pool.
    """
    warm_configs, cold_configs = _compile_mcp_configs()
    # Build the long-lived pool with ALL configs (warm + cold). The ``warm``
    # flag on each config controls eager-vs-lazy start inside pool.start():
    # warm servers connect eagerly with retry; cold servers are NOT connected
    # at start — they are lazily connected on first list_tools or call_tool.
    # Keeping cold configs in the pool is required so server-scoped dispatch
    # (_call_tool_scoped) and lazy connect (_ensure_connected) can resolve
    # ``server_id="mem0"`` against _configs.
    pool = MCPToolPool([*warm_configs, *cold_configs])
    _RUNTIME_STATE["mcp_tool_pool"] = pool
    _RUNTIME_STATE["mcp_cold_configs"] = cold_configs


def _is_empty_auth_header(value: str) -> bool:
    """True when an Authorization header is a Bearer token whose value is empty.

    The manifest expands ``"Bearer ${MEM0_API_KEY}"`` via
    :func:`_expand_env_value` at load time; an unset variable yields
    ``"Bearer "`` (trailing space, no token). That produces a 401 from the
    remote server, which inside MCP's streamable-HTTP transport cascades
    into a cross-task anyio cancel-scope RuntimeError. Detect it here so the
    host can refuse to wire the backend instead of crashing opaquely.

    Uses ``lstrip`` (not ``strip``) so the trailing space survives —
    ``"Bearer ".strip()`` becomes ``"Bearer"`` and would hide the empty token.
    """
    if not isinstance(value, str):
        return False
    # Case-insensitive "Bearer " prefix, then the token after it.
    lowered = value.lstrip().lower()
    if not lowered.startswith("bearer "):
        return False
    token = value.lstrip()[len("bearer "):].strip()
    return token == ""


def _register_memory_synthetic_tools(pool: Any) -> None:
    """Register the generic memory.* synthetic tools on a started pool.

    The three generic aliases (``memory.add`` / ``memory.search`` /
    ``memory.list``) route through the pool to whichever MCP server the
    manifest wired as the default memory backend (``lisa-memory`` by
    default, ``mem0`` when its HTTP endpoint is configured). Modes and
    operations (and the model) get the default backend's three core tools
    without knowing their concrete MCP tool names. Modes that want a
    backend's full surface (e.g. Mem0's ``update_memory`` /
    ``delete_memory`` / ``list_entities``) declare those tools directly —
    they are in the catalog because the backend is also registered as a
    tool server.

    Call this after ``pool.start()`` so the alias handlers forward to live
    server sessions. The default backend's tool names come from the
    manifest's ``memory.tool_names`` mapping.

    Ambient scope (user_id) is injected by the proxy layer
    (:class:`~kernel.mcp_launch.policy.ProxyPolicy` on the server's
    config), not by these aliases. The aliases only do generic→native
    field renaming (e.g. ``content``→``text`` for mem0). The proxy strips
    the user_id field from the schema the model sees and injects it at
    dispatch time — covering both the alias path and direct native-tool
    calls (``mem0.get_memories``) from the same chokepoint. agent_id and
    app_id were dropped from the mem0 policy because they were metadata-
    only and broke store→retrieve round-trip (add_memory injects them
    top-level while get_memories filters via filters.AND).
    """
    from kernel.tools.synthetic import mcp_alias_tool  # noqa: PLC0415

    memory_config = _RUNTIME_STATE.get("memory_config")
    default_backend_id = getattr(memory_config, "default_backend", None)
    declared = getattr(memory_config, "backends", ()) or ()
    default_spec = next((s for s in declared if s.id == default_backend_id), None)
    # Memory backends + memory tool servers are contributed by a pack (the
    # learning pack by default). Pack-contributed servers are compiled into
    # namespaced runtime ids (``<pack_slug>.<server_id>``); the synthetic
    # aliases pin dispatch to the namespaced default-backend id. When no
    # pack owns memory, fall back to the bare first-party ``lisa-memory``
    # server id (registered in code, not via a pack).
    owning_slug = _RUNTIME_STATE.get("memory_config_pack_slug")
    prefix = f"{owning_slug}." if owning_slug else ""
    _BACKEND_ID_TO_SERVER_ID = {"lisa_memory": f"{prefix}lisa-memory"}
    if default_spec is None:
        # No external default declared: lisa-memory is the first-party default.
        # Its tool names are the generic aliases themselves.
        tool_names = {"add": "add_memory", "search": "search_memory", "list": "list_memories"}
        # Pin dispatch to lisa-memory explicitly so the alias routes to it
        # even if another server in the pool also exposes the same tool names.
        # When a pack owns memory, the server is registered as
        # ``<pack_slug>.lisa-memory``; otherwise it is the bare
        # first-party ``lisa-memory``.
        target_server_id = f"{prefix}lisa-memory"
        # lisa-memory's tools take the generic shape (user_slug, memory_scope,
        # memory_tier) natively — no field renaming needed. Ambient scope
        # is not proxy-enforced for lisa-memory (it has no proxy_policy),
        # so the generic user_slug field is passed through and the model
        # controls it (lisa-memory is local SQLite, no cross-user risk).
        add_arg_map = None
        search_arg_map = None
        list_arg_map = None
    else:
        tool_names = dict(default_spec.tool_names)
        # Pin dispatch to the declared default backend's server. This
        # bypasses the pool's first-registered-wins bare-name index so the
        # default backend serves the alias even when its tool name collides
        # with another server's (e.g. mem0's add_memory vs lisa-memory's
        # add_memory). The backend id in the manifest is bare (``mem0`` or
        # ``lisa_memory``); the pool's server id is namespaced when memory
        # is pack-owned (``<pack_slug>.<server_id>``). The lisa_memory
        # backend id maps to the ``lisa-memory`` server id; any other
        # backend id is its own server id.
        lisa_memory_server_id = _BACKEND_ID_TO_SERVER_ID.get("lisa_memory", "lisa-memory")
        target_server_id = (
            lisa_memory_server_id
            if default_spec.id == "lisa_memory"
            else f"{prefix}{default_spec.id}"
        )
        # Per-backend field renaming: the generic aliases expose a stable
        # shape (content/user_slug/memory_scope/...) but each backend's MCP
        # tools may expect different argument names. Mem0's add_memory
        # requires `text` (not `content`). Ambient scope injection (user_id,
        # agent_id, app_id) is handled by the proxy layer — the alias only
        # renames fields, it does not inject ambient values.
        if default_spec.id == "mem0":
            def _mem0_add_arg_map(args: dict[str, Any]) -> dict[str, Any]:
                mapped = {
                    "text": args.get("content", ""),
                    "user_slug": args.get("user_slug", "default"),
                    "memory_scope": args.get("memory_scope", "personal"),
                    "memory_tier": args.get("memory_tier", "daily"),
                }
                return {k: v for k, v in mapped.items() if v is not None}

            def _mem0_search_or_list_arg_map(args: dict[str, Any]) -> dict[str, Any]:
                return {k: v for k, v in args.items() if k != "user_slug"}

            # Mem0 add_memory: content → text. The proxy injects user_id /
            # agent_id / app_id from the runtime; we only rename content→text
            # and pass through the rest (user_slug is kept for the alias's
            # generic schema; the proxy overrides user_id at dispatch time).
            add_arg_map = _mem0_add_arg_map
            # search_memories / get_memories: query + limit pass through.
            # The proxy injects the filters.AND with ambient scope.
            search_arg_map = _mem0_search_or_list_arg_map
            list_arg_map = _mem0_search_or_list_arg_map
        else:
            add_arg_map = None
            search_arg_map = None
            list_arg_map = None

    add_schema = {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Memory content to store."},
            "user_slug": {"type": "string", "default": "default"},
            "memory_scope": {"type": "string", "default": "personal"},
            "memory_tier": {"type": "string", "default": "daily"},
        },
        "required": ["content"],
    }
    search_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "user_slug": {"type": "string", "default": "default"},
            "limit": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    }
    list_schema = {
        "type": "object",
        "properties": {
            "user_slug": {"type": "string", "default": "default"},
            "limit": {"type": "integer", "default": 20},
        },
    }

    pool.add_synthetic_tools([
        mcp_alias_tool(
            name="memory.add",
            description="Persist a memory record via the default memory backend.",
            pool=pool,
            target_tool=tool_names["add"],
            input_schema=add_schema,
            target_server_id=target_server_id,
            arg_map=add_arg_map,
        ),
        mcp_alias_tool(
            name="memory.search",
            description="Recall memories matching a free-text query via the default backend.",
            pool=pool,
            target_tool=tool_names["search"],
            input_schema=search_schema,
            target_server_id=target_server_id,
            arg_map=search_arg_map,
        ),
        mcp_alias_tool(
            name="memory.list",
            description="List recent memory records for a user scope via the default backend.",
            pool=pool,
            target_tool=tool_names["list"],
            input_schema=list_schema,
            target_server_id=target_server_id,
            arg_map=list_arg_map,
        ),
    ])


async def astart_warm_mcp_pool() -> None:
    """Eagerly start the warm MCP pool inside the host's event loop.

    Called at the top of each async entry point (REPL, ``once``, desktop).
    A required warm server that cannot start after bounded retry raises
    :class:`MCPConnectionError`; the caller surfaces it as a startup
    error. Idempotent: a no-op when the pool is already started.
    """
    pool = _RUNTIME_STATE.get("mcp_tool_pool")
    if pool is None:
        return
    try:
        await pool.start()
    except MCPConnectionError as exc:
        _render_runtime_error(exc, title="MCP Startup Error")
        raise typer.Exit(code=2) from exc
    # Register ambient resolvers so the proxy layer can inject runtime
    # scope (user_slug, mode_id, app_id) into tool arguments at dispatch
    # time. The proxy strips these fields from the schema the model sees
    # and injects the resolved values before forwarding to the MCP server.
    pool.set_ambient_resolver("user_slug", _runtime_user_slug)
    pool.set_ambient_resolver("mode_id", _runtime_current_mode_id)
    pool.set_ambient_resolver("app_id", lambda: "little-lisa")
    # Register host-owned synthetic tools (generic memory aliases) on the
    # now-live pool. Done after start() so handlers can close over the live
    # router; the aliases dispatch in-process via the pool.
    _register_memory_synthetic_tools(pool)
    _append_log_event(
        level="info",
        event="mcp.warm_pool.started",
        payload={
            "warm_server_ids": pool.warm_server_ids(),
            "warm_failed_ids": pool.warm_failed_ids(),
            "cold_server_ids": [
                c.server_id for c in (_RUNTIME_STATE.get("mcp_cold_configs") or [])
            ],
        },
    )


async def aclose_warm_mcp_pool() -> None:
    """Asynchronously close the warm MCP pool at process exit.

    Called from the CLI REPL finally block and the desktop shutdown hook.
    Safe to call when no pool exists or the pool is already closed.
    """
    pool = _RUNTIME_STATE.pop("mcp_tool_pool", None)
    _RUNTIME_STATE.pop("mcp_cold_configs", None)
    if pool is None:
        return
    try:
        await pool.aclose()
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("warm MCP pool close failed: %s: %s", type(exc).__name__, exc)


def _init_core_state(*, debug: bool, log_level: str) -> None:
    """Resolve debug flag and log level before any framework code runs.

    Runs first so ``_configure_root_logging`` can install the FileHandler at
    the resolved level before ``setup_framework()`` emits INFO lines.
    """
    _load_local_env_file()
    _RUNTIME_STATE["debug"] = debug
    _RUNTIME_STATE["log_level"] = _resolve_log_level(log_level, debug_enabled=debug)


def _init_core_framework() -> None:
    """Register content/operation roots and reset provider clients.

    Runs **after** ``_configure_root_logging`` so the ``registry.py`` INFO
    lines it triggers land in ``host.log``, not FastMCP's stderr RichHandler.
    """
    setup_framework()
    configure_runtime()


def _init_core(*, debug: bool, log_level: str) -> None:
    """Set up framework, configure runtime, and resolve debug/log-level.

    Legacy single-call seam; retained for any external caller. New code
    should use ``_init_core_state`` + ``_init_core_framework`` so logging
    can be wedged between them.
    """
    _init_core_state(debug=debug, log_level=log_level)
    _init_core_framework()


def _init_paths(*, data_dir: str, log_path: str, no_log: bool, memory_db: str) -> None:
    """Resolve and set data-dir, log-path, memory-db, and thread-dir paths."""
    data_from_cli = Path(data_dir).expanduser() if data_dir.strip() else None
    data_from_env = (
        Path(os.getenv(_LITTLE_LISA_DATA_DIR_ENV, "")).expanduser()
        if os.getenv(_LITTLE_LISA_DATA_DIR_ENV, "").strip()
        else None
    )
    _RUNTIME_STATE["data_dir"] = data_from_cli or data_from_env or _DEFAULT_DATA_DIR

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


def _init_host_log(*, host_log_path: str) -> None:
    """Resolve and set the stdlib host-log path (distinct from the JSONL turn log).

    ``--host-log`` wins; otherwise the host log defaults to
    ``<data_dir>/host.log``. Must run after ``_init_paths`` has set
    ``data_dir``.
    """
    if host_log_path.strip():
        _RUNTIME_STATE["host_log_path"] = Path(host_log_path).expanduser()
    else:
        _RUNTIME_STATE["host_log_path"] = _runtime_data_dir() / "host.log"


def _init_session_context_and_threads(
    *,
    user_slug: str,
    thread_dir: str,
    no_thread: bool,
) -> tuple[str, str, bool]:
    """Resolve session context and thread strategy, set runtime state."""
    thread_from_cli = Path(thread_dir).expanduser() if thread_dir.strip() else None
    thread_from_env = (
        Path(os.getenv(_LITTLE_LISA_THREAD_DIR_ENV, "")).expanduser()
        if os.getenv(_LITTLE_LISA_THREAD_DIR_ENV, "").strip()
        else None
    )

    slug = str(user_slug).strip()
    normalized_user, context_ref, context_active = _resolve_session_context(slug or "")

    resolved_thread_dir, thread_path = _resolve_runtime_thread_strategy(
        no_thread=no_thread,
        context_active=context_active,
        context_ref=context_ref,
        normalized_user=normalized_user,
        thread_from_cli=thread_from_cli,
        thread_from_env=thread_from_env,
        data_dir=_runtime_data_dir(),
    )
    _RUNTIME_STATE["thread_dir"] = resolved_thread_dir
    _RUNTIME_STATE["thread_path"] = thread_path
    # Back-compat test seam: legacy key retained alongside thread_path.
    _RUNTIME_STATE["thread_log_path"] = thread_path
    _RUNTIME_STATE["user_slug"] = normalized_user
    _RUNTIME_STATE["session_context_ref"] = context_ref
    _RUNTIME_STATE["session_context_active"] = context_active
    return normalized_user, context_ref, context_active


def _init_timezone(*, timezone: str) -> None:
    """Resolve and set timezone and session counters."""
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


def _init_state_snapshot(state_snapshot_path: str) -> None:
    """Load and set the state snapshot from the given path."""
    _RUNTIME_STATE["state_snapshot_path"] = (
        Path(state_snapshot_path).expanduser() if state_snapshot_path.strip() else None
    )
    _RUNTIME_STATE["state_snapshot"] = _load_state_snapshot(state_snapshot_path)


def _init_streaming(streaming: str) -> None:
    """Resolve and set streaming mode and char delay."""
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


def _log_runtime_initialized(
    *,
    normalized_user: str,
    context_ref: str,
    context_active: bool,
) -> None:
    """Log the runtime.initialized event."""
    _append_log_event(
        level="info",
        event="runtime.initialized",
        payload={
            "session_context": normalized_user or "none",
            "session_context_ref": context_ref,
            "session_context_active": context_active,
            "timezone": _runtime_timezone(),
            "timezone_source": _runtime_timezone_source(),
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
    host_log: HostLogOption = "",
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
        timezone=timezone,
        state_snapshot=state_snapshot,
        streaming=streaming,
        host_log=host_log,
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
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "none",
    host_log: HostLogOption = "",
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
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
        host_log_path=host_log,
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
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "none",
    host_log: HostLogOption = "",
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
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
        host_log_path=host_log,
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


def _verify_check_memory_backend() -> tuple[str, str]:
    """Check the active memory backend's required env vars are set.

    The default backend from ``mcp_servers.yaml`` is what the memory seam
    actually talks to. An HTTP backend with an empty ``${VAR}`` in its
    Authorization header (the common MEM0_API_KEY-not-set case) is a hard
    failure: it would 401 at runtime. Surface it here so ``verify`` catches
    it before the user runs into the anyio cascade.
    """
    memory_config = _RUNTIME_STATE.get("memory_config")
    default_backend_id = getattr(memory_config, "default_backend", None)
    declared = getattr(memory_config, "backends", ()) or ()
    if not default_backend_id or not declared:
        return _VERIFY_PASS, "Memory backend: lisa-memory (default)"

    default_spec = next((s for s in declared if s.id == default_backend_id), None)
    if default_spec is None:
        return _VERIFY_WARN, (
            f"default_backend={default_backend_id!r} not declared; will fall back"
        )

    if default_spec.kind == "http":
        for value in default_spec.headers.values():
            if _is_empty_auth_header(value):
                return _VERIFY_FAIL, (
                    f"Memory backend {default_backend_id!r}: Authorization header "
                    "has an empty ${VAR} — set the env var (e.g. MEM0_API_KEY) in .env"
                )
        return _VERIFY_PASS, f"Memory backend: {default_backend_id!r} (http)"
    if default_spec.kind == "stdio":
        return _VERIFY_PASS, f"Memory backend: {default_backend_id!r} (stdio)"
    return _VERIFY_WARN, (
        f"Memory backend: {default_backend_id!r} "
        f"(unknown kind {default_spec.kind!r})"
    )


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
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "none",
    host_log: HostLogOption = "",
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
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
        host_log_path=host_log,
    )
    checks: list[tuple[str, str, str]] = [
        ("Python", *_verify_check_python()),
        (".env", *_verify_check_env()),
        ("Session context", *_verify_check_session_context()),
        ("Model manifest", *_verify_check_manifest()),
        ("Pack catalog", *_verify_check_pack_catalog()),
        ("Provider keys", *_verify_check_provider_keys()),
        ("Ollama", *_verify_check_ollama()),
        ("Memory backend", *_verify_check_memory_backend()),
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
def _setup_command() -> None:
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
    _SETUP_PROVIDER_STEPS[provider]()
    updates = _SETUP_PROVIDER_UPDATES.get(provider, {})

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


def _setup_step_ollama() -> dict[str, str]:
    """Ollama setup step — no API key needed, just validate the local server."""
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
    return {}


def _setup_step_openai() -> dict[str, str]:
    """OpenAI setup step — collect and validate API key."""
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
    return {"LITTLE_LISA_OPENAI_API_KEY": api_key}


def _setup_step_azure() -> dict[str, str]:
    """Azure OpenAI setup step — collect endpoint and API key, validate."""
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
    return {
        "LITTLE_LISA_AZURE_OPENAI_ENDPOINT": endpoint,
        "LITTLE_LISA_AZURE_OPENAI_API_KEY": api_key,
    }


def _setup_step_anthropic() -> dict[str, str]:
    """Anthropic setup step — collect and validate API key."""
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
    return {"LITTLE_LISA_ANTHROPIC_API_KEY": api_key}


def _setup_step_google() -> dict[str, str]:
    """Google Gemini setup step — collect and validate API key."""
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
    return {"LITTLE_LISA_GOOGLE_API_KEY": api_key}


_SETUP_PROVIDER_STEPS: dict[str, Callable[[], dict[str, str]]] = {
    "ollama": _setup_step_ollama,
    "openai": _setup_step_openai,
    "azure_openai": _setup_step_azure,
    "anthropic": _setup_step_anthropic,
    "google": _setup_step_google,
}

_SETUP_PROVIDER_UPDATES: dict[str, dict[str, str]] = {}  # populated by steps at call time


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
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "none",
    host_log: HostLogOption = "",
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
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
        host_log_path=host_log,
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
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "none",
    host_log: HostLogOption = "",
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
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
        host_log_path=host_log,
    )
    console.print(format_models())


@app.command("packs", help="Manage runtime pack installs (list/install/uninstall).")
def _packs_command(
    action: str = typer.Argument(..., help="Action: list, install, uninstall."),
    target: str = typer.Argument(
        "",
        help="Install spec or package name (required for install/uninstall).",
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
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "none",
    host_log: HostLogOption = "",
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
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
        host_log_path=host_log,
    )

    normalized = action.strip().lower()
    if normalized == "list":
        console.print(_render_pack_list())
        return
    if normalized == "install":
        ok, details = asyncio.run(_run_pack_install(target))
        if not ok:
            console.print(f"[red]{details}[/red]")
            raise typer.Exit(code=1)
        console.print(details)
        raise typer.Exit(code=0)
    if normalized == "uninstall":
        ok, details = asyncio.run(_run_pack_uninstall(target))
        if not ok:
            console.print(f"[red]{details}[/red]")
            raise typer.Exit(code=1)
        console.print(details)
        raise typer.Exit(code=0)

    console.print("[red]Unknown action. Use: list, install, uninstall.[/red]")
    raise typer.Exit(code=2)


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
    timezone: TimezoneOption = "",
    state_snapshot: StateSnapshotOption = "",
    streaming: StreamingOption = "none",
    host_log: HostLogOption = "",
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
        timezone=timezone,
        state_snapshot_path=state_snapshot,
        streaming=streaming,
        host_log_path=host_log,
    )
    asyncio.run(_run_once_with_teardown(message))


async def _run_once_with_teardown(message: str) -> None:
    """Run one turn, then close the warm MCP pool."""
    await astart_warm_mcp_pool()
    try:
        await _run_once(message)
    finally:
        await aclose_warm_mcp_pool()


@app.command("bottom", help="Run one bottom-half tick: collect signals, process, write state.")
def _bottom_command(
    data_dir: DataDirOption = "",
    log_level: LogLevelOption = "",
    host_log: HostLogOption = "",
) -> None:
    """One-shot bottom-half process. Triggered by systemd timer or cron."""
    resolved_data_dir = Path(data_dir).expanduser() if data_dir else _DEFAULT_DATA_DIR
    if log_level or host_log:
        _initialize_runtime_or_exit(
            debug=False,
            log_level=log_level,
            log_path="",
            no_log=False,
            data_dir=data_dir,
            memory_db="",
            thread_dir="",
            no_thread=True,
            user_slug="",
            timezone="",
            state_snapshot_path="",
            streaming="none",
            host_log_path=host_log,
        )
    try:
        from . import bottom  # noqa: PLC0415
    except ImportError:  # pragma: no cover - direct script execution fallback
        import bottom  # type: ignore[no-redef]

    state = asyncio.run(bottom.run_tick(data_dir=resolved_data_dir))
    console.print(
        f"[green]Bottom-half tick complete.[/green] "
        f"{len(state.signal_snapshots)} signal(s), "
        f"{len(state.active_alerts)} alert(s). "
        f"State written to {resolved_data_dir / 'state.json'}"
    )


def main() -> None:
    """CLI entry point for Little Lisa."""

    app(args=_normalize_legacy_cli_args(sys.argv[1:]))


if __name__ == "__main__":
    main()

