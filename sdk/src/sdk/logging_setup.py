"""Shared logging infrastructure for the Lisa Harness.

This module is the single source of truth for stdlib ``logging``
configuration across all hosts (CLI, web) and the first-party MCP
servers. It replaces the per-host ``_configure_root_logging`` duplicates
and the bespoke JSONL writer in ``host/turn_logging.py``.

Public surface
--------------
- :class:`JSONFormatter` — one JSON line per log record.
- :class:`RateLimitFilter` — collapses repeated identical records.
- :func:`configure_logging` — root + handlers + per-module debug levels.
- :func:`apply_log_levels` — re-apply levels without touching handlers.
- :func:`emit_structured_event` — emit a namespaced event through stdlib.
- :func:`install_log_excepthook` — route uncaught exceptions to ``CRITICAL``.

Conventions enforced here (see ``docs/LOGGING.md``)
---------------------------------------------------
- Module loggers are named ``_log`` and obtained via ``logging.getLogger(__name__)``.
- Messages use lazy ``%s`` positional args — never f-strings.
- Phrasing: ``"<noun>: <lowercase clause, no trailing period>"``.
- Every logged ``except`` block includes ``exc_info=True`` or calls
  ``_log.exception(...)``.
- Structured fields ride on ``extra={...}``; the ``event`` key is the
  namespaced junction name (e.g. ``"approval.granted"``, ``"fsm.transition"``).
- Debug verbosity is scoped per module via ``LISA_LOG_DEBUG`` — the global
  default stays ``INFO``.

Dependency direction
--------------------
``sdk.logging_setup`` ← ``hosts.*``, ``mcp_server.*``. This module
imports only the stdlib so it can be reused by every entry point without
pulling the kernel into subprocesses that do not need it. It lives in the
SDK layer (rather than the kernel) because logging *infrastructure* —
file rotation, stderr defaults, env-var parsing, the uncaught-exception
hook — is runtime/host concern, not a kernel contract. The kernel
contract forbids importing from higher layers; the SDK contract permits
importing from the kernel, and ``mcp_server``/``hosts`` already depend on
``lisa-harness-sdk``.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "LISA_AUDIT_GUARD_ENV",
    "LISA_LOG_DEBUG_ENV",
    "LISA_LOG_STDERR_ENV",
    "JSONFormatter",
    "RateLimitFilter",
    "apply_log_levels",
    "configure_logging",
    "emit_structured_event",
    "install_log_excepthook",
]

# ── Environment-variable name constants ──────────────────────────────

LISA_LOG_DEBUG_ENV = "LISA_LOG_DEBUG"
LISA_LOG_STDERR_ENV = "LISA_LOG_STDERR"
LISA_AUDIT_GUARD_ENV = "LISA_AUDIT_GUARD"

# ── RateLimitFilter defaults (code-level constants; no env knobs in v1) ──

_DEDUP_WINDOW_SECONDS = 60.0
_DEDUP_SUPPRESS_AFTER = 5

# ── File-handler rotation defaults ───────────────────────────────────

_ROTATE_MAX_BYTES = 5 * 1024 * 1024  # 5 MiB
_ROTATE_BACKUP_COUNT = 3

_log = logging.getLogger(__name__)

# Keys injected by the stdlib ``LogRecord`` that must not collide with
# user-supplied ``extra`` keys. ``JSONFormatter`` strips these from the
# emitted object and renders them in dedicated fields.
_RESERVED_RECORD_KEYS = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
        "message",
        "event",
    }
)


class JSONFormatter(logging.Formatter):
    """Emit one JSON line per log record.

    Fields: ``ts`` (ISO 8601 UTC), ``level``, ``name``, ``msg``, ``event``
    (when supplied via ``extra={"event": ...}``), all other ``extra`` keys,
    and ``exc`` (serialised traceback) when ``exc_info`` is present.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname.lower(),
            "name": record.name,
            "msg": record.getMessage(),
        }

        event = getattr(record, "event", None)
        if event:
            payload["event"] = event

        # Surface caller-supplied ``extra`` keys, skipping stdlib internals.
        for key, value in record.__dict__.items():
            if key in _RESERVED_RECORD_KEYS or key.startswith("_"):
                continue
            payload[key] = _json_safe(value)

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = record.stack_info

        return json.dumps(payload, ensure_ascii=False, default=_json_default)


def _json_safe(value: Any) -> Any:
    """Coerce values that ``json.dumps`` cannot natively serialize."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseException):
        return f"{type(value).__name__}: {value}"
    return str(value)


def _json_default(value: Any) -> Any:
    """Fallback ``default=`` hook for ``json.dumps`` in :class:`JSONFormatter`."""
    return _json_safe(value)


# ── Rate-limiting / dedup filter ─────────────────────────────────────


class RateLimitFilter(logging.Filter):
    """Suppress repeated identical records within a sliding window.

    Two records are "identical" when their ``(name, levelname, message)``
    tuple matches. After ``suppress_after`` duplicates within
    ``window_seconds``, subsequent duplicates are dropped and a single
    summary line is emitted at the end of the window.

    This is the cure for the warm-pool chatter in ``kernel/tools/mcp_client.py``
    and the per-turn tool-schema lines in ``turn_runner.py``: enabling
    ``DEBUG`` no longer floods the log with hundreds of repeats.
    """

    def __init__(
        self,
        *,
        window_seconds: float = _DEDUP_WINDOW_SECONDS,
        suppress_after: int = _DEDUP_SUPPRESS_AFTER,
    ) -> None:
        super().__init__()
        self._window_seconds = window_seconds
        self._suppress_after = suppress_after
        self._lock = threading.Lock()
        # key -> (first_seen_ts, suppressed_count)
        self._seen: dict[tuple[str, str, str], tuple[float, int]] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        key = (record.name, record.levelname, record.getMessage())
        now = time.monotonic()
        with self._lock:
            entry = self._seen.get(key)
            if entry is None:
                self._seen[key] = (now, 0)
                return True
            first_seen, count = entry
            if now - first_seen >= self._window_seconds:
                # Window elapsed — flush a summary if anything was suppressed.
                if count > 0:
                    self._emit_summary(record, count)
                self._seen[key] = (now, 0)
                return True
            # Inside the window.
            if count >= self._suppress_after:
                self._seen[key] = (first_seen, count + 1)
                return False
            self._seen[key] = (first_seen, count + 1)
            return True

    def _emit_summary(self, template: logging.LogRecord, count: int) -> None:
        """Emit one summary record on the same logger/level as *template*."""
        summary = logging.LogRecord(
            name=template.name,
            level=template.levelno,
            pathname=template.pathname,
            lineno=template.lineno,
            msg="rate-limited: suppressed %d duplicate records in %.0fs",
            args=(count, self._window_seconds),
            exc_info=None,
        )
        # Hand the summary straight to the logger's handlers; bypass our filter
        # so the summary itself is not deduped.
        logger = logging.getLogger(template.name)
        for handler in logger.handlers:
            if handler.level <= summary.levelno:
                handler.emit(summary)


# ── Configuration ────────────────────────────────────────────────────


def _parse_debug_modules(raw: str) -> list[str]:
    """Parse a comma-separated ``LISA_LOG_DEBUG`` value into logger names."""
    modules: list[str] = []
    for token in raw.split(","):
        name = token.strip()
        if name:
            modules.append(name)
    return modules


def _resolve_debug_modules_from_env() -> list[str]:
    """Read ``LISA_LOG_DEBUG`` from the environment."""
    return _parse_debug_modules(os.getenv(LISA_LOG_DEBUG_ENV, ""))


def _stderr_default_for(host_kind: str | None) -> bool:
    """Stderr handler default: on for the web host, off elsewhere.

    The web host does not own stderr with Rich, so mirroring logs there is
    safe and useful for ``docker logs``. The CLI host reserves stderr for
    the Rich UI and stays opt-in.
    """
    return host_kind == "web"


def _resolve_stderr_enabled(host_kind: str | None) -> bool:
    """Resolve stderr-handler enablement from env + host-kind default."""
    raw = os.getenv(LISA_LOG_STDERR_ENV, "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return _stderr_default_for(host_kind)


def configure_logging(
    *,
    log_path: Path,
    level: str,
    debug_modules: list[str] | None = None,
    host_kind: str | None = None,
    formatter: logging.Formatter | None = None,
) -> logging.Logger:
    """Configure the root logger with a rotating file handler and optional stderr.

    Parameters
    ----------
    log_path:
        Destination for the rotating file handler. Parent directories are
        created lazily.
    level:
        Global log level (``"debug"``, ``"info"``, ``"warning"``, ``"error"``).
    debug_modules:
        Logger names to force to ``DEBUG`` regardless of the global level
        (parsed from ``LISA_LOG_DEBUG`` by the caller or
        :func:`_resolve_debug_modules_from_env`).
    host_kind:
        ``"cli"``, ``"web"``, or ``None`` (MCP servers). Controls the
        stderr-handler default; ``LISA_LOG_STDERR`` overrides either way.
    formatter:
        Override the default :class:`JSONFormatter` (used by tests).
    """
    root = logging.getLogger()
    root.setLevel(_level_value(level))
    for handler in list(root.handlers):
        root.removeHandler(handler)

    fmt = formatter or JSONFormatter()
    rate_filter = RateLimitFilter()

    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=_ROTATE_MAX_BYTES,
        backupCount=_ROTATE_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(root.level)
    file_handler.setFormatter(fmt)
    file_handler.addFilter(rate_filter)
    root.addHandler(file_handler)

    if _resolve_stderr_enabled(host_kind):
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(root.level)
        stderr_handler.setFormatter(fmt)
        stderr_handler.addFilter(rate_filter)
        root.addHandler(stderr_handler)

    apply_log_levels(root, level=level, debug_modules=debug_modules or [])
    install_log_excepthook()
    return root


def apply_log_levels(
    root: logging.Logger,
    *,
    level: str,
    debug_modules: list[str],
) -> None:
    """Apply the global level and per-module DEBUG overrides.

    Idempotent: safe to call repeatedly to react to env changes. Resets
    previously-overridden loggers back to the global level before applying
    the new set, so removing a name from ``LISA_LOG_DEBUG`` takes effect.
    """
    root.setLevel(_level_value(level))
    for handler in root.handlers:
        handler.setLevel(_level_value(level))

    # Reset any loggers we previously forced to DEBUG.
    for name in list(_DEBUG_OVERRIDDEN):
        logging.getLogger(name).setLevel(root.level)
    _DEBUG_OVERRIDDEN.clear()

    for name in debug_modules:
        target = logging.getLogger(name)
        target.setLevel(logging.DEBUG)
        _DEBUG_OVERRIDDEN.add(name)


_DEBUG_OVERRIDDEN: set[str] = set()


def _level_value(level: str) -> int:
    """Resolve a level name (case-insensitive) to its stdlib numeric value."""
    normalized = level.strip().lower()
    if normalized == "warn":
        normalized = "warning"
    value = logging.getLevelName(normalized.upper())
    if isinstance(value, int):
        return value
    return logging.INFO


# ── Structured event emission ────────────────────────────────────────


def emit_structured_event(
    logger: logging.Logger,
    *,
    level: int,
    event: str,
    message: str,
    **fields: Any,
) -> None:
    """Emit a namespaced structured event through *logger*.

    Wraps ``logger.log(level, message, extra={...})`` so callers do not
    have to repeat the ``extra={"event": ...}`` boilerplate. The ``event``
    key is always set; remaining ``fields`` become top-level JSON keys via
    :class:`JSONFormatter`.

    Example::

        emit_structured_event(
            _log,
            level=logging.INFO,
            event="approval.granted",
            message="approve: granted request_id=%s",
            request_id=request_id,
        )
    """
    extra = {"event": event, **fields}
    logger.log(level, message, extra=extra)


# ── Uncaught-exception hook ──────────────────────────────────────────


def install_log_excepthook() -> None:
    """Route uncaught exceptions through ``logging.critical``.

    Safe to call repeatedly — the previous hook is preserved as a fallback.
    """
    previous = sys.excepthook

    def _hook(exc_type: type[BaseException], exc: BaseException, tb: Any) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            # Let the default hook handle Ctrl+C so the REPL stays responsive.
            previous(exc_type, exc, tb)
            return
        logging.getLogger("lisa.excepthook").critical(
            "excepthook: uncaught %s: %s",
            exc_type.__name__,
            exc,
            exc_info=(exc_type, exc, tb),
        )

    sys.excepthook = _hook
