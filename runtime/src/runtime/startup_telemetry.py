"""Monotonic boot-stage instrumentation (Slice C of ``docs/SPEC/PROMPT_STARTUP.md``).

Every framework boot gets one ``boot_invocation_id`` (a uuid4 prefix).
Stages emit structured begin/end (or error) lines on the ``lisa.startup``
logger with monotonic-clock durations and durations/counts only — never
prompt text, credentials, raw environment values, or private content
(spec §5). ``warn_if_repeat_setup`` is the evidence generator behind the
"apparent repeated bootstrap/setup calls" investigation: it records each
distinct ``setup_framework`` invocation and warns on the second one in a
process, so cProfile call counts (which also count coroutine resumes) can
be cross-checked against real invocations instead of trusted.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger("lisa.startup")

__all__ = [
    "boot_invocation_id",
    "log_boot_complete",
    "seconds_since_boot",
    "stage",
    "warn_if_repeat_setup",
]

_BOOT_INVOCATION_ID: str | None = None
_PREVIOUS_BOOT_INVOCATION_ID: str | None = None
_BOOT_SETUP_COUNT = 0

#: Monotonic process reference for "seconds since boot" metrics. Captured
#: at first import of this module — the earliest common point on every
#: instrumented boot path.
BOOT_T0 = time.perf_counter()


def boot_invocation_id() -> str:
    """Return the process-wide boot invocation id, creating it on first call."""
    global _BOOT_INVOCATION_ID  # noqa: PLW0603
    if _BOOT_INVOCATION_ID is None:
        _BOOT_INVOCATION_ID = uuid.uuid4().hex[:12]
    return _BOOT_INVOCATION_ID


def seconds_since_boot() -> float:
    """Return monotonic seconds since this module was first imported."""
    return time.perf_counter() - BOOT_T0


def log_boot_complete() -> None:
    """Emit the boot_complete boundary marker on the uniform payload stream."""
    logger.info(
        "startup %s",
        json.dumps(
            {
                "boot_id": boot_invocation_id(),
                "event": "boot_complete",
                "t_seconds": round(seconds_since_boot(), 6),
                "pid": os.getpid(),
            },
            sort_keys=True,
        ),
    )


def _log_stage(
    invocation_id: str,
    name: str,
    event: str,
    seconds: float | None = None,
) -> None:
    """Emit one structured startup-stage line (durations/counts only)."""
    record: dict[str, Any] = {
        "boot_id": invocation_id,
        "stage": name,
        "event": event,
        "pid": os.getpid(),
    }
    if seconds is not None:
        record["seconds"] = round(seconds, 6)
    logger.info("startup %s", json.dumps(record, sort_keys=True))


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Log begin/end (+ monotonic duration) around one boot stage.

    The payload carries the boot invocation id, stage name, event kind,
    pid and duration — nothing else. A failing stage logs ``error`` with
    the elapsed duration and re-raises.
    """
    invocation_id = boot_invocation_id()
    _log_stage(invocation_id, name, "begin")
    started = time.perf_counter()
    try:
        yield
    except BaseException:
        _log_stage(invocation_id, name, "error", time.perf_counter() - started)
        raise
    _log_stage(invocation_id, name, "end", time.perf_counter() - started)


def warn_if_repeat_setup() -> None:
    """Record one ``setup_framework`` invocation; warn when it repeats.

    The spec's repeated-bootstrap work must distinguish actual distinct
    invocations from cProfile artifacts (coroutine resumes inflate call
    counts). Each call increments the per-process invocation count and a
    second call logs a WARNING carrying both boot ids and the count — the
    trace-side hook for deciding whether a rebuild is required reload
    behavior or a genuine duplicate.
    """
    global _BOOT_SETUP_COUNT, _PREVIOUS_BOOT_INVOCATION_ID  # noqa: PLW0603
    _BOOT_SETUP_COUNT += 1
    invocation_id = boot_invocation_id()
    if _BOOT_SETUP_COUNT > 1:
        logger.warning(
            "startup repeat_setup_detected %s",
            json.dumps(
                {
                    "boot_id": invocation_id,
                    "previous_boot_id": _PREVIOUS_BOOT_INVOCATION_ID,
                    "setup_count": _BOOT_SETUP_COUNT,
                    "pid": os.getpid(),
                },
                sort_keys=True,
            ),
        )
    _PREVIOUS_BOOT_INVOCATION_ID = invocation_id
