"""Host-side wiring for spill-on-arrival (the cache-aware default).

The spill mechanics are kernel-owned (``kernel.turn_phases._spill``); the
kernel invokes the ``ToolHooks.persist_tool_result`` closure with a plain
record and this module binds that closure to the host's thread store — the
same store and record shape the result projector persists to, so a
preview's ``r<sha256>`` ref dereferences via ``thread_read_content`` and
via the projector's loader alike.

Cache policy note: spilling happens at delivery, BEFORE a large result
ever enters the provider's cached prefix, so normal operation is pure
append and the catastrophic mid-turn rewrite (the turn-1116 cache bust)
never happens. Eviction is one-way; recovery is a dereference.
Under ``--no-thread`` the store is the session's cache-backed result-object
path (``HostPaths.ephemeral_results_path``); no thread artifact is created.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from runtime.thread import persist_thread_tool_result


def build_persist_tool_result_hook(
    thread_path: Path | None,
    ephemeral_results_path: Path | None = None,
) -> Callable[[dict[str, Any]], str] | None:
    """Return the ``persist_tool_result`` closure for ``thread_path``.

    ``ephemeral_results_path`` is the ``--no-thread`` result store: it takes
    over when the durable ``thread_path`` is ``None`` so large tool results
    stay spillable with the same record shape.
    ``None`` when the host has no thread path: spill-on-arrival stays off
    and only the governor's epoch (which drives the host projector) can
    evict large results.
    """
    base = thread_path if thread_path is not None else ephemeral_results_path
    if base is None:
        return None
    return lambda record: persist_thread_tool_result(base, record)
