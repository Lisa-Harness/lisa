"""Lisa Thread SDK — lossless, segmented conversation-thread store.

Public API (import from :mod:`sdk.thread`, not the ``_backend`` module):

- :class:`ThreadStore` — owns the three-tier layout (active segment,
  sealed segments, sparse index) and the zstd compression sidecars.
  Exposes read-only operations: ``load_recent``, ``load_range``,
  ``build_recent_excerpt``, ``build_manifest``. The write path stays
  in the host because it owns the per-turn response and effect-receipt
  persistence.

The ``lisa-thread`` MCP server (``mcp_server/thread/server.py``) wraps
this; hosts or tools that want in-process thread access can use the
store directly without spawning the MCP server.

This module owns the pure read path. The host's ``runtime.thread`` module
keeps the write path and a thin re-export of the read functions for
backward compatibility.
"""

from __future__ import annotations

from sdk.thread._backend import SegmentRecord, ThreadStore
from sdk.thread._readout import (
    DEFAULT_READOUT_BYTES,
    MAX_READOUT_BYTES,
    PageCache,
    ThreadReader,
)
from sdk.thread._results import load_result_object, project_tool_messages, result_object_path
from sdk.thread._taskview import DEFAULT_TASK_VIEW_BYTES, build_task_view
from sdk.thread.redaction import (
    DEFAULT_REDACTION_PATTERNS,
    ThreadRedactionPolicy,
    resolve_redaction_policy,
)

__all__ = [
    "DEFAULT_READOUT_BYTES",
    "DEFAULT_REDACTION_PATTERNS",
    "DEFAULT_TASK_VIEW_BYTES",
    "MAX_READOUT_BYTES",
    "PageCache",
    "SegmentRecord",
    "ThreadReader",
    "ThreadRedactionPolicy",
    "ThreadStore",
    "build_task_view",
    "load_result_object",
    "project_tool_messages",
    "resolve_redaction_policy",
    "result_object_path",
]
