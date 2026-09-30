"""Lisa Thread MCP server — thin facade over :class:`sdk.thread.ThreadStore`.

This is the read-only thread-access server. It wraps the SDK's
``ThreadStore`` (segmented, zstd-sealed, lossless conversation log) and
exposes bounded views so the model can pull older entries on demand:

- ``thread_read_range`` — read entries by global turn index range.
- ``thread_read_recent`` — read the most recent N entries (compact form).
- ``thread_search`` — find prior turns by lexical relevance.
- ``thread_read_content`` — expand a referenced value through bounded pages.

The write path (append, seal, dictionary training) stays in
``runtime.thread``; this server is read-only.

The server resolves the thread base path from ``--thread-dir`` and
``--user-slug`` CLI args (the host's ``runtime.paths.thread_path`` property
produces ``thread_dir / user_slug``). No env-var path resolution.

**Redaction**: the store is built with the ``ThreadRedactionPolicy``
resolved from ``LISA_THREAD_REDACTION_ENABLED`` /
``LISA_THREAD_REDACTION_PATTERNS`` (the same env contract
``runtime.config`` uses for the host's in-process reads). The host
propagates both variables to this subprocess in ``runtime.mcp_pool``;
unset/pattern-less means framework defaults (``DEFAULT_REDACTION_PATTERNS``),
``LISA_THREAD_REDACTION_PATTERNS=none`` or ``LISA_THREAD_REDACTION_ENABLED=0``
disables. Without this, the model could pull its own prior harness-authored
budget/cap prose back into context verbatim via ``thread_read_range`` /
``thread_read_recent`` / ``thread_search`` and re-fire the claim (the
2026-08-30 remote-box budget-claim loop).

Tools are read-only and carry ``read_only_hint=True`` so the receipt
contract treats them as reads, not writes.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.context import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from sdk.layout import DATA_DIR
from sdk.thread import (
    DEFAULT_READOUT_BYTES,
    DEFAULT_REDACTION_PATTERNS,
    ThreadReader,
    ThreadRedactionPolicy,
    ThreadStore,
)

__all__ = ["ThreadDeps", "init_thread", "main", "mcp", "set_deps_for_test"]

logger = logging.getLogger(__name__)

_REDACTION_ENABLED_ENV = "LISA_THREAD_REDACTION_ENABLED"
_REDACTION_PATTERNS_ENV = "LISA_THREAD_REDACTION_PATTERNS"


def _resolve_redaction_policy() -> ThreadRedactionPolicy | None:
    """Resolve the redaction policy from the environment contract.

    The store's own redaction only covers ``build_recent_excerpt`` /
    ``build_manifest`` / ``search`` — ``load_range`` / ``load_recent``
    return RAW entries by design (the lossless read contract). The server
    therefore holds the policy on ``ThreadDeps`` and applies it itself
    when projecting entries for the model. Inheritance: unset patterns →
    framework defaults; non-empty ``LISA_THREAD_REDACTION_PATTERNS`` →
    those patterns only (hard cutover, no merge — same contract as the
    host config); ``none`` or ``LISA_THREAD_REDACTION_ENABLED=0`` → no
    redaction (``None``).
    """
    enabled = os.getenv(_REDACTION_ENABLED_ENV, "").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return None
    raw = os.getenv(_REDACTION_PATTERNS_ENV, "").strip()
    if raw.lower() == "none":
        return None
    if raw:
        patterns = tuple(p for p in raw.splitlines() if p.strip())
    else:
        patterns = DEFAULT_REDACTION_PATTERNS
    if not patterns:
        return None
    return ThreadRedactionPolicy.from_patterns(patterns)


def _build_thread_store(base: Path) -> tuple[ThreadStore, ThreadRedactionPolicy | None]:
    """Build the ``(store, redaction_policy)`` pair from the env contract.

    Single constructor for every store this server builds (lifespan, CLI
    init, and the ``init_thread`` test seam) so all read paths apply the
    same policy — the store for its excerpt/manifest/search builders, the
    returned policy for the server-side projection of raw ``load_range``
    / ``load_recent`` results.
    """
    policy = _resolve_redaction_policy()
    if policy is None:
        return ThreadStore(base), None
    return ThreadStore(base, redaction=policy), policy


# ── Lifespan deps ───────────────────────────────────────────────────────────


@dataclass
class ThreadDeps:
    """Server-scoped dependencies owned by the lifespan.

    Constructed from the resolved ``--thread-dir`` / ``--user-slug`` CLI
    args. ``redaction`` is applied to every tool projection
    (``_compact_entry`` / search results); ``None`` disables redaction.
    No teardown is needed.
    """

    store: ThreadStore
    redaction: ThreadRedactionPolicy | None = None


# Test seam: populated by the lifespan in production, or by
# ``set_deps_for_test`` in tests (direct ``call_tool`` does not run the
# lifespan). This is an explicit seam, not a hidden production singleton.
_deps: ThreadDeps | None = None


def set_deps_for_test(deps: ThreadDeps) -> None:
    """Test-only seam: inject ``ThreadDeps`` without spawning stdio."""
    global _deps  # noqa: PLW0603 — test seam; lifespan owns this in production
    _deps = deps


def _resolve_deps(ctx: Context[Any, Any] | None) -> ThreadDeps:
    """Return the active deps, preferring lifespan context when available."""
    if ctx is not None:
        try:
            rc = ctx.request_context
            deps = rc.lifespan_context
            if isinstance(deps, ThreadDeps):
                return deps
        except ValueError:
            pass  # direct call_tool path — no request_context
    if _deps is None:
        raise ToolError("lisa-thread is not initialized. Call set_deps_for_test(deps).")
    return _deps


# ── CLI config holder (populated by _init before run()) ────────────────────


@dataclass
class _ThreadConfig:
    thread_dir: Path = Path()
    user_slug: str = ""


_cfg = _ThreadConfig()


@asynccontextmanager
async def _lifespan(_app: MCPServer) -> AsyncIterator[ThreadDeps]:
    """Build the ThreadStore from the resolved CLI config and yield deps."""
    base = _cfg.thread_dir / _cfg.user_slug if _cfg.user_slug else _cfg.thread_dir
    store, policy = _build_thread_store(base)
    deps = ThreadDeps(store=store, redaction=policy)
    set_deps_for_test(deps)
    yield deps


# ── Server ────────────────────────────────────────────────────────────────

mcp = MCPServer(
    name="lisa-thread",
    version="1.0.0",
    lifespan=_lifespan,
)


# ── Tools ────────────────────────────────────────────────────────────────


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def thread_read_range(
    start_turn: int,
    end_turn: int,
    *,
    include_transcript: bool = False,
    cursor: str | None = None,
    budget_bytes: int = DEFAULT_READOUT_BYTES,
    ctx: Context[ThreadDeps, Any] | None = None,
) -> str:
    """Read a bounded range of thread entries by inclusive global turn index.

    Search with thread_search first, then read the relevant turn range.
    Returns {entries, next_cursor, has_more}. Small fields remain inline;
    large fields have {ref, head, tail, total_chars, omitted_chars} previews.
    Expand a ref with thread_read_content to obtain the complete permitted
    value through pages. The archive is lossless; previews are not full data.

    include_transcript adds tool calls/results as bounded message previews.
    Follow next_cursor with the same range and include_transcript setting;
    a page may continue within a turn (transcript_start gives its position).
    budget_bytes defaults to 16384 serialized UTF-8 bytes, clamped to
    16384-32768 including metadata. Cursors anchor the original turn window.
    """
    deps = _resolve_deps(ctx)
    try:
        return ThreadReader(deps.store, deps.redaction).read_range(
            start_turn,
            end_turn,
            include_transcript=include_transcript,
            cursor=cursor,
            budget_bytes=budget_bytes,
        )
    except ValueError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def thread_read_recent(
    limit: int = 10,
    include_transcript: bool = False,
    cursor: str | None = None,
    budget_bytes: int = DEFAULT_READOUT_BYTES,
    ctx: Context[ThreadDeps, Any] | None = None,
) -> str:
    """Read a bounded recent window, oldest-first (limit defaults to 10, capped at 50).

    Returns {entries, next_cursor, has_more} as in thread_read_range.
    Large fields are explicit previews; expand refs with thread_read_content.
    include_transcript adds bounded tool call/result previews. Continue with
    next_cursor and the same include_transcript setting; later appends do not
    shift the window. budget_bytes defaults to 16384, clamped to 16384-32768
    serialized UTF-8 bytes including metadata. Use thread_search for topics.
    """
    deps = _resolve_deps(ctx)
    try:
        return ThreadReader(deps.store, deps.redaction).read_recent(
            limit,
            include_transcript=include_transcript,
            cursor=cursor,
            budget_bytes=budget_bytes,
        )
    except ValueError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def thread_search(
    query: str,
    limit: int = 10,
    cursor: str | None = None,
    budget_bytes: int = DEFAULT_READOUT_BYTES,
    ctx: Context[ThreadDeps, Any] | None = None,
) -> str:
    """Find prior turns by lexical overlap in query, response, and mode.

    Returns {entries, next_cursor, has_more}, ranked by relevance then
    recency, with turn numbers, query/response previews, and score (0-1).
    Use thread_read_range for the relevant turns (include_transcript=true
    for tool calls/results), or thread_read_content to expand a preview ref.
    An empty query returns recent matches. limit defaults to 10, capped at 20.
    Continue with next_cursor and the same query/limit. budget_bytes defaults
    to 16384, clamped to 16384-32768 serialized UTF-8 bytes including metadata.
    """
    deps = _resolve_deps(ctx)
    try:
        return ThreadReader(deps.store, deps.redaction).search(
            query,
            limit,
            cursor=cursor,
            budget_bytes=budget_bytes,
        )
    except ValueError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def thread_read_content(
    ref: str,
    offset_chars: int = 0,
    limit_chars: int = 4000,
    budget_bytes: int = DEFAULT_READOUT_BYTES,
    ctx: Context[ThreadDeps, Any] | None = None,
) -> str:
    """Expand a thread preview ref without dumping unrelated history.

    Returns {ref, content, offset_chars, total_chars, next_offset_chars,
    has_more}. Follow next_offset_chars with the same ref to recover the
    complete permitted value. Offsets count Unicode characters AFTER
    redaction; structured values are serialized JSON. limit_chars defaults
    to 4000. budget_bytes defaults to 16384, clamped to 16384-32768 UTF-8
    bytes including metadata; a page may contain fewer characters than asked.

    Refs are scoped to this thread: t391.query, t391.response_text,
    t391.m20.content, t391.m19.tool_calls, or t391.m20 for a full message.
    Full reasoning_text and raw turn_input are not available through this tool.
    """
    deps = _resolve_deps(ctx)
    try:
        return ThreadReader(deps.store, deps.redaction).read_content(
            ref,
            offset_chars,
            limit_chars,
            budget_bytes=budget_bytes,
        )
    except ValueError as exc:
        raise ToolError(str(exc)) from exc


# ── Entry point ─────────────────────────────────────────────────────────────


def _add_thread_args(parser: Any) -> None:
    """Add ``--thread-dir`` and ``--user-slug`` to the standard parser."""
    parser.add_argument(
        "--thread-dir",
        required=False,
        type=str,
        default="",
        help="Directory containing per-user thread stores. The active "
        "store is <thread-dir>/<user-slug>. Defaults to --data-dir/threads "
        "when omitted.",
    )
    parser.add_argument(
        "--user-slug",
        required=False,
        type=str,
        default="",
        help="User slug selecting the per-user thread store. When empty, "
        "the server reads from --thread-dir directly (single-user mode).",
    )


def init_thread(
    data_dir: Path,
    *,
    thread_dir: str = "",
    user_slug: str = "",
) -> None:
    """Initialize the thread server with resolved paths.

    Kept as a thin back-compat shim for tests that call ``init_thread``
    directly.
    """
    resolved_thread_dir = Path(thread_dir) if thread_dir else data_dir / DATA_DIR / "threads"
    _cfg.thread_dir = resolved_thread_dir
    _cfg.user_slug = user_slug
    base = resolved_thread_dir / user_slug if user_slug else resolved_thread_dir
    store, policy = _build_thread_store(base)
    set_deps_for_test(ThreadDeps(store=store, redaction=policy))


def _init(args: Any) -> None:
    """Populate ``_cfg`` from parsed args so the lifespan can build the store."""
    data_dir = Path(args.data_dir)
    thread_dir = getattr(args, "thread_dir", "") or ""
    user_slug = getattr(args, "user_slug", "") or ""
    _cfg.thread_dir = Path(thread_dir) if thread_dir else data_dir / DATA_DIR / "threads"
    _cfg.user_slug = user_slug
    base = _cfg.thread_dir / user_slug if user_slug else _cfg.thread_dir
    store, policy = _build_thread_store(base)
    set_deps_for_test(ThreadDeps(store=store, redaction=policy))


def main() -> None:
    """Entry point for the lisa-thread MCP server (stdio transport).

    Parses the standard CLI contract via the shared ``run()`` helper.
    ``--thread-dir`` (defaults to ``--data-dir/threads``) and
    ``--user-slug`` select the per-user thread store.
    """
    from mcp_server.config.cli import run

    run(
        "lisa-thread",
        mcp,
        _init,
        supports_readonly=False,
        add_args=_add_thread_args,
    )


if __name__ == "__main__":
    main()
