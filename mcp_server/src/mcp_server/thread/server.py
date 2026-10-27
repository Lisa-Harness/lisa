"""Lisa Thread MCP server — thin facade over :class:`sdk.thread.ThreadStore`.

This is the read-only thread-access server. It wraps the SDK's
``ThreadStore`` (segmented, zstd-sealed, lossless conversation log) and
exposes bounded views so the model can pull older entries on demand:

- ``thread_read_range`` — read entries by global turn index range.
- ``thread_read_recent`` — read the most recent N entries (compact form).
- ``thread_search`` — find prior turns by lexical relevance.
- ``thread_read_content`` — expand a referenced value through bounded pages.
- ``thread_find_content`` — locate literal evidence inside a referenced value.
- ``thread_task_view`` — summarize one task's recorded state and evidence.

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
    PageCache,
    ThreadReader,
    ThreadRedactionPolicy,
    ThreadStore,
    resolve_redaction_policy,
)

__all__ = ["ThreadDeps", "init_thread", "main", "mcp", "set_deps_for_test"]

logger = logging.getLogger(__name__)

_REDACTION_ENABLED_ENV = "LISA_THREAD_REDACTION_ENABLED"
_REDACTION_PATTERNS_ENV = "LISA_THREAD_REDACTION_PATTERNS"


def _resolve_redaction_policy(
    *,
    enabled_value: str | None = None,
    patterns_value: str | None = None,
) -> ThreadRedactionPolicy | None:
    """Resolve the redaction policy from the environment contract.

    Delegates to :func:`sdk.thread.resolve_redaction_policy` so this server and
    the forensic reader cannot disagree about whether a deployment redacts.
    The store's own redaction covers ``build_recent_excerpt`` /
    ``build_manifest`` / ``search`` only — ``load_range`` / ``load_recent``
    return RAW entries by design — so any surface that projects entries into a
    prompt applies the policy itself.
    """
    return resolve_redaction_policy(
        enabled_value=enabled_value,
        patterns_value=patterns_value,
    )


def _build_thread_store(
    base: Path,
    *,
    enabled_value: str | None = None,
    patterns_value: str | None = None,
) -> tuple[ThreadStore, ThreadRedactionPolicy | None]:
    """Build the ``(store, redaction_policy)`` pair from the env contract.

    Single constructor for every store this server builds (lifespan, CLI
    init, and the ``init_thread`` test seam) so all read paths apply the
    same policy — the store for its excerpt/manifest/search builders, the
    returned policy for the server-side projection of raw ``load_range``
    / ``load_recent`` results.
    """
    policy = _resolve_redaction_policy(
        enabled_value=enabled_value,
        patterns_value=patterns_value,
    )
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
    page_cache: PageCache | None = None


def _reader(deps: ThreadDeps) -> ThreadReader:
    """Build a reader over the deps store, sharing the bounded page cache."""
    if deps.page_cache is None:
        deps.page_cache = PageCache()
    return ThreadReader(deps.store, deps.redaction, page_cache=deps.page_cache)


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
    redaction_enabled: str | None = None
    redaction_patterns: str | None = None


_cfg = _ThreadConfig()


@asynccontextmanager
async def _lifespan(_app: MCPServer) -> AsyncIterator[ThreadDeps]:
    """Build the ThreadStore from the resolved CLI config and yield deps."""
    base = _cfg.thread_dir / _cfg.user_slug if _cfg.user_slug else _cfg.thread_dir
    store, policy = _build_thread_store(
        base,
        enabled_value=_cfg.redaction_enabled,
        patterns_value=_cfg.redaction_patterns,
    )
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
    """Read recorded past exchanges by inclusive global turn index, not the active request.

    Search with thread_search first, then read the relevant turn range.
    Returns {scope, interpretation, entries, next_cursor, has_more}.
    Entries carry recorded timestamps (null if unknown) and provenance sources.
    Earlier instructions and diagnoses are historical, not live runtime state.
    Small fields remain inline;
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
        return _reader(deps).read_range(
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
    """Read recorded history, oldest-first (limit defaults to 10, capped at 50).

    Returns the labeled history envelope as in thread_read_range, including
    recorded timestamps and provenance sources. Latest recorded is not current.
    Large fields are explicit previews; expand refs with thread_read_content.
    include_transcript adds bounded tool call/result previews. Continue with
    next_cursor and the same include_transcript setting; later appends do not
    shift the window. budget_bytes defaults to 16384, clamped to 16384-32768
    serialized UTF-8 bytes including metadata. Use thread_search for topics.
    """
    deps = _resolve_deps(ctx)
    try:
        return _reader(deps).read_recent(
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
    *,
    cursor: str | None = None,
    budget_bytes: int = DEFAULT_READOUT_BYTES,
    tool_name: str | None = None,
    path: str | None = None,
    task_id: str | None = None,
    start_turn: int = 0,
    end_turn: int | None = None,
    ctx: Context[ThreadDeps, Any] | None = None,
) -> str:
    """Find prior turns and tool evidence by literal phrase/token matches.

    Returns a recorded_history envelope with timestamps and provenance sources,
    not live task state. Earlier instructions and diagnoses are historical.
    Entries are ranked by relevance then
    recency, with turn numbers, query/response previews, and score (0-1).
    Use thread_read_range for the relevant turns (include_transcript=true
    for tool calls/results), or thread_read_content to expand a preview ref.
    An empty query returns recent matches. limit defaults to 10, capped at 20.
    Continue with next_cursor and the same query/limit. budget_bytes defaults
    to 16384, clamped to 16384-32768 serialized UTF-8 bytes including metadata.
    Search includes tool arguments/results. Filter by exact tool_name, path
    substring in structured arguments, task_id, and inclusive start/end_turn.
    Each match has ref, excerpt, and offset_chars for thread_read_content.
    All filters must remain the same when following next_cursor. Use
    thread_find_content to locate text inside a particular large result.
    """
    deps = _resolve_deps(ctx)
    try:
        return _reader(deps).search(
            query,
            limit,
            cursor=cursor,
            budget_bytes=budget_bytes,
            tool_name=tool_name,
            path=path,
            task_id=task_id,
            start_turn=start_turn,
            end_turn=end_turn,
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

    Returns {ref, source, content, offset_chars, total_chars, next_offset_chars,
    has_more}. Follow next_offset_chars with the same ref to recover the
    complete permitted value. Offsets count Unicode characters AFTER
    redaction; structured values are serialized JSON. limit_chars defaults
    to 4000. budget_bytes defaults to 16384, clamped to 16384-32768 UTF-8
    bytes including metadata; a page may contain fewer characters than asked.

    Refs are scoped to this thread: t391.query, t391.response_text,
    t391.m20.content, t391.m19.tool_calls, or t391.m20 for a full message.
    Search can also return t391.m19.c0.arguments or .name for one call.
    Live/replayed result previews carry r<sha256> refs, available before
    the current turn completes and after restart. inventory_ref points to
    omitted tool groups with their arguments and full-result references.
    Full reasoning_text and raw turn_input are not available through this tool.
    Turn refs retain historical source labels and recorded timestamps. Hash refs
    are stored tool output with unknown timing; they may come from a live turn.
    """
    deps = _resolve_deps(ctx)
    try:
        return _reader(deps).read_content(
            ref,
            offset_chars,
            limit_chars,
            budget_bytes=budget_bytes,
        )
    except ValueError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def thread_find_content(
    ref: str,
    query: str,
    *,
    offset_chars: int = 0,
    limit: int = 10,
    budget_bytes: int = DEFAULT_READOUT_BYTES,
    ctx: Context[ThreadDeps, Any] | None = None,
) -> str:
    """Find literal text inside a referenced value, case-insensitively.

    Returns bounded {ref, source, matches, next_offset_chars, has_more, total_chars}.
    Source labels distinguish recorded history from stored tool output of unknown timing.
    Matches give excerpt, offset_chars, and match_offset_chars after redaction.
    Read the relevant span with thread_read_content(ref, offset_chars).
    Follow next_offset_chars with the same ref/query to find further matches.
    limit defaults to 10, capped at 20; budget_bytes is clamped to 16384-32768.
    No regex execution or raw-policy bypass is available.
    """
    deps = _resolve_deps(ctx)
    try:
        return _reader(deps).find_content(
            ref,
            query,
            offset_chars,
            limit,
            budget_bytes=budget_bytes,
        )
    except ValueError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def thread_task_view(
    task_id: str = "",
    *,
    budget_bytes: int = DEFAULT_READOUT_BYTES,
    ctx: Context[ThreadDeps, Any] | None = None,
) -> str:
    """Summarize one task's recorded state instead of replaying its history.

    task_id defaults to the most recent recorded task identity. Scans at
    most 200 recent entries. This is NOT the active invocation's task identity.
    scope, selection, and as_of_utc label the view as recorded history;
    status and blockers are recorded observations, not verified live state.
    Returns objective/latest previews and refs,
    blocking status, modes, receipt groups (capability_id, effect, count,
    artifact_ids), and archive evidence refs for thread_read_content or
    thread_find_content. Fields derive from recorded state, not model summaries.
    history_may_be_incomplete warns that the objective or earlier work may lie
    outside the scan; use task-filtered thread_search for older evidence.
    Truncation flags and identity previews mark omitted text. Expand refs
    rather than treating shortened IDs as complete. budget_bytes is positive,
    clamped to 1024-32768 UTF-8 bytes including metadata.
    """
    deps = _resolve_deps(ctx)
    try:
        return _reader(deps).task_view(
            task_id,
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
    # The in-process path hands the host-resolved redaction contract over
    # the compiled config; the subprocess path leaves these unset so the
    # server reads them from its own environment.
    _cfg.redaction_enabled = getattr(args, "thread_redaction_enabled", None)
    _cfg.redaction_patterns = getattr(args, "thread_redaction_patterns", None)
    base = _cfg.thread_dir / user_slug if user_slug else _cfg.thread_dir
    store, policy = _build_thread_store(
        base,
        enabled_value=_cfg.redaction_enabled,
        patterns_value=_cfg.redaction_patterns,
    )
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
