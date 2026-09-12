"""Lisa Thread MCP server — thin facade over :class:`sdk.thread.ThreadStore`.

This is the read-only thread-access server. It wraps the SDK's
``ThreadStore`` (segmented, zstd-sealed, lossless conversation log) and
exposes two tools so the model can pull older thread entries on demand:

- ``thread_read_range`` — read entries by global turn index range.
- ``thread_read_recent`` — read the most recent N entries (compact form).

The write path (append, seal, dictionary training) stays in
``host.thread``; this server is read-only.

The server resolves the thread base path from ``--thread-dir`` and
``--user-slug`` CLI args (the host's ``runtime.paths.thread_path`` property
produces ``thread_dir / user_slug``). No env-var path resolution.

**Redaction**: the store is built with the ``ThreadRedactionPolicy``
resolved from ``LISA_THREAD_REDACTION_ENABLED`` /
``LISA_THREAD_REDACTION_PATTERNS`` (the same env contract
``host.config`` uses for the host's in-process reads). The host
propagates both variables to this subprocess in ``host.mcp_pool``;
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

import json
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

from sdk.thread import DEFAULT_REDACTION_PATTERNS, ThreadRedactionPolicy, ThreadStore

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


def _compact_entry(entry: dict[str, Any], redaction: Any = None) -> dict[str, Any]:
    """Project a full thread entry to a compact form for the model.

    When ``redaction`` is a :class:`ThreadRedactionPolicy`, ``query`` and
    ``response_text`` are redacted at projection time — the raw store
    entry is never mutated, and harness-authored budget/cap prose never
    reaches the model through this tool.
    """
    response = str(entry.get("response_text") or "")
    query = str(entry.get("query") or "")
    if redaction is not None:
        response = redaction.redact(response)
        query = redaction.redact(query)
    max_chars = 2000
    if len(response) > max_chars:
        response = response[:max_chars] + "\n[…truncated…]"
    receipts = entry.get("effect_receipts") or []
    receipt_caps = [
        str(r.get("capability_id") or "") for r in receipts if isinstance(r, dict)
    ]
    return {
        "turn": entry.get("turn"),
        "mode_id": entry.get("mode_id"),
        "query": query,
        "response_text": response,
        "handoff_to": entry.get("handoff_to"),
        "transitioned_to": entry.get("transitioned_to"),
        "receipt_capability_ids": receipt_caps,
    }


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def thread_read_range(
    start_turn: int,
    end_turn: int,
    ctx: Context[ThreadDeps, Any] | None = None,
) -> str:
    """Read a range of entries from the conversation thread by global turn index.

    Use this to recall prior turns beyond the recent excerpt and thread
    manifest injected in the system prompt. Returns a JSON array of thread
    entries, each containing mode_id, query, response_text (truncated to
    2000 chars), handoff_to, transitioned_to, and receipt_capability_ids.

    Pass start_turn=0 and a large end_turn (e.g. 999999) for the full
    thread. Use ``thread_search`` to find turn numbers for topics beyond
    the manifest window, then pull those turns with this tool.

    Args:
        start_turn: The global (0-based) turn index to start reading from,
            inclusive.
        end_turn: The global turn index to stop reading at, inclusive. Use
            a large number for "to the end".

    Returns:
        JSON array of compact thread entries.
    """
    deps = _resolve_deps(ctx)
    entries = deps.store.load_range(start_turn=start_turn, end_turn=end_turn)
    compact = [_compact_entry(e, redaction=deps.redaction) for e in entries]
    return json.dumps(compact, ensure_ascii=False)


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def thread_read_recent(
    limit: int = 10,
    ctx: Context[ThreadDeps, Any] | None = None,
) -> str:
    """Read the most recent N thread entries in compact form.

    Returns the same compact JSON shape as ``thread_read_range``, but for
    the most recent ``limit`` entries (default 10). Useful when the system
    prompt's recent excerpt (3 entries) is not enough and you need a few
    more turns of context. For content-based lookups across deeper
    history, use ``thread_search`` instead.

    Args:
        limit: Maximum number of recent entries to return (default 10,
            capped at 50).

    Returns:
        JSON array of compact thread entries (oldest-first).
    """
    deps = _resolve_deps(ctx)
    capped_limit = max(1, min(limit, 50))
    entries = deps.store.load_recent(limit=capped_limit)
    compact = [_compact_entry(e, redaction=deps.redaction) for e in entries]
    return json.dumps(compact, ensure_ascii=False)


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def thread_search(
    query: str,
    limit: int = 10,
    ctx: Context[ThreadDeps, Any] | None = None,
) -> str:
    """Search the conversation thread by content and return ranked matches.

    Use this to find prior turns by topic when the thread manifest (last
    30 turns) does not cover what you need. Returns a JSON array of search
    results ranked by lexical relevance, each containing turn, mode_id,
    query (full text), response_preview (first 200 chars), handoff_to,
    transitioned_to, and score (match confidence, 0–1).

    After finding the right turn numbers with this tool, use
    ``thread_read_range`` to pull the full entry content for those turns.

    An empty query returns the most recent entries (recency fallback),
    which is useful for a quick "what just happened" check without
    scanning the manifest.

    Args:
        query: Search text. Matched against the query, response_text, and
            mode_id of every thread entry using whitespace-tokenized
            lexical overlap.
        limit: Maximum number of results to return (default 10, capped at
            20).

    Returns:
        JSON array of search result objects (oldest match first within
        equal-score ties).
    """
    deps = _resolve_deps(ctx)
    results = deps.store.search(query=query, limit=limit)
    return json.dumps(results, ensure_ascii=False)


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
    resolved_thread_dir = Path(thread_dir) if thread_dir else data_dir / "threads"
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
    _cfg.thread_dir = Path(thread_dir) if thread_dir else data_dir / "threads"
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