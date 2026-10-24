"""Shared, long-lived stdio MCP sessions for the synchronous catalog/content adapters.

The catalog and content adapters (:mod:`kernel.context.mcp_catalog`,
:mod:`kernel.context.mcp_content_registry`) expose a synchronous API but talk
to ``lisa-catalog`` / ``lisa-content`` over MCP stdio. Spawning a fresh Python
server per call costs a full interpreter start plus imports (seconds each),
so this module keeps one subprocess per distinct server config alive for the
host process lifetime and multiplexes every call over it.

All sessions are owned by a single daemon thread running its own event loop.
Each session lives inside one long-running owner task so the anyio task-group
scoping of ``stdio_client`` / ``ClientSession`` is entered and exited in the
same task. Nothing is spawned on import; sessions start on first use (or on an
explicit :func:`prewarm`) and are closed by :func:`close_shared_sessions`, which
is also registered with :mod:`atexit`.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import logging
import threading
from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import Any, TypeVar

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

logger = logging.getLogger(__name__)

__all__ = ["SharedStdioSession", "close_shared_sessions"]

_T = TypeVar("_T")
_CLOSE_TIMEOUT_SECONDS = 5.0

_SessionKey = tuple[str, tuple[str, ...], tuple[tuple[str, str], ...], str | None]


@dataclass
class _Entry:
    """One live (or connecting) server session owned by the loop thread."""

    ready: asyncio.Future[ClientSession]
    stop: asyncio.Event
    owner: asyncio.Task[None] | None = None
    established: bool = field(default=False)


class _SessionHost:
    """Background event loop that owns every shared stdio session."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._entries: dict[_SessionKey, _Entry] = {}

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is not None and self._thread is not None and self._thread.is_alive():
                return self._loop
            loop = asyncio.new_event_loop()
            started = threading.Event()

            def _run() -> None:
                asyncio.set_event_loop(loop)
                loop.call_soon(started.set)
                loop.run_forever()

            thread = threading.Thread(target=_run, name="lisa-shared-mcp-stdio", daemon=True)
            thread.start()
            started.wait()
            self._loop = loop
            self._thread = thread
            self._entries = {}
            return loop

    def submit(self, coro: Coroutine[Any, Any, _T]) -> concurrent.futures.Future[_T]:
        """Schedule ``coro`` on the session loop and return a thread-safe future."""
        loop = self._ensure_loop()
        return asyncio.run_coroutine_threadsafe(coro, loop)

    def run(self, coro: Coroutine[Any, Any, _T]) -> _T:
        """Run ``coro`` on the session loop and block until it completes."""
        if threading.current_thread() is self._thread:
            coro.close()
            raise RuntimeError("Shared MCP stdio calls must not run on the session loop thread.")
        return self.submit(coro).result()

    # ── Loop-thread helpers (only called on the session loop) ────────────

    def entry_for(self, key: _SessionKey, params: StdioServerParameters) -> _Entry:
        """Return the entry for ``key``, starting its owner task if needed."""
        entry = self._entries.get(key)
        if entry is not None and not (entry.owner is not None and entry.owner.done()):
            return entry
        loop = asyncio.get_running_loop()
        entry = _Entry(ready=loop.create_future(), stop=asyncio.Event())
        # Mark a startup failure as retrieved even if no caller awaits it
        # (e.g. a prewarm that nobody follows up on).
        entry.ready.add_done_callback(lambda f: f.cancelled() or f.exception())
        entry.owner = loop.create_task(
            self._own(params, entry), name=f"lisa-shared-mcp-{params.command}"
        )
        self._entries[key] = entry
        return entry

    def evict(self, key: _SessionKey, entry: _Entry) -> None:
        """Drop ``entry`` (if still current) and ask its owner task to exit."""
        if self._entries.get(key) is entry:
            del self._entries[key]
        entry.stop.set()

    @staticmethod
    async def _own(params: StdioServerParameters, entry: _Entry) -> None:
        try:
            async with (
                stdio_client(params) as (read, write_stream),
                ClientSession(read, write_stream) as session,
            ):
                await session.initialize()
                entry.established = True
                entry.ready.set_result(session)
                await entry.stop.wait()
        except BaseException as exc:
            if not entry.ready.done():
                if isinstance(exc, asyncio.CancelledError):
                    entry.ready.cancel()
                else:
                    entry.ready.set_exception(exc)
            elif not isinstance(exc, asyncio.CancelledError):
                logger.warning("Shared MCP stdio session for %s ended: %s", params.command, exc)
            if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                raise

    async def _close_all(self) -> None:
        entries = list(self._entries.values())
        self._entries.clear()
        for entry in entries:
            entry.stop.set()
        owners = [e.owner for e in entries if e.owner is not None]
        if not owners:
            return
        _, pending = await asyncio.wait(owners, timeout=_CLOSE_TIMEOUT_SECONDS)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.wait(pending, timeout=_CLOSE_TIMEOUT_SECONDS)

    def close(self) -> None:
        """Close every shared session and stop the loop thread."""
        with self._lock:
            loop, thread = self._loop, self._thread
            self._loop = None
            self._thread = None
        if loop is None or thread is None or not thread.is_alive():
            return
        try:
            asyncio.run_coroutine_threadsafe(self._close_all(), loop).result(
                timeout=_CLOSE_TIMEOUT_SECONDS * 2 + 1
            )
        except Exception as exc:  # best-effort shutdown
            logger.debug("Shared MCP stdio shutdown did not complete cleanly: %s", exc)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=_CLOSE_TIMEOUT_SECONDS)
        if not thread.is_alive():
            loop.close()


_HOST = _SessionHost()
atexit.register(_HOST.close)


def close_shared_sessions() -> None:
    """Terminate every shared stdio MCP server started by the catalog/content adapters.

    Safe to call multiple times; a later adapter call transparently starts a
    new session.
    """
    _HOST.close()


class SharedStdioSession:
    """Synchronous handle to a shared, long-lived stdio MCP server session.

    Adapters constructed from equal configs (same command, args, env and cwd)
    share one subprocess. The subprocess starts lazily on the first
    :meth:`call` or on :meth:`prewarm`.

    Args:
        params: Stdio launch parameters for the server.
    """

    def __init__(self, params: StdioServerParameters) -> None:
        self._params = params
        env = params.env or {}
        self._key: _SessionKey = (
            params.command,
            tuple(params.args),
            tuple(sorted(env.items())),
            None if params.cwd is None else str(params.cwd),
        )

    def prewarm(self) -> None:
        """Start the server session in the background without waiting for it."""

        async def _start() -> None:
            _HOST.entry_for(self._key, self._params)

        _HOST.submit(_start()).result()

    def call(self, tool: str, arguments: dict[str, Any]) -> Any:
        """Call ``tool`` on the shared session and return the raw ``CallToolResult``.

        A session whose subprocess died after a successful start is replaced
        and the call retried once. A server that fails to start raises the
        startup error without retrying.
        """
        return _HOST.run(self._call(tool, arguments))

    async def _call(self, tool: str, arguments: dict[str, Any]) -> Any:
        for attempt in range(2):
            entry = _HOST.entry_for(self._key, self._params)
            try:
                session = await asyncio.shield(entry.ready)
            except BaseException:
                _HOST.evict(self._key, entry)
                raise
            try:
                result = await session.call_tool(tool, arguments=arguments)
            except Exception:
                _HOST.evict(self._key, entry)
                if attempt == 0 and entry.established:
                    logger.info("Retrying %s on a fresh %s session", tool, self._params.command)
                    continue
                raise
            return result
        raise AssertionError("unreachable")  # pragma: no cover
