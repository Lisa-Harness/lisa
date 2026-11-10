"""Shared, long-lived stdio MCP sessions for the synchronous catalog/content adapters.

The catalog and content adapters (:mod:`kernel.context.mcp_catalog`,
:mod:`kernel.context.mcp_content_registry`) expose a synchronous API but talk
to ``lisa-catalog`` / ``lisa-content`` over MCP stdio. Spawning a fresh Python
server per call costs a full interpreter start plus imports (seconds each),
so this module keeps one subprocess per distinct server config alive while
the framework build that started it needs it, and multiplexes every call
over it.

All sessions are owned by a single daemon thread running its own event loop.
Each session lives inside one long-running owner task so the anyio task-group
scoping of ``stdio_client`` / ``ClientSession`` is entered and exited in the
same task. Nothing is spawned on import; sessions start on first use (or on an
explicit :func:`prewarm`). ``atexit`` remains a last-resort cleanup only.

Framework builds own their sessions through epoch leases: a build calls
:func:`acquire_lease` before its adapters create any
:class:`SharedStdioSession`, and those sessions are stamped with the lease's
epoch. A released lease stops exactly its own build's sessions — a second,
concurrently live framework build keeps its subprocesses — and a later build
reusing the same server config always gets a fresh subprocess, whose
server-side read caches therefore never survive past the build that created
them (the stale-content-after-edit gap in Slice A of
``docs/SPEC/PROMPT_STARTUP.md``). Using a handle whose lease was released
raises instead of silently spawning a replacement.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import itertools
import logging
import threading
from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import Any, TypeVar

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

logger = logging.getLogger(__name__)

__all__ = ["SharedSessionLease", "SharedStdioSession", "acquire_lease", "close_shared_sessions"]

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
    epoch: int = -1
    """The lease epoch this session was started under."""


@dataclass
class SharedSessionLease:
    """One framework build's ownership claim on shared sessions.

    Created by :func:`acquire_lease`; released explicitly by the owner
    (normal shutdown or partial-construction failure). Sessions started
    under the lease's epoch are stopped exactly once, by whichever of
    :meth:`release` or :func:`close_shared_sessions` runs first.
    """

    epoch: int
    _released: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def released(self) -> bool:
        """Whether :meth:`release` has already stopped this build's sessions."""
        with self._lock:
            return self._released

    def release(self) -> None:
        """Stop this build's sessions; idempotent and failure-visible."""
        with self._lock:
            if self._released:
                return
            self._released = True
        _HOST.close_epoch(self.epoch)


class _SessionHost:
    """Background event loop that owns every shared stdio session."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._entries: dict[tuple[int, _SessionKey], _Entry] = {}
        self._epoch_counter = itertools.count()

    def _next_epoch(self) -> int:
        with self._lock:
            return next(self._epoch_counter)

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

    def entry_for(self, key: _SessionKey, params: StdioServerParameters, epoch: int) -> _Entry:
        """Return the entry for ``(epoch, key)``, starting its owner if needed.

        Sharing is scoped to the lease epoch: equal-config handles from the
        same framework build share one subprocess, while a later build's
        handles always start a fresh one (fresh server-side caches).
        """
        storage_key = (epoch, key)
        entry = self._entries.get(storage_key)
        if entry is not None and not (entry.owner is not None and entry.owner.done()):
            return entry
        loop = asyncio.get_running_loop()
        entry = _Entry(ready=loop.create_future(), stop=asyncio.Event(), epoch=epoch)
        # Mark a startup failure as retrieved even if no caller awaits it
        # (e.g. a prewarm that nobody follows up on).
        entry.ready.add_done_callback(lambda f: f.cancelled() or f.exception())
        entry.owner = loop.create_task(
            self._own(params, entry), name=f"lisa-shared-mcp-{params.command}"
        )
        self._entries[storage_key] = entry
        return entry

    def evict(self, storage_key: tuple[int, _SessionKey], entry: _Entry) -> None:
        """Drop ``entry`` (if still current) and ask its owner task to exit."""
        if self._entries.get(storage_key) is entry:
            del self._entries[storage_key]
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

    async def _close_epoch(self, epoch: int) -> None:
        """Stop and await the owners of one epoch, leaving other epochs live."""
        stale = [
            (storage_key, entry)
            for storage_key, entry in self._entries.items()
            if entry.epoch == epoch
        ]
        for storage_key, entry in stale:
            self.evict(storage_key, entry)
        owners = [entry.owner for _, entry in stale if entry.owner is not None]
        if not owners:
            return
        _, pending = await asyncio.wait(owners, timeout=_CLOSE_TIMEOUT_SECONDS)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.wait(pending, timeout=_CLOSE_TIMEOUT_SECONDS)

    async def _close_all(self) -> None:
        # Explicitly stop every entry still registered (an evicted owner's
        # stop was already set when it was evicted): close() must await ALL
        # owner tasks — including ones a released lease evicted — so the
        # process holds no owned sessions when it returns.
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

    def close_epoch(self, epoch: int) -> None:
        """Close one build's sessions; the loop thread and other epochs stay."""
        with self._lock:
            loop, thread = self._loop, self._thread
        if loop is None or thread is None or not thread.is_alive():
            return
        try:
            asyncio.run_coroutine_threadsafe(self._close_epoch(epoch), loop).result(
                timeout=_CLOSE_TIMEOUT_SECONDS * 2 + 1
            )
        except Exception as exc:  # best-effort shutdown
            logger.debug(
                "Shared MCP stdio epoch-%d shutdown did not complete cleanly: %s", epoch, exc
            )


_HOST = _SessionHost()
atexit.register(_HOST.close)


def close_shared_sessions() -> None:
    """Terminate every shared stdio MCP server started by the catalog/content adapters.

    Safe to call multiple times; a later adapter call transparently starts a
    new session.
    """
    _HOST.close()


def acquire_lease() -> SharedSessionLease:
    """Return an epoch lease for one framework build.

    Call once per ``setup_framework`` build before its adapters construct
    their :class:`SharedStdioSession` handles: every session those handles
    start belongs to the lease, and releasing it stops exactly those
    sessions. Concurrent builds holding their own leases keep running.
    """
    return SharedSessionLease(epoch=_HOST._next_epoch())


class SharedStdioSession:
    """Synchronous handle to a shared, long-lived stdio MCP server session.

    Adapters constructed from equal configs (same command, args, env and cwd)
    share one subprocess *within the owning lease's epoch*: the subprocess
    starts lazily on the first :meth:`call` or on :meth:`prewarm`, and
    releasing the lease stops it, so a later framework build gets a fresh
    subprocess — and therefore fresh server-side caches — instead of
    inheriting a server that predates an edit to the content it serves.
    Calling into a handle whose lease was released raises instead of
    silently starting a new session outside the build's ownership.

    Args:
        params: Stdio launch parameters for the server.
        lease: Lease owning this handle's sessions. ``None`` (the default)
            is the process-global fallback used by direct consumers (tests
            and other non-framework callers); those sessions live until
            :func:`close_shared_sessions`.
    """

    def __init__(
        self, params: StdioServerParameters, lease: SharedSessionLease | None = None
    ) -> None:
        self._params = params
        self._lease = lease
        self._epoch = -1 if lease is None else lease.epoch
        env = params.env or {}
        self._key: _SessionKey = (
            params.command,
            tuple(params.args),
            tuple(sorted(env.items())),
            None if params.cwd is None else str(params.cwd),
        )

    def prewarm(self) -> None:
        """Start the server session in the background without waiting for it."""
        self._ensure_admitted()

        async def _start() -> None:
            _HOST.entry_for(self._key, self._params, self._epoch)

        _HOST.submit(_start()).result()

    def call(self, tool: str, arguments: dict[str, Any]) -> Any:
        """Call ``tool`` on the shared session and return the raw ``CallToolResult``.

        A session whose subprocess died after a successful start is replaced
        (under the same lease epoch) and the call retried once. A server that
        fails to start raises the startup error without retrying. The retry
        covers transport loss only — an MCP application error is never
        retried, and no other write is ever replayed.
        """
        self._ensure_admitted()
        return _HOST.run(self._call(tool, arguments))

    def _ensure_admitted(self) -> None:
        if self._lease is not None and self._lease.released:
            raise RuntimeError(
                "Shared MCP stdio session used after its framework build released it; "
                "acquire a new lease (setup_framework) before constructing adapters."
            )

    async def _call(self, tool: str, arguments: dict[str, Any]) -> Any:
        for attempt in range(2):
            entry = _HOST.entry_for(self._key, self._params, self._epoch)
            try:
                session = await asyncio.shield(entry.ready)
            except BaseException:
                _HOST.evict((self._epoch, self._key), entry)
                raise
            try:
                result = await session.call_tool(tool, arguments=arguments)
            except Exception:
                _HOST.evict((self._epoch, self._key), entry)
                if attempt == 0 and entry.established:
                    logger.info("Retrying %s on a fresh %s session", tool, self._params.command)
                    continue
                raise
            return result
        raise AssertionError("unreachable")  # pragma: no cover
