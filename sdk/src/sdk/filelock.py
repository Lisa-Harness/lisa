"""Cross-process advisory file locking, atomic file writes, and PID liveness checks."""

from __future__ import annotations

import asyncio
import contextlib
import errno
import os
import sys
import tempfile
import threading
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

if sys.platform == "win32":  # pragma: no cover - Windows-specific
    import msvcrt
else:
    import fcntl

__all__ = [
    "FileLockTimeoutError",
    "async_file_lock",
    "atomic_write_bytes",
    "atomic_write_text",
    "file_lock",
    "is_process_alive",
    "sidecar_lock_path",
]

_DEFAULT_TIMEOUT_S = 30.0
_INITIAL_POLL_INTERVAL_S = 0.001
_MAX_POLL_INTERVAL_S = 0.025

# Active lock descriptors keyed by (resolved_lock_path, thread_id, task_id) -> [fd, depth]
_ACTIVE_LOCKS_GUARD = threading.Lock()
_ACTIVE_LOCKS: dict[tuple[Path, int, int | None], list[int]] = {}


class FileLockTimeoutError(TimeoutError):
    """Raised when an advisory file lock cannot be acquired within the timeout."""


def sidecar_lock_path(target_path: Path) -> Path:
    """Return the companion ``.lock`` path for a state or config file."""
    return Path(f"{target_path}.lock")


def _owner_key(resolved_path: Path) -> tuple[Path, int, int | None]:
    """Build a reentrancy key scoped to the current OS thread and asyncio Task."""
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    task_id = id(task) if task is not None else None
    return (resolved_path, threading.get_ident(), task_id)


def _try_flock_nonblocking(fd: int) -> bool:
    """Attempt a non-blocking exclusive advisory lock on ``fd``."""
    if sys.platform == "win32":  # pragma: no cover - Windows-specific
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK):
            return False
        raise


def _unlock_and_close(fd: int) -> None:
    """Release the advisory lock on ``fd`` and close the descriptor."""
    try:
        if sys.platform == "win32":  # pragma: no cover - Windows-specific
            with contextlib.suppress(OSError):
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        with contextlib.suppress(OSError):
            os.close(fd)


def _open_lock_fd(resolved: Path) -> int:
    """Open or create the sidecar lock file without truncating it."""
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return os.open(str(resolved), os.O_CREAT | os.O_RDWR, 0o600)


@contextlib.contextmanager
def file_lock(
    lock_path: Path | str,
    *,
    timeout: float = _DEFAULT_TIMEOUT_S,
) -> Iterator[None]:
    """Acquire an exclusive cross-process advisory file lock.

    Lock files are intentionally never unlinked on release so concurrent
    waiters always contend on the same inode. Reentrant within the same
    thread + asyncio Task.
    """
    resolved = Path(lock_path).resolve()
    key = _owner_key(resolved)

    with _ACTIVE_LOCKS_GUARD:
        existing = _ACTIVE_LOCKS.get(key)
        if existing is not None:
            existing[1] += 1
            reentered = True
        else:
            reentered = False

    if reentered:
        try:
            yield
        finally:
            with _ACTIVE_LOCKS_GUARD:
                entry = _ACTIVE_LOCKS.get(key)
                if entry is not None:
                    entry[1] -= 1
        return

    fd = _open_lock_fd(resolved)
    deadline = time.monotonic() + max(0.0, timeout)
    interval = _INITIAL_POLL_INTERVAL_S
    acquired = False
    try:
        while True:
            if _try_flock_nonblocking(fd):
                acquired = True
                break
            now = time.monotonic()
            if now >= deadline:
                raise FileLockTimeoutError(
                    f"Timed out after {timeout:.2f}s waiting for file lock '{resolved}'"
                )
            time.sleep(min(interval, max(0.0, deadline - now)))
            interval = min(interval * 1.5, _MAX_POLL_INTERVAL_S)

        with _ACTIVE_LOCKS_GUARD:
            _ACTIVE_LOCKS[key] = [fd, 1]
        try:
            yield
        finally:
            with _ACTIVE_LOCKS_GUARD:
                _ACTIVE_LOCKS.pop(key, None)
    finally:
        if acquired:
            _unlock_and_close(fd)
        else:
            with contextlib.suppress(OSError):
                os.close(fd)


@contextlib.asynccontextmanager
async def async_file_lock(
    lock_path: Path | str,
    *,
    timeout: float = _DEFAULT_TIMEOUT_S,
) -> AsyncIterator[None]:
    """Async counterpart to :func:`file_lock` that yields to the event loop while waiting."""
    resolved = Path(lock_path).resolve()
    key = _owner_key(resolved)

    with _ACTIVE_LOCKS_GUARD:
        existing = _ACTIVE_LOCKS.get(key)
        if existing is not None:
            existing[1] += 1
            reentered = True
        else:
            reentered = False

    if reentered:
        try:
            yield
        finally:
            with _ACTIVE_LOCKS_GUARD:
                entry = _ACTIVE_LOCKS.get(key)
                if entry is not None:
                    entry[1] -= 1
        return

    fd = _open_lock_fd(resolved)
    deadline = time.monotonic() + max(0.0, timeout)
    interval = _INITIAL_POLL_INTERVAL_S
    acquired = False
    try:
        while True:
            if _try_flock_nonblocking(fd):
                acquired = True
                break
            now = time.monotonic()
            if now >= deadline:
                raise FileLockTimeoutError(
                    f"Timed out after {timeout:.2f}s waiting for file lock '{resolved}'"
                )
            await asyncio.sleep(min(interval, max(0.0, deadline - now)))
            interval = min(interval * 1.5, _MAX_POLL_INTERVAL_S)

        with _ACTIVE_LOCKS_GUARD:
            _ACTIVE_LOCKS[key] = [fd, 1]
        try:
            yield
        finally:
            with _ACTIVE_LOCKS_GUARD:
                _ACTIVE_LOCKS.pop(key, None)
    finally:
        if acquired:
            _unlock_and_close(fd)
        else:
            with contextlib.suppress(OSError):
                os.close(fd)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically via a unique sibling temp file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_raw = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_raw)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
        os.replace(tmp_path, path)
    except Exception:
        with contextlib.suppress(OSError):
            tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write ``text`` to ``path`` atomically via a unique sibling temp file."""
    atomic_write_bytes(path, text.encode(encoding))


def _is_process_alive_windows(pid: int) -> bool:  # pragma: no cover - Windows-specific
    if sys.platform != "win32":  # narrows ``ctypes.windll`` for type checkers
        return False
    import ctypes

    process_query_limited_information = 0x1000
    still_active = 259
    handle = ctypes.windll.kernel32.OpenProcess(
        process_query_limited_information,
        False,
        int(pid),
    )
    if not handle:
        return False
    try:
        exit_code = ctypes.c_ulong()
        ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        return bool(ok and exit_code.value == still_active)
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def _proc_stat_state(pid: int) -> str | None:
    """Return the Linux ``/proc/<pid>/stat`` state character, or ``None`` if unavailable."""
    proc_stat = Path(f"/proc/{pid}/stat")
    try:
        stat_text = proc_stat.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "MISSING" if Path("/proc").is_dir() else None
    except OSError:
        return None
    rparen = stat_text.rfind(")")
    if rparen == -1:
        return None
    rest = stat_text[rparen + 1 :].strip()
    return rest.split()[0] if rest else None


def is_process_alive(pid: int) -> bool:
    """Return ``True`` if OS process ``pid`` is currently running and not a zombie."""
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if sys.platform == "win32":  # pragma: no cover - Windows-specific
        return _is_process_alive_windows(pid)
    if _proc_stat_state(pid) in {"MISSING", "Z", "X", "x"}:
        return False
    try:
        os.kill(pid, 0)
    except OSError as exc:
        return exc.errno == errno.EPERM
    return True
