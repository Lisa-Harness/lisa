"""File-lock and atomic-write failure paths must preserve data and descriptors."""

from __future__ import annotations

import asyncio
import errno
import os
import sys
from pathlib import Path

import pytest

from sdk import filelock


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX advisory lock errors")
@pytest.mark.parametrize("error", [errno.EACCES, errno.EAGAIN, errno.EIO])
def test_advisory_lock_distinguishes_contention_from_io_failure(
    monkeypatch: pytest.MonkeyPatch, error: int
) -> None:
    def fail(*args: object) -> None:
        raise OSError(error, "injected lock error")

    monkeypatch.setattr(filelock.fcntl, "flock", fail)
    if error == errno.EIO:
        with pytest.raises(OSError) as raised:
            filelock._try_flock_nonblocking(-1)
        assert raised.value.errno == error
    else:
        assert not filelock._try_flock_nonblocking(-1)


def test_failed_atomic_replace_preserves_original_and_removes_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "state.json"
    target.write_text("original", encoding="utf-8")

    def fail(*args: object) -> None:
        raise OSError(errno.EIO, "replace failed")

    monkeypatch.setattr(filelock.os, "replace", fail)
    with pytest.raises(OSError, match="replace failed"):
        filelock.atomic_write_bytes(target, b"replacement")
    assert target.read_text(encoding="utf-8") == "original"
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_failed_async_acquisition_closes_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancelled: bool
) -> None:
    closed: list[int] = []
    close = os.close

    def record_close(fd: int) -> None:
        closed.append(fd)
        close(fd)

    async def cancel(*args: object) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(filelock, "_try_flock_nonblocking", lambda fd: False)
    monkeypatch.setattr(filelock.os, "close", record_close)
    if cancelled:
        monkeypatch.setattr(filelock.asyncio, "sleep", cancel)
    error = asyncio.CancelledError if cancelled else filelock.FileLockTimeoutError
    with pytest.raises(error):
        async with filelock.async_file_lock(tmp_path / "busy.lock", timeout=10 if cancelled else 0):
            pytest.fail("an unacquired lock must never yield")
    assert len(closed) == 1
    with pytest.raises(OSError) as raised:
        os.fstat(closed[0])
    assert raised.value.errno == errno.EBADF


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("123 (name with ) parentheses) S 1 2", "S"),
        ("invalid", None),
        ("123 (name)", None),
        (PermissionError(), None),
        (FileNotFoundError(), "MISSING"),
    ],
)
def test_proc_stat_handles_names_and_unreadable_records(
    monkeypatch: pytest.MonkeyPatch, content: str | OSError, expected: str | None
) -> None:
    def read(*args: object, **kwargs: object) -> str:
        if isinstance(content, OSError):
            raise content
        return content

    monkeypatch.setattr(filelock.Path, "read_text", read)
    monkeypatch.setattr(filelock.Path, "is_dir", lambda path: True)
    assert filelock._proc_stat_state(123) == expected


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal-zero liveness check")
@pytest.mark.parametrize(
    ("error", "expected"), [(None, True), (errno.EPERM, True), (errno.ESRCH, False)]
)
def test_pid_probe_distinguishes_permission_denial_from_missing_process(
    monkeypatch: pytest.MonkeyPatch, error: int | None, expected: bool
) -> None:
    monkeypatch.setattr(filelock, "_proc_stat_state", lambda pid: None)

    def kill(pid: int, signal: int) -> None:
        assert signal == 0
        if error is not None:
            raise OSError(error, "probe failed")

    monkeypatch.setattr(filelock.os, "kill", kill)
    assert filelock.is_process_alive(os.getpid() + 100000) is expected
    assert not filelock.is_process_alive(0)
