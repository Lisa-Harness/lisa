"""Descriptor-relative, no-overwrite file moves for the storage sandbox."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import stat
import sys
from contextlib import suppress
from functools import lru_cache
from pathlib import Path
from typing import Any

_RENAME_NOREPLACE = 1
_HASH_CHUNK_BYTES = 1024 * 1024
_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
)


class SecureMoveError(RuntimeError):
    """A move refusal or partial failure with JSON-ready details."""

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.details = details


def _platform_supported() -> bool:
    required_dir_fd = (os.open, os.mkdir, os.stat, os.rename)
    return (
        sys.platform.startswith("linux")
        and bool(getattr(os, "O_DIRECTORY", 0))
        and bool(getattr(os, "O_NOFOLLOW", 0))
        and all(fn in os.supports_dir_fd for fn in required_dir_fd)
        and os.stat in os.supports_follow_symlinks
    )


@lru_cache(maxsize=1)
def _libc_renameat2() -> Any:
    """Resolve glibc's no-overwrite rename primitive or fail closed."""
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as exc:
        raise SecureMoveError("move_file is unsupported: renameat2 is unavailable") from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    return renameat2


def _rename_noreplace(
    source_parent_fd: int,
    source_name: str,
    destination_parent_fd: int,
    destination_name: str,
) -> None:
    """Atomically rename one final name without replacing any destination."""
    renameat2 = _libc_renameat2()
    result = renameat2(
        source_parent_fd,
        os.fsencode(source_name),
        destination_parent_fd,
        os.fsencode(destination_name),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise SecureMoveError("move destination already exists")
    if error_number == errno.EXDEV:
        raise SecureMoveError("move crosses filesystems; the source was not changed")
    if error_number in {
        errno.ENOSYS,
        errno.EINVAL,
        errno.ENOTSUP,
        getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
    }:
        raise SecureMoveError("move_file is unsupported: atomic no-overwrite rename is unavailable")
    raise SecureMoveError(f"move failed: {os.strerror(error_number)}")


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _unchanged_before_rename(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        _same_identity(left, right)
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_ctime_ns == right.st_ctime_ns
        and left.st_nlink == right.st_nlink
    )


def _open_parent(
    root_fd: int,
    parts: tuple[str, ...],
    *,
    create: bool,
    label: str,
) -> int:
    """Open a parent directory without following any symlink component."""
    current_fd = os.dup(root_fd)
    traversed: list[str] = []
    try:
        for component in parts:
            traversed.append(component)
            if create:
                with suppress(FileExistsError):
                    os.mkdir(component, mode=0o700, dir_fd=current_fd)
            try:
                next_fd = os.open(component, _DIRECTORY_FLAGS, dir_fd=current_fd)
            except FileNotFoundError as exc:
                raise SecureMoveError(
                    f"{label} parent directory does not exist",
                    component="/".join(traversed),
                ) from exc
            except OSError as exc:
                raise SecureMoveError(
                    f"{label} path component is not a real directory",
                    component="/".join(traversed),
                ) from exc
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _hash_open_file(parent_fd: int, name: str, *, label: str) -> tuple[os.stat_result, str]:
    """Hash one no-follow regular file and prove the inode did not change."""
    try:
        path_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise SecureMoveError(f"{label} file not found") from exc
    except OSError as exc:
        raise SecureMoveError(f"cannot inspect {label} file: {exc}") from exc
    if not stat.S_ISREG(path_stat.st_mode):
        raise SecureMoveError(f"{label} path is not a regular file")
    if path_stat.st_nlink != 1:
        raise SecureMoveError(f"{label} file has hard links and cannot be moved safely")

    try:
        file_fd = os.open(name, _FILE_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        raise SecureMoveError(f"cannot safely open {label} file: {exc}") from exc

    try:
        opened_stat = os.fstat(file_fd)
        if (
            not stat.S_ISREG(opened_stat.st_mode)
            or not _same_identity(path_stat, opened_stat)
            or opened_stat.st_nlink != 1
        ):
            raise SecureMoveError(f"{label} file changed while it was being opened")
        digest = hashlib.sha256()
        while chunk := os.read(file_fd, _HASH_CHUNK_BYTES):
            digest.update(chunk)
        final_stat = os.fstat(file_fd)
        if not _unchanged_before_rename(opened_stat, final_stat):
            raise SecureMoveError(f"{label} file changed while it was being hashed")
        return final_stat, digest.hexdigest()
    except OSError as exc:
        raise SecureMoveError(f"cannot read {label} file safely: {exc}") from exc
    finally:
        os.close(file_fd)


def _stat_relative(parent_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _sync_directories(fds: tuple[int, ...]) -> list[str]:
    failures: list[str] = []
    seen: set[int] = set()
    for fd in fds:
        if fd in seen:
            continue
        seen.add(fd)
        try:
            os.fsync(fd)
        except OSError as exc:
            failures.append(os.strerror(exc.errno or errno.EIO))
    return failures


def move_regular_file(
    root: Path,
    *,
    source_parts: tuple[str, ...],
    destination_parts: tuple[str, ...],
    expected_sha256: str = "",
    create_parents: bool = False,
) -> dict[str, Any]:
    """Move a regular file inside ``root`` using pinned directory handles."""
    if not _platform_supported():
        raise SecureMoveError("move_file is unsupported on this platform")

    try:
        root_fd = os.open(root, _DIRECTORY_FLAGS)
    except OSError as exc:
        raise SecureMoveError(f"cannot safely open the files sandbox: {exc}") from exc

    source_parent_fd = -1
    destination_parent_fd = -1
    try:
        source_parent_fd = _open_parent(root_fd, source_parts[:-1], create=False, label="source")
        source_stat, digest = _hash_open_file(source_parent_fd, source_parts[-1], label="source")
        expected = expected_sha256.strip().lower()
        if expected and expected != digest:
            raise SecureMoveError(
                "file changed since it was read (sha256 mismatch); "
                "re-read and move the current content",
                sha256=digest,
            )

        destination_parent_fd = _open_parent(
            root_fd,
            destination_parts[:-1],
            create=create_parents,
            label="destination",
        )
        _rename_noreplace(
            source_parent_fd,
            source_parts[-1],
            destination_parent_fd,
            destination_parts[-1],
        )

        moved_stat = _stat_relative(destination_parent_fd, destination_parts[-1])
        if (
            moved_stat is None
            or not stat.S_ISREG(moved_stat.st_mode)
            or moved_stat.st_nlink != 1
            or not _same_identity(source_stat, moved_stat)
        ):
            try:
                _rename_noreplace(
                    destination_parent_fd,
                    destination_parts[-1],
                    source_parent_fd,
                    source_parts[-1],
                )
            except SecureMoveError as rollback_exc:
                raise SecureMoveError(
                    "source file identity changed during move; rollback failed",
                    rollback_error=str(rollback_exc),
                ) from rollback_exc
            raise SecureMoveError(
                "source file identity changed during move; the move was rolled back"
            )

        sync_errors = _sync_directories((source_parent_fd, destination_parent_fd))
        receipt: dict[str, Any] = {
            "ok": True,
            "source": "/".join(source_parts),
            "destination": "/".join(destination_parts),
            "bytes": moved_stat.st_size,
            "sha256": digest,
            "durable": not sync_errors,
        }
        if sync_errors:
            receipt["warning"] = "move completed but directory durability sync failed"
            receipt["durability_errors"] = sync_errors
        return receipt
    finally:
        if destination_parent_fd >= 0:
            os.close(destination_parent_fd)
        if source_parent_fd >= 0:
            os.close(source_parent_fd)
        os.close(root_fd)


def _verify_moved_directory(
    source_stat: os.stat_result,
    destination_parent_fd: int,
    destination_name: str,
    source_parent_fd: int,
    source_name: str,
) -> None:
    """Confirm the moved directory kept its inode identity; roll back if not.

    A rename can race with an attacker swapping the source directory
    between the nofollow stat and the rename. If the destination entry
    no longer carries the source's ``(dev, ino)``, the rename is undone
    before the caller reports anything.
    """
    moved_stat = _stat_relative(destination_parent_fd, destination_name)
    if (
        moved_stat is None
        or not stat.S_ISDIR(moved_stat.st_mode)
        or not _same_identity(source_stat, moved_stat)
    ):
        try:
            _rename_noreplace(
                destination_parent_fd,
                destination_name,
                source_parent_fd,
                source_name,
            )
        except SecureMoveError as rollback_exc:
            raise SecureMoveError(
                "source directory identity changed during move; rollback failed",
                rollback_error=str(rollback_exc),
            ) from rollback_exc
        raise SecureMoveError(
            "source directory identity changed during move; the move was rolled back"
        )


def move_directory(
    root: Path,
    *,
    source_parts: tuple[str, ...],
    destination_parts: tuple[str, ...],
    create_parents: bool = False,
) -> dict[str, Any]:
    """Move a directory inside ``root`` using pinned directory handles.

    Mirrors :func:`move_regular_file` without the content hash: the
    source must be a real directory (nofollow stat), the destination
    must not exist, and the whole subtree moves as one atomic rename.
    A destination inside the source subtree is refused — a file move
    cannot have that cycle. Symlinked source components are refused by
    the ``O_NOFOLLOW`` parent walk, and a symlink at the destination
    name is an ``EEXIST``-style refusal, never a traversal.
    """
    if not _platform_supported():
        raise SecureMoveError("move_directory is unsupported on this platform")
    if destination_parts[: len(source_parts)] == source_parts:
        raise SecureMoveError("move destination is inside the source directory")

    try:
        root_fd = os.open(root, _DIRECTORY_FLAGS)
    except OSError as exc:
        raise SecureMoveError(f"cannot safely open the files sandbox: {exc}") from exc

    source_parent_fd = -1
    destination_parent_fd = -1
    try:
        source_parent_fd = _open_parent(root_fd, source_parts[:-1], create=False, label="source")
        try:
            source_stat = os.stat(source_parts[-1], dir_fd=source_parent_fd, follow_symlinks=False)
        except FileNotFoundError as exc:
            raise SecureMoveError("source directory not found") from exc
        except OSError as exc:
            raise SecureMoveError(f"cannot inspect source directory: {exc}") from exc
        if not stat.S_ISDIR(source_stat.st_mode):
            raise SecureMoveError("source path is not a directory")

        destination_parent_fd = _open_parent(
            root_fd,
            destination_parts[:-1],
            create=create_parents,
            label="destination",
        )
        _rename_noreplace(
            source_parent_fd,
            source_parts[-1],
            destination_parent_fd,
            destination_parts[-1],
        )

        _verify_moved_directory(
            source_stat,
            destination_parent_fd,
            destination_parts[-1],
            source_parent_fd,
            source_parts[-1],
        )

        sync_errors = _sync_directories((source_parent_fd, destination_parent_fd))
        receipt: dict[str, Any] = {
            "ok": True,
            "source": "/".join(source_parts),
            "destination": "/".join(destination_parts),
            "durable": not sync_errors,
        }
        if sync_errors:
            receipt["warning"] = "move completed but directory durability sync failed"
            receipt["durability_errors"] = sync_errors
        return receipt
    finally:
        if destination_parent_fd >= 0:
            os.close(destination_parent_fd)
        if source_parent_fd >= 0:
            os.close(source_parent_fd)
        os.close(root_fd)
