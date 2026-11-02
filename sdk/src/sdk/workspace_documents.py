"""Bounded project documents and protected authority records for trusted packs.

DESIGN -- Domain semantics stay in packs; this backend supplies path safety,
cross-process transactions, compare-and-swap publication, and private records.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import subprocess
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from sdk.filelock import async_file_lock, atomic_write_text, file_lock

logger = logging.getLogger(__name__)

_MAX_DOCUMENT_BYTES = 8 * 1024 * 1024
_MAX_FINGERPRINT_BYTES = 512 * 1024 * 1024


class DocumentConflict(ValueError):
    """A document no longer matches its expected revision."""


@dataclass(frozen=True)
class GitState:
    """Actual repository HEAD, tree, and working-state observations."""

    head: str
    tree: str
    dirty: str


class WorkspaceDocuments:
    """Access project files separately from host-protected authority records.

    Args:
        root: Existing project directory supplied by trusted configuration.
        authority_root: Private host data directory outside the project.

    Raises:
        ValueError: Roots are invalid or private records would enter the project.
    """

    def __init__(self, root: Path, authority_root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.authority_root = authority_root.expanduser().resolve()
        if not self.root.is_dir() or self.authority_root.is_relative_to(self.root):
            raise ValueError("Authority records must be outside an existing project")

    def resolve(self, relative: str, *, private: bool = False) -> Path:
        """Resolve a canonical relative path, rejecting symlinks and traversal.

        Raises:
            ValueError: The path is unsafe or traverses a symlink.
        """
        path = PurePosixPath(relative)
        if (
            not relative
            or path.is_absolute()
            or "\\" in relative
            or any(part in ("", ".", "..", ".git") for part in relative.split("/"))
            or str(path) != relative
        ):
            raise ValueError(f"Unsafe document path: {relative!r}")
        root = self.authority_root if private else self.root
        current = root
        for part in path.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError(f"Symlink document path: {relative!r}")
        return current

    def read(self, relative: str, *, private: bool = False) -> str:
        """Read a bounded UTF-8 document without substituting missing content.

        Raises:
            OSError: The document cannot be read.
            ValueError: The document is too large or the path unsafe.
        """
        path = self.resolve(relative, private=private)
        if path.stat().st_size > _MAX_DOCUMENT_BYTES:
            raise ValueError(f"Document exceeds byte limit: {relative}")
        return path.read_text(encoding="utf-8")

    def exists(self, relative: str, *, private: bool = False) -> bool:
        """Return whether a safe document path names a file."""
        return self.resolve(relative, private=private).is_file()

    def git_state(self) -> GitState:
        """Read fixed Git observations without executing a supplied command.

        Raises:
            ValueError: Git cannot identify the configured repository.
            subprocess.TimeoutExpired: A bounded Git read times out.
        """
        observations = []
        for args in (
            ("rev-parse", "HEAD"),
            ("rev-parse", "HEAD^{tree}"),
            ("status", "--porcelain", "--untracked-files=all"),
        ):
            result = subprocess.run(
                ("git", "-C", str(self.root), *args),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.returncode:
                raise ValueError("Cannot verify Git repository state")
            observations.append(result.stdout.strip())
        return GitState(*observations)

    def list(self, directory: str) -> tuple[str, ...]:
        """List regular non-symlink documents in one project directory."""
        root = self.resolve(directory)
        if not root.exists():
            return ()
        return tuple(
            sorted(
                str(path.relative_to(self.root))
                for path in root.iterdir()
                if path.is_file() and not path.is_symlink()
            )
        )

    def write(
        self,
        relative: str,
        content: str,
        *,
        expected_digest: str,
        private: bool = False,
    ) -> str:
        """Publish a complete document atomically under compare-and-swap.

        Raises:
            DocumentConflict: Existing content differs from the expected digest.
        """
        with self.transaction():
            current = self.digest(relative, private=private)
            if current != expected_digest:
                raise DocumentConflict(f"Document changed: {relative}")
            if len(content.encode()) > _MAX_DOCUMENT_BYTES:
                raise ValueError("Document exceeds byte limit")
            atomic_write_text(self.resolve(relative, private=private), content)
            return self.digest(relative, private=private)

    def digest(self, relative: str, *, private: bool = False) -> str:
        """Hash current document bytes, or identify absence as an empty digest."""
        path = self.resolve(relative, private=private)
        if not path.exists():
            return ""
        if path.stat().st_size > _MAX_FINGERPRINT_BYTES:
            raise ValueError("Artifact exceeds fingerprint byte limit")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def move(self, source: str, target: str, *, expected_digest: str) -> None:
        """Move unchanged content without replacing another historical record.

        Raises:
            DocumentConflict: Source changed or the target already exists.
        """
        with self.transaction():
            if self.digest(source) != expected_digest or self.exists(target):
                raise DocumentConflict("Move conflicts with current document state")
            destination = self.resolve(target)
            destination.parent.mkdir(parents=True, exist_ok=True)
            self.resolve(source).rename(destination)

    @contextlib.contextmanager
    def transaction(self) -> Iterator[None]:
        """Serialize cooperating writers using the existing reentrant file lock."""
        with file_lock(self.authority_root / "workspace.lock"):
            yield

    @contextlib.asynccontextmanager
    async def async_transaction(self) -> AsyncIterator[None]:
        """Serialize async composition without blocking another event-loop task."""
        async with async_file_lock(self.authority_root / "workspace.lock"):
            yield
