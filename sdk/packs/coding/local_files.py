"""Bounded local filesystem helpers shared by coding pack tools."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

from sdk.packs.coding.config import CodingPackConfig

__all__ = [
    "CodingPathError",
    "excluded_relative_path",
    "iter_source_files",
    "read_text_file",
    "relative_path",
    "resolve_relative_path",
]

_EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
    }
)


class CodingPathError(ValueError):
    """Raised when a local coding request crosses the configured root boundary."""


def resolve_relative_path(
    config: CodingPackConfig,
    requested_path: str,
    *,
    allow_root: bool = False,
) -> Path:
    """Resolve a non-excluded relative path without allowing traversal or symlink escape."""

    root = config.resolve_workspace_root()
    raw = requested_path.strip()
    if not raw or raw == ".":
        if allow_root:
            return root
        raise CodingPathError("A relative path is required.")
    candidate = Path(raw)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise CodingPathError("Path traversal and absolute paths are not allowed.")
    resolved = (root / candidate).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise CodingPathError("Path escapes the configured workspace root.") from exc
    if excluded_relative_path(relative):
        raise CodingPathError("Path is excluded from coding workspace access.")
    return resolved


def relative_path(root: Path, path: Path) -> str:
    """Render a validated workspace-relative path with POSIX separators."""

    return path.resolve().relative_to(root.resolve()).as_posix()


def excluded_relative_path(path: Path) -> bool:
    """Return whether a workspace-relative path belongs to generated/noisy content."""

    return any(part in _EXCLUDED_PARTS for part in path.parts)


def iter_source_files(root: Path) -> Iterable[Path]:
    """Yield non-excluded regular files under ``root`` without following links."""

    root_resolved = root.resolve()
    for directory, child_dirs, child_files in os.walk(root_resolved, followlinks=False):
        directory_path = Path(directory)
        child_dirs[:] = [
            name
            for name in child_dirs
            if not excluded_relative_path((directory_path / name).relative_to(root_resolved))
        ]
        for child_name in child_files:
            candidate = directory_path / child_name
            try:
                resolved = candidate.resolve()
                relative = resolved.relative_to(root_resolved)
            except ValueError:
                continue
            if excluded_relative_path(relative) or not resolved.is_file():
                continue
            yield resolved


def read_text_file(path: Path, *, max_bytes: int) -> tuple[str, bool]:
    """Read a bounded UTF-8 text file and return ``(text, was_truncated)``."""

    with path.open("rb") as handle:
        payload = handle.read(max_bytes + 1)
    if b"\x00" in payload:
        raise CodingPathError("Binary files cannot be read as source text.")
    was_truncated = len(payload) > max_bytes
    text = payload[:max_bytes].decode("utf-8", errors="replace")
    return text, was_truncated