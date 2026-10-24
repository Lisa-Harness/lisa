"""Shared metadata for version-pinned MCP server binaries."""

from __future__ import annotations

import re
from dataclasses import dataclass

_GITHUB_REPOSITORY_RE = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?/"
    r"[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?"
)
_RELEASE_VERSION_RE = re.compile(r"v\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?")
_BINARY_NAME_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


@dataclass(frozen=True, slots=True)
class MCPBinaryDownload:
    """A binary asset hosted in a versioned GitHub release."""

    repository: str
    version: str
    binary_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.repository, str) or not _GITHUB_REPOSITORY_RE.fullmatch(
            self.repository
        ):
            raise ValueError("repository must be a GitHub owner/repository slug")
        if not isinstance(self.version, str) or not _RELEASE_VERSION_RE.fullmatch(self.version):
            raise ValueError("version must be a tagged semantic version such as v1.2.3")
        if not isinstance(self.binary_name, str) or not _BINARY_NAME_RE.fullmatch(self.binary_name):
            raise ValueError("binary_name must be a lowercase executable name")


__all__ = ["MCPBinaryDownload"]
