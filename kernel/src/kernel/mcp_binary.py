"""Shared metadata for release-hosted MCP server binaries."""

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
    """A binary asset hosted in a versioned GitHub release.

    ``version`` is ``None`` when the declaring pack leaves the release to its
    own installed version; the host fills it in before download.
    """

    repository: str
    binary_name: str
    version: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.repository, str) or not _GITHUB_REPOSITORY_RE.fullmatch(
            self.repository
        ):
            raise ValueError("repository must be a GitHub owner/repository slug")
        if self.version is not None and not is_release_version(self.version):
            raise ValueError("version must be a tagged semantic version such as v1.2.3")
        if not isinstance(self.binary_name, str) or not _BINARY_NAME_RE.fullmatch(self.binary_name):
            raise ValueError("binary_name must be a lowercase executable name")


def is_release_version(version: object) -> bool:
    """Return whether ``version`` is a ``vX.Y.Z`` release tag."""
    return isinstance(version, str) and _RELEASE_VERSION_RE.fullmatch(version) is not None


__all__ = ["MCPBinaryDownload", "is_release_version"]
