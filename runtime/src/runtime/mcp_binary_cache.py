"""Download and cache release-hosted MCP server binaries."""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import re
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote

import httpx
from kernel.mcp_binary import MCPBinaryDownload

if TYPE_CHECKING:
    from kernel.tools.mcp_client import MCPClientConfig

_CHECKSUMS_ASSET = "checksums.txt"
_CHECKSUM_PATTERN = re.compile(r"[0-9a-fA-F]{64}")
_MAX_CHECKSUMS_BYTES = 1024 * 1024
_MAX_BINARY_BYTES = 128 * 1024 * 1024
_CHECKSUM_FIELD_COUNT = 2

logger = logging.getLogger(__name__)


class MCPBinaryDownloadError(RuntimeError):
    """A pinned MCP server binary could not be prepared for launch."""


def _platform_suffix() -> tuple[str, str, str]:
    operating_system = {
        "windows": "windows",
        "linux": "linux",
        "darwin": "darwin",
    }.get(platform.system().lower())
    architecture = {
        "amd64": "amd64",
        "x86_64": "amd64",
        "x64": "amd64",
        "arm64": "arm64",
        "aarch64": "arm64",
    }.get(platform.machine().lower())
    if operating_system is None or architecture is None:
        raise MCPBinaryDownloadError(
            f"unsupported platform for Go MCP binaries: {platform.system()} {platform.machine()}"
        )
    extension = ".exe" if operating_system == "windows" else ""
    return operating_system, architecture, extension


def _release_url(source: MCPBinaryDownload, asset: str) -> str:
    if source.version is None:
        raise MCPBinaryDownloadError(f"{source.binary_name} has no binary release version")
    repository = quote(source.repository, safe="/-._")
    version = quote(source.version, safe="-._")
    asset_name = quote(asset, safe="-._")
    return f"https://github.com/{repository}/releases/download/{version}/{asset_name}"


def _checksum_for_asset(checksums: bytes, asset: str) -> str:
    if len(checksums) > _MAX_CHECKSUMS_BYTES:
        raise MCPBinaryDownloadError(f"{_CHECKSUMS_ASSET} exceeds {_MAX_CHECKSUMS_BYTES} bytes")
    try:
        text = checksums.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MCPBinaryDownloadError(f"{_CHECKSUMS_ASSET} is not valid UTF-8") from exc

    matches: list[str] = []
    for line in text.splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != _CHECKSUM_FIELD_COUNT or fields[1].lstrip("*") != asset:
            continue
        digest = fields[0].lower()
        if not _CHECKSUM_PATTERN.fullmatch(digest):
            raise MCPBinaryDownloadError(
                f"{_CHECKSUMS_ASSET} contains an invalid checksum for {asset}"
            )
        matches.append(digest)
    if len(matches) != 1:
        raise MCPBinaryDownloadError(
            f"{_CHECKSUMS_ASSET} must contain exactly one checksum for {asset}"
        )
    return matches[0]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def _read_checksums(
    client: httpx.AsyncClient,
    source: MCPBinaryDownload,
) -> bytes:
    try:
        content = bytearray()
        async with client.stream("GET", _release_url(source, _CHECKSUMS_ASSET)) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                if len(content) + len(chunk) > _MAX_CHECKSUMS_BYTES:
                    raise MCPBinaryDownloadError(
                        f"{_CHECKSUMS_ASSET} exceeds {_MAX_CHECKSUMS_BYTES} bytes"
                    )
                content.extend(chunk)
    except httpx.HTTPError as exc:
        raise MCPBinaryDownloadError(
            f"failed to fetch {_CHECKSUMS_ASSET} for {source.repository}@{source.version}: {exc}"
        ) from exc
    return bytes(content)


async def _download_binary(
    client: httpx.AsyncClient,
    *,
    source: MCPBinaryDownload,
    asset: str,
    expected_digest: str,
    destination: Path,
) -> None:
    temporary_path: Path | None = None
    digest = hashlib.sha256()
    total_bytes = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{asset}.",
            suffix=".download",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            async with client.stream("GET", _release_url(source, asset)) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    total_bytes += len(chunk)
                    if total_bytes > _MAX_BINARY_BYTES:
                        raise MCPBinaryDownloadError(f"{asset} exceeds {_MAX_BINARY_BYTES} bytes")
                    digest.update(chunk)
                    temporary.write(chunk)
        if temporary_path is None:
            raise MCPBinaryDownloadError(f"failed to create a cache file for {asset}")
        if digest.hexdigest() != expected_digest:
            raise MCPBinaryDownloadError(f"SHA-256 mismatch for {asset}")
        if os.name != "nt":
            temporary_path.chmod(0o755)
        os.replace(temporary_path, destination)
        temporary_path = None
    except (httpx.HTTPError, OSError) as exc:
        raise MCPBinaryDownloadError(f"failed to download {asset}: {exc}") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


async def _ensure_binary(
    client: httpx.AsyncClient,
    config: MCPClientConfig,
) -> MCPClientConfig:
    source = config.binary_download
    if source is None:
        return config
    asset, destination, checksum_path = _download_paths(config, source)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if _cached_binary_is_valid(destination, checksum_path, asset):
        logger.debug(
            "Using verified cached MCP binary %s@%s (%s)",
            source.repository,
            source.version,
            asset,
        )
        return replace(config, command=str(destination))

    logger.info("Downloading MCP binary %s@%s (%s)", source.repository, source.version, asset)
    checksums = await _read_checksums(client, source)
    expected_digest = _checksum_for_asset(checksums, asset)
    await _download_binary(
        client,
        source=source,
        asset=asset,
        expected_digest=expected_digest,
        destination=destination,
    )
    _write_cached_checksum(checksum_path, expected_digest)
    logger.info(
        "Downloaded and verified MCP binary %s@%s (%s)",
        source.repository,
        source.version,
        asset,
    )
    return replace(config, command=str(destination))


def _download_paths(
    config: MCPClientConfig,
    source: MCPBinaryDownload,
) -> tuple[str, Path, Path]:
    if config.runtime_kind != "binary":
        raise MCPBinaryDownloadError(
            f"{config.server_id} declares a binary download but is not a binary server"
        )
    if config.command:
        raise MCPBinaryDownloadError(
            f"{config.server_id} has both a command path and a binary download source"
        )
    if config.binary_cache_dir is None:
        raise MCPBinaryDownloadError(
            f"{config.server_id} has no runtime-managed binary cache directory"
        )
    if source.version is None:
        raise MCPBinaryDownloadError(
            f"{config.server_id} has no binary release version; install its pack from a "
            "vX.Y.Z tag or set binary_download.version"
        )

    operating_system, architecture, extension = _platform_suffix()
    asset = f"{source.binary_name}-{source.version}-{operating_system}-{architecture}{extension}"
    cache_root = Path(config.binary_cache_dir)
    destination = cache_root / source.repository / source.version / asset
    checksum_path = destination.with_name(f"{asset}.sha256")
    return asset, destination, checksum_path


def _cached_binary_is_valid(
    destination: Path,
    checksum_path: Path,
    asset: str,
) -> bool:
    if destination.is_symlink() or checksum_path.is_symlink():
        raise MCPBinaryDownloadError(f"refusing symlink in MCP binary cache for {asset}")
    if destination.exists() and not destination.is_file():
        raise MCPBinaryDownloadError(f"cache destination for {asset} is not a regular file")
    if not destination.is_file() or not checksum_path.is_file():
        return False
    try:
        cached_digest = checksum_path.read_text(encoding="ascii").strip().lower()
    except (OSError, UnicodeError) as exc:
        raise MCPBinaryDownloadError(f"failed to read cached checksum for {asset}: {exc}") from exc
    if not _CHECKSUM_PATTERN.fullmatch(cached_digest):
        return False
    try:
        return _file_sha256(destination) == cached_digest
    except OSError as exc:
        raise MCPBinaryDownloadError(f"failed to verify cached binary {asset}: {exc}") from exc


def _write_cached_checksum(checksum_path: Path, digest: str) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="ascii",
            prefix=f".{checksum_path.name}.",
            suffix=".tmp",
            dir=checksum_path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(digest)
        if temporary_path is None:
            raise MCPBinaryDownloadError(f"failed to create cached checksum {checksum_path.name}")
        os.replace(temporary_path, checksum_path)
        temporary_path = None
    except OSError as exc:
        raise MCPBinaryDownloadError(
            f"failed to cache checksum {checksum_path.name}: {exc}"
        ) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


async def resolve_mcp_binary_configs(
    configs: list[MCPClientConfig],
    *,
    client: httpx.AsyncClient | None = None,
) -> list[MCPClientConfig]:
    """Resolve configured downloads before any MCP server process is spawned."""
    if not any(config.binary_download is not None for config in configs):
        return configs
    if client is not None:
        return [await _ensure_binary(client, config) for config in configs]

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(90.0, connect=10.0),
    ) as download_client:
        return [await _ensure_binary(download_client, config) for config in configs]


__all__ = ["MCPBinaryDownloadError", "resolve_mcp_binary_configs"]
