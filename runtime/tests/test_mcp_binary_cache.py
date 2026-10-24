from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
import pytest
from kernel.mcp_binary import MCPBinaryDownload
from kernel.tools.mcp_client import MCPClientConfig
from runtime.mcp_binary_cache import (
    MCPBinaryDownloadError,
    resolve_mcp_binary_configs,
)


def _config(cache_dir: Path) -> MCPClientConfig:
    return MCPClientConfig(
        server_id="lisa-storage-go",
        command="",
        args=[],
        runtime_kind="binary",
        binary_download=MCPBinaryDownload(
            repository="Lisa-Harness/lisa-go",
            version="v0.1.0",
            binary_name="lisa-storage",
        ),
        binary_cache_dir=cache_dir,
    )


@pytest.mark.asyncio
async def test_download_verifies_and_reuses_cached_binary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("runtime.mcp_binary_cache.platform.system", lambda: "Linux")
    monkeypatch.setattr("runtime.mcp_binary_cache.platform.machine", lambda: "x86_64")
    binary = b"pinned Go executable"
    digest = hashlib.sha256(binary).hexdigest()
    asset = "lisa-storage-v0.1.0-linux-amd64"
    requests: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        name = request.url.path.rsplit("/", maxsplit=1)[-1]
        requests.append(name)
        if name == "checksums.txt":
            return httpx.Response(200, content=f"{digest}  {asset}\n", request=request)
        if name == asset:
            return httpx.Response(200, content=binary, request=request)
        return httpx.Response(404, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        first = await resolve_mcp_binary_configs([_config(tmp_path / "cache")], client=client)
        second = await resolve_mcp_binary_configs([_config(tmp_path / "cache")], client=client)

    assert first[0].command == second[0].command
    assert Path(first[0].command).read_bytes() == binary
    assert requests == ["checksums.txt", asset]

    Path(first[0].command).write_bytes(b"tampered cache")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        repaired = await resolve_mcp_binary_configs([_config(tmp_path / "cache")], client=client)

    assert Path(repaired[0].command).read_bytes() == binary
    assert requests == ["checksums.txt", asset, "checksums.txt", asset]


@pytest.mark.asyncio
async def test_checksum_mismatch_does_not_leave_a_cached_binary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("runtime.mcp_binary_cache.platform.system", lambda: "Linux")
    monkeypatch.setattr("runtime.mcp_binary_cache.platform.machine", lambda: "amd64")
    asset = "lisa-storage-v0.1.0-linux-amd64"

    def handle(request: httpx.Request) -> httpx.Response:
        name = request.url.path.rsplit("/", maxsplit=1)[-1]
        if name == "checksums.txt":
            return httpx.Response(200, content=f"{'0' * 64}  {asset}\n", request=request)
        return httpx.Response(200, content=b"not the pinned binary", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(MCPBinaryDownloadError, match="SHA-256 mismatch"):
            await resolve_mcp_binary_configs([_config(tmp_path / "cache")], client=client)

    assert not list((tmp_path / "cache").rglob(asset))
    assert not list((tmp_path / "cache").rglob("*.download"))


@pytest.mark.asyncio
async def test_download_rejects_unsupported_platform(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("runtime.mcp_binary_cache.platform.system", lambda: "Plan9")
    monkeypatch.setattr("runtime.mcp_binary_cache.platform.machine", lambda: "amd64")

    with pytest.raises(MCPBinaryDownloadError, match="unsupported platform"):
        await resolve_mcp_binary_configs([_config(tmp_path / "cache")])


@pytest.mark.asyncio
async def test_configs_without_binary_sources_need_no_download() -> None:
    configs = [MCPClientConfig(server_id="host", command="python", args=[])]

    assert await resolve_mcp_binary_configs(configs) is configs
