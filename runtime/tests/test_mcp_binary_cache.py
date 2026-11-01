from __future__ import annotations

import asyncio
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

from runtime import mcp_binary_cache


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
async def test_download_rejects_unresolved_release_version(tmp_path: Path) -> None:
    config = MCPClientConfig(
        server_id="lisa-storage-go",
        command="",
        args=[],
        runtime_kind="binary",
        binary_download=MCPBinaryDownload(
            repository="Lisa-Harness/lisa-go",
            binary_name="lisa-storage",
        ),
        binary_cache_dir=tmp_path / "cache",
    )

    with pytest.raises(MCPBinaryDownloadError, match="no binary release version"):
        await resolve_mcp_binary_configs([config])


@pytest.mark.asyncio
async def test_configs_without_binary_sources_need_no_download() -> None:
    configs = [MCPClientConfig(server_id="host", command="python", args=[])]

    assert await resolve_mcp_binary_configs(configs) is configs


@pytest.mark.asyncio
async def test_downloads_run_concurrently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("runtime.mcp_binary_cache.platform.system", lambda: "Linux")
    monkeypatch.setattr("runtime.mcp_binary_cache.platform.machine", lambda: "x86_64")
    binary = b"pinned Go executable"
    digest = hashlib.sha256(binary).hexdigest()
    asset = "lisa-storage-v0.1.0-linux-amd64"
    in_flight = 0
    second_request_started = asyncio.Event()

    async def handle(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight
        name = request.url.path.rsplit("/", maxsplit=1)[-1]
        if name == "checksums.txt":
            return httpx.Response(200, content=f"{digest}  {asset}\n", request=request)
        if name == asset:
            in_flight += 1
            if in_flight == 2:
                second_request_started.set()
            await second_request_started.wait()
            in_flight -= 1
            return httpx.Response(200, content=binary, request=request)
        return httpx.Response(404, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        resolved = await asyncio.wait_for(
            resolve_mcp_binary_configs(
                [_config(tmp_path / "cache-one"), _config(tmp_path / "cache-two")],
                client=client,
            ),
            timeout=10.0,
        )

    assert second_request_started.is_set()
    assert Path(resolved[0].command).read_bytes() == binary
    assert Path(resolved[1].command).read_bytes() == binary


@pytest.mark.asyncio
async def test_stat_fast_path_skips_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("runtime.mcp_binary_cache.platform.system", lambda: "Linux")
    monkeypatch.setattr("runtime.mcp_binary_cache.platform.machine", lambda: "x86_64")
    binary = b"pinned Go executable"
    digest = hashlib.sha256(binary).hexdigest()
    asset = "lisa-storage-v0.1.0-linux-amd64"
    requests: list[str] = []
    hash_calls: list[Path] = []

    def spy(path: Path) -> str:
        hash_calls.append(path)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    monkeypatch.setattr("runtime.mcp_binary_cache._file_sha256", spy)

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

    assert requests == ["checksums.txt", asset]
    assert hash_calls == []
    assert Path(first[0].command) == Path(second[0].command)
    assert Path(second[0].command).read_bytes() == binary
    cached = Path(second[0].command)
    record = cached.with_name(f"{asset}.sha256").read_text(encoding="ascii")
    assert record.strip().split() == [
        digest,
        str(cached.stat().st_size),
        str(cached.stat().st_mtime_ns),
    ]


@pytest.mark.asyncio
async def test_legacy_digest_sidecar_migrates_to_stat_record(
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
        return httpx.Response(404, request=request)

    config = _config(tmp_path / "cache")
    destination = tmp_path / "cache" / "Lisa-Harness" / "lisa-go" / "v0.1.0" / asset
    destination.parent.mkdir(parents=True)
    destination.write_bytes(binary)
    destination.with_name(f"{asset}.sha256").write_text(digest, encoding="ascii")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        resolved = await resolve_mcp_binary_configs([config], client=client)

    assert requests == []
    assert Path(resolved[0].command) == destination

    stat = destination.stat()
    migrated = destination.with_name(f"{asset}.sha256").read_text(encoding="ascii")
    assert migrated.strip().split() == [
        digest,
        str(stat.st_size),
        str(stat.st_mtime_ns),
    ]

    hash_calls: list[Path] = []

    def spy(path: Path) -> str:
        hash_calls.append(path)
        return digest

    monkeypatch.setattr("runtime.mcp_binary_cache._file_sha256", spy)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        again = await resolve_mcp_binary_configs([config], client=client)

    assert hash_calls == []
    assert Path(again[0].command) == destination


_BINARY = b"pinned Go executable"
_DIGEST = hashlib.sha256(_BINARY).hexdigest()
_ASSET = "lisa-storage-v0.1.0-linux-amd64"


def _linux_amd64(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("runtime.mcp_binary_cache.platform.system", lambda: "Linux")
    monkeypatch.setattr("runtime.mcp_binary_cache.platform.machine", lambda: "x86_64")


def _mock_client(
    checksums_body: bytes | int, binary_body: bytes | int = _BINARY
) -> httpx.AsyncClient:
    def handle(request: httpx.Request) -> httpx.Response:
        name = request.url.path.rsplit("/", maxsplit=1)[-1]
        if name == "checksums.txt":
            if isinstance(checksums_body, int):
                return httpx.Response(checksums_body, request=request)
            return httpx.Response(200, content=checksums_body, request=request)
        if name == _ASSET:
            if isinstance(binary_body, int):
                return httpx.Response(binary_body, request=request)
            return httpx.Response(200, content=binary_body, request=request)
        return httpx.Response(404, request=request)

    return httpx.AsyncClient(transport=httpx.MockTransport(handle))


@pytest.mark.asyncio
async def test_checksums_http_error_raises_download_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _linux_amd64(monkeypatch)
    async with _mock_client(503) as client:
        with pytest.raises(MCPBinaryDownloadError, match="failed to fetch checksums"):
            await resolve_mcp_binary_configs([_config(tmp_path / "cache")], client=client)


@pytest.mark.asyncio
async def test_binary_http_error_raises_download_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _linux_amd64(monkeypatch)
    checksums = f"{_DIGEST}  {_ASSET}\n".encode()
    async with _mock_client(checksums, 404) as client:
        with pytest.raises(MCPBinaryDownloadError, match="failed to download"):
            await resolve_mcp_binary_configs([_config(tmp_path / "cache")], client=client)
    assert not list((tmp_path / "cache").rglob("*.download"))


@pytest.mark.parametrize(
    "checksums",
    [
        pytest.param(b"\xff\xfe binary garbage", id="non-utf8"),
        pytest.param(b"0" * (1024 * 1024 + 1), id="oversized"),
    ],
)
@pytest.mark.asyncio
async def test_invalid_checksums_asset_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checksums: bytes,
) -> None:
    _linux_amd64(monkeypatch)
    expected_error = "not valid UTF-8" if checksums.startswith(b"\xff") else "exceeds"
    async with _mock_client(checksums) as client:
        with pytest.raises(MCPBinaryDownloadError, match=expected_error):
            await resolve_mcp_binary_configs([_config(tmp_path / "cache")], client=client)


@pytest.mark.parametrize(
    "checksums",
    [
        pytest.param(
            b"0" * 64 + b"  lisa-storage-v0.1.0-darwin-arm64\n",
            id="missing-asset",
        ),
        pytest.param(
            f"{_DIGEST}  {_ASSET}\n{_DIGEST}  {_ASSET}\n".encode(),
            id="duplicate-asset",
        ),
        pytest.param(b"not-a-sha  " + _ASSET.encode() + b"\n", id="invalid-digest"),
    ],
)
@pytest.mark.asyncio
async def test_invalid_checksum_entries_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checksums: bytes,
) -> None:
    _linux_amd64(monkeypatch)
    expected_error = (
        "invalid checksum" if checksums.startswith(b"not-a-sha") else "exactly one checksum"
    )
    async with _mock_client(checksums) as client:
        with pytest.raises(MCPBinaryDownloadError, match=expected_error):
            await resolve_mcp_binary_configs([_config(tmp_path / "cache")], client=client)


@pytest.mark.asyncio
async def test_oversized_binary_is_rejected_and_temp_file_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _linux_amd64(monkeypatch)
    monkeypatch.setattr(mcp_binary_cache, "_MAX_BINARY_BYTES", 4)
    async with _mock_client(f"{_DIGEST}  {_ASSET}\n".encode()) as client:
        with pytest.raises(MCPBinaryDownloadError, match="exceeds"):
            await resolve_mcp_binary_configs([_config(tmp_path / "cache")], client=client)
    assert not list((tmp_path / "cache").rglob("*.download"))


def _stage_valid_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _linux_amd64(monkeypatch)
    destination = tmp_path / "cache" / "Lisa-Harness" / "lisa-go" / "v0.1.0" / _ASSET
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_BINARY)
    destination.with_name(f"{_ASSET}.sha256").write_text(_DIGEST, encoding="ascii")
    return destination


@pytest.mark.asyncio
async def test_cached_checksum_garbage_triggers_redownload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = _stage_valid_cache(tmp_path, monkeypatch)
    checksum_path = destination.with_name(f"{_ASSET}.sha256")
    checksum_path.write_text("garbage-not-a-sha", encoding="ascii")

    async with _mock_client(f"{_DIGEST}  {_ASSET}\n".encode()) as client:
        resolved = await resolve_mcp_binary_configs([_config(tmp_path / "cache")], client=client)

    assert Path(resolved[0].command).read_bytes() == _BINARY
    assert checksum_path.read_text(encoding="ascii").strip().split()[0] == _DIGEST


@pytest.mark.asyncio
async def test_cached_checksum_is_normalized_without_network_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = _stage_valid_cache(tmp_path, monkeypatch)
    destination.with_name(f"{_ASSET}.sha256").write_text(f"  {_DIGEST.upper()}\n", encoding="ascii")

    async def fail_if_fetched(*_args, **_kwargs):
        raise AssertionError("a valid cached binary should not require a fetch")

    monkeypatch.setattr("runtime.mcp_binary_cache._read_checksums", fail_if_fetched)

    resolved = await resolve_mcp_binary_configs([_config(tmp_path / "cache")])

    assert Path(resolved[0].command) == destination
    assert destination.read_bytes() == _BINARY


@pytest.mark.asyncio
async def test_symlinked_cached_binary_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = _stage_valid_cache(tmp_path, monkeypatch)
    elsewhere = tmp_path / "elsewhere.bin"
    elsewhere.write_bytes(_BINARY)
    destination.unlink()
    try:
        destination.symlink_to(elsewhere)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    with pytest.raises(MCPBinaryDownloadError, match="refusing symlink"):
        await resolve_mcp_binary_configs([_config(tmp_path / "cache")])


@pytest.mark.asyncio
async def test_directory_at_cache_destination_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = _stage_valid_cache(tmp_path, monkeypatch)
    destination.unlink()
    destination.mkdir()

    with pytest.raises(MCPBinaryDownloadError, match="not a regular file"):
        await resolve_mcp_binary_configs([_config(tmp_path / "cache")])
