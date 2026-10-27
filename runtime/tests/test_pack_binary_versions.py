from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest
from kernel.mcp_binary import MCPBinaryDownload
from kernel.mcp_launch.policy import MCPServerRegistry, MCPServerSpec, RuntimeKind
from runtime.capability_servers import CapabilityCandidate, CapabilitySelectionError
from runtime.content_roots import installed_pack_version
from runtime.mcp_pool import (
    _PackManifestSpecs,
    _register_pack_servers,
    _resolve_capability_owners,
    _with_pack_binary_versions,
)


def _install_pack(site_packages: Path, version: str) -> Path:
    root = site_packages / "lisa_packs" / "demo"
    root.mkdir(parents=True)
    dist_info = site_packages / f"lisa_pack_demo-{version}.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: lisa-pack-demo\nVersion: {version}\n",
        encoding="utf-8",
    )
    return root


def _spec(version: str | None = None) -> MCPServerSpec:
    return MCPServerSpec(
        server_id="demo-go",
        runtime_kind=RuntimeKind.BINARY,
        binary_download=MCPBinaryDownload(
            repository="Lisa-Harness/lisa-go",
            binary_name="lisa-storage",
            version=version,
        ),
    )


def test_installed_pack_version_reads_dist_info_beside_pack(tmp_path: Path) -> None:
    root = _install_pack(tmp_path / "site-packages", "0.2.0")

    assert installed_pack_version(root, "lisa-pack-demo") == "0.2.0"


def test_installed_pack_version_falls_back_to_sys_path_for_editable_installs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_packages = tmp_path / "site-packages"
    _install_pack(site_packages, "0.3.1")
    editable_root = tmp_path / "src" / "lisa_packs" / "demo"
    editable_root.mkdir(parents=True)
    monkeypatch.setattr(sys, "path", [str(site_packages), *sys.path])

    assert installed_pack_version(editable_root, "lisa-pack-demo") == "0.3.1"


def test_unversioned_binary_download_uses_pack_release(tmp_path: Path) -> None:
    root = _install_pack(tmp_path / "site-packages", "0.2.0")
    explicit = _spec("v0.1.0")

    resolved = _with_pack_binary_versions([_spec(), explicit], root, "lisa-pack-demo")

    assert resolved[0].binary_download.version == "v0.2.0"
    assert resolved[1] is explicit


@pytest.mark.parametrize(
    "version",
    ["0.2.1.dev3+g1234abc", "0.2.0.post1.dev3+gd492fe7f8", "0.2.1rc1", "0.2.0.post1"],
)
def test_untagged_pack_skips_unversioned_binary_servers(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, version: str
) -> None:
    root = _install_pack(tmp_path / "site-packages", version)
    explicit = _spec("v0.1.0")
    local = MCPServerSpec(
        server_id="local-binary", runtime_kind=RuntimeKind.BINARY, command_path="/bin/demo"
    )

    resolved = _with_pack_binary_versions([_spec(), explicit, local], root, "lisa-pack-demo")

    assert resolved == [explicit, local]
    assert resolved[0] is explicit
    assert resolved[1] is local
    assert "skipping binary MCP servers without an explicit version: demo-go" in caplog.text


def test_missing_pack_version_skips_unversioned_binary_servers(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = tmp_path / "uninstalled-pack"
    root.mkdir()

    assert _with_pack_binary_versions([_spec()], root, "lisa-pack-uninstalled-demo") == []
    assert "has version None" in caplog.text
    assert "demo-go" in caplog.text


@pytest.mark.parametrize("selected", [False, True])
def test_unversioned_pack_server_cannot_be_selected_as_capability_owner(
    tmp_path: Path, selected: bool
) -> None:
    root = _install_pack(tmp_path / "site-packages", "0.2.0.post1.dev3+gd492fe7f8")
    pack_spec = replace(_spec(), capability="storage", capability_tools=("read_file",))
    builtin_spec = MCPServerSpec(
        server_id="lisa-storage",
        runtime_kind=RuntimeKind.VENV_BIN,
        capability="storage",
        capability_tools=("read_file",),
    )
    registry = MCPServerRegistry()
    candidates = [
        CapabilityCandidate(
            capability="storage",
            spec=builtin_spec,
            source="builtin",
            manifest_path="builtin.yaml",
        )
    ]
    _register_pack_servers(
        registry,
        [_PackManifestSpecs(root, "demo", root / "mcp_servers.yaml", [pack_spec])],
        set(),
        candidates,
    )

    assert registry.all_specs() == []
    assert [candidate.spec for candidate in candidates] == [builtin_spec]
    if selected:
        with pytest.raises(CapabilitySelectionError, match="does not match"):
            _resolve_capability_owners(
                registry, candidates, selected_capability_providers={"storage": "demo"}
            )
        assert registry.all_specs() == []
    else:
        _resolve_capability_owners(registry, candidates, selected_capability_providers={})
        assert registry.all_specs() == [builtin_spec]
