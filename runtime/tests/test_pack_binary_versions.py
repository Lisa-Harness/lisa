from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from kernel.mcp_binary import MCPBinaryDownload
from kernel.mcp_launch.policy import MCPServerRegistry, MCPServerSpec, RuntimeKind
from runtime.capability_servers import CapabilityCandidate, CapabilitySelectionError
from runtime.content_roots import installed_pack_requested_revision, installed_pack_version
from runtime.mcp_pool import (
    _PackManifestSpecs,
    _register_pack_servers,
    _resolve_capability_owners,
    _SkippedCandidate,
    _with_pack_binary_versions,
)


def _install_pack(
    site_packages: Path, version: str, *, direct_url: dict[str, object] | str | None = None
) -> Path:
    root = site_packages / "lisa_packs" / "demo"
    root.mkdir(parents=True)
    dist_info = site_packages / f"lisa_pack_demo-{version}.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: lisa-pack-demo\nVersion: {version}\n",
        encoding="utf-8",
    )
    if direct_url is not None:
        (dist_info / "direct_url.json").write_text(
            direct_url if isinstance(direct_url, str) else json.dumps(direct_url),
            encoding="utf-8",
        )
    return root


def _git_install(revision: str) -> dict[str, object]:
    return {
        "url": "https://github.com/Lisa-Harness/lisa-go.git",
        "vcs_info": {
            "vcs": "git",
            "commit_id": "dc25023348a36d1900409c7198b09b18ab3bd3e8",
            "requested_revision": revision,
        },
        "subdirectory": "pack",
    }


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

    assert resolved.specs[0].binary_download.version == "v0.2.0"
    assert resolved.specs[1] is explicit
    assert resolved.skipped == []


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

    unversioned = _spec()

    resolved = _with_pack_binary_versions([unversioned, explicit, local], root, "lisa-pack-demo")

    assert resolved.specs == [explicit, local]
    assert resolved.specs[0] is explicit
    assert resolved.specs[1] is local
    assert resolved.skipped == [unversioned]
    assert repr(version) in resolved.skip_reason
    assert "skipping binary MCP servers without an explicit version: demo-go" in caplog.text


def test_missing_pack_version_skips_unversioned_binary_servers(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = tmp_path / "uninstalled-pack"
    root.mkdir()

    resolved = _with_pack_binary_versions([_spec()], root, "lisa-pack-uninstalled-demo")

    assert resolved.specs == []
    assert [spec.server_id for spec in resolved.skipped] == ["demo-go"]
    assert "has version None" in caplog.text
    assert "demo-go" in caplog.text


@pytest.mark.parametrize(
    ("revision", "version", "release"),
    [
        # The tag moved after uv cached the clone, so the build backend
        # derived a post-release version; the requested tag still names it.
        ("v0.2.2", "0.2.2.post1.dev1+gdc2502334", "v0.2.2"),
        # The checkout predates the tag in the cached clone entirely.
        ("v0.2.1", "0.2.0.post1.dev3+gd492fe7f8", "v0.2.1"),
        ("0.3.0", "0.3.0", "v0.3.0"),
        # A branch install falls back to the distribution version.
        ("main", "0.2.0", "v0.2.0"),
        ("v0.2.2-rc1", "0.2.2", "v0.2.2"),
    ],
)
def test_requested_release_tag_selects_binary_release(
    tmp_path: Path, revision: str, version: str, release: str
) -> None:
    root = _install_pack(tmp_path / "site-packages", version, direct_url=_git_install(revision))

    resolved = _with_pack_binary_versions([_spec()], root, "lisa-pack-demo")

    assert [spec.binary_download.version for spec in resolved.specs] == [release]
    assert resolved.skipped == []


@pytest.mark.parametrize("revision", ["main", "d492fe7f8", "v0.2.2-rc1", "release-0.2.2"])
def test_non_release_ref_with_dev_version_skips_binary_servers(
    tmp_path: Path, revision: str
) -> None:
    root = _install_pack(
        tmp_path / "site-packages",
        "0.2.2.post1.dev1+gdc2502334",
        direct_url=_git_install(revision),
    )

    resolved = _with_pack_binary_versions([_spec()], root, "lisa-pack-demo")

    assert resolved.specs == []
    assert [spec.server_id for spec in resolved.skipped] == ["demo-go"]


@pytest.mark.parametrize(
    ("direct_url", "expected"),
    [
        (None, None),
        ({"url": "file:///home/adam/Code/lisa-go/pack", "dir_info": {}}, None),
        ({"url": "https://example.test/x.git", "vcs_info": {"vcs": "git"}}, None),
        ("{not json", None),
        ("[]", None),
        (_git_install("v0.2.2"), "v0.2.2"),
    ],
)
def test_installed_pack_requested_revision_reads_direct_url(
    tmp_path: Path, direct_url: dict[str, object] | str | None, expected: str | None
) -> None:
    root = _install_pack(tmp_path / "site-packages", "0.2.2", direct_url=direct_url)

    assert installed_pack_requested_revision(root, "lisa-pack-demo") == expected


def _storage_selection_fixture(
    tmp_path: Path,
) -> tuple[MCPServerRegistry, list[CapabilityCandidate], list[_SkippedCandidate], MCPServerSpec]:
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
    skipped: list[_SkippedCandidate] = []
    _register_pack_servers(
        registry,
        [_PackManifestSpecs(root, "demo", root / "mcp_servers.yaml", [pack_spec])],
        set(),
        candidates,
        skipped,
    )
    return registry, candidates, skipped, builtin_spec


def test_unversioned_pack_server_loses_to_builtin_without_selection(tmp_path: Path) -> None:
    registry, candidates, skipped, builtin_spec = _storage_selection_fixture(tmp_path)

    assert registry.all_specs() == []
    assert [candidate.spec for candidate in candidates] == [builtin_spec]
    assert [entry.candidate.spec.server_id for entry in skipped] == ["demo-go"]

    _resolve_capability_owners(
        registry, candidates, selected_capability_providers={}, skipped_candidates=skipped
    )

    assert registry.all_specs() == [builtin_spec]


@pytest.mark.parametrize("selection", ["demo", "demo-go", "lisa-pack-demo"])
def test_selecting_skipped_pack_server_explains_why(tmp_path: Path, selection: str) -> None:
    registry, candidates, skipped, _ = _storage_selection_fixture(tmp_path)

    with pytest.raises(CapabilitySelectionError) as excinfo:
        _resolve_capability_owners(
            registry,
            candidates,
            selected_capability_providers={"storage": selection},
            skipped_candidates=skipped,
        )

    message = str(excinfo.value)
    assert "does not match" in message
    assert "demo-go was skipped: pack lisa-pack-demo version" in message
    assert "'0.2.0.post1.dev3+gd492fe7f8' is not a release" in message
    assert registry.all_specs() == []


def test_unrelated_selection_error_is_not_annotated(tmp_path: Path) -> None:
    registry, candidates, skipped, _ = _storage_selection_fixture(tmp_path)

    with pytest.raises(CapabilitySelectionError) as excinfo:
        _resolve_capability_owners(
            registry,
            candidates,
            selected_capability_providers={"storage": "other-pack"},
            skipped_candidates=skipped,
        )

    assert "was skipped" not in str(excinfo.value)


def test_excluded_pack_server_is_not_reported_as_skipped(tmp_path: Path) -> None:
    root = _install_pack(tmp_path / "site-packages", "0.2.0.post1.dev3+gd492fe7f8")
    pack_spec = replace(_spec(), capability="storage", capability_tools=("read_file",))
    skipped: list[_SkippedCandidate] = []

    _register_pack_servers(
        MCPServerRegistry(),
        [_PackManifestSpecs(root, "demo", root / "mcp_servers.yaml", [pack_spec])],
        {"demo.demo-go"},
        [],
        skipped,
    )

    assert skipped == []
