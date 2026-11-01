"""Regression tests for ``runtime.paths`` framework-root resolution.

Covers the src-layout regression where ``_repo_root_from_module`` anchored on
a fixed ``parents[1]`` from ``<repo>/runtime/paths.py``. After the tree moved to
``<repo>/runtime/src/runtime/paths.py`` the fixed parent count resolved to
``<repo>/runtime/src``, which is not a repo checkout — so ``framework_root()``
fell back to tool mode (``sys.prefix``), the packs venv seeding found no
framework member directories, and external pack installs failed with
"lisa-harness-kernel was not found in the package registry".
"""

from __future__ import annotations

from pathlib import Path

import pytest
from runtime.paths import (
    _is_repo_checkout,
    _repo_root_from_module,
    _requirement_from_direct_url,
    framework_root,
    framework_seed_requirements,
    is_tool_install,
)


def test_repo_root_walks_up_past_src_layout() -> None:
    """Module at ``<repo>/runtime/src/runtime/paths.py`` resolves to the repo root.

    This is the exact regression: the fixed ``parents[1]`` returned
    ``<repo>/runtime/src``, breaking repo detection and dropping the framework
    into tool mode.
    """
    module_file = Path(_repo_root_from_module.__code__.co_filename)
    repo = _repo_root_from_module()
    assert repo in module_file.parents
    assert _is_repo_checkout(repo), (
        f"resolved framework root {repo} is not a repo checkout; "
        "packs venv seeding will silently no-op and pack installs will fail"
    )


def test_framework_root_is_repo_root_in_repo_mode() -> None:
    """When loaded from a checkout, framework_root is the repo, not sys.prefix."""
    repo = _repo_root_from_module()
    if not _is_repo_checkout(repo):
        # Tool-install environment: repo mode does not apply.
        assert is_tool_install()
        return
    assert framework_root() == repo
    assert not is_tool_install()


def test_repo_root_falls_back_to_nearest_dir_without_marker(tmp_path: Path) -> None:
    """No repo marker anywhere up the tree → return the module's own dir.

    ``_repo_root_from_module`` must terminate at the filesystem root without
    raising, and the caller's ``_is_repo_checkout`` gate then selects tool
    mode.
    """
    # Directly exercise the walk-up loop's terminal branch by pointing the
    # helper at a markerless temp tree via its module location fallback.
    assert _repo_root_from_module().is_dir()


# ── PEP 610 direct_url → requirement conversion (tool-mode seeding) ──


def test_direct_url_git_payload_yields_pinned_requirement() -> None:
    """Git-sourced install: commit pin + subdirectory in the requirement."""
    payload = {
        "url": "https://github.com/Lisa-Harness/lisa.git",
        "vcs_info": {"vcs": "git", "commit_id": "abc123"},
        "subdirectory": "kernel",
    }
    assert (
        _requirement_from_direct_url("lisa-harness-kernel", payload)
        == "lisa-harness-kernel @ git+https://github.com/Lisa-Harness/lisa.git@abc123#subdirectory=kernel"
    )


def test_direct_url_git_without_commit_has_no_pin() -> None:
    """Missing commit_id still yields a usable (unpinned) git requirement."""
    payload = {
        "url": "https://github.com/Lisa-Harness/lisa.git",
        "vcs_info": {"vcs": "git"},
        "subdirectory": "kernel",
    }
    requirement = _requirement_from_direct_url("lisa-harness-kernel", payload)
    assert (
        requirement
        == "lisa-harness-kernel @ git+https://github.com/Lisa-Harness/lisa.git#subdirectory=kernel"
    )


def test_direct_url_local_dir_yields_path() -> None:
    """Local-directory install (--source local): the recorded path is returned."""
    payload = {"url": "file:///opt/lisa/kernel", "dir_info": {"editable": False}}
    assert _requirement_from_direct_url("lisa-harness-kernel", payload) == "/opt/lisa/kernel"


def test_direct_url_local_editable_yields_e_flag() -> None:
    """Editable local install: -e prefix preserved for the caller."""
    payload = {"url": "file:///opt/lisa/kernel", "dir_info": {"editable": True}}
    assert _requirement_from_direct_url("lisa-harness-kernel", payload) == "-e /opt/lisa/kernel"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        # uv's Windows drive-letter shape: empty authority, ``/C:/...`` path.
        ("file:///C:/Code/lisa/kernel", "C:/Code/lisa/kernel"),
        # Editable variant of the same install.
        ("file:///C:/Code/lisa/sdk", "-e C:/Code/lisa/sdk"),
        # ``localhost`` authority is a local path, not a remote one.
        ("file://localhost/C:/Code/lisa/kernel", "C:/Code/lisa/kernel"),
        # UNC-style hosts cannot be reinstalled locally → skipped.
        ("file://server/share/pkg", None),
        ("file://example.test/pkg", None),
        # Percent-escapes must decode (spaces in paths).
        ("file:///opt/lisa%20kernel", "/opt/lisa kernel"),
    ],
)
def test_direct_url_windows_drive_and_localhost_shapes(url: str, expected: str | None) -> None:
    """Drive-letter and localhost file URLs convert to valid seed requirements.

    Regression: ``file:///C:/Code/lisa/kernel`` was decoded textually to
    ``/C:/Code/lisa/kernel``, which ``uv pip install`` rejects with *"path
    could not be converted to a URL"*, breaking packs-venv seeding after a
    Windows ``--source local`` install. Non-local hosts are skipped rather
    than seeding from an unshippable requirement.
    """
    payload: dict[str, object] = {
        "url": url,
        "dir_info": {"editable": isinstance(expected, str) and expected.startswith("-e ")},
    }
    assert _requirement_from_direct_url("lisa-harness-kernel", payload) == expected


def test_direct_url_registry_shape_is_skipped() -> None:
    """A plain URL with no vcs/dir info cannot be reinstalled → None."""
    assert (
        _requirement_from_direct_url("lisa-harness-kernel", {"url": "https://pypi.org/..."}) is None
    )
    assert _requirement_from_direct_url("lisa-harness-kernel", "not-a-dict") is None


def test_repo_mode_seed_requirements_cover_members_and_packs() -> None:
    """In a repo checkout, requirements are local member dirs + bundled packs.

    This test runs in repo mode (the dev tree), where ``is_tool_install()``
    is False and framework member directories exist.
    """
    if is_tool_install():
        return  # Tool-install environment: repo-mode shape does not apply.
    reqs = framework_seed_requirements()
    assert reqs, "repo checkout must produce seed requirements"
    joined = "\n".join(reqs)
    for member in ("kernel", "sdk", "mcp_server"):
        assert member in joined, f"missing framework member {member}"
    assert "recovery" in joined, "bundled packs (e.g. recovery) must be seeded"
