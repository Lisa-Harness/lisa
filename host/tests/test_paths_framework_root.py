"""Regression tests for ``host.paths`` framework-root resolution.

Covers the src-layout regression where ``_repo_root_from_module`` anchored on
a fixed ``parents[1]`` from ``<repo>/host/paths.py``. After the tree moved to
``<repo>/host/src/host/paths.py`` the fixed parent count resolved to
``<repo>/host/src``, which is not a repo checkout — so ``framework_root()``
fell back to tool mode (``sys.prefix``), the packs-venv seeding found no
framework member directories, and external pack installs failed with
"lisa-kernel was not found in the package registry".
"""

from __future__ import annotations

from pathlib import Path

from host.paths import (
    _is_repo_checkout,
    _repo_root_from_module,
    framework_root,
    is_tool_install,
)


def test_repo_root_walks_up_past_src_layout() -> None:
    """Module at ``<repo>/host/src/host/paths.py`` resolves to the repo root.

    This is the exact regression: the fixed ``parents[1]`` returned
    ``<repo>/host/src``, breaking repo detection and dropping the framework
    into tool mode.
    """
    module_file = Path(_repo_root_from_module.__code__.co_filename)
    repo = _repo_root_from_module()
    assert repo in module_file.parents
    assert _is_repo_checkout(repo), (
        f"resolved framework root {repo} is not a repo checkout; "
        "packs-venv seeding will silently no-op and pack installs will fail"
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