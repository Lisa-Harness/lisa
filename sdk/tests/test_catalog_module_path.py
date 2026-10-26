"""Tests for :func:`sdk.content._catalog._infer_module_path`.

Locks in module-path inference for both regular packages (with
``__init__.py``) and PEP 420 namespace packages (no ``__init__.py``).
The namespace case is the regression for the startup failure where
externally installed packs under the ``lisa_packs`` namespace produced
bare, un-importable paths like ``coding.tools.status.status`` because the
shared namespace ``__init__.py`` was absent from site-packages.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from sdk.content._catalog import _infer_module_path, _is_package_dir


class TestIsPackageDir:
    def test_regular_package_with_init(self, tmp_path: Path) -> None:
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "__init__.py").write_text("")
        assert _is_package_dir(tmp_path / "pkg") is True

    def test_plain_directory_without_init(self, tmp_path: Path) -> None:
        (tmp_path / "not_a_pkg").mkdir()
        assert _is_package_dir(tmp_path / "not_a_pkg") is False


class TestReachedPackageTop:
    def test_unrelated_same_named_package_does_not_anchor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A same-named package elsewhere on ``sys.path`` must not anchor the walk.

        Regression for the order-dependent suite failure: targeted runs that
        collect colocated ``src/**/tests`` put the package source dir (e.g.
        ``sdk/src/sdk``) itself on ``sys.path``, so a bare
        ``find_spec('packs')`` resolved to the real repo package while the
        walk visited an unrelated tmp ``packs`` directory. A spec that does
        not cover the walked directory is not evidence of a package top.
        """

        from sdk.content._catalog import _reached_package_top

        # Put a directory on sys.path that contains a *different* ``packs``
        # package with an __init__.py (mirrors the colocated-test pollution).
        other_root = tmp_path / "other-src"
        (other_root / "packs").mkdir(parents=True)
        (other_root / "packs" / "__init__.py").write_text("")
        monkeypatch.syspath_prepend(str(other_root))

        walked_root = tmp_path / "site-packages"
        walked_packs = walked_root / "sdk" / "packs"
        walked_packs.mkdir(parents=True)

        assert _reached_package_top(walked_packs) is False


class TestInferModulePath:
    def test_regular_package_operations_path(self, tmp_path: Path) -> None:
        ops = tmp_path / "sdk" / "packs" / "introspect" / "operations"
        ops.mkdir(parents=True)
        for pkg in ("sdk", "sdk/packs", "sdk/packs/introspect", "sdk/packs/introspect/operations"):
            (tmp_path / pkg / "__init__.py").write_text("")
        py = ops / "self_model" / "self_model.py"
        py.parent.mkdir()
        py.write_text("")
        assert (
            _infer_module_path(py, ops) == "sdk.packs.introspect.operations.self_model.self_model"
        )

    def test_namespace_package_path_without_init(self, tmp_path: Path) -> None:
        # Simulate the real venv layout: lisa_packs/ is a PEP 420 namespace
        # package (no __init__.py at the namespace root) but its pack
        # subpackages (coding) DO have __init__.py, and the tool dirs do not.
        # An ancestor of lisa_packs must be on sys.path so the import system
        # resolves lisa_packs as a namespace package.
        site_dir = tmp_path / "site-packages"
        namespace_root = site_dir / "lisa_packs"
        coding = namespace_root / "coding"
        tools = coding / "tools"
        tools.mkdir(parents=True)
        # Regular subpackage init under the namespace; no namespace __init__.
        (coding / "__init__.py").write_text("")
        py = tools / "status" / "status.py"
        py.parent.mkdir()
        py.write_text("")
        sys.path.insert(0, str(site_dir))
        try:
            assert _is_package_dir(namespace_root) is True
            assert _infer_module_path(py, tools) == "lisa_packs.coding.tools.status.status"
        finally:
            sys.path.remove(str(site_dir))

    def test_walk_terminates_at_filesystem_root(self, tmp_path: Path) -> None:
        """The package-base walk must stop at ``/``, not spin forever.

        Regression for the 2026-09-03 hang: an editable-installed pack
        rooted at ``/home/<user>/<dir>/<pack>/lisa_packs/<name>`` has no
        ``src/`` or ``site-packages`` ancestor, so the walk-up loop ran past
        every boundary into ``/`` where ``Path('/').parent == Path('/')``
        and the leaf name is ``''`` — spinning ``find_spec('')`` at 50k+
        calls/second until the process was killed. The guard must break the
        loop when ``package_dir`` reaches the filesystem root.
        """
        # Lay out a pack with no src/ or site-packages ancestor: the walk
        # climbs tmp_path → tmp_path.parent → … → / and would previously
        # spin there. ``somepack`` is not importable, so no anchor is found
        # and the result falls back to the root-relative parts.
        tools = tmp_path / "somepack" / "tools"
        tools.mkdir(parents=True)
        py = tools / "status" / "status.py"
        py.parent.mkdir()
        py.write_text("")
        assert _infer_module_path(py, tools) == "tools.status.status"

    def test_walk_terminates_for_home_rooted_pack(self, tmp_path: Path) -> None:
        """A real-world home-rooted pack path terminates without hanging.

        Mirrors ``/home/<user>/<repo>/<pack>/lisa_packs/<pack>/operations/…``
        with no ``src/`` or ``site-packages`` anywhere in the ancestor
        chain. The dotted path still includes the ``lisa_packs.<pack>``
        anchor chain when import resolution succeeds; the contract under
        test is that the call **returns** rather than spinning.

        The layout is built under ``tmp_path`` instead of the real home
        directory: the walk has no ``$HOME`` special-casing (it climbs to
        the filesystem root and resolves anchors via ``sys.path``), so a
        deep tmp_path chain exercises the identical code path while keeping
        the test hermetic. The real home dir is read-only in check sandboxes
        and would otherwise be littered on every dev machine.
        """
        scratch = tmp_path / "_lisa_test_walk_home_pack"
        pack_pkg = scratch / "lisa_packs" / "walkhome"
        ops = pack_pkg / "operations"
        ops.mkdir(parents=True)
        (pack_pkg / "__init__.py").write_text("")
        py = ops / "demo" / "demo.py"
        py.parent.mkdir()
        py.write_text("")
        result = _infer_module_path(py, ops)
        # Termination is the contract; the tail (operations.demo.demo)
        # must still be present regardless of how the anchor resolved.
        assert result.endswith("operations.demo.demo")

    def test_site_packages_nested_package_path(self, tmp_path: Path) -> None:
        """Site-packages paths for non-top-level packages resolve full dotted path.

        Regression for startup failure on installed hosts where
        'site-packages' in str(parent) matched on /.../site-packages/sdk/packs
        because 'site-packages' appeared in the string representation of every
        child directory under site-packages, prematurely stopping the walk
        and producing un-importable paths like 'packs.introspect...'.
        """
        site_dir = tmp_path / "venv" / "lib" / "python3.13" / "site-packages"
        ops = site_dir / "sdk" / "packs" / "introspect" / "operations"
        ops.mkdir(parents=True)
        (site_dir / "sdk" / "packs" / "introspect" / "__init__.py").write_text("")
        (site_dir / "sdk" / "packs" / "__init__.py").write_text("")
        py = ops / "self_model" / "self_model.py"
        py.parent.mkdir()
        py.write_text("")

        sys.path.insert(0, str(site_dir))
        try:
            result = _infer_module_path(py, ops)
            assert result == "sdk.packs.introspect.operations.self_model.self_model"
        finally:
            sys.path.remove(str(site_dir))
