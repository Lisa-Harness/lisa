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

from sdk.content._catalog import _infer_module_path, _is_package_dir


class TestIsPackageDir:
    def test_regular_package_with_init(self, tmp_path: Path) -> None:
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "__init__.py").write_text("")
        assert _is_package_dir(tmp_path / "pkg") is True

    def test_plain_directory_without_init(self, tmp_path: Path) -> None:
        (tmp_path / "not_a_pkg").mkdir()
        assert _is_package_dir(tmp_path / "not_a_pkg") is False


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

    def test_walk_terminates_for_home_rooted_pack(self) -> None:
        """A real-world home-rooted pack path terminates without hanging.

        Mirrors ``/home/<user>/<repo>/<pack>/lisa_packs/<pack>/operations/…``
        with no ``src/`` or ``site-packages`` anywhere in the ancestor
        chain. The dotted path still includes the ``lisa_packs.<pack>``
        anchor chain when import resolution succeeds; the contract under
        test is that the call **returns** rather than spinning.
        """
        import shutil

        scratch = Path.home() / "_lisa_test_walk_home_pack"
        pack_pkg = scratch / "lisa_packs" / "walkhome"
        ops = pack_pkg / "operations"
        ops.mkdir(parents=True)
        (pack_pkg / "__init__.py").write_text("")
        py = ops / "demo" / "demo.py"
        py.parent.mkdir()
        py.write_text("")
        try:
            result = _infer_module_path(py, ops)
            # Termination is the contract; the tail (operations.demo.demo)
            # must still be present regardless of how the anchor resolved.
            assert result.endswith("operations.demo.demo")
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
