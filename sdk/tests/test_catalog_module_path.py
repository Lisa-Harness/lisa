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
            _infer_module_path(py, ops)
            == "sdk.packs.introspect.operations.self_model.self_model"
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
            assert (
                _infer_module_path(py, tools)
                == "lisa_packs.coding.tools.status.status"
            )
        finally:
            sys.path.remove(str(site_dir))