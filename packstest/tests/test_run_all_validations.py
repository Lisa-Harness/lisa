"""The parameterized pack-validation runner selects and validates targets."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "run_all_validations.py"
    spec = importlib.util.spec_from_file_location("packstest_run_all_validations", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pack_with_tool(tmp_path: Path, slug: str) -> Path:
    pkg = tmp_path / slug / "lisa_packs" / slug
    pkg.mkdir(parents=True)
    (tmp_path / slug / "pyproject.toml").write_text(
        f'[project]\nname = "lisa-pack-{slug}"\n', encoding="utf-8"
    )
    tools = pkg / "tools" / "hello"
    tools.mkdir(parents=True)
    (tools / "hello.py").write_text(
        "from kernel.models import ToolExecutionContext, ToolResult\n"
        "async def execute(ctx: ToolExecutionContext) -> ToolResult:\n"
        "    return ToolResult(tool_id=ctx.tool_id)\n",
        encoding="utf-8",
    )
    (tools / "hello.tool.yml").write_text(
        "tool_id: hello\ntool_provider_id: lisa-pack-check\ndescription: probe\neffect: read\n",
        encoding="utf-8",
    )
    return tmp_path / slug


def test_worker_validates_pack_subprocess(tmp_path: Path, runner: ModuleType) -> None:
    """One pack checkout validates through the subprocess worker."""
    pack = _pack_with_tool(tmp_path, "check")
    code = runner.main(["--pack", str(pack)])
    assert code == 0, "pack with a healthy tool must validate"


def test_worker_reports_failing_pack(tmp_path: Path, runner: ModuleType) -> None:
    """A broken pack makes the run fail with OVERALL FAILURES PRESENT."""
    broken = _pack_with_tool(tmp_path, "broken")
    module = broken / "lisa_packs" / "broken" / "tools" / "hello" / "hello.py"
    module.write_text("raise SystemExit('boom at import')\n", encoding="utf-8")
    code = runner.main(["--pack", str(broken)])
    assert code == 1


def test_missing_repo_is_a_setup_error(tmp_path: Path, runner: ModuleType) -> None:
    """An unknown --repo exits with the setup code instead of raising."""
    code = runner.main(["--repo", str(tmp_path / "nope")])
    assert code == 2


def test_python_prefers_workspace_venv(runner: ModuleType) -> None:
    """The runner uses the workspace venv python when it exists."""
    src = runner._packstest_src()
    python = runner._python_for(src)
    assert Path(python).is_file()
