"""Isolated run directory: data dir, config, fixture repo, and process env."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from string import Template

from dotenv import load_dotenv

from e2etest.scenario import Scenario

_DEFAULT_POLICY = """\
version: 1
mode: ask
sandbox_downgrade: true
bare_requires_approval: false
allow: []
ask: []
deny: []
"""
# Operator location overrides that would point the run back at real state.
_LOCATION_ENV = (
    "LISA_PROFILE",
    "LISA_THREAD_DIR",
    "LISA_SESSION_CONTEXT_PATH",
    "LISA_MCP_LOG_PATH",
)
_GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "lisa-e2e",
    "GIT_AUTHOR_EMAIL": "lisa-e2e@localhost",
    "GIT_COMMITTER_NAME": "lisa-e2e",
    "GIT_COMMITTER_EMAIL": "lisa-e2e@localhost",
}


@dataclass(frozen=True)
class RunPaths:
    """Every location a run touches; all live under ``run_dir``."""

    run_dir: Path
    data_dir: Path
    workspace: Path | None
    remote: Path | None
    initial_main_sha: str

    @property
    def session_context(self) -> Path:
        return self.data_dir / "config" / "session_context.md"


def base_data_dir() -> Path:
    """The operator's real data dir; only models.yaml and .env are read from it."""
    override = os.getenv("LISA_E2E_BASE_DATA_DIR", "").strip()
    return Path(override).expanduser() if override else Path.home() / ".lisa"


def git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, **_GIT_IDENTITY},
    )
    return proc.stdout.strip()


def _seed_fixture(scenario: Scenario, run_dir: Path) -> tuple[Path | None, Path | None, str]:
    if scenario.fixture_repo is None:
        return None, None, ""
    workspace = run_dir / "workspace" / scenario.fixture_repo.name
    shutil.copytree(
        scenario.fixture_repo, workspace, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    git(workspace, "init", "--quiet", "--initial-branch", "main")
    for key, value in (("user.name", "lisa-e2e"), ("user.email", "lisa-e2e@localhost")):
        git(workspace, "config", key, value)
    git(workspace, "add", "-A")
    git(workspace, "commit", "--quiet", "-m", "fixture: initial state")
    remote = run_dir / "remote.git"
    git(run_dir, "init", "--quiet", "--bare", str(remote))
    git(workspace, "remote", "add", "origin", str(remote))
    git(workspace, "push", "--quiet", "origin", "main")
    return workspace, remote, git(workspace, "rev-parse", "HEAD")


def prepare_run(scenario: Scenario, run_dir: Path, packs_venv: Path) -> RunPaths:
    """Create the run dir, seed config + fixture, and point the process env at it."""
    data_dir = run_dir / "data"
    config_dir = data_dir / "config"
    config_dir.mkdir(parents=True)
    base = base_data_dir()
    models = base / "config" / "models.yaml"
    if not models.is_file():
        raise FileNotFoundError(f"base models.yaml not found: {models}")
    shutil.copy2(models, config_dir / "models.yaml")
    policy_text = (
        scenario.policy.read_text(encoding="utf-8") if scenario.policy else _DEFAULT_POLICY
    )
    (config_dir / "policy.yaml").write_text(policy_text, encoding="utf-8")

    workspace, remote, initial_sha = _seed_fixture(scenario, run_dir)
    paths = RunPaths(run_dir, data_dir, workspace, remote, initial_sha)
    template = Template(scenario.session_context.read_text(encoding="utf-8"))
    paths.session_context.write_text(
        template.substitute(
            RUN_DIR=str(run_dir),
            WORKSPACE=str(workspace or ""),
            WORKSPACE_PARENT=str(workspace.parent if workspace else ""),
        ),
        encoding="utf-8",
    )

    load_dotenv(base / ".env", override=False)
    for name in _LOCATION_ENV:
        os.environ.pop(name, None)
    os.environ["LISA_DATA_DIR"] = str(data_dir)
    os.environ["LISA_PACKS_VENV"] = str(packs_venv)
    os.environ.update(_GIT_IDENTITY)
    os.environ.update(scenario.env)
    return paths
