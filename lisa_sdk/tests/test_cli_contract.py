"""Tests for the shared CLI argument contract across all first-party servers.

Verifies that every server:
- Requires ``--data-dir`` (exits with code 2 when missing)
- Accepts ``--data-dir``, ``--transport``, ``--log-level``
- Derives paths from ``--data-dir`` (no env-var fallback)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# All first-party servers that must follow the CLI contract.
# (module, server_name, supports_readonly)
ALL_SERVERS: list[tuple[str, str, bool]] = [
    ("lisa_sdk.lisa_context.server", "lisa-context", True),
    ("lisa_sdk.lisa_memory.server", "lisa-memory", True),
    ("lisa_sdk.lisa_search.server", "lisa-search", False),
    ("lisa_sdk.lisa_surf.server", "lisa-surf", False),
    ("lisa_sdk.lisa_guard.server", "lisa-guard", False),
    ("lisa_sdk.lisa_storage.server", "lisa-storage", True),
    ("lisa_sdk.lisa_content.server", "lisa-content", False),
]


def _run_server(module: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run a server module with the given args and capture the result."""
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        capture_output=True,
        text=True,
        timeout=10,
    )


@pytest.mark.parametrize("module,server_name,supports_readonly", ALL_SERVERS)
def test_server_requires_data_dir(module: str, server_name: str, supports_readonly: bool) -> None:
    """Every server must exit with code 2 when --data-dir is missing."""
    result = _run_server(module)
    assert result.returncode == 2, (
        f"{module} should exit with code 2 when --data-dir is missing, "
        f"got {result.returncode}. stderr: {result.stderr}"
    )
    assert "--data-dir" in result.stderr, f"{module} should mention --data-dir in the error message"


@pytest.mark.parametrize("module,server_name,supports_readonly", ALL_SERVERS)
def test_server_accepts_standard_args(
    module: str, server_name: str, supports_readonly: bool, tmp_path: Path
) -> None:
    """Every server must accept --data-dir, --transport, --log-level."""
    # We can't fully run the server (it blocks on stdio), but we can verify
    # argparse doesn't reject these args by checking it doesn't exit with
    # code 2 for unknown args. We use --help which exits with code 0.
    result = _run_server(module, "--help")
    assert result.returncode == 0, (
        f"{module} --help should exit 0, got {result.returncode}. stderr: {result.stderr}"
    )
    assert "--data-dir" in result.stdout
    assert "--transport" in result.stdout
    assert "--log-level" in result.stdout


@pytest.mark.parametrize("module,server_name,supports_readonly", ALL_SERVERS)
def test_server_help_lists_server_name(
    module: str, server_name: str, supports_readonly: bool
) -> None:
    """The --help output should show the server name as prog."""
    result = _run_server(module, "--help")
    assert server_name in result.stdout, f"{module} --help should show '{server_name}' as prog name"


def test_guard_server_has_backend_arg() -> None:
    """lisa-guard should accept --backend in addition to standard args."""
    result = _run_server("lisa_sdk.lisa_guard.server", "--help")
    assert result.returncode == 0
    assert "--backend" in result.stdout


def test_storage_server_has_readonly_arg() -> None:
    """lisa-storage should accept --readonly."""
    result = _run_server("lisa_sdk.lisa_storage.server", "--help")
    assert result.returncode == 0
    assert "--readonly" in result.stdout


def test_context_server_has_readonly_arg() -> None:
    """lisa-context should accept --readonly."""
    result = _run_server("lisa_sdk.lisa_context.server", "--help")
    assert result.returncode == 0
    assert "--readonly" in result.stdout


def test_no_env_var_path_fallback_in_source() -> None:
    """No first-party server source should reference removed env vars."""

    src_root = Path(__file__).resolve().parents[1] / "src"
    forbidden = {"LISA_MCP_DB_PATH", "LISA_MEMORY_DB_PATH", "LISA_STORAGE_ROOT", "LISA_DATA_DIR"}
    violations: list[str] = []

    for py_file in src_root.rglob("*.py"):
        source = py_file.read_text(encoding="utf-8")
        # Simple string search is sufficient — these are env var names
        # that should never appear in source.
        for var in forbidden:
            if var in source:
                violations.append(f"{py_file}: contains '{var}'")

    assert not violations, "Removed env vars found in server source:\n" + "\n".join(violations)
