"""Regression tests for host→subprocess env var propagation in ``compile_mcp_configs``.

Covers the 2026-08-25 incident where ``os.environ.setdefault(...)`` silently
kept a parent-shell empty-string value (e.g. ``LISA_USER_SLUG=""``), causing
the ``lisa-thread`` MCP server to read ``<data_dir>/threads`` instead of
``<data_dir>/threads/<slug>`` and return ``[]`` from ``thread_read_range``.
The host is the authoritative source for these values once resolved from
CLI flags and config, so propagation must overwrite a present-but-empty
env var rather than skip it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import host.mcp_pool as mcp_pool
import host.pack_autoinstall as pack_autoinstall
from host.runtime import HostRuntime


def _make_runtime(
    tmp_path: Path, *, user_slug: str = "testing", no_thread: bool = False
) -> HostRuntime:
    from kernel.testing.runtime_factory import build_test_runtime

    return build_test_runtime(
        tmp_path, models_yaml="", user_slug=user_slug, no_thread=no_thread
    )


def _stub_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``discover_framework`` hermetic: no pack auto-install, no roots."""
    monkeypatch.setattr(
        pack_autoinstall, "auto_install_required_packs", lambda _runtime: []
    )
    monkeypatch.setattr(mcp_pool, "sdk_content_roots", lambda *a, **kw: [])
    monkeypatch.setattr(mcp_pool, "load_pack_server_exclusions", lambda _path: set())


@pytest.mark.asyncio
async def test_compile_mcp_configs_overrides_empty_lisa_user_slug_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A pre-set empty ``LISA_USER_SLUG`` must not suppress the host's slug.

    Reproduces the thread-read-empty root cause: the parent shell exported
    ``LISA_USER_SLUG=""`` and ``setdefault`` left it empty, so the
    lisa-thread server expanded ``${LISA_USER_SLUG}`` to ``""`` and read
    the wrong (user-less) thread store.
    """
    runtime = _make_runtime(tmp_path, user_slug="testing")
    # Pre-set the env vars to empty strings, mirroring a polluted shell.
    monkeypatch.setenv("LISA_THREAD_DIR", "")
    monkeypatch.setenv("LISA_USER_SLUG", "")
    monkeypatch.setenv("FRAMEWORK_ROOT", "")
    monkeypatch.setenv("LISA_TIMEZONE", "")
    _stub_discovery(monkeypatch)

    mcp_pool.compile_mcp_configs(runtime)

    assert os.environ["LISA_USER_SLUG"] == "testing"
    assert os.environ["LISA_THREAD_DIR"] == str(runtime.config.resolved_thread_dir)
    assert os.environ["FRAMEWORK_ROOT"]
    assert os.environ["LISA_TIMEZONE"] == runtime.config.timezone


@pytest.mark.asyncio
async def test_compile_mcp_configs_skips_thread_dir_env_when_no_thread(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """When ``no_thread`` is set, ``LISA_THREAD_DIR`` is not exported.

    ``LISA_USER_SLUG`` is still exported when a user slug is configured,
    because the memory subsystem uses it independently of the thread store.
    """
    runtime = _make_runtime(tmp_path, user_slug="testing", no_thread=True)
    monkeypatch.delenv("LISA_THREAD_DIR", raising=False)
    monkeypatch.delenv("LISA_USER_SLUG", raising=False)
    _stub_discovery(monkeypatch)

    mcp_pool.compile_mcp_configs(runtime)

    assert "LISA_THREAD_DIR" not in os.environ
    assert os.environ["LISA_USER_SLUG"] == "testing"


@pytest.mark.asyncio
async def test_compile_mcp_configs_exports_thread_redaction_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The thread redaction env contract reaches the server subprocess.

    The lisa-thread server reads ``LISA_THREAD_REDACTION_ENABLED`` /
    ``LISA_THREAD_REDACTION_PATTERNS`` to build its redaction policy. The
    host is the authoritative source (the same fields
    ``host.config`` resolves for in-process reads); pre-set empty
    shell values must not leak, and unset custom patterns must be popped
    so the server can fall back to framework defaults.
    """
    # Set AFTER nothing: HostConfig.from_env reads these vars at construction,
    # so a polluted patterns var would leak into runtime.config. Set the
    # polluted shell value FIRST, then construct → config picks it up →
    # the export path must still be correct. For the defaults case, start
    # from a clean patterns var so the runtime resolves defaults.
    monkeypatch.setenv("LISA_THREAD_REDACTION_ENABLED", "")
    monkeypatch.delenv("LISA_THREAD_REDACTION_PATTERNS", raising=False)

    runtime = _make_runtime(tmp_path, user_slug="testing")
    _stub_discovery(monkeypatch)

    mcp_pool.compile_mcp_configs(runtime)

    # Default config: enabled with no custom patterns → "1" and the
    # patterns var is popped (server falls back to DEFAULT_REDACTION_PATTERNS).
    assert os.environ["LISA_THREAD_REDACTION_ENABLED"] == "1"
    assert "LISA_THREAD_REDACTION_PATTERNS" not in os.environ


@pytest.mark.asyncio
async def test_compile_mcp_configs_exports_custom_redaction_patterns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Configured custom patterns are newline-joined into the env var."""
    monkeypatch.setenv("LISA_THREAD_REDACTION_PATTERNS", "stale-shell-pattern")

    runtime = _make_runtime(tmp_path, user_slug="testing")
    # Rebuild the runtime config with custom patterns (HostConfig is frozen;
    # ``runtime.config`` is a plain attribute so this assignment sticks).
    import dataclasses  # noqa: PLC0415

    runtime.config = dataclasses.replace(
        runtime.config,
        thread_redaction_patterns=(r"leaked_secret_\d+", r"stale_warning[^\n]*"),
    )
    _stub_discovery(monkeypatch)

    mcp_pool.compile_mcp_configs(runtime)

    assert (
        os.environ["LISA_THREAD_REDACTION_PATTERNS"]
        == "leaked_secret_\\d+\nstale_warning[^\\n]*"
    )