"""Tests that ``/consolidate`` reads its explicit-slice budget from HostConfig.

Before the fix, ``run_explicit_slice_command`` re-read the env var via the
legacy ``lisa_config._resolve_slice_tuning`` at command time, bypassing
``HostConfig`` — so the config object, its clamping logic, and its
``LISA_SLICE_BUDGET_MS`` resolution were all dead weight. Now the command
reads ``runtime.config.slice_config.explicit_budget_ms`` (resolved once at
startup), making ``HostConfig`` the single source of truth.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from host.commands._slice import run_explicit_slice_command
from host.testing import build_test_runtime


@pytest.fixture(autouse=True)
def _clean_timezone_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop LISA_TIMEZONE/TZ before each test.

    The full suite's env-propagation tests export ``LISA_TIMEZONE`` into
    ``os.environ`` (compile_mcp_configs writes the host's resolved zone for
    subprocesses). On hosts whose system-local tz is a POSIX alias
    (``EDT`` → ``EST5EDT``) that exported value is not re-resolvable by
    ``HostConfig.from_env``, poisoning every later ``build_test_runtime``
    call (see ``test_policy_command.py`` for the same fixture).
    """
    for env_name in ("LISA_TIMEZONE", "TZ", "TIMEZONE", "LOCAL_TIMEZONE"):
        monkeypatch.delenv(env_name, raising=False)


@pytest.mark.asyncio
async def test_explicit_slice_budget_comes_from_config(tmp_path: Path) -> None:
    """The runtime's default explicit-slice budget is the HostConfig default."""

    runtime = build_test_runtime(
        tmp_path,
        user_slug="user",
        surface_type="personal_dm",
        surface_id="surface",
    )

    assert runtime.config.slice_config.explicit_budget_ms == 2000


@pytest.mark.asyncio
async def test_explicit_slice_command_uses_configured_budget(
    tmp_path: Path,
) -> None:
    """``/consolidate`` passes the configured budget into the SlicePolicy.

    A monkeypatched provider records the policy it was invoked with so the
    test can pin that the budget flowed from ``HostConfig`` (default 2000,
    or the value resolved from ``LISA_SLICE_BUDGET_MS`` at startup).
    """

    runtime = build_test_runtime(
        tmp_path,
        user_slug="user",
        surface_type="personal_dm",
        surface_id="surface",
    )

    recorded: dict[str, int] = {}

    class _RecordingProvider:
        def __init__(self, *, thread_path: object) -> None:
            pass

        async def slice(self, *, ctx: object, policy: object) -> object:
            from kernel.slice import SliceOutcome

            recorded["budget_ms"] = policy.budget_ms
            return SliceOutcome()

    import host.commands._slice as slice_module

    original = slice_module.FileBackedStateProvider
    slice_module.FileBackedStateProvider = _RecordingProvider  # type: ignore[assignment]
    try:
        await run_explicit_slice_command(runtime, message="/consolidate")
    finally:
        slice_module.FileBackedStateProvider = original  # type: ignore[assignment]

    assert recorded["budget_ms"] == runtime.config.slice_config.explicit_budget_ms
    assert recorded["budget_ms"] == 2000
