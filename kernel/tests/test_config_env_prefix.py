"""``kernel.config.Settings`` optional ``LISA_`` env-var prefix.

Shell (global) env vars accept an optional ``LISA_`` prefix
(``LISA_STAGE_NUDGE_CAP`` and ``STAGE_NUDGE_CAP`` are both honored);
``.env`` files keep the bare unprefixed names only.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kernel.config import Settings


# ── Helpers ────────────────────────────────────────────────────────────

_LISA_PREFIXED = {
    "LISA_STAGE_NUDGE_CAP",
    "LISA_MAX_TOOL_ITERATIONS",
    "LISA_BUDGET_BRIEFING_ENABLED",
}
_BARE = {"STAGE_NUDGE_CAP", "MAX_TOOL_ITERATIONS", "BUDGET_BRIEFING_ENABLED"}
_ALL = _LISA_PREFIXED | _BARE


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip any LISA_/bare settings vars that leak from the host shell."""

    for var in _ALL:
        monkeypatch.delenv(var, raising=False)


# ── Shell env: bare and LISA_ prefixed ──────────────────────────────────


def test_bare_env_overrides_default() -> None:
    """Bare ``STAGE_NUDGE_CAP`` is honored in the live environment."""

    os.environ["STAGE_NUDGE_CAP"] = "9"
    try:
        assert Settings().stage_nudge_cap == 9
    finally:
        del os.environ["STAGE_NUDGE_CAP"]


def test_lisa_prefixed_env_overrides_default() -> None:
    """``LISA_STAGE_NUDGE_CAP`` is honored in the live environment."""

    os.environ["LISA_STAGE_NUDGE_CAP"] = "11"
    try:
        assert Settings().stage_nudge_cap == 11
    finally:
        del os.environ["LISA_STAGE_NUDGE_CAP"]


def test_bare_env_wins_over_lisa_prefixed() -> None:
    """Bare name takes precedence over the ``LISA_``-prefixed name."""

    os.environ["STAGE_NUDGE_CAP"] = "7"
    os.environ["LISA_STAGE_NUDGE_CAP"] = "13"
    try:
        assert Settings().stage_nudge_cap == 7
    finally:
        del os.environ["STAGE_NUDGE_CAP"]
        del os.environ["LISA_STAGE_NUDGE_CAP"]


def test_lisa_prefix_not_required() -> None:
    """Without any prefix, the bare name still works (back-compat)."""

    os.environ["MAX_TOOL_ITERATIONS"] = "123"
    try:
        assert Settings().max_tool_iterations == 123
    finally:
        del os.environ["MAX_TOOL_ITERATIONS"]


def test_lisa_prefix_works_for_bool_field() -> None:
    """``LISA_`` prefix applies to bool fields, not just ints."""

    os.environ["LISA_BUDGET_BRIEFING_ENABLED"] = "false"
    try:
        assert Settings().budget_briefing_enabled is False
    finally:
        del os.environ["LISA_BUDGET_BRIEFING_ENABLED"]


# ── .env files: bare names only ────────────────────────────────────────


def test_env_file_bare_name_honored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``.env`` files use bare unprefixed names."""

    (tmp_path / ".env").write_text("STAGE_NUDGE_CAP=5\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert Settings().stage_nudge_cap == 5


def test_env_file_lisa_prefix_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``LISA_``-prefixed keys in ``.env`` are NOT honored (bare-only)."""

    (tmp_path / ".env").write_text("LISA_STAGE_NUDGE_CAP=99\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    # Default 9 — the prefixed .env entry is ignored.
    assert Settings().stage_nudge_cap == 9


def test_env_file_does_not_shadow_bare_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bare shell env beats a .env file (standard pydantic precedence)."""

    (tmp_path / ".env").write_text("STAGE_NUDGE_CAP=5\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STAGE_NUDGE_CAP", "8")
    assert Settings().stage_nudge_cap == 8