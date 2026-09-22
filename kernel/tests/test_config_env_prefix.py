"""``kernel.config.Settings`` optional ``LISA_`` env-var prefix.

Shell (global) env vars accept an optional ``LISA_`` prefix
(``LISA_STAGE_NUDGE_CAP`` and ``STAGE_NUDGE_CAP`` are both honored).
Settings load from the live environment only — ``.env`` file handling is
owned by the host/CLI transport layer, not the kernel.
"""

from __future__ import annotations

import os

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
