"""Tests for the ``/pack`` dispatcher and lazy-import table in ``runtime.commands``."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from runtime import commands


class _FakeSession:
    def __init__(self) -> None:
        self.current_mode_id = ""
        self.current_stage_id = ""
        self.next_model_id = ""
        self.restarts: list[str] = []
        self.restart_reason = ""

    def request_restart(self, *, reason: str) -> None:
        self.restarts.append(reason)


def _fake_runtime() -> SimpleNamespace:
    return SimpleNamespace(session=_FakeSession())


@pytest.fixture
def _capture_simple(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    captured: list[tuple[str, str]] = []

    def _record(runtime: Any, message: str, mode_id: str, response_text: str) -> str:
        captured.append((mode_id, response_text))
        session = getattr(runtime, "session", None)
        if session is not None:
            session.current_mode_id = mode_id
            session.current_stage_id = "-"
            session.next_model_id = "-"
        return response_text

    monkeypatch.setattr(commands, "handle_simple_command", _record)
    return captured


# ── __getattr__ lazy-import table ────────────────────────────────────


def test_lazy_getattr_resolves_real_symbols() -> None:
    from runtime.model_runtime import resolve_model as real_resolve

    assert commands.resolve_little_lisa_model is real_resolve


def test_lazy_getattr_missing_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="has no attribute"):
        commands.definitely_not_a_symbol  # noqa: B018


def test_lazy_getattr_shadowed_by_monkeypatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """``setattr`` on the package shadows the lazy import (existing test seam)."""

    sentinel = object()
    monkeypatch.setattr(commands, "discover_models", sentinel)

    assert commands.discover_models is sentinel


@pytest.mark.parametrize("name", sorted(commands._LAZY_IMPORTS))
def test_lazy_import_table_entries_resolve(name: str) -> None:
    assert getattr(commands, name) is not None


# ── /pack dispatcher ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pack_bare_command_shows_usage(_capture_simple: list[tuple[str, str]]) -> None:
    result = await commands.handle_pack_command(_fake_runtime(), "/pack", "/pack")

    assert result.startswith("Usage: /pack list")
    assert _capture_simple[-1][0] == "pack"


@pytest.mark.asyncio
async def test_pack_unknown_action(_capture_simple: list[tuple[str, str]]) -> None:
    result = await commands.handle_pack_command(_fake_runtime(), "/pack frobnicate", "raw")

    assert result.startswith("Unknown /pack action")


@pytest.mark.asyncio
async def test_pack_list_delegates_to_render(
    _capture_simple: list[tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(commands, "render_pack_list", lambda runtime: "pack list output")

    result = await commands.handle_pack_command(_fake_runtime(), "/pack list", "raw")

    assert result == "pack list output"


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["enable", "disable"])
async def test_pack_enable_disable_missing_arg(
    _capture_simple: list[tuple[str, str]],
    action: str,
) -> None:
    result = await commands.handle_pack_command(_fake_runtime(), f"/pack {action}", "raw")

    assert result == f"Usage: /pack {action} <pack>"


@pytest.mark.asyncio
async def test_pack_enable_calls_set_pack_enabled(
    _capture_simple: list[tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        commands,
        "set_pack_enabled",
        lambda runtime, name, *, enabled: seen.append((name, enabled)) or (True, "ok: enabled"),
    )

    result = await commands.handle_pack_command(_fake_runtime(), "/pack enable search", "raw")

    assert result == "ok: enabled"
    assert seen == [("search", True)]


@pytest.mark.asyncio
async def test_pack_disable_calls_set_pack_enabled(
    _capture_simple: list[tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        commands,
        "set_pack_enabled",
        lambda runtime, name, *, enabled: seen.append((name, enabled)) or (True, "ok: disabled"),
    )

    result = await commands.handle_pack_command(_fake_runtime(), "/pack disable memory", "raw")

    assert result == "ok: disabled"
    assert seen == [("memory", False)]


@pytest.mark.asyncio
async def test_pack_install_success_no_resume(
    _capture_simple: list[tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _install(runtime: Any, arg: str) -> tuple[bool, str]:
        return True, f"installed {arg}"

    monkeypatch.setattr(commands, "run_pack_install", _install)
    runtime = _fake_runtime()

    result = await commands.handle_pack_command(runtime, "/pack install search", "raw")

    assert result == "installed search"
    assert runtime.session.restarts == []


@pytest.mark.asyncio
async def test_pack_install_failure_skips_restart(
    _capture_simple: list[tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _install(runtime: Any, arg: str) -> tuple[bool, str]:
        return False, "install failed"

    monkeypatch.setattr(commands, "run_pack_install", _install)
    runtime = _fake_runtime()

    result = await commands.handle_pack_command(runtime, "/pack install bad --resume go on", "raw")

    assert result == "install failed"
    assert runtime.session.restarts == []


@pytest.mark.asyncio
async def test_pack_install_success_with_resume_requests_restart(
    _capture_simple: list[tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    async def _install(runtime: Any, arg: str) -> tuple[bool, str]:
        return True, "installed"

    monkeypatch.setattr(commands, "run_pack_install", _install)
    monkeypatch.delenv("LISA_RESTART_RESUME_MESSAGE", raising=False)
    runtime = _fake_runtime()
    runtime.session.restart_reason = "installing search"

    result = await commands.handle_pack_command(
        runtime,
        "/pack install search --resume continue the setup",
        "raw",
    )

    assert result == "installed"
    assert runtime.session.restarts == ["installing search"]
    assert os.environ["LISA_RESTART_RESUME_MESSAGE"] == "continue the setup"
    monkeypatch.delenv("LISA_RESTART_RESUME_MESSAGE", raising=False)


@pytest.mark.asyncio
async def test_pack_uninstall_success(
    _capture_simple: list[tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _uninstall(runtime: Any, arg: str) -> tuple[bool, str]:
        return True, f"uninstalled {arg}"

    monkeypatch.setattr(commands, "run_pack_uninstall", _uninstall)

    result = await commands.handle_pack_command(_fake_runtime(), "/pack uninstall memory", "raw")

    assert result == "uninstalled memory"


@pytest.mark.asyncio
async def test_pack_uninstall_failure(
    _capture_simple: list[tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _uninstall(runtime: Any, arg: str) -> tuple[bool, str]:
        return False, "no such pack"

    monkeypatch.setattr(commands, "run_pack_uninstall", _uninstall)

    result = await commands.handle_pack_command(_fake_runtime(), "/pack uninstall ghost", "raw")

    assert result == "no such pack"


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", ["help", "--help", "-h"])
async def test_pack_update_help_variants(
    _capture_simple: list[tuple[str, str]],
    flag: str,
) -> None:
    result = await commands.handle_pack_command(_fake_runtime(), f"/pack update {flag}", "raw")

    assert result == "Usage: /pack update <pack> | /pack update --all"


@pytest.mark.asyncio
async def test_pack_update_no_arg(_capture_simple: list[tuple[str, str]]) -> None:
    result = await commands.handle_pack_command(_fake_runtime(), "/pack update", "raw")

    assert result == "Usage: /pack update <pack> | /pack update --all"


@pytest.mark.asyncio
@pytest.mark.parametrize("word", ["--all", "all"])
async def test_pack_update_all(
    _capture_simple: list[tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    word: str,
) -> None:
    async def _update_all(runtime: Any) -> tuple[bool, str]:
        return True, "all updated"

    monkeypatch.setattr(commands, "run_pack_update_all", _update_all)

    result = await commands.handle_pack_command(_fake_runtime(), f"/pack update {word}", "raw")

    assert result == "all updated"


@pytest.mark.asyncio
async def test_pack_update_all_failure(
    _capture_simple: list[tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _update_all(runtime: Any) -> tuple[bool, str]:
        return False, "update-all failed"

    monkeypatch.setattr(commands, "run_pack_update_all", _update_all)

    result = await commands.handle_pack_command(_fake_runtime(), "/pack update --all", "raw")

    assert result == "update-all failed"


@pytest.mark.asyncio
async def test_pack_update_single_with_resume(
    _capture_simple: list[tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    async def _update(runtime: Any, arg: str) -> tuple[bool, str]:
        return True, f"updated {arg}"

    monkeypatch.setattr(commands, "run_pack_update", _update)
    monkeypatch.delenv("LISA_RESTART_RESUME_MESSAGE", raising=False)
    runtime = _fake_runtime()
    runtime.session.restart_reason = "explicit reason"

    result = await commands.handle_pack_command(
        runtime, "/pack update search --resume keep going", "raw"
    )

    assert result == "updated search"
    assert runtime.session.restarts == ["explicit reason"]
    assert os.environ["LISA_RESTART_RESUME_MESSAGE"] == "keep going"
    monkeypatch.delenv("LISA_RESTART_RESUME_MESSAGE", raising=False)


@pytest.mark.asyncio
async def test_pack_update_single_failure_no_restart(
    _capture_simple: list[tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _update(runtime: Any, arg: str) -> tuple[bool, str]:
        return False, "nope"

    monkeypatch.setattr(commands, "run_pack_update", _update)
    runtime = _fake_runtime()

    result = await commands.handle_pack_command(
        runtime, "/pack update search --resume after", "raw"
    )

    assert result == "nope"
    assert runtime.session.restarts == []


@pytest.mark.asyncio
async def test_pack_install_resume_reason_defaults(
    _capture_simple: list[tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``runtime.session.restart_reason or "pack install"`` picks the fallback."""

    import os

    async def _install(runtime: Any, arg: str) -> tuple[bool, str]:
        return True, "installed"

    monkeypatch.setattr(commands, "run_pack_install", _install)
    monkeypatch.delenv("LISA_RESTART_RESUME_MESSAGE", raising=False)
    runtime = _fake_runtime()
    runtime.session.restart_reason = ""  # falsy → default reason

    await commands.handle_pack_command(runtime, "/pack install x --resume msg", "raw")

    assert runtime.session.restarts == ["pack install"]
    assert os.environ["LISA_RESTART_RESUME_MESSAGE"] == "msg"
    monkeypatch.delenv("LISA_RESTART_RESUME_MESSAGE", raising=False)


@pytest.mark.asyncio
async def test_pack_verb_case_insensitive(
    _capture_simple: list[tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(commands, "render_pack_list", lambda runtime: "listed")

    result = await commands.handle_pack_command(_fake_runtime(), "/pack LIST", "raw")

    assert result == "listed"
