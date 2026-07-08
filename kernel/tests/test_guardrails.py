"""Tests for the InputGuard/OutputGuard guardrail seam and turn-runner wiring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from kernel.context import PassThroughInputGuard, PassThroughOutputGuard
from kernel.ipsar_fsm import register_content_roots, register_operation_callbacks
from kernel.models import SurfaceContext
from kernel.operations.registry import execute_operation, get_operation_catalog
from kernel.seams.protocols import GuardResult
from kernel.turn_runner import TurnHooks, TurnResult, run_turn

# ── Module-level setup ────────────────────────────────────────────────
_CORE_ROOT: Path = (Path(__file__).resolve().parents[1] / ".." / "sdk_packs" / "core").resolve()


@pytest.fixture(scope="module", autouse=True)
def _guardrails_setup() -> None:
    """Wire the operation callbacks and content roots so run_turn works."""
    register_content_roots(_CORE_ROOT)
    register_operation_callbacks(get_operation_catalog, execute_operation)


# ── Test helpers ──────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _StubCompletion:
    """Fake model completion for tests."""

    content: str
    tool_calls: list[dict[str, Any]] | None = None


def _make_stub_complete(response: str, captured: dict[str, Any] | None = None):
    """Return an async complete stub that records the query it received."""

    async def stub(**kwargs: Any) -> _StubCompletion:
        if captured is not None:
            messages = kwargs.get("messages") or []
            for msg in messages:
                if msg.get("role") == "user":
                    captured["user_content"] = msg.get("content", "")
                    break
        return _StubCompletion(content=response)

    return stub


def _build_minimal_hooks(complete_fn) -> TurnHooks:
    return TurnHooks(complete=complete_fn)


# ── Pass-through guard tests ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_passthrough_input_guard_allows_everything() -> None:
    """PassThroughInputGuard returns allowed=True with no sanitized text."""
    guard = PassThroughInputGuard()
    surface = SurfaceContext(surface_type="personal_dm", surface_id="test")

    result = await guard.validate("anything goes", surface)

    assert isinstance(result, GuardResult)
    assert result.allowed is True
    assert result.sanitized_text is None
    assert result.blocked_reason is None


@pytest.mark.asyncio
async def test_passthrough_output_guard_allows_everything() -> None:
    """PassThroughOutputGuard returns allowed=True with no sanitized text."""
    guard = PassThroughOutputGuard()
    surface = SurfaceContext(surface_type="personal_dm", surface_id="test")

    result = await guard.validate("any output", surface)

    assert isinstance(result, GuardResult)
    assert result.allowed is True
    assert result.sanitized_text is None
    assert result.blocked_reason is None


# ── Turn runner integration — input guard ────────────────────────────


class _BlockingInputGuard:
    """Test guard that blocks all input with a custom reason."""

    def __init__(self, reason: str = "Blocked by test policy") -> None:
        self._reason = reason

    async def validate(self, query: str, surface: SurfaceContext) -> GuardResult:
        return GuardResult(allowed=False, blocked_reason=self._reason)


class _SanitizingInputGuard:
    """Test guard that sanitizes the query, replacing it with a safe version."""

    def __init__(self, sanitized: str = "sanitized query") -> None:
        self._sanitized = sanitized

    async def validate(self, query: str, surface: SurfaceContext) -> GuardResult:
        return GuardResult(allowed=True, sanitized_text=self._sanitized)


@pytest.mark.asyncio
async def test_blocking_input_guard_short_circuits_before_model_call() -> None:
    """A blocking InputGuard returns TurnResult(blocked=True) without calling
    the model (the complete hook is never awaited)."""
    complete_called = False

    async def exploding_complete(**kwargs: Any) -> _StubCompletion:
        nonlocal complete_called
        complete_called = True
        return _StubCompletion(content="should not reach here")

    hooks = _build_minimal_hooks(exploding_complete)
    hooks.context_seams.input_guard = _BlockingInputGuard("Unsafe input detected")

    result = await run_turn(mode_id="default", query="dangerous", hooks=hooks)

    assert isinstance(result, TurnResult)
    assert result.blocked is True
    assert result.response_text == "Unsafe input detected"
    assert result.tool_call_count == 0
    assert result.iteration_count == 0
    assert complete_called is False


@pytest.mark.asyncio
async def test_sanitizing_input_guard_replaces_query_before_model_call() -> None:
    """A sanitizing InputGuard rewrites the query before the model sees it."""
    captured: dict[str, Any] = {}
    hooks = _build_minimal_hooks(_make_stub_complete("ok", captured=captured))
    hooks.context_seams.input_guard = _SanitizingInputGuard("clean query")

    result = await run_turn(mode_id="default", query="dirty query", hooks=hooks)

    assert result.blocked is False
    assert result.response_text == "ok"
    # The user message content should contain the sanitized query, not the original
    assert "clean query" in captured.get("user_content", "")
    assert "dirty query" not in captured.get("user_content", "")


# ── Turn runner integration — output guard ───────────────────────────


class _BlockingOutputGuard:
    """Test guard that blocks all output."""

    def __init__(self, reason: str = "Unsafe output") -> None:
        self._reason = reason

    async def validate(self, text: str, surface: SurfaceContext) -> GuardResult:
        return GuardResult(allowed=False, blocked_reason=self._reason)


class _SanitizingOutputGuard:
    """Test guard that sanitizes the output, replacing it with a safe version."""

    def __init__(self, sanitized: str = "sanitized output") -> None:
        self._sanitized = sanitized

    async def validate(self, text: str, surface: SurfaceContext) -> GuardResult:
        return GuardResult(allowed=True, sanitized_text=self._sanitized)


@pytest.mark.asyncio
async def test_blocking_output_guard_replaces_with_safe_fallback() -> None:
    """A blocking OutputGuard replaces content with the safe fallback and
    sets blocked=True."""

    hooks = _build_minimal_hooks(_make_stub_complete("dangerous output"))
    hooks.context_seams.output_guard = _BlockingOutputGuard()

    result = await run_turn(mode_id="default", query="hi", hooks=hooks)

    assert result.blocked is True
    assert "unable to provide that response" in result.response_text
    assert "dangerous output" not in result.response_text


@pytest.mark.asyncio
async def test_sanitizing_output_guard_replaces_content() -> None:
    """A sanitizing OutputGuard replaces content with sanitized text."""

    hooks = _build_minimal_hooks(_make_stub_complete("original output"))
    hooks.context_seams.output_guard = _SanitizingOutputGuard("safe output")

    result = await run_turn(mode_id="default", query="hi", hooks=hooks)

    assert result.blocked is False
    assert result.response_text == "safe output"
    assert result.response_text == "safe output"


@pytest.mark.asyncio
async def test_default_pass_through_guards_do_not_change_behavior() -> None:
    """With default pass-through guards (no override), the turn behaves
    identically to the pre-guard behavior."""
    hooks = _build_minimal_hooks(_make_stub_complete("normal response"))

    result = await run_turn(mode_id="default", query="hello", hooks=hooks)

    assert result.blocked is False
    assert result.response_text == "normal response"
