"""Tests for the model-router seam: ``ModelRouter`` Protocol, discovery, and
the three-tier fallback in ``_resolve_iteration_model``.

Covers:
- ``ContextSeams.model_router`` defaults to ``None`` (no behavior change).
- ``kernel.model_router_discovery.discover_model_router_factories`` returns
  an empty list when no pack is registered, and the entry-point group name
  is stable.
- The turn loop's ``_resolve_iteration_model`` consults a wired ``ModelRouter``
  first, falls back to the legacy ``ModelPolicy.choose`` path, then to the
  static ``model_def``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from kernel.model_router_discovery import (
    MODEL_ROUTERS_ENTRY_POINT_GROUP,
    discover_model_router_factories,
)
from kernel.seams.protocols import RouterDecision, RouterRequest
from kernel.turn_runner import ContextSeams, TurnHooks, run_turn

_CORE_ROOT: Path = (Path(__file__).resolve().parents[1] / ".." / "sdk" / "packs" / "core").resolve()


# ── Seam default tests ──────────────────────────────────────────────────────


def test_context_seams_model_router_defaults_to_none() -> None:
    """The new seam field defaults to None — no behavior change when unwired."""
    seams = ContextSeams()
    assert seams.model_router is None


def test_model_routers_entry_point_group_name() -> None:
    """The new entry-point group is stable and named as documented."""
    assert MODEL_ROUTERS_ENTRY_POINT_GROUP == "lisa.extensions.model_routers"


def test_discover_model_router_factories_returns_callable_entries() -> None:
    """Discovery returns entries with callable factories (or an empty list)."""
    entries = discover_model_router_factories()
    for entry in entries:
        assert callable(entry.factory)
        assert isinstance(entry.pack_name, str)


# ── Turn-loop integration tests ─────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _StubCompletion:
    content: str
    tool_calls: list[dict[str, Any]] | None = None


def _make_stub_complete(response: str):
    async def stub(**kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content=response)

    return stub


def _build_minimal_hooks(complete_fn) -> TurnHooks:
    return TurnHooks(complete=complete_fn)


@pytest.mark.asyncio
async def test_router_is_consulted_per_iteration() -> None:
    """When a ModelRouter is wired, the loop calls choose() each iteration
    and uses the returned model_id."""

    choose_calls: list[RouterRequest] = []

    class _StubRouter:
        def choose(self, req: RouterRequest) -> RouterDecision:
            choose_calls.append(req)
            return RouterDecision(
                model_id="gpt-5.4",
                reason="test router",
                routed_by="stub",
            )

    resolved_ids: list[str] = []

    def resolve_model(model_id: str) -> Any:
        resolved_ids.append(model_id)
        return SimpleNamespace(model_id=model_id, reasoning_model=False)

    hooks = _build_minimal_hooks(_make_stub_complete("Done."))
    hooks.resolve_model = resolve_model
    hooks.context_seams.model_router = _StubRouter()

    result = await run_turn(mode_id="default", query="hello", hooks=hooks)

    assert result.response_text == "Done."
    # The router was consulted at least once (iteration 1).
    assert len(choose_calls) >= 1
    # The resolved model was the router's choice, not the default.
    assert "gpt-5.4" in resolved_ids


@pytest.mark.asyncio
async def test_router_returning_none_falls_back_to_static() -> None:
    """When the router returns None, the loop falls back to the static model_def."""

    class _DecliningRouter:
        def choose(self, req: RouterRequest) -> None:
            return None

    resolved_ids: list[str] = []

    def resolve_model(model_id: str) -> Any:
        resolved_ids.append(model_id)
        return SimpleNamespace(model_id=model_id, reasoning_model=False)

    hooks = _build_minimal_hooks(_make_stub_complete("Done."))
    hooks.resolve_model = resolve_model
    hooks.context_seams.model_router = _DecliningRouter()

    result = await run_turn(mode_id="default", query="hello", hooks=hooks)

    assert result.response_text == "Done."
    # The static default was used.
    assert resolved_ids == ["default"]


@pytest.mark.asyncio
async def test_router_exception_falls_back_gracefully() -> None:
    """When the router raises, the loop catches it and falls back to static."""

    class _CrashingRouter:
        def choose(self, req: RouterRequest) -> RouterDecision:
            raise RuntimeError("router exploded")

    resolved_ids: list[str] = []

    def resolve_model(model_id: str) -> Any:
        resolved_ids.append(model_id)
        return SimpleNamespace(model_id=model_id, reasoning_model=False)

    hooks = _build_minimal_hooks(_make_stub_complete("Done."))
    hooks.resolve_model = resolve_model
    hooks.context_seams.model_router = _CrashingRouter()

    result = await run_turn(mode_id="default", query="hello", hooks=hooks)

    assert result.response_text == "Done."
    assert resolved_ids == ["default"]


@pytest.mark.asyncio
async def test_legacy_choose_still_works_without_router() -> None:
    """When model_router is None but ModelPolicy has a choose method,
    the legacy duck-typed path is used."""

    class _PolicyWithChoose:
        def resolve_alias(self, alias: str) -> str:
            return alias

        def is_available(self, model_id: str) -> bool:
            return True

        def choose(self, **kwargs: Any) -> str:
            return "haiku"

    resolved_ids: list[str] = []

    def resolve_model(model_id: str) -> Any:
        resolved_ids.append(model_id)
        return SimpleNamespace(model_id=model_id, reasoning_model=False)

    hooks = _build_minimal_hooks(_make_stub_complete("Done."))
    hooks.resolve_model = resolve_model
    hooks.context_seams.model_policy = _PolicyWithChoose()

    result = await run_turn(mode_id="default", query="hello", hooks=hooks)

    assert result.response_text == "Done."
    assert "haiku" in resolved_ids


@pytest.mark.asyncio
async def test_no_router_no_choose_uses_static_default() -> None:
    """With no router and no choose, the static preferred_model_id is used."""

    resolved_ids: list[str] = []

    def resolve_model(model_id: str) -> Any:
        resolved_ids.append(model_id)
        return SimpleNamespace(model_id=model_id, reasoning_model=False)

    hooks = _build_minimal_hooks(_make_stub_complete("Done."))
    hooks.resolve_model = resolve_model

    result = await run_turn(
        mode_id="default", query="hello", hooks=hooks, preferred_model_id="default"
    )

    assert result.response_text == "Done."
    assert resolved_ids == ["default"]
