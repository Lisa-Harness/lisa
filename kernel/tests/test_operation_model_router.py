"""Tests for operation-scoped model-router consultation.

Covers the ``_resolve_operation_model`` helper in
``kernel.operations.registry`` that consults the model router before
falling back to the definition's authored ``preferred_model_id``.
"""

from __future__ import annotations

from typing import Any

from kernel.models import OperationDefinition, OperationExecutionContext
from kernel.operations.registry import _resolve_operation_model
from kernel.seams.protocols import RouterDecision, RouterRequest


def _definition(
    *,
    operation_id: str = "consolidate_semantic",
    preferred_model_id: str | None = "default",
) -> OperationDefinition:
    """Build a minimal OperationDefinition for testing."""
    return OperationDefinition(
        operation_id=operation_id,
        description="test op",
        provider_id="lisa-pack-memory",
        preferred_model_id=preferred_model_id,
    )


def _ctx(*, model_router: Any = None, mode_id: str = "default") -> OperationExecutionContext:
    return OperationExecutionContext(
        operation_id="consolidate_semantic",
        mode_id=mode_id,
        model_router=model_router,
    )


# ── No router wired ──────────────────────────────────────────────────────────


def test_no_router_returns_authored_model() -> None:
    """When no router is wired, the definition's authored model is used."""
    definition = _definition(preferred_model_id="default")
    ctx = _ctx(model_router=None)
    assert _resolve_operation_model(definition, ctx) == "default"


def test_no_router_returns_none_when_authored_is_none() -> None:
    """When no router and no authored model, returns None."""
    definition = _definition(preferred_model_id=None)
    ctx = _ctx(model_router=None)
    assert _resolve_operation_model(definition, ctx) is None


# ── Router returns a matching rule ──────────────────────────────────────────


def test_router_rule_match_overrides_authored_model() -> None:
    """A router rule matching the operation_id overrides the authored model."""

    class _StubRouter:
        def choose(self, req: RouterRequest) -> RouterDecision:
            assert req.operation_id == "consolidate_semantic"
            return RouterDecision(
                model_id="gpt-5.2-codex",
                reason="op rule",
                routed_by="stub",
            )

    definition = _definition(preferred_model_id="default")
    ctx = _ctx(model_router=_StubRouter())
    assert _resolve_operation_model(definition, ctx) == "gpt-5.2-codex"


# ── Router fallback is ignored ──────────────────────────────────────────────


def test_router_fallback_ignored_returns_authored() -> None:
    """A fallback decision (from_fallback=True) is ignored; authored model used."""

    class _FallbackRouter:
        def choose(self, req: RouterRequest) -> RouterDecision:
            return RouterDecision(
                model_id="gemma4:cloud",
                reason="fallback",
                routed_by="stub",
                from_fallback=True,
            )

    definition = _definition(preferred_model_id="default")
    ctx = _ctx(model_router=_FallbackRouter())
    assert _resolve_operation_model(definition, ctx) == "default"


# ── Router returns None ─────────────────────────────────────────────────────


def test_router_returning_none_returns_authored() -> None:
    """When the router returns None, the authored model is used."""

    class _DecliningRouter:
        def choose(self, req: RouterRequest) -> None:
            return None

    definition = _definition(preferred_model_id="default")
    ctx = _ctx(model_router=_DecliningRouter())
    assert _resolve_operation_model(definition, ctx) == "default"


# ── Router raises ───────────────────────────────────────────────────────────


def test_router_exception_returns_authored() -> None:
    """When the router raises, the authored model is used (graceful fallback)."""

    class _CrashingRouter:
        def choose(self, req: RouterRequest) -> RouterDecision:
            raise RuntimeError("router exploded")

    definition = _definition(preferred_model_id="default")
    ctx = _ctx(model_router=_CrashingRouter())
    assert _resolve_operation_model(definition, ctx) == "default"


# ── Router returns empty model_id ───────────────────────────────────────────


def test_router_empty_model_id_returns_authored() -> None:
    """When the router returns an empty model_id, the authored model is used."""

    class _EmptyRouter:
        def choose(self, req: RouterRequest) -> RouterDecision:
            return RouterDecision(model_id="", reason="empty", routed_by="stub")

    definition = _definition(preferred_model_id="default")
    ctx = _ctx(model_router=_EmptyRouter())
    assert _resolve_operation_model(definition, ctx) == "default"


# ── Router returns finish/transition_to ─────────────────────────────────────


def test_router_finish_action_ignored_returns_authored() -> None:
    """A finish action is a turn-loop lifecycle action; ignored for operations."""

    class _FinishRouter:
        def choose(self, req: RouterRequest) -> RouterDecision:
            return RouterDecision(finish="tool_budget_exhausted", routed_by="stub")

    definition = _definition(preferred_model_id="default")
    ctx = _ctx(model_router=_FinishRouter())
    assert _resolve_operation_model(definition, ctx) == "default"