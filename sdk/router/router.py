"""CEL-based model router engine.

Evaluates the ``router.rules`` block from the manifest against the turn
state in ``RouterRequest``. Each rule's ``when`` condition is a
`CEL <https://github.com/google/cel-spec>`_ expression compiled eagerly at
router construction time. The first matching rule wins (first-match-wins).

A rule selects one action:

- ``use: <model_id>`` — route this iteration to the named model.
- ``finish: <reason>`` — stop the tool loop and force one final no-tools
  completion.
- ``transition_to: <stage_id>`` — move the active workflow to a named
  terminal stage, then produce its closing summary.

If no rule matches, the ``router.fallback`` model is used; if no fallback
is configured, the router returns ``None`` so the turn loop falls back to
the statically-resolved ``preferred_model_id``.

Supported CEL bindings (see ``router/README.md`` for the full guide):

- ``turn.iteration`` — the 1-based iteration number.
- ``turn.tool_calls_so_far`` — tool calls dispatched so far this turn.
- ``turn.budget_remaining`` — remaining tool-call budget, or ``-1`` when
  unlimited/unknown (CEL has no nullable int; ``-1`` is the sentinel).
- ``turn.last_tool_result`` — the last tool result text, or ``""`` when
  none. Use ``turn.has_tool_result`` for a boolean.
- ``route.query`` — the user query text.
- ``route.mode_id`` — the active mode id.
- ``route.required_capabilities`` — a list of required capability names.
- ``model.model_id`` / ``model.provider`` / ``model.cost_tier`` /
  ``model.latency_tier`` / ``model.capabilities`` /
  ``model.reasoning_model`` / ``model.thinking_model`` /
  ``model.structured_reliability`` — the metadata of
  the model effective in the preceding iteration (or the static preferred
  model on iteration 1).

Example rules::

    rules:
      - when: '"code" in route.required_capabilities'
        use: gpt-5.2-codex
      - when: turn.iteration > 1 && turn.has_tool_result
        use: gpt-5.4
      - when: turn.tool_calls_so_far >= 8
        finish: tool_budget_exhausted
      - when: turn.iteration >= 10
        transition_to: summary

Conditions are standard CEL — boolean expressions with ``&&``, ``||``,
``!``, comparisons, ``in`` (list/map membership), and function calls. This
keeps the engine free of custom parsing while remaining safe (CEL has no
side effects, no I/O, no arbitrary code execution). Complex routing needs
can still use a vendor backend (e.g. ``lisa-pack-aoi-router``).
"""

from __future__ import annotations

import logging
from typing import Any

import celpy

from kernel.model_manifest import ModelManifest, RouterRule
from kernel.seams.protocols import (
    ModelRouter,
    RouterDecision,
    RouterModelRef,
    RouterRequest,
)

logger = logging.getLogger(__name__)

_ROUTE_REASON_PREFIX = "lisa-router: "

# Sentinel for "no budget information" in CEL, which lacks nullable ints.
_BUDGET_UNKNOWN = -1


class LisaRouter(ModelRouter):
    """Built-in CEL-based model router.

    Constructed with a loaded ``ModelManifest`` whose ``router`` block is
    active (``backend == "lisa-router"``). Every rule's ``when`` expression
    is compiled to a CEL program at construction time; a parse error is
    raised immediately so an invalid manifest fails at startup rather than
    silently mis-routing turns. The ``choose`` method evaluates each compiled
    rule in order; the first match returns a ``RouterDecision`` carrying the
    rule's action.
    """

    def __init__(self, manifest: ModelManifest) -> None:
        self._manifest = manifest
        self._rules: tuple[tuple[RouterRule, celpy.Runner], ...] = ()
        if manifest.router is not None:
            env = celpy.Environment()
            compiled: list[tuple[RouterRule, celpy.Runner]] = []
            for rule in manifest.router.rules:
                try:
                    ast = env.compile(rule.when)
                    program = env.program(ast)
                except Exception as exc:  # pylint: disable=broad-except
                    raise ValueError(
                        f"lisa-router: failed to compile CEL expression {rule.when!r}: {exc}"
                    ) from exc
                compiled.append((rule, program))
            self._rules = tuple(compiled)
        self._fallback: str = manifest.router.fallback if manifest.router else ""

    def choose(self, req: RouterRequest) -> RouterDecision | None:
        activation = _build_activation(req)
        for rule, program in self._rules:
            try:
                result = program.evaluate(activation)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(
                    "lisa-router: CEL evaluation failed for %r: %s",
                    rule.when,
                    exc,
                )
                continue
            if not _truthy(result):
                continue
            return _decision_for_rule(rule, req)
        if self._fallback:
            return RouterDecision(
                model_id=self._fallback,
                reason=f"{_ROUTE_REASON_PREFIX}no rule matched; using fallback",
                routed_by="lisa-router",
            )
        # No rule matched and no fallback — return None so the turn loop
        # falls back to the statically-resolved preferred_model_id.
        return None


def _truthy(value: Any) -> bool:
    """Coerce a CEL result to a Python bool, tolerating celpy's bool types."""
    if isinstance(value, bool):
        return value
    # celpy returns BoolType, a bool subclass, but be defensive.
    try:
        return bool(value)
    except Exception:  # pylint: disable=broad-except
        return False


def _decision_for_rule(rule: RouterRule, req: RouterRequest) -> RouterDecision:
    """Build the RouterDecision for one matching rule."""
    reason = f"{_ROUTE_REASON_PREFIX}rule '{rule.when}' matched at iteration {req.iteration}"
    kind = rule.action_kind()
    if kind == "use":
        return RouterDecision(model_id=rule.use, reason=reason, routed_by="lisa-router")
    if kind == "finish":
        return RouterDecision(
            finish=rule.finish,
            reason=f"{reason} (finish: {rule.finish})",
            routed_by="lisa-router",
        )
    return RouterDecision(
        transition_to=rule.transition_to,
        reason=f"{reason} (transition_to: {rule.transition_to})",
        routed_by="lisa-router",
    )


def _build_activation(req: RouterRequest) -> dict[str, Any]:
    """Build the CEL activation context from a RouterRequest."""

    budget = req.budget_remaining
    budget_val = budget if budget is not None else _BUDGET_UNKNOWN

    last_result = req.last_tool_result or ""

    turn_ctx: dict[str, Any] = {
        "iteration": req.iteration,
        "tool_calls_so_far": req.tool_calls_so_far,
        "budget_remaining": budget_val,
        "last_tool_result": last_result,
        "has_tool_result": last_result != "",
    }
    route_ctx: dict[str, Any] = {
        "query": req.query,
        "mode_id": req.mode_id,
        "required_capabilities": list(req.required_capabilities),
    }

    if req.effective_model is not None:
        m = req.effective_model
        model_ctx: dict[str, Any] = {
            "model_id": m.model_id,
            "provider": m.provider,
            "cost_tier": m.cost_tier,
            "latency_tier": m.latency_tier,
            "capabilities": list(m.capabilities),
            "reasoning_model": m.reasoning_model,
            "thinking_model": m.thinking_model,
            "structured_reliability": float(getattr(m, "structured_reliability", 0.0) or 0.0),
        }
    else:
        model_ctx = {
            "model_id": req.default_model_id,
            "provider": "",
            "cost_tier": "",
            "latency_tier": "",
            "capabilities": [],
            "reasoning_model": False,
            "thinking_model": False,
            "structured_reliability": 0.0,
        }

    return {
        "turn": celpy.json_to_cel(turn_ctx),
        "route": celpy.json_to_cel(route_ctx),
        "model": celpy.json_to_cel(model_ctx),
    }


__all__ = ["LisaRouter", "RouterModelRef"]
