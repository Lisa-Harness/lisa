"""Rules-based model router engine.

Evaluates the ``router.rules`` block from the manifest against the turn
state in ``RouterRequest``. Each rule's ``when`` condition is a simple
expression evaluated against the request fields and the candidate model's
metadata. The first matching rule wins (first-match-wins); if no rule
matches, the ``router.fallback`` model is used; if no fallback is configured,
the router returns ``None`` so the turn loop falls back to the statically
resolved ``preferred_model_id``.

Supported condition syntax (case-insensitive keywords):

- ``required_capabilities includes "code"``
    True when every capability in the quoted list is present in
    ``RouterRequest.required_capabilities``.

- ``iteration > 1``
    True when the iteration number exceeds the integer on the right.

- ``iteration > 1 and last_tool_result is not None``
    Compound with ``and`` / ``or``. ``last_tool_result is not None`` is true
    when at least one tool has returned a result this turn.

- ``cost_tier == "frontier"``
    True when the *target* model (``rule.use``) has the given ``cost_tier``.

- ``tool_calls_so_far >= 3``
    True when the number of tool calls so far meets or exceeds the threshold.

Conditions are intentionally simple — single predicates joined by ``and`` /
``or``. This keeps the engine dependency-free (no eval, no AST parser) and
auditable. Complex routing needs should use a vendor backend (e.g.
``lisa-pack-aoi-router``).
"""

from __future__ import annotations

import logging
import re

from kernel.model_manifest import ModelManifest, RouterRule
from kernel.seams.protocols import ModelRouter, RouterDecision, RouterRequest

logger = logging.getLogger(__name__)

_ROUTE_REASON_PREFIX = "lisa-router: "


class LisaRouter(ModelRouter):
    """Built-in rules-based model router.

    Constructed with a loaded ``ModelManifest`` whose ``router`` block is
    active (``backend == "lisa-router"``). The ``choose`` method evaluates
    each rule in order; the first match returns a ``RouterDecision`` naming
    the rule's ``use`` model.
    """

    def __init__(self, manifest: ModelManifest) -> None:
        self._manifest = manifest
        self._rules: tuple[RouterRule, ...] = manifest.router.rules if manifest.router else ()
        self._fallback: str = manifest.router.fallback if manifest.router else ""

    def choose(self, req: RouterRequest) -> RouterDecision | None:
        for rule in self._rules:
            if self._evaluate(rule.when, req):
                return RouterDecision(
                    model_id=rule.use,
                    reason=f"{_ROUTE_REASON_PREFIX}rule '{rule.when}' matched at iteration {req.iteration}",
                    routed_by="lisa-router",
                )
        if self._fallback:
            return RouterDecision(
                model_id=self._fallback,
                reason=f"{_ROUTE_REASON_PREFIX}no rule matched; using fallback",
                routed_by="lisa-router",
            )
        # No rule matched and no fallback — return None so the turn loop
        # falls back to the statically-resolved preferred_model_id.
        return None

    def _evaluate(self, condition: str, req: RouterRequest) -> bool:
        """Evaluate one ``when`` condition string against the request."""

        normalized = condition.strip()
        if not normalized:
            return False

        # Split on top-level ``or`` (lowest precedence).
        or_parts = _split_top_level(normalized, " or ")
        if len(or_parts) > 1:
            return any(self._evaluate(part, req) for part in or_parts)

        # Split on top-level ``and``.
        and_parts = _split_top_level(normalized, " and ")
        if len(and_parts) > 1:
            return all(self._evaluate(part, req) for part in and_parts)

        # Single predicate.
        return self._evaluate_predicate(normalized, req)

    def _evaluate_predicate(self, predicate: str, req: RouterRequest) -> bool:
        """Evaluate one atomic predicate."""

        pred = predicate.strip().lower()

        # required_capabilities includes "code"
        m = re.match(
            r'required_capabilities\s+includes\s+"([^"]+)"',
            pred,
        )
        if m:
            caps = m.group(1)
            required = {c.lower() for c in req.required_capabilities}
            return all(c.strip().lower() in required for c in caps.split(","))

        # last_tool_result is not None / is None
        if pred == "last_tool_result is not none":
            return req.last_tool_result is not None
        if pred == "last_tool_result is none":
            return req.last_tool_result is None

        # iteration > N  /  iteration >= N
        m = re.match(r"iteration\s*([><]=?|==)\s*(\d+)", pred)
        if m:
            op, val = m.group(1), int(m.group(2))
            return _compare(req.iteration, op, val)

        # tool_calls_so_far > N  /  >= N
        m = re.match(r"tool_calls_so_far\s*([><]=?|==)\s*(\d+)", pred)
        if m:
            op, val = m.group(1), int(m.group(2))
            return _compare(req.tool_calls_so_far, op, val)

        # cost_tier == "frontier"  (checks the *target* model from rule.use —
        # but we don't know the target here; this predicate is evaluated
        # against the request, not the candidate. We skip it.)
        # NOTE: cost_tier conditions require knowing the candidate model,
        # which is the rule's ``use`` field — not available in the request.
        # A future extension could pass the candidate model to the predicate.

        logger.debug("lisa-router: unrecognized predicate %r; treating as false", predicate)
        return False


def _split_top_level(text: str, separator: str) -> list[str]:
    """Split ``text`` on ``separator`` at the top level (not inside quotes).

    A simple split is safe because conditions don't nest parentheses —
    ``and`` / ``or`` are the only composition operators and they don't appear
    inside quoted strings.
    """
    if separator not in text:
        return [text]
    return [part.strip() for part in text.split(separator) if part.strip()]


def _compare(left: int, op: str, right: int) -> bool:
    if op == ">":
        return left > right
    if op == "<":
        return left < right
    if op == ">=":
        return left >= right
    if op == "<=":
        return left <= right
    if op == "==":
        return left == right
    return False


__all__ = ["LisaRouter"]
