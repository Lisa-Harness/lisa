"""Model-aware completion budgets and completion-shape predicates.

Shared by two callers that must agree on what a model can spend and how a
budget-exhausted generation looks:

- ``kernel.turn_runner`` — per-iteration turn budgets plus the streamed /
  non-streamed / tool-calls-fallback truncation and reasoning-burnout
  rescues.
- Content packs that call ``ctx.host.complete`` directly (e.g. the
  markdown pack's forge operation). A reasoning model bills its
  chain-of-thought against ``max_tokens`` (Ollama cloud models bill it
  against the OpenAI ``max_tokens`` wire parameter), so an operation that
  derives a small first-pass budget from a document target — words ×
  tokens-per-word — can exhaust the wall on reasoning *before* emitting
  any visible content (the 2026-10-14 forge Empty Response incident:
  ``target_length: 1000`` → ``max_tokens: 1300`` → ``finish_reason=length,
  content=""``, every token on ``reasoning``).

Packs import ``kernel`` only (never ``runtime``), so these helpers live at
the kernel public surface instead of inside the turn runner where they
originated. The turn runner imports from here under its historical
private names; behavior is unchanged for both callers.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from kernel.config import settings as _kernel_settings
from kernel.provider_contracts import CompletionRequest

__all__ = [
    "_LEGACY_TURN_COMPLETION_MAX_TOKENS",
    "_REASONING_BURNOUT_SHARE",
    "_REASONING_CAPABILITY_MARKER",
    "_burnout_or_truncation",
    "_completion_usage_tokens",
    "_effective_completion_max_tokens",
    "_is_reasoning_burnout_completion",
    "_model_completion_ceiling",
    "_reasoning_capable",
    "_replacement_retry_request",
    "_retry_budget_tokens",
    "_turn_completion_max_tokens",
]

#: Legacy lean per-iteration completion budget for non-reasoning models.
#: Reasoning-capable models get ``settings.turn_completion_max_tokens``
#: instead; see ``turn_completion_max_tokens``.
_LEGACY_TURN_COMPLETION_MAX_TOKENS = 4096

#: A reasoning-burnout completion must have consumed at least this share of
#: its per-call budget on the one visible proxy (completion tokens) before
#: the retry fires — below the share, a cheap empty completion keeps its
#: prior end-of-turn behavior (reasoning-tail caveat, no retry).
_REASONING_BURNOUT_SHARE = 0.9

_REASONING_CAPABILITY_MARKER = "reasoning"


def _reasoning_capable(model_def: Any) -> bool:
    """Return True when the model declaration marks reasoning capability."""
    flags_reasoning = bool(
        getattr(model_def, "reasoning_model", False) or getattr(model_def, "thinking_model", False)
    )
    caps_reasoning = _REASONING_CAPABILITY_MARKER in {
        str(cap).lower() for cap in (getattr(model_def, "capabilities", ()) or ())
    }
    return flags_reasoning or caps_reasoning


def _turn_completion_max_tokens(model_def: Any) -> int:
    """Return the per-iteration completion budget for a model definition.

    Reasoning-capable models (``reasoning_model``/``thinking_model`` flags
    or a ``reasoning`` capability string) get ``settings.turn_completion_max_tokens``
    (default 16384); everything else keeps the legacy 4096. Always bounded by
    the model's ``context_window`` when the manifest declares one, and by
    the model's declared ``max_completion_tokens`` ceiling when set
    (BUDGET_SCOPES_RFC scope 1 — the per-model value may only tighten:
    min of settings budget and model ceiling).
    """
    budget = int(getattr(_kernel_settings, "turn_completion_max_tokens", 0) or 0)
    if budget <= 0:
        budget = _LEGACY_TURN_COMPLETION_MAX_TOKENS
    model_ceiling = int(getattr(model_def, "max_completion_tokens", 0) or 0)
    if model_ceiling > 0:
        budget = min(budget, model_ceiling)
    if not _reasoning_capable(model_def):
        # Never exceed the legacy budget for plain models, even when the
        # configured reasoning budget is lower.
        return min(_LEGACY_TURN_COMPLETION_MAX_TOKENS, budget)
    cap_window = int(getattr(model_def, "context_window", 0) or 0)
    if cap_window > 0:
        return min(budget, cap_window)
    return budget


def _effective_completion_max_tokens(model_def: Any, *, mode_budget_tokens: int = 0) -> int:
    """Effective per-call completion budget after the mode's declaration.

    BUDGET_SCOPES_RFC scope 2 precedence: min(mode, model, settings), the
    narrowest declaration wins. A mode value may only tighten — a mode
    declaring more than its model/settings budget inherits the smaller
    value. ``mode_budget_tokens`` 0/absent keeps the model/settings path
    exactly as before.
    """
    budget = _turn_completion_max_tokens(model_def)
    if mode_budget_tokens > 0:
        budget = min(budget, int(mode_budget_tokens))
    return budget


def _model_completion_ceiling(model_def: Any) -> int | None:
    """Return the per-model declared completion ceiling, or ``None``.

    ``max_completion_tokens`` 0/absent means no per-model declaration —
    the caller's own budget logic remains authoritative.
    """
    ceiling = int(getattr(model_def, "max_completion_tokens", 0) or 0)
    return ceiling if ceiling > 0 else None


def _is_reasoning_burnout_completion(completion: Any, *, base_max_tokens: int) -> bool:
    """Return True when a completion burned (almost) its whole budget on thinking.

    The burnout shape (the 2026-10-13 turn 789 stall, glm-5.3-flash:cloud):
    a reasoning model re-deliberates its payload until the wall —
    ``finish_reason='stop'`` (a reasoning loop running out of budget does
    not always surface ``length`` on every backend) — with empty visible
    ``content``, no tool calls, and substantial ``reasoning_content``.
    The ``length`` predicate misses it and the turn would end with nothing
    dispatched and a reasoning-tail caveat. Mirrors the numeric guard of
    ``_empty_content_detail`` (operations/declarative.py): completion tokens
    near the per-call cap mark a budget-consumed generation; anything below
    the share stays a cheap empty completion. A truncated completion
    (``finish_reason='length'``) never enters here — the truncation rescues
    own that shape; call the :func:`_burnout_or_truncation` decision helper
    when a caller needs both predicates in one gate.

    Usage is consulted when present (the shared ``_completion_spend`` helper)
    so a stream without usage cannot prove a burn; direct-attribute usage
    from ``CompletionResult``/provider shapes both work.
    """
    if base_max_tokens <= 0:
        return False
    if getattr(completion, "finish_reason", None) != "stop":
        return False
    if str(getattr(completion, "content", "") or "").strip():
        return False
    if getattr(completion, "tool_calls", None):
        return False
    reasoning = getattr(completion, "reasoning_content", None) or getattr(
        completion, "reasoning", None
    )
    if not isinstance(reasoning, str) or not reasoning.strip():
        return False
    usage = getattr(completion, "usage", None)
    completion_tokens = int(
        getattr(completion, "completion_tokens", 0)
        or (getattr(usage, "completion_tokens", 0) if usage is not None else 0)
    )
    return completion_tokens >= base_max_tokens * _REASONING_BURNOUT_SHARE


def _burnout_or_truncation(completion: Any, *, base_max_tokens: int) -> str | None:
    """Classify a budget-exhausted completion: ``burnout``, ``length``, or ``None``.

    One shared gate for callers that own the retry policy but must agree on
    the diagnosis:
    - ``"length"``   — provider-cut generation (``finish_reason='length'``).
    - ``"burnout"``  — clean stop, empty content, whole budget on thinking,
                       no tool calls (the reasoning-loop shape).
    - ``None``       — neither; the completion needs no rescue.

    ``base_max_tokens`` <= 0 leaves cheap-empty diagnostics to the caller
    (nothing was spent to diagnose).
    """
    if getattr(completion, "finish_reason", None) == "length":
        return "length"
    if base_max_tokens <= 0:
        return None
    if _is_reasoning_burnout_completion(completion, base_max_tokens=base_max_tokens):
        return "burnout"
    return None


def _completion_usage_tokens(completion: Any) -> int:
    """Return the completion-token spend of one completion result (0 when unknown)."""
    usage = getattr(completion, "usage", None)
    return int(
        getattr(completion, "completion_tokens", 0)
        or (getattr(usage, "completion_tokens", 0) if usage is not None else 0)
    )


def _retry_budget_tokens(
    *,
    base_max_tokens: int,
    model_def: Any,
    context_budget: Any = None,
    multiplier: int = 2,
) -> int:
    """Return the one-shot retry budget for a budget-exhausted completion.

    Shared math for the turn runner's ``_truncation_retry_budget`` clone and
    pack operation rescues: base × ``multiplier``, context-window-capped
    (including a supplied ``request.context_budget`` whose window/margin/
    input already bound the wire request), never below the base — a
    non-positive result means "cannot exceed the base; no retry".
    """
    retry_max_tokens = max(0, int(base_max_tokens)) * max(1, multiplier)
    cap_window = int(getattr(model_def, "context_window", 0) or 0)
    if context_budget is not None:
        budget = context_budget
        cap_window = (
            int(getattr(budget, "window_tokens", 0) or 0)
            - int(getattr(budget, "margin_tokens", 0) or 0)
            - int(getattr(budget, "input_tokens", 0) or 0)
        )
    if cap_window > 0:
        retry_max_tokens = min(retry_max_tokens, cap_window)
    return retry_max_tokens


def _replacement_retry_request(
    base_request: CompletionRequest,
    *,
    retry_max_tokens: int,
) -> CompletionRequest:
    """Return ``base_request`` re-issued with the retry budget (dataclasses.replace)."""
    return replace(base_request, max_tokens=retry_max_tokens)
