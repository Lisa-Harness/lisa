"""Shared model-aware completion budgets (``kernel.model_budgets``).

The reasoning-aware budget, the burnout predicate, and the retry-budget
math moved out of ``kernel.turn_runner`` into ``kernel.model_budgets`` so
pack operations that call ``ctx.host.complete`` directly (the markdown
forge's burnout rescue) share the exact same policy as the turn-loop
rescues. These tests lock in the shared surface; turn-loop behavior tests
remain in ``test_turn_completion_budget.py`` and ``test_streaming.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

# The turn runner keeps its historical private names as aliases of the
# shared module so its call sites and tests stay valid.
import kernel.turn_runner as tr
import pytest
from kernel.config import Settings
from kernel.model_budgets import (
    _LEGACY_TURN_COMPLETION_MAX_TOKENS,
    _REASONING_BURNOUT_SHARE,
    _burnout_or_truncation,
    _completion_usage_tokens,
    _effective_completion_max_tokens,
    _is_reasoning_burnout_completion,
    _model_completion_ceiling,
    _reasoning_capable,
    _replacement_retry_request,
    _retry_budget_tokens,
    _turn_completion_max_tokens,
)


def _model(**overrides: object) -> SimpleNamespace:
    base: dict[str, object] = {
        "reasoning_model": False,
        "thinking_model": False,
        "capabilities": (),
        "context_window": 0,
        "max_completion_tokens": 0,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_turn_runner_aliases_bind_to_shared_module() -> None:
    """``kernel.turn_runner`` delegates to ``kernel.model_budgets``.

    The budget entry points are thin delegates so monkeypatched tests that
    patch the shared module's settings object flip both callers; identity
    holds for the moved constants and the burnout predicate.
    """

    assert tr._is_reasoning_burnout_completion is _is_reasoning_burnout_completion
    assert tr._REASONING_BURNOUT_SHARE == _REASONING_BURNOUT_SHARE
    assert tr._REASONING_CAPABILITY_MARKER == "reasoning"
    assert tr._LEGACY_TURN_COMPLETION_MAX_TOKENS == _LEGACY_TURN_COMPLETION_MAX_TOKENS
    # Equivalent outputs at the same inputs — the delegation contract —
    # rather than function identity (the turn runner keeps docstring-bearing
    # wrappers for its own call sites).
    sample = _model(reasoning_model=True, context_window=131072)
    assert tr._turn_completion_max_tokens(sample) == _turn_completion_max_tokens(sample)


def test_budget_matches_turn_runner_behavior() -> None:
    """Shared budget math reproduces the turn-loop documented behavior."""

    reasoning = _model(capabilities=("reasoning", "tools"), context_window=131072)
    assert _turn_completion_max_tokens(reasoning) == 16384
    plain = _model(capabilities=("tools", "text"))
    assert _turn_completion_max_tokens(plain) == _LEGACY_TURN_COMPLETION_MAX_TOKENS
    # Per-model ceiling may only tighten.
    assert (
        _turn_completion_max_tokens(_model(reasoning_model=True, max_completion_tokens=8192))
        == 8192
    )


def test_mode_budget_only_tightens() -> None:
    """``_effective_completion_max_tokens`` applies min(mode, model, settings)."""

    model = _model(reasoning_model=True, context_window=131072)
    assert _effective_completion_max_tokens(model) == 16384
    assert _effective_completion_max_tokens(model, mode_budget_tokens=4096) == 4096
    # A mode declaring MORE than the resolved budget cannot loosen it.
    assert _effective_completion_max_tokens(model, mode_budget_tokens=999_999) == 16384


def test_model_completion_ceiling() -> None:
    """``_model_completion_ceiling`` surfaces only a positive declaration."""

    assert _model_completion_ceiling(_model(max_completion_tokens=32768)) == 32768
    assert _model_completion_ceiling(_model(max_completion_tokens=0)) is None
    assert _model_completion_ceiling(_model()) is None


def test_retry_budget_tokens_caps_at_context_window() -> None:
    """The shared retry math doubles, then caps at the model context window."""

    model = _model(context_window=8192)
    assert _retry_budget_tokens(base_max_tokens=4096, model_def=model) == 8192
    assert _retry_budget_tokens(base_max_tokens=6000, model_def=model) == 8192
    # No declared window: the multiplier is the only bound.
    assert _retry_budget_tokens(base_max_tokens=1000, model_def=_model()) == 2000


def test_retry_budget_tokens_context_budget_overrides_window() -> None:
    """A supplied ``context_budget`` caps the retry below the model window.

    The doubled budget (2 × 1000 = 2000) is min()ed against the budget's
    remaining room (window − margin − input = 3500) — the cap binds only
    when the doubled value exceeds the remaining room, not automatically.
    """
    model = _model(context_window=131072)
    budget = SimpleNamespace(window_tokens=10_000, margin_tokens=500, input_tokens=6_000)
    # Doubled (2000) fits inside the 3500-token remaining room: the
    # multiplier result stands, the cap is only an upper bound.
    doubled = _retry_budget_tokens(base_max_tokens=1000, model_def=model, context_budget=budget)
    assert doubled == 2_000
    # A tighter remaining room (window − margin − input = 2500) caps the
    # doubled value below the multiplier result.
    tight = SimpleNamespace(window_tokens=10_000, margin_tokens=500, input_tokens=7_000)
    capped = _retry_budget_tokens(base_max_tokens=2000, model_def=model, context_budget=tight)
    assert capped == 2_500


def test_retry_budget_tokens_custom_multiplier() -> None:
    """``multiplier`` scales the base; zero/negative base stays non-positive."""

    model = _model()
    assert _retry_budget_tokens(base_max_tokens=1500, model_def=model, multiplier=4) == 6000
    assert _retry_budget_tokens(base_max_tokens=0, model_def=model) == 0


def test_replacement_retry_request_swaps_budget() -> None:
    """The request clone updates ``max_tokens`` and nothing else."""

    request = _request(max_tokens=1000)
    cloned = _replacement_retry_request(request, retry_max_tokens=2000)
    assert cloned.max_tokens == 2000
    assert cloned.messages == request.messages
    assert cloned.model_def is request.model_def


def test_burnout_predicate_and_classifier_agree() -> None:
    """``_burnout_or_truncation`` routes the three completion shapes."""

    burnout = _burnout_completion(max_tokens=1300)
    assert _burnout_or_truncation(burnout, base_max_tokens=1300) == "burnout"
    assert _is_reasoning_burnout_completion(burnout, base_max_tokens=1300) is True
    truncated = SimpleNamespace(
        finish_reason="length",
        content="",
        tool_calls=None,
        reasoning_content="partial",
        usage=SimpleNamespace(completion_tokens=1300),
    )
    assert _burnout_or_truncation(truncated, base_max_tokens=1300) == "length"
    healthy = SimpleNamespace(
        finish_reason="stop",
        content="# doc",
        tool_calls=None,
        reasoning_content="thinking",
        usage=SimpleNamespace(completion_tokens=400),
    )
    assert _burnout_or_truncation(healthy, base_max_tokens=1300) is None
    # Below the burn share: a cheap empty completion stays unclassified.
    cheap = SimpleNamespace(
        finish_reason="stop",
        content="",
        tool_calls=None,
        reasoning_content="a little",
        usage=SimpleNamespace(completion_tokens=100),
    )
    assert _burnout_or_truncation(cheap, base_max_tokens=1300) is None


def test_burnout_predicate_accepts_direct_completion_tokens() -> None:
    """Completions exposing ``completion_tokens`` without a usage object qualify."""

    completion = SimpleNamespace(
        finish_reason="stop",
        content="",
        tool_calls=None,
        reasoning_content="long deliberation",
        completion_tokens=1250,
    )
    assert _completion_usage_tokens(completion) == 1250
    assert _is_reasoning_burnout_completion(completion, base_max_tokens=1300) is True


def test_burnout_predicate_requires_reasoning_and_no_tools() -> None:
    """Empty content without reasoning, or with tool calls, is not a burnout."""

    no_reasoning = SimpleNamespace(
        finish_reason="stop",
        content="",
        tool_calls=None,
        reasoning_content="",
        usage=SimpleNamespace(completion_tokens=1300),
    )
    assert _burnout_or_truncation(no_reasoning, base_max_tokens=1300) is None
    with_tools = SimpleNamespace(
        finish_reason="stop",
        content="",
        tool_calls=[{"id": "t"}],
        reasoning_content="long deliberation",
        usage=SimpleNamespace(completion_tokens=1300),
    )
    assert _burnout_or_truncation(with_tools, base_max_tokens=1300) is None


def test_reasoning_capable_detection() -> None:
    """Flags and capabilities both mark the model; neither keeps it plain."""

    assert _reasoning_capable(_model(reasoning_model=True)) is True
    assert _reasoning_capable(_model(thinking_model=True)) is True
    assert _reasoning_capable(_model(capabilities=("REASONING",))) is True
    assert _reasoning_capable(_model(capabilities=("tools", "text"))) is False


# ── Fixtures ───────────────────────────────────────────────────────────


def _request(*, max_tokens: int) -> Any:
    from kernel.provider_contracts import CompletionRequest

    return CompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        model_def=_model(context_window=131072),
        max_tokens=max_tokens,
        temperature=0.4,
    )


def _burnout_completion(*, max_tokens: int) -> Any:
    """One budget-consumed reasoning burnout at ``max_tokens`` (>= 90% share)."""

    tokens = int(max_tokens * _REASONING_BURNOUT_SHARE)
    return SimpleNamespace(
        finish_reason="stop",
        content="",
        tool_calls=None,
        reasoning_content="x" * 100,
        usage=SimpleNamespace(completion_tokens=tokens),
    )


@pytest.mark.parametrize(
    ("configured", "expected"),
    [(0, 4096), (12000, 12000)],
)
def test_settings_budget_plumbing(
    monkeypatch: pytest.MonkeyPatch, *, configured: int, expected: int
) -> None:
    """The shared helper reads the live ``Settings`` value like the turn loop."""

    from kernel.config import settings as live_settings

    original = live_settings.turn_completion_max_tokens
    try:
        object.__setattr__(live_settings, "turn_completion_max_tokens", configured)
        assert _turn_completion_max_tokens(_model(reasoning_model=True)) == expected
    finally:
        object.__setattr__(live_settings, "turn_completion_max_tokens", original)


def test_settings_env_parsing_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    """Env override of ``TURN_COMPLETION_MAX_TOKENS`` reaches the shared helper."""

    monkeypatch.setenv("TURN_COMPLETION_MAX_TOKENS", "12000")
    assert Settings().turn_completion_max_tokens == 12000
