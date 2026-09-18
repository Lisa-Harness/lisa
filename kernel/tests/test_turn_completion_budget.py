"""Reasoning-aware turn-loop completion budget.

Reasoning-capable models (``reasoning_model``/``thinking_model`` flags or a
``reasoning`` capability string) get ``Settings.turn_completion_max_tokens``
(default 8192) instead of the lean legacy 4096: some providers (Ollama) bill
chain-of-thought against ``max_tokens``, so the lean budget starves the
visible answer or the tool-call payload (the 2026-09-02 empty-turn and
2026-09-17 meta-analysis incidents). Non-reasoning models keep 4096.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from kernel.config import Settings
from kernel.turn_runner import _turn_completion_max_tokens


def _model(**overrides: object) -> SimpleNamespace:
    """Build a minimal ManifestModelEntry-shaped stub."""

    base: dict[str, object] = {
        "reasoning_model": False,
        "thinking_model": False,
        "capabilities": (),
        "context_window": 0,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_reasoning_capability_gets_reasoning_budget() -> None:
    """A model with the ``reasoning`` capability gets the 8192 budget."""

    model = _model(capabilities=("reasoning", "tools", "text"), context_window=131072)
    assert _turn_completion_max_tokens(model) == 8192


def test_reasoning_model_flag_gets_reasoning_budget() -> None:
    """The ``reasoning_model`` manifest flag drives the budget bump."""

    model = _model(reasoning_model=True)
    assert _turn_completion_max_tokens(model) == 8192


def test_thinking_model_flag_gets_reasoning_budget() -> None:
    """The ``thinking_model`` manifest flag drives the budget bump."""

    model = _model(thinking_model=True)
    assert _turn_completion_max_tokens(model) == 8192


def test_plain_model_keeps_legacy_budget() -> None:
    """A tools-only text model keeps the lean legacy 4096."""

    model = _model(capabilities=("tools", "text", "code"))
    assert _turn_completion_max_tokens(model) == 4096


def test_budget_capped_by_context_window() -> None:
    """A small ``context_window`` caps the reasoning budget."""

    model = _model(reasoning_model=True, context_window=4096)
    assert _turn_completion_max_tokens(model) == 4096


def test_env_override_via_turn_completion_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``TURN_COMPLETION_MAX_TOKENS`` (and its LISA_ twin) override the default."""

    monkeypatch.setenv("TURN_COMPLETION_MAX_TOKENS", "12000")
    settings = Settings()
    assert settings.turn_completion_max_tokens == 12000


def test_lower_configured_budget_applies_to_both_model_kinds() -> None:
    """A configured budget below 4096 applies to plain AND reasoning models.

    The bump only ever *raises* the budget for reasoning models; a host
    that explicitly lowers ``TURN_COMPLETION_MAX_TOKENS`` constrains both
    model kinds (min of the legacy floor and the configured value for
    plain models, exactly the configured value for reasoning models).
    """

    import kernel.turn_runner as tr

    original = tr._kernel_settings.turn_completion_max_tokens
    try:
        object.__setattr__(tr._kernel_settings, "turn_completion_max_tokens", 2048)
        assert _turn_completion_max_tokens(_model()) == 2048
        assert _turn_completion_max_tokens(_model(reasoning_model=True)) == 2048
    finally:
        object.__setattr__(tr._kernel_settings, "turn_completion_max_tokens", original)
