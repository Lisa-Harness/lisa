"""Reasoning-aware turn-loop completion budget.

Reasoning-capable models (``reasoning_model``/``thinking_model`` flags or a
``reasoning`` capability string) get ``Settings.turn_completion_max_tokens``
(default 16384) instead of the lean legacy 4096: some providers (Ollama) bill
chain-of-thought against ``max_tokens``, so the lean budget starves the
visible answer or the tool-call payload (the 2026-09-02 empty-turn and
2026-09-17 meta-analysis incidents). The 2026-09-19 narration-loop incident
raised the default from 8192: a completion holding deliberation PLUS an
artifact-sized write payload still truncated mid-tool-call. Non-reasoning
models keep 4096.
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
    """A model with the ``reasoning`` capability gets the 16384 budget."""

    model = _model(capabilities=("reasoning", "tools", "text"), context_window=131072)
    assert _turn_completion_max_tokens(model) == 16384


def test_reasoning_model_flag_gets_reasoning_budget() -> None:
    """The ``reasoning_model`` manifest flag drives the budget bump."""

    model = _model(reasoning_model=True)
    assert _turn_completion_max_tokens(model) == 16384


def test_thinking_model_flag_gets_reasoning_budget() -> None:
    """The ``thinking_model`` manifest flag drives the budget bump."""

    model = _model(thinking_model=True)
    assert _turn_completion_max_tokens(model) == 16384


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


def test_reasoning_budget_capped_by_context_window() -> None:
    """A reasoning model with a small context window gets the window, not 16384.

    The 16384 default is only safe when the model's context window allows
    it. A reasoning model with ``context_window=8192`` (e.g. a small
    local deployment) must get 8192 — never more than its window.
    """

    model = _model(capabilities=("reasoning",), context_window=8192)
    assert _turn_completion_max_tokens(model) == 8192


# ── Per-model manifest budget (BUDGET_SCOPES_RFC scope 1) ─────────────


def test_model_declared_budget_bounds_reasoning_budget() -> None:
    """A manifest ``max_completion_tokens`` bounds the per-call budget.

    Scope 1 of the per-scope completion budgets: the manifest entry
    declares provider behavior — "this model may not bill more than N
    completion tokens per call". The 2026-10-06 glm-5.3:cloud marathons
    burned the host override at the settings level only; a per-model
    ceiling tames reasoning-class models for every mode at once, and
    makes the truncation-rescue math proportionally smaller for free.
    """

    model = _model(reasoning_model=True, max_completion_tokens=8192)
    assert _turn_completion_max_tokens(model) == 8192


def test_model_declared_budget_above_settings_uses_settings() -> None:
    """The effective budget is the min of model ceiling and settings value.

    A per-model value may only tighten; a declaration above the host
    budget never raises the effective budget.
    """

    model = _model(reasoning_model=True, max_completion_tokens=131072)
    assert _turn_completion_max_tokens(model) == 16384


def test_model_declared_budget_absent_inherits_settings() -> None:
    """Absent field (0) inherits the settings path unchanged."""

    model = _model(reasoning_model=True, max_completion_tokens=0)
    assert _turn_completion_max_tokens(model) == 16384


def test_model_declared_budget_respects_context_window() -> None:
    """The window cap applies after the model ceiling: min of all three."""

    model = _model(
        reasoning_model=True, max_completion_tokens=8192, context_window=4096
    )
    assert _turn_completion_max_tokens(model) == 4096


def test_manifest_load_validates_model_completion_budget() -> None:
    """The manifest loader accepts and rejects per-model budgets.

    ``max_completion_tokens`` must be >= 1024 and, when both are set,
    <= the entry's ``context_window`` — violations are load-time errors
    (the manifest is not the place to hide a silent clamp).
    """
    import pathlib
    import tempfile

    import yaml
    from kernel.model_manifest import load_model_manifest

    def _load(model: dict) -> None:
        payload = {"version": 1, "models": [model]}
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            yaml.dump(payload, f)
            f.flush()
            load_model_manifest(pathlib.Path(f.name))

    _load({"model_id": "ok", "provider": "openai", "max_completion_tokens": 8192})
    with pytest.raises(ValueError, match="max_completion_tokens"):
        _load({"model_id": "tiny", "provider": "openai", "max_completion_tokens": 512})
    with pytest.raises(ValueError, match="max_completion_tokens"):
        _load(
            {
                "model_id": "over",
                "provider": "openai",
                "max_completion_tokens": 131072,
                "context_window": 65536,
            }
        )


# ── Per-mode completion budget (BUDGET_SCOPES_RFC scope 2) ─────────────


def test_mode_budget_bounds_model_and_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """A mode's declared budget tightens the effective per-call budget.

    Precedence is min(mode, model, settings): a mode declares its work
    shape; a model entry declares provider behavior; the settings value is
    the process default. The narrowest declaration wins, so the 2026-10-06
    shape (a report-final host override becoming every mode's ceiling)
    cannot recur — a mode that only ever produces small payloads can opt
    down even when the host raised the global budget.
    """
    from kernel.turn_runner import _effective_completion_max_tokens

    model = _model(reasoning_model=True, max_completion_tokens=8192)
    assert _effective_completion_max_tokens(model, mode_budget_tokens=4096) == 4096
    # Mode absent → model ceiling applies.
    assert _effective_completion_max_tokens(model, mode_budget_tokens=0) == 8192
    # Both absent → settings path unchanged.
    plain = _model(reasoning_model=True)
    assert _effective_completion_max_tokens(plain, mode_budget_tokens=0) == 16384


def test_mode_budget_never_raises_effective_budget() -> None:
    """A mode value above the model ceiling does not raise the budget."""

    from kernel.turn_runner import _effective_completion_max_tokens

    model = _model(reasoning_model=True, max_completion_tokens=8192)
    assert _effective_completion_max_tokens(model, mode_budget_tokens=65536) == 8192
