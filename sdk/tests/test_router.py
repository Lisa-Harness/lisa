"""Tests for the built-in lisa-router CEL rules engine."""

from __future__ import annotations

import pytest

from kernel.model_manifest import ManifestModelEntry, ModelManifest, RouterConfig, RouterRule
from kernel.seams.protocols import RouterRequest
from sdk.router.router import LisaRouter

# ── Fixtures ─────────────────────────────────────────────────────────────────

_MODELS = (
    ManifestModelEntry(
        key="gpt-5.4",
        model_id="gpt-5.4",
        provider="azure_openai",
        display_name="GPT-5.4",
        cost_tier="frontier",
        latency_tier="medium",
        capabilities=("text", "tools", "structured_output", "code", "vision"),
    ),
    ManifestModelEntry(
        key="gpt-5.2-codex",
        model_id="gpt-5.2-codex",
        provider="azure_openai",
        display_name="gpt-5.2-codex",
        cost_tier="frontier",
        latency_tier="medium",
        capabilities=("code", "structured_output", "text", "tools"),
    ),
    ManifestModelEntry(
        key="gemma4:cloud",
        model_id="gemma4:cloud",
        provider="ollama",
        display_name="Gemma4:Cloud",
        cost_tier="standard",
        latency_tier="medium",
        capabilities=("text", "tools", "structured_output"),
    ),
)


def _manifest(rules: tuple[RouterRule, ...], fallback: str = "") -> ModelManifest:
    return ModelManifest(
        models=_MODELS,
        router=RouterConfig(backend="lisa-router", rules=rules, fallback=fallback),
    )


def _req(**kwargs: object) -> RouterRequest:
    defaults: dict[str, object] = {
        "default_model_id": "gemma4:cloud",
        "iteration": 1,
        "tool_calls_so_far": 0,
        "budget_remaining": None,
    }
    defaults.update(kwargs)
    return RouterRequest(**defaults)  # type: ignore[arg-type]


# ── Factory tests ────────────────────────────────────────────────────────────


def test_factory_returns_none_when_no_router_block() -> None:
    from sdk.router import get_router_factory

    factory = get_router_factory()
    manifest = ModelManifest(models=_MODELS)
    assert factory(manifest, None) is None


def test_factory_returns_none_when_backend_mismatches() -> None:
    from sdk.router import get_router_factory

    factory = get_router_factory()
    manifest = ModelManifest(
        models=_MODELS,
        router=RouterConfig(backend="aoi-router", rules=(), fallback=""),
    )
    assert factory(manifest, None) is None


# ── Default config tests ─────────────────────────────────────────────────────


def test_default_router_config_has_correct_backend() -> None:
    """The backend's default config declares the lisa-router backend."""
    from sdk.router.default_config import get_default_router_config

    config = get_default_router_config()
    assert config["backend"] == "lisa-router"


def test_default_router_config_has_rules() -> None:
    """The default config includes at least one routing rule."""
    from sdk.router.default_config import get_default_router_config

    config = get_default_router_config()
    assert isinstance(config["rules"], list)
    assert len(config["rules"]) >= 1
    for rule in config["rules"]:
        assert "when" in rule
        # Each rule must have exactly one action.
        actions = [k for k in ("use", "finish", "transition_to") if k in rule]
        assert len(actions) == 1


def test_default_router_config_uses_symbolic_refs() -> None:
    """Default rules use symbolic mode refs (default), not literal ids."""
    from sdk.router.default_config import get_default_router_config

    config = get_default_router_config()
    uses = {rule["use"] for rule in config["rules"]}
    assert config["fallback"] == "default"
    assert uses == {"default"}


def test_factory_returns_router_when_backend_matches() -> None:
    from sdk.router import get_router_factory

    factory = get_router_factory()
    manifest = _manifest(rules=(), fallback="gemma4:cloud")
    router = factory(manifest, None)
    assert router is not None
    assert isinstance(router, LisaRouter)


# ── CEL rule evaluation tests ───────────────────────────────────────────────


def test_required_capabilities_rule_matches() -> None:
    manifest = _manifest(
        rules=(RouterRule(when='"code" in route.required_capabilities', use="gpt-5.2-codex"),),
        fallback="gemma4:cloud",
    )
    router = LisaRouter(manifest)
    decision = router.choose(_req(required_capabilities=("code", "text")))
    assert decision is not None
    assert decision.model_id == "gpt-5.2-codex"
    assert decision.routed_by == "lisa-router"


def test_required_capabilities_rule_does_not_match() -> None:
    manifest = _manifest(
        rules=(RouterRule(when='"code" in route.required_capabilities', use="gpt-5.2-codex"),),
        fallback="gemma4:cloud",
    )
    router = LisaRouter(manifest)
    decision = router.choose(_req(required_capabilities=("text",)))
    assert decision is not None
    assert decision.model_id == "gemma4:cloud"


def test_iteration_gt_rule() -> None:
    manifest = _manifest(
        rules=(RouterRule(when="turn.iteration > 1", use="gpt-5.4"),),
        fallback="gemma4:cloud",
    )
    router = LisaRouter(manifest)
    assert router.choose(_req(iteration=1)).model_id == "gemma4:cloud"
    assert router.choose(_req(iteration=2)).model_id == "gpt-5.4"


def test_compound_and_with_last_tool_result() -> None:
    manifest = _manifest(
        rules=(
            RouterRule(
                when="turn.iteration > 1 && turn.has_tool_result",
                use="gpt-5.4",
            ),
        ),
        fallback="gemma4:cloud",
    )
    router = LisaRouter(manifest)
    assert router.choose(_req(iteration=1, last_tool_result=None)).model_id == "gemma4:cloud"
    assert router.choose(_req(iteration=2, last_tool_result=None)).model_id == "gemma4:cloud"
    assert router.choose(_req(iteration=2, last_tool_result="result")).model_id == "gpt-5.4"


def test_compound_or() -> None:
    manifest = _manifest(
        rules=(
            RouterRule(
                when='"code" in route.required_capabilities || turn.iteration > 5',
                use="gpt-5.2-codex",
            ),
        ),
        fallback="gemma4:cloud",
    )
    router = LisaRouter(manifest)
    assert router.choose(_req(required_capabilities=("code",))).model_id == "gpt-5.2-codex"
    assert router.choose(_req(iteration=6)).model_id == "gpt-5.2-codex"
    assert router.choose(_req(iteration=1)).model_id == "gemma4:cloud"


def test_tool_calls_so_far_rule() -> None:
    manifest = _manifest(
        rules=(RouterRule(when="turn.tool_calls_so_far >= 3", use="gpt-5.4"),),
        fallback="gemma4:cloud",
    )
    router = LisaRouter(manifest)
    assert router.choose(_req(tool_calls_so_far=2)).model_id == "gemma4:cloud"
    assert router.choose(_req(tool_calls_so_far=3)).model_id == "gpt-5.4"


def test_no_match_no_fallback_returns_none() -> None:
    manifest = _manifest(
        rules=(RouterRule(when="turn.iteration > 100", use="gpt-5.4"),),
        fallback="",
    )
    router = LisaRouter(manifest)
    assert router.choose(_req(iteration=1)) is None


def test_first_match_wins() -> None:
    manifest = _manifest(
        rules=(
            RouterRule(when="turn.iteration > 1", use="gpt-5.4"),
            RouterRule(when="turn.iteration > 5", use="gpt-5.2-codex"),
        ),
        fallback="gemma4:cloud",
    )
    router = LisaRouter(manifest)
    assert router.choose(_req(iteration=2)).model_id == "gpt-5.4"
    assert router.choose(_req(iteration=6)).model_id == "gpt-5.4"


def test_decision_reason_contains_rule_text() -> None:
    manifest = _manifest(
        rules=(RouterRule(when="turn.iteration > 1", use="gpt-5.4"),),
        fallback="gemma4:cloud",
    )
    router = LisaRouter(manifest)
    decision = router.choose(_req(iteration=2))
    assert decision is not None
    assert "turn.iteration > 1" in decision.reason


# ── Effective-model metadata tests ──────────────────────────────────────────


def test_model_cost_tier_condition() -> None:
    """model.cost_tier reflects the effective model from the preceding iteration."""
    from kernel.seams.protocols import RouterModelRef

    manifest = _manifest(
        rules=(
            RouterRule(
                when='model.cost_tier == "frontier" && turn.tool_calls_so_far >= 4',
                use="gemma4:cloud",
            ),
        ),
        fallback="gpt-5.4",
    )
    router = LisaRouter(manifest)
    effective_frontier = RouterModelRef(model_id="gpt-5.4", cost_tier="frontier")
    effective_standard = RouterModelRef(model_id="gemma4:cloud", cost_tier="standard")
    # frontier model + 4 tool calls → downgrade
    decision = router.choose(_req(tool_calls_so_far=4, effective_model=effective_frontier))
    assert decision is not None
    assert decision.model_id == "gemma4:cloud"
    # standard model + 4 tool calls → fallback
    decision = router.choose(_req(tool_calls_so_far=4, effective_model=effective_standard))
    assert decision is not None
    assert decision.model_id == "gpt-5.4"


def test_model_capabilities_condition() -> None:
    from kernel.seams.protocols import RouterModelRef

    manifest = _manifest(
        rules=(RouterRule(when='"vision" in model.capabilities', use="gpt-5.4"),),
        fallback="gemma4:cloud",
    )
    router = LisaRouter(manifest)
    effective_vision = RouterModelRef(model_id="gpt-5.4", capabilities=("vision", "text"))
    effective_no_vision = RouterModelRef(model_id="gpt-5.2-codex", capabilities=("code", "text"))
    assert router.choose(_req(effective_model=effective_vision)).model_id == "gpt-5.4"
    assert router.choose(_req(effective_model=effective_no_vision)).model_id == "gemma4:cloud"


def test_model_structured_reliability_condition() -> None:
    """model.structured_reliability reflects the effective model and lets a
    rule escalate JSON-bound work away from models that ignore
    ``response_format``."""
    from kernel.seams.protocols import RouterModelRef

    manifest = _manifest(
        rules=(RouterRule(when="model.structured_reliability < 0.8", use="deepseek-v4-pro:cloud"),),
        fallback="deepseek-v4-flash:0731-cloud",
    )
    router = LisaRouter(manifest)
    unreliable = RouterModelRef(
        model_id="deepseek-v4-flash:0731-cloud",
        structured_reliability=0.0,
    )
    reliable = RouterModelRef(
        model_id="deepseek-v4-pro:cloud",
        structured_reliability=0.8,
    )
    # Unreliable effective model → escalate.
    decision = router.choose(_req(effective_model=unreliable))
    assert decision is not None
    assert decision.model_id == "deepseek-v4-pro:cloud"
    # Reliable effective model → fallback.
    decision = router.choose(_req(effective_model=reliable))
    assert decision is not None
    assert decision.model_id == "deepseek-v4-flash:0731-cloud"


# ── Terminal action tests ───────────────────────────────────────────────────


def test_finish_action() -> None:
    manifest = _manifest(
        rules=(RouterRule(when="turn.tool_calls_so_far >= 8", finish="tool_budget_exhausted"),),
        fallback="gemma4:cloud",
    )
    router = LisaRouter(manifest)
    # Not enough tool calls → fallback
    decision = router.choose(_req(tool_calls_so_far=3))
    assert decision is not None
    assert decision.model_id == "gemma4:cloud"
    assert decision.finish == ""
    # 8 tool calls → finish action
    decision = router.choose(_req(tool_calls_so_far=8))
    assert decision is not None
    assert decision.finish == "tool_budget_exhausted"
    assert decision.model_id == ""
    assert decision.transition_to == ""


def test_transition_to_action() -> None:
    manifest = _manifest(
        rules=(RouterRule(when="turn.iteration >= 10", transition_to="summary"),),
        fallback="gemma4:cloud",
    )
    router = LisaRouter(manifest)
    # iteration 5 → fallback
    decision = router.choose(_req(iteration=5))
    assert decision is not None
    assert decision.model_id == "gemma4:cloud"
    assert decision.transition_to == ""
    # iteration 10 → transition_to action
    decision = router.choose(_req(iteration=10))
    assert decision is not None
    assert decision.transition_to == "summary"
    assert decision.model_id == ""
    assert decision.finish == ""


# ── CEL compilation error tests ─────────────────────────────────────────────


def test_invalid_cel_expression_raises_at_construction() -> None:
    """An invalid CEL expression fails at router construction, not at choose()."""
    manifest = _manifest(
        rules=(RouterRule(when="this is not valid cel !!!", use="gpt-5.4"),),
        fallback="gemma4:cloud",
    )
    with pytest.raises(ValueError, match="failed to compile CEL"):
        LisaRouter(manifest)


def test_cel_evaluation_error_treated_as_no_match() -> None:
    """A CEL runtime error (not a compile error) is logged and the rule is skipped."""
    manifest = _manifest(
        rules=(
            RouterRule(when="model.nonexistent_field == 1", use="gpt-5.4"),
            RouterRule(when="turn.iteration > 1", use="gpt-5.2-codex"),
        ),
        fallback="gemma4:cloud",
    )
    router = LisaRouter(manifest)
    # The first rule's evaluation error is skipped; the second rule matches.
    decision = router.choose(_req(iteration=2))
    assert decision is not None
    assert decision.model_id == "gpt-5.2-codex"


# ── Manifest validation tests ───────────────────────────────────────────────


def test_router_rule_requires_an_action() -> None:
    """A rule with no action (use/finish/transition_to) is rejected."""
    import tempfile

    import yaml

    from kernel.model_manifest import load_model_manifest
    from kernel.seams.protocols import RouterRequest  # noqa: F401

    payload = {
        "version": 1,
        "models": [{"model_id": "m1", "provider": "openai"}],
        "router": {
            "backend": "lisa-router",
            "rules": [{"when": "turn.iteration > 1"}],
        },
    }
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        yaml.dump(payload, f)
        f.flush()
        with pytest.raises(ValueError, match="must include one of"):
            load_model_manifest(__import__("pathlib").Path(f.name))


def test_router_rule_rejects_multiple_actions() -> None:
    """A rule with more than one action is rejected."""
    import pathlib
    import tempfile

    import yaml

    from kernel.model_manifest import load_model_manifest

    payload = {
        "version": 1,
        "models": [{"model_id": "m1", "provider": "openai"}],
        "router": {
            "backend": "lisa-router",
            "rules": [
                {"when": "turn.iteration > 1", "use": "m1", "finish": "done"},
            ],
        },
    }
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        yaml.dump(payload, f)
        f.flush()
        with pytest.raises(ValueError, match="exactly one"):
            load_model_manifest(pathlib.Path(f.name))


def test_manifest_parses_structured_reliability() -> None:
    """``structured_reliability`` flows from the manifest into the model entry."""
    import pathlib
    import tempfile

    import yaml

    from kernel.model_manifest import load_model_manifest

    payload = {
        "version": 1,
        "models": [
            {"model_id": "m1", "provider": "openai", "structured_reliability": 1.0},
            {"model_id": "m2", "provider": "ollama", "structured_reliability": 0.4},
            {"model_id": "m3", "provider": "ollama"},  # omitted → 0.0
        ],
        "mode_models": {"default": "m1"},
    }
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        yaml.dump(payload, f)
        f.flush()
        manifest = load_model_manifest(pathlib.Path(f.name))
    by_id = {model.model_id: model for model in manifest.models}
    assert by_id["m1"].structured_reliability == 1.0
    assert by_id["m2"].structured_reliability == 0.4
    assert by_id["m3"].structured_reliability == 0.0


def test_manifest_rejects_out_of_range_structured_reliability() -> None:
    """Values outside [0.0, 1.0] are rejected so a malformed manifest cannot
    silently over-claim a model's structured-output reliability."""
    import pathlib
    import tempfile

    import yaml

    from kernel.model_manifest import load_model_manifest

    payload = {
        "version": 1,
        "models": [{"model_id": "m1", "provider": "openai", "structured_reliability": 1.5}],
        "mode_models": {"default": "m1"},
    }
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        yaml.dump(payload, f)
        f.flush()
        with pytest.raises(ValueError, match="range"):
            load_model_manifest(pathlib.Path(f.name))
