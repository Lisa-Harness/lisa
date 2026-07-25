"""Tests for the built-in lisa-router rules engine."""

from __future__ import annotations

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
        assert "use" in rule


def test_default_router_config_uses_symbolic_refs() -> None:
    """Default rules use symbolic mode/role refs (default, deep), not literal ids.

    This lets the config work across any discovered model set — the manifest
    builder resolves the symbols at /models discover time.
    """
    from sdk.router.default_config import get_default_router_config

    config = get_default_router_config()
    uses = {rule["use"] for rule in config["rules"]}
    assert config["fallback"] in {"default", "deep"} or config["fallback"]
    # At least one rule uses a symbolic ref.
    assert uses & {"default", "deep"}


def test_factory_returns_router_when_backend_matches() -> None:
    from sdk.router import get_router_factory

    factory = get_router_factory()
    manifest = _manifest(rules=(), fallback="gemma4:cloud")
    router = factory(manifest, None)
    assert router is not None
    assert isinstance(router, LisaRouter)


# ── Rule evaluation tests ───────────────────────────────────────────────────


def test_required_capabilities_rule_matches() -> None:
    manifest = _manifest(
        rules=(RouterRule(when='required_capabilities includes "code"', use="gpt-5.2-codex"),),
        fallback="gemma4:cloud",
    )
    router = LisaRouter(manifest)
    decision = router.choose(_req(required_capabilities=("code", "text")))
    assert decision is not None
    assert decision.model_id == "gpt-5.2-codex"
    assert decision.routed_by == "lisa-router"


def test_required_capabilities_rule_does_not_match() -> None:
    manifest = _manifest(
        rules=(RouterRule(when='required_capabilities includes "code"', use="gpt-5.2-codex"),),
        fallback="gemma4:cloud",
    )
    router = LisaRouter(manifest)
    decision = router.choose(_req(required_capabilities=("text",)))
    assert decision is not None
    assert decision.model_id == "gemma4:cloud"


def test_iteration_gt_rule() -> None:
    manifest = _manifest(
        rules=(RouterRule(when="iteration > 1", use="gpt-5.4"),),
        fallback="gemma4:cloud",
    )
    router = LisaRouter(manifest)
    assert router.choose(_req(iteration=1)).model_id == "gemma4:cloud"
    assert router.choose(_req(iteration=2)).model_id == "gpt-5.4"


def test_compound_and_with_last_tool_result() -> None:
    manifest = _manifest(
        rules=(
            RouterRule(
                when="iteration > 1 and last_tool_result is not None",
                use="gpt-5.4",
            ),
        ),
        fallback="gemma4:cloud",
    )
    router = LisaRouter(manifest)
    # iteration 1, no tool result → fallback
    assert router.choose(_req(iteration=1, last_tool_result=None)).model_id == "gemma4:cloud"
    # iteration 2, no tool result → fallback
    assert router.choose(_req(iteration=2, last_tool_result=None)).model_id == "gemma4:cloud"
    # iteration 2, with tool result → rule matches
    assert router.choose(_req(iteration=2, last_tool_result="result")).model_id == "gpt-5.4"


def test_compound_or() -> None:
    manifest = _manifest(
        rules=(
            RouterRule(
                when='required_capabilities includes "code" or iteration > 5',
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
        rules=(RouterRule(when="tool_calls_so_far >= 3", use="gpt-5.4"),),
        fallback="gemma4:cloud",
    )
    router = LisaRouter(manifest)
    assert router.choose(_req(tool_calls_so_far=2)).model_id == "gemma4:cloud"
    assert router.choose(_req(tool_calls_so_far=3)).model_id == "gpt-5.4"


def test_no_match_no_fallback_returns_none() -> None:
    manifest = _manifest(
        rules=(RouterRule(when="iteration > 100", use="gpt-5.4"),),
        fallback="",
    )
    router = LisaRouter(manifest)
    assert router.choose(_req(iteration=1)) is None


def test_first_match_wins() -> None:
    manifest = _manifest(
        rules=(
            RouterRule(when="iteration > 1", use="gpt-5.4"),
            RouterRule(when="iteration > 5", use="gpt-5.2-codex"),
        ),
        fallback="gemma4:cloud",
    )
    router = LisaRouter(manifest)
    # iteration 2 → first rule matches
    assert router.choose(_req(iteration=2)).model_id == "gpt-5.4"
    # iteration 6 → first rule still matches (first-match-wins)
    assert router.choose(_req(iteration=6)).model_id == "gpt-5.4"


def test_decision_reason_contains_rule_text() -> None:
    manifest = _manifest(
        rules=(RouterRule(when="iteration > 1", use="gpt-5.4"),),
        fallback="gemma4:cloud",
    )
    router = LisaRouter(manifest)
    decision = router.choose(_req(iteration=2))
    assert decision is not None
    assert "iteration > 1" in decision.reason