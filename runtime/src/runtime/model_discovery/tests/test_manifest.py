"""Tests for manifest construction, rendering, and atomic writes."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import yaml

from runtime.model_discovery.manifest import (
    build_manifest_payload,
    render_manifest_yaml,
    write_manifest_yaml,
)
from runtime.model_discovery.types import DiscoveredModel


class TestBuildManifestPayload:
    def test_empty_models(self) -> None:
        payload = build_manifest_payload(())
        assert payload == {"version": 1, "models": []}

    def test_cloud_gpt5_cheapest_size_wins_default(self) -> None:
        """Cloud default = cheapest gpt-5-family model: nano < mini < full."""
        models = (
            DiscoveredModel(model_id="gpt-5", provider="openai"),
            DiscoveredModel(model_id="gpt-5-mini", provider="openai"),
            DiscoveredModel(model_id="gpt-5.4", provider="openai"),
        )
        payload = build_manifest_payload(models)
        assert payload["mode_models"] == {"default": "gpt-5-mini"}

    def test_cloud_gpt5_nano_beats_mini(self) -> None:
        """nano is cheaper than mini; both beat full gpt-5."""
        models = (
            DiscoveredModel(model_id="gpt-5", provider="openai"),
            DiscoveredModel(model_id="gpt-5-mini", provider="openai"),
            DiscoveredModel(model_id="gpt-5-nano", provider="openai"),
        )
        payload = build_manifest_payload(models)
        assert payload["mode_models"] == {"default": "gpt-5-nano"}

    def test_cloud_gpt5_full_when_no_mini_nano(self) -> None:
        """If only full gpt-5 variants exist, the lowest variant wins."""
        models = (
            DiscoveredModel(model_id="gpt-5.4", provider="openai"),
            DiscoveredModel(model_id="gpt-5", provider="openai"),
        )
        payload = build_manifest_payload(models)
        assert payload["mode_models"] == {"default": "gpt-5"}

    def test_cloud_falls_back_to_first_cloud_when_no_gpt5(self) -> None:
        """No gpt-5 family probed → first cloud model in discovery order."""
        models = (
            DiscoveredModel(model_id="claude-opus-4", provider="anthropic"),
            DiscoveredModel(model_id="gemini-2.5-pro", provider="google"),
        )
        payload = build_manifest_payload(models)
        assert payload["mode_models"] == {"default": "claude-opus-4"}

    def test_local_mid_range_wins_when_local_present(self) -> None:
        """Local present → default is the median by context_window (mid-range)."""
        models = (
            DiscoveredModel(model_id="small:7b", provider="ollama", context_window=8192),
            DiscoveredModel(model_id="big:70b", provider="ollama", context_window=65536),
            DiscoveredModel(model_id="mid:13b", provider="ollama", context_window=16384),
        )
        payload = build_manifest_payload(models)
        # Sorted by context_window: 8192, 16384, 65536 → median = mid:13b.
        assert payload["mode_models"] == {"default": "mid:13b"}

    def test_local_even_count_picks_lower_median(self) -> None:
        """Even count: the lower of the two middle entries wins (cheaper)."""
        models = (
            DiscoveredModel(model_id="a:1b", provider="ollama", context_window=2048),
            DiscoveredModel(model_id="b:7b", provider="ollama", context_window=8192),
            DiscoveredModel(model_id="c:13b", provider="ollama", context_window=16384),
            DiscoveredModel(model_id="d:70b", provider="ollama", context_window=65536),
        )
        payload = build_manifest_payload(models)
        # Sorted: 2048, 8192, 16384, 65536 → lower median = b:7b (8192).
        assert payload["mode_models"] == {"default": "b:7b"}

    def test_local_tiebreak_prefers_reasoning(self) -> None:
        """Equal context_window → reasoning_model wins the tie."""
        models = (
            DiscoveredModel(model_id="plain:70b", provider="ollama", context_window=65536),
            DiscoveredModel(
                model_id="reason:70b",
                provider="ollama",
                context_window=65536,
                reasoning_model=True,
            ),
        )
        payload = build_manifest_payload(models)
        # Two models, even count, lower median. Same context_window, reasoning
        # sorts earlier (not reasoning=False sorts first), so reason:70b wins.
        assert payload["mode_models"] == {"default": "reason:70b"}

    def test_local_wins_default_over_cloud_when_both_present(self) -> None:
        """Local (ollama) present → local mid-range wins as default, not cloud."""
        models = (
            DiscoveredModel(model_id="gpt-5-mini", provider="openai"),
            DiscoveredModel(model_id="small:7b", provider="ollama", context_window=8192),
            DiscoveredModel(model_id="big:70b", provider="ollama", context_window=65536),
        )
        payload = build_manifest_payload(models)
        # Two local models, even count, lower median = small:7b.
        assert payload["mode_models"] == {"default": "small:7b"}

    def test_emits_mode_models_no_designations(self) -> None:
        """Output shape: mode_models at top level; no per-model designations."""
        models = (
            DiscoveredModel(model_id="gpt-5", provider="openai"),
            DiscoveredModel(
                model_id="o3-pro",
                provider="openai",
                reasoning_model=True,
            ),
        )
        payload = build_manifest_payload(models)
        assert "mode_models" in payload
        assert "roles" not in payload
        for row in payload["models"]:
            assert "designations" not in row

    def test_dedup_same_provider_same_id(self) -> None:
        models = (
            DiscoveredModel(model_id="gpt-5", provider="openai"),
            DiscoveredModel(model_id="gpt-5", provider="openai"),
        )
        payload = build_manifest_payload(models)
        assert len(payload["models"]) == 1

    def test_dedup_case_insensitive(self) -> None:
        models = (
            DiscoveredModel(model_id="gpt-5", provider="openai"),
            DiscoveredModel(model_id="GPT-5", provider="openai"),
        )
        payload = build_manifest_payload(models)
        assert len(payload["models"]) == 1

    def test_dedup_different_providers_same_id_drops_second(self) -> None:
        """The manifest loader rejects same model_id across providers,
        so _dedupe must collapse them (keeping the first-seen entry)."""

        models = (
            DiscoveredModel(model_id="gpt-5", provider="openai"),
            DiscoveredModel(model_id="gpt-5", provider="azure_openai"),
        )
        payload = build_manifest_payload(models)
        assert len(payload["models"]) == 1
        assert payload["models"][0]["provider"] == "openai"

    # ── Pin viability (retirement awareness) ──────────────────────────
    #
    # Regression guards for the 2026-10 demo incident: from-scratch
    # discovery pinned a retired Ollama cloud tag as default. Expired,
    # soon-expiring, and batch-only models must never win a pin.

    def test_local_default_skips_expired_model(self) -> None:
        """An expired local model listed by the provider loses the default pin."""
        models = (
            DiscoveredModel(
                model_id="retired:cloud",
                provider="ollama",
                context_window=1048576,
                expires_at=date(2026, 9, 25),
            ),
            DiscoveredModel(model_id="alive:cloud", provider="ollama", context_window=1048576),
        )
        payload = build_manifest_payload(models)
        assert payload["mode_models"] == {"default": "alive:cloud"}
        # The expired model stays listed (with its expiry visible) — it
        # only loses pin eligibility.
        listed_ids = {row["model_id"] for row in payload["models"]}
        assert "retired:cloud" in listed_ids

    def test_local_default_skips_soon_expiring_model(self) -> None:
        """A model dying within the pin horizon is not pinned."""
        soon = date.today() + timedelta(days=7)
        models = (
            DiscoveredModel(
                model_id="dying:cloud",
                provider="ollama",
                context_window=262144,
                expires_at=soon,
            ),
            DiscoveredModel(model_id="stable:cloud", provider="ollama", context_window=262144),
        )
        payload = build_manifest_payload(models)
        assert payload["mode_models"] == {"default": "stable:cloud"}

    def test_far_future_expiry_stays_pin_viable(self) -> None:
        """A model expiring beyond the horizon is still pin-eligible."""
        far = date.today() + timedelta(days=200)
        models = (
            DiscoveredModel(
                model_id="stable:cloud",
                provider="ollama",
                context_window=262144,
                expires_at=far,
            ),
        )
        payload = build_manifest_payload(models)
        assert payload["mode_models"] == {"default": "stable:cloud"}

    def test_cloud_default_skips_batch_variant(self) -> None:
        """A ``:batch`` (Batch-API-only) twin never wins the cloud default."""
        models = (
            DiscoveredModel(model_id="openai/gpt-5.4-nano:batch", provider="openrouter"),
            DiscoveredModel(model_id="openai/gpt-5.4-nano", provider="openrouter"),
            DiscoveredModel(model_id="openai/gpt-5.4-mini", provider="openrouter"),
        )
        payload = build_manifest_payload(models)
        assert payload["mode_models"] == {"default": "openai/gpt-5.4-nano"}

    def test_cloud_default_prefers_cheaper_variant_with_prefix(self) -> None:
        """Provider-prefixed gpt-5 ids rank by variant, not discovery order.

        The old rank parser could not see behind ``openai/`` — every
        prefixed variant collapsed to (tier, 5.0), so discovery order
        decided the cloud default.
        """
        models = (
            DiscoveredModel(model_id="openai/gpt-5.4-mini", provider="openrouter"),
            DiscoveredModel(model_id="openai/gpt-5.2-mini", provider="openrouter"),
            DiscoveredModel(model_id="openai/gpt-5.4", provider="openrouter"),
        )
        payload = build_manifest_payload(models)
        # Same size tier (mini); lower variant 5.2 wins over 5.4.
        assert payload["mode_models"] == {"default": "openai/gpt-5.2-mini"}

    def test_expired_only_set_falls_back_to_listing_default(self) -> None:
        """When every model is expired, discovery still names a default
        rather than emitting an unusable manifest."""
        models = (
            DiscoveredModel(
                model_id="retired-a:cloud",
                provider="ollama",
                context_window=1048576,
                expires_at=date(2026, 9, 25),
            ),
            DiscoveredModel(
                model_id="retired-b:cloud",
                provider="ollama",
                context_window=262144,
                expires_at=date(2026, 9, 25),
            ),
        )
        payload = build_manifest_payload(models)
        # A default is still pinned (to the mid-range of the expired set)
        # so the manifest stays loadable; the operator sees the expiry
        # dates in the emitted rows and the probe's dropped note.
        assert payload["mode_models"]["default"] in {"retired-a:cloud", "retired-b:cloud"}

    def test_code_pin_skips_retired_model(self) -> None:
        """Preset annotations never resolve to a retired code model."""
        from runtime.model_discovery.manifest import _pick_code_model

        models = (
            DiscoveredModel(
                model_id="retired-coder:cloud",
                provider="ollama",
                capabilities=("text", "tools", "code"),
                expires_at=date(2026, 9, 25),
            ),
            DiscoveredModel(
                model_id="alive-coder:cloud",
                provider="ollama",
                capabilities=("text", "tools", "code"),
            ),
        )
        assert _pick_code_model(models, "fallback") == "alive-coder:cloud"

    def test_reasoning_pin_skips_retired_model(self) -> None:
        """Preset annotations never resolve to a retired reasoning model."""
        from runtime.model_discovery.manifest import _pick_reasoning_model

        models = (
            DiscoveredModel(
                model_id="retired-thinker:cloud",
                provider="ollama",
                reasoning_model=True,
                expires_at=date(2026, 9, 25),
            ),
            DiscoveredModel(
                model_id="alive-thinker:cloud",
                provider="ollama",
                reasoning_model=True,
            ),
        )
        fallback = models[1]
        picked = _pick_reasoning_model(models, fallback)
        assert picked.model_id == "alive-thinker:cloud"

    def test_expires_at_emitted_on_model_rows(self) -> None:
        """Upcoming expiry dates are written on the emitted model rows so
        operators see a model's retirement horizon in the proposal."""
        expiry = date.today() + timedelta(days=60)
        models = (
            DiscoveredModel(model_id="dying-soon:cloud", provider="ollama", expires_at=expiry),
        )
        payload = build_manifest_payload(models)
        assert payload["models"][0]["expires_at"] == expiry.isoformat()


class TestRouterBlock:
    """Tests for the pack-seeded ``router:`` block in ``build_manifest_payload``.

    The router block is seeded from the first pack-declared router config
    (``lisa.extensions.model_router_configs``). Symbolic refs (``"default"``)
    are resolved to concrete model ids. Rules referencing unknown
    models are stripped. When no pack declares a config, no ``router`` key
    is emitted.
    """

    def test_no_router_block_when_no_pack_config(self) -> None:
        """When no pack declares a router config, no router key is emitted."""
        models = (DiscoveredModel(model_id="gpt-5", provider="openai"),)
        with patch(
            "kernel.model_router_config_discovery.discover_pack_router_config",
            return_value=None,
        ):
            payload = build_manifest_payload(models)
        assert "router" not in payload

    def test_router_block_seeded_from_pack_config(self) -> None:
        """Pack config with symbolic refs resolved to concrete model ids."""
        models = (
            DiscoveredModel(model_id="gpt-5", provider="openai"),
            DiscoveredModel(model_id="o3-pro", provider="openai", reasoning_model=True),
        )
        pack_config = {
            "backend": "lisa-router",
            "rules": [
                {"when": '"code" in route.required_capabilities', "use": "default"},
            ],
        }
        with patch(
            "kernel.model_router_config_discovery.discover_pack_router_config",
            return_value=pack_config,
        ):
            payload = build_manifest_payload(models)
        assert "router" in payload
        router = payload["router"]
        assert router["backend"] == "lisa-router"
        # Symbolic ref resolved: "default" → gpt-5.
        assert router["rules"][0]["use"] == "gpt-5"

    def test_router_block_preserves_finish_action(self) -> None:
        """A ``finish`` action in the pack config is preserved, not dropped."""
        models = (DiscoveredModel(model_id="gpt-5", provider="openai"),)
        pack_config = {
            "backend": "lisa-router",
            "rules": [
                {"when": '"code" in route.required_capabilities', "use": "gpt-5"},
                {"when": "turn.tool_calls_so_far >= 8", "finish": "tool_budget_exhausted"},
            ],
        }
        with patch(
            "kernel.model_router_config_discovery.discover_pack_router_config",
            return_value=pack_config,
        ):
            payload = build_manifest_payload(models)
        router = payload["router"]
        assert len(router["rules"]) == 2
        assert router["rules"][0]["use"] == "gpt-5"
        assert router["rules"][1]["finish"] == "tool_budget_exhausted"
        assert "use" not in router["rules"][1]

    def test_router_block_preserves_transition_to_action(self) -> None:
        """A ``transition_to`` action in the pack config is preserved."""
        models = (DiscoveredModel(model_id="gpt-5", provider="openai"),)
        pack_config = {
            "backend": "lisa-router",
            "rules": [
                {"when": "turn.iteration >= 10", "transition_to": "summary"},
            ],
        }
        with patch(
            "kernel.model_router_config_discovery.discover_pack_router_config",
            return_value=pack_config,
        ):
            payload = build_manifest_payload(models)
        router = payload["router"]
        assert len(router["rules"]) == 1
        assert router["rules"][0]["transition_to"] == "summary"
        assert "use" not in router["rules"][0]

    def test_router_block_drops_rules_with_multiple_actions(self) -> None:
        """A rule with multiple actions is dropped (the loader would reject it)."""
        models = (DiscoveredModel(model_id="gpt-5", provider="openai"),)
        pack_config = {
            "backend": "lisa-router",
            "rules": [
                {"when": "turn.iteration > 1", "use": "gpt-5", "finish": "done"},
            ],
        }
        with patch(
            "kernel.model_router_config_discovery.discover_pack_router_config",
            return_value=pack_config,
        ):
            payload = build_manifest_payload(models)
        router = payload["router"]
        # The multi-action rule is dropped; rules key is absent when empty.
        assert len(router.get("rules", [])) == 0

    def test_router_strips_rules_referencing_unknown_models(self) -> None:
        """Rules whose ``use`` cannot be resolved are dropped."""
        models = (
            DiscoveredModel(model_id="gpt-5", provider="openai"),
            DiscoveredModel(model_id="o3-pro", provider="openai", reasoning_model=True),
        )
        pack_config = {
            "backend": "lisa-router",
            "rules": [
                {"when": "turn.iteration > 1", "use": "default"},  # resolvable
                {"when": "turn.tool_calls_so_far >= 3", "use": "gpt-99-nonexistent"},  # stripped
            ],
        }
        with patch(
            "kernel.model_router_config_discovery.discover_pack_router_config",
            return_value=pack_config,
        ):
            payload = build_manifest_payload(models)
        router = payload["router"]
        assert len(router["rules"]) == 1
        assert router["rules"][0]["use"] == "gpt-5"

    def test_router_strips_fallback_referencing_unknown_model(self) -> None:
        """Fallback referencing an unknown model is dropped (not the whole block)."""
        models = (DiscoveredModel(model_id="gpt-5", provider="openai"),)
        pack_config = {
            "backend": "lisa-router",
            "rules": [{"when": "iteration > 1", "use": "default"}],
        }
        with patch(
            "kernel.model_router_config_discovery.discover_pack_router_config",
            return_value=pack_config,
        ):
            payload = build_manifest_payload(models)
        router = payload["router"]
        assert "fallback" not in router
        assert len(router["rules"]) == 1

    def test_router_block_dropped_when_no_rules_survive(self) -> None:
        """When all rules are stripped and no fallback resolves, no router key."""
        models = (DiscoveredModel(model_id="gpt-5", provider="openai"),)
        pack_config = {
            "backend": "lisa-router",
            "rules": [{"when": "iteration > 1", "use": "gpt-99-nonexistent"}],
        }
        with patch(
            "kernel.model_router_config_discovery.discover_pack_router_config",
            return_value=pack_config,
        ):
            payload = build_manifest_payload(models)
        # No rules survive, no fallback, but backend is set. The block is
        # still emitted (backend alone is valid — just selects the router
        # backend with no rules; the router will always return None).
        assert "router" in payload
        assert payload["router"]["backend"] == "lisa-router"
        assert "rules" not in payload["router"]

    def test_router_block_resolves_literal_model_id_refs(self) -> None:
        """Literal model ids in use/fallback pass through when in the discovered set."""
        models = (
            DiscoveredModel(model_id="gpt-5", provider="openai"),
            DiscoveredModel(model_id="gpt-5-mini", provider="openai"),
        )
        pack_config = {
            "backend": "lisa-router",
            "rules": [{"when": "iteration > 1", "use": "gpt-5-mini"}],
        }
        with patch(
            "kernel.model_router_config_discovery.discover_pack_router_config",
            return_value=pack_config,
        ):
            payload = build_manifest_payload(models)
        router = payload["router"]
        assert router["rules"][0]["use"] == "gpt-5-mini"

    def test_discover_regenerates_router_block_from_pack_default(self) -> None:
        """``/models discover`` regenerates the router block from the pack default
        on every run — it does not preserve a previously hand-edited block.

        Locks the A26 contract: ``build_manifest_payload`` has no
        ``existing_router`` input and always seeds from the pack-declared
        config. "User wins" means editing the active ``models.yaml`` directly
        or the proposal before ``/models apply`` — not re-running discover.
        """
        models = (DiscoveredModel(model_id="gpt-5", provider="openai"),)
        pack_config = {
            "backend": "lisa-router",
            "rules": [{"when": '"code" in route.required_capabilities', "use": "default"}],
        }
        # ``build_manifest_payload`` takes no active-manifest argument, so a
        # prior hand-edited router block cannot be passed in — the pack
        # default is the only source. Two consecutive calls yield the same
        # block, proving regenerate-from-scratch (no merge with prior state).
        with patch(
            "kernel.model_router_config_discovery.discover_pack_router_config",
            return_value=pack_config,
        ):
            first = build_manifest_payload(models)["router"]
        with patch(
            "kernel.model_router_config_discovery.discover_pack_router_config",
            return_value=pack_config,
        ):
            second = build_manifest_payload(models)["router"]
        assert first == second
        assert first["rules"][0]["use"] == "gpt-5"


class TestRouterPresetAnnotations:
    """Tests for ``_router_preset_annotations`` and annotation injection."""

    def _resolved_router_block(self) -> dict:
        """A representative resolved router block for annotation tests."""
        return {
            "backend": "lisa-router",
            "rules": [
                {"when": '"code" in route.required_capabilities', "use": "o3-pro"},
                {"when": "turn.tool_calls_so_far >= 5", "use": "gpt-5"},
                {"when": "turn.iteration > 1 && turn.has_tool_result", "use": "o3-pro"},
            ],
        }

    def test_includes_four_presets(self) -> None:
        """The annotation string names all four presets (including terminal-actions)."""
        from runtime.model_discovery.manifest import _router_preset_annotations

        models = (DiscoveredModel(model_id="gpt-5", provider="openai"),)
        annotations = _router_preset_annotations(
            mode_models={"default": "gpt-5"},
            models=models,
            router_block=self._resolved_router_block(),
        )
        assert "cost-conscious" in annotations
        assert "quality-first" in annotations
        assert "coding-focused" in annotations
        assert "terminal-actions" in annotations

    def test_render_annotates_active_router_inline(self) -> None:
        """The active router block is rendered without inline rule comments.

        Inline per-rule explanation trailers were removed from the rendered
        YAML (the annotation comments duplicated the rule predicates and
        complicated downstream parsing). The active block still appears
        with its rules and fallback.
        """
        payload = build_manifest_payload(
            (
                DiscoveredModel(model_id="gpt-5", provider="openai"),
                DiscoveredModel(model_id="o3-pro", provider="openai", reasoning_model=True),
            )
        )
        router_block = payload.get("router")
        assert router_block is not None
        rendered = render_manifest_yaml(payload, router_block=router_block)
        # The active block is present with its rules; the removed fallback
        # key must not be rendered.
        assert "rules:" in rendered
        assert "fallback:" not in rendered
        # No inline explanation trailers remain on rule lines.
        assert "→" not in rendered

    def test_includes_best_for_lines(self) -> None:
        """Each preset carries a 'Best for:' guidance line."""
        from runtime.model_discovery.manifest import _router_preset_annotations

        models = (DiscoveredModel(model_id="gpt-5", provider="openai"),)
        annotations = _router_preset_annotations(
            mode_models={"default": "gpt-5"},
            models=models,
            router_block=self._resolved_router_block(),
        )
        assert "Best for:" in annotations

    def test_uses_resolved_model_ids(self) -> None:
        """Annotation text contains the concrete model ids from mode_models/roles."""
        from runtime.model_discovery.manifest import _router_preset_annotations

        models = (
            DiscoveredModel(model_id="gpt-5-chat", provider="openai"),
            DiscoveredModel(model_id="o3-pro", provider="openai", reasoning_model=True),
        )
        annotations = _router_preset_annotations(
            mode_models={"default": "gpt-5-chat"},
            models=models,
            router_block=self._resolved_router_block(),
        )
        assert "gpt-5-chat" in annotations
        assert "o3-pro" in annotations

    def test_picks_code_model_from_capabilities(self) -> None:
        """When a model has ``code`` capability, it is used for the coding preset."""
        from runtime.model_discovery.manifest import _router_preset_annotations

        models = (
            DiscoveredModel(model_id="gpt-5", provider="openai"),
            DiscoveredModel(
                model_id="gpt-5.2-codex",
                provider="openai",
                capabilities=("code", "text"),
            ),
        )
        annotations = _router_preset_annotations(
            mode_models={"default": "gpt-5"},
            models=models,
            router_block=self._resolved_router_block(),
        )
        assert "gpt-5.2-codex" in annotations

    def test_returns_empty_when_no_mode_models(self) -> None:
        """Empty string when mode_models is missing."""
        from runtime.model_discovery.manifest import _router_preset_annotations

        models = (DiscoveredModel(model_id="gpt-5", provider="openai"),)
        assert (
            _router_preset_annotations({}, models, router_block=self._resolved_router_block()) == ""
        )

    def test_no_active_config_section_in_annotations(self) -> None:
        """The annotation string only carries presets, not the active mirror.

        The active ``router:`` block is annotated inline by
        ``_annotated_router_block`` via ``render_manifest_yaml``; the
        ``_router_preset_annotations`` helper only emits the alternatives.
        """
        from runtime.model_discovery.manifest import _router_preset_annotations

        models = (DiscoveredModel(model_id="gpt-5", provider="openai"),)
        annotations = _router_preset_annotations(
            mode_models={"default": "gpt-5"},
            models=models,
            router_block=None,
        )
        assert "Active config" not in annotations
        # Presets are still present.
        assert "cost-conscious" in annotations

    def test_render_manifest_yaml_injects_annotations(self) -> None:
        """Annotation text appears in the rendered output."""
        payload = {"version": 1, "models": []}
        annotations = "# test annotation\n"
        rendered = render_manifest_yaml(payload, annotations=annotations)
        assert "# test annotation" in rendered

    def test_render_manifest_yaml_injects_before_models(self) -> None:
        """Annotations are injected before the ``models:`` list, not at the end."""
        payload = build_manifest_payload((DiscoveredModel(model_id="gpt-5", provider="openai"),))
        annotations = "# MARKER\n"
        rendered = render_manifest_yaml(payload, annotations=annotations)
        marker_idx = rendered.index("# MARKER")
        # Find the top-level ``models:`` line (not the substring inside
        # ``mode_models:``).
        models_line_idx = next(
            i for i, ln in enumerate(rendered.split("\n")) if ln.rstrip() == "models:"
        )
        models_char_idx = sum(len(ln) + 1 for ln in rendered.split("\n")[:models_line_idx])
        assert marker_idx < models_char_idx, "annotations must appear before the models: list"

    def test_render_manifest_yaml_annotates_router_inline(self) -> None:
        """The active router block is rendered without inline rule comments.

        Inline per-rule comments were removed; the block still appears at
        the top with rules and fallback, before the models list.
        """
        payload = build_manifest_payload((DiscoveredModel(model_id="gpt-5", provider="openai"),))
        router_block = payload.get("router")
        assert router_block is not None
        rendered = render_manifest_yaml(payload, router_block=router_block)
        # The active block appears at the top.
        assert "router:" in rendered
        # No inline comment markers remain on rule lines.
        assert "→" not in rendered
        # The annotated block sits before the models list. Match the whole
        # ``models:`` line so we don't hit the substring in ``mode_models:``.
        lines = rendered.split("\n")
        router_line_idx = next(i for i, ln in enumerate(lines) if ln.rstrip() == "router:")
        models_line_idx = next(i for i, ln in enumerate(lines) if ln.rstrip() == "models:")
        assert router_line_idx < models_line_idx

    def test_render_manifest_yaml_no_duplicate_router_block(self) -> None:
        """When router_block is passed, the dumper's plain block is replaced."""
        payload = build_manifest_payload((DiscoveredModel(model_id="gpt-5", provider="openai"),))
        router_block = payload.get("router")
        rendered = render_manifest_yaml(payload, router_block=router_block)
        # Exactly one ``router:`` top-level key (the annotated one).
        assert rendered.count("\nrouter:\n") == 1

    def test_render_manifest_yaml_no_annotations_when_none(self) -> None:
        """No annotation text when annotations is None."""
        payload = {"version": 1, "models": []}
        rendered = render_manifest_yaml(payload)
        assert "# test annotation" not in rendered

    def test_write_manifest_yaml_with_annotations(self, tmp_path: Path) -> None:
        """Annotations are written to the proposal file."""
        path = tmp_path / "models.proposed.yaml"
        payload = build_manifest_payload((DiscoveredModel(model_id="gpt-5", provider="openai"),))
        annotations = "# preset annotations\n"
        write_manifest_yaml(path, payload, annotations=annotations)
        content = path.read_text(encoding="utf-8")
        assert "preset annotations" in content


class TestRenderManifestYaml:
    def test_roundtrip(self) -> None:
        models = (
            DiscoveredModel(model_id="gpt-4o", provider="openai"),
            DiscoveredModel(model_id="o3-mini", provider="openai", reasoning_model=True),
        )
        payload = build_manifest_payload(models)
        rendered = render_manifest_yaml(payload)
        assert rendered.startswith("# generated-by: lisa /models discover")
        parsed = yaml.safe_load(rendered)
        assert parsed == payload

    def test_is_valid_yaml(self) -> None:
        payload = {"version": 1, "models": []}
        rendered = render_manifest_yaml(payload)
        parsed = yaml.safe_load(rendered)
        assert parsed == payload


class TestWriteManifestYaml:
    def test_atomic_write(self, tmp_path: Path) -> None:
        path = tmp_path / "models.proposed.yaml"
        payload = build_manifest_payload((DiscoveredModel(model_id="gpt-4o", provider="openai"),))
        write_manifest_yaml(path, payload)
        assert path.exists()
        assert not (tmp_path / "models.proposed.yaml.tmp").exists()
        content = path.read_text(encoding="utf-8")
        assert "gpt-4o" in content

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "deep" / "models.proposed.yaml"
        write_manifest_yaml(path, {"version": 1, "models": []})
        assert path.exists()

    def test_overwrites_existing(self, tmp_path: Path) -> None:
        path = tmp_path / "models.proposed.yaml"
        path.write_text("old", encoding="utf-8")
        write_manifest_yaml(
            path,
            build_manifest_payload((DiscoveredModel(model_id="new-model", provider="openai"),)),
        )
        assert "new-model" in path.read_text(encoding="utf-8")
        assert "old" not in path.read_text(encoding="utf-8")
