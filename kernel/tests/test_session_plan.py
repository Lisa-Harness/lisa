"""Focused coverage for strict SessionManifestV2 compilation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from kernel.session_plan import (
    PackInstallSpec,
    SessionPlanError,
    compile_session_plan,
)

_SCHEMA_VERSION = 2
_SHA256_LENGTH = 64


def _manifest(*, extra: str = "", body: str = "# Primary Session\n\nAct precisely.\n") -> str:
    return f"""---
schema_version: 2
name: test-session
surface:
  type: personal_dm
  id: host-console
  memory_scope: testing
resources:
  packs:
    include: [lisa-pack-core, lisa-pack-memory]
    required: [lisa-pack-core]
  servers:
    exclude: [legacy-server]
providers:
  capabilities:
    memory: lisa-pack-mem0
routing:
  start_mode: default
context:
  lanes:
    - lane_id: operational
      min_tokens: 128
      max_tokens: 512
      render_order: 0
    - lane_id: history
      max_tokens: 256
      render_order: 10
  sources:
    - rule_id: alerts
      lane_id: operational
      evidence_types: [alert]
      priority: 100
    - rule_id: fallback
      lane_id: history
      fallback: true
{extra}---
{body}"""


def test_compile_session_plan_preserves_primary_instruction_and_plan_digest() -> None:
    text = _manifest()

    plan = compile_session_plan(text=text, source_uri="workspace:session_context.md")

    assert plan.manifest.schema_version == _SCHEMA_VERSION
    assert plan.manifest.resources.packs.required == ("lisa-pack-core",)
    assert plan.manifest.providers.capabilities == {"memory": "lisa-pack-mem0"}
    assert plan.instructions.primary_body == "# Primary Session\n\nAct precisely."
    assert len(plan.instructions.source_sha256) == _SHA256_LENGTH
    assert len(plan.plan_digest) == _SHA256_LENGTH
    assert (
        plan.plan_digest
        == compile_session_plan(
            text=text,
            source_uri="workspace:session_context.md",
        ).plan_digest
    )


def test_compile_session_plan_preserves_owner_pack_configuration() -> None:
    text = _manifest(
        extra=(
            "resources:\n"
            "  packs:\n"
            "    include: [lisa-pack-core, lisa-pack-coding]\n"
            "    configuration:\n"
            "      lisa-pack-coding:\n"
            "        workspace_root: .\n"
            "        checks:\n"
            "          - id: unit\n"
            "            argv: [uv, run, pytest, -q]\n"
        )
    )

    plan = compile_session_plan(text=text, source_uri="workspace:session_context.md")

    assert plan.manifest.resources.packs.configuration == {
        "lisa-pack-coding": {
            "workspace_root": ".",
            "checks": [{"id": "unit", "argv": ["uv", "run", "pytest", "-q"]}],
        }
    }


def test_compile_session_plan_rejects_legacy_and_unknown_front_matter() -> None:
    text = _manifest(extra="replay:\n  trigger_mode: manual\n")

    with pytest.raises(SessionPlanError, match="Extra inputs are not permitted"):
        compile_session_plan(text=text, source_uri="workspace:session_context.md")


def test_compile_session_plan_rejects_conflicting_resource_selection() -> None:
    text = _manifest(
        extra="resources:\n  packs:\n    include: [lisa-pack-core]\n    exclude: [lisa-pack-core]\n"
    )

    with pytest.raises(SessionPlanError, match="include and exclude overlap"):
        compile_session_plan(text=text, source_uri="workspace:session_context.md")


def test_compile_session_plan_accepts_sequential_connectors() -> None:
    """The ``routing.sequential_connectors`` list compiles into the plan."""
    text = _manifest(
        extra=("routing:\n  start_mode: default\n  sequential_connectors: [then, based on, once]\n")
    )

    plan = compile_session_plan(text=text, source_uri="workspace:session_context.md")

    assert plan.manifest.routing.sequential_connectors == ("then", "based on", "once")


def test_compile_session_plan_defaults_sequential_connectors_empty() -> None:
    """Omitting ``routing.sequential_connectors`` yields an empty tuple
    (the kernel falls back to its built-in list at runtime)."""
    plan = compile_session_plan(text=_manifest(), source_uri="workspace:session_context.md")

    assert plan.manifest.routing.sequential_connectors == ()


def test_compile_session_plan_accepts_resume_phrases() -> None:
    """The ``routing.resume_phrases`` list compiles into the plan."""
    text = _manifest(
        extra=(
            "routing:\n  start_mode: default\n  resume_phrases: [proceed, onward, carry forward]\n"
        )
    )

    plan = compile_session_plan(text=text, source_uri="workspace:session_context.md")

    assert plan.manifest.routing.resume_phrases == ("proceed", "onward", "carry forward")


def test_compile_session_plan_defaults_resume_phrases_empty() -> None:
    """Omitting ``routing.resume_phrases`` yields an empty tuple (the kernel
    falls back to its built-in list at runtime)."""
    plan = compile_session_plan(text=_manifest(), source_uri="workspace:session_context.md")

    assert plan.manifest.routing.resume_phrases == ()


def test_compile_session_plan_accepts_execution_request_block() -> None:
    text = _manifest(
        extra=(
            "execution:\n"
            "  max_model_calls: 12\n"
            "  max_capability_calls: 20\n"
            "  max_total_tokens: 40000\n"
            "  max_completion_tokens: 500000\n"
            "  max_wall_ms: 180000\n"
            "  max_session_tokens: 15000000\n"
            "  capabilities:\n"
            "    fetch_page:\n"
            "      max_invocations_per_turn: 6\n"
        )
    )

    plan = compile_session_plan(text=text, source_uri="workspace:session_context.md")

    assert plan.manifest.execution.max_model_calls == 12
    assert plan.manifest.execution.max_capability_calls == 20
    assert plan.manifest.execution.max_total_tokens == 40_000
    assert plan.manifest.execution.max_completion_tokens == 500_000
    assert plan.manifest.execution.max_wall_ms == 180_000
    assert plan.manifest.execution.max_session_tokens == 15_000_000
    assert plan.manifest.execution.capabilities["fetch_page"].max_invocations_per_turn == 6


def test_compile_session_plan_rejects_non_positive_execution_limits() -> None:
    text = _manifest(extra=("execution:\n  max_model_calls: 0\n"))

    with pytest.raises(SessionPlanError, match="greater than or equal to 1"):
        compile_session_plan(text=text, source_uri="workspace:session_context.md")


def test_compile_session_plan_rejects_context_rules_for_unknown_lanes() -> None:
    text = _manifest(
        extra=(
            "context:\n"
            "  lanes:\n"
            "    - lane_id: operational\n"
            "  sources:\n"
            "    - rule_id: invalid\n"
            "      lane_id: missing\n"
            "      fallback: true\n"
        )
    )

    with pytest.raises(SessionPlanError, match="unknown lanes"):
        compile_session_plan(text=text, source_uri="workspace:session_context.md")


def test_compile_session_plan_requires_nonempty_body() -> None:
    with pytest.raises(SessionPlanError, match="must not be empty"):
        compile_session_plan(
            text=_manifest(body="\n"),
            source_uri="workspace:session_context.md",
        )


def test_bundled_default_session_compiles_with_core_mode_provider() -> None:
    """The bundled default must compile and assign the core pack to the default mode.

    The bundled default intentionally does not declare ``providers.tools`` for
    ``web_search`` — that catalog collision belongs to the host's
    ``session_context.md``, not the core pack's bundled fallback.  The search
    pack ships its own graceful ``web_search`` stub, so the bundled default
    stays pack-encapsulated (no reference to external search provider packs).
    """
    default_session_path = (
        Path(__file__).resolve().parents[2] / "sdk" / "packs" / "core" / "sessions" / "default.md"
    )
    text = default_session_path.read_text(encoding="utf-8")

    plan = compile_session_plan(text=text, source_uri=str(default_session_path))

    assert plan.manifest.providers.modes == {"default": "lisa-pack-core"}
    assert plan.manifest.providers.tools == {}


# ── MemoryConsolidationTuning ──────────────────────────────────────────


def test_memory_consolidation_defaults_enabled_and_post_turn() -> None:
    """MemoryConsolidationTuning defaults to enabled, post_turn, safe thresholds.

    Consolidation is on by default (per the "all work items on by default"
    preference); the pressure/budget gates are conservative enough that it
    only fires when episodic pressure crosses the threshold.
    """
    from kernel.session_plan import MemoryConsolidationTuning

    tuning = MemoryConsolidationTuning()
    assert tuning.enabled is True
    assert tuning.trigger == "post_turn"
    assert tuning.pressure_threshold == 0.6
    assert tuning.hard_threshold == 0.85
    assert tuning.min_interval_turns == 8
    assert tuning.batch_size == 10
    assert tuning.min_episodic_count == 5
    assert tuning.max_records == 200
    assert tuning.consolidate_operation == "consolidate_semantic"


# ── PackSelection.auto_install ────────────────────────────────────────


def test_pack_install_spec_requires_nonempty_source() -> None:
    """PackInstallSpec.source must be a non-empty string."""
    with pytest.raises(ValidationError):
        PackInstallSpec.model_validate({"source": ""})


def test_compile_session_plan_accepts_auto_install_for_required_pack() -> None:
    """auto_install entries that match a required pack compile cleanly."""
    text = _manifest(
        extra=(
            "resources:\n"
            "  packs:\n"
            "    required: [lisa-pack-core, lisa-pack-cloudflare]\n"
            "    auto_install:\n"
            "      lisa-pack-cloudflare:\n"
            "        source: git+https://github.com/lisa-framework/lisa-pack-cloudflare.git@v0.2.0\n"
            "        pin: v0.2.0\n"
        )
    )
    plan = compile_session_plan(text=text, source_uri="workspace:session_context.md")
    auto = plan.manifest.resources.packs.auto_install
    assert "lisa-pack-cloudflare" in auto
    assert auto["lisa-pack-cloudflare"].source.startswith("git+https://")
    assert auto["lisa-pack-cloudflare"].pin == "v0.2.0"


def test_compile_session_plan_rejects_auto_install_key_not_in_required() -> None:
    """auto_install keys must be a subset of required."""
    text = _manifest(
        extra=(
            "resources:\n"
            "  packs:\n"
            "    required: [lisa-pack-core]\n"
            "    auto_install:\n"
            "      lisa-pack-cloudflare:\n"
            "        source: git+https://github.com/lisa-framework/lisa-pack-cloudflare.git@v0.2.0\n"
        )
    )
    with pytest.raises(SessionPlanError, match="auto_install keys must also appear in required"):
        compile_session_plan(text=text, source_uri="workspace:session_context.md")


def test_compile_session_plan_rejects_empty_auto_install_source() -> None:
    """auto_install entries require a non-empty source string."""
    text = _manifest(
        extra=(
            "resources:\n"
            "  packs:\n"
            "    required: [lisa-pack-core, lisa-pack-cloudflare]\n"
            "    auto_install:\n"
            "      lisa-pack-cloudflare:\n"
            '        source: ""\n'
        )
    )
    with pytest.raises(SessionPlanError):
        compile_session_plan(text=text, source_uri="workspace:session_context.md")


def test_compile_session_plan_rejects_unknown_auto_install_key() -> None:
    """Strict extra=forbid applies inside auto_install entries."""
    text = _manifest(
        extra=(
            "resources:\n"
            "  packs:\n"
            "    required: [lisa-pack-core, lisa-pack-cloudflare]\n"
            "    auto_install:\n"
            "      lisa-pack-cloudflare:\n"
            "        source: lisa-pack-cloudflare\n"
            "        bogus: true\n"
        )
    )
    with pytest.raises(SessionPlanError, match="Extra inputs are not permitted"):
        compile_session_plan(text=text, source_uri="workspace:session_context.md")


# ── ContextPlan.models (per-session model manifest path) ─────────────


def test_compile_session_plan_accepts_context_models_path() -> None:
    """context.models.path compiles when set to a non-empty string."""
    text = _manifest(extra=("context:\n  models:\n    path: ./roster.yaml\n"))
    plan = compile_session_plan(text=text, source_uri="workspace:session_context.md")
    assert plan.manifest.context.models is not None
    assert plan.manifest.context.models.path == "./roster.yaml"


def test_compile_session_plan_defaults_context_models_to_none() -> None:
    """Omitting context.models leaves it None (host falls back to data_dir/models.yaml)."""
    plan = compile_session_plan(text=_manifest(), source_uri="workspace:session_context.md")
    assert plan.manifest.context.models is None


def test_compile_session_plan_rejects_empty_context_models_path() -> None:
    """context.models.path must be non-empty."""
    text = _manifest(extra=('context:\n  models:\n    path: ""\n'))
    with pytest.raises(SessionPlanError):
        compile_session_plan(text=text, source_uri="workspace:session_context.md")


def test_compile_session_plan_rejects_unknown_context_models_key() -> None:
    """Strict extra=forbid applies inside context.models."""
    text = _manifest(extra=("context:\n  models:\n    path: ./roster.yaml\n    bogus: true\n"))
    with pytest.raises(SessionPlanError, match="Extra inputs are not permitted"):
        compile_session_plan(text=text, source_uri="workspace:session_context.md")
