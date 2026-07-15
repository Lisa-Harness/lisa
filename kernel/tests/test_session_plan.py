"""Focused coverage for strict SessionManifestV2 compilation."""

from __future__ import annotations

from pathlib import Path

import pytest

from kernel.session_plan import SessionPlanError, compile_session_plan

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
    include: [lisa-pack-core, lisa-pack-learning]
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
    assert plan.plan_digest == compile_session_plan(
        text=text,
        source_uri="workspace:session_context.md",
    ).plan_digest


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


def test_bundled_default_session_selects_duplicate_catalog_providers() -> None:
    default_session_path = (
        Path(__file__).resolve().parents[2]
        / "sdk"
        / "packs"
        / "core"
        / "sessions"
        / "default.md"
    )
    text = default_session_path.read_text(encoding="utf-8")

    plan = compile_session_plan(text=text, source_uri=str(default_session_path))

    assert plan.manifest.providers.modes == {"default": "lisa-pack-core"}
    assert plan.manifest.providers.tools == {"web_search": "lisa-pack-brave-search"}