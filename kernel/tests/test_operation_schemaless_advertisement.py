"""Regression tests: schema-less operations must stay advertised.

The 2026-09-23 finances incident: ``build_operation_tool_schemas`` silently
skipped any operation whose module defines no ``Parameters`` model, while
``mode_completion_effects`` still required those operations for mode
completion. Mode-bound, invocable, invisible — the model was ordered to
complete work with tools it could never see, and degraded to improvising
with whatever tools it could see (hand-building the beancount ledger via
raw storage tools).

These tests pin the contract: advertisement follows mode binding, not
schema presence. A schema-less operation advertises an empty-arguments
schema.
"""

from __future__ import annotations

from typing import Any

import pytest
from kernel.function_tools import build_strict_openai_function_tool
from kernel.models import WorkflowModeConfig
from kernel.operations import registry as registry_module
from kernel.operations.registry import (
    OperationDefinition,
    build_operation_tool_schemas,
    mode_completion_effects,
)
from pydantic import BaseModel


class _ParametersLessModule:
    """An implementation module that defines NO Parameters model."""

    @staticmethod
    async def execute(ctx: Any) -> Any:  # pragma: no cover - shape only
        return None


class _Parameters(BaseModel):
    """A minimal real Parameters model."""


class _WithParametersModule:
    """An implementation module that defines a Parameters model."""

    Parameters = _Parameters

    @staticmethod
    async def execute(ctx: Any) -> Any:  # pragma: no cover - shape only
        return None


def _definition(
    operation_id: str,
    *,
    effect: str = "read",
    parameters_schema: Any = None,
) -> OperationDefinition:
    return OperationDefinition(
        operation_id=operation_id,
        description=f"op {operation_id}",
        pack_id="test-pack",
        provider_id="test-pack",
        effect=effect,
        parameters_schema=parameters_schema,
        mode_allowlist=["incident"],
        implementation_module=_ParametersLessModule(),
    )


@pytest.fixture()
def seeded_catalog(monkeypatch: pytest.MonkeyPatch) -> dict[str, OperationDefinition]:
    """Seed the operation catalog with one schema-less and one schema'd op."""
    catalog = {
        "op_no_schema": _definition("op_no_schema", effect="write"),
        "op_with_schema": _definition(
            "op_with_schema",
            effect="write",
            parameters_schema=_Parameters,
        ),
    }
    monkeypatch.setattr(registry_module, "_OPERATION_CATALOG", catalog)
    return catalog


class TestSchemalessOperationsAreAdvertised:
    def test_operation_without_parameters_model_advertises(
        self, seeded_catalog: dict[str, OperationDefinition]
    ) -> None:
        """The incident scenario: a mode-bound op with no Parameters model."""
        schemas = build_operation_tool_schemas("incident", tools_authoritative=True)
        names = {s["function"]["name"] for s in schemas}
        assert "op_no_schema" in names, (
            "schema-less mode-bound operations must still advertise — "
            "the 2026-09-23 finances incident hid every finances op from "
            "the model while the completion contract required them"
        )

    def test_schemaless_op_advertises_empty_arguments(
        self, seeded_catalog: dict[str, OperationDefinition]
    ) -> None:
        """The empty-args schema is a valid object schema, not a stub."""
        schemas = build_operation_tool_schemas("incident", tools_authoritative=True)
        by_name = {s["function"]["name"]: s for s in schemas}
        parameters = by_name["op_no_schema"]["function"]["parameters"]
        assert isinstance(parameters, dict)
        assert parameters.get("type") == "object"

    def test_advertised_set_matches_mode_bound_set(
        self, seeded_catalog: dict[str, OperationDefinition]
    ) -> None:
        """Advertisement and binding stay aligned for schema-less ops.

        Guards the divergence at the heart of the incident: the completion
        contract required ops the schema builder refused to advertise.
        """
        schemas = build_operation_tool_schemas("incident", tools_authoritative=True)
        advertised = {s["function"]["name"] for s in schemas}
        bound = set(seeded_catalog)
        assert advertised == bound, (
            f"advertised {sorted(advertised)} != mode-bound {sorted(bound)}: "
            "advertisement and the completion contract must not diverge"
        )

    def test_completion_contract_still_includes_schemaless_writes(
        self,
        seeded_catalog: dict[str, OperationDefinition],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """mode_completion_effects never skipped schema-less ops — keep it that way."""
        from kernel import ipsar_fsm

        base_catalog = dict(ipsar_fsm.get_workflow_catalog())
        base_catalog["incident"] = WorkflowModeConfig(
            mode_id="incident",
            start_stage_id="none",
            mode_kind="action",
            routable=True,
        )
        monkeypatch.setattr(ipsar_fsm, "get_workflow_catalog", lambda: base_catalog)
        effects = mode_completion_effects("incident")
        assert "op_no_schema" in effects, (
            "the write-effect schema-less op must remain in the completion "
            "contract — this is the invariant the advertiser must match"
        )


class TestStrictFunctionToolEmptySchema:
    def test_none_parameters_builds_valid_schema(self) -> None:
        """build_strict_openai_function_tool(None) yields an empty object schema."""
        tool = build_strict_openai_function_tool(
            name="bare_op",
            description="An operation that takes no arguments.",
            parameters_schema=None,
        )
        parameters = tool["function"]["parameters"]
        assert isinstance(parameters, dict)
        assert parameters.get("type") == "object"
