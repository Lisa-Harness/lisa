"""Conformance tests for provider argument normalization.

Replays the nine incident shapes from
``docs/SPEC/PROVIDER_ARGUMENT_NORMALIZATION.md`` (~252 rejected dispatches
in nine days) and pins the R1-R4 contract: encodings normalize with zero
rejections and dispatch-equivalent payloads, over-cap strings reject with
intent-naming messages instead of truncating, required fields are never
touched, strict schemas opt out entirely, and every coercion lands one
warning plus a ``normalized_fields`` stamp on the tool telemetry row.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest
from kernel.models import (
    OperationDefinition,
    OperationExecutionContext,
    ToolDefinition,
    ToolExecutionContext,
)
from kernel.operations.registry import (
    OperationExecutionError,
)
from kernel.operations.registry import (
    _validate_parameters as validate_operation_parameters,
)
from kernel.tools.parameter_normalization import (
    describe_validation_failure,
    normalize_payload,
)
from kernel.tools.registry import (
    ToolExecutionError,
    _publish_tool_telemetry,
)
from kernel.tools.registry import (
    _validate_parameters as validate_tool_parameters,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class SearchCodeParams(BaseModel):
    """The optional-sentinel shape behind the 48-incident bucket."""

    query: str
    path: str = ""
    artifact_id: str | None = None
    max_results: int = Field(default=25, ge=1, le=200)


class RecordFactParams(BaseModel):
    """The over-cap string shape behind the 44-incident bucket."""

    claim: str = Field(max_length=200)
    revision: str | None = None
    base: str | None = None


class ReadFileParams(BaseModel):
    path: str
    artifact_id: str | None = None
    revision: str | None = None


class GitLogParams(BaseModel):
    artifact_id: str | None = None
    base: str | None = None


class EditFilesParams(BaseModel):
    artifact_id: str | None = None
    edits: list[dict[str, Any]] | None = None


class FetchPageParams(BaseModel):
    url: str
    max_chars: int | None = Field(default=12000, ge=500, le=20000)


class BashRunParams(BaseModel):
    commands: list[str] | None = None


class StrictParams(BaseModel):
    model_config = ConfigDict(strict_parameters=True)

    artifact_id: str | None = None


def _tool_definition(
    schema: type[BaseModel],
    tool_id: str = "tool_under_test",
) -> ToolDefinition:
    return ToolDefinition.model_construct(tool_id=tool_id, parameters_schema=schema)


def _operation_definition(
    schema: type[BaseModel],
    operation_id: str = "operation_under_test",
) -> OperationDefinition:
    return OperationDefinition.model_construct(operation_id=operation_id, parameters_schema=schema)


@pytest.fixture()
def published(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture both registries' fire-and-forget telemetry publishes."""

    rows: list[dict[str, Any]] = []

    def _capture(event_bus: Any, topic: str, event: Any, *, source: str = "") -> None:
        rows.append(dict(event))

    monkeypatch.setattr("kernel.tools.registry.schedule_telemetry_publish", _capture)
    monkeypatch.setattr("kernel.operations.registry.schedule_telemetry_publish", _capture)
    return rows


# --- R1: optional-field null sentinels (shapes 1-3, 6: 48+35+14+12+7) ---


def test_r1_null_string_sentinel_becomes_none() -> None:
    raw = {"query": "provider normalization", "artifact_id": "null"}
    result = normalize_payload(SearchCodeParams, raw, target_id="coding_search_code")
    assert result.normalized_fields == ("artifact_id",)
    assert result.payload["artifact_id"] is None
    assert SearchCodeParams.model_validate(result.payload).artifact_id is None


def test_r1_empty_string_sentinel_becomes_none() -> None:
    raw = {"path": "kernel/src/kernel/models.py", "artifact_id": ""}
    result = normalize_payload(ReadFileParams, raw, target_id="coding_read_file")
    assert result.payload["artifact_id"] is None
    ReadFileParams.model_validate(result.payload)


def test_r1_revision_and_base_sentinels() -> None:
    result = normalize_payload(ReadFileParams, {"path": "x", "revision": "", "artifact_id": "null"})
    assert result.normalized_fields == ("artifact_id", "revision")
    git = normalize_payload(
        GitLogParams, {"base": "null", "artifact_id": "None"}, target_id="coding_git_log"
    )
    assert git.payload == {"base": None, "artifact_id": None}
    GitLogParams.model_validate(git.payload)


def test_r1_required_fields_are_never_coerced() -> None:
    result = normalize_payload(RecordFactParams, {"claim": ""})
    assert result.payload["claim"] == ""
    assert result.normalized_fields == ()


# --- R2: string-encoded arrays and shape drift (shape 7: ~44 + 7) ---


def test_r2_string_encoded_array_with_the_composite_incident_shape() -> None:
    # The coding_edit_files bucket verbatim: string-encoded array, flat
    # record, empty content, "null" digest — dispatched, not rejected.
    edits = json.dumps([{"path": "a.txt", "content": "", "expected_sha256": "null"}])
    raw = {"artifact_id": "98df14d6", "edits": edits}
    result = normalize_payload(EditFilesParams, raw, target_id="coding_edit_files")
    assert result.normalized_fields == ("edits",)
    assert result.payload["edits"] == [{"path": "a.txt", "content": "", "expected_sha256": "null"}]
    EditFilesParams.model_validate(result.payload)


def test_r2_flat_single_record_lifts_to_one_element_list() -> None:
    raw = {"edits": json.dumps({"path": "a.txt", "content": "x"})}
    result = normalize_payload(EditFilesParams, raw)
    assert result.payload["edits"] == [{"path": "a.txt", "content": "x"}]


def test_r2_python_literal_and_bare_line_fallbacks() -> None:
    literal = normalize_payload(BashRunParams, {"commands": "['ls -la', 'git status']"})
    assert literal.payload["commands"] == ["ls -la", "git status"]
    bare = normalize_payload(BashRunParams, {"commands": "git status"})
    assert bare.payload["commands"] == ["git status"]


def test_r2_ambiguous_prose_is_left_for_pydantic() -> None:
    result = normalize_payload(BashRunParams, {"commands": "line one\nline two"})
    assert result.payload["commands"] == "line one\nline two"
    assert result.normalized_fields == ()
    with pytest.raises(ValidationError):
        BashRunParams.model_validate(result.payload)


# --- R3: bounded numerics (shape 8: 9 fetch_page incidents) ---


def test_r3_out_of_range_numeric_clamps_to_declared_bounds() -> None:
    high = normalize_payload(
        FetchPageParams,
        {"url": "https://example.com", "max_chars": 1_200_000},
        target_id="fetch_page",
    )
    assert high.payload["max_chars"] == 20000
    assert high.normalized_fields == ("max_chars",)
    low = normalize_payload(FetchPageParams, {"url": "https://example.com", "max_chars": 10})
    assert low.payload["max_chars"] == 500


def test_r3_numeric_strings_dispatch_and_garbage_rejects() -> None:
    coerced = normalize_payload(FetchPageParams, {"url": "u", "max_chars": "12000"})
    assert FetchPageParams.model_validate(coerced.payload).max_chars == 12000
    garbage = normalize_payload(FetchPageParams, {"url": "u", "max_chars": "12,000"})
    assert garbage.normalized_fields == ()
    with pytest.raises(ValidationError):
        FetchPageParams.model_validate(garbage.payload)


# --- R4: over-cap strings reject with intent-naming messages (shape 5) ---


def test_r4_over_cap_string_is_never_truncated() -> None:
    raw = {"claim": "x" * 500}
    result = normalize_payload(RecordFactParams, raw, target_id="coding_record_fact")
    assert result.payload["claim"] == "x" * 500
    assert result.normalized_fields == ()


def test_r4_rejection_message_names_field_cap_and_remedy() -> None:
    raw = {"claim": "x" * 500}
    result = normalize_payload(RecordFactParams, raw)
    with pytest.raises(ValidationError) as excinfo:
        RecordFactParams.model_validate(result.payload)
    message = describe_validation_failure(excinfo.value)
    assert "claim" in message
    assert "200" in message
    assert "condense" in message


# --- Opt-out, purity, logging ---


def test_strict_schema_opts_out_of_normalization() -> None:
    raw = {"artifact_id": "null"}
    result = normalize_payload(StrictParams, raw)
    assert result.payload["artifact_id"] == "null"
    assert result.normalized_fields == ()


def test_input_mapping_is_never_mutated() -> None:
    raw = {"query": "q", "artifact_id": "null"}
    snapshot = dict(raw)
    normalize_payload(SearchCodeParams, raw)
    assert raw == snapshot


def test_one_warning_per_coercion_names_target_and_fields(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="kernel.tools.parameter_normalization"):
        normalize_payload(
            SearchCodeParams,
            {"query": "q", "artifact_id": "null"},
            target_id="coding_search_code",
        )
    records = [r for r in caplog.records if r.name == "kernel.tools.parameter_normalization"]
    assert len(records) == 1
    assert "coding_search_code" in records[0].getMessage()
    assert "artifact_id" in records[0].getMessage()


# --- Registry wiring: tools ---


def test_tool_registry_normalizes_and_stamps_context(
    published: list[dict[str, Any]],
) -> None:
    definition = _tool_definition(SearchCodeParams, "coding_search_code")
    ctx = ToolExecutionContext(event_bus=object(), parameters={"query": "q", "artifact_id": "null"})
    validate_tool_parameters(definition, ctx)
    assert ctx.normalized_fields == ["artifact_id"]
    assert ctx.parameters["artifact_id"] is None
    _publish_tool_telemetry(ctx, "coding_search_code", "success", 5, None)
    assert published[-1]["normalized_fields"] == ["artifact_id"]


def test_tool_registry_rejection_names_cap_and_publishes_telemetry(
    published: list[dict[str, Any]],
) -> None:
    definition = _tool_definition(RecordFactParams, "coding_record_fact")
    ctx = ToolExecutionContext(event_bus=object(), parameters={"claim": "x" * 500})
    with pytest.raises(ToolExecutionError) as excinfo:
        validate_tool_parameters(definition, ctx)
    assert "200" in str(excinfo.value)
    assert "condense" in str(excinfo.value)
    assert len(published) == 1
    assert published[0]["status"] == "error"
    assert published[0]["normalized_fields"] is None


def test_rejection_telemetry_stamps_partial_normalization(
    published: list[dict[str, Any]],
) -> None:
    definition = _tool_definition(RecordFactParams, "coding_record_fact")
    ctx = ToolExecutionContext(event_bus=object(), parameters={"claim": "x" * 500, "revision": ""})
    with pytest.raises(ToolExecutionError):
        validate_tool_parameters(definition, ctx)
    assert published[0]["normalized_fields"] == ["revision"]


# --- Registry wiring: operations twin ---


def test_operation_registry_twin_normalizes_and_stamps(
    published: list[dict[str, Any]],
) -> None:
    definition = _operation_definition(SearchCodeParams, "memory_recall")
    ctx = OperationExecutionContext(
        event_bus=object(), params={"query": "q", "artifact_id": "null"}
    )
    validate_operation_parameters(definition, ctx)
    assert ctx.normalized_fields == ["artifact_id"]
    assert ctx.params["artifact_id"] is None


def test_operation_registry_twin_rejection_publishes_telemetry(
    published: list[dict[str, Any]],
) -> None:
    definition = _operation_definition(RecordFactParams, "record_fact")
    ctx = OperationExecutionContext(event_bus=object(), params={"claim": "x" * 500})
    with pytest.raises(OperationExecutionError) as excinfo:
        validate_operation_parameters(definition, ctx)
    assert "condense" in str(excinfo.value)
    assert len(published) == 1
    assert published[0]["status"] == "error"
