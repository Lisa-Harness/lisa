"""Conformance tests for the generic validation-recovery hook.

Pins the coding-edit-schema-recovery contract for the kernel half:

- a tool that declares neither catalog content nor classifier keeps its
  exact previous validation error;
- a declared tool appends the rendered authored section (classified key,
  missing-key-safe substitutions, original summary preserved);
- a rendering failure (missing content, unknown key, no classifier,
  unset registry) is a pack installation error: it is logged and the
  original message still rejects — the hook never masks a rejection;
- the recovery text is bounded and carries no raw arguments.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from kernel.context.content_registry import set_content_registry
from kernel.models import ToolDefinition, ToolExecutionContext, ValidationRecoveryDeclaration
from kernel.tools.registry import (
    ToolExecutionError,
)
from kernel.tools.registry import (
    _validate_parameters as validate_tool_parameters,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError

_RECOVERY_MD = """# recovery content

## digest_format

Re-read the target with coding_read_file and copy its digest at {field_path}.
Validation detail: {validation_detail}

## edits_not_array

Send one JSON array under edits. Detail: {validation_detail}
"""


class _Record(BaseModel):
    """One edit record requiring the digest field the classifier keys on."""

    kind: str
    path: str
    expected_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EditParams(BaseModel):
    """The minimal shape the classifier discriminates over."""

    model_config = ConfigDict(strict=False)

    artifact_id: str = Field(min_length=1)
    edits: list[_Record] = Field(min_length=1)


def _module() -> Any:
    """A fake implementation module owning the pure classifier."""

    def validation_recovery_key(exc: BaseException) -> str:
        if not isinstance(exc, ValidationError):
            return "unknown_or_other"
        for error in exc.errors():
            if "valid list" in str(error.get("msg", "")):
                return "edits_not_array"
        return "digest_format"

    return type("Module", (), {"validation_recovery_key": staticmethod(validation_recovery_key)})


def _definition(*, declare: bool, with_classifier: bool = True) -> ToolDefinition:
    return ToolDefinition.model_construct(
        tool_id="coding_edit_files",
        parameters_schema=EditParams,
        module_path="lisa_packs.coding.tools.edit_files.edit_files",
        implementation_module=_module() if with_classifier else None,
        validation_recovery=(
            ValidationRecoveryDeclaration(content_path="tools/edit_files/recovery.md")
            if declare
            else None
        ),
    )


class _Registry:
    """Minimal content-registry double."""

    def __init__(self, content: str) -> None:
        self._content = content

    def read_authored_text(self, relative_path: str) -> str:
        return self._content if relative_path == "tools/edit_files/recovery.md" else ""


@pytest.fixture()
def published(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def _capture(event_bus: Any, topic: str, event: Any, *, source: str = "") -> None:
        rows.append(dict(event))

    monkeypatch.setattr("kernel.tools.registry.schedule_telemetry_publish", _capture)
    return rows


@pytest.fixture(autouse=True)
def registry() -> Any:
    """Install the authored content for the duration of each test."""

    set_content_registry(_Registry(_RECOVERY_MD))
    yield
    set_content_registry(None)  # type: ignore[arg-type]


def test_undeclared_tool_keeps_its_exact_error(published: list[dict[str, Any]]) -> None:
    definition = _definition(declare=False)
    ctx = ToolExecutionContext(event_bus=object(), parameters={"edits": "not json"})
    with pytest.raises(ToolExecutionError) as excinfo:
        validate_tool_parameters(definition, ctx)
    message = str(excinfo.value)
    assert "valid list" in message
    assert "Re-read the target" not in message
    assert len(published) == 1


def test_declared_tool_appends_classified_recovery(published: list[dict[str, Any]]) -> None:
    definition = _definition(declare=True)
    ctx = ToolExecutionContext(event_bus=object(), parameters={"artifact_id": "wt", "edits": 5})
    with pytest.raises(ToolExecutionError) as excinfo:
        validate_tool_parameters(definition, ctx)
    message = str(excinfo.value)
    # Original bounded summary is preserved...
    assert "valid list" in message
    # ...and the classified section renders with its substitutions filled.
    assert "Send one JSON array under edits." in message
    assert "edits" in message
    assert len(published) == 1
    assert published[0]["status"] == "error"


def test_recovery_substitutes_field_path_and_detail() -> None:
    definition = _definition(declare=True)
    ctx = ToolExecutionContext(
        event_bus=object(),
        parameters={"artifact_id": "wt", "edits": [{"kind": "update", "path": "x"}]},
    )
    with pytest.raises(ToolExecutionError) as excinfo:
        validate_tool_parameters(definition, ctx)
    message = str(excinfo.value)
    # The digest-format section renders with the first failing field path.
    assert "copy its digest at edits.0.expected_sha256." in message
    # Substitutions are all filled — no raw braces survive in the section.
    assert "Validation detail:" in message
    assert "{" not in message.split("Re-read the target", 1)[1]


def test_missing_classifier_logs_and_keeps_message(
    published: list[dict[str, Any]], caplog: pytest.LogCaptureFixture
) -> None:
    definition = _definition(declare=True, with_classifier=False)
    ctx = ToolExecutionContext(event_bus=object(), parameters={"edits": "not json"})
    with (
        caplog.at_level(logging.WARNING, logger="kernel.tools.registry"),
        pytest.raises(ToolExecutionError) as excinfo,
    ):
        validate_tool_parameters(definition, ctx)
    assert "valid list" in str(excinfo.value)
    assert "validation recovery rendering failed" in caplog.text
    assert len(published) == 1


def test_missing_content_logs_and_keeps_message(
    published: list[dict[str, Any]], caplog: pytest.LogCaptureFixture
) -> None:
    set_content_registry(_Registry(""))
    definition = _definition(declare=True)
    ctx = ToolExecutionContext(event_bus=object(), parameters={"edits": "not json"})
    with (
        caplog.at_level(logging.WARNING, logger="kernel.tools.registry"),
        pytest.raises(ToolExecutionError) as excinfo,
    ):
        validate_tool_parameters(definition, ctx)
    assert "valid list" in str(excinfo.value)
    assert "validation recovery rendering failed" in caplog.text


def test_unknown_recovery_key_logs_and_keeps_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    set_content_registry(_Registry("## only_known_key\n\nunrelated content\n"))
    definition = _definition(declare=True)
    ctx = ToolExecutionContext(event_bus=object(), parameters={"edits": 5})
    with (
        caplog.at_level(logging.WARNING, logger="kernel.tools.registry"),
        pytest.raises(ToolExecutionError) as excinfo,
    ):
        validate_tool_parameters(definition, ctx)
    assert "valid list" in str(excinfo.value)
    assert "no section for recovery key" in caplog.text


def test_unset_registry_logs_and_keeps_message(caplog: pytest.LogCaptureFixture) -> None:
    set_content_registry(None)  # type: ignore[arg-type]
    definition = _definition(declare=True)
    ctx = ToolExecutionContext(event_bus=object(), parameters={"edits": 5})
    with (
        caplog.at_level(logging.WARNING, logger="kernel.tools.registry"),
        pytest.raises(ToolExecutionError) as excinfo,
    ):
        validate_tool_parameters(definition, ctx)
    assert "valid list" in str(excinfo.value)
    assert "content registry is not set" in caplog.text


def test_recovery_text_never_carries_raw_arguments() -> None:
    definition = _definition(declare=True)
    secret = "x" * 200
    ctx = ToolExecutionContext(
        event_bus=object(),
        parameters={"artifact_id": "wt", "edits": [{"kind": "update", "path": secret}]},
    )
    with pytest.raises(ToolExecutionError) as excinfo:
        validate_tool_parameters(definition, ctx)
    assert secret not in str(excinfo.value)
