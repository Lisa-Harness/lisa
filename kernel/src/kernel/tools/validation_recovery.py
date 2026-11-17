"""Generic validation-recovery hook shared by both dispatch registries.

When pre-execution parameter validation rejects a model-written payload, the
raw pydantic summary names the failing field but not the shape to send next
(the coding-edit-schema-recovery design: under long-context pressure the model
repeats the same rejected batch and burns its call allowance). A tool may
declare authored recovery content:

- the ``.tool.yml`` ``validation_recovery.content_path`` declaration names a
  packaged Markdown resource whose ``## <key>`` sections hold the recovery
  text (catalog translation; the SDK validates the resource exists);
- the implementation module defines a pure classifier
  ``validation_recovery_key(exc) -> str`` with no execution side effects.

On a ``ValidationError`` the registry classifies the error, loads the
declared section through the kernel's content-registry seam, formats the
named substitutions with missing-key-safe ``str.format_map``, and appends
the rendered text to the original bounded validation summary. The error
still publishes rejection telemetry and still raises through the existing
``ToolExecutionError`` / ``OperationExecutionError`` path: no write effect
is emitted and the failure never becomes a successful result.

Failure modes of the hook itself are pack installation errors and fail
loudly — a missing/unreadable/empty resource or an unknown recovery key
raises with the content path and key rather than substituting Python
fallback prose (the pack-content rule: no silent fallbacks).
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from kernel.models import ValidationRecoveryDeclaration

__all__ = ["ValidationRecoveryError", "render_validation_recovery"]


class ValidationRecoveryError(RuntimeError):
    """Raised when a declared recovery resource is missing or malformed."""

    def __init__(self, message: str, *, content_path: str = "", key: str = "") -> None:
        super().__init__(message)
        self.content_path = content_path
        self.key = key


def render_validation_recovery(
    definition: Any,
    exc: ValidationError,
    *,
    field_paths: list[str],
    validation_detail: str,
) -> str:
    """Return the rendered recovery text for one rejected dispatch.

    ``definition`` is the tool or operation definition carrying a
    ``validation_recovery`` declaration, an ``implementation_module``, a
    ``module_path``, and an id field (``tool_id`` or ``operation_id``).
    Returns ``""`` when the definition declares neither the catalog content
    path nor the module classifier — its error stays exactly as before. Any
    declared half that is missing is a pack installation error: an empty
    content path raises with the definition id, an absent classifier raises
    with the module name, and missing/empty/unknown-key content raises with
    the content path and key.

    ``field_paths`` and ``validation_detail`` are the only substitutions;
    the rendered text must never embed raw arguments, file contents, or
    digests.
    """

    declaration = _declaration(definition)
    if declaration is None:
        return ""
    definition_id = _definition_id(definition)
    module = getattr(definition, "implementation_module", None)
    classifier = getattr(module, "validation_recovery_key", None)
    if not callable(classifier):
        raise ValidationRecoveryError(
            f"'{definition_id}' declares validation_recovery content "
            f"but its module '{definition.module_path}' defines no "
            "validation_recovery_key(exc) classifier.",
            content_path=declaration.content_path,
        )
    key = str(classifier(exc))
    body = _section(definition_id, declaration, key)
    return _format(body, field_paths=field_paths, validation_detail=validation_detail)


def _definition_id(definition: Any) -> str:
    """Return the definition's id for error messages (tool or operation)."""

    return str(getattr(definition, "tool_id", None) or getattr(definition, "operation_id", ""))


def _declaration(definition: Any) -> ValidationRecoveryDeclaration | None:
    """Return the definition's recovery declaration, validating its content path."""

    declaration = getattr(definition, "validation_recovery", None)
    if declaration is None:
        return None
    if not isinstance(declaration, ValidationRecoveryDeclaration):
        raise ValidationRecoveryError(
            f"'{_definition_id(definition)}' declares validation_recovery "
            "with a malformed declaration.",
        )
    path = declaration.content_path
    if not path or not isinstance(path, str):
        raise ValidationRecoveryError(
            f"'{_definition_id(definition)}' declares validation_recovery with an "
            "empty content_path.",
        )
    return declaration


def _section(
    definition_id: str,
    declaration: ValidationRecoveryDeclaration,
    key: str,
) -> str:
    """Load the authored resource and extract one keyed section.

    A missing/unreadable resource, an empty resource, or a key without a
    section are all pack installation errors: fail loudly with the resource
    path and key, never substitute fallback prose.
    """

    registry = _content_registry()
    if registry is None:
        raise ValidationRecoveryError(
            f"'{definition_id}' declares validation recovery content at "
            f"'{declaration.content_path}' but the content registry is not set.",
            content_path=declaration.content_path,
            key=key,
        )
    raw = registry.read_authored_text(declaration.content_path)
    if not str(raw or "").strip():
        raise ValidationRecoveryError(
            f"'{definition_id}' validation recovery content is missing "
            f"or empty: {declaration.content_path}",
            content_path=declaration.content_path,
            key=key,
        )
    section = _extract_section(raw, key)
    if not section.strip():
        raise ValidationRecoveryError(
            f"'{definition_id}' validation recovery content "
            f"'{declaration.content_path}' has no section for recovery key "
            f"'{key}'.",
            content_path=declaration.content_path,
            key=key,
        )
    return section


def _content_registry() -> Any:
    """Return the module-level content registry holder (``None`` when unset).

    Late import: the holder module is import-safe, but reaching it through
    ``kernel.context`` executes the package ``__init__`` whose catalog
    adapters circle back into this registry module.
    """

    # lazy: breaks the cycle with the kernel.context package init
    from kernel.context.content_registry import get_content_registry

    return get_content_registry()


def _extract_section(raw: str, key: str) -> str:
    """Return the body of the ``## <key>`` heading up to the next heading."""

    lines = raw.split("\n")
    start = -1
    for index, line in enumerate(lines):
        if line.strip() == f"## {key}":
            start = index + 1
            break
    if start < 0:
        return ""
    end = len(lines)
    for index in range(start, len(lines)):
        stripped = lines[index].strip()
        if stripped.startswith("## "):
            end = index
            break
    return "\n".join(lines[start:end]).strip()


def _format(body: str, *, field_paths: list[str], validation_detail: str) -> str:
    """Missing-key-safe ``str.format_map`` over the two named substitutions."""

    fields: dict[str, Any] = {
        "field_path": field_paths[0] if field_paths else "",
        "validation_detail": validation_detail,
    }

    class _Missing(dict[str, str]):  # pylint: disable=missing-class-docstring
        def __missing__(self, name: str) -> str:
            return f"{{{name}}}"

    return body.format_map(_Missing(fields))
