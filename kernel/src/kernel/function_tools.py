"""Function-tool schema builder with OpenAI strict-mode support.

Produces OpenAI-compatible function-tool dicts from Pydantic models. The
strict-mode schema transformation (inlining ``$ref``, setting
``additionalProperties: False``, marking all properties ``required``) is
implemented locally — this module has no ``openai`` SDK dependency.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class _EmptyToolParameters(BaseModel):
    """Placeholder schema for function tools that take no arguments."""


def _has_additional_properties_true(schema: Any) -> bool:
    if isinstance(schema, dict):
        if schema.get("additionalProperties") is True:
            return True
        return any(_has_additional_properties_true(v) for v in schema.values())
    if isinstance(schema, list):
        return any(_has_additional_properties_true(item) for item in schema)
    return False


def _has_union_items(schema: Any) -> bool:
    """Detect ``anyOf``/``oneOf`` inside an ``items`` schema.

    OpenAI strict mode requires every ``items`` node to declare an explicit
    ``type``. pydantic emits ``anyOf``/``oneOf`` in ``items`` for
    ``list[Union[...]]`` fields (e.g. ``list[str | dict[str, str]]``), which
    strict mode rejects with ``schema must have a 'type' key``. Top-level
    ``anyOf`` on a property (e.g. ``Optional[str]``) is permitted by strict
    mode and is NOT flagged here, so the common optional-field case keeps
    strict mode.
    """
    if isinstance(schema, dict):
        items = schema.get("items")
        if isinstance(items, dict) and ("anyOf" in items or "oneOf" in items):
            return True
        return any(_has_union_items(v) for v in schema.values())
    if isinstance(schema, list):
        return any(_has_union_items(item) for item in schema)
    return False


def _strict_incompatible(schema: Any) -> bool:
    """True when the schema is incompatible with OpenAI strict mode."""
    return _has_additional_properties_true(schema) or _has_union_items(schema)


# ── Strict JSON schema transformation ────────────────────────────────────
#
# Faithful reimplementation of openai.lib._pydantic.to_strict_json_schema.
# The OpenAI strict mode requires:
#   - every ``type: object`` node has ``additionalProperties: false``
#   - every property is listed in ``required``
#   - ``$ref`` nodes are inlined (no $ref + other keys)
#   - ``$defs`` / ``definitions`` are recursively processed
#   - ``None`` defaults are stripped
# See: https://platform.openai.com/docs/guides/structured-outputs


def _is_dict(obj: object) -> bool:
    return isinstance(obj, dict)


def _is_list(obj: object) -> bool:
    return isinstance(obj, list)


def _resolve_ref(*, root: dict[str, object], ref: str) -> dict[str, Any]:
    if not ref.startswith("#/"):
        raise ValueError(f"Unexpected $ref format {ref!r}; Does not start with #/")

    path = ref[2:].split("/")
    resolved: Any = root
    for key in path:
        resolved = resolved[key]  # type: ignore[index]
        if not _is_dict(resolved):
            raise TypeError(f"Encountered non-dictionary entry while resolving {ref}")
    return resolved  # type: ignore[return-value]


def _has_more_than_n_keys(obj: dict[str, object], n: int) -> bool:
    return sum(1 for _ in obj) > n


def _inline_nested_defs(
    schema: dict[str, Any],
    *,
    path: tuple[str, ...],
    root: dict[str, object],
) -> None:
    """Recursively strictify ``$defs`` / ``definitions`` subschemas (in place)."""
    defs = schema.get("$defs")
    if _is_dict(defs):
        for def_name, def_schema in defs.items():
            _ensure_strict_json_schema(def_schema, path=(*path, "$defs", def_name), root=root)

    definitions = schema.get("definitions")
    if _is_dict(definitions):
        for definition_name, definition_schema in definitions.items():
            _ensure_strict_json_schema(
                definition_schema,
                path=(*path, "definitions", definition_name),
                root=root,
            )


def _inline_all_of(
    schema: dict[str, Any],
    *,
    path: tuple[str, ...],
    root: dict[str, object],
) -> None:
    """Strictify ``allOf`` entries, inlining a single-entry intersection (in place)."""
    all_of = schema.get("allOf")
    if not _is_list(all_of):
        return
    if len(all_of) == 1:
        schema.update(_ensure_strict_json_schema(all_of[0], path=(*path, "allOf", "0"), root=root))
        schema.pop("allOf")
    else:
        schema["allOf"] = [
            _ensure_strict_json_schema(entry, path=(*path, "allOf", str(i)), root=root)
            for i, entry in enumerate(all_of)
        ]


def _ensure_strict_json_schema(
    json_schema: object,
    *,
    path: tuple[str, ...],
    root: dict[str, object],
) -> dict[str, Any]:
    """Mutate ``json_schema`` in place to conform to OpenAI strict mode."""

    if not _is_dict(json_schema):
        raise TypeError(f"Expected {json_schema} to be a dictionary; path={path}")

    schema: dict[str, Any] = json_schema  # type: ignore[assignment]

    _inline_nested_defs(schema, path=path, root=root)

    typ = schema.get("type")
    if typ == "object" and "additionalProperties" not in schema:
        schema["additionalProperties"] = False

    # object types
    properties = schema.get("properties")
    if _is_dict(properties):
        schema["required"] = list(properties.keys())
        schema["properties"] = {
            key: _ensure_strict_json_schema(prop_schema, path=(*path, "properties", key), root=root)
            for key, prop_schema in properties.items()
        }

    # arrays
    items = schema.get("items")
    if _is_dict(items):
        schema["items"] = _ensure_strict_json_schema(items, path=(*path, "items"), root=root)

    # unions
    any_of = schema.get("anyOf")
    if _is_list(any_of):
        schema["anyOf"] = [
            _ensure_strict_json_schema(variant, path=(*path, "anyOf", str(i)), root=root)
            for i, variant in enumerate(any_of)
        ]

    # intersections
    _inline_all_of(schema, path=path, root=root)

    # strip None defaults
    if schema.get("default") is None:
        schema.pop("default", None)

    # inline $ref when other keys are present
    ref = schema.get("$ref")
    if ref and _has_more_than_n_keys(schema, 1):
        resolved = _resolve_ref(root=root, ref=ref)
        # properties from the json schema take priority over the $ref
        schema.update({**resolved, **schema})
        schema.pop("$ref")
        return _ensure_strict_json_schema(schema, path=path, root=root)

    return schema


def _to_strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return a strict-mode JSON schema for ``model``."""

    schema = model.model_json_schema()
    return _ensure_strict_json_schema(schema, path=(), root=schema)


def build_strict_openai_function_tool(
    *,
    name: str,
    description: str,
    parameters_schema: type[BaseModel] | None = None,
) -> dict[str, Any]:
    """Return one function-tool schema compatible with SDK auto-parsing.

    Uses structured outputs (`strict=True`) unless the schema is
    incompatible with OpenAI strict mode — arbitrary object keys
    (`additionalProperties: True`) or union shapes (`anyOf`/`oneOf`),
    both of which strict mode rejects. Incompatible schemas fall back to
    JSON mode (`strict=False`) so the model can still call the tool.

    The all-properties-``required`` rewrite is a strict-mode artifact: it
    tells the model every field (including optional ones) must be present.
    That is correct under ``strict=True`` (the API then ignores unset
    optional fields), but under the ``strict=False`` fallback it actively
    misleads weaker models into *fabricating* values for optional fields —
    observed in production as ``overrides: ""`` and both ``command`` AND
    ``commands`` supplied to ``bash_run``. So the strict rewrite only
    ships when the tool actually runs in strict mode; JSON-mode tools keep
    pydantic's honest ``required`` (only genuinely-required fields) and
    their ``default`` annotations.
    """

    model = parameters_schema or _EmptyToolParameters
    parameters = _to_strict_json_schema(model)
    tool: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": name,
            "strict": True,
            "parameters": parameters,
        },
    }
    if description is not None:
        tool["function"]["description"] = description

    if _strict_incompatible(tool["function"].get("parameters", {})):
        tool["function"]["strict"] = False
        # Re-emit the schema from the raw pydantic model so optional fields
        # are not marked required and None defaults are preserved (see
        # docstring: the all-required rewrite misleads JSON-mode callers).
        tool["function"]["parameters"] = model.model_json_schema()

    return tool
