"""Focused failure handling tests for MCP-backed catalog adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from kernel.context.mcp_catalog import (
    CatalogResponseError,
    MCPModeCatalog,
    MCPOperationCatalog,
    MCPToolCatalog,
)
from kernel.tools.mcp_client import MCPClientConfig

CatalogFactory = Callable[[MCPClientConfig], Any]

_CATALOG_CONFIG = MCPClientConfig(
    server_id="catalog-error-test",
    command="unused",
    args=[],
)
_CATALOG_ERROR = "Error executing tool: duplicate default mode"


@pytest.mark.parametrize(
    ("catalog_factory", "getter_name", "tool_name"),
    (
        (MCPModeCatalog, "get_workflow_catalog", "list_modes"),
        (MCPOperationCatalog, "get_operation_catalog", "list_operations"),
        (MCPToolCatalog, "get_tool_catalog", "list_tools"),
    ),
)
def test_catalog_adapters_expose_textual_mcp_failures(
    monkeypatch: pytest.MonkeyPatch,
    catalog_factory: CatalogFactory,
    getter_name: str,
    tool_name: str,
) -> None:
    catalog = catalog_factory(_CATALOG_CONFIG)
    monkeypatch.setattr(catalog, "_call", lambda _tool, _arguments: _CATALOG_ERROR)

    with pytest.raises(
        CatalogResponseError,
        match=rf"lisa-catalog tool '{tool_name}' returned invalid JSON",
    ) as error:
        getattr(catalog, getter_name)()

    assert _CATALOG_ERROR in str(error.value)


def test_catalog_adapter_rejects_non_object_json(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = MCPModeCatalog(_CATALOG_CONFIG)
    monkeypatch.setattr(catalog, "_call", lambda _tool, _arguments: "[]")

    with pytest.raises(CatalogResponseError, match="returned a JSON list"):
        catalog.get_workflow_catalog()
