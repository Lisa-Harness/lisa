"""Tests for the ``stream_output`` tool declaration flag.

Covers the SDK catalog loader (``.tool.yml`` front matter →
``ToolDefinition.stream_output``) and the kernel propagation of the flag
onto ``ToolExecutionContext`` via ``kernel.tools.registry``.
"""

from __future__ import annotations

import json
from pathlib import Path

from sdk.content import (
    build_tool_catalog_json,
    invalidate_catalog_cache,
    resolve_tool_roots,
)


def _write_tool(
    root: Path,
    *,
    tool_id: str,
    provider_id: str = "test-provider",
    stream_output: bool | None = None,
) -> None:
    """Write one minimal tool (``.py`` + ``.tool.yml``) under ``root``."""
    tools_dir = root / "tools" / tool_id
    tools_dir.mkdir(parents=True, exist_ok=True)
    (tools_dir / f"{tool_id}.py").write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "from kernel.models import ToolExecutionContext, ToolResult",
                "",
                "",
                "async def execute(ctx: ToolExecutionContext) -> ToolResult:",
                f"    return ToolResult(tool_id=ctx.tool_id or '{tool_id}', output_text='ok')",
                "",
            ]
        ),
        encoding="utf-8",
    )
    lines = [
        f"tool_id: {tool_id}",
        f"tool_provider_id: {provider_id}",
        f'description: "{tool_id} tool"',
        "mode_allowlist: null",
        "timeout_seconds: 30",
    ]
    if stream_output is not None:
        lines.append(f"stream_output: {str(stream_output).lower()}")
    lines.append("")
    (tools_dir / f"{tool_id}.tool.yml").write_text("\n".join(lines), encoding="utf-8")


def _tool_catalog(*roots: Path) -> dict:
    """Build the tool catalog JSON and return it as a dict."""
    invalidate_catalog_cache()
    tool_roots = resolve_tool_roots(list(roots))
    return json.loads(build_tool_catalog_json(tool_roots))


class TestStreamOutputCatalog:
    """``stream_output`` flows from ``.tool.yml`` into the tool catalog JSON."""

    def test_declared_true_round_trips(self, tmp_path: Path) -> None:
        _write_tool(tmp_path, tool_id="streamer", stream_output=True)
        catalog = _tool_catalog(tmp_path)
        assert catalog["streamer"]["stream_output"] is True

    def test_declared_false_round_trips(self, tmp_path: Path) -> None:
        _write_tool(tmp_path, tool_id="silent", stream_output=False)
        catalog = _tool_catalog(tmp_path)
        assert catalog["silent"]["stream_output"] is False

    def test_omitted_defaults_false(self, tmp_path: Path) -> None:
        _write_tool(tmp_path, tool_id="legacy")  # no stream_output key at all
        catalog = _tool_catalog(tmp_path)
        assert catalog["legacy"]["stream_output"] is False