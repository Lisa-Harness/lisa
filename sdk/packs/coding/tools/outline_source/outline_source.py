"""Extract a bounded structural outline from local source files."""

from __future__ import annotations

import ast
import re

from pydantic import BaseModel, Field

from kernel.models import ToolExecutionContext, ToolResult
from sdk.packs.coding.config import get_coding_config
from sdk.packs.coding.local_files import CodingPathError, read_text_file, resolve_relative_path

TOOL_ID = "coding_outline_source"
_TS_DECLARATION = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?"
    r"(?:class|function|interface|type)\s+([A-Za-z_$][\w$]*)"
)
_TS_ARROW = re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=")


class Parameters(BaseModel):
    """Structured source outline request."""

    path: str = Field(min_length=1)


def _python_outline(source: str) -> list[dict[str, object]]:
    """Return top-level Python classes and functions through the AST."""

    tree = ast.parse(source)
    items: list[dict[str, object]] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            items.append({"kind": "class", "name": node.name, "line": node.lineno})
            for member in node.body:
                if isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef):
                    items.append(
                        {
                            "kind": "method",
                            "name": member.name,
                            "line": member.lineno,
                            "async": isinstance(member, ast.AsyncFunctionDef),
                        }
                    )
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            items.append(
                {
                    "kind": "function",
                    "name": node.name,
                    "line": node.lineno,
                    "async": isinstance(node, ast.AsyncFunctionDef),
                }
            )
    return items


def _typescript_outline(source: str) -> list[dict[str, object]]:
    """Use an explicitly labeled bounded regex fallback for TS and JS files."""

    items: list[dict[str, object]] = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        declaration = _TS_DECLARATION.match(line)
        arrow = _TS_ARROW.match(line)
        name = declaration.group(1) if declaration else arrow.group(1) if arrow else ""
        if name:
            items.append({"kind": "declaration", "name": name, "line": line_number})
    return items


async def execute(ctx: ToolExecutionContext) -> ToolResult:
    """Return an AST or explicitly labeled fallback outline for one source file."""

    params = Parameters.model_validate(ctx.parameters)
    try:
        config = get_coding_config(ctx)
        path = resolve_relative_path(config, params.path)
        if not path.is_file():
            return ToolResult(tool_id=TOOL_ID, error="Requested path is not a file.")
        source, byte_truncated = read_text_file(path, max_bytes=config.max_file_bytes)
        suffix = path.suffix.lower()
        if suffix == ".py":
            output_type = "python_ast"
            items = _python_outline(source)
        elif suffix in {".ts", ".tsx", ".js", ".jsx"}:
            output_type = "typescript_javascript_regex"
            items = _typescript_outline(source)
        else:
            output_type = "text_lines"
            items = [
                {"kind": "line", "name": line[:240], "line": index}
                for index, line in enumerate(source.splitlines()[:20], start=1)
            ]
    except SyntaxError as exc:
        return ToolResult(tool_id=TOOL_ID, error=f"Python parse failed: {exc}")
    except (CodingPathError, OSError, ValueError) as exc:
        return ToolResult(tool_id=TOOL_ID, error=str(exc))

    return ToolResult(
        tool_id=TOOL_ID,
        output_text=f"Extracted {len(items)} outline items from {params.path}.",
        output_json={
            "path": params.path,
            "output_type": output_type,
            "truncated_by_bytes": byte_truncated,
            "items": items,
        },
    )
