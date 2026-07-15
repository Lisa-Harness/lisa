"""Read a bounded line window from one local text source file."""

from __future__ import annotations

from pydantic import BaseModel, Field

from kernel.models import ToolExecutionContext, ToolResult
from sdk.packs.coding.config import get_coding_config
from sdk.packs.coding.local_files import CodingPathError, read_text_file, resolve_relative_path

TOOL_ID = "coding_read_file"


class Parameters(BaseModel):
    """Structured bounded line-range request."""

    path: str = Field(min_length=1)
    start_line: int = Field(default=1, ge=1)
    end_line: int = Field(default=200, ge=1)


async def execute(ctx: ToolExecutionContext) -> ToolResult:
    """Read a bounded text excerpt without materializing an unbounded source file."""

    params = Parameters.model_validate(ctx.parameters)
    try:
        config = get_coding_config(ctx)
        path = resolve_relative_path(config, params.path)
        if not path.is_file():
            return ToolResult(tool_id=TOOL_ID, error="Requested path is not a file.")
        text, byte_truncated = read_text_file(path, max_bytes=config.max_file_bytes)
    except (CodingPathError, OSError, ValueError) as exc:
        return ToolResult(tool_id=TOOL_ID, error=str(exc))

    start_line = params.start_line
    end_line = max(start_line, min(params.end_line, start_line + config.max_read_lines - 1))
    all_lines = text.splitlines()
    excerpt = [
        {"line": number, "text": all_lines[number - 1]}
        for number in range(start_line, min(end_line, len(all_lines)) + 1)
    ]
    return ToolResult(
        tool_id=TOOL_ID,
        output_text=f"Read {len(excerpt)} lines from {params.path}.",
        output_json={
            "path": params.path,
            "start_line": start_line,
            "end_line": excerpt[-1]["line"] if excerpt else start_line,
            "total_lines_observed": len(all_lines),
            "truncated_by_bytes": byte_truncated,
            "lines": excerpt,
        },
    )
