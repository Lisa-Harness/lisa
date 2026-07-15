"""List bounded workspace directory entries."""

from __future__ import annotations

from pydantic import BaseModel, Field

from kernel.models import ToolExecutionContext, ToolResult
from sdk.packs.coding.config import get_coding_config
from sdk.packs.coding.local_files import (
    CodingPathError,
    excluded_relative_path,
    relative_path,
    resolve_relative_path,
)

TOOL_ID = "coding_list_directory"


class Parameters(BaseModel):
    """Structured directory listing request."""

    path: str = Field(default="", description="Relative workspace directory, or empty for root.")
    max_entries: int = Field(default=100, ge=1, le=1000)


async def execute(ctx: ToolExecutionContext) -> ToolResult:
    """Return stable, bounded metadata for visible workspace entries."""

    params = Parameters.model_validate(ctx.parameters)
    try:
        config = get_coding_config(ctx)
        root = config.resolve_workspace_root()
        directory = resolve_relative_path(config, params.path, allow_root=True)
        if not directory.is_dir():
            return ToolResult(tool_id=TOOL_ID, error="Requested path is not a directory.")
        limit = min(params.max_entries, config.max_directory_entries)
        entries = []
        for item in sorted(
            directory.iterdir(), key=lambda value: (not value.is_dir(), value.name.lower())
        ):
            try:
                resolved = item.resolve()
                relative = resolved.relative_to(root)
            except ValueError:
                continue
            if excluded_relative_path(relative):
                continue
            entries.append(
                {
                    "path": relative_path(root, resolved),
                    "name": item.name,
                    "kind": "directory" if resolved.is_dir() else "file",
                    "size_bytes": resolved.stat().st_size if resolved.is_file() else None,
                }
            )
            if len(entries) >= limit:
                break
    except (CodingPathError, OSError, ValueError) as exc:
        return ToolResult(tool_id=TOOL_ID, error=str(exc))

    return ToolResult(
        tool_id=TOOL_ID,
        output_text=f"Listed {len(entries)} entries.",
        output_json={
            "path": params.path or ".",
            "entries": entries,
            "truncated": len(entries) >= limit,
        },
    )
