"""Search local source using a fixed ripgrep JSON invocation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from pydantic import BaseModel, Field

from kernel.models import ToolExecutionContext, ToolResult
from sdk.packs.coding.config import get_coding_config
from sdk.packs.coding.local_files import (
    CodingPathError,
    excluded_relative_path,
    resolve_relative_path,
)

TOOL_ID = "coding_search_code"


class Parameters(BaseModel):
    """Structured code search request."""

    query: str = Field(min_length=1, max_length=500)
    path: str = Field(default="", description="Optional relative file or directory scope.")
    max_results: int = Field(default=25, ge=1, le=200)


async def _read_limited(stream: asyncio.StreamReader, limit: int) -> tuple[bytes, bool]:
    """Drain a subprocess stream while retaining at most ``limit`` bytes."""

    retained = bytearray()
    overflowed = False
    while chunk := await stream.read(8192):
        remaining = limit - len(retained)
        if remaining > 0:
            retained.extend(chunk[:remaining])
        if len(chunk) > remaining:
            overflowed = True
    return bytes(retained), overflowed


async def _run_ripgrep(
    *,
    root: str,
    path: str,
    query: str,
    max_results: int,
    max_output_bytes: int,
    timeout_seconds: int,
) -> tuple[int, bytes, bytes, bool]:
    """Run a bounded, no-shell ripgrep search and return captured output."""

    command = [
        "rg",
        "--json",
        "--line-number",
        "--color=never",
        "--max-count",
        str(max_results),
        "--glob",
        "!.git/**",
        "--glob",
        "!node_modules/**",
        "--glob",
        "!.venv/**",
        "--regexp",
        query,
    ]
    if path:
        command.extend(("--", path))
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=root,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    stdout_task = asyncio.create_task(_read_limited(process.stdout, max_output_bytes))
    stderr_task = asyncio.create_task(_read_limited(process.stderr, max_output_bytes))
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
    except TimeoutError:
        process.kill()
        await process.wait()
        await asyncio.gather(stdout_task, stderr_task)
        raise
    stdout, stdout_overflowed = await stdout_task
    stderr, stderr_overflowed = await stderr_task
    return process.returncode or 0, stdout, stderr, stdout_overflowed or stderr_overflowed


def _parse_matches(
    stdout: bytes,
    *,
    max_results: int,
    excluded_parts: frozenset[str],
) -> list[dict[str, object]]:
    """Parse bounded ripgrep JSON output into safe local match summaries."""

    matches: list[dict[str, object]] = []
    for raw_line in stdout.decode("utf-8", errors="replace").splitlines():
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "match":
            continue
        data = event.get("data") or {}
        raw_path = ((data.get("path") or {}).get("text") or "").replace("\\", "/")
        if not raw_path:
            continue
        if any(part in excluded_parts for part in Path(raw_path).parts):
            continue
        line_text = str((data.get("lines") or {}).get("text") or "").rstrip("\r\n")
        matches.append(
            {
                "path": raw_path,
                "line": int(data.get("line_number") or 0),
                "text": line_text[:500],
            }
        )
        if len(matches) >= max_results:
            break
    return matches


async def execute(ctx: ToolExecutionContext) -> ToolResult:
    """Run a fixed-argument ripgrep search under the configured workspace root."""

    params = Parameters.model_validate(ctx.parameters)
    try:
        config = get_coding_config(ctx)
        root = config.resolve_workspace_root()
        scope = resolve_relative_path(config, params.path, allow_root=True)
        relative_scope = "" if scope == root else scope.relative_to(root).as_posix()
        max_results = min(params.max_results, config.max_search_results)
        code, stdout, stderr, overflowed = await _run_ripgrep(
            root=str(root),
            path=relative_scope,
            query=params.query,
            max_results=max_results,
            max_output_bytes=config.max_search_output_bytes,
            timeout_seconds=config.search_timeout_seconds,
        )
    except TimeoutError:
        return ToolResult(tool_id=TOOL_ID, error="Code search timed out.")
    except FileNotFoundError:
        return ToolResult(tool_id=TOOL_ID, error="ripgrep is not available on this host.")
    except (CodingPathError, OSError, ValueError) as exc:
        return ToolResult(tool_id=TOOL_ID, error=str(exc))

    if overflowed:
        return ToolResult(
            tool_id=TOOL_ID, error="Code search output exceeded the configured limit."
        )
    if code not in (0, 1):
        return ToolResult(
            tool_id=TOOL_ID,
            error="ripgrep failed: " + stderr.decode("utf-8", errors="replace")[:500],
        )
    matches = _parse_matches(
        stdout,
        max_results=max_results,
        excluded_parts=frozenset(config.excluded_parts),
    )
    return ToolResult(
        tool_id=TOOL_ID,
        output_text=f"Found {len(matches)} code matches.",
        output_json={"query": params.query, "path": params.path or ".", "matches": matches},
    )
