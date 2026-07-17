"""Typed, session-owned policy for local coding tools."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kernel.models import ToolExecutionContext

__all__ = ["CodingCheck", "CodingPackConfig", "get_coding_config"]


class CodingCheck(BaseModel):
    """One finite command vector allowed to run in an owned worktree."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, max_length=80)
    argv: tuple[str, ...] = Field(min_length=1)
    timeout_seconds: int = Field(default=300, ge=1, le=1800)


class CodingPackConfig(BaseModel):
    """Portable policy supplied only through session pack configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    workspace_root: str = Field(min_length=1)
    excluded_parts: tuple[str, ...] = Field(
        default=(
            ".git",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            ".venv",
            "__pycache__",
            "build",
            "dist",
            "node_modules",
        )
    )
    checks: tuple[CodingCheck, ...] = ()
    required_check_ids: tuple[str, ...] = ()
    branch_prefix: str = Field(default="lisa/", min_length=1, max_length=80)
    protected_branches: tuple[str, ...] = ("main", "master")
    max_directory_entries: int = Field(default=200, ge=1, le=1000)
    max_file_bytes: int = Field(default=512_000, ge=1024, le=5_000_000)
    max_read_lines: int = Field(default=400, ge=1, le=2000)
    max_search_results: int = Field(default=50, ge=1, le=200)
    max_search_output_bytes: int = Field(default=200_000, ge=4096, le=2_000_000)
    search_timeout_seconds: int = Field(default=20, ge=1, le=120)
    max_patch_bytes: int = Field(default=200_000, ge=1024, le=2_000_000)
    max_command_output_bytes: int = Field(default=100_000, ge=4096, le=2_000_000)
    git_timeout_seconds: int = Field(default=60, ge=1, le=300)
    push_remote: str = Field(default="origin", min_length=1, max_length=100)

    @model_validator(mode="after")
    def _validate_checks(self) -> CodingPackConfig:
        check_ids = [check.id for check in self.checks]
        if len(check_ids) != len(set(check_ids)):
            raise ValueError("checks must use unique IDs")
        unknown_required = sorted(set(self.required_check_ids) - set(check_ids))
        if unknown_required:
            raise ValueError(
                "required_check_ids reference unknown checks: " + ", ".join(unknown_required)
            )
        if any(branch in self.protected_branches for branch in ("", self.branch_prefix)):
            raise ValueError("branch_prefix must not be an empty or protected branch")
        return self

    def resolve_workspace_root(self) -> Path:
        """Resolve the configured root and require it to be a Git worktree."""

        raw = Path(self.workspace_root).expanduser()
        root = (raw if raw.is_absolute() else Path.cwd() / raw).resolve()
        if not root.is_dir():
            raise ValueError(f"workspace_root is not a directory: {root}")
        if not (root / ".git").exists():
            raise ValueError(f"workspace_root is not a Git worktree: {root}")
        return root


def get_coding_config(ctx: ToolExecutionContext) -> CodingPackConfig:
    """Parse the current target's immutable coding configuration."""

    return CodingPackConfig.model_validate(ctx.pack_config)
