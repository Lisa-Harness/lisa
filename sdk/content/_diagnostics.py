"""Catalog diagnostics for the ``lisa-content`` MCP server.

Collects failures that are normally silently swallowed (broken entry points,
missing metadata siblings, malformed front-matter, missing ``_mode.md``) so
the kernel can inspect *why* a pack contributed nothing instead of seeing an
empty catalog with no explanation.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
from pathlib import Path
from typing import Any

from kernel.content_roots import _CONTENT_ROOT_ENTRY_POINT_GROUP
from kernel.pack_activation import load_disabled_pack_names_from_env

from sdk.content._catalog import (
    _DECLARATIVE_OPERATION_SUFFIX,
    _MODE_CONFIG_FILENAME,
    _OPERATION_METADATA_SUFFIX,
    _TOOL_METADATA_SUFFIX,
)
from sdk.content._front_matter import FrontMatterError, parse_front_matter

logger = logging.getLogger(__name__)

__all__ = ["build_catalog_diagnostics_json"]


def _entry_point_pack_name(ep: importlib.metadata.EntryPoint) -> str:
    dist = getattr(ep, "dist", None)
    if dist is None:
        return ""
    name = dist.metadata.get("Name", "") if dist.metadata else ""
    return str(name or "")


def _collect_root_discovery_failures(
    excluded: set[str],
) -> list[dict[str, Any]]:
    """Re-scan entry points and capture load failures + missing dirs."""
    failures: list[dict[str, Any]] = []
    try:
        eps = importlib.metadata.entry_points(group=_CONTENT_ROOT_ENTRY_POINT_GROUP)
    except Exception:  # pylint: disable=broad-except
        failures.append({
            "source": "entry_point_group",
            "entry_point": "",
            "error": "Failed to enumerate entry-point group",
        })
        return failures

    for ep in eps:
        if excluded and _entry_point_pack_name(ep) in excluded:
            continue
        try:
            root_loader = ep.load()
            candidate = Path(root_loader())
        except Exception as exc:  # pylint: disable=broad-except
            failures.append({
                "source": "entry_point_load",
                "entry_point": ep.name,
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue
        if not candidate.is_dir():
            failures.append({
                "source": "entry_point_missing_dir",
                "entry_point": ep.name,
                "path": str(candidate),
                "error": f"Content root path does not exist or is not a directory: {candidate}",
            })
    return failures


def _collect_mode_diagnostics(mode_roots: list[Path]) -> list[dict[str, Any]]:
    """Check each mode directory for missing _mode.md or invalid front-matter."""
    issues: list[dict[str, Any]] = []
    for root in mode_roots:
        if not root.is_dir():
            continue
        for mode_dir in sorted(root.iterdir()):
            if not mode_dir.is_dir():
                continue
            mode_config_path = mode_dir / _MODE_CONFIG_FILENAME
            if not mode_config_path.is_file():
                issues.append({
                    "source": "mode_missing_config",
                    "path": str(mode_config_path),
                    "mode_dir": mode_dir.name,
                    "error": f"Mode folder '{mode_dir.name}' is missing _mode.md",
                })
                continue
            try:
                text = mode_config_path.read_text(encoding="utf-8")
                front_matter, _ = parse_front_matter(text)
            except FrontMatterError as exc:
                issues.append({
                    "source": "mode_front_matter",
                    "path": str(mode_config_path),
                    "mode_dir": mode_dir.name,
                    "error": str(exc),
                })
                continue
            except OSError as exc:
                issues.append({
                    "source": "mode_read_error",
                    "path": str(mode_config_path),
                    "mode_dir": mode_dir.name,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue
            mode_id = front_matter.get("mode_id")
            if not mode_id or not isinstance(mode_id, str):
                issues.append({
                    "source": "mode_missing_id",
                    "path": str(mode_config_path),
                    "mode_dir": mode_dir.name,
                    "error": "Front-matter is missing 'mode_id' or it is not a string",
                })
    return issues


def _collect_operation_diagnostics(op_roots: list[Path]) -> list[dict[str, Any]]:
    """Check runtime operation modules for missing .yml metadata siblings."""
    issues: list[dict[str, Any]] = []
    for root in op_roots:
        if not root.is_dir():
            continue
        for py_path in sorted(root.rglob("*.py")):
            if py_path.name == "__init__.py" or "__pycache__" in py_path.parts:
                continue
            declarative_sibling = py_path.with_suffix(_DECLARATIVE_OPERATION_SUFFIX)
            if declarative_sibling.exists():
                continue
            yml_path = py_path.with_suffix(_OPERATION_METADATA_SUFFIX)
            if not yml_path.is_file():
                issues.append({
                    "source": "operation_missing_metadata",
                    "path": str(py_path),
                    "error": f"Missing metadata sibling '{yml_path.name}'",
                })
    return issues


def _collect_tool_diagnostics(tool_roots: list[Path]) -> list[dict[str, Any]]:
    """Check runtime tool modules for missing .tool.yml metadata siblings."""
    issues: list[dict[str, Any]] = []
    for root in tool_roots:
        if not root.is_dir():
            continue
        for py_path in sorted(root.rglob("*.py")):
            if py_path.name == "__init__.py" or "__pycache__" in py_path.parts:
                continue
            explicit = py_path.with_suffix(f".{_TOOL_METADATA_SUFFIX}")
            legacy = py_path.with_suffix(_OPERATION_METADATA_SUFFIX)
            if not explicit.is_file() and not legacy.is_file():
                issues.append({
                    "source": "tool_missing_metadata",
                    "path": str(py_path),
                    "error": f"Missing metadata sibling ('{explicit.name}' or '{legacy.name}')",
                })
    return issues


def build_catalog_diagnostics_json(
    all_roots: list[Path],
    mode_roots: list[Path],
    operation_roots: list[Path],
    tool_roots: list[Path],
) -> str:
    """Return a JSON diagnostics report covering root discovery and catalog scans.

    The report has this shape::

        {
          "root_discovery_failures": [...],
          "mode_issues": [...],
          "operation_issues": [...],
          "tool_issues": [...],
          "summary": {
            "total_roots": N,
            "total_issues": N
          }
        }
    """
    excluded = load_disabled_pack_names_from_env()
    root_failures = _collect_root_discovery_failures(excluded)
    mode_issues = _collect_mode_diagnostics(mode_roots)
    operation_issues = _collect_operation_diagnostics(operation_roots)
    tool_issues = _collect_tool_diagnostics(tool_roots)

    total_issues = len(root_failures) + len(mode_issues) + len(operation_issues) + len(tool_issues)
    report: dict[str, Any] = {
        "root_discovery_failures": root_failures,
        "mode_issues": mode_issues,
        "operation_issues": operation_issues,
        "tool_issues": tool_issues,
        "summary": {
            "total_roots": len(all_roots),
            "total_issues": total_issues,
        },
    }
    return json.dumps(report)