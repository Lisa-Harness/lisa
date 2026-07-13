"""Catalog and import-boundary tests for the Gmail pack.

These are deterministic evals (no model calls, no I/O). They mirror the
contract used by the framework: the pack must be importable, its content
root discoverable, and its operations/tools/modes well-formed.
"""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path

from kernel.models import ToolExecutionContext
from lisa_pack_gmail import get_content_root
from lisa_pack_gmail.config import gmail_settings
from lisa_pack_gmail.tools.gmail_create_draft.gmail_create_draft import (
    Parameters,
    execute,
)


def test_content_root_is_pack_dir() -> None:
    """``get_content_root`` returns the pack package directory."""

    pkg = importlib.import_module("lisa_pack_gmail")
    root = pkg.get_content_root()
    pkg_file = pkg.__file__
    assert pkg_file is not None
    assert root == Path(pkg_file).parent
    assert (root / "pyproject.toml").exists()


def test_import_boundaries() -> None:
    """The pack must not import ``hosts`` or any private app module."""

    # These imports are allowed and expected.
    importlib.import_module("lisa_pack_gmail")
    importlib.import_module("lisa_pack_gmail.config")
    importlib.import_module("lisa_pack_gmail.models")
    importlib.import_module("lisa_pack_gmail.tools.gmail_create_draft.gmail_create_draft")
    importlib.import_module("lisa_pack_gmail.operations.gmail_draft.gmail_draft")


def test_tool_metadata_present() -> None:
    """The ``gmail_create_draft`` tool metadata file exists."""

    yml = get_content_root() / "tools" / "gmail_create_draft" / "gmail_create_draft.tool.yml"
    assert yml.exists(), "tool yml missing"
    text = yml.read_text(encoding="utf-8")
    assert "tool_id: gmail_create_draft" in text
    assert "mode_allowlist: [gmail]" in text


def test_operation_metadata_present() -> None:
    """The ``gmail_draft`` operation metadata file exists."""

    yml = get_content_root() / "operations" / "gmail_draft" / "gmail_draft.yml"
    assert yml.exists(), "operation yml missing"
    text = yml.read_text(encoding="utf-8")
    assert "operation_id: gmail_draft" in text
    assert "tool_ids: [gmail_create_draft]" in text


def test_mode_metadata_present() -> None:
    """The ``gmail`` mode metadata file exists and is well-formed."""

    mode_md = get_content_root() / "modes" / "gmail" / "_mode.md"
    assert mode_md.exists(), "mode md missing"
    text = mode_md.read_text(encoding="utf-8")
    assert "mode_id: gmail" in text
    assert "mode_kind: action" in text
    assert "entry_operation_ids: [gmail_draft]" in text


def test_config_defaults() -> None:
    """``gmail_settings`` defaults the API base and reports unconfigured."""

    # No GMAIL_TOKEN in the test environment.
    settings = gmail_settings()
    assert settings.api_base == "https://gmail.googleapis.com/gmail/v1"
    assert settings.is_configured is False


def test_tool_parameters_schema() -> None:
    """The tool's ``Parameters`` model has the expected fields."""

    fields = set(Parameters.model_fields)
    assert {"to", "subject", "body"} <= fields


def test_tool_not_configured_message() -> None:
    """The tool returns a clear error when Gmail is not configured."""

    ctx = ToolExecutionContext(
        tool_id="gmail_create_draft",
        parameters={"to": "a@b.com", "subject": "x", "body": "y"},
    )
    result = asyncio.run(execute(ctx))
    assert result.error
    assert "GMAIL_TOKEN" in result.error