"""Tests for the pure first-party MCP server config builders in sdk.mcp_host.

These cover the reusable spawning-config surface that any host (CLI, web,
external) calls to wire ``lisa-content`` and ``lisa-catalog``. The builders
are pure projections: they take content roots, disabled-pack sets, and
provider selections and return
:class:`~kernel.tools.mcp_client.MCPClientConfig` entries without I/O or
runtime state. Lifecycle (pool start / close) is host-owned and tested
elsewhere.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from kernel.session_plan import ProviderSelections
from kernel.tools.mcp_client import MCPClientConfig
from sdk.mcp_host import (
    build_catalog_mcp_config,
    build_content_mcp_config,
    session_provider_selection_payload,
)


def test_build_content_mcp_config_uses_discovered_roots_and_data_dir(tmp_path: Path) -> None:
    pack_root = tmp_path / "external-pack"
    (pack_root / "content").mkdir(parents=True, exist_ok=True)
    data_dir = tmp_path / "data"

    config = build_content_mcp_config(
        data_dir=data_dir,
        content_roots=[pack_root],
        disabled_packs=set(),
        provider_selections=ProviderSelections(),
    )

    assert config.server_id == "lisa-content"
    assert config.command == sys.executable
    assert config.args is not None
    assert "-m" in config.args
    assert "mcp_server.content.server" in config.args
    assert "--data-dir" in config.args
    data_dir_index = config.args.index("--data-dir")
    assert config.args[data_dir_index + 1] == str(data_dir)

    assert config.env is not None
    roots_value = str(config.env.get("SDK_CONTENT_ROOTS", ""))
    configured_roots = [part for part in roots_value.split(os.pathsep) if part]
    assert str(pack_root.resolve()) in configured_roots
    assert str((pack_root / "content").resolve()) in configured_roots


def test_build_content_mcp_config_without_roots_keeps_empty_sdk_content_roots(
    tmp_path: Path,
) -> None:
    config = build_content_mcp_config(
        data_dir=tmp_path,
        content_roots=[],
        disabled_packs=set(),
        provider_selections=ProviderSelections(),
    )

    assert config.env is not None
    assert str(config.env.get("SDK_CONTENT_ROOTS", "")) == ""


def test_build_content_mcp_config_serializes_session_provider_selections(
    tmp_path: Path,
) -> None:
    config = build_content_mcp_config(
        data_dir=tmp_path,
        content_roots=[],
        disabled_packs=set(),
        provider_selections=ProviderSelections(
            modes={"default": "lisa-pack-core"},
            operations={"web_search": "lisa-pack-search"},
            tools={"web_search": "lisa-pack-brave"},
        ),
    )

    assert config.env is not None
    payload = json.loads(config.env["LISA_SESSION_PROVIDER_SELECTIONS"])
    assert payload == {
        "modes": {"default": "lisa-pack-core"},
        "operations": {"web_search": "lisa-pack-search"},
        "tools": {"web_search": "lisa-pack-brave"},
    }


def test_build_content_mcp_config_passes_disabled_packs_to_child(tmp_path: Path) -> None:
    config = build_content_mcp_config(
        data_dir=tmp_path,
        content_roots=[],
        disabled_packs={"lisa-pack-markdown", "lisa-pack-search"},
        provider_selections=ProviderSelections(),
    )

    assert config.env is not None
    disabled = config.env.get("LISA_DISABLED_PACKS", "")
    assert "lisa-pack-markdown" in disabled
    assert "lisa-pack-search" in disabled


def test_build_content_mcp_config_injects_packs_venv_site_packages(tmp_path: Path) -> None:
    venv_site = tmp_path / "packs-venv" / "site-packages"
    venv_site.mkdir(parents=True)

    config = build_content_mcp_config(
        data_dir=tmp_path,
        content_roots=[],
        disabled_packs=set(),
        provider_selections=ProviderSelections(),
        packs_venv_site_packages=venv_site,
    )

    assert config.env is not None
    assert config.env.get("LISA_PACKS_VENV_SITE_PACKAGES") == str(venv_site)


def test_build_content_mcp_config_omits_packs_venv_when_missing(tmp_path: Path) -> None:
    config = build_content_mcp_config(
        data_dir=tmp_path,
        content_roots=[],
        disabled_packs=set(),
        provider_selections=ProviderSelections(),
        packs_venv_site_packages=None,
    )

    assert config.env is not None
    assert "LISA_PACKS_VENV_SITE_PACKAGES" not in config.env


def test_build_content_mcp_config_prepends_overlays_dir(tmp_path: Path) -> None:
    pack_root = tmp_path / "pack"
    pack_root.mkdir()
    overlays = tmp_path / "overlays"
    overlays.mkdir()

    config = build_content_mcp_config(
        data_dir=tmp_path,
        content_roots=[pack_root],
        disabled_packs=set(),
        provider_selections=ProviderSelections(),
        overlays_dir=overlays,
    )

    assert config.env is not None
    roots = config.env.get("SDK_CONTENT_ROOTS", "")
    assert roots.startswith(str(overlays) + os.sep) or roots.startswith(str(overlays) + os.pathsep)
    assert config.env.get("LISA_OVERLAY_ROOT") == str(overlays)


def test_build_content_mcp_config_logs_to_data_dir(tmp_path: Path) -> None:
    config = build_content_mcp_config(
        data_dir=tmp_path,
        content_roots=[],
        disabled_packs=set(),
        provider_selections=ProviderSelections(),
    )

    assert config.env is not None
    assert config.env.get("LISA_MCP_LOG_PATH") == str(tmp_path / "lisa-content.jsonl")


def test_build_catalog_mcp_config_reuses_content_env_and_splits_log_path(
    tmp_path: Path,
) -> None:
    pack_root = tmp_path / "pack"
    pack_root.mkdir()
    data_dir = tmp_path / "data"

    content_config = build_content_mcp_config(
        data_dir=data_dir,
        content_roots=[pack_root],
        disabled_packs={"lisa-pack-markdown"},
        provider_selections=ProviderSelections(),
    )

    catalog_config = build_catalog_mcp_config(content_config=content_config, data_dir=data_dir)

    assert catalog_config.server_id == "lisa-catalog"
    assert catalog_config.command == content_config.command
    assert catalog_config.args is not None
    assert "mcp_server.catalog.server" in catalog_config.args
    assert "--data-dir" in catalog_config.args
    data_dir_index = catalog_config.args.index("--data-dir")
    assert catalog_config.args[data_dir_index + 1] == str(data_dir)

    assert catalog_config.env is not None
    # Reused env from content config.
    assert catalog_config.env.get("SDK_CONTENT_ROOTS") == content_config.env.get(
        "SDK_CONTENT_ROOTS"
    )
    assert catalog_config.env.get("LISA_DISABLED_PACKS") == "lisa-pack-markdown"
    # Log path overridden to the catalog's own file.
    assert catalog_config.env.get("LISA_MCP_LOG_PATH") == str(data_dir / "lisa-catalog.jsonl")


def test_build_catalog_mcp_config_handles_none_env(tmp_path: Path) -> None:
    # Construct a content config with no env to exercise the defensive copy.
    content_config = MCPClientConfig(
        server_id="lisa-content",
        command=sys.executable,
        args=["-m", "mcp_server.content.server"],
        env=None,
        runtime_kind="python",
    )

    catalog_config = build_catalog_mcp_config(content_config=content_config, data_dir=tmp_path)

    assert catalog_config.env is not None
    assert catalog_config.env.get("LISA_MCP_LOG_PATH") == str(tmp_path / "lisa-catalog.jsonl")


def test_session_provider_selection_payload_round_trips() -> None:
    selections = ProviderSelections(
        modes={"default": "lisa-pack-core"},
        operations={"web_search": "lisa-pack-search"},
        tools={"web_search": "lisa-pack-brave"},
    )

    payload = session_provider_selection_payload(selections)

    assert payload == {
        "modes": {"default": "lisa-pack-core"},
        "operations": {"web_search": "lisa-pack-search"},
        "tools": {"web_search": "lisa-pack-brave"},
    }
    # Empty selections yield empty payload values (no env var set downstream).
    empty = ProviderSelections()
    assert session_provider_selection_payload(empty) == {
        "modes": {},
        "operations": {},
        "tools": {},
    }
