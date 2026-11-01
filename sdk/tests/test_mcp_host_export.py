"""Tests for the pure MCP client-config export in ``sdk.mcp_host.export``.

The export is the interoperability surface: it turns the host's compiled
first-party server configs into an ``mcpServers`` document another agent
client can paste. These tests hold it to the two properties that make that
safe — never leak inherited environment or credentials, and never silently
drop or silently widen a server — plus the exact document shape.
"""

from __future__ import annotations

import json

import pytest
from kernel.mcp_launch.policy import ToolPolicy
from kernel.tools.mcp_client import MCPClientConfig
from sdk.mcp_host.export import (
    EXPORT_FORMAT_MCPSERVERS,
    EXPORTABLE_ENV_KEYS,
    MCPServerExport,
    export_mcp_servers,
    render_mcp_export,
    render_mcp_export_markdown,
)


def _config(server_id: str = "lisa-storage", **overrides: object) -> MCPClientConfig:
    values: dict[str, object] = {
        "server_id": server_id,
        "command": f"/venv/bin/{server_id}",
        "args": ["--data-dir", "/home/u/.lisa"],
        "runtime_kind": "venv_bin",
    }
    values.update(overrides)
    return MCPClientConfig(**values)  # type: ignore[arg-type]


# ── Document shape ────────────────────────────────────────────────────


def test_export_emits_mcpservers_document() -> None:
    export = export_mcp_servers([_config()])

    assert export.format == EXPORT_FORMAT_MCPSERVERS
    assert export.to_payload() == {
        "mcpServers": {
            "lisa-storage": {
                "command": "/venv/bin/lisa-storage",
                "args": ["--data-dir", "/home/u/.lisa"],
            }
        }
    }


def test_render_export_is_valid_json_matching_payload() -> None:
    export = export_mcp_servers([_config()])

    assert json.loads(render_mcp_export(export)) == export.to_payload()


def test_export_omits_empty_env_and_cwd() -> None:
    """``env: {}`` / ``cwd: null`` would advertise configuration that is not there."""
    entry = export_mcp_servers([_config(env={}, cwd=None)]).servers[0].to_entry()

    assert "env" not in entry
    assert "cwd" not in entry


def test_export_preserves_args_order_and_embedded_values() -> None:
    config = _config(
        args=["--data-dir", "/d", "--thread-dir", "/t", "--user-slug", "alice", "--readonly"]
    )

    assert export_mcp_servers([config]).servers[0].args == tuple(config.args)


def test_export_sorts_servers_by_id() -> None:
    export = export_mcp_servers([_config("zeta"), _config("alpha"), _config("mid")])

    assert [server.server_id for server in export.servers] == ["alpha", "mid", "zeta"]
    assert list(export.to_payload()["mcpServers"]) == ["alpha", "mid", "zeta"]  # type: ignore[index]


def test_export_preserves_pinned_cwd() -> None:
    entry = export_mcp_servers([_config(cwd="/home/u/project")]).servers[0].to_entry()

    assert entry["cwd"] == "/home/u/project"


# ── Environment discipline ────────────────────────────────────────────


def test_export_never_leaks_inherited_parent_environment() -> None:
    """A compiled env is the whole parent environment plus overrides.

    Serializing it verbatim would put every secret in the operator's shell
    into a file they paste into another tool.
    """
    config = _config(
        env={
            "PATH": "/usr/bin",
            "HOME": "/home/u",
            "AWS_SECRET_ACCESS_KEY": "super-secret",
            "GITHUB_TOKEN": "ghp_secret",
            "LISA_STORAGE_DENY_GLOBS": "*.env",
        }
    )

    env = export_mcp_servers([config]).servers[0].env

    assert set(env) == {"LISA_STORAGE_DENY_GLOBS"}
    assert env["LISA_STORAGE_DENY_GLOBS"] == "*.env"
    assert "super-secret" not in render_mcp_export(export_mcp_servers([config]))
    assert "ghp_secret" not in render_mcp_export(export_mcp_servers([config]))


def test_export_placeholders_credential_valued_variables() -> None:
    export = export_mcp_servers([_config(env={"LISA_EMBEDDING_API_KEY": "sk-live-secret"})])

    env = export.servers[0].env
    assert env["LISA_EMBEDDING_API_KEY"] == "${LISA_EMBEDDING_API_KEY}"
    assert "sk-live-secret" not in render_mcp_export(export)
    assert any("LISA_EMBEDDING_API_KEY" in warning for warning in export.warnings)


def test_export_does_not_warn_when_credential_is_absent() -> None:
    """An unset credential is not "substitute this" — it was never configured."""
    export = export_mcp_servers([_config(env={"LISA_EMBEDDING_API_KEY": ""})])

    assert not [w for w in export.warnings if "credential" in w]


def test_exportable_env_table_excludes_obvious_secret_names() -> None:
    """Guard the allowlist itself: it is the review point for env forwarding."""
    forbidden = {"PATH", "HOME", "AWS_SECRET_ACCESS_KEY", "GITHUB_TOKEN", "OPENAI_API_KEY"}

    assert EXPORTABLE_ENV_KEYS & forbidden == set()


# ── Selection and skips ───────────────────────────────────────────────


def test_export_restricts_to_requested_server_ids() -> None:
    export = export_mcp_servers(
        [_config("lisa-memory"), _config("lisa-storage")], server_ids=["lisa-memory"]
    )

    assert [server.server_id for server in export.servers] == ["lisa-memory"]
    assert export.skipped == ()


def test_export_unknown_server_id_raises() -> None:
    with pytest.raises(ValueError, match="unknown server id"):
        export_mcp_servers([_config()], server_ids=["nope"])


def test_export_unknown_server_id_error_lists_configured_ids() -> None:
    with pytest.raises(ValueError, match="lisa-storage"):
        export_mcp_servers([_config()], server_ids=["nope"])


def test_export_rejects_unknown_format() -> None:
    with pytest.raises(ValueError, match="unsupported export format"):
        export_mcp_servers([_config()], format="claude_desktop")


def test_export_skips_in_process_without_a_resolver() -> None:
    export = export_mcp_servers([_config("lisa-memory", command="", runtime_kind="in_process")])

    assert export.servers == ()
    assert export.skipped[0][0] == "lisa-memory"
    assert "command_resolver" in export.skipped[0][1]


def test_export_uses_command_resolver_for_in_process_configs() -> None:
    export = export_mcp_servers(
        [_config("lisa-memory", command="", runtime_kind="in_process")],
        command_resolver=lambda config: f"/venv/bin/{config.server_id}",
    )

    assert export.servers[0].command == "/venv/bin/lisa-memory"
    assert export.skipped == ()


def test_export_skips_when_resolver_returns_none() -> None:
    export = export_mcp_servers(
        [_config("lisa-memory", command="", runtime_kind="in_process")],
        command_resolver=lambda _config: None,
    )

    assert export.is_empty
    assert "no spawnable command" in export.skipped[0][1]


def test_export_skips_download_only_binary() -> None:
    """The binary path is resolved by Lisa's async startup, not by this projection."""
    from kernel.mcp_binary import MCPBinaryDownload

    config = _config(
        "remote-bin",
        command="",
        runtime_kind="binary",
        binary_download=MCPBinaryDownload(
            repository="owner/repo", binary_name="server", version="v1.0.0"
        ),
    )

    export = export_mcp_servers([config])

    assert export.is_empty
    assert "binary download" in export.skipped[0][1]


def test_export_skips_internal_servers_by_default() -> None:
    """``lisa-content`` is host plumbing: it is hidden from Lisa's own model."""
    export = export_mcp_servers([_config(tool_policy=ToolPolicy(allowed_tools=[]))])

    assert export.is_empty
    assert "internal plumbing" in export.skipped[0][1]


def test_export_can_include_internal_servers_explicitly() -> None:
    export = export_mcp_servers(
        [_config(tool_policy=ToolPolicy(allowed_tools=[]))], include_internal=True
    )

    assert [server.server_id for server in export.servers] == ["lisa-storage"]


def test_export_skips_http_by_default_and_reports_why() -> None:
    config = _config("remote", command="", runtime_kind="http", url="https://example.test/mcp")

    export = export_mcp_servers([config])

    assert export.is_empty
    assert "credential headers" in export.skipped[0][1]


def test_export_http_entry_carries_url_and_header_names_only() -> None:
    config = _config(
        "remote",
        command="",
        runtime_kind="http",
        url="https://example.test/mcp",
        headers={"Authorization": "Bearer secret-token", "X-Scope": "team"},
    )

    export = export_mcp_servers([config], include_http=True)

    server = export.servers[0]
    assert server.transport == "http"
    assert server.header_names == ("Authorization", "X-Scope")
    assert server.to_entry() == {"type": "http", "url": "https://example.test/mcp"}
    assert "secret-token" not in render_mcp_export(export)
    assert any("Authorization" in warning for warning in export.warnings)


def test_export_skips_http_without_url() -> None:
    export = export_mcp_servers(
        [_config("remote", command="", runtime_kind="http", url="")], include_http=True
    )

    assert export.is_empty
    assert "no resolved url" in export.skipped[0][1]


def test_export_warns_that_a_restrictive_tool_policy_is_widened() -> None:
    config = _config(tool_policy=ToolPolicy(allowed_tools=["read_file"]))

    export = export_mcp_servers([config])

    assert export.servers[0].server_id == "lisa-storage"
    assert any("restricted to read_file" in warning for warning in export.warnings)
    assert any("full native surface" in note for note in export.servers[0].notes)


def test_export_warns_about_a_denied_tool_policy() -> None:
    config = _config(tool_policy=ToolPolicy(denied_tools=["delete_file"]))

    export = export_mcp_servers([config])

    assert any("hides delete_file" in warning for warning in export.warnings)


def test_export_notes_missing_log_path() -> None:
    """Without ``LISA_MCP_LOG_PATH`` a server logs into the client's cwd."""
    export = export_mcp_servers([_config(env={})])

    assert any("LISA_MCP_LOG_PATH" in note for note in export.servers[0].notes)


def test_export_notes_capability_publishing() -> None:
    config = _config(
        "lisa-memory",
        capability="memory",
        capability_tools=("add", "search"),
        capability_optional_tools=("get",),
        tool_map={"add": "add_memory"},
    )

    server = export_mcp_servers([config]).servers[0]

    assert server.capability == "memory"
    assert server.published_tools == ("memory.add", "memory.get", "memory.search")
    assert any("capability 'memory'" in note for note in server.notes)


def test_export_records_data_dir() -> None:
    assert export_mcp_servers([_config()], data_dir="/home/u/.lisa").data_dir == "/home/u/.lisa"


# ── Shell rendering ───────────────────────────────────────────────────


def test_shell_command_quotes_each_token() -> None:
    server = MCPServerExport(
        server_id="s",
        transport="stdio",
        command="/venv/bin/lisa-storage",
        args=("--data-dir", "/home/u/My Data"),
        env={"LISA_STORAGE_DENY_GLOBS": "*.env"},
    )

    command = server.to_shell_command()

    assert command.startswith("LISA_STORAGE_DENY_GLOBS=")
    assert "'/home/u/My Data'" in command
    assert command.endswith("/venv/bin/lisa-storage --data-dir '/home/u/My Data'")


# ── Markdown rendering ────────────────────────────────────────────────


def test_markdown_includes_document_warnings_and_skips() -> None:
    export = export_mcp_servers(
        [
            _config(env={"LISA_EMBEDDING_API_KEY": "sk-secret"}),
            _config(
                "remote",
                command="",
                runtime_kind="http",
                url="https://example.test/mcp",
            ),
        ],
        data_dir="/home/u/.lisa",
    )

    text = render_mcp_export_markdown(export)

    assert "```json" in text
    assert "Data directory: `/home/u/.lisa`" in text
    assert "### Before you use this" in text
    assert "### Not exported" in text
    assert "`remote`" in text
    assert "sk-secret" not in text


def test_markdown_reports_an_empty_export_explicitly() -> None:
    text = render_mcp_export_markdown(export_mcp_servers([]))

    assert "No configured server can be exported" in text
    assert "```json" not in text
    assert "### Direct invocation" not in text


def test_markdown_omits_direct_invocation_for_http_only_exports() -> None:
    export = export_mcp_servers(
        [_config("remote", command="", runtime_kind="http", url="https://example.test/mcp")],
        include_http=True,
    )

    text = render_mcp_export_markdown(export)

    assert "### Direct invocation" not in text
