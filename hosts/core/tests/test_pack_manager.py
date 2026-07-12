"""Tests for pack management functions.

Migrated from ``hosts/little_lisa/test_pack_manager.py``.

The pack management functions were re-homed from the
``hosts/little_lisa/main.py`` monolith into two modules:

- ``hosts.core.commands`` — pack install/uninstall, validation, enable/disable
- ``hosts.core.content_roots`` — installed pack lock load/write, sdk_content_roots
- ``hosts.core.mcp_pool`` — compile_mcp_configs, load_pack_server_exclusions

The ``_RUNTIME_STATE`` dict is gone. Tests construct a ``HostRuntime``
from a ``HostConfig`` (with ``data_dir`` set to ``tmp_path``) and pass
it to the runtime-aware functions.

Functions that were module-level globals in ``main.py`` (e.g.
``_sdk_content_roots``, ``_load_pack_server_exclusions``) are now
module-level functions in ``hosts.core.content_roots`` and
``hosts.core.mcp_pool`` respectively. Monkeypatch them at their new
import sites.
"""

from __future__ import annotations

# pylint: disable=protected-access

import asyncio
import os
from pathlib import Path
from typing import Any, cast

import pytest

from hosts.core import commands, content_roots, mcp_pool
from hosts.core.config import HostConfig
from hosts.core.runtime import HostRuntime


def _make_runtime(tmp_path: Path) -> HostRuntime:
    """Build a HostRuntime whose data_dir is tmp_path."""
    config = HostConfig.from_env(data_dir=str(tmp_path))
    return HostRuntime(config)


def _load_lock(runtime: HostRuntime) -> dict[str, Any]:
    return content_roots.load_installed_packs_lock(runtime.config.data_dir)


def _write_lock(runtime: HostRuntime, payload: dict[str, Any]) -> None:
    path = content_roots.runtime_pack_lock_path(runtime.config.data_dir)
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


@pytest.mark.asyncio
async def test_run_pack_install_rejected_by_policy_does_not_mutate_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(tmp_path)
    monkeypatch.setattr(
        commands,
        "_validate_pack_spec_for_install",
        lambda spec: (False, f"rejected:{spec}"),
    )

    runtime.restart_required = False
    runtime.restart_reason = ""

    ok, message = await commands.run_pack_install(runtime, "requests")

    assert ok is False
    assert message == "rejected:requests"
    assert runtime.restart_required is False
    assert runtime.restart_reason == ""


@pytest.mark.asyncio
async def test_run_pack_install_success_records_lock_and_restart(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(tmp_path)

    class _FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return (b"installed", b"")

    async def _fake_create_subprocess_exec(*args: object, **kwargs: object) -> _FakeProc:
        _ = args
        _ = kwargs
        return _FakeProc()

    monkeypatch.setattr(
        commands,
        "_validate_pack_spec_for_install",
        lambda spec: (True, ""),
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)
    runtime.restart_required = False
    runtime.restart_reason = ""

    ok, details = await commands.run_pack_install(runtime, "lisa-pack-cloudflare==0.1.0")

    assert ok is True
    assert "Installed 'lisa-pack-cloudflare'" in details
    assert runtime.restart_required is True
    assert runtime.restart_reason == "pack install: lisa-pack-cloudflare"

    lock = _load_lock(runtime)
    assert isinstance(lock.get("packs"), list)
    assert any(
        str(item.get("package_name")) == "lisa-pack-cloudflare"
        and str(item.get("source_kind")) == "package"
        for item in lock["packs"]
    )


@pytest.mark.asyncio
async def test_run_pack_install_local_folder_records_local_source_and_restart(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(tmp_path)

    class _FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return (b"installed", b"")

    async def _fake_create_subprocess_exec(*args: object, **kwargs: object) -> _FakeProc:
        _ = args
        _ = kwargs
        return _FakeProc()

    local_pack = tmp_path / "lisa-pack-learning"
    local_pack.mkdir(parents=True, exist_ok=True)
    (local_pack / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "lisa-pack-learning"',
                'version = "0.1.0"',
                'requires-python = ">=3.11"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        commands,
        "_validate_pack_spec_for_install",
        lambda spec: (True, ""),
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)
    runtime.restart_required = False
    runtime.restart_reason = ""

    ok, details = await commands.run_pack_install(runtime, str(local_pack))

    assert ok is True
    assert "Installed 'lisa-pack-learning' from local source" in details
    assert runtime.restart_required is True
    assert runtime.restart_reason == "pack install: lisa-pack-learning"

    lock = _load_lock(runtime)
    assert isinstance(lock.get("packs"), list)
    assert any(
        str(item.get("package_name")) == "lisa-pack-learning"
        and str(item.get("source_kind")) == "local"
        and str(item.get("source_ref")) == str(local_pack)
        for item in lock["packs"]
    )


@pytest.mark.asyncio
async def test_run_pack_uninstall_success_updates_lock_and_restart(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(tmp_path)

    class _FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return (b"removed", b"")

    async def _fake_create_subprocess_exec(*args: object, **kwargs: object) -> _FakeProc:
        _ = args
        _ = kwargs
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)
    _write_lock(
        runtime,
        {
            "packs": [
                {
                    "package_name": "lisa-pack-cloudflare",
                    "source_kind": "git",
                    "source_ref": "git+https://github.com/acme/lisa-pack-cloudflare.git@v0.1.0#egg=lisa-pack-cloudflare",
                }
            ]
        },
    )
    runtime.restart_required = False
    runtime.restart_reason = ""

    ok, details = await commands.run_pack_uninstall(runtime, "lisa-pack-cloudflare")

    assert ok is True
    assert "Uninstalled 'lisa-pack-cloudflare'" in details
    assert runtime.restart_required is True
    assert runtime.restart_reason == "pack uninstall: lisa-pack-cloudflare"

    lock = _load_lock(runtime)
    assert lock["packs"] == []


def test_validate_pack_spec_for_install_package_prefix_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LISA_PACK_ALLOWED_PREFIXES", "lisa-pack-,acme-pack-")

    ok, reason = commands._validate_pack_spec_for_install("acme-pack-observability==0.2.0")
    assert ok is True
    assert reason == ""

    ok, reason = commands._validate_pack_spec_for_install("requests==2.32.0")
    assert ok is False
    assert "allowed package prefixes" in reason


def test_validate_pack_spec_for_install_git_host_and_ref_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LISA_PACK_GIT_ALLOW_HOSTS", "github.com,dev.azure.com")
    monkeypatch.setenv("LISA_PACK_REQUIRE_GIT_REF", "true")

    ok, reason = commands._validate_pack_spec_for_install(
        "git+https://github.com/acme/lisa-pack-cloudflare.git@v1.2.3#egg=lisa-pack-cloudflare"
    )
    assert ok is True
    assert reason == ""

    ok, reason = commands._validate_pack_spec_for_install(
        "git+https://evil.example.com/acme/lisa-pack-cloudflare.git@main#egg=lisa-pack-cloudflare"
    )
    assert ok is False
    assert "not in the allowlist" in reason

    ok, reason = commands._validate_pack_spec_for_install(
        "git+https://github.com/acme/lisa-pack-cloudflare.git#egg=lisa-pack-cloudflare"
    )
    assert ok is False
    assert "must pin @tag/@branch/@sha" in reason


def test_validate_pack_spec_for_install_local_folder_accepts_prefixed_project_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LISA_PACK_ALLOWED_PREFIXES", "lisa-pack-")
    local_pack = tmp_path / "lisa-pack-learning"
    local_pack.mkdir(parents=True, exist_ok=True)
    (local_pack / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "lisa-pack-learning"',
                'version = "0.1.0"',
                'requires-python = ">=3.11"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    ok, reason = commands._validate_pack_spec_for_install(str(local_pack))
    assert ok is True
    assert reason == ""


def test_validate_pack_spec_for_install_local_folder_rejects_missing_pyproject_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LISA_PACK_ALLOWED_PREFIXES", "lisa-pack-")
    local_pack = tmp_path / "lisa-pack-learning"
    local_pack.mkdir(parents=True, exist_ok=True)
    (local_pack / "pyproject.toml").write_text(
        "\n".join(
            [
                "[build-system]",
                'requires = ["setuptools>=68"]',
                'build-backend = "setuptools.build_meta"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    ok, reason = commands._validate_pack_spec_for_install(str(local_pack))
    assert ok is False
    assert "missing a valid pyproject.toml project.name" in reason


def test_request_and_consume_restart_resume_message(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Restart request + resume-message env-var round-trip.

    In the new structure, ``runtime.request_restart(reason)`` sets the
    restart flag/reason, and the resume message is stored in the
    ``LISA_RESTART_RESUME_MESSAGE`` env var (read by the host on next
    launch). This test exercises the env-var half directly.
    """
    monkeypatch.delenv("LISA_RESTART_RESUME_MESSAGE", raising=False)
    runtime = _make_runtime(tmp_path)
    runtime.restart_required = False
    runtime.restart_reason = ""

    runtime.request_restart(reason="pack install: lisa-pack-cloudflare")
    monkeypatch.setenv("LISA_RESTART_RESUME_MESSAGE", "check cloudflare status")

    assert runtime.restart_required is True
    assert runtime.restart_reason == "pack install: lisa-pack-cloudflare"
    assert os.environ["LISA_RESTART_RESUME_MESSAGE"] == "check cloudflare status"

    # Simulate the next-launch consumption clearing the env var.
    monkeypatch.delenv("LISA_RESTART_RESUME_MESSAGE", raising=False)
    assert os.environ.get("LISA_RESTART_RESUME_MESSAGE", "") == ""


def test_load_pack_server_exclusions_supports_multiple_formats(tmp_path: Path) -> None:
    host_manifest = tmp_path / "mcp_servers.yaml"
    host_manifest.write_text(
        "\n".join(
            [
                "version: 1",
                "pack_policy:",
                "  exclude_servers:",
                "    - cloudflare.github",
                "    - cloudflare:workers",
                "    - analytics/search",
                "    - blocked-pack.*",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    exclusions = mcp_pool.load_pack_server_exclusions(host_manifest)

    assert "cloudflare.github" in exclusions
    assert "cloudflare:workers" in exclusions
    assert "analytics/search" in exclusions
    assert "blocked-pack.*" in exclusions


def test_compile_mcp_configs_namespaces_pack_server_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(tmp_path)
    pack_a_root = tmp_path / "pack-a"
    pack_b_root = tmp_path / "pack-b"
    pack_a_root.mkdir(parents=True, exist_ok=True)
    pack_b_root.mkdir(parents=True, exist_ok=True)

    pack_manifest_text = "\n".join(
        [
            "version: 1",
            "servers:",
            "  - server_id: shared-server",
            "    runtime_kind: http",
            "    url: https://example.com/mcp",
        ]
    ) + "\n"
    (pack_a_root / "mcp_servers.yaml").write_text(pack_manifest_text, encoding="utf-8")
    (pack_b_root / "mcp_servers.yaml").write_text(pack_manifest_text, encoding="utf-8")

    monkeypatch.setattr(mcp_pool, "sdk_content_roots", lambda *a, **kw: [pack_a_root, pack_b_root])
    monkeypatch.setattr(mcp_pool, "load_pack_server_exclusions", lambda _path: set())

    warm, cold = mcp_pool.compile_mcp_configs(runtime)
    server_ids = {config.server_id for config in [*warm, *cold]}

    assert "pack-a.shared-server" in server_ids
    assert "pack-b.shared-server" in server_ids


def test_compile_mcp_configs_applies_pack_server_exclusions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(tmp_path)
    pack_a_root = tmp_path / "pack-a"
    pack_b_root = tmp_path / "pack-b"
    pack_a_root.mkdir(parents=True, exist_ok=True)
    pack_b_root.mkdir(parents=True, exist_ok=True)

    pack_manifest_text = "\n".join(
        [
            "version: 1",
            "servers:",
            "  - server_id: shared-server",
            "    runtime_kind: http",
            "    url: https://example.com/mcp",
        ]
    ) + "\n"
    (pack_a_root / "mcp_servers.yaml").write_text(pack_manifest_text, encoding="utf-8")
    (pack_b_root / "mcp_servers.yaml").write_text(pack_manifest_text, encoding="utf-8")

    monkeypatch.setattr(mcp_pool, "sdk_content_roots", lambda *a, **kw: [pack_a_root, pack_b_root])
    monkeypatch.setattr(
        mcp_pool,
        "load_pack_server_exclusions",
        lambda _path: {"pack-a.shared-server"},
    )

    warm, cold = mcp_pool.compile_mcp_configs(runtime)
    server_ids = {config.server_id for config in [*warm, *cold]}

    assert "pack-a.shared-server" not in server_ids
    assert "pack-b.shared-server" in server_ids


def test_compile_mcp_configs_namespaces_pack_memory_servers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(tmp_path)
    pack_root = tmp_path / "learning"
    pack_root.mkdir(parents=True, exist_ok=True)

    pack_manifest_text = "\n".join(
        [
            "version: 1",
            "memory:",
            "  default_backend: lisa_memory",
            "  backends:",
            "    - id: lisa_memory",
            "      kind: stdio",
            '      command: "${LISA_MCP_COMMAND:lisa-memory}"',
            "      args:",
            "        - --data-dir",
            '        - "${LISA_DATA_DIR}"',
            "      tool_names:",
            "        add: add_memory",
            "        search: search_memory",
            "        list: list_memories",
            "    - id: mem0",
            "      kind: http",
            "      url: https://mcp.mem0.ai/mcp",
            "      headers:",
            '        Authorization: "Bearer ${MEM0_API_KEY}"',
            "      tool_names:",
            "        add: add_memory",
            "        search: search_memories",
            "        list: get_memories",
            "servers:",
            "  - server_id: lisa-memory",
            "    runtime_kind: venv_bin",
            "    entry_point: lisa-memory",
            '    venv_dir: "${FRAMEWORK_ROOT}/.venv"',
            "    warm: true",
            "    required: true",
            "  - server_id: mem0",
            "    runtime_kind: http",
            "    url: https://mcp.mem0.ai/mcp",
            "    headers:",
            '      Authorization: "Bearer ${MEM0_API_KEY}"',
        ]
    ) + "\n"
    (pack_root / "mcp_servers.yaml").write_text(pack_manifest_text, encoding="utf-8")

    monkeypatch.setattr(mcp_pool, "sdk_content_roots", lambda *a, **kw: [pack_root])
    monkeypatch.setattr(mcp_pool, "load_pack_server_exclusions", lambda _path: set())

    warm, cold = mcp_pool.compile_mcp_configs(runtime)
    server_ids = {config.server_id for config in [*warm, *cold]}

    assert "learning.lisa-memory" in server_ids
    assert "learning.mem0" in server_ids
    assert runtime.memory_config_pack_slug == "learning"


def test_set_pack_enabled_rejects_disabling_always_on_pack(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(tmp_path)
    monkeypatch.setattr(commands, "always_on_pack_names", lambda _path=None, **kw: {"lisa-pack-core"})

    ok, details = commands.set_pack_enabled(runtime, "core", enabled=False)

    assert ok is False
    assert "always on" in details


def test_set_pack_enabled_for_bundled_pack_records_lock_and_restart(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(tmp_path)
    bundled_root = tmp_path / "lisa_pack_cloudflare"
    bundled_root.mkdir(parents=True, exist_ok=True)

    runtime.restart_required = False
    runtime.restart_reason = ""
    monkeypatch.setattr(commands, "always_on_pack_names", lambda _path=None, **kw: {"lisa-pack-core"})
    monkeypatch.setattr(commands, "discover_installed_content_roots", lambda: [bundled_root])

    ok, details = commands.set_pack_enabled(runtime, "cloudflare", enabled=True)

    assert ok is True
    assert "Enabled 'lisa-pack-cloudflare'" in details
    assert runtime.restart_required is True
    assert runtime.restart_reason == "pack enable: lisa-pack-cloudflare"

    lock = _load_lock(runtime)
    assert any(
        entry.get("package_name") == "lisa-pack-cloudflare"
        and bool(entry.get("enabled", False)) is True
        and entry.get("source_kind") == "bundled"
        for entry in lock.get("packs", [])
    )


def test_sdk_content_roots_filters_bundled_activation_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    core_root = tmp_path / "sdk" / "packs" / "core"
    bundled_root = tmp_path / "lisa_pack_cloudflare"
    configured_root = tmp_path / "configured_root"
    core_root.mkdir(parents=True, exist_ok=True)
    bundled_root.mkdir(parents=True, exist_ok=True)
    configured_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(content_roots, "always_on_pack_names", lambda _path=None, **kw: {"lisa-pack-core"})
    monkeypatch.setattr(
        content_roots,
        "discover_installed_content_roots",
        lambda: [core_root, bundled_root],
    )
    monkeypatch.setattr(
        content_roots,
        "discover_configured_content_roots",
        lambda: [configured_root],
    )

    _write_lock(
        HostRuntime(HostConfig.from_env(data_dir=str(tmp_path))),
        {
            "packs": [
                {
                    "package_name": "lisa-pack-cloudflare",
                    "source_kind": "bundled",
                    "enabled": False,
                }
            ]
        },
    )

    roots_disabled = content_roots.sdk_content_roots(tmp_path)
    assert core_root in roots_disabled
    assert configured_root in roots_disabled
    assert bundled_root not in roots_disabled

    _write_lock(
        HostRuntime(HostConfig.from_env(data_dir=str(tmp_path))),
        {
            "packs": [
                {
                    "package_name": "lisa-pack-cloudflare",
                    "source_kind": "bundled",
                    "enabled": True,
                }
            ]
        },
    )
    roots_enabled = content_roots.sdk_content_roots(tmp_path)
    assert bundled_root in roots_enabled


@pytest.mark.asyncio
async def test_run_pack_install_autoselects_memory_provider_when_unset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(tmp_path)

    class _FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return (b"installed", b"")

    async def _fake_create_subprocess_exec(*args: object, **kwargs: object) -> _FakeProc:
        _ = args
        _ = kwargs
        return _FakeProc()

    mem0_root = tmp_path / "lisa_pack_mem0"
    mem0_root.mkdir(parents=True, exist_ok=True)
    (mem0_root / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "lisa-pack-mem0"',
                'version = "0.1.0"',
                'requires-python = ">=3.11"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (mem0_root / "mcp_servers.yaml").write_text(
        "\n".join(
            [
                "version: 1",
                "memory:",
                "  default_backend: mem0",
                "  backends:",
                "    - id: mem0",
                "      kind: http",
                "      url: https://mcp.mem0.ai/mcp",
                "      headers:",
                '        Authorization: "Bearer ${MEM0_API_KEY}"',
                "      tool_names:",
                "        add: add_memory",
                "        search: search_memories",
                "        list: get_memories",
                "servers:",
                "  - server_id: mem0",
                "    runtime_kind: http",
                "    url: https://mcp.mem0.ai/mcp",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        commands,
        "_validate_pack_spec_for_install",
        lambda spec: (True, ""),
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)
    monkeypatch.setattr(commands, "discover_installed_content_roots", lambda: [mem0_root])
    monkeypatch.setattr(commands, "always_on_pack_names", lambda _path=None, **kw: {"lisa-pack-core"})

    ok, _details = await commands.run_pack_install(runtime, "lisa-pack-mem0==0.1.0")

    assert ok is True
    lock = _load_lock(runtime)
    assert lock.get("selected_capability_providers", {}).get("memory") == "lisa-pack-mem0"


def test_compile_mcp_configs_prefers_runtime_installed_memory_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(tmp_path)
    learning_root = tmp_path / "learning-pack"
    mem0_root = tmp_path / "mem0-pack"
    learning_root.mkdir(parents=True, exist_ok=True)
    mem0_root.mkdir(parents=True, exist_ok=True)

    (learning_root / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "lisa-pack-learning"',
                'version = "0.1.0"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (mem0_root / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "lisa-pack-mem0"',
                'version = "0.1.0"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (learning_root / "mcp_servers.yaml").write_text(
        "\n".join(
            [
                "version: 1",
                "memory:",
                "  default_backend: lisa_memory",
                "  backends:",
                "    - id: lisa_memory",
                "      kind: stdio",
                '      command: "${LISA_MCP_COMMAND:lisa-memory}"',
                "      args: []",
                "      tool_names:",
                "        add: add_memory",
                "        search: search_memory",
                "        list: list_memories",
                "servers:",
                "  - server_id: lisa-memory",
                "    runtime_kind: venv_bin",
                "    entry_point: lisa-memory",
                '    venv_dir: "${FRAMEWORK_ROOT}/.venv"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (mem0_root / "mcp_servers.yaml").write_text(
        "\n".join(
            [
                "version: 1",
                "memory:",
                "  default_backend: mem0",
                "  backends:",
                "    - id: mem0",
                "      kind: http",
                "      url: https://mcp.mem0.ai/mcp",
                "      headers:",
                '        Authorization: "Bearer ${MEM0_API_KEY}"',
                "      tool_names:",
                "        add: add_memory",
                "        search: search_memories",
                "        list: get_memories",
                "servers:",
                "  - server_id: mem0",
                "    runtime_kind: http",
                "    url: https://mcp.mem0.ai/mcp",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _write_lock(
        runtime,
        {
            "packs": [
                {
                    "package_name": "lisa-pack-mem0",
                    "source_kind": "package",
                    "enabled": True,
                }
            ]
        },
    )
    monkeypatch.setattr(mcp_pool, "sdk_content_roots", lambda *a, **kw: [learning_root, mem0_root])
    monkeypatch.setattr(mcp_pool, "load_pack_server_exclusions", lambda _path: set())
    monkeypatch.setattr(mcp_pool, "load_capability_provider_preferences", lambda _path: {})

    _warm, _cold = mcp_pool.compile_mcp_configs(runtime)

    assert runtime.memory_config_pack_slug == "mem0-pack"


def test_compile_mcp_configs_honors_explicit_memory_provider_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(tmp_path)
    learning_root = tmp_path / "learning-pack"
    mem0_root = tmp_path / "mem0-pack"
    learning_root.mkdir(parents=True, exist_ok=True)
    mem0_root.mkdir(parents=True, exist_ok=True)

    (learning_root / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "lisa-pack-learning"',
                'version = "0.1.0"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (mem0_root / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "lisa-pack-mem0"',
                'version = "0.1.0"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (learning_root / "mcp_servers.yaml").write_text(
        "\n".join(
            [
                "version: 1",
                "memory:",
                "  default_backend: lisa_memory",
                "  backends:",
                "    - id: lisa_memory",
                "      kind: stdio",
                '      command: "${LISA_MCP_COMMAND:lisa-memory}"',
                "      args: []",
                "      tool_names:",
                "        add: add_memory",
                "        search: search_memory",
                "        list: list_memories",
                "servers:",
                "  - server_id: lisa-memory",
                "    runtime_kind: venv_bin",
                "    entry_point: lisa-memory",
                '    venv_dir: "${FRAMEWORK_ROOT}/.venv"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (mem0_root / "mcp_servers.yaml").write_text(
        "\n".join(
            [
                "version: 1",
                "memory:",
                "  default_backend: mem0",
                "  backends:",
                "    - id: mem0",
                "      kind: http",
                "      url: https://mcp.mem0.ai/mcp",
                "      headers:",
                '        Authorization: "Bearer ${MEM0_API_KEY}"',
                "      tool_names:",
                "        add: add_memory",
                "        search: search_memories",
                "        list: get_memories",
                "servers:",
                "  - server_id: mem0",
                "    runtime_kind: http",
                "    url: https://mcp.mem0.ai/mcp",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _write_lock(
        runtime,
        {
            "packs": [
                {
                    "package_name": "lisa-pack-mem0",
                    "source_kind": "package",
                    "enabled": True,
                }
            ],
            "selected_capability_providers": {
                "memory": "lisa-pack-learning",
            },
        },
    )
    monkeypatch.setattr(mcp_pool, "sdk_content_roots", lambda *a, **kw: [learning_root, mem0_root])
    monkeypatch.setattr(mcp_pool, "load_pack_server_exclusions", lambda _path: set())
    monkeypatch.setattr(mcp_pool, "load_capability_provider_preferences", lambda _path: {})

    _warm, _cold = mcp_pool.compile_mcp_configs(runtime)

    assert runtime.memory_config_pack_slug == "learning-pack"