from __future__ import annotations

# pylint: disable=protected-access

import asyncio
from typing import Any, cast
from pathlib import Path

import pytest

from . import main as little_lisa


def _state() -> dict[str, object]:
    return cast(dict[str, object], getattr(little_lisa, "_RUNTIME_STATE"))


def _fn(name: str) -> Any:
    return getattr(little_lisa, name)


@pytest.mark.asyncio
async def test_run_pack_install_rejected_by_policy_does_not_mutate_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        little_lisa,
        "_validate_pack_spec_for_install",
        lambda spec: (False, f"rejected:{spec}"),
    )

    _state()["restart_required"] = False
    _state()["restart_reason"] = ""

    ok, message = await _fn("_run_pack_install")("requests")

    assert ok is False
    assert message == "rejected:requests"
    assert _state()["restart_required"] is False
    assert _state()["restart_reason"] == ""


@pytest.mark.asyncio
async def test_run_pack_install_success_records_lock_and_restart(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return (b"installed", b"")

    async def _fake_create_subprocess_exec(*args: object, **kwargs: object) -> _FakeProc:
        _ = args
        _ = kwargs
        return _FakeProc()

    monkeypatch.setattr(
        little_lisa,
        "_validate_pack_spec_for_install",
        lambda spec: (True, ""),
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)
    monkeypatch.setitem(_state(), "data_dir", tmp_path)
    monkeypatch.setitem(_state(), "restart_required", False)
    monkeypatch.setitem(_state(), "restart_reason", "")

    ok, details = await _fn("_run_pack_install")("lisa-pack-cloudflare==0.1.0")

    assert ok is True
    assert "Installed 'lisa-pack-cloudflare'" in details
    assert _state()["restart_required"] is True
    assert _state()["restart_reason"] == "pack install: lisa-pack-cloudflare"

    lock = _fn("_load_installed_packs_lock")()
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
        little_lisa,
        "_validate_pack_spec_for_install",
        lambda spec: (True, ""),
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)
    monkeypatch.setitem(_state(), "data_dir", tmp_path)
    monkeypatch.setitem(_state(), "restart_required", False)
    monkeypatch.setitem(_state(), "restart_reason", "")

    ok, details = await _fn("_run_pack_install")(str(local_pack))

    assert ok is True
    assert "Installed 'lisa-pack-learning' from local source" in details
    assert _state()["restart_required"] is True
    assert _state()["restart_reason"] == "pack install: lisa-pack-learning"

    lock = _fn("_load_installed_packs_lock")()
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
    class _FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return (b"removed", b"")

    async def _fake_create_subprocess_exec(*args: object, **kwargs: object) -> _FakeProc:
        _ = args
        _ = kwargs
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)
    monkeypatch.setitem(_state(), "data_dir", tmp_path)
    _fn("_write_installed_packs_lock")(
        {
            "packs": [
                {
                    "package_name": "lisa-pack-cloudflare",
                    "source_kind": "git",
                    "source_ref": "git+https://github.com/acme/lisa-pack-cloudflare.git@v0.1.0#egg=lisa-pack-cloudflare",
                }
            ]
        }
    )
    monkeypatch.setitem(_state(), "restart_required", False)
    monkeypatch.setitem(_state(), "restart_reason", "")

    ok, details = await _fn("_run_pack_uninstall")("lisa-pack-cloudflare")

    assert ok is True
    assert "Uninstalled 'lisa-pack-cloudflare'" in details
    assert _state()["restart_required"] is True
    assert _state()["restart_reason"] == "pack uninstall: lisa-pack-cloudflare"

    lock = _fn("_load_installed_packs_lock")()
    assert lock["packs"] == []


def test_validate_pack_spec_for_install_package_prefix_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITTLE_LISA_PACK_ALLOWED_PREFIXES", "lisa-pack-,acme-pack-")

    ok, reason = _fn("_validate_pack_spec_for_install")("acme-pack-observability==0.2.0")
    assert ok is True
    assert reason == ""

    ok, reason = _fn("_validate_pack_spec_for_install")("requests==2.32.0")
    assert ok is False
    assert "allowed package prefixes" in reason


def test_validate_pack_spec_for_install_git_host_and_ref_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITTLE_LISA_PACK_GIT_ALLOW_HOSTS", "github.com,dev.azure.com")
    monkeypatch.setenv("LITTLE_LISA_PACK_REQUIRE_GIT_REF", "true")

    ok, reason = _fn("_validate_pack_spec_for_install")(
        "git+https://github.com/acme/lisa-pack-cloudflare.git@v1.2.3#egg=lisa-pack-cloudflare"
    )
    assert ok is True
    assert reason == ""

    ok, reason = _fn("_validate_pack_spec_for_install")(
        "git+https://evil.example.com/acme/lisa-pack-cloudflare.git@main#egg=lisa-pack-cloudflare"
    )
    assert ok is False
    assert "not in the allowlist" in reason

    ok, reason = _fn("_validate_pack_spec_for_install")(
        "git+https://github.com/acme/lisa-pack-cloudflare.git#egg=lisa-pack-cloudflare"
    )
    assert ok is False
    assert "must pin @tag/@branch/@sha" in reason


def test_validate_pack_spec_for_install_local_folder_accepts_prefixed_project_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LITTLE_LISA_PACK_ALLOWED_PREFIXES", "lisa-pack-")
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

    ok, reason = _fn("_validate_pack_spec_for_install")(str(local_pack))
    assert ok is True
    assert reason == ""


def test_validate_pack_spec_for_install_local_folder_rejects_missing_pyproject_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LITTLE_LISA_PACK_ALLOWED_PREFIXES", "lisa-pack-")
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

    ok, reason = _fn("_validate_pack_spec_for_install")(str(local_pack))
    assert ok is False
    assert "missing a valid pyproject.toml project.name" in reason


def test_request_and_consume_restart_resume_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITTLE_LISA_RESTART_RESUME_MESSAGE", raising=False)
    monkeypatch.setitem(_state(), "restart_required", False)
    monkeypatch.setitem(_state(), "restart_reason", "")

    _fn("_request_runtime_restart")(
        reason="pack install: lisa-pack-cloudflare",
        resume_message="check cloudflare status",
    )

    assert _state()["restart_required"] is True
    assert _state()["restart_reason"] == "pack install: lisa-pack-cloudflare"
    assert (
        _fn("_consume_restart_resume_message")()
        == "check cloudflare status"
    )
    assert _fn("_consume_restart_resume_message")() == ""


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

    exclusions = _fn("_load_pack_server_exclusions")(host_manifest)

    assert "cloudflare.github" in exclusions
    assert "cloudflare:workers" in exclusions
    assert "analytics/search" in exclusions
    assert "blocked-pack.*" in exclusions


def test_compile_mcp_configs_namespaces_pack_server_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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

    monkeypatch.setattr(little_lisa, "_sdk_content_roots", lambda: [pack_a_root, pack_b_root])
    monkeypatch.setattr(little_lisa, "_load_pack_server_exclusions", lambda _path: set())

    warm, cold = _fn("_compile_mcp_configs")()
    server_ids = {config.server_id for config in [*warm, *cold]}

    assert "pack-a.shared-server" in server_ids
    assert "pack-b.shared-server" in server_ids


def test_compile_mcp_configs_applies_pack_server_exclusions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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

    monkeypatch.setattr(little_lisa, "_sdk_content_roots", lambda: [pack_a_root, pack_b_root])
    monkeypatch.setattr(
        little_lisa,
        "_load_pack_server_exclusions",
        lambda _path: {"pack-a.shared-server"},
    )

    warm, cold = _fn("_compile_mcp_configs")()
    server_ids = {config.server_id for config in [*warm, *cold]}

    assert "pack-a.shared-server" not in server_ids
    assert "pack-b.shared-server" in server_ids


def test_compile_mcp_configs_namespaces_pack_memory_servers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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
            "      command: \"${LISA_MCP_COMMAND:lisa-memory}\"",
            "      args:",
            "        - --data-dir",
            "        - \"${LISA_DATA_DIR}\"",
            "      tool_names:",
            "        add: add_memory",
            "        search: search_memory",
            "        list: list_memories",
            "    - id: mem0",
            "      kind: http",
            "      url: https://mcp.mem0.ai/mcp",
            "      headers:",
            "        Authorization: \"Bearer ${MEM0_API_KEY}\"",
            "      tool_names:",
            "        add: add_memory",
            "        search: search_memories",
            "        list: get_memories",
            "servers:",
            "  - server_id: lisa-memory",
            "    runtime_kind: venv_bin",
            "    entry_point: lisa-memory",
            "    venv_dir: \"${FRAMEWORK_ROOT}/.venv\"",
            "    warm: true",
            "    required: true",
            "  - server_id: mem0",
            "    runtime_kind: http",
            "    url: https://mcp.mem0.ai/mcp",
            "    headers:",
            "      Authorization: \"Bearer ${MEM0_API_KEY}\"",
        ]
    ) + "\n"
    (pack_root / "mcp_servers.yaml").write_text(pack_manifest_text, encoding="utf-8")

    monkeypatch.setattr(little_lisa, "_sdk_content_roots", lambda: [pack_root])
    monkeypatch.setattr(little_lisa, "_load_pack_server_exclusions", lambda _path: set())

    warm, cold = _fn("_compile_mcp_configs")()
    server_ids = {config.server_id for config in [*warm, *cold]}

    assert "learning.lisa-memory" in server_ids
    assert "learning.mem0" in server_ids
    assert _state()["memory_config_pack_slug"] == "learning"


def test_set_pack_enabled_rejects_disabling_always_on_pack(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setitem(_state(), "data_dir", tmp_path)
    monkeypatch.setattr(little_lisa, "_always_on_pack_names", lambda _path=None: {"lisa-pack-core"})

    ok, details = _fn("_set_pack_enabled")("core", enabled=False)

    assert ok is False
    assert "always on" in details


def test_set_pack_enabled_for_bundled_pack_records_lock_and_restart(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundled_root = tmp_path / "lisa_pack_cloudflare"
    bundled_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setitem(_state(), "data_dir", tmp_path)
    monkeypatch.setitem(_state(), "restart_required", False)
    monkeypatch.setitem(_state(), "restart_reason", "")
    monkeypatch.setattr(little_lisa, "_always_on_pack_names", lambda _path=None: {"lisa-pack-core"})
    monkeypatch.setattr(little_lisa, "discover_installed_content_roots", lambda: [bundled_root])

    ok, details = _fn("_set_pack_enabled")("cloudflare", enabled=True)

    assert ok is True
    assert "Enabled 'lisa-pack-cloudflare'" in details
    assert _state()["restart_required"] is True
    assert _state()["restart_reason"] == "pack enable: lisa-pack-cloudflare"

    lock = _fn("_load_installed_packs_lock")()
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

    monkeypatch.setitem(_state(), "data_dir", tmp_path)
    monkeypatch.setattr(little_lisa, "_always_on_pack_names", lambda _path=None: {"lisa-pack-core"})
    monkeypatch.setattr(
        little_lisa,
        "discover_installed_content_roots",
        lambda: [core_root, bundled_root],
    )
    monkeypatch.setattr(
        little_lisa,
        "discover_configured_content_roots",
        lambda: [configured_root],
    )

    _fn("_write_installed_packs_lock")(
        {
            "packs": [
                {
                    "package_name": "lisa-pack-cloudflare",
                    "source_kind": "bundled",
                    "enabled": False,
                }
            ]
        }
    )

    roots_disabled = _fn("_sdk_content_roots")()
    assert core_root in roots_disabled
    assert configured_root in roots_disabled
    assert bundled_root not in roots_disabled

    _fn("_write_installed_packs_lock")(
        {
            "packs": [
                {
                    "package_name": "lisa-pack-cloudflare",
                    "source_kind": "bundled",
                    "enabled": True,
                }
            ]
        }
    )
    roots_enabled = _fn("_sdk_content_roots")()
    assert bundled_root in roots_enabled


@pytest.mark.asyncio
async def test_run_pack_install_autoselects_memory_provider_when_unset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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
                "        Authorization: \"Bearer ${MEM0_API_KEY}\"",
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

    monkeypatch.setitem(_state(), "data_dir", tmp_path)
    monkeypatch.setattr(
        little_lisa,
        "_validate_pack_spec_for_install",
        lambda spec: (True, ""),
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)
    monkeypatch.setattr(little_lisa, "discover_installed_content_roots", lambda: [mem0_root])
    monkeypatch.setattr(little_lisa, "_always_on_pack_names", lambda _path=None: {"lisa-pack-core"})

    ok, _details = await _fn("_run_pack_install")("lisa-pack-mem0==0.1.0")

    assert ok is True
    lock = _fn("_load_installed_packs_lock")()
    assert lock.get("selected_capability_providers", {}).get("memory") == "lisa-pack-mem0"


def test_compile_mcp_configs_prefers_runtime_installed_memory_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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
                "      command: \"${LISA_MCP_COMMAND:lisa-memory}\"",
                "      args: []",
                "      tool_names:",
                "        add: add_memory",
                "        search: search_memory",
                "        list: list_memories",
                "servers:",
                "  - server_id: lisa-memory",
                "    runtime_kind: venv_bin",
                "    entry_point: lisa-memory",
                "    venv_dir: \"${FRAMEWORK_ROOT}/.venv\"",
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
                "        Authorization: \"Bearer ${MEM0_API_KEY}\"",
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

    monkeypatch.setitem(_state(), "data_dir", tmp_path)
    _fn("_write_installed_packs_lock")(
        {
            "packs": [
                {
                    "package_name": "lisa-pack-mem0",
                    "source_kind": "package",
                    "enabled": True,
                }
            ]
        }
    )
    monkeypatch.setattr(little_lisa, "_sdk_content_roots", lambda: [learning_root, mem0_root])
    monkeypatch.setattr(little_lisa, "_load_pack_server_exclusions", lambda _path: set())
    monkeypatch.setattr(little_lisa, "_load_capability_provider_preferences", lambda _path: {})

    _warm, _cold = _fn("_compile_mcp_configs")()

    assert _state()["memory_config_pack_slug"] == "mem0-pack"


def test_compile_mcp_configs_honors_explicit_memory_provider_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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
                "      command: \"${LISA_MCP_COMMAND:lisa-memory}\"",
                "      args: []",
                "      tool_names:",
                "        add: add_memory",
                "        search: search_memory",
                "        list: list_memories",
                "servers:",
                "  - server_id: lisa-memory",
                "    runtime_kind: venv_bin",
                "    entry_point: lisa-memory",
                "    venv_dir: \"${FRAMEWORK_ROOT}/.venv\"",
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
                "        Authorization: \"Bearer ${MEM0_API_KEY}\"",
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

    monkeypatch.setitem(_state(), "data_dir", tmp_path)
    _fn("_write_installed_packs_lock")(
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
        }
    )
    monkeypatch.setattr(little_lisa, "_sdk_content_roots", lambda: [learning_root, mem0_root])
    monkeypatch.setattr(little_lisa, "_load_pack_server_exclusions", lambda _path: set())
    monkeypatch.setattr(little_lisa, "_load_capability_provider_preferences", lambda _path: {})

    _warm, _cold = _fn("_compile_mcp_configs")()

    assert _state()["memory_config_pack_slug"] == "learning-pack"
