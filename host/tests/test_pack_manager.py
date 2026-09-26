"""Tests for pack management functions.

Migrated from ``hosts/little_lisa/test_pack_manager.py``.

The pack management functions were re-homed from the
``hosts/little_lisa/main.py`` monolith into two modules:

- ``host.commands`` — pack install/uninstall, validation, enable/disable
- ``host.content_roots`` — installed pack lock load/write, sdk_content_roots
- ``host.mcp_pool`` — compile_mcp_configs, load_pack_server_exclusions

The ``_RUNTIME_STATE`` dict is gone. Tests construct a ``HostRuntime``
from a ``HostConfig`` (with ``data_dir`` set to ``tmp_path``) and pass
it to the runtime-aware functions.

Functions that were module-level globals in ``main.py`` (e.g.
``_sdk_content_roots``, ``_load_pack_server_exclusions``) are now
module-level functions in ``host.content_roots`` and
``host.mcp_pool`` respectively. Monkeypatch them at their new
import sites.
"""

from __future__ import annotations

# pylint: disable=protected-access
import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
from host.config import HostConfig
from host.paths import framework_seed_requirements
from host.runtime import HostRuntime
from kernel.session_plan import compile_session_plan

from host import commands, content_roots, mcp_pool


def _make_runtime(tmp_path: Path) -> HostRuntime:
    """Build a HostRuntime whose data_dir is tmp_path."""
    from host.testing import build_test_runtime

    return build_test_runtime(tmp_path, models_yaml="")


def _make_runtime_with_memory_provider(
    tmp_path: Path,
    provider_id: str,
) -> HostRuntime:
    config = HostConfig.from_env(data_dir=str(tmp_path))
    plan = compile_session_plan(
        text=(
            "---\n"
            "schema_version: 2\n"
            "name: pack-manager-test\n"
            "surface: { type: personal_dm, id: host-console, memory_scope: testing }\n"
            "providers:\n"
            "  capabilities:\n"
            f"    memory: {provider_id}\n"
            "---\n"
            "# Session\n"
        ),
        source_uri="workspace:pack-manager-test.md",
    )
    return HostRuntime(config, session_plan=plan)


def _load_lock(runtime: HostRuntime) -> dict[str, Any]:
    return content_roots.load_installed_packs_lock(runtime.config.data_dir)


def _write_lock(runtime: HostRuntime, payload: dict[str, Any]) -> None:
    path = content_roots.runtime_pack_lock_path(runtime.config.data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


@pytest.mark.asyncio
async def test_run_pack_install_rejected_by_policy_does_not_mutate_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(tmp_path)
    monkeypatch.setattr(
        commands._packs,
        "_validate_pack_spec_for_install",
        lambda spec: (False, f"rejected:{spec}"),
    )

    runtime.session.restart_required = False
    runtime.session.restart_reason = ""

    ok, message = await commands.run_pack_install(runtime, "requests")

    assert ok is False
    assert message.startswith("rejected:")
    assert runtime.session.restart_required is False
    assert runtime.session.restart_reason == ""


@pytest.mark.asyncio
async def test_run_pack_install_success_records_lock_and_restart(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(tmp_path)

    class _FakeCompleted:
        returncode = 0
        stdout = b"installed"

    monkeypatch.setattr(
        commands._packs,
        "_validate_pack_spec_for_install",
        lambda spec: (True, ""),
    )
    import subprocess as _subprocess

    monkeypatch.setattr(_subprocess, "run", lambda *a, **kw: _FakeCompleted())
    runtime.session.restart_required = False
    runtime.session.restart_reason = ""

    ok, details = await commands.run_pack_install(runtime, "lisa-pack-cloudflare==0.1.0")

    assert ok is True
    assert "Installed 'lisa-pack-cloudflare'" in details
    assert runtime.session.restart_required is True
    assert runtime.session.restart_reason == "pack install: lisa-pack-cloudflare"

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

    class _FakeCompleted:
        returncode = 0
        stdout = b"installed"

    local_pack = tmp_path / "lisa-pack-memory"
    local_pack.mkdir(parents=True, exist_ok=True)
    (local_pack / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "lisa-pack-memory"',
                'version = "0.1.0"',
                'requires-python = ">=3.11"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        commands._packs,
        "_validate_pack_spec_for_install",
        lambda spec: (True, ""),
    )
    import subprocess as _subprocess

    monkeypatch.setattr(_subprocess, "run", lambda *a, **kw: _FakeCompleted())
    runtime.session.restart_required = False
    runtime.session.restart_reason = ""

    ok, details = await commands.run_pack_install(runtime, str(local_pack))

    assert ok is True
    assert "Installed 'lisa-pack-memory' from local source" in details
    assert runtime.session.restart_required is True
    assert runtime.session.restart_reason == "pack install: lisa-pack-memory"

    lock = _load_lock(runtime)
    assert isinstance(lock.get("packs"), list)
    assert any(
        str(item.get("package_name")) == "lisa-pack-memory"
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
    runtime.session.restart_required = False
    runtime.session.restart_reason = ""

    ok, details = await commands.run_pack_uninstall(runtime, "lisa-pack-cloudflare")

    assert ok is True
    assert "Uninstalled 'lisa-pack-cloudflare'" in details
    assert runtime.session.restart_required is True
    assert runtime.session.restart_reason == "pack uninstall: lisa-pack-cloudflare"

    lock = _load_lock(runtime)
    assert lock["packs"] == []


# ---------------------------------------------------------------------------
# /pack update — in-place upgrade from recorded install source
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_pack_update_reinstalls_from_recorded_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A runtime-installed pack is re-installed from its recorded source_ref."""
    runtime = _make_runtime(tmp_path)
    seen_specs: list[str] = []

    class _FakeCompleted:
        returncode = 0
        stdout = b"installed"

    def _fake_install_sync(rt: HostRuntime, spec: str, **kwargs: object) -> tuple[bool, str]:
        _ = rt
        seen_specs.append(spec)
        return True, (
            "Installed 'lisa-pack-cloudflare' from git source. "
            "Restart required to reload entry points, pack catalogs, and MCP manifests."
        )

    _write_lock(
        runtime,
        {
            "packs": [
                {
                    "package_name": "lisa-pack-cloudflare",
                    "source_kind": "git",
                    "source_ref": (
                        "git+https://github.com/acme/lisa-pack-cloudflare.git@v0.1.0"
                        "#egg=lisa-pack-cloudflare"
                    ),
                    "enabled": True,
                }
            ]
        },
    )
    monkeypatch.setattr(commands._packs, "run_pack_install_sync", _fake_install_sync)
    runtime.session.restart_required = False
    runtime.session.restart_reason = ""

    ok, details = await commands.run_pack_update(runtime, "cloudflare")

    assert ok is True
    assert len(seen_specs) == 1
    assert seen_specs[0].startswith("git+https://github.com/acme/lisa-pack-cloudflare.git@v0.1.0")
    assert "Updated 'lisa-pack-cloudflare'" in details
    assert runtime.session.restart_required is True
    assert runtime.session.restart_reason == "pack update: lisa-pack-cloudflare"


@pytest.mark.asyncio
async def test_run_pack_update_rejects_unknown_pack(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An unknown pack name fails with the /pack list hint and no restart."""
    runtime = _make_runtime(tmp_path)
    monkeypatch.setattr(
        commands._packs,
        "run_pack_install_sync",
        lambda *a, **kw: pytest.fail("install core must not run for unknown packs"),
    )
    runtime.session.restart_required = False
    runtime.session.restart_reason = ""

    ok, details = await commands.run_pack_update(runtime, "lisa-pack-nowhere")

    assert ok is False
    assert "Unknown pack" in details
    assert "/pack list" in details
    assert runtime.session.restart_required is False


@pytest.mark.asyncio
async def test_run_pack_update_rejects_bundled_pack(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Bundled packs have no install source; update explains the framework path."""
    runtime = _make_runtime(tmp_path)
    _write_lock(
        runtime,
        {
            "packs": [
                {
                    "package_name": "lisa-pack-core",
                    "source_kind": "bundled",
                    "source_ref": "bundled",
                    "enabled": True,
                }
            ]
        },
    )
    monkeypatch.setattr(
        commands._packs,
        "run_pack_install_sync",
        lambda *a, **kw: pytest.fail("install core must not run for bundled packs"),
    )
    runtime.session.restart_required = False
    runtime.session.restart_reason = ""

    ok, details = await commands.run_pack_update(runtime, "lisa-pack-core")

    assert ok is False
    assert "bundled" in details
    assert runtime.session.restart_required is False


@pytest.mark.asyncio
async def test_run_pack_update_propagates_install_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An install-core failure surfaces its message and requests no restart."""
    runtime = _make_runtime(tmp_path)
    _write_lock(
        runtime,
        {
            "packs": [
                {
                    "package_name": "lisa-pack-cloudflare",
                    "source_kind": "package",
                    "source_ref": "lisa-pack-cloudflare==0.1.0",
                    "enabled": True,
                }
            ]
        },
    )
    monkeypatch.setattr(
        commands._packs,
        "run_pack_install_sync",
        lambda *a, **kw: (False, "Install failed for 'lisa-pack-cloudflare==0.1.0'.\nboom"),
    )
    runtime.session.restart_required = False
    runtime.session.restart_reason = ""

    ok, details = await commands.run_pack_update(runtime, "lisa-pack-cloudflare")

    assert ok is False
    assert "Install failed" in details
    assert runtime.session.restart_required is False


@pytest.mark.asyncio
async def test_run_pack_update_all_refreshes_runtime_packs_and_skips_bundled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """update --all re-installs every runtime entry and skips bundled packs."""
    runtime = _make_runtime(tmp_path)
    seen_specs: list[str] = []

    def _fake_install_sync(rt: HostRuntime, spec: str, **kwargs: object) -> tuple[bool, str]:
        _ = rt
        seen_specs.append(spec)
        assert kwargs.get("request_restart") is False
        return True, f"Installed '{spec}'. Restart required."

    _write_lock(
        runtime,
        {
            "packs": [
                {
                    "package_name": "lisa-pack-core",
                    "source_kind": "bundled",
                    "source_ref": "bundled",
                },
                {
                    "package_name": "lisa-pack-cloudflare",
                    "source_kind": "git",
                    "source_ref": "git+https://github.com/acme/lisa-pack-cloudflare.git@v0.2.0",
                },
                {
                    "package_name": "lisa-pack-mem0",
                    "source_kind": "package",
                    "source_ref": "lisa-pack-mem0==0.3.1",
                },
                {
                    "package_name": "lisa-pack-no-source",
                    "source_kind": "package",
                    "source_ref": "",
                },
            ]
        },
    )
    monkeypatch.setattr(commands._packs, "run_pack_install_sync", _fake_install_sync)
    runtime.session.restart_required = False
    runtime.session.restart_reason = ""

    ok, details = await commands.run_pack_update_all(runtime)

    assert ok is True
    assert sorted(seen_specs) == [
        "git+https://github.com/acme/lisa-pack-cloudflare.git@v0.2.0",
        "lisa-pack-mem0==0.3.1",
    ]
    assert "Updated lisa-pack-cloudflare, lisa-pack-mem0" in details
    assert runtime.session.restart_required is True
    assert runtime.session.restart_reason == "pack update-all: 2 pack(s)"


@pytest.mark.asyncio
async def test_run_pack_update_all_without_runtime_packs_is_noop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """No updatable entries → friendly message, no install, no restart."""
    runtime = _make_runtime(tmp_path)
    _write_lock(
        runtime,
        {
            "packs": [
                {
                    "package_name": "lisa-pack-core",
                    "source_kind": "bundled",
                    "source_ref": "bundled",
                }
            ]
        },
    )
    monkeypatch.setattr(
        commands._packs,
        "run_pack_install_sync",
        lambda *a, **kw: pytest.fail("install core must not run with nothing to update"),
    )
    runtime.session.restart_required = False
    runtime.session.restart_reason = ""

    ok, details = await commands.run_pack_update_all(runtime)

    assert ok is True
    assert "No runtime-installed packs" in details
    assert runtime.session.restart_required is False


@pytest.mark.asyncio
async def test_run_pack_update_all_reports_partial_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """One failing pack does not stop the batch; failures are listed per pack."""
    runtime = _make_runtime(tmp_path)

    def _fake_install_sync(rt: HostRuntime, spec: str, **kwargs: object) -> tuple[bool, str]:
        _ = rt
        if spec == "lisa-pack-mem0==0.3.1":
            return False, "Install failed for 'lisa-pack-mem0==0.3.1'.\nresolver boom"
        return True, f"Installed '{spec}'."

    _write_lock(
        runtime,
        {
            "packs": [
                {
                    "package_name": "lisa-pack-cloudflare",
                    "source_kind": "package",
                    "source_ref": "lisa-pack-cloudflare==0.2.0",
                },
                {
                    "package_name": "lisa-pack-mem0",
                    "source_kind": "package",
                    "source_ref": "lisa-pack-mem0==0.3.1",
                },
            ]
        },
    )
    monkeypatch.setattr(commands._packs, "run_pack_install_sync", _fake_install_sync)
    runtime.session.restart_required = False
    runtime.session.restart_reason = ""

    ok, details = await commands.run_pack_update_all(runtime)

    assert ok is True
    assert "Updated lisa-pack-cloudflare" in details
    assert "Update failed for 'lisa-pack-mem0'" in details
    assert "resolver boom" in details
    assert runtime.session.restart_required is True
    assert runtime.session.restart_reason == "pack update-all: 1 pack(s)"


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
    local_pack = tmp_path / "lisa-pack-memory"
    local_pack.mkdir(parents=True, exist_ok=True)
    (local_pack / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "lisa-pack-memory"',
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
    local_pack = tmp_path / "lisa-pack-memory"
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


def test_normalize_pack_install_spec_expands_bare_pack_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bare ``lisa-pack-*`` names expand to the framework git root template."""
    monkeypatch.delenv("LISA_PACK_GIT_ROOT", raising=False)

    result = commands._normalize_pack_install_spec("lisa-pack-brave")
    assert result == "git+https://github.com/Lisa-Harness/lisa-pack-brave@main"


def test_normalize_pack_install_spec_bare_name_respects_custom_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``LISA_PACK_GIT_ROOT`` overrides the expansion template."""
    monkeypatch.setenv(
        "LISA_PACK_GIT_ROOT",
        "git+https://gitlab.example.com/lisa-mirror/{pack}@main",
    )

    result = commands._normalize_pack_install_spec("lisa-pack-brave")
    assert result == "git+https://gitlab.example.com/lisa-mirror/lisa-pack-brave@main"


def test_normalize_pack_install_spec_bare_name_disabled_when_root_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setting ``LISA_PACK_GIT_ROOT`` to empty disables bare-name expansion."""
    monkeypatch.setenv("LISA_PACK_GIT_ROOT", "")

    result = commands._normalize_pack_install_spec("lisa-pack-brave")
    assert result == "lisa-pack-brave"


def test_normalize_pack_install_spec_bare_name_preserves_version_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Specs with version pins or extras are not expanded (not bare names)."""
    monkeypatch.delenv("LISA_PACK_GIT_ROOT", raising=False)

    assert (
        commands._normalize_pack_install_spec("lisa-pack-cloudflare==0.1.0")
        == "lisa-pack-cloudflare==0.1.0"
    )
    assert commands._normalize_pack_install_spec("lisa-pack-core[extra]") == "lisa-pack-core[extra]"


def test_normalize_pack_install_spec_git_and_local_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """git+ specs and local paths are returned as-is (not bare names)."""
    monkeypatch.delenv("LISA_PACK_GIT_ROOT", raising=False)

    git_spec = "git+https://github.com/acme/lisa-pack-cloudflare.git@v1.2.3"
    assert commands._normalize_pack_install_spec(git_spec) == git_spec

    local_pack = tmp_path / "lisa-pack-memory"
    local_pack.mkdir(parents=True, exist_ok=True)
    assert commands._normalize_pack_install_spec(str(local_pack)) == str(local_pack)


def test_validate_pack_spec_for_install_expanded_bare_name_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare name expanded to the default framework git root passes validation."""
    monkeypatch.delenv("LISA_PACK_GIT_ROOT", raising=False)
    monkeypatch.setenv("LISA_PACK_GIT_ALLOW_HOSTS", "github.com")
    monkeypatch.setenv("LISA_PACK_REQUIRE_GIT_REF", "true")

    expanded = commands._normalize_pack_install_spec("lisa-pack-brave")
    ok, reason = commands._validate_pack_spec_for_install(expanded)
    assert ok is True
    assert reason == ""


@pytest.mark.asyncio
async def test_ensure_packs_venv_creates_venv_without_seeding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """_ensure_packs_venv creates the venv but does NOT seed it (caller owns seeding)."""
    packs_venv = tmp_path / "packs venv"
    # Simulate that pyvenv.cfg does NOT exist → venv will be created.
    assert not (packs_venv / "pyvenv.cfg").is_file()

    calls: list[list[str]] = []

    class _FakeCompleted:
        returncode = 0
        stdout = b"ok"

    def _fake_run(args: list[str], **kwargs: object) -> _FakeCompleted:
        calls.append([str(a) for a in args])
        return _FakeCompleted()

    import subprocess as _subprocess

    monkeypatch.setattr(_subprocess, "run", _fake_run)

    ok, err = await commands._ensure_packs_venv(packs_venv)
    assert ok is True
    assert err == ""

    # Exactly one call: uv venv <path>. Seeding is now the caller's job.
    assert len(calls) == 1
    assert calls[0][:3] == ["uv", "venv", str(packs_venv)]


@pytest.mark.asyncio
async def test_seed_packs_venv_seeds_framework_and_bundled_packs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """_seed_packs_venv_with_framework seeds framework members AND bundled packs."""
    packs_venv = tmp_path / "packs venv"
    packs_venv.mkdir(parents=True, exist_ok=True)
    (packs_venv / "pyvenv.cfg").write_text("home = /fake\n", encoding="utf-8")

    calls: list[list[str]] = []

    def _fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        calls.append([str(a) for a in args])
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=b"ok")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    def _fake_seed_requirements() -> list[str]:
        """Repo-mode style requirements: editable local member dirs."""
        fw = Path(commands._FRAMEWORK_ROOT)
        reqs = [f"-e {fw / member}" for member in ("kernel", "sdk", "mcp_server")]
        reqs.extend(f"-e {fw / 'sdk' / 'packs' / name}" for name in ("core", "recovery"))
        return reqs

    monkeypatch.setattr(commands._packs, "framework_seed_requirements", _fake_seed_requirements)

    ok, err = await commands._seed_packs_venv_with_framework(packs_venv)
    assert ok is True
    assert err == ""

    # Exactly one call: uv pip install --python <path> -e <member> ...
    assert len(calls) == 1
    assert calls[0][:5] == ["uv", "pip", "install", "--python", str(packs_venv)]
    assert "-e" in calls[0]
    e_indices = [i for i, arg in enumerate(calls[0]) if arg == "-e"]
    targets = [calls[0][idx + 1] for idx in e_indices]
    # Framework members are present.
    assert any("kernel" in t for t in targets)
    assert any("sdk" in t for t in targets)
    assert any("mcp_server" in t for t in targets)
    # Bundled packs are present.
    bundled_names = {"core", "recovery"}
    for name in bundled_names:
        assert any(name in t for t in targets), f"bundled pack {name!r} missing from seed targets"


@pytest.mark.asyncio
async def test_seed_packs_venv_tool_mode_uses_direct_url_requirements(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Tool-mode seeding installs commit-pinned git requirements, not -e paths.

    Regression for the clean-box failure: ``uv tool install`` from git gives
    the venv no framework source tree, so the seed list must come from PEP
    610 direct_url metadata (commit-pinned ``git+`` requirements) instead of
    local directories. An empty repo-mode-style list must not silently
    no-op the seeding.
    """
    packs_venv = tmp_path / "packs venv"
    packs_venv.mkdir(parents=True, exist_ok=True)
    (packs_venv / "pyvenv.cfg").write_text("home = /fake\n", encoding="utf-8")

    calls: list[list[str]] = []

    def _fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        calls.append([str(a) for a in args])
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=b"ok")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    def _fake_seed_requirements() -> list[str]:
        """Tool-mode style: pinned git direct requirements (no leading -e)."""
        return [
            "lisa-harness-kernel @ git+https://github.com/Lisa-Harness/lisa.git@abc123#subdirectory=kernel",
            "lisa-harness-sdk @ git+https://github.com/Lisa-Harness/lisa.git@abc123#subdirectory=sdk",
        ]

    monkeypatch.setattr(commands._packs, "framework_seed_requirements", _fake_seed_requirements)

    ok, err = await commands._seed_packs_venv_with_framework(packs_venv)
    assert ok is True
    assert err == ""

    assert len(calls) == 1
    args = calls[0]
    assert args[:5] == ["uv", "pip", "install", "--python", str(packs_venv)]
    # Git direct requirements pass through verbatim — no -e prefix.
    assert "-e" not in args
    assert (
        "lisa-harness-kernel @ git+https://github.com/Lisa-Harness/lisa.git@abc123#subdirectory=kernel"
        in args
    )
    assert (
        "lisa-harness-sdk @ git+https://github.com/Lisa-Harness/lisa.git@abc123#subdirectory=sdk"
        in args
    )


@pytest.mark.asyncio
async def test_seed_packs_venv_noop_when_no_requirements(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """No seed requirements (nothing installable) skips the uv call entirely."""
    packs_venv = tmp_path / "packs venv"
    packs_venv.mkdir(parents=True, exist_ok=True)
    (packs_venv / "pyvenv.cfg").write_text("home = /fake\n", encoding="utf-8")

    calls: list[list[str]] = []

    def _fail_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        calls.append([str(a) for a in args])
        raise AssertionError("uv must not be invoked when there is nothing to seed")

    monkeypatch.setattr(subprocess, "run", _fail_run)
    monkeypatch.setattr(commands._packs, "framework_seed_requirements", list)

    ok, err = await commands._seed_packs_venv_with_framework(packs_venv)
    assert ok is True
    assert err == ""
    assert calls == []


@pytest.mark.asyncio
async def test_framework_seed_requirements_repo_mode_covers_bundled_packs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repo-mode requirements cover kernel/sdk/mcp_server plus bundled packs."""
    monkeypatch.setattr("host.paths.is_tool_install", lambda: False, raising=False)
    reqs = framework_seed_requirements()
    assert reqs, "repo checkout must produce seed requirements"
    joined = "\n".join(reqs)
    for member in ("kernel", "sdk", "mcp_server"):
        assert member in joined
    # Bundled packs live under sdk/packs/ with a pyproject.toml each.
    assert "recovery" in joined


@pytest.mark.asyncio
async def test_ensure_packs_venv_skips_creation_when_already_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An existing valid packs venv (pyvenv.cfg present) skips creation."""
    packs_venv = tmp_path / "packs venv"
    packs_venv.mkdir(parents=True, exist_ok=True)
    (packs_venv / "pyvenv.cfg").write_text("home = /fake\n", encoding="utf-8")

    call_count = 0

    class _FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return (b"ok", b"")

    async def _fake_create_subprocess_exec(*args: object, **kwargs: object) -> _FakeProc:
        nonlocal call_count
        call_count += 1
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)

    ok, err = await commands._ensure_packs_venv(packs_venv)
    assert ok is True
    assert err == ""
    # No subprocess calls — venv already exists.
    assert call_count == 0


def test_request_and_consume_restart_resume_message(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Restart request + resume-message env-var round-trip.

    In the new structure, ``runtime.session.request_restart(reason)`` sets the
    restart flag/reason, and the resume message is stored in the
    ``LISA_RESTART_RESUME_MESSAGE`` env var (read by the host on next
    launch). This test exercises the env-var half directly.
    """
    monkeypatch.delenv("LISA_RESTART_RESUME_MESSAGE", raising=False)
    runtime = _make_runtime(tmp_path)
    runtime.session.restart_required = False
    runtime.session.restart_reason = ""

    runtime.session.request_restart(reason="pack install: lisa-pack-cloudflare")
    monkeypatch.setenv("LISA_RESTART_RESUME_MESSAGE", "check cloudflare status")

    assert runtime.session.restart_required is True
    assert runtime.session.restart_reason == "pack install: lisa-pack-cloudflare"
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


@pytest.mark.parametrize(
    ("exclusions", "expected_present", "expected_absent"),
    [
        pytest.param(
            set(),
            {"pack-a.shared-server", "pack-b.shared-server"},
            set(),
            id="namespaced",
        ),
        pytest.param(
            {"pack-a.shared-server"},
            {"pack-b.shared-server"},
            {"pack-a.shared-server"},
            id="excluded",
        ),
    ],
)
def test_compile_mcp_configs_namespaces_and_excludes_pack_servers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    exclusions: set[str],
    expected_present: set[str],
    expected_absent: set[str],
) -> None:
    """Pack servers are namespaced ``<pack>.<server>``; exclusions drop exactly the named one."""
    runtime = _make_runtime(tmp_path)
    pack_a_root = tmp_path / "pack-a"
    pack_b_root = tmp_path / "pack-b"
    pack_a_root.mkdir(parents=True, exist_ok=True)
    pack_b_root.mkdir(parents=True, exist_ok=True)

    pack_manifest_text = (
        "\n".join(
            [
                "version: 1",
                "servers:",
                "  - server_id: shared-server",
                "    runtime_kind: http",
                "    url: http://127.0.0.1:9000/mcp",
            ]
        )
        + "\n"
    )
    (pack_a_root / "mcp_servers.yaml").write_text(pack_manifest_text, encoding="utf-8")
    (pack_b_root / "mcp_servers.yaml").write_text(pack_manifest_text, encoding="utf-8")

    monkeypatch.setattr(mcp_pool, "sdk_content_roots", lambda *a, **kw: [pack_a_root, pack_b_root])
    monkeypatch.setattr(mcp_pool, "load_pack_server_exclusions", lambda _path: exclusions)

    warm, cold = mcp_pool.compile_mcp_configs(runtime)
    server_ids = {config.server_id for config in [*warm, *cold]}

    assert expected_present <= server_ids
    assert not expected_absent & server_ids


def test_compile_proxy_configs_honors_session_include_and_required(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(tmp_path)
    runtime.session_plan = compile_session_plan(
        text=(
            "---\n"
            "schema_version: 2\n"
            "name: proxy-selection\n"
            "surface:\n"
            "  type: personal_dm\n"
            "  id: host-console\n"
            "  memory_scope: testing\n"
            "resources:\n"
            "  proxies:\n"
            "    include: [pack-a.selected]\n"
            "    required: [pack-b.required]\n"
            "---\n"
            "# Proxy selection\n"
        ),
        source_uri="workspace:session_context.md",
    )
    pack_a_root = tmp_path / "pack-a"
    pack_b_root = tmp_path / "pack-b"
    pack_a_root.mkdir()
    pack_b_root.mkdir()
    for root, proxy_id in (
        (pack_a_root, "selected"),
        (pack_a_root, "not-selected"),
        (pack_b_root, "required"),
    ):
        manifest = root / "mcp_servers.yaml"
        if manifest.exists():
            existing = manifest.read_text(encoding="utf-8")
        else:
            existing = "version: 1\nproxies:\n"
        port = (
            "18787"
            if proxy_id == "selected"
            else "18788"
            if proxy_id == "not-selected"
            else "18789"
        )
        manifest.write_text(
            existing
            + "  - proxy_id: "
            + proxy_id
            + "\n    runtime_kind: binary\n    command_path: /bin/true\n"
            + "    protocol: openai\n    providers: [ollama]\n    port: "
            + port
            + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        mcp_pool,
        "sdk_content_roots",
        lambda *args, **kwargs: [pack_a_root, pack_b_root],
    )

    configs = mcp_pool.compile_proxy_configs(runtime)

    configs_by_id = {config.proxy_id: config for config in configs}
    assert set(configs_by_id) == {"pack-a.selected", "pack-b.required"}
    assert configs_by_id["pack-b.required"].required is True


def test_compile_proxy_configs_resolves_provider_overlay_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(tmp_path)
    pack_root = tmp_path / "pack-a"
    pack_root.mkdir()
    (pack_root / "mcp_servers.yaml").write_text(
        "\n".join(
            [
                "version: 1",
                "proxies:",
                "  - proxy_id: layered",
                "    runtime_kind: binary",
                "    command_path: /bin/true",
                "    protocol: openai",
                "    providers: [ollama]",
                "    port: 18787",
                "    provider_overrides:",
                "      ollama:",
                "        cli_args: [--ollama-switch]",
                "        env:",
                "          OPENAI_TARGET_API_URL:",
                "            value_from: direct_base_url",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(mcp_pool, "sdk_content_roots", lambda *args, **kwargs: [pack_root])
    monkeypatch.setenv("LISA_OLLAMA_HOST", "127.0.0.1:11434")

    configs = mcp_pool.compile_proxy_configs(runtime)

    assert len(configs) == 1
    config = configs[0]
    assert config.proxy_id == "pack-a.layered"
    assert config.args[-1] == "--ollama-switch"
    assert config.env["OPENAI_TARGET_API_URL"] == "http://127.0.0.1:11434/v1/"


def test_compile_mcp_configs_namespaces_pack_memory_servers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(tmp_path)
    pack_root = tmp_path / "memory"
    pack_root.mkdir(parents=True, exist_ok=True)

    pack_manifest_text = (
        "\n".join(
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
                "    url: http://127.0.0.1:8765/mcp",
                "    headers:",
                '      Authorization: "Bearer ${MEM0_API_KEY}"',
            ]
        )
        + "\n"
    )
    (pack_root / "mcp_servers.yaml").write_text(pack_manifest_text, encoding="utf-8")

    monkeypatch.setattr(mcp_pool, "sdk_content_roots", lambda *a, **kw: [pack_root])
    monkeypatch.setattr(mcp_pool, "load_pack_server_exclusions", lambda _path: set())

    warm, cold = mcp_pool.compile_mcp_configs(runtime)
    server_ids = {config.server_id for config in [*warm, *cold]}

    assert "memory.lisa-memory" in server_ids
    assert "memory.mem0" in server_ids
    assert runtime.services.memory_config_pack_slug == "memory"


def test_compile_mcp_configs_merges_user_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """User manifest servers are compiled alongside framework servers."""
    runtime = _make_runtime(tmp_path)
    user_manifest = tmp_path / "config" / "mcp_servers.yaml"
    user_manifest.parent.mkdir(parents=True, exist_ok=True)
    user_manifest.write_text(
        "\n".join(
            [
                "version: 1",
                "servers:",
                "  - server_id: my-search",
                "    runtime_kind: http",
                "    url: http://127.0.0.1:9000/mcp",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(mcp_pool, "sdk_content_roots", lambda *a, **kw: [])
    monkeypatch.setattr(mcp_pool, "load_pack_server_exclusions", lambda _path: set())

    warm, cold = mcp_pool.compile_mcp_configs(runtime)
    server_ids = {config.server_id for config in [*warm, *cold]}

    # User server appears with its bare id.
    assert "my-search" in server_ids


def test_compile_mcp_configs_user_manifest_collision_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A user manifest server_id colliding with a framework server raises."""
    runtime = _make_runtime(tmp_path)
    user_manifest = tmp_path / "config" / "mcp_servers.yaml"
    user_manifest.parent.mkdir(parents=True, exist_ok=True)
    user_manifest.write_text(
        "\n".join(
            [
                "version: 1",
                "servers:",
                "  - server_id: lisa-storage",
                "    runtime_kind: http",
                "    url: http://127.0.0.1:9000/mcp",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(mcp_pool, "sdk_content_roots", lambda *a, **kw: [])
    monkeypatch.setattr(mcp_pool, "load_pack_server_exclusions", lambda _path: set())

    # The framework manifest declares lisa-storage; the user manifest
    # redeclares it → MCPServerRegistry.register raises ValueError.
    with pytest.raises(ValueError, match="already registered"):
        mcp_pool.compile_mcp_configs(runtime)


def test_set_pack_enabled_rejects_disabling_always_on_pack(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(tmp_path)
    monkeypatch.setattr(
        commands._packs, "always_on_pack_names", lambda _path=None, **kw: {"lisa-pack-core"}
    )

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

    runtime.session.restart_required = False
    runtime.session.restart_reason = ""
    monkeypatch.setattr(
        commands._packs, "always_on_pack_names", lambda _path=None, **kw: {"lisa-pack-core"}
    )
    monkeypatch.setattr(commands._packs, "discover_installed_content_roots", lambda: [bundled_root])

    ok, details = commands.set_pack_enabled(runtime, "cloudflare", enabled=True)

    assert ok is True
    assert "Enabled 'lisa-pack-cloudflare'" in details
    assert runtime.session.restart_required is True
    assert runtime.session.restart_reason == "pack enable: lisa-pack-cloudflare"

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

    monkeypatch.setattr(
        content_roots, "always_on_pack_names", lambda _path=None, **kw: {"lisa-pack-core"}
    )
    monkeypatch.setattr(
        "kernel.content_roots.discover_installed_content_roots",
        lambda: [core_root, bundled_root],
    )
    monkeypatch.setattr(
        "kernel.content_roots.discover_packs_venv_content_roots",
        lambda *args, **kwargs: [],
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

    class _FakeCompleted:
        returncode = 0
        stdout = b"installed"

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
                "    url: http://127.0.0.1:8765/mcp",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        commands._packs,
        "_validate_pack_spec_for_install",
        lambda spec: (True, ""),
    )
    import subprocess as _subprocess

    monkeypatch.setattr(_subprocess, "run", lambda *a, **kw: _FakeCompleted())
    monkeypatch.setattr(commands._packs, "discover_installed_content_roots", lambda: [mem0_root])
    monkeypatch.setattr(
        commands._packs, "always_on_pack_names", lambda _path=None, **kw: {"lisa-pack-core"}
    )

    ok, _details = await commands.run_pack_install(runtime, "lisa-pack-mem0==0.1.0")

    assert ok is True
    lock = _load_lock(runtime)
    assert lock.get("selected_capability_providers", {}).get("memory") == "lisa-pack-mem0"


def test_compile_mcp_configs_uses_session_plan_memory_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _make_runtime_with_memory_provider(tmp_path, "lisa-pack-mem0")
    memory_root = tmp_path / "memory-pack"
    mem0_root = tmp_path / "mem0-pack"
    memory_root.mkdir(parents=True, exist_ok=True)
    mem0_root.mkdir(parents=True, exist_ok=True)

    (memory_root / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "lisa-pack-memory"',
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

    (memory_root / "mcp_servers.yaml").write_text(
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
                "    url: http://127.0.0.1:8765/mcp",
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
    monkeypatch.setattr(mcp_pool, "sdk_content_roots", lambda *a, **kw: [memory_root, mem0_root])
    monkeypatch.setattr(mcp_pool, "load_pack_server_exclusions", lambda _path: set())

    _warm, _cold = mcp_pool.compile_mcp_configs(runtime)

    assert runtime.services.memory_config_pack_slug == "mem0-pack"


def test_compile_mcp_configs_honors_explicit_memory_provider_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _make_runtime_with_memory_provider(tmp_path, "lisa-pack-memory")
    memory_root = tmp_path / "memory-pack"
    mem0_root = tmp_path / "mem0-pack"
    memory_root.mkdir(parents=True, exist_ok=True)
    mem0_root.mkdir(parents=True, exist_ok=True)

    (memory_root / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "lisa-pack-memory"',
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

    (memory_root / "mcp_servers.yaml").write_text(
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
                "    url: http://127.0.0.1:8765/mcp",
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
                "memory": "lisa-pack-mem0",
            },
        },
    )
    monkeypatch.setattr(mcp_pool, "sdk_content_roots", lambda *a, **kw: [memory_root, mem0_root])
    monkeypatch.setattr(mcp_pool, "load_pack_server_exclusions", lambda _path: set())

    _warm, _cold = mcp_pool.compile_mcp_configs(runtime)

    assert runtime.services.memory_config_pack_slug == "memory-pack"


# ---------------------------------------------------------------------------
# Bundled pack lock seeding (lifecycle closure)
# ---------------------------------------------------------------------------


def _bundled_root(tmp_path: Path, slug: str, package_name: str) -> Path:
    """Create a fake discovered pack root with a pyproject.toml name."""
    root = tmp_path / slug
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                f'name = "{package_name}"',
                'version = "0.1.0"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def test_seed_bundled_packs_lock_first_boot_seeds_discovered_packs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """First-boot seeding writes enabled=true bundled entries for discovered packs."""
    cloudflare_root = _bundled_root(tmp_path, "lisa_pack_cloudflare", "lisa-pack-cloudflare")
    github_root = _bundled_root(tmp_path, "lisa_pack_github", "lisa-pack-github")
    monkeypatch.setattr(
        content_roots,
        "discover_installed_content_roots",
        lambda: [cloudflare_root, github_root],
    )
    monkeypatch.setattr(content_roots, "_is_bundled_pack_root", lambda _root: True)

    written = content_roots.seed_bundled_packs_lock(tmp_path)

    assert written is True
    lock = content_roots.load_installed_packs_lock(tmp_path)
    by_name = {entry["package_name"]: entry for entry in lock["packs"]}
    assert set(by_name) == {"lisa-pack-cloudflare", "lisa-pack-github"}
    for entry in lock["packs"]:
        assert entry["enabled"] is True
        assert entry["source_kind"] == "bundled"
        assert entry["source_ref"] == "bundled"
        assert entry["updated_at"]


def test_seed_bundled_packs_lock_stamps_external_packs_as_package(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A discovered pack whose root is not under sdk/packs/ is source_kind=package."""
    external_root = _bundled_root(tmp_path, "lisa_pack_search", "lisa-pack-search")
    monkeypatch.setattr(
        content_roots,
        "discover_installed_content_roots",
        lambda: [external_root],
    )
    # Do NOT monkeypatch _is_bundled_pack_root — the tmp_path root is not
    # under sdk/packs/, so the real classifier returns False.

    written = content_roots.seed_bundled_packs_lock(tmp_path)

    assert written is True
    lock = content_roots.load_installed_packs_lock(tmp_path)
    by_name = {entry["package_name"]: entry for entry in lock["packs"]}
    assert set(by_name) == {"lisa-pack-search"}
    assert by_name["lisa-pack-search"]["source_kind"] == "package"
    assert by_name["lisa-pack-search"]["source_ref"] == "package"
    assert by_name["lisa-pack-search"]["enabled"] is True


def test_seed_bundled_packs_lock_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Second seed call does not duplicate or mutate existing entries."""
    cloudflare_root = _bundled_root(tmp_path, "lisa_pack_cloudflare", "lisa-pack-cloudflare")
    monkeypatch.setattr(
        content_roots,
        "discover_installed_content_roots",
        lambda: [cloudflare_root],
    )

    assert content_roots.seed_bundled_packs_lock(tmp_path) is True
    first = content_roots.load_installed_packs_lock(tmp_path)
    first_updated_at = first["packs"][0]["updated_at"]

    assert content_roots.seed_bundled_packs_lock(tmp_path) is False
    second = content_roots.load_installed_packs_lock(tmp_path)

    assert len(second["packs"]) == 1
    assert second["packs"][0]["updated_at"] == first_updated_at


def test_seed_bundled_packs_lock_preserves_explicit_disabled_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An explicitly disabled bundled pack stays disabled across seeding."""
    cloudflare_root = _bundled_root(tmp_path, "lisa_pack_cloudflare", "lisa-pack-cloudflare")
    monkeypatch.setattr(
        content_roots,
        "discover_installed_content_roots",
        lambda: [cloudflare_root],
    )

    _write_lock(
        HostRuntime(HostConfig.from_env(data_dir=str(tmp_path))),
        {
            "packs": [
                {
                    "package_name": "lisa-pack-cloudflare",
                    "source_kind": "bundled",
                    "source_ref": "bundled",
                    "enabled": False,
                    "updated_at": "2026-07-12T00:00:00+00:00",
                }
            ]
        },
    )

    written = content_roots.seed_bundled_packs_lock(tmp_path)

    assert written is False
    lock = content_roots.load_installed_packs_lock(tmp_path)
    assert len(lock["packs"]) == 1
    assert lock["packs"][0]["enabled"] is False
    assert lock["packs"][0]["updated_at"] == "2026-07-12T00:00:00+00:00"


def test_seed_bundled_packs_lock_preserves_runtime_installed_and_provider_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Seeding adds only missing bundled packs; runtime + provider state survives."""
    github_root = _bundled_root(tmp_path, "lisa_pack_github", "lisa-pack-github")
    monkeypatch.setattr(
        content_roots,
        "discover_installed_content_roots",
        lambda: [github_root],
    )
    monkeypatch.setattr(content_roots, "_is_bundled_pack_root", lambda _root: True)

    _write_lock(
        HostRuntime(HostConfig.from_env(data_dir=str(tmp_path))),
        {
            "packs": [
                {
                    "package_name": "lisa-pack-mem0",
                    "source_kind": "package",
                    "source_ref": "mem0==0.1.0",
                    "enabled": True,
                    "installed_at": "2026-07-12T00:00:00+00:00",
                }
            ],
            "selected_capability_providers": {"memory": "lisa-pack-mem0"},
        },
    )

    written = content_roots.seed_bundled_packs_lock(tmp_path)

    assert written is True
    lock = content_roots.load_installed_packs_lock(tmp_path)
    by_name = {entry["package_name"]: entry for entry in lock["packs"]}
    assert set(by_name) == {"lisa-pack-github", "lisa-pack-mem0"}
    # Runtime-installed entry untouched
    assert by_name["lisa-pack-mem0"]["source_kind"] == "package"
    assert by_name["lisa-pack-mem0"]["source_ref"] == "mem0==0.1.0"
    # Seeded bundled entry
    assert by_name["lisa-pack-github"]["source_kind"] == "bundled"
    assert by_name["lisa-pack-github"]["enabled"] is True
    # Provider selection preserved
    assert lock["selected_capability_providers"] == {"memory": "lisa-pack-mem0"}


def test_seed_bundled_packs_lock_empty_discovery_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """No discovered packs → no write, no lock file created."""
    monkeypatch.setattr(content_roots, "discover_installed_content_roots", lambda: [])

    written = content_roots.seed_bundled_packs_lock(tmp_path)

    assert written is False
    assert not content_roots.runtime_pack_lock_path(tmp_path).exists()


def test_seed_bundled_packs_lock_excludes_lisa_sdk(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The ``lisa-harness-sdk`` umbrella package is never seeded into the lock."""
    sdk_root = _bundled_root(tmp_path, "sdk", "lisa-harness-sdk")
    cloudflare_root = _bundled_root(tmp_path, "lisa_pack_cloudflare", "lisa-pack-cloudflare")
    monkeypatch.setattr(
        content_roots,
        "discover_installed_content_roots",
        lambda: [sdk_root, cloudflare_root],
    )

    written = content_roots.seed_bundled_packs_lock(tmp_path)

    assert written is True
    lock = content_roots.load_installed_packs_lock(tmp_path)
    by_name = {entry["package_name"]: entry for entry in lock["packs"]}
    assert "lisa-harness-sdk" not in by_name
    assert set(by_name) == {"lisa-pack-cloudflare"}


def test_render_pack_list_excludes_lisa_sdk(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``lisa-harness-sdk`` never appears in the ``/pack list`` output, even if a stale
    lock entry still references it."""
    runtime = _make_runtime(tmp_path)
    sdk_root = _bundled_root(tmp_path, "sdk", "lisa-harness-sdk")
    cloudflare_root = _bundled_root(tmp_path, "lisa_pack_cloudflare", "lisa-pack-cloudflare")
    monkeypatch.setattr(
        commands._lists,
        "installed_pack_roots_by_name",
        lambda: {
            "lisa-harness-sdk": sdk_root,
            "lisa-pack-cloudflare": cloudflare_root,
        },
    )
    monkeypatch.setattr(
        commands._lists,
        "always_on_pack_names",
        lambda *args, **kwargs: set(),
    )
    _write_lock(
        runtime,
        {
            "packs": [
                {
                    "package_name": "lisa-harness-sdk",
                    "enabled": True,
                    "source_kind": "bundled",
                    "source_ref": "bundled",
                    "updated_at": "2026-07-13T00:00:00+00:00",
                },
                {
                    "package_name": "lisa-pack-cloudflare",
                    "enabled": True,
                    "source_kind": "bundled",
                    "source_ref": "bundled",
                    "updated_at": "2026-07-13T00:00:00+00:00",
                },
            ]
        },
    )

    rendered = commands.render_pack_list(runtime)

    assert "lisa-harness-sdk" not in rendered
    assert "lisa-pack-cloudflare" in rendered


# ── resolve_default_mcp_tool_ids ────────────────────────────────────────────


def test_resolve_default_mcp_tool_ids_host_and_pack_namespacing(tmp_path: Path) -> None:
    """Host servers keep bare server_id; pack servers get <pack_slug>.<server_id>.

    The ambient tool id the model sees is the namespaced form the catalog
    merge and turn-runner filter use, so the resolver emits that exact form.
    """
    host_manifest = tmp_path / "mcp_servers.yaml"
    host_manifest.write_text(
        "\n".join(
            [
                "version: 1",
                "servers:",
                "  - server_id: lisa-storage",
                "    runtime_kind: venv_bin",
                '    venv_dir: "${FRAMEWORK_ROOT}"',
                "    default_available_tools:",
                "      - read_file",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pack_root = tmp_path / "pack-memory"
    pack_root.mkdir(parents=True, exist_ok=True)
    (pack_root / "mcp_servers.yaml").write_text(
        "\n".join(
            [
                "version: 1",
                "servers:",
                "  - server_id: lisa-memory",
                "    runtime_kind: venv_bin",
                '    venv_dir: "${FRAMEWORK_ROOT}"',
                "    default_available_tools:",
                "      - search_memory",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    ids = mcp_pool.resolve_default_mcp_tool_ids(
        host_manifest_path=host_manifest,
        pack_roots=[pack_root],
    )
    assert ids == {"lisa-storage.read_file", "pack-memory.lisa-memory.search_memory"}


def test_resolve_default_mcp_tool_ids_ignores_disabled(tmp_path: Path) -> None:
    """A disabled server contributes no ambient tool ids."""
    host_manifest = tmp_path / "mcp_servers.yaml"
    host_manifest.write_text(
        "\n".join(
            [
                "version: 1",
                "servers:",
                "  - server_id: lisa-storage",
                "    runtime_kind: venv_bin",
                '    venv_dir: "${FRAMEWORK_ROOT}"',
                "    disabled: true",
                "    default_available_tools:",
                "      - read_file",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ids = mcp_pool.resolve_default_mcp_tool_ids(
        host_manifest_path=host_manifest,
        pack_roots=[],
    )
    assert ids == set()


def test_resolve_default_mcp_tool_ids_empty_when_undeclared(tmp_path: Path) -> None:
    """No default_available_tools on any server → empty set."""
    host_manifest = tmp_path / "mcp_servers.yaml"
    host_manifest.write_text(
        "\n".join(
            [
                "version: 1",
                "servers:",
                "  - server_id: lisa-storage",
                "    runtime_kind: venv_bin",
                '    venv_dir: "${FRAMEWORK_ROOT}"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ids = mcp_pool.resolve_default_mcp_tool_ids(
        host_manifest_path=host_manifest,
        pack_roots=[],
    )
    assert ids == set()


def test_resolve_default_mcp_tool_ids_missing_manifest(tmp_path: Path) -> None:
    """A missing host manifest yields an empty set (defensive)."""
    ids = mcp_pool.resolve_default_mcp_tool_ids(
        host_manifest_path=tmp_path / "nope.yaml",
        pack_roots=[],
    )
    assert ids == set()


def test_resolve_default_mcp_tool_ids_user_manifest(tmp_path: Path) -> None:
    """User manifest default_available_tools are collected with bare server_id."""
    host_manifest = tmp_path / "mcp_servers.yaml"
    host_manifest.write_text(
        "\n".join(
            [
                "version: 1",
                "servers:",
                "  - server_id: lisa-storage",
                "    runtime_kind: venv_bin",
                '    venv_dir: "${FRAMEWORK_ROOT}"',
                "    default_available_tools:",
                "      - read_file",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    user_manifest = tmp_path / "user_mcp_servers.yaml"
    user_manifest.write_text(
        "\n".join(
            [
                "version: 1",
                "servers:",
                "  - server_id: my-search",
                "    runtime_kind: npx",
                '    command: ["-y", "@some/mcp-server"]',
                "    default_available_tools:",
                "      - web_search",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    ids = mcp_pool.resolve_default_mcp_tool_ids(
        host_manifest_path=host_manifest,
        pack_roots=[],
        user_manifest_path=user_manifest,
    )
    assert ids == {"lisa-storage.read_file", "my-search.web_search"}


def test_resolve_default_mcp_tool_ids_user_manifest_absent(tmp_path: Path) -> None:
    """A None or missing user manifest produces no user ambient ids."""
    host_manifest = tmp_path / "mcp_servers.yaml"
    host_manifest.write_text(
        "\n".join(
            [
                "version: 1",
                "servers:",
                "  - server_id: lisa-storage",
                "    runtime_kind: venv_bin",
                '    venv_dir: "${FRAMEWORK_ROOT}"',
                "    default_available_tools:",
                "      - read_file",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # None user_manifest_path — same as not passing it.
    ids_none = mcp_pool.resolve_default_mcp_tool_ids(
        host_manifest_path=host_manifest,
        pack_roots=[],
        user_manifest_path=None,
    )
    assert ids_none == {"lisa-storage.read_file"}

    # Non-existent path — defensive, same result.
    ids_missing = mcp_pool.resolve_default_mcp_tool_ids(
        host_manifest_path=host_manifest,
        pack_roots=[],
        user_manifest_path=tmp_path / "nope.yaml",
    )
    assert ids_missing == {"lisa-storage.read_file"}
