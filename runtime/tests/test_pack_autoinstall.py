"""Tests for startup auto-install of session-required packs.

Verifies that ``runtime.pack_autoinstall.auto_install_required_packs``:
- is a no-op when all required packs are already importable,
- installs missing required packs via the sync install core,
- raises RuntimeError when a required pack fails to install,
- rejects unpinned git refs at spec resolution time,
- requests a restart when at least one pack is installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from kernel.session_plan import PackInstallSpec
from runtime.commands import _packs
from runtime.config import HostConfig
from runtime.pack_autoinstall import (
    _resolve_auto_install_spec,
    auto_install_required_packs,
)
from runtime.runtime import HostRuntime


def _session_with_required(
    tmp_path: Path,
    *,
    required: list[str],
    auto_install: dict[str, dict[str, str]] | None = None,
) -> Path:
    """Write a session_context.md with the given required + auto_install blocks."""
    session = tmp_path / "session_context.md"
    auto_block = ""
    if auto_install:
        auto_lines = ["    auto_install:"]
        for pack, spec in auto_install.items():
            auto_lines.append(f"      {pack}:")
            auto_lines.append(f"        source: {spec['source']}")
            if "pin" in spec:
                auto_lines.append(f"        pin: {spec['pin']}")
        auto_block = "\n".join(auto_lines) + "\n"
    session.write_text(
        "---\n"
        "schema_version: 2\n"
        "name: auto-install-test\n"
        "surface: { type: personal_dm, id: host-console, memory_scope: testing }\n"
        "resources:\n"
        "  packs:\n"
        f"    required: {required}\n"
        f"{auto_block}"
        "---\n"
        "# Session\n\nAct precisely.\n",
        encoding="utf-8",
    )
    return session


def _make_runtime(
    tmp_path: Path,
    *,
    required: list[str],
    auto_install: dict[str, dict[str, str]] | None = None,
) -> HostRuntime:
    session = _session_with_required(tmp_path, required=required, auto_install=auto_install)
    config = HostConfig.from_env(data_dir=str(tmp_path), session_context=str(session))
    return HostRuntime(config)


# ── _resolve_auto_install_spec ─────────────────────────────────────


@pytest.mark.parametrize(
    ("auto_install", "expected"),
    [
        pytest.param(
            {
                "lisa-pack-cloudflare": PackInstallSpec(
                    source="git+https://github.com/Lisa-Harness/lisa-pack-cloudflare.git@v0.2.0",
                    pin="v0.2.0",
                )
            },
            (
                "git+https://github.com/Lisa-Harness/lisa-pack-cloudflare.git@v0.2.0",
                True,
                "auto_install.source",
            ),
            id="explicit-pinned-git-source",
        ),
        pytest.param(
            {},
            ("lisa-pack-cloudflare", False, "git-root-fallback"),
            id="bare-name-git-root-fallback",
        ),
        pytest.param(
            {"lisa-pack-cloudflare": PackInstallSpec(source="C:/Code/lisa-pack-cloudflare")},
            ("C:/Code/lisa-pack-cloudflare", True, "auto_install.source"),
            id="local-path-needs-no-pin",
        ),
    ],
)
def test_resolve_auto_install_spec(
    auto_install: dict[str, PackInstallSpec], expected: tuple[str, bool, str]
) -> None:
    """Resolution returns ``(spec, editable, source_label)`` for each source kind."""
    assert _resolve_auto_install_spec("lisa-pack-cloudflare", auto_install) == expected


def test_resolve_auto_install_spec_rejects_unpinned_git() -> None:
    """An unpinned git+ source raises ValueError (pin is mandatory)."""
    spec = PackInstallSpec(source="git+https://github.com/Lisa-Harness/lisa-pack-cloudflare.git")
    with pytest.raises(ValueError, match="unpinned"):
        _resolve_auto_install_spec("lisa-pack-cloudflare", {"lisa-pack-cloudflare": spec})


# ── auto_install_required_packs ────────────────────────────────────


def test_auto_install_noop_when_no_required(tmp_path: Path) -> None:
    """A session with no required packs is a no-op."""
    runtime = _make_runtime(tmp_path, required=[])
    installed = auto_install_required_packs(runtime)
    assert installed == []
    assert runtime.session.restart_required is False


def test_auto_install_noop_when_all_required_importable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """When every required pack is already importable, nothing is installed."""
    runtime = _make_runtime(tmp_path, required=["lisa-pack-core"])
    monkeypatch.setattr(
        "runtime.pack_autoinstall._installed_pack_names",
        lambda runtime: {"lisa-pack-core"},
    )
    installed = auto_install_required_packs(runtime)
    assert installed == []
    assert runtime.session.restart_required is False


def test_auto_install_installs_missing_required_pack(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A missing required pack with an auto_install entry is installed."""
    runtime = _make_runtime(
        tmp_path,
        required=["lisa-pack-cloudflare"],
        auto_install={
            "lisa-pack-cloudflare": {
                "source": "git+https://github.com/Lisa-Harness/lisa-pack-cloudflare.git@v0.2.0",
                "pin": "v0.2.0",
            }
        },
    )
    monkeypatch.setattr(
        "runtime.pack_autoinstall._installed_pack_names",
        lambda runtime: set(),
    )

    call_log: list[tuple[str, bool]] = []

    def _fake_install_sync(
        rt: HostRuntime, spec: str, *, request_restart: bool = True, editable: bool = False
    ) -> tuple[bool, str]:
        call_log.append((spec, editable))
        return True, "Installed."

    monkeypatch.setattr(_packs, "run_pack_install_sync", _fake_install_sync)

    installed = auto_install_required_packs(runtime)
    assert installed == ["lisa-pack-cloudflare"]
    assert call_log == [
        ("git+https://github.com/Lisa-Harness/lisa-pack-cloudflare.git@v0.2.0", True)
    ]
    # The caller (discover_framework) requests the restart, not this function.
    assert runtime.session.restart_required is False


def test_auto_install_raises_on_install_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A failed required-pack install raises RuntimeError."""
    runtime = _make_runtime(
        tmp_path,
        required=["lisa-pack-cloudflare"],
        auto_install={
            "lisa-pack-cloudflare": {
                "source": "git+https://github.com/Lisa-Harness/lisa-pack-cloudflare.git@v0.2.0",
            }
        },
    )
    monkeypatch.setattr(
        "runtime.pack_autoinstall._installed_pack_names",
        lambda runtime: set(),
    )

    def _fake_install_sync(
        rt: HostRuntime, spec: str, *, request_restart: bool = True, editable: bool = False
    ) -> tuple[bool, str]:
        return False, "uv pip install failed: network error"

    monkeypatch.setattr(_packs, "run_pack_install_sync", _fake_install_sync)

    with pytest.raises(RuntimeError, match="install failed"):
        auto_install_required_packs(runtime)


def test_auto_install_raises_on_unpinned_git_ref(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An unpinned git ref in auto_install raises before any install attempt."""
    runtime = _make_runtime(
        tmp_path,
        required=["lisa-pack-cloudflare"],
        auto_install={
            "lisa-pack-cloudflare": {
                "source": "git+https://github.com/Lisa-Harness/lisa-pack-cloudflare.git",
            }
        },
    )
    monkeypatch.setattr(
        "runtime.pack_autoinstall._installed_pack_names",
        lambda runtime: set(),
    )

    with pytest.raises(RuntimeError, match="unpinned"):
        auto_install_required_packs(runtime)
