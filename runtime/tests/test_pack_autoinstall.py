"""Tests for startup auto-install of session-required packs.

Verifies that ``runtime.pack_autoinstall.auto_install_required_packs``:
- is a no-op when all required packs are already importable,
- installs missing required packs via the sync install core,
- raises RuntimeError when a required pack fails to install,
- rejects unpinned git refs at spec resolution time,
- requests a restart when at least one pack is installed.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest
from kernel.session_plan import PackInstallSpec
from runtime.commands import _packs
from runtime.config import HostConfig
from runtime.pack_autoinstall import (
    _resolve_auto_install_spec,
    auto_install_required_packs,
)
from runtime.paths import packs_venv_for, packs_venv_site_packages
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
                "git+https://github.com/Lisa-Harness/lisa-pack-cloudflare.git@v0.2.0"
                "#egg=lisa-pack-cloudflare",
                True,
                "auto_install.source",
            ),
            id="explicit-pinned-git-source",
        ),
        pytest.param(
            {
                "lisa-pack-cloudflare": PackInstallSpec(
                    source="git+https://github.com/Lisa-Harness/lisa-go.git@v0.1.0#subdirectory=pack",
                    pin="v0.1.0",
                )
            },
            (
                "git+https://github.com/Lisa-Harness/lisa-go.git@v0.1.0"
                "#subdirectory=pack&egg=lisa-pack-cloudflare",
                True,
                "auto_install.source",
            ),
            id="monorepo-subdirectory-git-source",
        ),
        pytest.param(
            {},
            ("lisa-pack-cloudflare", False, "git-root-fallback"),
            id="bare-name-git-root-fallback",
        ),
        pytest.param(
            {"lisa-pack-cloudflare": PackInstallSpec(source="C:/Code/lisa-pack-cloudflare")},
            (str(Path("C:/Code/lisa-pack-cloudflare")), True, "auto_install.source"),
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
        (
            "git+https://github.com/Lisa-Harness/lisa-pack-cloudflare.git@v0.2.0"
            "#egg=lisa-pack-cloudflare",
            True,
        )
    ]
    # The caller (discover_framework) requests the restart, not this function.
    assert runtime.session.restart_required is False


def test_auto_install_activates_packs_venv_created_during_boot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The packs venv created by auto-install becomes importable this boot.

    On a first launch the CLI's startup injection no-ops (no venv yet); the
    install then creates it, so auto-install must re-inject or every
    ``lisa_packs.*`` import fails for the whole session.
    """
    runtime = _make_runtime(tmp_path, required=["lisa-pack-cloudflare"])
    monkeypatch.setattr(
        "runtime.pack_autoinstall._installed_pack_names",
        lambda runtime: set(),
    )

    site_packages = packs_venv_site_packages(packs_venv_for(tmp_path))

    def _fake_install_sync(
        rt: HostRuntime, spec: str, *, request_restart: bool = True, editable: bool = False
    ) -> tuple[bool, str]:
        (site_packages / "lisa_packs_probe").mkdir(parents=True)
        (site_packages / "lisa_packs_probe" / "__init__.py").write_text("", encoding="utf-8")
        return True, "Installed."

    monkeypatch.setattr(_packs, "run_pack_install_sync", _fake_install_sync)
    original_path = list(sys.path)
    try:
        assert auto_install_required_packs(runtime) == ["lisa-pack-cloudflare"]
        assert str(site_packages) in sys.path
        assert importlib.util.find_spec("lisa_packs_probe") is not None
    finally:
        sys.path[:] = original_path
        sys.modules.pop("lisa_packs_probe", None)
        importlib.invalidate_caches()


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


def test_auto_install_partial_success_is_hard_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A failed required pack prevents startup even after another pack succeeds."""
    runtime = _make_runtime(
        tmp_path,
        required=["lisa-pack-alpha", "lisa-pack-beta"],
        auto_install={
            "lisa-pack-alpha": {
                "source": "git+https://github.com/Lisa-Harness/lisa-pack-alpha.git@v1.0.0",
                "pin": "v1.0.0",
            },
            "lisa-pack-beta": {
                "source": "git+https://github.com/Lisa-Harness/lisa-pack-beta.git@v1.0.0",
                "pin": "v1.0.0",
            },
        },
    )
    monkeypatch.setattr(
        "runtime.pack_autoinstall._installed_pack_names",
        lambda runtime: set(),
    )

    attempted: list[str] = []

    def _fake_install_sync(
        rt: HostRuntime, spec: str, *, request_restart: bool = True, editable: bool = False
    ) -> tuple[bool, str]:
        attempted.append(spec)
        if "lisa-pack-alpha" in spec:
            return False, "uv pip install failed: network error"
        return True, "Installed."

    monkeypatch.setattr(_packs, "run_pack_install_sync", _fake_install_sync)
    monkeypatch.setattr(
        "runtime.pack_autoinstall._activate_packs_venv",
        lambda rt: None,
    )

    with pytest.raises(RuntimeError, match="install failed"):
        auto_install_required_packs(runtime)

    assert len(attempted) == 2


def test_auto_install_spec_resolution_error_aborts_immediately(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A spec-resolution error aborts before any install is attempted.

    Unlike an install failure, an unresolvable spec (unpinned git ref) raises
    immediately: later packs are not attempted and earlier partial success
    is not returned.
    """
    runtime = _make_runtime(
        tmp_path,
        required=["lisa-pack-alpha", "lisa-pack-beta"],
        auto_install={
            "lisa-pack-alpha": {
                "source": "git+https://example.invalid/lisa-pack-alpha.git",
            },
            "lisa-pack-beta": {
                "source": "git+https://example.invalid/lisa-pack-beta.git@v1.0.0",
                "pin": "v1.0.0",
            },
        },
    )
    monkeypatch.setattr(
        "runtime.pack_autoinstall._installed_pack_names",
        lambda runtime: set(),
    )

    attempted: list[str] = []

    def _fake_install_sync(
        rt: HostRuntime, spec: str, *, request_restart: bool = True, editable: bool = False
    ) -> tuple[bool, str]:
        attempted.append(spec)
        return True, "Installed."

    monkeypatch.setattr(_packs, "run_pack_install_sync", _fake_install_sync)

    with pytest.raises(RuntimeError, match="pin"):
        auto_install_required_packs(runtime)

    # The pass aborted before any install ran.
    assert attempted == []


def test_auto_install_trust_policy_error_in_later_spec_aborts_before_any_install(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Every spec is trust-validated before the first install begins."""
    runtime = _make_runtime(
        tmp_path,
        required=["lisa-pack-alpha", "lisa-pack-beta"],
        auto_install={
            "lisa-pack-alpha": {
                "source": "git+https://github.com/Lisa-Harness/lisa-pack-alpha.git@v1.0.0",
                "pin": "v1.0.0",
            },
            "lisa-pack-beta": {
                "source": "git+https://gitlab.com/Lisa-Harness/lisa-pack-beta.git@v1.0.0",
                "pin": "v1.0.0",
            },
        },
    )
    monkeypatch.setattr(
        "runtime.pack_autoinstall._installed_pack_names",
        lambda runtime: set(),
    )

    attempted: list[str] = []

    def _fake_install_sync(
        rt: HostRuntime, spec: str, *, request_restart: bool = True, editable: bool = False
    ) -> tuple[bool, str]:
        attempted.append(spec)
        return True, "Installed."

    monkeypatch.setattr(_packs, "run_pack_install_sync", _fake_install_sync)

    with pytest.raises(RuntimeError, match=r"Git host 'gitlab\.com'"):
        auto_install_required_packs(runtime)

    assert attempted == []


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


def test_auto_install_normalizes_underscore_pack_name_lookup() -> None:
    spec, editable, source_label = _resolve_auto_install_spec(
        "lisa-pack-beta",
        {
            "lisa_pack_beta": PackInstallSpec(
                source="git+https://github.com/Lisa-Harness/lisa-pack-beta.git@v1.0.0",
                pin="v1.0.0",
            )
        },
    )
    assert spec.endswith("#egg=lisa-pack-beta")
    assert editable is True
    assert source_label == "auto_install.source"


def test_auto_install_rejects_git_egg_name_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(
        tmp_path,
        required=["lisa-pack-beta"],
        auto_install={
            "lisa-pack-beta": {
                "source": "git+https://github.com/Lisa-Harness/lisa-go.git@v1.0.0"
                "#egg=lisa-pack-alpha",
                "pin": "v1.0.0",
            }
        },
    )
    monkeypatch.setattr(
        "runtime.pack_autoinstall._installed_pack_names",
        lambda runtime: set(),
    )
    attempted: list[str] = []
    monkeypatch.setattr(
        _packs,
        "run_pack_install_sync",
        lambda rt, spec, request_restart=False, editable=False: (
            attempted.append(spec)
            or (
                True,
                "Installed.",
            )
        ),
    )

    with pytest.raises(RuntimeError, match="names must match"):
        auto_install_required_packs(runtime)

    assert attempted == []


def test_auto_install_rejects_local_project_name_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pack_dir = tmp_path / "lisa-pack-beta"
    pack_dir.mkdir()
    (pack_dir / "pyproject.toml").write_text(
        '[project]\nname = "lisa-pack-beta"\nversion = "0.0.1"\n',
        encoding="utf-8",
    )
    runtime = _make_runtime(
        tmp_path,
        required=["lisa-pack-alpha"],
        auto_install={
            "lisa-pack-alpha": {
                "source": str(pack_dir),
                "editable": True,
            }
        },
    )
    monkeypatch.setattr(
        "runtime.pack_autoinstall._installed_pack_names",
        lambda runtime: set(),
    )
    attempted: list[str] = []
    monkeypatch.setattr(
        _packs,
        "run_pack_install_sync",
        lambda rt, spec, request_restart=False, editable=False: (
            attempted.append(spec)
            or (
                True,
                "Installed.",
            )
        ),
    )

    with pytest.raises(RuntimeError, match="names must match"):
        auto_install_required_packs(runtime)

    assert attempted == []


def test_auto_install_pinning_is_mandatory_when_general_policy_disables_refs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Auto-install remains pinned even if the general /pack policy is disabled."""
    monkeypatch.setenv("LISA_PACK_REQUIRE_GIT_REF", "false")
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
    attempted: list[str] = []

    def _fake_install_sync(
        rt: HostRuntime, spec: str, *, request_restart: bool = True, editable: bool = False
    ) -> tuple[bool, str]:
        attempted.append(spec)
        return True, "Installed."

    monkeypatch.setattr(_packs, "run_pack_install_sync", _fake_install_sync)

    with pytest.raises(RuntimeError, match="unpinned"):
        auto_install_required_packs(runtime)

    assert attempted == []
