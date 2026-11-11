"""Branch tests for the mcp_pool framework-wiring helpers.

These cover the in-flight seams the full-suite runs do not reach:
``_build_catalog_source_config``'s payload assembly (with and without
session provider selections), ``_release_previous_build``'s invalidation
and release steps across both transports (including the stdio-side
failure that continues), and ``_resolve_kernel_transports``' both
returns. The helpers are called with ``SimpleNamespace`` runtimes and a
``_FrameworkDiscovery``-shaped stub, so no subprocess is spawned.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from kernel.context._catalog_sources import CatalogSource
from kernel.session_plan import ProviderSelections
from kernel.tools.mcp_client import MCPClientConfig
from runtime.mcp_pool import (
    _build_catalog_source_config,
    _release_previous_build,
    _resolve_kernel_transports,
)


class _Discovery:
    """``_FrameworkDiscovery``-shaped stub (private contract, test-only)."""

    def __init__(self, tmp_path: Path) -> None:
        self.roots = [tmp_path / "roots" / "pack-a"]
        self.disabled_packs: set[str] = set()
        self.lock: dict[str, Any] = {}
        self.host_manifest_path = tmp_path / "mcp_servers.yaml"
        self.user_manifest_path: Path | None = None
        self.host_specs: list[Any] = []
        self.user_specs: list[Any] = []
        self.pack_specs: list[Any] = []


class _Lease:
    def __init__(self) -> None:
        self.released = False

    def release(self) -> None:
        self.released = True


def _runtime(tmp_path: Path, *, selections: ProviderSelections | None = None) -> Any:
    manifest = SimpleNamespace(
        providers=selections
        if selections is not None
        else SimpleNamespace(modes={}, operations={}, tools={}, capabilities={}),
    )
    return SimpleNamespace(
        paths=SimpleNamespace(data_dir=tmp_path, overlays_dir=tmp_path / "overlays"),
        session_plan=SimpleNamespace(manifest=manifest),
        resources=SimpleNamespace(shared_session_lease=None),
        _in_process_mcp=True,
    )


def test_build_catalog_source_config_assembles_the_frozen_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sdk.catalog_sources import CatalogSourceConfig

    monkeypatch.delenv("LISA_DISABLED_PACKS", raising=False)
    runtime = _runtime(tmp_path)
    discovery = _Discovery(tmp_path)
    config = _build_catalog_source_config(runtime, discovery)

    assert isinstance(config, CatalogSourceConfig)
    assert config.content_roots == discovery.roots
    assert config.provider_selections_json == ""
    assert config.overlays_dir == runtime.paths.overlays_dir


def test_release_previous_build_requires_a_lease(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.resources.shared_session_lease = None
    # With no lease the no-op returns without touching anything.
    _release_previous_build(runtime, _Discovery(tmp_path), _UnusedConfig().value)


def test_resolution_transports_in_process_returns_direct_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime(tmp_path)
    discovery = _Discovery(tmp_path)
    lease = _Lease()
    catalog_source, content_source = _resolve_kernel_transports(
        runtime,
        discovery,
        _UnusedConfig().value,
        lease,  # type: ignore[arg-type]
    )
    assert isinstance(catalog_source, CatalogSource)
    assert content_source is not None
    assert catalog_source is not None
    catalog_source.invalidate()


class _UnusedConfig:
    value: MCPClientConfig

    def __init__(self) -> None:
        self.value = MCPClientConfig(
            server_id="lisa-content", command="python", args=["-c", "pass"]
        )


class _ResettingLease:
    def __init__(self) -> None:
        self.released_count = 0

    def release(self) -> None:
        self.released_count += 1


def test_release_previous_build_stdio_invalidates_then_releases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stdio branch probes the content server, resets the snapshot, then releases."""
    from kernel.context import mcp_content_registry as registry_mod

    from kernel import ipsar_fsm

    invalidated: list[bool] = []

    class Probe:
        def invalidate(self) -> None:
            invalidated.append(True)

    monkeypatch.setattr(registry_mod, "MCPContentRegistry", lambda config, lease=None: Probe())

    resets: list[bool] = []
    monkeypatch.setattr(ipsar_fsm, "reset_workflow_catalog", lambda: resets.append(True))

    lease = _ResettingLease()
    runtime = _runtime(tmp_path)
    runtime.resources.shared_session_lease = lease
    runtime._in_process_mcp = False

    _release_previous_build(runtime, _Discovery(tmp_path), _UnusedConfig().value)
    assert invalidated == [True]
    assert resets == [True]
    assert lease.released_count == 1
    assert runtime.resources.shared_session_lease is None


def test_release_previous_build_stdio_swallows_probe_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing invalidation logs and continues: release + reset still happen."""
    from kernel.context import mcp_content_registry as registry_mod

    from kernel import ipsar_fsm

    class Boom:
        def invalidate(self) -> None:
            raise RuntimeError("probe failed")

    monkeypatch.setattr(registry_mod, "MCPContentRegistry", lambda config, lease=None: Boom())
    resets: list[bool] = []
    monkeypatch.setattr(ipsar_fsm, "reset_workflow_catalog", lambda: resets.append(True))

    lease = _ResettingLease()
    runtime = _runtime(tmp_path)
    runtime.resources.shared_session_lease = lease
    runtime._in_process_mcp = False

    _release_previous_build(runtime, _Discovery(tmp_path), _UnusedConfig().value)
    assert resets == [True]
    assert lease.released_count == 1


def test_release_previous_build_in_process_invalidates_via_direct_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sdk.catalog_sources as catalog_sources_mod

    from kernel import ipsar_fsm

    invalidated: list[bool] = []

    class Direct:
        def __init__(self, config: object) -> None:
            self.config = config

        def invalidate(self) -> None:
            invalidated.append(True)

    monkeypatch.setattr(catalog_sources_mod, "DirectCatalogSource", Direct)
    resets: list[bool] = []
    monkeypatch.setattr(ipsar_fsm, "reset_workflow_catalog", lambda: resets.append(True))

    lease = _ResettingLease()
    runtime = _runtime(tmp_path)
    runtime.resources.shared_session_lease = lease

    _release_previous_build(runtime, _Discovery(tmp_path), _UnusedConfig().value)
    assert invalidated == [True]
    assert resets == [True]
    assert lease.released_count == 1
