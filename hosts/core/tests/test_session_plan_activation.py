"""Focused SessionPlan-to-content-root activation coverage."""

from __future__ import annotations

from pathlib import Path

import pytest

from hosts.core import content_roots
from kernel.session_plan import compile_session_plan


def _plan_with_packs(
    packs: str,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    plan = compile_session_plan(
        text=(
            "---\n"
            "schema_version: 2\n"
            "name: activation-test\n"
            "surface: { type: personal_dm, id: host-console, memory_scope: testing }\n"
            "resources:\n"
            "  packs:\n"
            f"{packs}"
            "---\n"
            "# Session\n"
        ),
        source_uri="workspace:activation-test.md",
    )
    selection = plan.manifest.resources.packs
    return selection.include, selection.exclude, selection.required


def test_session_pack_selection_overrides_lock_default_without_overriding_host_mandatory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    core_root = tmp_path / "sdk" / "packs" / "core"
    cloudflare_root = tmp_path / "lisa_pack_cloudflare"
    search_root = tmp_path / "sdk" / "packs" / "search"
    core_root.mkdir(parents=True)
    cloudflare_root.mkdir()
    search_root.mkdir(parents=True)
    monkeypatch.setattr(
        content_roots,
        "always_on_pack_names",
        lambda _path=None, **_kwargs: {"lisa-pack-core"},
    )
    monkeypatch.setattr(
        content_roots,
        "discover_installed_content_roots",
        lambda: [core_root, cloudflare_root, search_root],
    )
    monkeypatch.setattr(content_roots, "discover_configured_content_roots", lambda: [])
    content_roots.save_installed_packs_lock(
        tmp_path,
        {
            "packs": [
                {
                    "package_name": "lisa-pack-cloudflare",
                    "source_kind": "bundled",
                    "enabled": False,
                },
                {
                    "package_name": "lisa-pack-search",
                    "source_kind": "bundled",
                    "enabled": True,
                },
            ]
        },
    )

    included, _, required = _plan_with_packs(
        "    include: [lisa-pack-cloudflare]\n"
    )
    included_roots = content_roots.sdk_content_roots(
        tmp_path,
        session_pack_include=(*included, *required),
    )
    assert cloudflare_root in included_roots
    assert search_root not in included_roots

    _, excluded, required = _plan_with_packs(
        "    exclude: [lisa-pack-cloudflare, lisa-pack-core]\n"
    )
    excluded_roots = content_roots.sdk_content_roots(
        tmp_path,
        session_pack_include=required,
        session_pack_exclude=excluded,
    )
    assert cloudflare_root not in excluded_roots
    assert core_root in excluded_roots


def test_infer_pack_package_name_maps_lisa_packs_site_packages_to_canonical_name(
    tmp_path: Path,
) -> None:
    """Regression: external packs installed under ``site-packages/lisa_packs/<name>``
    must infer the canonical ``lisa-pack-<name>`` distribution name (matching the
    pyproject name) so session-plan provider selections like
    ``memory: lisa-pack-mem0`` resolve. Without this, ``infer_pack_package_name``
    fell back to the bare directory name (``mem0``) and the memory capability
    owner selection silently returned None — routing ``memory.search`` to the
    excluded first-party ``lisa-memory`` server instead of mem0."""
    site_packages = tmp_path / "site-packages" / "lisa_packs"
    mem0_root = site_packages / "mem0"
    mem0_root.mkdir(parents=True)
    brave_root = site_packages / "brave_search"
    brave_root.mkdir()

    assert content_roots.infer_pack_package_name(mem0_root) == "lisa-pack-mem0"
    assert content_roots.infer_pack_package_name(brave_root) == "lisa-pack-brave-search"


def test_infer_pack_package_name_sdk_packs_keep_canonical_name(tmp_path: Path) -> None:
    """SDK workspace packs under ``sdk/packs/<name>`` keep resolving to
    ``lisa-pack-<name>`` — the new lisa_packs branch must not regress them."""
    sdk_root = tmp_path / "sdk" / "packs" / "memory"
    sdk_root.mkdir(parents=True)
    assert content_roots.infer_pack_package_name(sdk_root) == "lisa-pack-memory"