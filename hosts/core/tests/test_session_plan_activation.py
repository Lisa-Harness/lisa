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