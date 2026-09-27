"""Test-runtime factory for constructing pre-wired ``HostRuntime`` instances.

Centralizes the ``HostRuntime(HostConfig.from_env(data_dir=str(tmp_path), ...))``
boilerplate repeated across 50+ test sites. Writes a minimal ``models.yaml``,
constructs the runtime, and optionally pre-loads the model manifest and/or
runs ``setup_framework`` for tests that need catalog/content wiring.

Default ``setup_framework=False`` keeps tests side-effect-free: no kernel
module-global mutation, no MCP server subprocesses. Tests that need the
workflow catalog, operation catalog, or content registry pass
``setup_framework=True``.

Dependency direction: ``runtime.testing.runtime_factory`` ->
``runtime.config``, ``runtime.runtime``, ``runtime.model_runtime``,
``runtime.mcp_pool``. This module is test-only; it must not be imported by
production code. (Moved from ``kernel.testing`` to fix the kernel→host
layer violation.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from runtime.config import HostConfig
from runtime.runtime import HostRuntime

_DEFAULT_MODELS_YAML = "\n".join(
    [
        "version: 1",
        "mode_models: {default: gemma3:latest}",
        "models:",
        "  - model_id: gemma3:latest",
        "    provider: ollama",
    ]
)


def build_test_runtime(
    tmp_path: Path,
    *,
    models_yaml: str | None = None,
    manifest: Any | None = None,
    setup_framework: bool = False,
    **config_kwargs: Any,
) -> HostRuntime:
    """Build a ``HostRuntime`` pre-wired with stubs for unit testing.

    Writes ``models_yaml`` (or a minimal default) to ``tmp_path / "models.yaml"``,
    constructs ``HostRuntime`` from ``HostConfig.from_env(data_dir=tmp_path, ...)``,
    and optionally pre-loads the manifest and/or runs ``setup_framework``.

    Args:
        tmp_path: pytest ``tmp_path`` fixture — the runtime's ``data_dir``.
        models_yaml: Custom ``models.yaml`` content. When ``None``, writes a
            minimal default with one ``gemma3:latest`` model. When an empty
            string, skips writing the file (for tests that want no manifest).
        manifest: When provided, sets ``runtime.session.manifest`` directly after
            construction, bypassing the file load. Takes precedence over
            ``models_yaml``.
        setup_framework: When ``True``, runs ``runtime.mcp_pool.setup_framework``
            (which registers operation/tool/content MCP, routing vocabulary,
            workflow boot-load, and manifest pre-load). Default ``False`` —
            most unit tests only need the runtime, not the full catalog wiring.
        **config_kwargs: Forwarded to ``HostConfig.from_env`` (e.g.
            ``user_slug="user"``, ``no_thread=True``, ``surface_type="..."``).

    Returns:
        A ``HostRuntime`` ready for test assertions.
    """
    if models_yaml != "":
        from sdk.layout import CONFIG_DIR

        manifest_dir = tmp_path / CONFIG_DIR
        manifest_dir.mkdir(parents=True, exist_ok=True)
        (manifest_dir / "models.yaml").write_text(
            models_yaml or _DEFAULT_MODELS_YAML,
            encoding="utf-8",
        )

    config = HostConfig.from_env(data_dir=str(tmp_path), **config_kwargs)
    runtime = HostRuntime(config)

    if manifest is not None:
        runtime.session.manifest = manifest
    else:
        # Seed a default provider selection for ``web_search`` so the
        # catalog subprocess can resolve the collision when multiple
        # search provider packs (brave, serper, tavily) are installed
        # alongside the search contract pack. Production sessions provide
        # this via the session-context ``providers.tools`` block. Done
        # unconditionally (not just for ``setup_framework=True``) because
        # many tests call ``mcp_pool.setup_framework(runtime)`` directly
        # after construction.
        from kernel.session_plan import (
            ProviderSelections,
        )

        existing = runtime.session_plan.manifest.providers
        if not existing.tools.get("web_search"):
            new_providers = ProviderSelections(
                modes=existing.modes,
                operations=existing.operations,
                tools={**existing.tools, "web_search": "lisa-pack-brave"},
                capabilities=existing.capabilities,
            )
            runtime.session_plan = runtime.session_plan.model_copy(
                update={
                    "manifest": runtime.session_plan.manifest.model_copy(
                        update={"providers": new_providers}
                    )
                }
            )

    if setup_framework:
        from runtime.mcp_pool import setup_framework as _setup_framework

        _setup_framework(runtime)

    return runtime
