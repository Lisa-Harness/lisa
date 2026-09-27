"""Auto-capture operation-context wiring (2026-09-23 finances-profile incident).

The post-turn auto-capture path built its ``OperationExecutionContext``
WITHOUT ``capability_binding_provider`` and ``model_router`` — unlike every
turn-loop dispatch site (``_build_entry_ctx``, the approval-resume ``common``
context, the mid-turn operation dispatch). Consequences, observed on the
finances profile:

- ``synthesize_and_store_learning`` resolved ``memory_distill_max_tokens``
  from the module fallback (512) instead of the session's 5000 — on a
  reasoning model via Ollama, reasoning bills against ``max_tokens``, so the
  JSON array truncated mid-fact and the ``_looks_like_json`` guard (correctly)
  refused to fall back to prose.
- Operation-scoped router rules (finances routes learning distills to
  ``glm-5.3:cloud``) never applied on the auto-capture path.

These tests pin that ``build_turn_hooks`` wires both seams onto the
auto-capture context.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from kernel.capability_binding import CapabilityRef
from kernel.models import OperationDefinition
from runtime.turn_hooks_builder import build_turn_hooks

from runtime import HostRuntime


def _make_runtime(tmp_path: Path) -> HostRuntime:
    from runtime.testing import build_test_runtime

    models_yaml = "\n".join(
        [
            "version: 1",
            "mode_models: {default: gemma3:latest}",
            "router:",
            "  backend: lisa-router",
            "  rules:",
            '    - when: route.operation_id in ["synthesize_and_store_learning"]',
            "      use: gemma3:latest",
            "models:",
            "  - model_id: gemma3:latest",
            "    provider: ollama",
        ]
    )
    return build_test_runtime(
        tmp_path,
        models_yaml=models_yaml,
        user_slug="user",
        surface_type="personal_dm",
        surface_id="surface",
    )


def _seed_pack_config(runtime: HostRuntime, pack_config: dict[str, Any]) -> None:
    """Put a ``lisa-pack-memory`` configuration block into the session plan."""

    manifest = runtime.session_plan.manifest
    packs = manifest.resources.packs
    manifest = manifest.model_copy(
        update={
            "resources": manifest.resources.model_copy(
                update={
                    "packs": packs.model_copy(
                        update={
                            "configuration": {"lisa-pack-memory": pack_config},
                        }
                    ),
                }
            ),
            "providers": manifest.providers.model_copy(
                update={
                    "capabilities": {
                        **manifest.providers.capabilities,
                        "memory": "lisa-pack-memory",
                    }
                }
            ),
        }
    )
    runtime.session_plan = runtime.session_plan.model_copy(update={"manifest": manifest})


def test_memory_provider_config_without_catalog_capabilities_builds_hooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _make_runtime(tmp_path)
    _seed_pack_config(runtime, {"memory_distill_max_tokens": 5000})
    monkeypatch.setattr(
        "kernel.operations.registry.get_operation_catalog",
        lambda: {
            "other_operation": OperationDefinition(
                operation_id="other_operation", owner_id="lisa-pack-other"
            )
        },
    )

    hooks = build_turn_hooks(runtime, {"mode_id": "default"})

    assert hooks.services.capability_binding_provider is not None
    assert hooks.services.capability_binding_provider.binding_for(
        CapabilityRef("operation", "missing", "lisa-pack-memory")
    ) is None


@pytest.mark.asyncio
async def test_auto_capture_context_carries_pack_config_resolver_and_router(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_on_turn_complete``'s capture context must carry the resolver + router.

    Before the fix, the capture context omitted both seams: the resolver
    defaulted to ``EMPTY_PACK_CONFIGURATION`` (512-token module fallback for
    the distill budget) and the router was ``None`` (operation-scoped rules
    silently skipped). We intercept the kernel registry dispatch and assert
    both are live objects on the context for the auto-capture invocation.
    """
    runtime = _make_runtime(tmp_path)
    _seed_pack_config(runtime, {"memory_distill_max_tokens": 5000})
    monkeypatch.setattr(
        "kernel.operations.registry.get_operation_catalog",
        lambda: {
            "synthesize_and_store_learning": OperationDefinition(
                operation_id="synthesize_and_store_learning",
                owner_id="lisa-pack-memory",
            )
        },
    )
    hooks = build_turn_hooks(runtime, {"mode_id": "default"})
    assert hooks.lifecycle is not None

    captured: dict[str, Any] = {}

    import runtime.turn_hooks_builder as hooks_module

    async def _fake_execute_operation(
        operation_id: str,
        ctx: Any,
    ) -> Any:
        if operation_id == "synthesize_and_store_learning":
            captured["ctx"] = ctx
        from kernel.models import OperationResult

        return OperationResult(output_text="captured")

    # ``turn_hooks_builder`` imports ``execute_operation`` by name, so patch
    # the module-level reference it actually calls — not the registry.
    monkeypatch.setattr(
        hooks_module,
        "execute_operation",
        _fake_execute_operation,
    )

    # Fire the hook with a turn that trips the write-effect trigger.
    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "lisa-storage.write_file", "arguments": "{}"}}],
        },
        {
            "role": "tool",
            "content": "the approved action executed successfully",
        },
    ]
    receipts = [{"effect": "write", "capability_id": "lisa-storage.write_file"}]
    await hooks.lifecycle.on_turn_complete(
        mode_id="default",
        query="remember that the ledger uses USD",
        response_text="Done.",
        messages=messages,
        effect_receipts=receipts,
    )

    ctx = captured.get("ctx")
    assert ctx is not None, "auto-capture must dispatch synthesize_and_store_learning"

    # Pack config binding: must be session configuration, not an empty mapping.
    resolver = getattr(ctx, "capability_binding_provider", None)
    assert resolver is not None, "capture context must wire capability_binding_provider"
    config = resolver.binding_for(
        CapabilityRef("operation", "synthesize_and_store_learning", "lisa-pack-memory")
    ).configuration
    assert dict(config).get("memory_distill_max_tokens") == 5000, (
        "resolver must surface the session's lisa-pack-memory configuration"
    )

    # Model router: must be wired, not None.
    router = getattr(ctx, "model_router", None)
    assert router is not None, "capture context must wire model_router"
