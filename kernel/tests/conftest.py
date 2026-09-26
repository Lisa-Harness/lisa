"""Shared pytest fixtures for kernel tests.

All kernel tests that need the mode or operation catalog must use the
MCP-mediated catalog loading fixture from ``conftest_catalog.py``.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

# Ensure sibling conftest_catalog is importable regardless of pytest import mode.
sys.path.insert(0, str(Path(__file__).parent))
from conftest_catalog import _mcp_catalog_setup  # noqa: F401

# ── Provider-gateway bridge for TurnHooks fakes ─────────────────────
#
# The kernel's turn loop no longer calls the ``TurnHooks.models.complete``
# callable — every model call routes through
# ``ContextSeams.provider_gateways``. Kernel unit tests still construct
# ``TurnHooks(models=ModelHooks(complete=...))`` fakes. Rather than
# touch 175 test sites, the autouse fixture below patches ``TurnHooks.__init__``
# so a non-None ``models.complete`` callable is ALSO wrapped as a
# ``StubGatewayProvider`` and set onto ``context_seams.provider_gateways`` (when
# the caller did not already supply one). Tests that assert the *absence* of a
# completion path (e.g. ``test_run_turn_requires_complete_hook``) pass
# ``complete=None`` and get no gateway, so the kernel's "no gateway wired"
# ``NotImplementedError`` still surfaces.
_ORIGINAL_TURNHOOKS_INIT: Any = None


@pytest.fixture(autouse=True)
def _bridge_turnhooks_complete_to_gateway() -> Any:
    """Route ``TurnHooks(models=ModelHooks(complete=...))`` fakes through the
    provider gateway.

    Wraps the test's ``models.complete`` callable into a ``StubGatewayProvider``
    so the kernel's gateway call-site resolves it. Preserves the caller's
    explicit ``context_seams`` if provided. Yields so the patch is active only
    during each test.
    """

    global _ORIGINAL_TURNHOOKS_INIT  # noqa: PLW0603 — conftest snapshot/restore seam

    from kernel.testing.gateway_stub import StubGatewayProvider
    from kernel.turn_runner import ContextSeams, TurnHooks

    if _ORIGINAL_TURNHOOKS_INIT is None:
        _ORIGINAL_TURNHOOKS_INIT = TurnHooks.__init__

    def _patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        _ORIGINAL_TURNHOOKS_INIT(self, *args, **kwargs)
        if getattr(self, "context_seams", None) is None:
            # Should not happen (default_factory guarantees it), but be safe.
            self.context_seams = ContextSeams()
        complete_fn = getattr(self.models, "complete", None)
        if (
            complete_fn is not None
            and getattr(self.context_seams, "provider_gateways", None) is None
        ):
            self.context_seams = replace(
                self.context_seams,
                provider_gateways=StubGatewayProvider(
                    complete_fn,
                    complete_stream_fn=getattr(self.models, "complete_stream", None),
                ),
            )

    TurnHooks.__init__ = _patched_init  # type: ignore[method-assign]
    try:
        yield
    finally:
        if _ORIGINAL_TURNHOOKS_INIT is not None:
            TurnHooks.__init__ = _ORIGINAL_TURNHOOKS_INIT  # type: ignore[method-assign]


@pytest.fixture(scope="module", autouse=True)
def _wire_sub_turn_content_registry() -> None:
    """Wire a directory-based content registry for the sub-turn contract assets.

    ``kernel.context.sub_turn_contract`` loads the fork/delegate contract and
    midpoint-nudge prose from authored Markdown. Production wires the
    ``MCPContentRegistry`` at host startup (``host.mcp_pool``) against the
    ``sdk/packs/core`` content root; kernel unit tests that call
    ``_build_sub_turn_contract`` directly need the same prose but must not
    depend on the ``sdk`` package (kernel is runtime-neutral). The canonical
    prompt assets are mirrored under ``kernel/testing/fixtures/sub_turn/`` so
    the kernel test suite is self-contained. A lightweight direct-file
    registry avoids spawning the catalog MCP subprocess for these unit tests.
    """

    class _DirContentRegistry:
        def __init__(self, root: Path) -> None:
            self._root = root

        def read_authored_text(self, relative_path: str) -> str:
            return (self._root / relative_path).read_text(encoding="utf-8")

    # The prompt assets live under the kernel *package* (src layout:
    # kernel/src/kernel/testing/fixtures). parents[1] is the member dir;
    # the package itself is its src/kernel subtree.
    fixtures_root = (
        Path(__file__).resolve().parents[1] / "src" / "kernel" / "testing" / "fixtures"
    ).resolve()
    from kernel.context.sub_turn_contract import set_content_registry

    set_content_registry(_DirContentRegistry(fixtures_root))
    yield


@pytest.fixture(autouse=True)
def _reset_routing_vocabulary() -> None:
    """Reset and re-register the session routing vocabulary between tests.

    The routing-vocabulary registry is a process global injected by the
    host boot path from the session plan. The kernel has no hardcoded
    fallbacks (the 2026-08-11 hard cutover): a reader raises
    ``RuntimeError`` when no session vocabulary is wired. Kernel-direct
    tests that exercise ``_compute_capability_routing_hint`` or
    ``_query_is_sequential`` need a wired registry. Reset between tests
    to clear state, then register a representative vocabulary so the
    common case works without each test opting in.
    """
    from kernel.testing.routing_vocabulary import representative_routing_vocabulary

    with representative_routing_vocabulary():
        yield


# ── Synthetic target-mode augmentation for the ambient test catalog ──
#
# The routing/transition/fork tests route against target modes
# (``markdown``, ``remember``, ``ponder``, ``search``, ``coding``, …)
# that were removed from every authoritative pack by the 2026-08 mode
# migrations (093f6f8 moved search/markdown/letta to external submodules;
# 336a73f and 04d921d deleted the domain modes). This venv has no
# installed external packs, so the ambient fixture catalog serves only
# ``default`` / ``letta`` / ``letta-agent`` and every test that hands off
# to a named target falls through with "Unknown transition target".
#
# The fix is hermetic: after the ambient catalog loads, augment it with
# synthetic routable WorkflowModeConfigs for the target vocabulary the
# tests were authored against. Each synthetic mode is generic (tags and
# keywords keyed on its own id, fork_allowed) and carries no
# pack-specific semantics, so the kernel's pure mapping is what the tests
# exercise. Tests that need a *specific* mode shape still patch the
# catalog per-test (the kernel test-seams rule) and their patch wins.

_SYNTHETIC_TEST_MODES: dict[str, dict[str, Any]] = {
    # markdown mirrors the real pack's shape: a write-contract mode whose
    # completion contract names a write-effect operation the catalog cannot
    # resolve (unknown → conservative write), so the parent no_tools block
    # fires on unverified prose, matching the production contract.
    "markdown": {
        "tags": ["write"],
        "keywords": ["markdown", "document", "write"],
        "authored_ops": ["markdown_forge_document"],
    },
    # remember mirrors the pre-migration core pack shape: an action mode
    # whose authored write operation the tests' child-classification
    # expectations assume (mode_completion_effects non-empty). The ambient
    # memory ops (recall_memory / synthesize_and_store_learning) are bound
    # so the authoritative tool surface advertises them.
    "remember": {
        "tags": ["remember"],
        "keywords": ["remember", "note", "capture"],
        "authored_ops": ["remember_capture"],
        "bound_ops": ["recall_memory", "synthesize_and_store_learning"],
    },
    "ponder": {
        "tags": ["reasoning", "research"],
        "keywords": ["understand", "swot", "analyze", "tradeoffs", "reason"],
    },
    "search": {"tags": ["research"], "keywords": ["search", "web", "find"]},
    "coding": {"tags": ["coding"], "keywords": ["code", "coding", "implement"]},
    "plan": {"tags": ["plan"], "keywords": ["plan"]},
    "discuss": {"tags": ["discuss"], "keywords": ["discuss"]},
    "read": {"tags": ["read"], "keywords": ["read"]},
    "explainer": {"tags": ["explain"], "keywords": ["explain", "expound"]},
}


def _augment_catalog_with_synthetic_modes() -> None:
    """Add synthetic routable modes to the ambient workflow catalog.

    Idempotent: runs once per process; safe if the catalog was already
    loaded without the augmentation.
    """
    from kernel.ipsar_fsm import get_workflow_catalog
    from kernel.models import WorkflowModeConfig

    catalog = get_workflow_catalog()
    for mode_id, spec in _SYNTHETIC_TEST_MODES.items():
        if mode_id in catalog:
            continue
        authored_ops = list(spec.get("authored_ops", []))
        bound_ops = list(spec.get("bound_ops", []))
        # Keyword map: the spec's ``keywords`` list is flat (one pool per
        # mode), while ``routing_intent_keywords`` is keyed per tag. Attribute
        # the full pool to EVERY tag — intent detection is per-tag substring
        # matching over the pool, so sharing the pool keeps each tag able to
        # fire on its cue words.
        tags = list(spec["tags"])
        kw_pool = list(spec["keywords"])
        catalog[mode_id] = WorkflowModeConfig(
            mode_id=mode_id,
            start_stage_id="none",
            mode_kind="chat",
            routable=True,
            fork_allowed=True,
            satisfies_intents=sorted(set(tags) | {mode_id}),
            routing_intent_tags=tags,
            routing_intent_keywords={tag: list(kw_pool) for tag in tags},
            authored_allowed_operation_ids=authored_ops,
            allowed_operation_ids=authored_ops + bound_ops,
        )


@pytest.fixture(autouse=True, scope="module")
def _augmented_ambient_catalog(_mcp_catalog_setup):  # noqa: F811 — deliberate fixture dependency on the module-scope import
    """Augment the ambient test catalog with the synthetic target modes.

    Depends on ``_mcp_catalog_setup`` so the ambient catalog is loaded
    (and eagerly validated) BEFORE the augmentation runs. Individual tests
    that patch the catalog bindings override this per-test — their patch
    takes precedence while active.
    """
    _augment_catalog_with_synthetic_modes()
    yield
