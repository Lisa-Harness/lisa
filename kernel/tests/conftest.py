"""Shared pytest fixtures for kernel tests.

All kernel tests that need the mode or operation catalog must use the
MCP-mediated catalog loading fixture from ``conftest_catalog.py``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

# Import the catalog setup fixture so it's auto-use across all kernel tests.
from kernel.tests.conftest_catalog import _mcp_catalog_setup  # noqa: F401


# ── Provider-gateway bridge for TurnHooks fakes ─────────────────────
#
# The kernel's turn loop no longer calls the ``TurnHooks.complete`` callable —
# every model call routes through ``ContextSeams.provider_gateways``. Kernel
# unit tests still construct ``TurnHooks(complete=...)`` fakes. Rather than
# touch 175 test sites, the autouse fixture below patches ``TurnHooks.__init__``
# so a non-None ``complete`` callable is ALSO wrapped as a
# ``StubGatewayProvider`` and set onto ``context_seams.provider_gateways`` (when
# the caller did not already supply one). Tests that assert the *absence* of a
# completion path (e.g. ``test_run_turn_requires_complete_hook``) pass
# ``complete=None`` and get no gateway, so the kernel's "no gateway wired"
# ``NotImplementedError`` still surfaces.
_ORIGINAL_TURNHOOKS_INIT: Any = None


@pytest.fixture(autouse=True)
def _bridge_turnhooks_complete_to_gateway() -> Any:
    """Route ``TurnHooks(complete=...)`` fakes through the provider gateway.

    Wraps the test's ``complete`` callable into a ``StubGatewayProvider`` so
    the kernel's gateway call-site resolves it. Preserves the caller's
    explicit ``context_seams`` if provided. Yields so the patch is active only
    during each test.
    """

    global _ORIGINAL_TURNHOOKS_INIT  # pylint: disable=global-statement

    from kernel.testing.gateway_stub import StubGatewayProvider
    from kernel.turn_runner import ContextSeams, TurnHooks

    if _ORIGINAL_TURNHOOKS_INIT is None:
        _ORIGINAL_TURNHOOKS_INIT = TurnHooks.__init__

    def _patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        _ORIGINAL_TURNHOOKS_INIT(self, *args, **kwargs)
        if getattr(self, "context_seams", None) is None:
            # Should not happen (default_factory guarantees it), but be safe.
            self.context_seams = ContextSeams()
        complete_fn = getattr(self, "complete", None)
        if (
            complete_fn is not None
            and getattr(self.context_seams, "provider_gateways", None) is None
        ):
            self.context_seams = replace(
                self.context_seams,
                provider_gateways=StubGatewayProvider(
                    complete_fn,
                    complete_stream_fn=getattr(self, "complete_stream", None),
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

    fixtures_root = (Path(__file__).resolve().parents[1] / "testing" / "fixtures").resolve()
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
    from kernel.routing_vocabulary import (
        register_routing_vocabulary,
        reset_routing_vocabulary,
    )

    reset_routing_vocabulary()
    register_routing_vocabulary(
        sequential_connectors=["then", "and then", "after that", "next", "followed by"],
        producer_intents=["research", "gather", "learn", "read", "analyze", "review", "architect"],
        consumer_intents=["code", "write", "implement", "review"],
        resume_phrases=["proceed", "continue", "go ahead", "go for it", "do it", "carry on", "pick up", "resume", "next step", "keep going"],
    )
    yield
    reset_routing_vocabulary()
    reset_routing_vocabulary()
