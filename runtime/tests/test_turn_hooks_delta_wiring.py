"""Tests for streamed-delta wiring in ``build_turn_hooks``.

The kernel forwards each ``StreamedChunk`` through ``hooks.models.on_delta``
as it arrives; the runtime bundle must forward those deltas to the
transport's ``on_text_chunk(text, reasoning)``— and swallow transport
errors like stage events do, so a display bug never fails the turn.
Regression for the 2026-10-04 "spinner with nothing streaming" incident.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from runtime.testing import build_test_runtime
from runtime.turn_hooks_builder import build_turn_hooks


def _make_runtime(tmp_path: Path) -> Any:
    models_yaml = "\n".join(
        [
            "version: 1",
            "mode_models: {default: gemma3:latest}",
            "router:",
            "  backend: lisa-router",
            "  rules: []",
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
        streaming="hybrid",
    )


@pytest.mark.asyncio
async def test_on_delta_forwards_to_transport_text_chunk(tmp_path: Any) -> None:
    """Kernel model hooks get an ``on_delta`` that reaches the transport."""
    import asyncio

    runtime = _make_runtime(tmp_path)

    class _RecordingTransport:
        def __init__(self) -> None:
            self.chunks: list[tuple[str, str]] = []

        async def on_stage(self, label: str, data: Any = None) -> None: ...

        async def on_text_chunk(self, text: str, reasoning: str = "") -> None:
            await asyncio.sleep(0)
            self.chunks.append((text, reasoning))

        async def on_response(self, text: str, data: Any = None) -> None: ...

        async def on_error(self, message: str, exc: Any = None) -> None: ...

    transport = _RecordingTransport()
    hooks = build_turn_hooks(runtime, context=None, transport=transport)
    assert hooks.models.on_delta is not None
    await hooks.models.on_delta("answer text", "hidden thinking")
    assert transport.chunks == [("answer text", "hidden thinking")]
    await runtime.aclose()


@pytest.mark.asyncio
async def test_on_delta_swallows_transport_errors(tmp_path: Any) -> None:
    """A raising transport must not break the turn (same posture as stages)."""

    runtime = _make_runtime(tmp_path)

    class _BrokenTransport:
        async def on_stage(self, label: str, data: Any = None) -> None: ...

        async def on_text_chunk(self, text: str, reasoning: str = "") -> None:
            raise RuntimeError("display exploded")

        async def on_response(self, text: str, data: Any = None) -> None: ...

        async def on_error(self, message: str, exc: Any = None) -> None: ...

    hooks = build_turn_hooks(runtime, context=None, transport=_BrokenTransport())
    assert hooks.models.on_delta is not None
    await hooks.models.on_delta("text", "")  # must not raise
    await runtime.aclose()
