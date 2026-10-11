"""Model-sized tool-result residency budget (2026-10-09 turn-622)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from kernel.config import settings
from kernel.context.context_budget import model_context_window_tokens


def test_declared_window_sizes_the_budget() -> None:
    model = SimpleNamespace(context_window=400_000)
    assert model_context_window_tokens(model) == 400_000


@pytest.mark.parametrize("model", [None, SimpleNamespace(), SimpleNamespace(context_window=0)])
def test_undeclared_window_uses_the_shared_default(model: object) -> None:
    assert model_context_window_tokens(model) == settings.context_default_window_tokens


def test_small_window_gets_a_small_budget() -> None:
    assert model_context_window_tokens(SimpleNamespace(context_window=8192)) == 8192
