"""Tests for the pack router-config discovery seam.

Covers:
- ``MODEL_ROUTER_CONFIGS_ENTRY_POINT_GROUP`` name is stable.
- ``discover_pack_router_config`` returns the first pack-declared config dict.
- ``discover_pack_router_config`` returns ``None`` when no pack declares one.
- ``discover_pack_router_config`` skips broken entry points and continues.
"""

from __future__ import annotations

import importlib.metadata
from typing import Any
from unittest.mock import patch

from kernel.model_router_config_discovery import (
    MODEL_ROUTER_CONFIGS_ENTRY_POINT_GROUP,
    discover_pack_router_config,
)


class _StubEntryPoint:
    """Minimal stand-in for ``importlib.metadata.EntryPoint``."""

    def __init__(
        self,
        name: str,
        target: Any,
        *,
        load_raises: bool = False,
        load_returns: Any = None,
    ) -> None:
        self.name = name
        self._target = target
        self._load_raises = load_raises
        self._load_returns = load_returns

    def load(self) -> Any:
        if self._load_raises:
            raise RuntimeError("boom")
        if self._load_returns is not None:
            return self._load_returns
        return self._target


def test_entry_point_group_name() -> None:
    assert MODEL_ROUTER_CONFIGS_ENTRY_POINT_GROUP == "lisa.extensions.model_router_configs"


def test_discover_returns_none_when_no_entries() -> None:
    """No registered entry points → None."""
    with patch.object(
        importlib.metadata,
        "entry_points",
        return_value=[],
    ):
        assert discover_pack_router_config() is None


def test_discover_returns_first_dict_config() -> None:
    """First entry point returning a dict wins."""
    config: dict[str, Any] = {
        "backend": "lisa-router",
        "rules": [{"when": "iteration > 1", "use": "gpt-5"}],
        "fallback": "gpt-5-mini",
    }
    eps = [_StubEntryPoint("lisa-router", lambda: config)]
    with patch.object(importlib.metadata, "entry_points", return_value=eps):
        result = discover_pack_router_config()
    assert result == config


def test_discover_skips_broken_entry_points() -> None:
    """A broken entry point is logged and skipped; the next one wins."""
    good_config: dict[str, Any] = {"backend": "my-router"}
    eps = [
        _StubEntryPoint("broken-router", None, load_raises=True),
        _StubEntryPoint("good-router", lambda: good_config),
    ]
    with patch.object(importlib.metadata, "entry_points", return_value=eps):
        result = discover_pack_router_config()
    assert result == good_config


def test_discover_skips_non_dict_returns() -> None:
    """An entry point returning a non-dict (e.g. a string) is skipped."""
    good_config: dict[str, Any] = {"backend": "my-router"}
    eps = [
        _StubEntryPoint("string-router", lambda: "not-a-dict"),
        _StubEntryPoint("good-router", lambda: good_config),
    ]
    with patch.object(importlib.metadata, "entry_points", return_value=eps):
        result = discover_pack_router_config()
    assert result == good_config


def test_discover_returns_none_when_all_non_dict() -> None:
    """When every entry point returns a non-dict, result is None."""
    eps = [_StubEntryPoint("string-router", lambda: "not-a-dict")]
    with patch.object(importlib.metadata, "entry_points", return_value=eps):
        assert discover_pack_router_config() is None


def test_discover_first_wins_does_not_consult_later() -> None:
    """Once a dict is found, later entry points are not consulted."""
    first_config: dict[str, Any] = {"backend": "first"}
    second_config: dict[str, Any] = {"backend": "second"}
    eps = [
        _StubEntryPoint("first", lambda: first_config),
        _StubEntryPoint("second", lambda: second_config),
    ]
    with patch.object(importlib.metadata, "entry_points", return_value=eps):
        result = discover_pack_router_config()
    assert result == first_config
