"""Tests for the session-injected routing-vocabulary registry."""

from __future__ import annotations

import pytest

from kernel.routing_vocabulary import (
    DEFAULT_SEQUENTIAL_CONNECTORS,
    get_sequential_connectors,
    register_routing_vocabulary,
    reset_routing_vocabulary,
)


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    """Reset the registry before and after each test."""
    reset_routing_vocabulary()
    yield
    reset_routing_vocabulary()


def test_get_sequential_connectors_returns_default_when_unwired() -> None:
    """With no session vocabulary registered, the historical default is used."""
    assert get_sequential_connectors() == DEFAULT_SEQUENTIAL_CONNECTORS


def test_register_routing_vocabulary_replaces_default() -> None:
    """A registered session vocabulary replaces the default list."""
    register_routing_vocabulary(sequential_connectors=["then", "based on"])
    connectors = get_sequential_connectors()
    assert " then " in connectors
    assert " based on " in connectors
    # The historical default is NOT mixed in — the session list is authoritative.
    assert " followed by " not in connectors


def test_register_routing_vocabulary_normalizes_entries() -> None:
    """Entries are stripped, lowercased, padded, de-duplicated."""
    register_routing_vocabulary(sequential_connectors=["  Then  ", "BASED ON", "then", "", "  "])
    assert get_sequential_connectors() == (" then ", " based on ")


def test_register_routing_vocabulary_none_resets_to_default() -> None:
    """Passing ``None`` (or empty) resets to the historical default."""
    register_routing_vocabulary(sequential_connectors=["based on"])
    assert " based on " in get_sequential_connectors()

    register_routing_vocabulary(sequential_connectors=None)
    assert get_sequential_connectors() == DEFAULT_SEQUENTIAL_CONNECTORS

    register_routing_vocabulary(sequential_connectors=["based on"])
    register_routing_vocabulary(sequential_connectors=[])
    assert get_sequential_connectors() == DEFAULT_SEQUENTIAL_CONNECTORS


def test_reset_routing_vocabulary_clears_session_list() -> None:
    """``reset_routing_vocabulary`` returns to the default list."""
    register_routing_vocabulary(sequential_connectors=["based on"])
    reset_routing_vocabulary()
    assert get_sequential_connectors() == DEFAULT_SEQUENTIAL_CONNECTORS
