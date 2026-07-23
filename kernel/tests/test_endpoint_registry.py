"""Test that every registered endpoint has a producer and a consumer.

This test is the documentation proxy for the public API. If an endpoint is
orphaned (no producer in any installed pyproject.toml, or no consumer in
the codebase), the test fails loudly. External consumers can't be surveyed,
but the registry proves every contract has at least one producer and one
consumer in the reference implementation.
"""

from __future__ import annotations

import importlib
import importlib.metadata

import pytest

from kernel.testing.endpoint_registry import REGISTRY as ENDPOINTS


def _has_producer(group: str) -> bool:
    """Return True if at least one pack declares an entry point in this group."""
    try:
        eps = importlib.metadata.entry_points(group=group)
    except Exception:  # pylint: disable=broad-except
        return False
    return len(list(eps)) > 0


def _consumer_exists(consumer_site: str) -> bool:
    """Return True if the consumer site (module.attr) is importable."""
    if "." not in consumer_site:
        return False
    module_path, attr = consumer_site.rsplit(".", 1)
    try:
        mod = importlib.import_module(module_path)
    except Exception:  # pylint: disable=broad-except
        return False
    return hasattr(mod, attr)


def _contract_exists(contract_module: str, contract_name: str) -> bool:
    """Return True if the contract class/Protocol is importable."""
    try:
        mod = importlib.import_module(contract_module)
    except Exception:  # pylint: disable=broad-except
        return False
    return hasattr(mod, contract_name)


@pytest.mark.parametrize(
    "endpoint",
    ENDPOINTS,
    ids=[ep.group for ep in ENDPOINTS],
)
def test_endpoint_has_producer(endpoint) -> None:
    """Every registered endpoint must have at least one pack producing it."""
    assert _has_producer(endpoint.group), (
        f"{endpoint.group}: no producers declared in any installed pyproject.toml. "
        f"Either add an entry point to a pack, or remove the endpoint from the registry."
    )


@pytest.mark.parametrize(
    "endpoint",
    ENDPOINTS,
    ids=[ep.group for ep in ENDPOINTS],
)
def test_endpoint_has_consumer(endpoint) -> None:
    """Every registered endpoint must have a consumer site in the codebase."""
    assert _consumer_exists(endpoint.consumer_site), (
        f"{endpoint.group}: consumer site '{endpoint.consumer_site}' not found. "
        f"Wire the endpoint loader, or remove the endpoint from the registry."
    )


@pytest.mark.parametrize(
    "endpoint",
    ENDPOINTS,
    ids=[ep.group for ep in ENDPOINTS],
)
def test_endpoint_contract_exists(endpoint) -> None:
    """Every registered endpoint must have an importable contract class."""
    assert _contract_exists(endpoint.contract_module, endpoint.contract_name), (
        f"{endpoint.group}: contract '{endpoint.contract_module}.{endpoint.contract_name}' "
        f"not found. Create the contract, or remove the endpoint from the registry."
    )