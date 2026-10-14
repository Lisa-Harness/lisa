"""Test that every registered endpoint has a producer and a consumer.

This test is the documentation proxy for the public API. If an endpoint is
orphaned (no producer in any installed pyproject.toml or sibling pack
checkout, or no consumer in the codebase), the test fails loudly. External
consumers can't be surveyed, but the registry + test proves every contract
has at least one producer and one consumer in the reference implementation.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from pathlib import Path

import pytest
from kernel.testing.endpoint_registry import REGISTRY as ENDPOINTS

# Sibling pack checkouts (public + WIP). Several public endpoints are
# produced ONLY by packs that are deliberately not dev dependencies
# (the ext/wip extras were removed in 40067bd), so the installed-entry-point
# scan alone fails on every clean dev venv. A producer declared in a
# sibling checkout's pyproject.toml counts exactly like an installed one.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SIBLING_PACK_ROOTS = (
    _REPO_ROOT.parent / "lisa-packs",
    _REPO_ROOT.parent / "lisa-packs-wip",
)


def _declared_entry_point_groups(pyproject: Path) -> set[str]:
    """Return the entry-point group names declared in one pyproject.toml.

    Parses only the ``[project.entry-points."<group>"]`` section headers —
    enough to prove a producer declaration exists without a TOML dependency
    in the kernel test env.
    """
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return set()
    groups: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[project.entry-points.") and stripped.endswith("]"):
            group = stripped[len("[project.entry-points.") : -1].strip().strip('"')
            if group:
                groups.add(group)
    return groups


def _sibling_declared_groups() -> set[str]:
    """Union of entry-point groups declared across sibling pack checkouts."""
    declared: set[str] = set()
    for packs_root in _SIBLING_PACK_ROOTS:
        if not packs_root.is_dir():
            continue
        for pyproject in sorted(packs_root.glob("*/pyproject.toml")):
            declared |= _declared_entry_point_groups(pyproject)
    return declared


def _has_producer(group: str) -> bool:
    """Return True if at least one pack declares an entry point in this group.

    Producers count from either installed entry points (venv) or sibling
    checkout pyprojects (source-tree checkouts without a venv install).
    """
    try:
        eps = importlib.metadata.entry_points(group=group)
        if len(list(eps)) > 0:
            return True
    except Exception:  # pylint: disable=broad-except
        pass
    return group in _sibling_declared_groups()


def _producer_source(group: str) -> str:
    """Return a human-readable source description for the producer scan."""
    try:
        eps = importlib.metadata.entry_points(group=group)
        if len(list(eps)) > 0:
            return "installed entry points"
    except Exception:  # pylint: disable=broad-except
        pass
    if group in _sibling_declared_groups():
        return "sibling pack checkout"
    return "no known producer source"


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
    """Every registered endpoint must have at least one pack producing it.

    ``known_orphan`` endpoints are exempt: the registry documents them as
    intentionally producer-less (live consumer, intended surface) instead
    of failing the suite.
    """
    if endpoint.known_orphan:
        pytest.skip(f"{endpoint.group}: registered known orphan (live consumer, no producer yet)")
    assert _has_producer(endpoint.group), (
        f"{endpoint.group}: no producers declared in any installed pyproject.toml "
        f"or sibling pack checkout ({_producer_source(endpoint.group)}). "
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
