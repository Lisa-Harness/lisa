"""Tests for the run manifest's configuration content hashes.

The manifest's ``config`` block is the only thing that lets a reader tell two
hosts apart when they differ only in configuration. It is built from content
hashes of the RESOLVED policy, routing, and model manifest — values that are
pydantic models and frozen dataclasses, not plain JSON.

Hashing a model directly raised ``TypeError`` (``json.dumps`` cannot
serialize it), and the failure was swallowed at debug level, so a live
manifest recorded only the session-plan digest: two hosts differing only in
execution policy hashed identically. These tests pin the normalization that
fixes it.
"""

from __future__ import annotations

from kernel.execution_policy import ExecutionPolicy
from kernel.session_plan import SessionRouting
from runtime.run_manifest import config_digest, config_digest_of


def test_execution_policy_hashes_to_a_stable_value() -> None:
    """A resolved policy is hashable, and the same policy hashes the same."""
    policy = ExecutionPolicy(max_model_calls=12)
    digest = config_digest_of(policy)
    assert digest is not None
    assert digest == config_digest_of(ExecutionPolicy(max_model_calls=12))


def test_two_hosts_differing_only_in_policy_hash_differently() -> None:
    """The manifest's whole purpose: policy differences must be visible.

    Before the fix, both policies raised ``TypeError`` inside the hash, so
    neither was recorded and the two hosts produced the same manifest.
    """
    strict = config_digest_of(ExecutionPolicy(max_model_calls=12))
    lax = config_digest_of(ExecutionPolicy(max_model_calls=99))
    assert strict is not None
    assert lax is not None
    assert strict != lax


def test_two_hosts_differing_only_in_routing_hash_differently() -> None:
    """Routing differences must be visible for the same reason."""
    one = config_digest_of(SessionRouting(start_mode="default"))
    two = config_digest_of(SessionRouting(start_mode="research"))
    assert one is not None
    assert two is not None
    assert one != two


def test_a_plain_value_hashes_the_same_as_before() -> None:
    """Plain JSON values still hash through the same canonical form."""
    assert config_digest_of({"b": 2, "a": 1}) == config_digest({"a": 1, "b": 2})


def test_an_unserializable_value_is_unknown_not_a_placeholder() -> None:
    """An unhashable value yields no hash at all, never a hash of nothing.

    The manifest must never carry a placeholder hash: that would assert a
    provenance the host does not have.
    """
    assert config_digest_of(object()) is None
