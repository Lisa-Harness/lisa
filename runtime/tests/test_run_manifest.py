"""Tests for deterministic, host-wide run manifests (evidence envelope §3)."""

from __future__ import annotations

import json
from pathlib import Path

from runtime.run_manifest import (
    RunManifest,
    build_run_manifest,
    config_digest,
    config_digest_of,
    run_base_path,
)
from runtime.thread._results import _persist_result
from sdk.thread import load_result_object


def _manifest(**overrides: object) -> RunManifest:
    """Build a manifest with stable inputs, overridable per test."""
    kwargs: dict[str, object] = {
        "config_hashes": {"routing": "a" * 64, "execution_policy": "b" * 64},
        "model_data_version": "2026-11-01",
        "host_type": "cli",
        "host_protocol_version": None,
    }
    kwargs.update(overrides)
    return build_run_manifest(**kwargs)  # type: ignore[arg-type]


def test_same_configuration_produces_identical_bytes() -> None:
    """Content addressing only works if the document is order-stable."""
    first = _manifest().to_document()
    second = _manifest().to_document()
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_changed_config_hash_changes_the_document() -> None:
    """A pack or policy reload must move the address, or replay cannot tell."""
    before = _manifest().to_document()
    after = _manifest(config_hashes={"routing": "c" * 64, "execution_policy": "b" * 64})
    assert before != after.to_document()


def test_manifest_carries_no_paths_environment_or_credentials() -> None:
    """The manifest is host-wide shared state; it must stay non-identifying."""
    serialized = json.dumps(_manifest().to_document())
    assert "/home/" not in serialized
    assert "\\\\Users" not in serialized
    assert "api_key" not in serialized.lower()
    assert "token" not in serialized.lower()


def test_manifest_records_every_spec_field() -> None:
    """The §3 field list is the contract; a missing block is a silent gap."""
    document = _manifest().to_document()
    for name in ("lisa", "packs", "config", "model_data_version", "host", "python"):
        assert name in document, f"manifest omits {name}"
    assert set(document["lisa"]) == {"kernel", "runtime", "sdk", "mcp_server"}
    assert document["host"] == {"type": "cli", "protocol_version": None}


def test_run_manifest_is_stored_host_wide_and_content_addressed(tmp_path: Path) -> None:
    """The manifest lives in the run store, not in any thread's store."""
    base = run_base_path(tmp_path)
    manifest = _manifest()
    ref = _persist_result(base, manifest.to_stored_record())
    assert ref.startswith("r") and len(ref) == 65
    # The store derives ``<base>.results``; it never creates ``base`` itself.
    assert (tmp_path / "run.results" / f"{ref}.json").is_file()
    assert not base.exists()
    # Stored as canonical JSON text: ``load_result_object`` requires a string
    # ``content``, so a nested object would be write-only and unreadable.
    assert json.loads(load_result_object(base, ref)["content"]) == manifest.to_document()


def test_restoring_identical_manifest_reuses_one_ref(tmp_path: Path) -> None:
    """Rebuilding on startup must not accumulate duplicate objects."""
    base = run_base_path(tmp_path)
    first = _persist_result(base, _manifest().to_stored_record())
    second = _persist_result(base, _manifest().to_stored_record())
    assert first == second
    assert len(list((tmp_path / "run.results").glob("*.json"))) == 1


def test_config_digest_ignores_key_order() -> None:
    """Resolved config is a mapping; iteration order is never meaningful."""
    assert config_digest({"a": 1, "b": 2}) == config_digest({"b": 2, "a": 1})
    assert config_digest({"a": 1}) != config_digest({"a": 2})


def test_config_digest_of_normalizes_models_and_dataclasses() -> None:
    class _Model:
        def model_dump(self, *, mode: str) -> dict:
            assert mode == "json"
            return {"a": 1}

    from dataclasses import dataclass

    @dataclass
    class _Dataclass:
        b: int

    assert config_digest_of(_Model()) == config_digest({"a": 1})
    assert config_digest_of(_Dataclass(2)) == config_digest({"b": 2})


def test_config_digest_of_handles_dump_and_serialization_failures() -> None:
    class _BrokenDump:
        def model_dump(self, *, mode: str) -> dict:
            del mode
            raise RuntimeError("cannot normalize")

    class _Unserializable:
        pass

    assert config_digest_of(_BrokenDump()) is None
    assert config_digest_of(_Unserializable()) is None
