"""Host-wide, content-addressed run manifests (evidence envelope spec §3).

A turn's behaviour is a function of code and content, not only its input, so
replaying turn 900 under today's packs has to be able to say what changed.
This module builds one deterministic manifest describing the effective run
configuration and stores it in the existing result-object store.

The store is the thread result-object store, unchanged: it derives
``<base>.results/`` from whatever base path it is given and content-addresses
the payload as ``r<sha256>``. Nothing requires that base to be a thread file —
:func:`sdk.thread.result_object_path` never opens it — so a host-wide store is
simply a different base, ``<data-dir>/run``, yielding
``<data-dir>/run.results/r<sha256>.json``. The manifest is run configuration,
not thread data, so it must not be written through a thread base: keeping it
out of ``threads/`` keeps it out of per-thread archives and redaction scope.

Determinism is the whole point of content addressing here. The manifest holds
no timestamps, filesystem paths, environment values, or credentials, and every
collection is sorted, so identical effective configuration produces identical
bytes and therefore an identical ``run_ref``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

#: Distribution names whose installed versions describe the running framework.
#: Keyed by the manifest field name the spec names.
_LISA_DISTRIBUTIONS: dict[str, str] = {
    "kernel": "lisa-harness-kernel",
    "mcp_server": "lisa-harness-mcp-servers",
    "runtime": "lisa-harness-runtime",
    "sdk": "lisa-harness-sdk",
}

#: Entry-point group a pack registers its content roots under.
_PACK_ENTRY_POINT_GROUP = "lisa.extensions.content_roots"

#: Manifest schema revision, so a future field addition is detectable.
RUN_MANIFEST_VERSION = 1


def _installed_version(distribution_name: str) -> str | None:
    """Return an installed distribution's version, or ``None`` when absent.

    ``None`` rather than a guessed value: an absent distribution is unknown,
    never "version 0", and the manifest records that honestly.
    """
    try:
        return importlib_metadata.version(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _hash_text(text: str) -> str:
    """Return the hex SHA-256 of ``text``, for content-addressing config bytes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    """Serialize deterministically: sorted keys, no incidental whitespace."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def config_digest(value: object) -> str:
    """Return a stable content hash for a resolved configuration value.

    Used for the manifest's ``config`` block and for ``mode_digest``, which is
    the content hash of the resolved mode definition a turn actually ran
    (after overrides), not of the whole manifest.
    """
    return _hash_text(_canonical_json(value))


def config_digest_of(value: object) -> str | None:
    """Return the content hash of a resolved configuration value, or ``None``.

    A resolved configuration is usually a pydantic model (the execution
    policy, the session routing block), and ``json.dumps`` cannot serialize a
    model directly. ``model_dump(mode="json")`` normalizes it to plain JSON
    types first, so the hash covers the value's *content* rather than its
    class. Normalizing once here is what keeps the manifest able to tell two
    hosts apart when they differ only in policy or routing.

    The resolved model manifest is a frozen dataclass rather than a pydantic
    model, so it is normalized with ``dataclasses.asdict``. Both normalizations
    produce plain JSON types, which is what the hash covers.

    ``None`` means the value could not be serialized at all. An unknown hash
    is absent from the manifest; it is never a hash of a placeholder, because
    a placeholder hash would assert a provenance the host does not have.
    """
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            value = dump(mode="json")
        except Exception:  # pylint: disable=broad-except
            return None
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    try:
        return config_digest(value)
    except (TypeError, ValueError):
        return None


def _installed_packs() -> list[dict[str, Any]]:
    """Return one identity record per installed pack, sorted by name.

    Identity is name, version, and a hash of the pack's declared content
    roots — never the install location, which would make the manifest
    host-specific and therefore non-reproducible.
    """
    packs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry_point in importlib_metadata.entry_points(group=_PACK_ENTRY_POINT_GROUP):
        distribution = getattr(entry_point, "dist", None)
        name = getattr(distribution, "name", None) or entry_point.name
        if not isinstance(name, str) or not name or name in seen:
            continue
        seen.add(name)
        version = getattr(distribution, "version", None)
        packs.append(
            {
                "name": name,
                "version": version if isinstance(version, str) and version else None,
                # The entry point value names the module providing the roots;
                # hashing it identifies the pack's content wiring without
                # recording a filesystem path.
                "content_hash": _hash_text(f"{entry_point.name}:{entry_point.value}"),
            }
        )
    return sorted(packs, key=lambda pack: str(pack["name"]))


@dataclass(frozen=True, slots=True)
class RunManifest:
    """The deterministic description of one host run configuration.

    ``config`` holds content hashes of the resolved mode, policy, and router
    configuration; the raw resolved values are hashed in the builder and never
    stored, so the manifest cannot carry user data or configuration content.
    """

    lisa: dict[str, str | None] = field(default_factory=dict)
    packs: list[dict[str, Any]] = field(default_factory=list)
    config: dict[str, str] = field(default_factory=dict)
    model_data_version: str | None = None
    host: dict[str, str | None] = field(default_factory=dict)
    python: str = ""

    def to_document(self) -> dict[str, Any]:
        """Return the manifest as the ordered JSON document that gets stored."""
        return {
            "manifest_version": RUN_MANIFEST_VERSION,
            "lisa": dict(sorted(self.lisa.items())),
            "packs": list(self.packs),
            "config": dict(sorted(self.config.items())),
            "model_data_version": self.model_data_version,
            "host": dict(sorted(self.host.items())),
            "python": self.python,
        }

    def to_stored_record(self) -> dict[str, Any]:
        """Return the record to persist, shaped for the result-object store.

        ``load_result_object`` requires ``content`` to be a *string*; a nested
        object round-trips into a write-only object that no reader can load
        back (``Invalid result-object content``). The manifest is therefore
        stored as canonical JSON text, which keeps the bytes deterministic and
        therefore keeps ``run_ref`` stable for an unchanged configuration.
        """
        return {"content": _canonical_json(self.to_document())}


def build_run_manifest(
    *,
    config_hashes: dict[str, str],
    model_data_version: str | None,
    host_type: str | None,
    host_protocol_version: str | None,
) -> RunManifest:
    """Build the manifest from already-resolved, already-hashed inputs.

    Callers pass content hashes and identifiers, never paths or raw
    configuration, so this function cannot leak either one into the store.
    """
    return RunManifest(
        lisa={
            field_name: _installed_version(dist) for field_name, dist in _LISA_DISTRIBUTIONS.items()
        },
        packs=_installed_packs(),
        config=dict(sorted(config_hashes.items())),
        model_data_version=model_data_version,
        host={"type": host_type, "protocol_version": host_protocol_version},
        python=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )


def run_base_path(data_dir: Path) -> Path:
    """Return the result-object store base for host-wide run manifests.

    This path is a *naming root*, not a file: the store appends ``.results``
    to it, producing ``<data-dir>/run.results/``. It deliberately does not
    collide with ``HostPaths.run_dir()``, which owns the per-session
    ``<data-dir>/run/<session_key>`` tree.
    """
    return data_dir / "run"


__all__ = [
    "RUN_MANIFEST_VERSION",
    "RunManifest",
    "build_run_manifest",
    "config_digest",
    "run_base_path",
]
