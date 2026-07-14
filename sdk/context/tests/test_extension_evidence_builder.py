"""Tests for disabled-pack exclusion in the extension evidence builder."""

from __future__ import annotations

from importlib.metadata import EntryPoint

from sdk.context.extension_evidence_builder import EntryPointExtensionEvidenceBuilder


class _FakeEntryPoint:
    def __init__(self, *, name: str, factory: object, dist_name: str | None = None) -> None:
        self.name = name
        self._factory = factory

        class _FakeDist:
            def __init__(self, dname: str) -> None:
                self.metadata = {"Name": dname}

        self.dist = _FakeDist(dist_name) if dist_name else None

    def load(self):
        return self._factory


def test_extension_evidence_builder_excludes_disabled_packs(monkeypatch) -> None:
    """Entry points whose distribution name is in excluded_packs are skipped."""
    active_fn = lambda _ctx, _state: [{"source": "active"}]
    disabled_fn = lambda _ctx, _state: [{"source": "disabled"}]

    monkeypatch.setattr(
        "sdk.context.extension_evidence_builder.entry_points",
        lambda group: [
            _FakeEntryPoint(name="active", factory=active_fn, dist_name="lisa-pack-active"),
            _FakeEntryPoint(
                name="disabled",
                factory=disabled_fn,
                dist_name="lisa-pack-disabled",
            ),
        ],
    )

    builder = EntryPointExtensionEvidenceBuilder(
        excluded_packs={"lisa-pack-disabled"},
    )
    result = builder.build_extension_context_evidence(runtime_context=None, state=None)
    assert len(result) == 1
    assert result[0]["source"] == "active"
    assert result[0]["pack"] == "lisa-pack-active"


def test_extension_evidence_builder_none_returns_all(monkeypatch) -> None:
    """excluded_packs=None preserves backward-compatible behavior."""
    active_fn = lambda _ctx, _state: [{"source": "active"}]
    other_fn = lambda _ctx, _state: [{"source": "other"}]

    monkeypatch.setattr(
        "sdk.context.extension_evidence_builder.entry_points",
        lambda group: [
            _FakeEntryPoint(name="active", factory=active_fn, dist_name="lisa-pack-active"),
            _FakeEntryPoint(name="other", factory=other_fn, dist_name="lisa-pack-other"),
        ],
    )

    builder = EntryPointExtensionEvidenceBuilder(excluded_packs=None)
    result = builder.build_extension_context_evidence(runtime_context=None, state=None)
    assert any(r["source"] == "active" for r in result)
    assert any(r["source"] == "other" for r in result)
    # Each record should be tagged with its source pack name.
    assert any(r.get("pack") == "lisa-pack-active" for r in result)
    assert any(r.get("pack") == "lisa-pack-other" for r in result)


def test_extension_evidence_builder_empty_exclusion_returns_all(monkeypatch) -> None:
    """An empty exclusion set is equivalent to no filtering."""
    fn = lambda _ctx, _state: [{"source": "test"}]

    monkeypatch.setattr(
        "sdk.context.extension_evidence_builder.entry_points",
        lambda group: [
            _FakeEntryPoint(name="test", factory=fn, dist_name="lisa-pack-test"),
        ],
    )

    builder = EntryPointExtensionEvidenceBuilder(excluded_packs=set())
    result = builder.build_extension_context_evidence(runtime_context=None, state=None)
    assert len(result) == 1
    assert result[0]["source"] == "test"
    assert result[0]["pack"] == "lisa-pack-test"


def test_extension_evidence_builder_tags_records_with_pack_name(monkeypatch) -> None:
    """Each evidence record is tagged with its source distribution package name."""
    from kernel.models import ContextEvidence

    # Return a ContextEvidence with no pack field set — the builder should
    # populate it from the entry point's distribution name.
    def factory(_ctx, _state):
        return [
            ContextEvidence(
                evidence_id="collector:test",
                source="test",
                evidence_type="connector_status",
                title="Test",
                body="OK",
                collected_at="2026-01-01T00:00:00Z",
            )
        ]

    monkeypatch.setattr(
        "sdk.context.extension_evidence_builder.entry_points",
        lambda group: [
            _FakeEntryPoint(name="test", factory=factory, dist_name="lisa-pack-test"),
        ],
    )

    builder = EntryPointExtensionEvidenceBuilder(excluded_packs=None)
    result = builder.build_extension_context_evidence(runtime_context=None, state=None)
    assert len(result) == 1
    assert result[0].pack == "lisa-pack-test"


def test_extension_evidence_builder_does_not_overwrite_explicit_pack(monkeypatch) -> None:
    """When the evidence already has a pack, the builder does not overwrite it."""
    from kernel.models import ContextEvidence

    def factory(_ctx, _state):
        return [
            ContextEvidence(
                evidence_id="collector:custom",
                source="custom",
                evidence_type="connector_status",
                title="Custom",
                body="OK",
                collected_at="2026-01-01T00:00:00Z",
                pack="lisa-pack-explicit",
            )
        ]

    monkeypatch.setattr(
        "sdk.context.extension_evidence_builder.entry_points",
        lambda group: [
            _FakeEntryPoint(name="custom", factory=factory, dist_name="lisa-pack-other"),
        ],
    )

    builder = EntryPointExtensionEvidenceBuilder(excluded_packs=None)
    result = builder.build_extension_context_evidence(runtime_context=None, state=None)
    assert len(result) == 1
    assert result[0].pack == "lisa-pack-explicit"