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
    assert result == [{"source": "active"}]


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
    assert {"source": "active"} in result
    assert {"source": "other"} in result


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
    assert result == [{"source": "test"}]