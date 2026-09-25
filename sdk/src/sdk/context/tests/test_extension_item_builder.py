"""Tests for disabled-pack exclusion in the extension item builder."""

from __future__ import annotations

from sdk.context.extension_item_builder import EntryPointExtensionItemBuilder


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


def test_extension_item_builder_excludes_disabled_packs(monkeypatch) -> None:
    """Entry points whose distribution name is in excluded_packs are skipped."""

    def active_fn(_ctx, _state):
        return [{"source": "active"}]

    def disabled_fn(_ctx, _state):
        return [{"source": "disabled"}]

    monkeypatch.setattr(
        "sdk.context.extension_item_builder.entry_points",
        lambda group: [
            _FakeEntryPoint(name="active", factory=active_fn, dist_name="lisa-pack-active"),
            _FakeEntryPoint(
                name="disabled",
                factory=disabled_fn,
                dist_name="lisa-pack-disabled",
            ),
        ],
    )

    builder = EntryPointExtensionItemBuilder(
        excluded_packs={"lisa-pack-disabled"},
    )
    result = builder.build_extension_context_items(runtime_context=None, state=None)
    assert len(result) == 1
    assert result[0]["source"] == "active"
    assert result[0]["pack"] == "lisa-pack-active"


def test_extension_item_builder_none_returns_all(monkeypatch) -> None:
    """excluded_packs=None preserves backward-compatible behavior."""

    def active_fn(_ctx, _state):
        return [{"source": "active"}]

    def other_fn(_ctx, _state):
        return [{"source": "other"}]

    monkeypatch.setattr(
        "sdk.context.extension_item_builder.entry_points",
        lambda group: [
            _FakeEntryPoint(name="active", factory=active_fn, dist_name="lisa-pack-active"),
            _FakeEntryPoint(name="other", factory=other_fn, dist_name="lisa-pack-other"),
        ],
    )

    builder = EntryPointExtensionItemBuilder(excluded_packs=None)
    result = builder.build_extension_context_items(runtime_context=None, state=None)
    assert any(r["source"] == "active" for r in result)
    assert any(r["source"] == "other" for r in result)
    # Each record should be tagged with its source pack name.
    assert any(r.get("pack") == "lisa-pack-active" for r in result)
    assert any(r.get("pack") == "lisa-pack-other" for r in result)


def test_extension_item_builder_empty_exclusion_returns_all(monkeypatch) -> None:
    """An empty exclusion set is equivalent to no filtering."""

    def fn(_ctx, _state):
        return [{"source": "test"}]

    monkeypatch.setattr(
        "sdk.context.extension_item_builder.entry_points",
        lambda group: [
            _FakeEntryPoint(name="test", factory=fn, dist_name="lisa-pack-test"),
        ],
    )

    builder = EntryPointExtensionItemBuilder(excluded_packs=set())
    result = builder.build_extension_context_items(runtime_context=None, state=None)
    assert len(result) == 1
    assert result[0]["source"] == "test"
    assert result[0]["pack"] == "lisa-pack-test"


def test_extension_item_builder_tags_records_with_pack_name(monkeypatch) -> None:
    """Each item is tagged with its source distribution package name."""
    from kernel.models import ContextItem

    # Return a ContextItem with no pack field set — the builder should
    # populate it from the entry point's distribution name.
    def factory(_ctx, _state):
        return [
            ContextItem(
                item_id="collector:test",
                source="test",
                evidence_type="connector_status",
                title="Test",
                body="OK",
                collected_at="2026-01-01T00:00:00Z",
            )
        ]

    monkeypatch.setattr(
        "sdk.context.extension_item_builder.entry_points",
        lambda group: [
            _FakeEntryPoint(name="test", factory=factory, dist_name="lisa-pack-test"),
        ],
    )

    builder = EntryPointExtensionItemBuilder(excluded_packs=None)
    result = builder.build_extension_context_items(runtime_context=None, state=None)
    assert len(result) == 1
    assert result[0].pack == "lisa-pack-test"


def test_extension_item_builder_does_not_overwrite_explicit_pack(monkeypatch) -> None:
    """When the item already has a pack, the builder does not overwrite it."""
    from kernel.models import ContextItem

    def factory(_ctx, _state):
        return [
            ContextItem(
                item_id="collector:custom",
                source="custom",
                evidence_type="connector_status",
                title="Custom",
                body="OK",
                collected_at="2026-01-01T00:00:00Z",
                pack="lisa-pack-explicit",
            )
        ]

    monkeypatch.setattr(
        "sdk.context.extension_item_builder.entry_points",
        lambda group: [
            _FakeEntryPoint(name="custom", factory=factory, dist_name="lisa-pack-other"),
        ],
    )

    builder = EntryPointExtensionItemBuilder(excluded_packs=None)
    result = builder.build_extension_context_items(runtime_context=None, state=None)
    assert len(result) == 1
    assert result[0].pack == "lisa-pack-explicit"
