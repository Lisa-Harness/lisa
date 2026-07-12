from __future__ import annotations

import httpx
from pydantic import BaseModel

from kernel.intake.connectors import SignalCollector
from kernel.testing import extension_harness


class _FakeEntryPoint:
    def __init__(self, *, name: str, value: str, factory: object) -> None:
        self.name = name
        self.value = value
        self.module = value.split(":", 1)[0]
        self.attr = value.split(":", 1)[1] if ":" in value else ""
        self._factory = factory

    def load(self):
        return self._factory


class _HelloWorldSnapshot(BaseModel):
    collected_at: str
    status: str = "ok"
    message: str = "hello world"


class _HelloWorldCollector(SignalCollector[_HelloWorldSnapshot]):
    collector_id = "hello_world"
    display_name = "Hello World"

    async def collect(
        self,
        client: httpx.AsyncClient,
        collected_at: str,
    ) -> _HelloWorldSnapshot:
        del client
        return _HelloWorldSnapshot(collected_at=collected_at, status="ok", message="hello world")

    def build_unconfigured_snapshot(self, collected_at: str) -> _HelloWorldSnapshot:
        return _HelloWorldSnapshot(
            collected_at=collected_at,
            status="unconfigured",
            message="missing configuration",
        )

    def build_error_snapshot(self, collected_at: str) -> _HelloWorldSnapshot:
        return _HelloWorldSnapshot(
            collected_at=collected_at,
            status="error",
            message="collector failed",
        )


def test_list_extension_entries_reports_signal_collector_metadata(monkeypatch) -> None:
    fake_entry_point = _FakeEntryPoint(
        name="hello_world",
        value="sdk.packs.hello_world.collector:HelloWorldSignalCollector",
        factory=_HelloWorldCollector,
    )
    monkeypatch.setattr(
        extension_harness,
        "entry_points",
        lambda group: [fake_entry_point] if group == "lisa.extensions.signal_connectors" else [],
    )

    entries = extension_harness.list_extension_entries()

    assert [entry.entry_id for entry in entries] == ["hello_world"]
    assert entries[0].kind == "collector"
    assert (
        entries[0].import_path == "sdk.packs.hello_world.collector:HelloWorldSignalCollector"
    )


def test_load_signal_collector_and_build_contract_preview(monkeypatch) -> None:
    fake_entry_point = _FakeEntryPoint(
        name="hello_world",
        value="sdk.packs.hello_world.collector:HelloWorldSignalCollector",
        factory=_HelloWorldCollector,
    )
    monkeypatch.setattr(
        extension_harness,
        "entry_points",
        lambda group: [fake_entry_point] if group == "lisa.extensions.signal_connectors" else [],
    )

    collector = extension_harness.load_signal_collector("hello_world")
    preview = extension_harness.build_collector_contract_preview(
        collector,
        collected_at="2026-05-29T00:00:00+00:00",
    )

    assert collector.collector_id == "hello_world"
    assert preview["collector_id"] == "hello_world"
    assert preview["unconfigured_snapshot"]["status"] == "unconfigured"
    assert preview["error_snapshot"]["status"] == "error"
