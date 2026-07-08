from __future__ import annotations

import httpx
from pydantic import BaseModel

from kernel.intake.connectors import SignalConnector
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


class _HelloWorldConnector(SignalConnector[_HelloWorldSnapshot]):
    connector_id = "hello_world"
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
            message="connector failed",
        )


def test_list_extension_entries_reports_signal_connector_metadata(monkeypatch) -> None:
    fake_entry_point = _FakeEntryPoint(
        name="hello_world",
        value="sdk_extensions.hello_world.connector:HelloWorldSignalConnector",
        factory=_HelloWorldConnector,
    )
    monkeypatch.setattr(
        extension_harness,
        "entry_points",
        lambda group: [fake_entry_point] if group == "lisa.extensions.signal_connectors" else [],
    )

    entries = extension_harness.list_extension_entries()

    assert [entry.entry_id for entry in entries] == ["hello_world"]
    assert entries[0].kind == "connector"
    assert (
        entries[0].import_path == "sdk_extensions.hello_world.connector:HelloWorldSignalConnector"
    )


def test_load_signal_connector_and_build_contract_preview(monkeypatch) -> None:
    fake_entry_point = _FakeEntryPoint(
        name="hello_world",
        value="sdk_extensions.hello_world.connector:HelloWorldSignalConnector",
        factory=_HelloWorldConnector,
    )
    monkeypatch.setattr(
        extension_harness,
        "entry_points",
        lambda group: [fake_entry_point] if group == "lisa.extensions.signal_connectors" else [],
    )

    connector = extension_harness.load_signal_connector("hello_world")
    preview = extension_harness.build_connector_contract_preview(
        connector,
        collected_at="2026-05-29T00:00:00+00:00",
    )

    assert connector.connector_id == "hello_world"
    assert preview["connector_id"] == "hello_world"
    assert preview["unconfigured_snapshot"]["status"] == "unconfigured"
    assert preview["error_snapshot"]["status"] == "error"
