"""Slice C: shared-session warm-readiness inventory (counts/identifiers only)."""

from __future__ import annotations

import asyncio

import kernel.context._shared_stdio as shared_stdio


def test_inventory_empty_without_sessions() -> None:
    assert shared_stdio.shared_session_inventory() == []


def test_inventory_filters_by_epoch_and_reports_command_and_established() -> None:
    loop = asyncio.new_event_loop()
    try:
        # Constructed directly (no entry_for): the inventory test needs no
        # running loop, owner task, or session subprocess.
        entry_a = shared_stdio._Entry(ready=loop.create_future(), stop=asyncio.Event(), epoch=1)
        entry_a.command = "lisa-catalog"
        entry_b = shared_stdio._Entry(ready=loop.create_future(), stop=asyncio.Event(), epoch=2)
        entry_b.command = "lisa-content"
        entry_b.established = True
        with shared_stdio._HOST._lock:
            shared_stdio._HOST._entries[(1, ("k-a",))] = entry_a
            shared_stdio._HOST._entries[(2, ("k-b",))] = entry_b
        try:
            assert shared_stdio.shared_session_inventory(epoch=1) == [
                {"command": "lisa-catalog", "epoch": 1, "established": False}
            ]
            assert shared_stdio.shared_session_inventory() == [
                {"command": "lisa-catalog", "epoch": 1, "established": False},
                {"command": "lisa-content", "epoch": 2, "established": True},
            ]
        finally:
            with shared_stdio._HOST._lock:
                shared_stdio._HOST._entries.clear()
            entry_a.ready.cancel()
            entry_b.ready.cancel()
    finally:
        loop.close()
