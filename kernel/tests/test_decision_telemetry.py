"""Tests for the durable decision/dispatch telemetry appender.

Covers ``kernel.decision_telemetry``: the JSONL append contract, topic
tagging, non-mapping rejection, filter hooks, and the ``turn_index`` stamp
that makes the record joinable to the thread and the cost log.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from collections.abc import Iterator
from contextlib import AbstractContextManager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
from kernel.decision_telemetry import (
    TOOL_TELEMETRY_TOPIC,
    TOPIC_FIELD,
    TURN_DECISION_TOPIC,
    DecisionTelemetryHandler,
)
from kernel.models import ToolExecutionTelemetry
from kernel.token_usage import build_token_usage_event


@pytest.mark.asyncio
async def test_turn_decision_appended_as_jsonl_with_topic() -> None:
    """A decision payload lands as one line tagged with its source topic."""
    with TemporaryDirectory() as td:
        path = Path(td) / "logs" / "decisions.jsonl"
        handler = DecisionTelemetryHandler(path)

        await handler.publish_turn_decision(
            {
                "mode_id": "coding_design",
                "turn_index": 42,
                "iteration": 3,
                "action_type": "ContinueAction",
                "action_reason": "tool calls remain",
                "write_receipts": 1,
                "read_receipts": 2,
            }
        )

        assert path.exists()
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(records) == 1
        record = records[0]
        assert record[TOPIC_FIELD] == TURN_DECISION_TOPIC
        assert record["turn_index"] == 42
        assert record["mode_id"] == "coding_design"
        assert record["action_type"] == "ContinueAction"


@pytest.mark.asyncio
async def test_tool_telemetry_appended_and_keeps_turn_index() -> None:
    """A dispatch event keeps the turn key so the ledger joins to the thread."""
    with TemporaryDirectory() as td:
        path = Path(td) / "decisions.jsonl"
        handler = DecisionTelemetryHandler(path)

        event = ToolExecutionTelemetry(
            target_id="fetch_page",
            target_type="tool",
            status="error",
            duration_ms=812,
            error_message="timeout",
            turn_index=17,
        )
        # The bus delivers serialized payloads; the handler must accept that
        # shape as well as the live event object.
        await handler.publish_tool_telemetry(event.model_dump(mode="json"))

        record = json.loads(path.read_text(encoding="utf-8").strip())
        assert record[TOPIC_FIELD] == TOOL_TELEMETRY_TOPIC
        assert record["turn_index"] == 17
        assert record["target_id"] == "fetch_page"
        assert record["status"] == "error"
        assert record["error_message"] == "timeout"


@pytest.mark.asyncio
async def test_live_event_object_is_dumped_not_stringified() -> None:
    """An unserialized event is dumped into a mapping, not recorded as a repr."""
    with TemporaryDirectory() as td:
        path = Path(td) / "decisions.jsonl"
        handler = DecisionTelemetryHandler(path)

        await handler.publish_tool_telemetry(
            ToolExecutionTelemetry(
                target_id="self_model",
                target_type="operation",
                status="success",
                duration_ms=4,
                turn_index=5,
            )
        )

        record = json.loads(path.read_text(encoding="utf-8").strip())
        assert record["target_id"] == "self_model"
        assert record["turn_index"] == 5


@pytest.mark.asyncio
async def test_non_mapping_payload_is_rejected() -> None:
    """A payload that is not a mapping is dropped rather than wrapped."""
    with TemporaryDirectory() as td:
        path = Path(td) / "decisions.jsonl"
        handler = DecisionTelemetryHandler(path)

        await handler.publish_turn_decision("not-a-record")
        await handler.publish_turn_decision([1, 2, 3])

        assert not path.exists()


@pytest.mark.asyncio
async def test_filter_can_drop_a_record() -> None:
    """The filter hook sees the topic and can suppress persistence."""
    with TemporaryDirectory() as td:
        path = Path(td) / "decisions.jsonl"
        seen: list[tuple[str, dict[str, Any]]] = []

        def _filter(topic: str, record: dict[str, Any]) -> bool:
            seen.append((topic, record))
            return record.get("mode_id") != "noisy"

        handler = DecisionTelemetryHandler(path, filter_fn=_filter)
        await handler.publish_turn_decision({"mode_id": "noisy", "turn_index": 1})
        await handler.publish_turn_decision({"mode_id": "quiet", "turn_index": 2})

        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert [r["mode_id"] for r in records] == ["quiet"]
        assert [topic for topic, _ in seen] == [TURN_DECISION_TOPIC, TURN_DECISION_TOPIC]


@pytest.mark.asyncio
async def test_raising_filter_still_persists() -> None:
    """A broken filter must not lose evidence; the record still lands."""
    with TemporaryDirectory() as td:
        path = Path(td) / "decisions.jsonl"

        def _filter(_topic: str, _record: dict[str, Any]) -> bool:
            raise RuntimeError("filter exploded")

        handler = DecisionTelemetryHandler(path, filter_fn=_filter)
        await handler.publish_turn_decision({"mode_id": "kept", "turn_index": 9})

        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert [r["mode_id"] for r in records] == ["kept"]


@pytest.mark.asyncio
async def test_sequence_distinguishes_same_clock_records() -> None:
    """Two records written in the same tick are still distinguishable."""
    with TemporaryDirectory() as td:
        path = Path(td) / "decisions.jsonl"
        handler = DecisionTelemetryHandler(path)

        await handler.publish_turn_decision({"mode_id": "a", "turn_index": 1})
        await handler.publish_turn_decision({"mode_id": "a", "turn_index": 1})

        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(records) == 2
        assert records[0]["sequence"] != records[1]["sequence"]


@pytest.mark.asyncio
async def test_segmentation_is_lossless_across_seals() -> None:
    """Sealing never deletes evidence: every record survives across segments."""
    with TemporaryDirectory() as td:
        path = Path(td) / "decisions.jsonl"
        handler = DecisionTelemetryHandler(path, segment_max_bytes=200)

        written = 0
        for i in range(40):
            await handler.publish_turn_decision({"mode_id": "m", "turn_index": i, "pad": "x" * 20})
            written += 1

        assert (Path(td) / "decisions.jsonl.1").exists()
        assert (Path(td) / "decisions.jsonl.2").exists()
        assert path.exists()

        total = 0
        sequences: list[int] = []
        sealed = sorted(
            (p for p in Path(td).glob("decisions.jsonl.*") if p.name.split(".")[-1].isdigit()),
            key=lambda p: int(p.name.split(".")[-1]),
        )
        ordered = sealed + ([path] if path.exists() else [])
        for seg in ordered:
            for line in seg.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if isinstance(rec, dict):
                    total += 1
                    sequences.append(rec.get("sequence", 0))
        assert total == written
        assert sequences == sorted(sequences)
        assert len(set(sequences)) == len(sequences)


def test_token_usage_event_carries_turn_index() -> None:
    """The cost log is per-turn attributable, not just per-mode."""
    event = build_token_usage_event(
        provider="ollama",
        model_id="llama3.2:latest",
        prompt_tokens=100,
        completion_tokens=20,
        mode_id="coding_design",
        turn_index=88,
    )
    assert event.turn_index == 88
    assert event.mode_id == "coding_design"


# ── Cross-process lock seam ────────────────────────────────────────────


class _CountingLock:
    """A lock_factory double that records acquisitions and serializes writers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._acquired = 0
        self._active = 0
        self._max_concurrent = 0
        self._lock_factory = self._make
        # Bind eagerly so the handler can be reconstructed per seal iteration.
        self.handlers: list[DecisionTelemetryHandler] = []

    def _make(self) -> AbstractContextManager[None]:
        @contextlib.contextmanager
        def _guard() -> Iterator[None]:
            with self._lock:
                self._acquired += 1
                self._active += 1
                self._max_concurrent = max(self._max_concurrent, self._active)
                try:
                    yield
                finally:
                    self._active -= 1

        return _guard()


@pytest.mark.asyncio
async def test_lock_factory_serializes_append_and_seal() -> None:
    """The injected lock wraps the whole append + seal-decision critical section."""
    with TemporaryDirectory() as td:
        path = Path(td) / "decisions.jsonl"
        double = _CountingLock()
        handler = DecisionTelemetryHandler(path, segment_max_bytes=200, lock_factory=double._make)

        for i in range(30):
            await handler.publish_turn_decision({"mode_id": "m", "turn_index": i, "pad": "x" * 30})

        # One acquisition per record, and never two holders at once.
        assert double._acquired >= 30
        assert double._max_concurrent == 1

        # Sealing happened and every record survived it.
        sealed = sorted(p for p in Path(td).glob("decisions.jsonl.*"))
        assert sealed, "seal threshold crossed but no segment sidecar was written"
        total = 0
        for seg in [*sealed, path]:
            for line in seg.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    assert json.loads(line)
                    total += 1
        assert total == 30


def test_seal_number_is_recomputed_from_directory_not_cache() -> None:
    """Two handler instances (two processes' views of one file) never overwrite a sidecar.

    The pre-fix behavior served the next segment number from a per-instance
    cache seeded on first seal, so a second instance that had seen fewer
    segments renamed over the first's sidecar. The number must always be
    derived from the directory listing at seal time: a fresh instance
    continues past every sidecar the first one wrote.
    """
    with TemporaryDirectory() as td:
        path = Path(td) / "decisions.jsonl"
        first = DecisionTelemetryHandler(path, segment_max_bytes=200)
        second = DecisionTelemetryHandler(path, segment_max_bytes=200)

        written_first = 12
        for i in range(written_first):
            asyncio.run(first.publish_turn_decision({"turn_index": i, "pad": "x" * 40}))
        first_sidecars = {p.name for p in Path(td).glob("decisions.jsonl.*")}
        assert len(first_sidecars) >= 2, "seal threshold crossed but fewer than 2 segments exist"

        # Second instance (a different process that never observed the file)
        # seals next, then the first seals AGAIN: the first's cached counter
        # must not have gone stale relative to sidecars the second wrote
        # (the pre-fix failure overwrote a sealed segment).
        written_second = 12
        for i in range(written_second):
            asyncio.run(second.publish_turn_decision({"turn_index": i, "pad": "y" * 40}))
        for i in range(written_first):
            asyncio.run(first.publish_turn_decision({"turn_index": i, "pad": "z" * 40}))

        after = {p.name for p in Path(td).glob("decisions.jsonl.*")}
        assert first_sidecars <= after, "the second instance deleted or overwrote a sealed segment"
        numbers = sorted(int(n.rsplit(".", 1)[1]) for n in after)
        assert numbers == list(range(1, len(numbers) + 1)), (
            f"segment numbers must be contiguous: {numbers}"
        )
        total = 0
        for seg in Path(td).glob("decisions.jsonl.*"):
            total += sum(1 for line in seg.read_text(encoding="utf-8").splitlines() if line.strip())
        total += sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        assert total == written_first * 2 + written_second


@pytest.mark.asyncio
async def test_lock_timeout_drops_the_line_instead_of_raising() -> None:
    """A host lock that times out swallows the failure; telemetry never breaks a turn."""

    @contextlib.contextmanager
    def _never_acquires() -> Iterator[None]:
        raise TimeoutError("lock contention")
        yield  # pragma: no cover

    with TemporaryDirectory() as td:
        handler = DecisionTelemetryHandler(
            Path(td) / "decisions.jsonl", lock_factory=_never_acquires
        )
        # Must not raise despite the lock factory exploding.
        await handler.publish_turn_decision({"mode_id": "m", "turn_index": 1})
        assert not (Path(td) / "decisions.jsonl").exists()
