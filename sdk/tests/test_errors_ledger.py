"""The sealed errors ledger keeps every ERROR+ record and feeds forensics.

``errors.jsonl`` is the ERROR+ evidence ledger: neither rotated nor
rate-limited, sealed to never-deleted numbered sidecars when large. The
rotating ``runtime.jsonl`` evicts old records and suppresses repeats, so
a forensic read prefers the ledger when it exists and falls back to the
rotating log (naming which source it read) for hosts that predate it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from sdk.forensics import ForensicReader
from sdk.logging_setup import (
    JSONFormatter,
    SealedErrorsHandler,
    apply_log_levels,
    configure_logging,
)


def _error_record(msg: str, *, name: str = "test.errors") -> logging.LogRecord:
    return logging.LogRecord(
        name=name,
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )


def _json_handler(path: Path) -> SealedErrorsHandler:
    handler = SealedErrorsHandler(path)
    handler.setFormatter(JSONFormatter())
    return handler


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _seed_thread(threads: Path, turns: int, user: str = "adam") -> None:
    threads.mkdir(parents=True, exist_ok=True)
    entries = [
        {
            "timestamp_utc": f"2026-10-30T1{i}:00:00+00:00",
            "turn": i,
            "user_slug": user,
            "mode_id": "coding",
            "query": f"q{i}",
            "response_text": "r",
        }
        for i in range(turns)
    ]
    (threads / f"{user}.current.jsonl").write_text(
        "\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8"
    )


def _bundle(data_dir: Path):
    reader = ForensicReader(
        data_dir,
        thread_dir=data_dir / "data" / "threads",
        user_slug="adam",
        resolve_redaction_from_env=False,
    )
    return reader.bundle(turns="last")


# ── Handler behavior ─────────────────────────────────────────────────


def test_every_error_record_lands_including_repeats(tmp_path: Path) -> None:
    path = tmp_path / "errors.jsonl"
    handler = _json_handler(path)
    for _ in range(11):
        handler.emit(_error_record("same failure"))

    records = _read_jsonl(path)
    assert len(records) == 11
    assert all(r["level"] == "error" for r in records)


def test_seal_moves_full_segment_to_numbered_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "errors.jsonl"
    handler = _json_handler(path)
    handler._seal_max_bytes = 1
    handler.emit(_error_record("first"))
    handler.emit(_error_record("second"))

    sidecar = tmp_path / "errors.jsonl.1"
    assert [r["msg"] for r in _read_jsonl(sidecar)] == ["first"]
    assert [r["msg"] for r in _read_jsonl(path)] == ["second"]
    assert handler.stream is not None and not handler.stream.closed


def test_seal_continues_numbering_across_restarts(tmp_path: Path) -> None:
    path = tmp_path / "errors.jsonl"
    for _ in range(3):
        handler = _json_handler(path)
        handler._seal_max_bytes = 1
        handler.emit(_error_record("boom"))
        handler.close()

    # The first cycle creates the active file (nothing to seal yet); each
    # later cycle seals it and starts a fresh one.
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "errors.jsonl",
        "errors.jsonl.1",
        "errors.jsonl.2",
    ]


def test_seal_failure_still_appends_the_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "errors.jsonl"
    handler = _json_handler(path)
    handler._seal_max_bytes = 1
    monkeypatch.setattr(
        Path, "rename", lambda self, target: (_ for _ in ()).throw(OSError("disk full"))
    )
    handler.emit(_error_record("must survive"))

    # The contract: the record survives the failed seal…
    assert len(_read_jsonl(path)) == 1
    # …and the handler keeps working on the (unsealed) file.
    handler._seal_max_bytes = 1024 * 1024
    handler.emit(_error_record("still working"))
    assert [r["msg"] for r in _read_jsonl(path)] == ["must survive", "still working"]


def test_configure_logging_writes_errors_ledger(tmp_path: Path) -> None:
    log_path = tmp_path / "runtime.jsonl"
    configure_logging(log_path=log_path, level="info", host_kind="cli")
    root = logging.getLogger()
    try:
        logging.getLogger("test.errors").error("ledger check")
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)

    records = _read_jsonl(tmp_path / "errors.jsonl")
    assert any(r["level"] == "error" and r["msg"] == "ledger check" for r in records)


def test_configure_logging_can_disable_ledger(tmp_path: Path) -> None:
    log_path = tmp_path / "runtime.jsonl"
    configure_logging(log_path=log_path, level="info", host_kind="cli", errors_path=log_path)
    root = logging.getLogger()
    try:
        assert not any(
            isinstance(h, logging.FileHandler) and Path(h.baseFilename).name == "errors.jsonl"
            for h in root.handlers
        )
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)


def test_rate_limited_error_still_lands_in_ledger(tmp_path: Path) -> None:
    """The rotating log suppresses repeats; the ledger must not."""
    log_path = tmp_path / "runtime.jsonl"
    configure_logging(log_path=log_path, level="info", host_kind="cli")
    root = logging.getLogger()
    try:
        for _ in range(10):
            logging.getLogger("test.rate").error("same failure")
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)

    rotating = _read_jsonl(log_path)
    ledger = _read_jsonl(tmp_path / "errors.jsonl")
    assert sum(1 for r in rotating if r["msg"] == "same failure") < 10
    assert sum(1 for r in ledger if r["msg"] == "same failure") == 10


def test_level_floor_survives_apply_log_levels(tmp_path: Path) -> None:
    log_path = tmp_path / "runtime.jsonl"
    configure_logging(log_path=log_path, level="info", host_kind="cli")
    root = logging.getLogger()
    try:
        apply_log_levels(root, level="debug", debug_modules=[])
        errors_handler = next(
            h for h in root.handlers if getattr(h, "_lisa_level_floor", None) is not None
        )
        assert errors_handler.level == logging.ERROR
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)


# ── Forensic read ────────────────────────────────────────────────────


def test_forensics_prefers_errors_ledger(tmp_path: Path) -> None:
    threads = tmp_path / "data" / "threads"
    _seed_thread(threads, turns=2)
    logs = tmp_path / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    ledger_row = {
        "ts": "2026-10-30T19:00:00+00:00",
        "level": "error",
        "name": "runtime.tools",
        "msg": "tool exploded",
    }
    (logs / "errors.jsonl").write_text(json.dumps(ledger_row) + "\n", encoding="utf-8")

    bundle = _bundle(tmp_path)
    assert bundle.sources.errors_log is True
    assert bundle.sources.runtime_log is False
    assert any(e["msg"] == "tool exploded" for e in bundle.log_errors)


def test_forensics_falls_back_to_rotating_log(tmp_path: Path) -> None:
    threads = tmp_path / "data" / "threads"
    _seed_thread(threads, turns=2)
    logs = tmp_path / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_row = {
        "ts": "2026-10-30T19:00:00+00:00",
        "level": "error",
        "name": "runtime.tools",
        "msg": "old-style error",
    }
    (logs / "runtime.jsonl").write_text(json.dumps(log_row) + "\n", encoding="utf-8")

    bundle = _bundle(tmp_path)
    assert bundle.sources.runtime_log is True
    assert bundle.sources.errors_log is False
    assert any(e["msg"] == "old-style error" for e in bundle.log_errors)


def test_forensics_names_missing_log_sources(tmp_path: Path) -> None:
    threads = tmp_path / "data" / "threads"
    _seed_thread(threads, turns=2)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)

    bundle = _bundle(tmp_path)
    assert bundle.sources.errors_log is False
    assert bundle.sources.runtime_log is False
    assert any("runtime.jsonl" in note for note in bundle.notes)
