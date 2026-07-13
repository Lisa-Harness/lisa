"""Tests for the ``/models`` command parser and dispatch.

Covers the tokenized parser (tolerant of multi-space, rejects no-space),
the ``--force`` overwrite guard, and the typed-error rendering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hosts.core.commands import _parse_models_subcommand, _render_discover_notes
from hosts.core.model_discovery.types import (
    DiscoveredModel,
    ProbeError,
    ProbeResult,
    ProbeStatus,
)


class TestParseModelsSubcommand:
    def test_no_subcommand(self) -> None:
        assert _parse_models_subcommand("/models") == ("", ())

    def test_single_subcommand(self) -> None:
        assert _parse_models_subcommand("/models show") == ("show", ())

    def test_multi_space_tolerated(self) -> None:
        assert _parse_models_subcommand("/models  show") == ("show", ())

    def test_tab_tolerated(self) -> None:
        assert _parse_models_subcommand("/models\tshow") == ("show", ())

    def test_rejects_no_space(self) -> None:
        # /modelsdiscover has no space → parts[0] is the whole thing,
        # which is not "/models", so returns ("", ()) → help branch.
        sub, args = _parse_models_subcommand("/modelsdiscover")
        assert sub == ""

    def test_with_args(self) -> None:
        assert _parse_models_subcommand("/models discover --force") == ("discover", ("--force",))

    def test_case_insensitive(self) -> None:
        assert _parse_models_subcommand("/models SHOW") == ("show", ())


class TestRenderDiscoverNotes:
    def test_ok_produces_no_note(self) -> None:
        report = _build_report(
            ProbeResult(
                provider="openai",
                status=ProbeStatus.OK,
                models=(DiscoveredModel(model_id="gpt-4o", provider="openai"),),
            )
        )
        assert _render_discover_notes(report) == []

    def test_skipped_produces_no_note(self) -> None:
        report = _build_report(
            ProbeResult(provider="openai", status=ProbeStatus.SKIPPED)
        )
        assert _render_discover_notes(report) == []

    def test_auth_produces_note(self) -> None:
        report = _build_report(
            ProbeResult(
                provider="openai",
                status=ProbeStatus.AUTH,
                error=ProbeError(
                    provider="openai",
                    status=ProbeStatus.AUTH,
                    safe_message="openai: HTTP 401 Unauthorized",
                ),
            )
        )
        notes = _render_discover_notes(report)
        assert len(notes) == 1
        assert "openai" in notes[0]
        assert "401" in notes[0]

    def test_key_never_in_note(self) -> None:
        report = _build_report(
            ProbeResult(
                provider="google",
                status=ProbeStatus.UNKNOWN,
                error=ProbeError(
                    provider="google",
                    status=ProbeStatus.UNKNOWN,
                    safe_message="google: HTTP 500 Internal Server Error",
                    detail="httpx.HTTPStatusError: 500 for url https://...?key=AIzaSySECRET",
                ),
            )
        )
        notes = _render_discover_notes(report)
        for note in notes:
            assert "AIzaSySECRET" not in note


def _build_report(*results: ProbeResult):
    from hosts.core.model_discovery import DiscoveryReport

    return DiscoveryReport.from_results(results)