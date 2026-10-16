"""Tests for the ``/models`` command parser and dispatch.

Covers the tokenized parser (tolerant of multi-space, rejects no-space),
the ``--force`` overwrite guard, and the typed-error rendering.
"""

from __future__ import annotations

from runtime.commands import _parse_models_subcommand, _render_discover_notes
from runtime.model_discovery.types import (
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
        sub, _args = _parse_models_subcommand("/modelsdiscover")
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
        report = _build_report(ProbeResult(provider="openai", status=ProbeStatus.SKIPPED))
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

    def test_dropped_models_produce_note_on_ok(self) -> None:
        """OK results that dropped retired models surface a UI note.

        Regression guard for the 2026-10 demo incident: retired Ollama
        cloud tags must be visible in discovery output so operators see
        why a listed tag is missing from the proposal.
        """
        report = _build_report(
            ProbeResult(
                provider="ollama",
                status=ProbeStatus.OK,
                models=(DiscoveredModel(model_id="alive:cloud", provider="ollama"),),
                dropped_model_ids=("retired:cloud",),
            )
        )
        notes = _render_discover_notes(report)
        assert len(notes) == 1
        assert "retired:cloud" in notes[0]
        assert "retired/expired" in notes[0]

    def test_dropped_note_truncates_long_lists(self) -> None:
        """More than six dropped ids roll into a count, not a wall of text."""
        report = _build_report(
            ProbeResult(
                provider="ollama",
                status=ProbeStatus.OK,
                models=(DiscoveredModel(model_id="alive:cloud", provider="ollama"),),
                dropped_model_ids=tuple(f"retired-{i}:cloud" for i in range(9)),
            )
        )
        notes = _render_discover_notes(report)
        assert len(notes) == 1
        assert "retired-5:cloud" in notes[0]
        assert "retired-6:cloud" not in notes[0]
        assert "(+3 more)" in notes[0]


def _build_report(*results: ProbeResult):
    from runtime.model_discovery import DiscoveryReport

    return DiscoveryReport.from_results(results)
