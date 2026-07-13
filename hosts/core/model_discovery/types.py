"""Typed surface for model discovery.

Every provider probe returns a :class:`ProbeResult` carrying a
:class:`ProbeStatus` (a small enum distinguishing ``skipped`` from
``auth`` from ``not_found`` …) and — on failure — a :class:`ProbeError`
whose ``safe_message`` is pre-redacted and safe to render in any UI.

The raw exception is captured into ``ProbeError.detail`` for logging
only. It must never be rendered to the user. This is the structural
fix for the credential-leak class of bugs (Google's ``?key=...`` was
surfaced verbatim via ``str(httpx.HTTPStatusError)`` which embeds the
full request URL).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class ProbeStatus(str, Enum):
    """Discriminated outcome of a single provider probe.

    The UI layer branches on these to render actionable prose instead of
    raw exception strings.
    """

    OK = "ok"                       # probed, returned >=1 model
    EMPTY = "empty"                  # 200 but zero models returned
    SKIPPED = "skipped"              # not configured (no creds/endpoint)
    NETWORK = "network"              # connection/DNS/TLS failure
    AUTH = "auth"                    # 401 / 403
    NOT_FOUND = "not_found"          # 404 — wrong endpoint shape
    RATE_LIMITED = "rate_limited"    # 429
    SCHEMA = "schema"                # 200 but unparseable body
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class DiscoveredModel:
    """A single model candidate discovered from a provider."""

    model_id: str
    provider: str
    reasoning_model: bool = False


@dataclass(frozen=True, slots=True)
class ProbeError:
    """Structured, redacted error for a failed probe.

    ``safe_message`` is the only field that may be rendered to users.
    ``detail`` is for telemetry/logging only and must never be shown.
    """

    provider: str
    status: ProbeStatus
    safe_message: str
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Outcome of probing one provider."""

    provider: str
    status: ProbeStatus
    models: tuple[DiscoveredModel, ...] = ()
    error: ProbeError | None = None
    elapsed_ms: int = 0

    @property
    def succeeded(self) -> bool:
        return self.status is ProbeStatus.OK

    @property
    def configured(self) -> bool:
        """True if the provider was configured (i.e. not SKIPPED)."""

        return self.status is not ProbeStatus.SKIPPED

    def note_line(self) -> str | None:
        """Return a UI-safe note for this result, or None if nothing to say.

        ``OK`` and ``SKIPPED`` produce no note (they are the quiet cases).
        Everything else produces a single line drawn from ``safe_message``.
        """

        if self.status in (ProbeStatus.OK, ProbeStatus.SKIPPED):
            return None
        if self.error is not None:
            return self.error.safe_message
        return f"{self.provider}: {self.status.value}"


@dataclass(frozen=True, slots=True)
class DiscoveryReport:
    """Aggregate result of :func:`discover_models`.

    ``results`` is the canonical, per-provider breakdown. ``models`` and
    ``notes`` are derived conveniences for callers that want the flat
    projections (the host command layer uses both).
    """

    results: tuple[ProbeResult, ...]

    @property
    def models(self) -> tuple[DiscoveredModel, ...]:
        return tuple(model for result in self.results for model in result.models)

    @property
    def notes(self) -> tuple[str, ...]:
        return tuple(
            line
            for line in (result.note_line() for result in self.results)
            if line is not None
        )

    def results_for(self, provider: str) -> tuple[ProbeResult, ...]:
        lowered = provider.lower()
        return tuple(r for r in self.results if r.provider.lower() == lowered)

    @classmethod
    def from_results(cls, results: Iterable[ProbeResult]) -> "DiscoveryReport":
        return cls(results=tuple(results))