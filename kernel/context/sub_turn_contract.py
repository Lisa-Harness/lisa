"""Authored sub-turn contract helpers for prompt assembly.

Loads the model-facing prose for fork/delegate sub-turn briefings and the
midpoint write-nudge from versioned Markdown assets shipped as pack content.
Python callers supply only path constants, the ``sub_turn_kind`` discriminator,
and substitution glue (``tools_list`` / ``remaining_iterations`` /
``action_list``). No model-facing prose lives in this module.

Mirrors ``kernel.context.context_frames``: a module-level content registry
injected by the host at startup via ``set_content_registry``. No disk-based
default; callers that read before the registry is set raise ``RuntimeError``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from kernel.seams.protocols import ContentRegistry

SubTurnKind = Literal["fork", "delegate"]

_SUB_TURN_ASSET_PREFIX = "prompts/sub_turn"
_FORK_CONTRACT_PATH = f"{_SUB_TURN_ASSET_PREFIX}/fork_contract.md"
_DELEGATE_CONTRACT_PATH = f"{_SUB_TURN_ASSET_PREFIX}/delegate_contract.md"
_BUDGET_EXHAUSTED_PATH = f"{_SUB_TURN_ASSET_PREFIX}/budget_exhausted.md"
_MIDPOINT_NUDGE_PATH = f"{_SUB_TURN_ASSET_PREFIX}/midpoint_write_nudge.md"
_NO_WRITE_WARNING_PATH = f"{_SUB_TURN_ASSET_PREFIX}/no_write_warning.md"

_CONTRACT_PATHS: dict[str, str] = {
    "fork": _FORK_CONTRACT_PATH,
    "delegate": _DELEGATE_CONTRACT_PATH,
}

# Module-level content registry. Hosts MUST inject one via
# ``set_content_registry`` before any sub-turn asset is read. No disk-based
# default.
_content_registry: ContentRegistry | None = None


def set_content_registry(registry: ContentRegistry) -> None:
    """Override the module-level content registry and invalidate caches."""

    global _content_registry
    _content_registry = registry
    read_sub_turn_contract.cache_clear()
    read_sub_turn_budget_exhausted_tail.cache_clear()
    read_sub_turn_midpoint_nudge.cache_clear()
    read_no_write_warning.cache_clear()


class _MissingKeyDict(dict):
    """``str.format_map`` mapping that leaves unfilled placeholders visible.

    Per the pack-content rule, unfilled placeholders remain visible instead of
    raising ``KeyError`` so a missing substitution token is surfaced to the
    model rather than crashing the turn.
    """

    def __missing__(self, key: str) -> str:  # type: ignore[override]
        return "{" + key + "}"


def _strip_authored_heading(text: str) -> str:
    """Strip the leading ``# heading`` line from an authored asset.

    Authored prompt assets carry a top-level heading for lint cleanliness and
    discoverability. The heading
    is documentation metadata, not model-facing content; the loader drops it
    so only the prose body reaches ``str.format_map`` substitution.
    """

    lines = text.splitlines()
    if not lines or not lines[0].lstrip().startswith("#"):
        return text.strip()
    return "\n".join(lines[1:]).strip()


@lru_cache(maxsize=4)
def read_sub_turn_contract(kind: SubTurnKind) -> str:
    """Load the authored contract body for one sub-turn kind.

    Raises ``RuntimeError`` when the content registry is unset, and
    ``ValueError`` when the authored asset is missing or empty — per the
    pack-content rule, degraded sub-turns must fail loudly rather than run
    with an empty contract (the 2026-08-03 incident a missing contract caused).
    """

    if _content_registry is None:
        raise RuntimeError(
            "Content registry not set. Call set_content_registry() before "
            "reading sub-turn contract assets."
        )
    path = _CONTRACT_PATHS.get(kind)
    if path is None:
        raise ValueError(f"Unknown sub_turn_kind: {kind!r}")
    raw = _content_registry.read_authored_text(path)
    body = _strip_authored_heading(raw)
    if not body:
        raise ValueError(
            f"Authored sub-turn contract asset is empty: {path}. The "
            "sub-turn contract must not be empty — a fork/delegate child "
            "with no contract read-loops to the cap without writing."
        )
    return body


@lru_cache(maxsize=2)
def read_sub_turn_budget_exhausted_tail() -> str:
    """Load the authored budget-exhausted tail appended to a sub-turn contract."""

    if _content_registry is None:
        raise RuntimeError(
            "Content registry not set. Call set_content_registry() before "
            "reading sub-turn contract assets."
        )
    raw = _content_registry.read_authored_text(_BUDGET_EXHAUSTED_PATH)
    body = _strip_authored_heading(raw)
    if not body:
        raise ValueError(
            f"Authored sub-turn budget-exhausted asset is empty: {_BUDGET_EXHAUSTED_PATH}."
        )
    return body


@lru_cache(maxsize=2)
def read_sub_turn_midpoint_nudge() -> str:
    """Load the authored midpoint write-nudge body (lifecycle-briefing prose)."""

    if _content_registry is None:
        raise RuntimeError(
            "Content registry not set. Call set_content_registry() before "
            "reading sub-turn contract assets."
        )
    raw = _content_registry.read_authored_text(_MIDPOINT_NUDGE_PATH)
    body = _strip_authored_heading(raw)
    if not body:
        raise ValueError(
            f"Authored sub-turn midpoint-nudge asset is empty: {_MIDPOINT_NUDGE_PATH}."
        )
    return body


@lru_cache(maxsize=2)
def read_no_write_warning() -> str:
    """Load the authored no-verified-write warning footnote.

    Absence is omission: when the authored asset is not available (the content
    registry is not set, or reading raises any exception), returns the empty
    string — the warning footnote is omitted entirely rather than falling back
    to inlined Python prose. This matches the pack-content rule: all
    model-facing prose lives in versioned Markdown; no default fallback text
    in Python.
    """

    if _content_registry is None:
        return ""
    try:
        raw = _content_registry.read_authored_text(_NO_WRITE_WARNING_PATH)
    except Exception:
        return ""
    body = _strip_authored_heading(raw)
    if not body:
        return ""
    return body


def render_no_write_warning(*, action_names: str) -> str:
    """Render the no-verified-write warning footnote for chat-mode softening.

    The caller supplies ``action_names`` (a comma-separated list of backtick-
    wrapped action tool names). When the authored asset is absent the function
    returns the empty string — the warning is omitted.
    """

    body = read_no_write_warning()
    if not body:
        return ""
    return body.format_map(_MissingKeyDict(action_names=action_names))


def render_sub_turn_contract(
    *,
    kind: SubTurnKind,
    tools_list: str,
    remaining_iterations: int,
    budget_exhausted: bool,
) -> str:
    """Render the ``## Sub-Turn Contract`` block for one sub-turn child.

    Keeps only the structural heading and substitution glue in Python; the
    framing prose, deadline sentence, and budget-exhausted tail are authored
    Markdown loaded through the content registry.
    """

    body = read_sub_turn_contract(kind).format_map(
        _MissingKeyDict(
            tools_list=tools_list,
            remaining_iterations=remaining_iterations,
        )
    )
    parts = [body]
    if budget_exhausted:
        parts.append(read_sub_turn_budget_exhausted_tail())
    return "## Sub-Turn Contract\n\n" + " ".join(parts)


def render_sub_turn_midpoint_nudge(*, action_list: str) -> str:
    """Render the midpoint write-nudge briefing line for a sub-turn child."""

    return read_sub_turn_midpoint_nudge().format_map(_MissingKeyDict(action_list=action_list))
