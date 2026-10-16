"""Shared helpers for model discovery probes.

Environment resolution, model-classification heuristics, and
expiry/viability predicates live here so every probe reads credentials
through the same path and applies the same reasoning/low-latency tagging
and retirement-awareness rules.
"""

from __future__ import annotations

import os
import re
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Reasoning-model markers. Substring match against the lower-cased
# model id. Kept deliberately narrow to avoid false positives like
# matching "r1" inside "far1lab-..." — markers are now anchored to
# known product prefixes or explicit tokens.
_REASONING_MARKERS = (
    "reason",
    "thinking",
    "deepseek-r1",
    "-o1",
    "-o3",
    "-o4",
    "o1-",
    "o3-",
    "o4-",
    "o1:",
    "o3:",
    "o4:",
    "claude-opus-4",
    "gemini-2.5-pro",
)

# gpt-5-family markers. "gpt-5+" = the gpt-5 family and above, which is
# the floor for the cloud default-model pick now that gpt-4 is being
# deprecated. mini/nano variants count as gpt-5+ (they share the family).
_GPT_5_FAMILY_MARKERS = ("gpt-5",)

# Coding-specialized model markers. Substring match against the lower-cased
# model id. Conservative by design: only well-known coding model families /
# naming conventions. ``glm`` is included because GLM models (glm-4.5+ /
# glm-5) are strong code-capable general assistants and routinely serve as
# the coding default on Ollama. Over-matching is avoided by not using bare
# tokens like "code" or "gpt".
_CODE_MODEL_MARKERS = ("codex", "coder", "glm", "code-")

#: Providers treated as local (everything else is cloud). Single source of
#: truth for the local/cloud split — the discovery algorithm and any
#: pack-contributed role defaults consult this.
_LOCAL_PROVIDERS: frozenset[str] = frozenset({"ollama"})

_KNOWN_MODELS_PATH = Path(__file__).parent / "known_models.yaml"


def env_any(*names: str) -> str:
    """Return the first non-empty environment variable among ``names``."""

    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def is_reasoning_model(model_id: str) -> bool:
    """Heuristic: does ``model_id`` look like a reasoning-capable model?

    Anchored markers (``-o1``, ``o1-``, ``o1:``) avoid the previous
    false-positive where the bare substring ``"o1"`` matched anything
    containing those two characters.
    """

    value = model_id.strip().lower()
    if any(marker in value for marker in _REASONING_MARKERS):
        return True
    # gpt-5+ models (except ``*-chat``) reason by default and bill that
    # reasoning against ``max_completion_tokens``; classifying them as plain
    # models capped calls at the 4096 legacy budget and forced truncation
    # retries that resent the whole prompt.
    return is_gpt_5_family(value) and "chat" not in value


def is_code_model(model_id: str) -> bool:
    """Heuristic: does ``model_id`` look like a coding-specialized model?

    Mirrors :func:`is_reasoning_model`: conservative substring markers for
    well-known coding model families/naming conventions. Providers without
    capability metadata (notably Ollama) use this to advertise the ``code``
    capability so the router rule ``'"code" in route.required_capabilities'``
    and ``_pick_code_model`` can select a coding model. Overrides in
    ``known_models.yaml`` still win because each probe applies the registry
    after this heuristic.
    """

    value = model_id.strip().lower()
    return any(marker in value for marker in _CODE_MODEL_MARKERS)


def is_gpt_5_family(model_id: str) -> bool:
    """Heuristic: does ``model_id`` belong to the gpt-5 family or above?

    The cloud default-model pick floors at gpt-5 now that gpt-4 is being
    deprecated. mini/nano variants are included — they share the family
    and are valid low-cost defaults.
    """
    value = model_id.strip().lower()
    return any(marker in value for marker in _GPT_5_FAMILY_MARKERS)


_BREAKPOINT_CACHE_FAMILY = re.compile(r"gpt-(?:5\.(?:[6-9]|\d{2,})|[6-9]|\d{2,})(?![\d])")


def uses_prompt_cache_breakpoints(provider: str, model_id: str) -> bool:
    """Heuristic: does the deployment cache only at prompt-cache breakpoints?

    Azure GPT-5.6 and later families read cached prefixes only where a
    breakpoint ends them (the implicit one on the latest message, plus
    explicit ones). Older models reject the breakpoint field with a 400.
    """

    if provider.strip().lower() != "azure_openai":
        return False
    return bool(_BREAKPOINT_CACHE_FAMILY.search(model_id.strip().lower()))


def is_local_provider(provider: str) -> bool:
    """Return True if ``provider`` is treated as local (ollama today)."""
    return provider.strip().lower() in _LOCAL_PROVIDERS


def is_cloud_provider(provider: str) -> bool:
    """Return True if ``provider`` is treated as cloud (non-local)."""
    return not is_local_provider(provider)


def parse_expiry_date(raw: object) -> date | None:
    """Parse a provider-published expiry scalar into a ``date``.

    Accepts ISO 8601 dates/datetimes (``2026-10-20``,
    ``2026-10-20T00:00:00Z``, ``2026-10-20T00:00:00.000+00:00``). Returns
    ``None`` for empty or unparseable values so an unknown expiry never
    disqualifies a model.
    """
    if isinstance(raw, (datetime, date)):
        return raw.date() if isinstance(raw, datetime) else raw
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


#: Horizon (days) within which a model's expiry makes it ineligible for
#: default/mode pins. A model dying in under a month would leave a fresh
#: install broken almost immediately — skip it before pinning.
PIN_EXPIRY_HORIZON_DAYS = 30


def model_expires_within(
    expires_at: date | None,
    *,
    days: int,
    today: date | None = None,
) -> bool:
    """Return True if ``expires_at`` falls within the next ``days`` days."""
    if expires_at is None:
        return False
    return expires_at <= (today or date.today()) + timedelta(days=days)


def is_model_expired(expires_at: date | None, *, today: date | None = None) -> bool:
    """Return True if ``expires_at`` is today or in the past."""
    if expires_at is None:
        return False
    return expires_at <= (today or date.today())


def is_batch_only_model(model_id: str) -> bool:
    """Return True for Batch-API-only variants the host cannot drive live.

    OpenRouter exposes ``:batch`` twins of interactive models. They only
    accept asynchronous Batch API jobs — an interactive agent loop
    pinning one fails on the first turn.
    """
    return model_id.strip().lower().endswith(":batch")


@lru_cache(maxsize=1)
def load_known_models() -> dict[str, dict[str, Any]]:
    """Load the static ``known_models.yaml`` registry.

    Returns a ``{provider:model_id: {field: value}}`` mapping. The file is
    cached for the process lifetime; restart the host to pick up edits.

    Returns an empty dict if the file is missing or unreadable — discovery
    must never fail because the registry is corrupt.
    """

    if not _KNOWN_MODELS_PATH.exists():
        return {}
    try:
        payload = yaml.safe_load(_KNOWN_MODELS_PATH.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        str(key).lower(): dict(value) for key, value in payload.items() if isinstance(value, dict)
    }


def apply_known_override(
    provider: str,
    model_id: str,
    enrichment: dict[str, Any],
) -> dict[str, Any]:
    """Overlay static-registry values on top of provider-derived enrichment.

    Registry entries take precedence for the fields they specify. Fields
    not present in the registry fall through to the provider-derived values
    already in ``enrichment``.
    """

    registry = load_known_models()
    registry_key = f"{provider}:{model_id}".lower()
    override = registry.get(registry_key)
    if not override:
        return enrichment
    merged = dict(enrichment)
    for field, value in override.items():
        if value in (None, "", [], ()):
            continue
        merged[field] = value
    return merged
