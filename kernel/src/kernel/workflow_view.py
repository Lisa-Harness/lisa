"""Derived workflow view from the lossless thread store.

The thread is the single source of truth for workflow state. This module
derives a ``WorkflowView`` — a frozen snapshot of the active workflow —
from the thread tail at turn start. The view is never persisted; it is
rebuilt every turn from the append-only thread log.

The plan lives as a ``## Plan`` block in a turn's ``response_text``.
Completion is inferred from the plan turn and later thread entries:
``effect_receipts`` matching each step's ``write_capability`` (for write
steps) or the step's mode handing off (for read steps, via
``departed_modes``). When a match is found, the completing entry's
global turn index is recorded on the step as ``completing_turn`` — the
completion is turn-addressed and verifiable via ``thread_read_range`` or
the markdown render operation. The string match is the *discovery*; the
addressed turn is the *conclusion*. Mode/stage pointers are derived from
the most recent thread entry's ``mode_id`` and ``transitioned_to``.

This replaces the mutable ``WorkflowPosition`` + ``FileWorkflowPositionStore``
pair that drifted from the thread. One store, append-only, lossless.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from kernel.config import settings
from kernel.workflow_position import WorkflowStep

__all__ = [
    "WORKFLOW_CONCLUSION_KEY",
    "WorkflowView",
    "derive_workflow_view",
    "has_workflow_plan",
    "is_complete",
    "remaining_steps",
    "render_workflow_progress",
]

#: Thread-entry ``provenance`` key recording that a workflow was concluded by
#: the operator. The value carries the operator's reason (or an empty string
#: when none was given); the presence of the key is what retires the plan.
WORKFLOW_CONCLUSION_KEY = "workflow_conclusion"


# ── Plan block parser ────────────────────────────────────────────────

# Anchored at the start of the response: ``_render_plan_markdown`` prepends
# the block, so a plan that does not open the turn is a quotation, an
# example, or a discussion of a plan — not the active one. ``\A`` is used
# instead of ``^`` so no leading whitespace can satisfy it, since ``match``
# on a ``re.MULTILINE`` ``^`` pattern would re-anchor at every line.
_PLAN_HEADER_RE = re.compile(r"\A##[ \t]+Plan[ \t]*$", re.MULTILINE)
# Matches:  1. **search** — Search for the latest rust async runtime news
# Captures: mode_id="search", description="Search for the latest..."
_PLAN_STEP_RE = re.compile(r"^\s*(?:\d+\.|[-*])\s+\*{0,2}([a-zA-Z0-9_-]+)\*{0,2}\s*[—–-]\s*(.+)$")
# Optional rationale line:  _Provides web_search_
_RATIONALE_RE = re.compile(r"^\s*_\s*(.+?)\s*_\s*$")
# Optional deliverable line:  _writes: markdown_forge_document_  or  _reads_
_WRITES_RE = re.compile(r"^\s*_\s*writes:\s*([a-zA-Z0-9_.-]+)\s*_\s*$")
_READS_RE = re.compile(r"^\s*_\s*reads_\s*$")


def _parse_step_deliverable_lines(
    lines: list[str],
    step_index: int,
) -> tuple[str, str | None, int]:
    """Consume the optional rationale / deliverable lines following one plan step.

    Both the rationale and the ``_writes_`` / ``_reads_`` line use the ``_..._``
    wrapper; they are distinguished by content, and the deliverable line takes
    precedence for capability extraction. Returns ``(rationale, write_capability,
    next_index)`` where ``next_index`` is the line index after the consumed
    lookahead lines.
    """
    rationale = ""
    write_capability: str | None = None
    i = step_index
    if i + 1 >= len(lines):
        return rationale, write_capability, i

    writes_match = _WRITES_RE.match(lines[i + 1])
    reads_match = _READS_RE.match(lines[i + 1])
    if writes_match:
        write_capability = writes_match.group(1).strip()
        i += 1
        if i + 1 < len(lines):
            rat_match = _RATIONALE_RE.match(lines[i + 1])
            if rat_match:
                rationale = rat_match.group(1).strip()
                i += 1
    elif reads_match:
        i += 1
        if i + 1 < len(lines):
            rat_match = _RATIONALE_RE.match(lines[i + 1])
            if rat_match:
                rationale = rat_match.group(1).strip()
                i += 1
    else:
        rat_match = _RATIONALE_RE.match(lines[i + 1])
        if rat_match:
            rationale = rat_match.group(1).strip()
            i += 1
    return rationale, write_capability, i


def _parse_plan_block(text: str) -> tuple[str, list[tuple[str, str, str, str | None]]] | None:
    """Extract the ``## Plan`` block from ``text``.

    Returns ``(plan_text, steps)`` where ``steps`` is a list of
    ``(mode_id, description, rationale, write_capability)`` tuples.
    ``write_capability`` is the capability id parsed from a ``_writes: <id>_``
    line, ``None`` for read steps (``_reads_`` or no deliverable line), or
    ``None`` when no ``## Plan`` header is found.

    Only a plan that *starts* the response counts. ``_render_plan_markdown``
    emits the block at the top of the turn's text (the turn runner prepends
    it), so an unanchored match would treat any response that merely
    *mentions* a plan — a document containing a ``## Plan`` example, a
    quoted template, a review of a plan — as an active workflow. The same
    rule keeps fenced examples from counting: prose cannot precede the
    header, so a fenced block (which must follow prose to be introduced) is
    excluded along with every other quoted case.

    The parser is best-effort: it handles the standard format produced by
    ``_render_plan_markdown`` and common prose variants. Steps with
    unparseable mode_ids are skipped (not silently accepted with empty
    mode_id — the FSM theme: if preconditions are not met, stop).
    """
    match = _PLAN_HEADER_RE.match(text)
    if not match:
        return None

    # Extract from the header to the next ## heading or end of text.
    block_start = match.start()
    after = text[match.end() :]
    next_header = re.search(r"^##\s+", after, re.MULTILINE)
    if next_header:
        block_text = text[block_start : match.end() + next_header.start()]
    else:
        block_text = text[block_start:]

    lines = block_text.splitlines()
    steps: list[tuple[str, str, str, str | None]] = []
    i = 1  # skip the header line
    while i < len(lines):
        step_match = _PLAN_STEP_RE.match(lines[i])
        if step_match:
            mode_id = step_match.group(1).strip()
            description = step_match.group(2).strip()
            # Check next lines for rationale and/or deliverable line.
            rationale, write_capability, i = _parse_step_deliverable_lines(lines, i)
            # Skip steps with empty mode_id or description (hollow steps)
            if mode_id and description:
                steps.append((mode_id, description, rationale, write_capability))
        i += 1

    if not steps:
        # A ## Plan header with no parseable steps is not a plan.
        return None

    return block_text.strip(), steps


# ── WorkflowView model ────────────────────────────────────────────────


def has_workflow_plan(text: str) -> bool:
    """Return whether text contains a parseable plan, not just a plan heading."""
    return _parse_plan_block(text) is not None


class WorkflowView(BaseModel):
    """A derived, immutable snapshot of the active workflow.

    Drop-in compatible with ``WorkflowPosition``: same field names,
    same ``step_by_id`` method. Built from the thread tail by
    ``derive_workflow_view`` — never persisted, never mutated.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    # ── Identity & provenance ────────────────────────────────────────
    position_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    plan_digest: str = ""
    spawned_from: str = "thread_derived"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_progressed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # ── Inter-mode plan ───────────────────────────────────────────────
    declared_steps: tuple[WorkflowStep, ...] = ()
    completed_steps: tuple[str, ...] = ()
    current_step_id: str = ""
    plan_text: str = ""
    step_results: dict[str, str] = Field(default_factory=dict)

    # ── Intra-mode FSM pointer ────────────────────────────────────────
    mode_id: str = ""
    stage_id: str = ""
    turn_count: int = 0
    terminal_reached: bool = False

    def step_by_id(self, step_id: str) -> WorkflowStep | None:
        """Return the declared step with the given id, or ``None``."""
        for step in self.declared_steps:
            if step.step_id == step_id:
                return step
        return None


# ── Derivation ────────────────────────────────────────────────────────


def _step_id_for(mode_id: str, index: int) -> str:
    """Build a stable step id from a mode id and its position in the chain."""
    slug = "".join(c if c.isalnum() else "_" for c in mode_id.lower())
    return f"{slug}_{index}"


#: One unit of completion evidence inside a thread entry:
#: ``(kind, mode_id, capability_id)``. ``kind`` is ``"write"`` for a write
#: receipt (capability set) or ``"handoff"`` for a mode departing (capability
#: empty).
_Evidence = tuple[str, str, str]


def _write_evidence(entry: dict[str, Any]) -> list[_Evidence]:
    """Return one ``write`` evidence unit per write receipt in the entry.

    Mode and capability are paired per receipt: one thread entry can span
    several modes (a turn may enter in one mode and hand off mid-turn), so
    matching them independently would let one mode's receipt and another
    mode's capability combine into false completion evidence. Receipts
    persisted before ``EffectReceipt.mode_id`` existed carry an empty id, so
    the entry's own ``mode_id`` is used as their fallback.
    """
    receipts = entry.get("effect_receipts")
    if not receipts or not isinstance(receipts, list):
        return []
    entry_mode = str(entry.get("mode_id") or "")
    units: list[_Evidence] = []
    for r in receipts:
        if not isinstance(r, dict):
            continue
        if str(r.get("effect", "")) not in ("write", "external_write"):
            continue
        cap = str(r.get("capability_id") or "")
        if not cap:
            continue
        units.append(("write", str(r.get("mode_id") or "") or entry_mode, cap))
    return units


def _handoff_evidence(entry: dict[str, Any]) -> list[_Evidence]:
    """Return one ``handoff`` evidence unit per mode that finished in the entry.

    ``departed_modes`` lists every mode that handed off mid-turn, so a turn
    that ran ``planner → thinker → researcher`` credits ``thinker`` even
    though ``handoff_to`` only keeps the last hop. A stage transition
    (``transitioned_to``) credits the mode the turn ended in. Entries
    written before ``departed_modes`` existed fall back to the entry's own
    ``mode_id`` when it handed off or transitioned.
    """
    handoff = str(entry.get("handoff_to") or "")
    transitioned = str(entry.get("transitioned_to") or "")
    entry_mode = str(entry.get("mode_id") or "")
    departed = entry.get("departed_modes")
    if not isinstance(departed, list):
        return [("handoff", entry_mode, "")] if handoff or transitioned else []
    units: list[_Evidence] = [("handoff", str(m), "") for m in departed if m]
    if transitioned:
        units.append(("handoff", handoff or entry_mode, ""))
    return units


def _entry_modes(entry: dict[str, Any]) -> set[str]:
    """Return every mode a thread entry ran in.

    The entry's own ``mode_id`` is the mode the turn entered in. A turn that
    hands off mid-turn also runs in the mode it left, recorded in
    ``departed_modes``, and ends in the mode it handed off to.
    """
    modes: set[str] = set()
    entry_mode = str(entry.get("mode_id") or "")
    if entry_mode:
        modes.add(entry_mode)
    departed = entry.get("departed_modes")
    if isinstance(departed, list):
        modes.update(str(mode) for mode in departed if mode)
    handoff = str(entry.get("handoff_to") or "")
    if handoff:
        modes.add(handoff)
    return modes


def _count_idle_turns(
    thread_entries: list[dict[str, Any]],
    *,
    last_progress: int,
    open_step_modes: set[str],
) -> int:
    """Return the length of the trailing run of turns that are idle for the plan.

    A turn is idle when it neither completes a step nor runs in any open
    step's mode. The run is counted backwards from the newest entry and
    stops at the first turn that is not idle, so a turn that did the open
    step's work resets the count. Only a stretch of turns that ignore the
    plan entirely retires it — a plan whose step is still being worked on
    is not abandoned, however long the step takes.

    Entries at or before ``last_progress`` completed a step, so they are
    never idle and are excluded from the scan.
    """
    idle = 0
    for entry in reversed(thread_entries[last_progress + 1 :]):
        if open_step_modes & _entry_modes(entry):
            break
        idle += 1
    return idle


def _evidence_satisfies_step(unit: _Evidence, step: WorkflowStep) -> bool:
    """Return True when one evidence unit completes ``step``.

    - **Write step** (``step.write_capability`` set): a write receipt
      dispatched in the step's mode whose capability matches in raw or bare
      form (``storage.save`` or ``save``). No dense-content fallback — the
      receipt is the honest signal.
    - **Read step** (``step.write_capability`` is None): the step's mode
      handing off (or transitioning stage). The reads and prose are already
      in the lossless thread; the handoff is the completion signal.

    Mode scoping prevents one mode's work from completing another's step.
    """
    kind, mode_id, capability = unit
    if mode_id != step.mode_id:
        return False
    if step.write_capability is None:
        return kind == "handoff"
    if kind != "write":
        return False
    return step.write_capability in (capability, capability.rsplit(".", 1)[-1])


#: Maximum length of a completed step's one-line result preview.
_PREVIEW_MAX_CHARS = 300


def _strip_plan_lines(block_text: str) -> str:
    """Drop the header, step, and deliverable/rationale lines from a plan block."""
    plan_line_res = (_PLAN_HEADER_RE, _PLAN_STEP_RE, _RATIONALE_RE, _WRITES_RE, _READS_RE)
    return "\n".join(
        line
        for line in block_text.splitlines()
        if not any(pattern.match(line) for pattern in plan_line_res)
    )


def _result_preview(entry: dict[str, Any]) -> str:
    """Return a one-line preview of the completing entry's response.

    The preview is rendered inside a checklist line, so it must not carry
    the response's own structure: the plan block a plan-emitting turn opens
    with and any ``## Workflow progress`` footer are dropped, and the rest is
    collapsed to single spaces — a raw multi-line slice would spill headings
    and list items into the footer.
    """
    text = str(entry.get("response_text") or "")
    plan = _parse_plan_block(text)
    if plan is not None:
        # The block runs to the next heading, so a plan turn whose answer
        # has no heading of its own keeps it inside the block; fall back to
        # the block's non-plan lines then.
        after = text.strip()[len(plan[0]) :]
        text = after if after.strip() else _strip_plan_lines(plan[0])
    footer = re.search(r"^##[ \t]+Workflow progress[ \t]*$", text, re.MULTILINE)
    if footer:
        text = text[: footer.start()]
    flat = " ".join(text.split())
    if len(flat) <= _PREVIEW_MAX_CHARS:
        return flat
    return flat[: _PREVIEW_MAX_CHARS - 1].rstrip() + "…"


def _workflow_concluded_after(thread_entries: list[dict[str, Any]], plan_entry_index: int) -> bool:
    """Return True when an entry after the plan records a workflow conclusion.

    The conclusion is stamped by the host's ``/workflow conclude`` command as
    ``provenance["workflow_conclusion"]`` on a thread entry. Because the
    thread is append-only and the view is derived, that single entry is
    enough to retire the workflow durably — no mutable position store, and
    the reason stays readable for audit.
    """
    for entry in thread_entries[plan_entry_index + 1 :]:
        provenance = entry.get("provenance")
        if isinstance(provenance, dict) and WORKFLOW_CONCLUSION_KEY in provenance:
            return True
    return False


def _latest_plan_entry(
    thread_entries: list[dict[str, Any]],
) -> tuple[int, str, list[tuple[str, str, str, str | None]]] | None:
    """Find the most recent ``## Plan`` block, scanning backwards.

    Returns ``(entry_index, plan_text, parsed_steps)``, or None when no
    entry carries a plan block (no active workflow).
    """
    for i in range(len(thread_entries) - 1, -1, -1):
        entry = thread_entries[i]
        response = str(entry.get("response_text") or "")
        result = _parse_plan_block(response)
        if result is not None:
            plan_text, parsed_steps = result
            return i, plan_text, parsed_steps
    return None


def _build_view_steps(
    parsed_steps: list[tuple[str, str, str, str | None]],
) -> list[WorkflowStep]:
    """Build the declared WorkflowStep chain from one parsed plan."""
    steps: list[WorkflowStep] = []
    for idx, (mode_id, description, rationale, write_capability) in enumerate(parsed_steps):
        step_id = _step_id_for(mode_id, idx)
        steps.append(
            WorkflowStep(
                step_id=step_id,
                mode_id=mode_id,
                step_query=description,
                plan_text=rationale,
                is_terminal=(idx == len(parsed_steps) - 1),
                spawned_from="thread_derived",
                write_capability=write_capability,
            )
        )
    return steps


def _match_step_completions(
    steps: list[WorkflowStep],
    thread_entries: list[dict[str, Any]],
    plan_entry_index: int,
) -> tuple[list[str], dict[str, str], list[WorkflowStep], int]:
    """Determine step completion by consuming evidence units monotonically.

    Steps are matched in declared order, and each evidence unit — one write
    receipt or one mode departure — can complete at most one step (``used``
    tracks claimed units). That makes double-counting unrepresentable: one
    commit receipt cannot tick two ``coding`` steps. Claiming units rather
    than whole entries matters because one turn can run several plan steps
    back to back (``thinker`` hands off to ``researcher``, which writes).

    The scan starts at the plan entry itself: the planner opens the turn
    and the turn goes on to execute steps, so work done in the
    plan-emitting turn is genuine completion evidence (the 2026-10-28 turn
    1094 ran both of its steps in-turn, was never credited, and its plan
    stayed live for every later turn).

    Every step scans from the plan onwards, so a step can be satisfied by work
    that ran *before* an earlier step finished. A strict forward cursor would
    be wrong here: if step 1 never completes, the cursor stops on it and step
    2's receipt — already in the thread — is never seen, leaving genuinely
    finished work marked open forever. Completion reports what happened, not
    what the declared order predicted, and the per-mode handoff/receipt test
    still prevents a step from claiming another mode's work.

    Returns ``(completed_ids, step_results, updated_steps, last_progress)``
    with ``completing_turn`` stamped on completed steps; ``last_progress`` is
    the index of the latest entry that completed a step, or the plan entry
    when none did.
    """
    completed: list[str] = []
    step_results: dict[str, str] = {}
    updated_steps = list(steps)
    evidence = [
        _write_evidence(entry) + _handoff_evidence(entry)
        for entry in thread_entries[plan_entry_index:]
    ]

    used: set[tuple[int, int]] = set()
    last_progress = plan_entry_index
    for idx, step in enumerate(updated_steps):
        match = next(
            (
                (offset, unit_index)
                for offset, units in enumerate(evidence)
                for unit_index, unit in enumerate(units)
                if (offset, unit_index) not in used and _evidence_satisfies_step(unit, step)
            ),
            None,
        )
        if match is None:
            continue
        used.add(match)
        entry_index = plan_entry_index + match[0]
        last_progress = max(last_progress, entry_index)
        entry = thread_entries[entry_index]
        completed.append(step.step_id)
        preview = _result_preview(entry)
        if preview:
            step_results[step.step_id] = preview
        # Record the completing turn index so the completion is
        # turn-addressed and verifiable via ``thread_read_range``. The
        # entry's ``turn`` field is the global 0-based turn index
        # stamped at append time.
        completing_turn = entry.get("turn")
        if isinstance(completing_turn, int):
            updated_steps[idx] = step.model_copy(update={"completing_turn": completing_turn})

    return completed, step_results, updated_steps, last_progress


def _count_consecutive_same_mode_turns(
    thread_entries: list[dict[str, Any]],
    current_mode: str,
    current_stage: str,
) -> int:
    """Count consecutive tail entries sharing the same mode_id + stage_id."""
    turn_count = 1
    for j in range(len(thread_entries) - 2, -1, -1):
        entry = thread_entries[j]
        if (
            entry.get("mode_id") == current_mode
            and str(entry.get("transitioned_to") or "") == current_stage
        ):
            turn_count += 1
        else:
            break
    return turn_count


def derive_workflow_view(
    thread_entries: list[dict[str, Any]],
    catalog: dict[str, Any] | None = None,
    *,
    idle_turn_limit: int | None = None,
) -> WorkflowView | None:
    """Derive the active workflow view from the thread tail.

    Scans the thread entries (chronological order) for the most recent
    ``## Plan`` block, parses its steps, and determines completion by
    matching ``effect_receipts`` and mode departures in the plan entry and
    later entries against each step's ``mode_id``.

    Returns ``None`` when:
    - No ``## Plan`` block is found in any entry (no active workflow).
    - The plan is found but all steps are complete (the workflow is done;
      the view should not be injected — "if I finished a plan then quit,
      there shouldn't be a plan").
    - The plan was concluded by the operator (``/workflow conclude``).
    - ``idle_turn_limit`` consecutive *idle* turns followed the plan's latest
      progress — turns that neither completed a step nor ran in any open
      step's mode (the plan went stale). Defaults to
      ``Settings.workflow_plan_idle_turn_limit``; ``0`` disables the rule.

    The ``catalog`` parameter is optional and currently unused — it's
    reserved for future enrichment (e.g. resolving ``satisfies_intents``
    from the mode catalog).
    """
    if not thread_entries:
        return None

    # Scan from most recent backwards for the latest ## Plan block.
    latest_plan = _latest_plan_entry(thread_entries)
    if latest_plan is None:
        # No plan block found — no active workflow.
        return None
    plan_entry_index, plan_text, parsed_steps = latest_plan

    # A conclusion recorded after the plan retires it. The marker is a
    # provenance key on a later thread entry, so it needs no mutable store:
    # the thread stays the single source of truth, and the reason survives
    # for audit. Only entries *after* the plan count — an earlier conclusion
    # belongs to a previous workflow and must not silently kill this one.
    if _workflow_concluded_after(thread_entries, plan_entry_index):
        return None

    # Build WorkflowStep objects from the parsed plan.
    steps = _build_view_steps(parsed_steps)

    completed, step_results, steps, last_progress = _match_step_completions(
        steps, thread_entries, plan_entry_index
    )

    # Completion frontier: the declaration index of the latest completed
    # step. Steps complete by *evidence* (receipts and mode departures),
    # not in declared order — step 2 can run while step 1 is abandoned —
    # so the frontier, not the naive "first open step", decides what the
    # plan still routes to. Both the idle retirement and the current-step
    # pointer below read it.
    completed_indexes = [idx for idx, s in enumerate(steps) if s.step_id in completed]
    high_water = max(completed_indexes, default=-1)

    # A plan that stopped making progress is retired: without this, an
    # abandoned plan re-attaches its footer and chain-in-flight handoff to
    # every later turn, re-routing work that has nothing to do with it.
    #
    # The rule is "idle turns" — consecutive turns that neither complete a
    # step nor run in an open step's mode. Counting every turn since the
    # last completion would retire a plan whose step is still being worked
    # on (the step's own turns are not idle), which is the difference
    # between an abandoned plan and a slow one.
    #
    # Front-awareness matters here too: only open steps at or after the
    # completion frontier keep the plan alive. A turn that runs in an
    # abandoned pre-frontier step's mode (exactly what a backward routing
    # hint produces) must not reset the idle count and immortalize the
    # zombie plan — the t1107/t1108 loop sustained itself precisely
    # because each hijacked turn ended in `thinker`, the abandoned step's
    # mode, resetting the retirement clock.
    limit = settings.workflow_plan_idle_turn_limit if idle_turn_limit is None else idle_turn_limit
    if limit > 0:
        open_step_modes = {
            step.mode_id for step in steps[high_water + 1 :] if step.step_id not in completed
        }
        idle_turns = _count_idle_turns(
            thread_entries,
            last_progress=last_progress,
            open_step_modes=open_step_modes,
        )
        if idle_turns >= limit:
            return None

    # Derive mode/stage pointer from the most recent entry.
    last_entry = thread_entries[-1]
    current_mode = str(last_entry.get("mode_id") or "")
    current_stage = str(last_entry.get("transitioned_to") or "")

    # Derive turn_count: consecutive entries with same mode_id + stage_id.
    turn_count = _count_consecutive_same_mode_turns(thread_entries, current_mode, current_stage)

    # Determine current step: the first non-completed step at or after the
    # completion frontier computed above. The naive ``remaining[0]`` pinned
    # the pointer on an abandoned pre-frontier step forever, so every
    # post-handoff routing hint pointed backward at it and the
    # forced-action enforcer dragged a compliant model back on each
    # handoff (the 2026-10-30 t1107/t1108 loop: the view said `thinker`
    # while the turn was in `coding_review`). When no open step sits after
    # the frontier, the pointer is *empty* — the chain has no forward step
    # to name — but the view stays alive: the checklist still renders, and
    # only the routing (hint, resume, budget transition) goes silent.
    current_step_id = ""
    remaining = [s for s in steps if s.step_id not in completed]
    if not remaining:
        # All steps complete — the workflow is done. Return None so the
        # caller doesn't inject a stale plan ("if I finished a plan then
        # quit, there shouldn't be a plan").
        return None
    current_step_id = next(
        (s.step_id for s in steps[high_water + 1 :] if s.step_id not in completed),
        "",
    )

    # Deterministic position_id from plan_text hash (stable across turns
    # for the same plan, so recovery record correlation works).
    plan_digest = hashlib.sha256(plan_text.encode()).hexdigest()[:16]
    position_id = hashlib.sha256(
        f"{plan_digest}:{current_mode}:{current_stage}".encode()
    ).hexdigest()[:32]

    now = datetime.now(UTC)
    terminal = len(completed) == len(steps)

    return WorkflowView(
        position_id=position_id,
        plan_digest=plan_digest,
        spawned_from="thread_derived",
        created_at=now,
        last_progressed_at=now,
        declared_steps=tuple(steps),
        completed_steps=tuple(completed),
        current_step_id=current_step_id,
        plan_text=plan_text,
        step_results=step_results,
        mode_id=current_mode,
        stage_id=current_stage,
        turn_count=turn_count,
        terminal_reached=terminal,
    )


# ── Helpers (moved from workflow_position.py, adapted for WorkflowView) ──


def remaining_steps(view: WorkflowView) -> tuple[WorkflowStep, ...]:
    """Return the declared steps not yet completed, in declared order."""
    completed = set(view.completed_steps)
    return tuple(s for s in view.declared_steps if s.step_id not in completed)


def is_complete(view: WorkflowView) -> bool:
    """Return True when the view has no remaining work.

    Complete means: every declared step is completed AND the intra-mode
    pointer is either absent or parked on a terminal stage.
    """
    steps_open = bool(view.declared_steps) and len(view.completed_steps) < len(view.declared_steps)
    stage_open = bool(view.stage_id) and not view.terminal_reached
    return not steps_open and not stage_open


def render_workflow_progress(view: WorkflowView) -> str:
    """Render a human-readable summary of completed and remaining steps.

    Returns an empty string when the view has no declared steps or is
    complete (no progress to report). Otherwise emits a checklist of
    all declared steps (with a 300-char result preview and the
    completing turn index for completed steps), followed by a
    ``## Current step`` block giving the current step's full description
    and rationale so the model can orient narratively — not just
    structurally.
    """
    if not view.declared_steps:
        return ""
    if is_complete(view):
        return ""
    completed = set(view.completed_steps)
    lines: list[str] = ["## Workflow progress"]
    for step in view.declared_steps:
        mark = "[x]" if step.step_id in completed else "[ ]"
        label = f"{mark} {step.mode_id}"
        preview = view.step_results.get(step.step_id, "")
        if preview:
            label += f": {preview[:300]}"
        if step.step_id in completed and step.completing_turn is not None:
            label += f" ← turn {step.completing_turn}"
        lines.append(label)
    # ── Current step orientation block ───────────────────────────────
    # Give the model the full description and rationale for the step it
    # is about to run (or is running), not just a checkbox label. This
    # is the narrative orientation that lets the model "pick up where we
    # left off" rather than just seeing a structural checklist.
    current_step = view.step_by_id(view.current_step_id) if view.current_step_id else None
    if current_step is not None:
        lines.append("")
        lines.append("## Current step")
        lines.append("")
        lines.append(f"**{current_step.mode_id}**: {current_step.step_query}")
        if current_step.plan_text:
            lines.append("")
            lines.append(current_step.plan_text)
    return "\n".join(lines)
