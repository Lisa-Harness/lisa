"""Write a run bundle: verdict, raw outcome, and a readable transcript."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from e2etest.oracles import Evidence, OracleResult
from e2etest.scenario import Scenario

_TEXT_LIMIT = 4000


def verdict(scenario: Scenario, evidence: Evidence, results: list[OracleResult]) -> dict[str, Any]:
    failed = [r for r in results if not r.passed]
    return {
        "scenario": scenario.scenario_id,
        "scenario_file": str(scenario.source),
        "run_dir": str(evidence.paths.run_dir),
        "passed": not failed,
        "first_failure": asdict(failed[0]) if failed else None,
        "exchanges": len(evidence.outcome.exchanges),
        "aborted": evidence.outcome.aborted,
        "results": [asdict(r) for r in results],
    }


def _transcript(evidence: Evidence) -> str:
    lines = ["# Transcript", ""]
    for ex in evidence.outcome.exchanges:
        lines.append(f"## #{ex.index} {ex.kind} ({ex.duration_s}s)")
        lines += ["", "```text", ex.message, "```", ""]
        flags = []
        if ex.approval:
            flags.append(
                f"approval: {ex.approval.get('target_id')} ({ex.approval.get('request_id')})"
            )
        if ex.operator_exit:
            flags.append(f"operator_exit: {ex.operator_exit}")
        if ex.yielded_question:
            flags.append(f"yielded: {ex.yielded_question}")
        flags += [f"error: {e}" for e in ex.errors]
        lines += [f"- {f}" for f in flags]
        text = ex.text if len(ex.text) <= _TEXT_LIMIT else ex.text[:_TEXT_LIMIT] + "\n…[truncated]"
        lines += ["", text, ""]
    lines += ["# Thread entries", ""]
    for entry in evidence.thread_entries:
        receipts = [r.get("capability_id") for r in entry.get("effect_receipts") or ()]
        provenance = entry.get("provenance") or {}
        lines.append(
            f"- turn {entry.get('turn')} mode={entry.get('mode_id')} "
            f"handoff={entry.get('handoff_to') or '-'} model={provenance.get('model_id', '-')} "
            f"exit={provenance.get('operator_exit') or '-'} receipts={receipts}"
        )
    return "\n".join(lines) + "\n"


def write_bundle(run_dir: Path, payload: dict[str, Any], evidence: Evidence) -> Path:
    (run_dir / "outcome.json").write_text(
        json.dumps(evidence.outcome.to_json(), indent=1), encoding="utf-8"
    )
    (run_dir / "transcript.md").write_text(_transcript(evidence), encoding="utf-8")
    path = run_dir / "verdict.json"
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return path
