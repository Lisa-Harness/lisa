"""Tests for the ``/policy`` command — render, mode dial, bare toggle.

``/policy`` is the host-side surface for the approval policy engine:

- bare ``/policy`` renders the active document (mode, guards, rule lists).
- ``/policy mode <m>`` persists a new fallback dial to ``policy.yaml``
  (preserving any rule lists) and the reply confirms it.
- ``/policy bare on|off`` toggles ``bare_requires_approval``; bad input
  echoes the current state and does not write.
- Unknown subcommands return usage, never a crash.

Writes go through ``host.policy.write_policy`` so the document stays
hand-editable; the engine re-reads the file on every approval request, so an
edit must be observable by the very next ``load_policy``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from host.commands import handle_policy_command, render_policy_document
from host.policy import PolicyDocument, load_policy


@pytest.fixture(autouse=True)
def _clean_timezone_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop LISA_TIMEZONE/TZ before each test.

    The full suite's env-propagation tests export ``LISA_TIMEZONE`` into
    ``os.environ`` (compile_mcp_configs writes the host's resolved zone for
    subprocesses). On hosts whose system-local tz is a POSIX alias
    (``EDT`` → ``EST5EDT``) that exported value is not re-resolvable by
    ``HostConfig.from_env``, which poisons every later ``build_test_runtime``
    call. This file builds runtimes directly, so scrub the var first.
    """
    for env_name in ("LISA_TIMEZONE", "TZ", "TIMEZONE", "LOCAL_TIMEZONE"):
        monkeypatch.delenv(env_name, raising=False)


def _make_runtime(tmp_path: Path) -> object:
    from host.testing import build_test_runtime

    models_yaml = "\n".join(
        [
            "version: 1",
            "mode_models: {default: gemma3:latest}",
            "models:",
            "  - model_id: gemma3:latest",
            "    provider: ollama",
        ]
    )
    return build_test_runtime(tmp_path, models_yaml=models_yaml)


# ── Render ────────────────────────────────────────────────────────────


def test_render_policy_document_defaults(tmp_path: Path) -> None:
    """A missing policy.yaml renders the fail-closed defaults."""
    runtime = _make_runtime(tmp_path)
    text = render_policy_document(runtime)  # type: ignore[arg-type]
    assert "## Approval Policy" in text
    assert "mode: **ask**" in text
    assert "sandbox_downgrade: true" in text
    assert "bare_requires_approval: true" in text
    assert "deny rules (0)" in text
    assert "(none)" in text


def _write_doc(tmp_path: Path, doc: PolicyDocument) -> None:
    """Serialize a PolicyDocument to config/policy.yaml (mode as its string value)."""
    payload = doc.model_dump()
    payload["mode"] = doc.mode.value
    path = tmp_path / "config" / "policy.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_render_policy_document_reflects_file(tmp_path: Path) -> None:
    """An existing policy.yaml's mode and rules appear in the render."""
    runtime = _make_runtime(tmp_path)
    doc = PolicyDocument(
        mode="trust",
        allow=[{"tool": "bash_run", "command_prefix": "git status"}],
    )
    _write_doc(tmp_path, doc)
    text = render_policy_document(runtime)  # type: ignore[arg-type]
    assert "mode: **trust**" in text
    assert "allow rules (1)" in text
    assert "git status" in text


# ── Mode dial ─────────────────────────────────────────────────────────


def test_policy_mode_set_persists_and_preserves_rules(tmp_path: Path) -> None:
    """mode trust writes the dial and keeps existing rule lists."""
    runtime = _make_runtime(tmp_path)
    doc = PolicyDocument(deny=[{"tool": "bash_run"}])
    _write_doc(tmp_path, doc)
    reply = handle_policy_command(runtime, "/policy mode trust", "/policy mode trust")  # type: ignore[arg-type]
    assert "**trust**" in reply
    reloaded = load_policy(tmp_path)
    assert reloaded.mode.value == "trust"
    # The pre-existing deny rule survived the write.
    assert any(r.tool == "bash_run" for r in reloaded.deny)


def test_policy_mode_unknown_is_rejected_without_write(tmp_path: Path) -> None:
    """An invalid mode returns usage and leaves the document untouched."""
    runtime = _make_runtime(tmp_path)
    reply = handle_policy_command(runtime, "/policy mode banana", "/policy mode banana")  # type: ignore[arg-type]
    assert "Unknown mode" in reply
    assert not (tmp_path / "config" / "policy.yaml").exists()


# ── Bare toggle ───────────────────────────────────────────────────────


def test_policy_bare_off_toggles_guard(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    reply = handle_policy_command(runtime, "/policy bare off", "/policy bare off")  # type: ignore[arg-type]
    assert "**false**" in reply
    assert load_policy(tmp_path).bare_requires_approval is False


def test_policy_bare_bad_input_reports_current(tmp_path: Path) -> None:
    """Bad bare input echoes the current state and writes nothing."""
    runtime = _make_runtime(tmp_path)
    reply = handle_policy_command(runtime, "/policy bare", "/policy bare")  # type: ignore[arg-type]
    assert "currently true" in reply
    assert not (tmp_path / "config" / "policy.yaml").exists()


# ── Unknown / usage ───────────────────────────────────────────────────


def test_policy_unknown_subcommand_returns_usage(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    reply = handle_policy_command(runtime, "/policy frobnicate", "/policy frobnicate")  # type: ignore[arg-type]
    assert "Usage: /policy" in reply


def test_policy_mode_roundtrip_via_engine(tmp_path: Path) -> None:
    """End-to-end: set trust, then the engine's decide() honors it.

    A write-mode 'write' action would prompt under ask but is allowed under
    trust (no rules, no containment consulted for non-execute effects).
    """
    from host.policy import decide

    runtime = _make_runtime(tmp_path)
    from kernel.approval import ApprovalIdentity, build_approval_invocation

    identity = ApprovalIdentity(workspace_id="w", user_slug="u")
    invocation = build_approval_invocation(
        target_kind="tool",
        target_id="write_file",
        pack_id="lisa-pack-core",
        effect="write",
        parameters={"path": "x.txt"},
        identity=identity,
        pack_config={},
        safe_preview="write x.txt",
    )
    # Default ask mode -> prompt.
    policy = load_policy(tmp_path)
    assert decide(policy=policy, invocation=invocation, data_dir=tmp_path).verdict == "ask"
    handle_policy_command(runtime, "/policy mode trust", "/policy mode trust")  # type: ignore[arg-type]
    policy = load_policy(tmp_path)
    assert decide(policy=policy, invocation=invocation, data_dir=tmp_path).verdict == "allow"


# ── Rule CRUD ─────────────────────────────────────────────────────────


def test_rule_add_persists_and_decides(tmp_path: Path) -> None:
    """A rule added via /policy rule add drives the engine immediately."""
    runtime = _make_runtime(tmp_path)
    reply = handle_policy_command(
        runtime,
        '/policy rule add deny --tool bash_run --when action.argv[0] == "rm"',
        "/policy rule add deny --tool bash_run --when ...",
    )  # type: ignore[arg-type]
    assert "Rule added to **deny**" in reply
    loaded = load_policy(tmp_path)
    assert loaded.deny[0].tool == "bash_run"
    # The double-quoted string literal survived verbatim.
    assert loaded.deny[0].when == 'action.argv[0] == "rm"'


def test_rule_add_preserves_other_lists(tmp_path: Path) -> None:
    """An add to one list never drops rules in another."""
    runtime = _make_runtime(tmp_path)
    doc = PolicyDocument(allow=[{"tool": "bash_run", "command_prefix": "git status"}])
    path = tmp_path / "config" / "policy.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc.model_dump(mode="json")), encoding="utf-8")
    handle_policy_command(runtime, "/policy rule add ask --effect external_write", "x")  # type: ignore[arg-type]
    loaded = load_policy(tmp_path)
    assert any(r.command_prefix == "git status" for r in loaded.allow)
    assert loaded.ask[0].effect == "external_write"


def test_rule_add_rejects_bad_expression_without_write(tmp_path: Path) -> None:
    """A non-compiling when is rejected and the file stays untouched."""
    runtime = _make_runtime(tmp_path)
    reply = handle_policy_command(runtime, "/policy rule add allow --when this is (bad cel", "x")  # type: ignore[arg-type]
    assert "Rejected" in reply
    loaded = load_policy(tmp_path)
    assert loaded.allow == []


def test_rule_add_rejects_empty_rule(tmp_path: Path) -> None:
    """An add with no matchers is rejected — an empty rule matches nothing."""
    runtime = _make_runtime(tmp_path)
    reply = handle_policy_command(runtime, "/policy rule add allow", "x")  # type: ignore[arg-type]
    assert "at least one matcher" in reply
    assert load_policy(tmp_path).allow == []


def test_rule_add_rejects_unknown_list(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    reply = handle_policy_command(runtime, "/policy rule add maybe --tool x", "x")  # type: ignore[arg-type]
    assert "Unknown rule list" in reply


def test_rule_add_single_quoted_expression_is_unquoted(tmp_path: Path) -> None:
    """REPL-style whole-expression quoting is stripped.

    A single-quoted remainder is valid but ALWAYS-TRUTHY CEL persisted
    verbatim — an allow rule matching everything. The quoting probe strips
    the one-token wrapper while leaving inner literals verbatim.
    """
    runtime = _make_runtime(tmp_path)
    handle_policy_command(
        runtime, "/policy rule add allow --tool bash_run --when 'action.argv.size() <= 2'", "x"
    )  # type: ignore[arg-type]
    loaded = load_policy(tmp_path)
    assert loaded.allow[0].when == "action.argv.size() <= 2"


def test_rule_add_expression_case_is_preserved(tmp_path: Path) -> None:
    """The command parses flags from the original text — never lowercased.

    A lowercased expression would silently rewrite identifiers and string
    literals (``GitStatus`` -> ``gitstatus``).
    """
    runtime = _make_runtime(tmp_path)
    handle_policy_command(
        runtime, "/policy RULE ADD ask --tool BashRun --when 'action.argv[0] == \"Make\"'", "x"
    )  # type: ignore[arg-type]
    loaded = load_policy(tmp_path)
    assert loaded.ask[0].tool == "BashRun"
    assert loaded.ask[0].when == 'action.argv[0] == "Make"'


def test_rule_remove_drops_addressed_rule(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    handle_policy_command(runtime, "/policy rule add allow --tool a", "x")  # type: ignore[arg-type]
    handle_policy_command(runtime, "/policy rule add allow --tool b", "x")  # type: ignore[arg-type]
    reply = handle_policy_command(runtime, "/policy rule remove allow:1", "x")  # type: ignore[arg-type]
    assert "Removed allow:1" in reply
    loaded = load_policy(tmp_path)
    assert [r.tool for r in loaded.allow] == ["b"]


def test_rule_remove_out_of_range_is_rejected(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    reply = handle_policy_command(runtime, "/policy rule remove allow:3", "x")  # type: ignore[arg-type]
    assert "No rule allow:3" in reply


def test_rule_remove_bad_address_is_usage(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    for bad in ("allow", "allow:", "allow:x", "maybe:1", ":1"):
        reply = handle_policy_command(runtime, f"/policy rule remove {bad}", "x")  # type: ignore[arg-type]
        assert "Usage: /policy rule remove" in reply, bad


def test_rule_list_shows_addresses(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    handle_policy_command(runtime, "/policy rule add deny --tool bash_run", "x")  # type: ignore[arg-type]
    reply = handle_policy_command(runtime, "/policy rule list", "x")  # type: ignore[arg-type]
    assert "`deny:1`" in reply
    assert "bash_run" in reply
    assert "remove" in reply


def test_rule_unknown_verb_is_usage(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    reply = handle_policy_command(runtime, "/policy rule frobnicate", "x")  # type: ignore[arg-type]
    assert "Usage: /policy rule" in reply


# ── Render with when expressions and load errors ─────────────────────


def test_render_shows_when_expression(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    doc = PolicyDocument(allow=[{"tool": "bash_run", "when": '"git" in action.argv'}])
    path = tmp_path / "config" / "policy.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc.model_dump(mode="json")), encoding="utf-8")
    text = render_policy_document(runtime)  # type: ignore[arg-type]
    assert 'when: `"git" in action.argv`' in text


def test_render_shows_load_errors_for_broken_expression(tmp_path: Path) -> None:
    """A non-compiling expression fails the document; the render says why."""
    runtime = _make_runtime(tmp_path)
    path = tmp_path / "config" / "policy.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("allow:\n  - when: $[not cel]\n", encoding="utf-8")
    text = render_policy_document(runtime)  # type: ignore[arg-type]
    assert "load errors" in text
    assert "allow rule 1" in text
    assert "ask-mode defaults active" in text


def test_render_clean_document_has_no_load_errors_section(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    text = render_policy_document(runtime)  # type: ignore[arg-type]
    assert "load errors" not in text
