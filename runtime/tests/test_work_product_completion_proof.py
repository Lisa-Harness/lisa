"""Producer-bound work products cannot bypass reconciliation at commit resume."""

from pathlib import Path

from runtime.code_artifacts import _completion_proof_problem
from sdk.workproducts import WorkProduct, WorkProductInput
from sdk.workspace_documents import WorkspaceDocuments


def test_required_producer_and_fresh_inputs(tmp_path: Path) -> None:
    """A matching commit alone cannot replace a required producer's proof."""
    root = tmp_path / "project"
    root.mkdir()
    private = tmp_path / "private"
    documents = WorkspaceDocuments(root, private)
    digest = documents.write("gate.md", "contract", expected_digest="")
    artifact = WorkProduct(
        workspace_id="test",
        user_id="user",
        user_slug="user",
        surface_type="personal_dm",
        surface_id="worker",
        session_plan_digest="plan",
        source_root=str(root),
        revision=1,
        commit_sha="commit",
        completion_producer_id="reconcile",
    )
    assert _completion_proof_problem(artifact, producer="reconcile", authority_root=private)
    artifact.completion_revision = 1
    artifact.completion_commit_sha = "commit"
    artifact.completion_inputs = [WorkProductInput(root=str(root), path="gate.md", sha256=digest)]
    assert not _completion_proof_problem(artifact, producer="reconcile", authority_root=private)
    assert _completion_proof_problem(artifact, producer="raw_commit", authority_root=private)
    documents.write("gate.md", "changed", expected_digest=digest)
    assert _completion_proof_problem(artifact, producer="reconcile", authority_root=private)
    artifact.completion_inputs[0].root = str(tmp_path / "absent")
    assert "cannot be verified" in _completion_proof_problem(
        artifact,
        producer="reconcile",
        authority_root=private,
    )
