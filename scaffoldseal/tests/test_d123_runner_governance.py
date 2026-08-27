import json
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from d123_runner_governance import (
    claim_next_dependency_ready_stage,
    complete_claimed_stage,
    fail_claimed_stage,
    load_metric_labels_after_accepted_gate,
    next_dependency_ready_stage,
    validate_label_release_dependencies,
    write_stage_artifact_manifest,
    write_label_release_receipt,
)


class FakeLedger:
    def __init__(self, accepted, valid):
        self.accepted = set(accepted)
        self.valid = set(valid)

    def accepted_output_namespace(self, stage):
        identity = stage["scientific_identity_sha256"]
        return f"artifacts/fake/{identity}" if identity in self.accepted else None

    def completed_artifacts_valid(self, stage, output):
        return stage["scientific_identity_sha256"] in self.valid

    def latest_attempt(self, stage):
        identity = stage["scientific_identity_sha256"]
        if identity not in self.accepted:
            return None
        return {
            "status": "COMPLETED_ACCEPTED",
            "details": {
                "artifacts": [
                    {
                        "relative_path": "predictions.lossless.json",
                        "size_bytes": 123,
                        "sha256": identity,
                    }
                ]
            },
        }


class Claim:
    def __init__(self, identity):
        self.scientific_identity_sha256 = identity


class RecoveryLedger(FakeLedger):
    def __init__(self, *, latest=None, accept_recovery=True):
        super().__init__(set(), set())
        self.latest = latest
        self.accept_recovery = accept_recovery
        self.recoveries = []
        self.claims = []

    def latest_attempt(self, stage):
        return self.latest

    def recover_interrupted(self, stage, **kwargs):
        self.recoveries.append((stage["scientific_identity_sha256"], kwargs))
        if not self.accept_recovery:
            raise RuntimeError("live owner")
        self.latest = {"status": "INTERRUPTED_RECORDED"}
        return self.latest

    def claim(self, stage, *, attempt_metadata):
        self.claims.append((stage["scientific_identity_sha256"], attempt_metadata))
        return Claim(stage["scientific_identity_sha256"])


def _plan():
    root = Path(__file__).resolve().parents[1]
    return root, json.loads((root / "d123_plan_candidate.json").read_text())


def test_gate_rejects_one_missing_or_hash_invalid_prediction():
    root, plan = _plan()
    gate = next(
        s
        for s in plan["stages"]
        if s["kind"] == "d123_metric_label_release_gate"
        and s["key"]["variant"] == "D1"
    )
    accepted = set(gate["dependencies"])
    missing = next(iter(accepted))
    try:
        validate_label_release_dependencies(
            plan, "D1", FakeLedger(accepted - {missing}, accepted), root
        )
    except RuntimeError as error:
        assert "not COMPLETED_ACCEPTED" in str(error)
    else:
        raise AssertionError("Missing prediction was accepted by label gate")
    try:
        validate_label_release_dependencies(
            plan, "D1", FakeLedger(accepted, accepted - {missing}), root
        )
    except RuntimeError as error:
        assert "artifact validation" in str(error)
    else:
        raise AssertionError("Hash-invalid prediction was accepted by label gate")


def test_labels_remain_unopened_until_gate_is_accepted(monkeypatch):
    root, plan = _plan()
    gate = next(
        s
        for s in plan["stages"]
        if s["kind"] == "d123_metric_label_release_gate"
        and s["key"]["variant"] == "D2"
    )
    reads = []

    def forbidden_read(*args, **kwargs):
        reads.append(args)
        raise AssertionError("Labels were opened before gate acceptance")

    monkeypatch.setattr("d123_runner_governance.pd.read_csv", forbidden_read)
    deps = set(gate["dependencies"])
    try:
        load_metric_labels_after_accepted_gate(
            plan, "D2", FakeLedger(deps, deps), root
        )
    except RuntimeError as error:
        assert "remain sealed" in str(error)
    else:
        raise AssertionError("Unaccepted gate released metric labels")
    assert reads == []


def test_scheduler_skips_valid_completed_and_requires_valid_dependencies():
    root, plan = _plan()
    first = plan["stages"][0]
    second = plan["stages"][1]
    assert next_dependency_ready_stage(plan, FakeLedger(set(), set()), root) == first
    assert next_dependency_ready_stage(
        plan,
        FakeLedger(
            {first["scientific_identity_sha256"]},
            {first["scientific_identity_sha256"]},
        ),
        root,
    ) == second
    try:
        next_dependency_ready_stage(
            plan,
            FakeLedger({first["scientific_identity_sha256"]}, set()),
            root,
        )
    except RuntimeError as error:
        assert "artifact validation" in str(error)
    else:
        raise AssertionError("Scheduler skipped hash-invalid accepted evidence")


def test_claim_wrapper_claims_one_ready_identity_with_plan_binding():
    root, plan = _plan()
    ledger = RecoveryLedger()
    result = claim_next_dependency_ready_stage(
        plan, ledger, root, recovery_reason="executor restart"
    )
    stage, claim = result
    assert stage == plan["stages"][0]
    assert claim.scientific_identity_sha256 == stage["scientific_identity_sha256"]
    assert ledger.recoveries == []
    assert ledger.claims[0][1] == {
        "d123_plan_sha256": plan["plan_sha256"],
        "stage_kind": stage["kind"],
        "recovered_previous_attempt": False,
    }


def test_claim_wrapper_recovers_only_through_ledger_dead_owner_proof():
    root, plan = _plan()
    live = {"status": "CLAIMED_BEFORE_NAMESPACE_OR_MODEL"}
    ledger = RecoveryLedger(latest=live)
    result = claim_next_dependency_ready_stage(
        plan,
        ledger,
        root,
        recovery_reason="provably dead prior executor",
        inspector=lambda pid: None,
    )
    stage, _ = result
    assert ledger.recoveries[0][0] == stage["scientific_identity_sha256"]
    assert ledger.claims[0][1]["recovered_previous_attempt"] is True

    rejecting = RecoveryLedger(latest=live, accept_recovery=False)
    try:
        claim_next_dependency_ready_stage(
            plan, rejecting, root, recovery_reason="must not steal live owner"
        )
    except RuntimeError as error:
        assert "live owner" in str(error)
    else:
        raise AssertionError("A live scientific identity was claimed twice")
    assert rejecting.claims == []


class TerminalLedger:
    def __init__(self):
        self.completed = []
        self.failed = []

    def record_completed(self, claim, stage, output, evidence):
        self.completed.append((claim, stage, output, evidence))
        return ["accepted"]

    def record_failed(self, claim, reason, evidence):
        self.failed.append((claim, reason, evidence))


def test_manifest_is_exclusive_and_completion_happens_last(tmp_path):
    stage = {
        "scientific_identity_sha256": "a" * 64,
        "stage_spec_sha256": "b" * 64,
        "expected_outputs": ["result.json", "artifact_manifest.json"],
    }
    (tmp_path / "result.json").write_text('{"ok":true}\\n', encoding="utf-8")
    ledger = TerminalLedger()
    claim = Claim("a" * 64)
    assert complete_claimed_stage(
        stage, claim, ledger, tmp_path, {"runtime_seconds": 0.0}
    ) == ["accepted"]
    manifest = json.loads((tmp_path / "artifact_manifest.json").read_text())
    assert manifest["scientific_identity_sha256"] == "a" * 64
    assert manifest["stage_spec_sha256"] == "b" * 64
    assert manifest["artifacts"][0]["relative_path"] == "result.json"
    assert len(manifest["artifacts"][0]["sha256"]) == 64
    assert len(manifest["manifest_sha256"]) == 64
    assert len(ledger.completed) == 1
    try:
        write_stage_artifact_manifest(stage, tmp_path)
    except RuntimeError as error:
        assert "overwrite" in str(error)
    else:
        raise AssertionError("Immutable artifact manifest was overwritten")


def test_missing_output_never_reaches_completion_and_failure_is_recorded(tmp_path):
    stage = {
        "scientific_identity_sha256": "c" * 64,
        "stage_spec_sha256": "d" * 64,
        "expected_outputs": ["missing.json", "artifact_manifest.json"],
    }
    ledger = TerminalLedger()
    claim = Claim("c" * 64)
    try:
        complete_claimed_stage(stage, claim, ledger, tmp_path, {})
    except RuntimeError as error:
        assert "absent" in str(error)
    else:
        raise AssertionError("A stage completed without its expected artifact")
    assert ledger.completed == []
    fail_claimed_stage(
        claim,
        ledger,
        reason="zero-training synthetic failure",
        resource_evidence={"runtime_seconds": 0.0},
    )
    assert ledger.failed[0][1] == "zero-training synthetic failure"


def test_label_release_receipt_binds_90_predictions_without_reading_labels(
    tmp_path, monkeypatch
):
    root, plan = _plan()
    gate = next(
        stage
        for stage in plan["stages"]
        if stage["kind"] == "d123_metric_label_release_gate"
        and stage["key"]["variant"] == "D3"
    )
    dependencies = set(gate["dependencies"])
    ledger = FakeLedger(dependencies, dependencies)
    reads = []

    def forbidden_read(*args, **kwargs):
        reads.append(args)
        raise AssertionError("Metric labels were opened while writing gate receipt")

    monkeypatch.setattr("d123_runner_governance.pd.read_csv", forbidden_read)
    payload = write_label_release_receipt(
        plan, "D3", ledger, root, tmp_path
    )
    assert payload["prediction_receipt_count"] == 90
    assert payload["metric_label_file_opened"] is False
    assert len(payload["receipt_sha256"]) == 64
    assert reads == []
    observed = json.loads((tmp_path / "label_release_receipt.json").read_text())
    assert observed == payload
