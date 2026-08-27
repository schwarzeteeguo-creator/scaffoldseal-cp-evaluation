"""Runtime-only fail-closed governance for D123 label release."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import pandas as pd


def stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _by_id(plan: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    stages = plan.get("stages")
    if not isinstance(stages, list):
        raise RuntimeError("D123 plan stages are absent")
    result = {
        str(stage["scientific_identity_sha256"]): stage
        for stage in stages
        if isinstance(stage, Mapping)
    }
    if len(result) != len(stages):
        raise RuntimeError("D123 plan identities are missing or duplicated")
    return result


def _one_variant_stage(
    plan: Mapping[str, object], kind: str, variant: str
) -> Mapping[str, object]:
    matches = [
        stage
        for stage in plan["stages"]
        if stage["kind"] == kind and stage["key"].get("variant") == variant
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {kind} stage for {variant}")
    return matches[0]


def accepted_output_if_valid(stage, ledger, project_root: Path) -> Path | None:
    """Return an accepted output, rejecting corrupted accepted evidence."""

    root = project_root.resolve()
    relative = ledger.accepted_output_namespace(stage)
    if relative is None:
        return None
    output = (root / str(relative)).resolve()
    try:
        output.relative_to(root)
    except ValueError as error:
        raise RuntimeError("Accepted D123 namespace escapes project") from error
    if not ledger.completed_artifacts_valid(stage, output):
        raise RuntimeError("Accepted D123 stage failed artifact validation")
    return output


def next_dependency_ready_stage(
    plan: Mapping[str, object], ledger, project_root: Path
) -> Mapping[str, object] | None:
    """Select the first incomplete stage whose dependencies are accepted-valid."""

    by_id = _by_id(plan)
    for stage in plan["stages"]:
        if accepted_output_if_valid(stage, ledger, project_root) is not None:
            continue
        dependencies = tuple(map(str, stage["dependencies"]))
        for identity in dependencies:
            if identity not in by_id:
                raise RuntimeError("D123 stage dependency is outside the plan")
        ready = all(
            accepted_output_if_valid(by_id[identity], ledger, project_root) is not None
            for identity in dependencies
        )
        if ready:
            return stage
    incomplete = [
        stage
        for stage in plan["stages"]
        if accepted_output_if_valid(stage, ledger, project_root) is None
    ]
    if incomplete:
        raise RuntimeError("D123 ledger has incomplete stages but no dependency-ready stage")
    return None


def claim_next_dependency_ready_stage(
    plan: Mapping[str, object],
    ledger,
    project_root: Path,
    *,
    recovery_reason: str,
    inspector=None,
):
    """Recover a provably dead owner, then claim exactly one ready identity.

    The underlying append-only ledger remains the authority for live-owner
    proof and claim exclusivity.  A live owner, unreadable owner evidence, or
    a completed/hash-invalid stage therefore fails closed instead of being
    bypassed here.
    """

    if not recovery_reason.strip():
        raise ValueError("D123 recovery requires an auditable reason")
    stage = next_dependency_ready_stage(plan, ledger, project_root)
    if stage is None:
        return None
    latest = ledger.latest_attempt(stage)
    recovered = None
    if latest is not None and latest.get("status") == "CLAIMED_BEFORE_NAMESPACE_OR_MODEL":
        kwargs = {"reason": recovery_reason}
        if inspector is not None:
            kwargs["inspector"] = inspector
        recovered = ledger.recover_interrupted(stage, **kwargs)
        if recovered.get("status") != "INTERRUPTED_RECORDED":
            raise RuntimeError("D123 dead-owner recovery was not durably recorded")
    claim = ledger.claim(
        stage,
        attempt_metadata={
            "d123_plan_sha256": str(plan["plan_sha256"]),
            "stage_kind": str(stage["kind"]),
            "recovered_previous_attempt": recovered is not None,
        },
    )
    if claim.scientific_identity_sha256 != stage["scientific_identity_sha256"]:
        raise RuntimeError("D123 ledger returned a claim for another identity")
    return stage, claim


def write_stage_artifact_manifest(
    stage: Mapping[str, object], output_root: Path
) -> dict[str, object]:
    """Seal all non-manifest outputs before ledger completion."""

    root = output_root.resolve()
    expected = tuple(map(str, stage["expected_outputs"]))
    if expected.count("artifact_manifest.json") != 1:
        raise RuntimeError("D123 stage must expect exactly one artifact manifest")
    records = []
    for relative in expected:
        if relative == "artifact_manifest.json":
            continue
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise RuntimeError("D123 artifact escapes output namespace") from error
        if not path.is_file():
            raise RuntimeError(f"D123 expected artifact is absent: {relative}")
        records.append(
            {
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": stream_sha256(path),
            }
        )
    payload = {
        "schema_version": "scaffoldseal-d123-artifact-manifest-v1",
        "scientific_identity_sha256": str(stage["scientific_identity_sha256"]),
        "stage_spec_sha256": str(stage["stage_spec_sha256"]),
        "artifacts": records,
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    target = root / "artifact_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(
            str(target),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
    except FileExistsError as error:
        raise RuntimeError("Refusing to overwrite immutable D123 artifact manifest") from error
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return payload


def complete_claimed_stage(
    stage: Mapping[str, object],
    claim,
    ledger,
    output_root: Path,
    resource_evidence: Mapping[str, object],
):
    """Write the last immutable artifact, then let the ledger verify all bytes."""

    write_stage_artifact_manifest(stage, output_root)
    return ledger.record_completed(
        claim, stage, output_root.resolve(), resource_evidence
    )


def fail_claimed_stage(
    claim, ledger, *, reason: str, resource_evidence: Mapping[str, object]
) -> None:
    """Durably record a failed attempt without deleting its evidence."""

    if not reason.strip():
        raise ValueError("D123 stage failure requires an auditable reason")
    ledger.record_failed(claim, reason, resource_evidence)


def validate_label_release_dependencies(
    plan: Mapping[str, object], variant: str, ledger, project_root: Path
) -> Mapping[str, object]:
    """Validate all sealed predictions without opening the metric label file."""

    root = project_root.resolve()
    gate = _one_variant_stage(plan, "d123_metric_label_release_gate", variant)
    by_id = _by_id(plan)
    dependencies = tuple(map(str, gate["dependencies"]))
    expected = {
        str(stage["scientific_identity_sha256"])
        for stage in plan["stages"]
        if stage["kind"] == "d123_pampa_outer_fit_prediction"
        and stage["key"].get("variant") == variant
    }
    if len(dependencies) != 90 or set(dependencies) != expected:
        raise RuntimeError("Label gate does not bind the exact 90 variant predictions")
    for identity in dependencies:
        stage = by_id[identity]
        output = accepted_output_if_valid(stage, ledger, root)
        if output is None:
            raise RuntimeError("Label gate dependency is not COMPLETED_ACCEPTED")
    return gate


def write_label_release_receipt(
    plan: Mapping[str, object],
    variant: str,
    ledger,
    project_root: Path,
    output_root: Path,
) -> dict[str, object]:
    """Seal proof of all predictions without opening the metric-label file."""

    gate = validate_label_release_dependencies(plan, variant, ledger, project_root)
    by_id = _by_id(plan)
    receipts = []
    for identity in map(str, gate["dependencies"]):
        prediction_stage = by_id[identity]
        output = accepted_output_if_valid(prediction_stage, ledger, project_root)
        if output is None:
            raise RuntimeError("Label-release receipt lacks an accepted prediction")
        latest = ledger.latest_attempt(prediction_stage)
        if latest is None or latest.get("status") != "COMPLETED_ACCEPTED":
            raise RuntimeError("Prediction ledger receipt is not completed")
        artifacts = latest.get("details", {}).get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise RuntimeError("Prediction ledger receipt lacks artifact hashes")
        receipts.append(
            {
                "scientific_identity_sha256": identity,
                "stage_spec_sha256": str(prediction_stage["stage_spec_sha256"]),
                "output_namespace": str(
                    output.resolve().relative_to(project_root.resolve())
                ).replace("\\", "/"),
                "artifacts": artifacts,
            }
        )
    source = dict(plan["scientific_lock"]["metric_label_source"])
    payload = {
        "schema_version": "scaffoldseal-d123-label-release-receipt-v1",
        "variant": variant,
        "gate_scientific_identity_sha256": str(
            gate["scientific_identity_sha256"]
        ),
        "gate_stage_spec_sha256": str(gate["stage_spec_sha256"]),
        "prediction_receipt_count": len(receipts),
        "prediction_receipts": receipts,
        "metric_label_source_lock": source,
        "metric_label_file_opened": False,
    }
    if len(receipts) != 90:
        raise RuntimeError("Label-release receipt must bind exactly 90 predictions")
    payload["receipt_sha256"] = canonical_sha256(payload)
    target = output_root.resolve() / "label_release_receipt.json"
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            str(target),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
    except FileExistsError as error:
        raise RuntimeError("Refusing to overwrite label-release receipt") from error
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return payload


def load_metric_labels_after_accepted_gate(
    plan: Mapping[str, object], variant: str, ledger, project_root: Path
) -> pd.DataFrame:
    """Open labels only after the gate itself is accepted and hash-valid."""

    root = project_root.resolve()
    gate = validate_label_release_dependencies(plan, variant, ledger, root)
    gate_output = accepted_output_if_valid(gate, ledger, root)
    if gate_output is None:
        raise RuntimeError("Metric labels remain sealed until the release gate is accepted")

    source = plan["scientific_lock"]["metric_label_source"]
    label_path = (root / str(source["relative_path"])).resolve()
    try:
        label_path.relative_to(root)
    except ValueError as error:
        raise RuntimeError("Metric label source escapes project") from error
    if (
        not label_path.is_file()
        or label_path.stat().st_size != int(source["size_bytes"])
        or stream_sha256(label_path) != str(source["sha256"])
    ):
        raise RuntimeError("Metric label source failed exact byte validation")
    labels = pd.read_csv(label_path)
    if len(labels) != 6895 or labels["curated_id"].duplicated().any():
        raise RuntimeError("Metric label population drifted after release")
    return labels
