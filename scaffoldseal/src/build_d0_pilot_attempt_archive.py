"""Build the D-036 zero-training D0 attempt/provenance archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from d0_pilot_governance import (
    LOSSLESS_PREDICTION_SCHEMA,
    canonical_json_sha256,
    corrected_joint_lobo_projection,
    frozen_pilot_identity,
    inspect_checkpoint_read_only,
    inventory_attempt_roots,
    stream_sha256,
    summarize_cumulative_execution,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_PROTOCOL_SHA256 = "b5fc086cb2c6fb81e9cd009fbaed24cdb62405b22192899255ff82e762c91680"
PROTECTED_R0_HASHES = {
    "analysis_all_labels.csv": "d8679508392c943068e2797f30492d716cf0e20385e6f6f24fb5d642ff52981d",
    "comparison_fold_manifest.csv": "b45d19a9b939a8a824842cf1aa20ef9a12383f9a329dd694bfeb9bf163d72b40",
    "data_manifest_raw.sha256": "b091a8297b16a5db1031c1ffb7e204b32292264f5c4ad76e98fbda423dc413ac",
    "inner_basket_manifest.csv": "42bdf42099f9bd7c8540d6e45d596919b6489fb997272b50344d286f2ed8db73",
    "outer_fold_manifest.csv": "f53bea95dd9635a7dd815b9ba0ac20bc3be176ad7427f17dc441927f450237e6",
    "outer_record_assignments.csv": "f655b5129bdd69485b3381e0f5d54693ac184b3f98f4e85ee93cd04e57cb5eb9",
    "pre_fit_contract_manifest.csv": "3bc173d5a3253bfb13b2a0d676cc6939609680b078e14b499eda0fff706779c0",
}


def directory_evidence(relative_directory: str) -> dict[str, object]:
    root = PROJECT_ROOT / relative_directory
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        files.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": stream_sha256(path),
            }
        )
    return {
        "relative_directory": relative_directory,
        "file_count": len(files),
        "files_canonical_sha256": canonical_json_sha256(files),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-root", type=Path, required=True)
    parser.add_argument("--v2-root", type=Path, required=True)
    parser.add_argument("--v3-root", type=Path, required=True)
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "r1c0_dmpnn_attempt_archive_v1",
    )
    args = parser.parse_args()
    attempt_roots = {"v1": args.v1_root, "v2": args.v2_root, "v3": args.v3_root}
    archive_dir = args.archive_dir.resolve()
    archive_dir.mkdir(parents=True, exist_ok=True)

    inventory = inventory_attempt_roots(attempt_roots, archive_dir)
    write_json(archive_dir / "attempt_inventory.json", inventory)

    partial_checkpoint = inspect_checkpoint_read_only(
        args.v2_root
        / "pampa"
        / "outer_01"
        / "inner_04"
        / "d0_locked"
        / "seed_0"
        / "checkpoint1.pt",
        restored_checkpoint_path=args.v2_root / "pretraining_a" / "checkpoint1.pt",
        training_rows=6370,
        batch_size=64,
    )
    v3_inner04_checkpoint = (
        args.v3_root
        / "pampa"
        / "outer_01"
        / "inner_04"
        / "d0_locked"
        / "seed_0"
        / "checkpoint1.pt"
    )
    partial_checkpoint["v3_inner04_checkpoint_sha256"] = stream_sha256(
        v3_inner04_checkpoint
    )
    partial_checkpoint["matches_v3_inner04_best_checkpoint"] = (
        partial_checkpoint["checkpoint_sha256_before_after"]
        == partial_checkpoint["v3_inner04_checkpoint_sha256"]
    )
    if not partial_checkpoint["matches_v3_inner04_best_checkpoint"]:
        raise RuntimeError("v2 partial checkpoint no longer matches retained v3 inner-4 evidence")
    write_json(archive_dir / "v2_inner04_checkpoint_inspection.json", partial_checkpoint)

    cumulative = summarize_cumulative_execution(attempt_roots, partial_checkpoint)
    write_json(archive_dir / "cumulative_execution.json", cumulative)

    v3_inner = []
    for basket in range(1, 5):
        path = (
            args.v3_root
            / "pampa"
            / "outer_01"
            / f"inner_{basket:02d}"
            / "d0_locked"
            / "seed_0"
            / "training_trace.json"
        )
        v3_inner.append(json.loads(path.read_text(encoding="utf-8")))
    v3_outer_path = (
        args.v3_root
        / "pampa"
        / "outer_01"
        / "outer_refit"
        / "d0_locked"
        / "seed_0"
        / "training_trace.json"
    )
    v3_outer = json.loads(v3_outer_path.read_text(encoding="utf-8"))
    v3_pretraining = json.loads(
        (args.v3_root / "pretraining_a" / "pretraining_trace.json").read_text(
            encoding="utf-8"
        )
    )
    assignments = pd.read_csv(PROJECT_ROOT / "artifacts" / "v2_r0" / "outer_record_assignments.csv")
    corrected_projection = corrected_joint_lobo_projection(
        v3_inner, v3_outer, assignments, float(v3_pretraining["runtime_seconds"])
    )
    corrected_projection["original_v3_reported_gpu_hours"] = 49.2420571224639
    corrected_projection["difference_gpu_hours"] = (
        float(corrected_projection["projected_joint_lobo_gpu_hours"])
        - corrected_projection["original_v3_reported_gpu_hours"]
    )
    write_json(archive_dir / "corrected_resource_projection.json", corrected_projection)

    committed = {
        "schema_version": "scaffoldseal-d0-committed-candidate-hashes-v1",
        "directories": [
            directory_evidence("artifacts/r1c0_dmpnn_pilot_v2"),
            directory_evidence("artifacts/r1c0_dmpnn_pilot_v3"),
        ],
    }
    write_json(archive_dir / "committed_candidate_hashes.json", committed)

    protected = []
    for name, expected in PROTECTED_R0_HASHES.items():
        path = PROJECT_ROOT / "artifacts" / "v2_r0" / name
        observed = stream_sha256(path)
        protected.append(
            {
                "relative_path": f"artifacts/v2_r0/{name}",
                "expected_sha256": expected,
                "observed_sha256": observed,
                "match": observed == expected,
            }
        )
    accepted_r1 = [
        directory_evidence("artifacts/r1a_classical"),
        directory_evidence("artifacts/r1b1_random_forest"),
        directory_evidence("artifacts/r1b2_xgboost"),
    ]
    protected_payload = {
        "schema_version": "scaffoldseal-protected-artifact-check-v1",
        "r0": protected,
        "all_r0_match": all(item["match"] for item in protected),
        "accepted_r1_read_only_snapshots": accepted_r1,
    }
    write_json(archive_dir / "protected_artifact_check.json", protected_payload)
    if not protected_payload["all_r0_match"]:
        raise RuntimeError("A protected R0 artifact hash differs")

    prediction_format = {
        "schema_version": LOSSLESS_PREDICTION_SCHEMA,
        "purpose": "lossless future D0 prediction interchange plus readable CSV companion",
        "record_fields": {
            "curated_id": "UTF-8 string",
            "outer_fold": "exact integer",
            "seed": "exact integer",
            "observed_log10_papp_ieee754_be": "16 lowercase hex digits",
            "prediction_normalized_ieee754_be": "16 lowercase hex digits",
            "prediction_log10_papp_ieee754_be": "16 lowercase hex digits",
        },
        "float_encoding": "finite IEEE-754 binary64, big-endian hexadecimal",
        "integrity": "canonical JSON SHA-256 over ordered records",
        "readable_companion": "CSV with 17 significant digits; JSON remains authoritative",
        "retroactive_v3_conversion_performed": False,
        "model_execution_required": False,
    }
    write_json(archive_dir / "lossless_prediction_format.json", prediction_format)

    protocol_path = PROJECT_ROOT / "d0_pilot_protocol.json"
    if stream_sha256(protocol_path) != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError("Frozen pilot protocol hash differs")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    identity = frozen_pilot_identity(protocol)
    ledger = {
        "schema_version": "scaffoldseal-single-dispatch-ledger-v1",
        "scientific_identity_sha256": identity,
        "status": "RETIRED_AFTER_CUMULATIVE_EXECUTION_DEVIATION",
        "attempt": {
            "retained_namespaces": {
                label: str(path.resolve()) for label, path in sorted(attempt_roots.items())
            },
            "governing_decision": "D-036",
            "full_lobo_started": False,
        },
        "one_shot": True,
        "relaunch_under_new_namespace_allowed": False,
        "scientific_execution_authorized": False,
    }
    ledger_path = archive_dir / "dispatch_ledger" / f"{identity}.json"
    if ledger_path.exists():
        if json.loads(ledger_path.read_text(encoding="utf-8")) != ledger:
            raise RuntimeError("Existing historical dispatch ledger differs")
    else:
        write_json(ledger_path, ledger)

    archive_files = []
    for path in sorted(item for item in archive_dir.rglob("*") if item.is_file()):
        if path.name == "archive_manifest.json":
            continue
        archive_files.append(
            {
                "relative_path": path.relative_to(archive_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": stream_sha256(path),
            }
        )
    forbidden_copies = [
        item
        for item in archive_files
        if Path(str(item["relative_path"])).suffix.lower() in {".pt", ".npy", ".gzip", ".csv"}
    ]
    if forbidden_copies:
        raise RuntimeError(f"Large/data binary copied into attempt archive: {forbidden_copies}")
    archive_manifest = {
        "schema_version": "scaffoldseal-d0-attempt-archive-manifest-v1",
        "scientific_training_performed": False,
        "optimizer_or_weight_updates_performed": False,
        "full_lobo_dispatcher_present": False,
        "full_lobo_authorized": False,
        "future_runner_requires_prefit_review": True,
        "archive_files": archive_files,
        "archive_files_canonical_sha256": canonical_json_sha256(archive_files),
    }
    write_json(archive_dir / "archive_manifest.json", archive_manifest)
    print(
        json.dumps(
            {
                "status": "PASS",
                "files_hashed": inventory["totals"]["file_count"],
                "bytes_hashed": inventory["totals"]["size_bytes"],
                "cumulative_optimizer_updates_lower_bound": cumulative["cumulative"][
                    "optimizer_updates"
                ],
                "projected_gpu_hours": corrected_projection[
                    "projected_joint_lobo_gpu_hours"
                ],
                "scientific_identity_sha256": identity,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
