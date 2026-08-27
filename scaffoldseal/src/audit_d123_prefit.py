"""Independent-style, zero-training structural audit for the D123 candidate."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import d123_runner


EXPECTED_COUNTS = {
    "d23_descriptor_delaney_pretraining": 5,
    "d123_pampa_inner_fit": 216,
    "d123_stopping_epoch_selection": 54,
    "d123_pampa_outer_fit_prediction": 270,
    "d123_metric_label_release_gate": 3,
    "d123_sealed_oof_metrics": 3,
}


def _must_reject(plan, root: Path, label: str) -> str:
    try:
        d123_runner.validate_plan(plan, root)
    except RuntimeError:
        return label
    raise AssertionError(f"D123 pre-fit mutation was accepted: {label}")


def audit(project_root: Path) -> dict[str, object]:
    root = project_root.resolve()
    plan_path = root / "d123_plan_candidate.json"
    acceptance_path = root / "d123_acceptance.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    d123_runner.validate_plan(plan, root)

    if any(
        acceptance.get(key) is not False
        for key in ("accepted", "execution_authorized", "gpu_training_allowed")
    ):
        raise RuntimeError("D123 pre-fit audit requires all acceptance flags false")
    if (
        acceptance.get("plan_sha256") != plan["plan_sha256"]
        or acceptance.get("plan_file_sha256")
        != d123_runner.stream_sha256(plan_path)
        or acceptance.get("runner_file_sha256")
        != d123_runner.stream_sha256(Path(d123_runner.__file__))
    ):
        raise RuntimeError("D123 pending acceptance is not bound to exact bytes")
    if d123_runner.PREFIT_REVIEW_SOURCE_LOCK or d123_runner.REAL_EXECUTION_SOURCE_LOCK:
        raise RuntimeError("D123 source execution lock was enabled before review")

    counts = {
        kind: sum(stage["kind"] == kind for stage in plan["stages"])
        for kind in EXPECTED_COUNTS
    }
    if counts != EXPECTED_COUNTS or len(plan["stages"]) != 551:
        raise RuntimeError("D123 stage geometry drifted")
    identities = {
        str(stage["scientific_identity_sha256"]) for stage in plan["stages"]
    }
    namespaces = {
        str(stage["namespace"][key])
        for stage in plan["stages"]
        for key in ("work_attempt_template", "output_attempt_template")
    }
    if len(identities) != 551 or len(namespaces) != 1102:
        raise RuntimeError("D123 identities or namespaces are not unique")

    methods = (
        "_run_descriptor_pretraining_stage",
        "_run_inner_stage",
        "_run_selection_stage",
        "_run_outer_stage",
        "_run_label_gate_stage",
        "_run_metric_stage",
        "_dispatch_stage",
        "run",
    )
    if not all(callable(getattr(d123_runner.D123Executor, name, None)) for name in methods):
        raise RuntimeError("D123 executor lacks a required stage body")

    mutations = []
    changed = copy.deepcopy(plan)
    changed["stages"][0]["key"]["global_features_size"] = 26
    mutations.append(_must_reject(changed, root, "scientific_identity"))
    changed = copy.deepcopy(plan)
    changed["stages"][0]["stage_spec_sha256"] = "0" * 64
    mutations.append(_must_reject(changed, root, "stage_spec"))
    changed = copy.deepcopy(plan)
    changed["stages"][0]["expected_outputs"][0] = "../escape"
    mutations.append(_must_reject(changed, root, "artifact_escape"))
    changed = copy.deepcopy(plan)
    changed["protocol_lock_sha256"] = "0" * 64
    mutations.append(_must_reject(changed, root, "protocol_lock"))

    preexecution_paths = (
        root / "artifacts/d123_ledger_v1",
        root / "runs/d123_v1",
        root / "artifacts/d123_v1",
    )
    existing = [path.as_posix() for path in preexecution_paths if path.exists()]
    if existing:
        raise RuntimeError(f"D123 pre-fit audit found execution namespaces: {existing}")

    try:
        d123_runner.assert_execution_authorized(
            plan, acceptance, plan_path, Path(d123_runner.__file__)
        )
    except RuntimeError as error:
        if "source locks" not in str(error):
            raise
    else:
        raise RuntimeError("D123 execution was authorized during zero-training audit")

    return {
        "schema_version": "scaffoldseal-d123-prefit-audit-v1",
        "status": "PASS_ZERO_TRAINING_CANDIDATE",
        "plan_sha256": plan["plan_sha256"],
        "stages": 551,
        "scientific_fits": 491,
        "identities": len(identities),
        "namespaces": len(namespaces),
        "stage_counts": counts,
        "mutation_rejections": mutations,
        "execution_namespaces_absent": True,
        "acceptance_all_false": True,
        "source_locks_false": True,
        "gpu_training_invoked": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Refusing to overwrite D123 pre-fit audit")
    report = audit(args.project_root)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
