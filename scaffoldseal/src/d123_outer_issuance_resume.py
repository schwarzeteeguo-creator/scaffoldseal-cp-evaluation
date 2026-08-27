"""Fail-closed D123 resume for authoritative outer train/test record issuance."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import d123_external_dependency_resume as base
import d123_runner as runner


AMENDMENT_NAME = "d123_outer_issuance_resume_amendment.json"
EXPECTED_BASE_WRAPPER_SHA256 = (
    "e38fdac94de301482591a5868f8b81819d550a8435d18dbfa4e2f34886474044"
)


def authoritative_outer_records(frames):
    """Combine labeled train rows with strictly label-free outer-test features."""
    import numpy as np
    import pandas as pd

    train = frames["outer_train"].copy(deep=True)
    test = frames["outer_test_label_free"].copy(deep=True)
    if "normalized_pampa" in test.columns:
        raise RuntimeError("D123 outer-test target unexpectedly materialized")
    test["normalized_pampa"] = np.nan
    if set(train.columns) != set(test.columns):
        raise RuntimeError("D123 authoritative outer frame schema differs")
    combined = pd.concat([train, test.loc[:, train.columns]], ignore_index=True)
    if combined["curated_id"].astype(str).duplicated().any():
        raise RuntimeError("D123 authoritative outer frame contains duplicate IDs")
    return combined


class OuterIssuanceResumeExecutor(runner.D123Executor):
    """Preserve the frozen runner except for complete outer-record issuance."""

    def _run_outer_stage(self, stage, claim, work: Path, output: Path) -> None:
        import numpy as np
        import pandas as pd
        from d123_fold_data import load_fold_frames
        from d123_runner_governance import complete_claimed_stage
        from d123_sealed_outputs import seal_outer_predictions
        from h1_random_cv_runner import resource_evidence_from_trace
        from split_safe import FitAuditTrail, SplitSafeFitExecutor

        frames = load_fold_frames(self.plan, stage, self.project_root)
        adapter, contract = self._load_adapter_and_contract(stage, frames)
        checkpoint, checkpoint_sha = self._pretrained_checkpoint(stage)
        selection_dependencies = [
            self.by_identity.get(str(identity))
            for identity in stage["dependencies"]
            if str(identity) in self.by_identity
            and self.by_identity[str(identity)]["kind"]
            == "d123_stopping_epoch_selection"
        ]
        if len(selection_dependencies) != 1:
            raise RuntimeError("D123 outer fit lacks one stopping selection")
        selection = selection_dependencies[0]
        if (
            selection["key"]["variant"] != stage["key"]["variant"]
            or int(selection["key"]["outer_fold"])
            != int(stage["key"]["outer_fold"])
        ):
            raise RuntimeError("D123 outer fit binds another stopping selection")
        selection_receipt = json.loads(
            (self._accepted_output(selection) / "stopping_epoch.json").read_text(
                encoding="utf-8"
            )
        )
        fixed_epoch = int(selection_receipt["selected_epoch"])
        if (
            selection_receipt.get("variant") != stage["key"]["variant"]
            or int(selection_receipt.get("outer_fold"))
            != int(stage["key"]["outer_fold"])
            or selection_receipt.get("rule") != "ceil_median_4"
            or fixed_epoch < 1
            or fixed_epoch > 2000
        ):
            raise RuntimeError("D123 selected epoch receipt drifted")
        audit = FitAuditTrail()
        executor = SplitSafeFitExecutor(contract, audit)
        run = contract.mint_run_context(
            work / "checkpoint_root",
            config_id=f"d123_{stage['key']['variant']}",
            seed=int(stage["key"]["effective_rng_seed"]),
            inner_basket=None,
            pretrained_checkpoint=checkpoint,
            pretrained_checkpoint_sha256=checkpoint_sha,
        )
        self.ledger.mark_scientific_execution_started(claim)
        with self.ledger.lease_heartbeat(claim):
            handle = executor.fit_outer_frame(
                adapter,
                contract.outer_frame_training_batch(
                    authoritative_outer_records(frames)
                ),
                feature_columns=["SMILES"],
                target_column="normalized_pampa",
                run_context=run,
                fixed_epoch=fixed_epoch,
            )
            prediction = executor.predict_outer_frame(handle)
        checkpoint_dir = Path(run.checkpoint_dir)
        for name in ("checkpoint1.pt", "training_trace.json"):
            shutil.copy2(checkpoint_dir / name, output / name)
        if str(stage["key"]["variant"]) in ("D2", "D3"):
            shutil.copy2(
                checkpoint_dir / "descriptor_transform.json",
                output / "descriptor_transform.json",
            )
        ids = tuple(map(str, prediction.ids))
        values = np.asarray(prediction.predictions, dtype=np.float64).reshape(-1)
        if ids != tuple(
            sorted(frames["outer_test_label_free"]["curated_id"].astype(str))
        ):
            raise RuntimeError("D123 outer prediction IDs differ from the sealed test")
        seal_outer_predictions(
            stage,
            pd.DataFrame({"curated_id": ids, "prediction_normalized": values}),
            output,
        )
        trace_path = output / "training_trace.json"
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        if (
            trace.get("role") != "outer_refit"
            or trace.get("variant") != stage["key"]["variant"]
            or int(trace.get("outer_fold")) != int(stage["key"]["outer_fold"])
            or int(trace.get("seed")) != int(stage["key"]["effective_rng_seed"])
            or int(trace.get("fixed_epoch")) != fixed_epoch
        ):
            raise RuntimeError("D123 outer training trace identity drifted")
        evidence = resource_evidence_from_trace(trace, trace_path, gpu_stage=True)
        self.ledger.record_scientific_execution_finished(claim, evidence)
        complete_claimed_stage(stage, claim, self.ledger, output, evidence)


def validate_outer_resume_boundary(project_root: Path):
    root = project_root.resolve()
    plan, acceptance = base.validate_resume_boundary(root)
    amendment = json.loads((root / AMENDMENT_NAME).read_text(encoding="utf-8"))
    base_path = root / "src/d123_external_dependency_resume.py"
    wrapper_path = Path(__file__).resolve()
    if runner.stream_sha256(base_path) != EXPECTED_BASE_WRAPPER_SHA256:
        raise RuntimeError("D123 outer resume base wrapper bytes drifted")
    if (
        amendment.get("schema_version")
        != "scaffoldseal-d123-outer-issuance-resume-v1"
        or amendment.get("accepted") is not True
        or amendment.get("plan_sha256") != base.EXPECTED_PLAN_SHA256
        or amendment.get("base_wrapper_sha256") != EXPECTED_BASE_WRAPPER_SHA256
        or amendment.get("wrapper_file_sha256")
        != runner.stream_sha256(wrapper_path)
    ):
        raise RuntimeError("D123 outer issuance resume lacks exact acceptance")
    return plan, acceptance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    plan, acceptance = validate_outer_resume_boundary(root)
    base.governance.next_dependency_ready_stage = (
        base.next_stage_with_frozen_external_receipts
    )
    base.ResumeSafeD123LockedAdapter.runtime_root = root
    import d123_locked_adapter

    d123_locked_adapter.D123LockedAdapter = base.ResumeSafeD123LockedAdapter
    if args.validate_only:
        print(json.dumps({"status": "PASS", "plan_sha256": plan["plan_sha256"]}))
    if args.execute:
        OuterIssuanceResumeExecutor(
            plan, acceptance, root, root / runner.DEFAULT_PLAN
        ).run()


if __name__ == "__main__":
    main()
