"""Fail-closed D1/D2/D3 runner boundary; real execution remains source-locked."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Mapping


PREFIT_REVIEW_SOURCE_LOCK = True
REAL_EXECUTION_SOURCE_LOCK = True
DEFAULT_PLAN = "d123_plan_candidate.json"
DEFAULT_ACCEPTANCE = "d123_acceptance.json"


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_plan(plan: Mapping[str, object], project_root: Path) -> None:
    if plan.get("schema_version") != "scaffoldseal-d123-plan-v1":
        raise RuntimeError("D123 plan schema drifted")
    unhashed = dict(plan)
    observed_plan_sha = unhashed.pop("plan_sha256", None)
    if observed_plan_sha != canonical_sha256(unhashed):
        raise RuntimeError("D123 plan self-hash mismatch")
    authorization = plan.get("authorization")
    if not isinstance(authorization, Mapping) or any(
        authorization.get(key) is not False
        for key in ("accepted", "execution_authorized", "gpu_training_allowed")
    ):
        raise RuntimeError("Candidate plan authorization must remain false")
    protocol_lock = str(plan.get("protocol_lock_sha256"))
    expected_protocol = canonical_sha256(
        {
            "scientific_lock": plan["scientific_lock"],
            "locked_files": plan["locked_files"],
        }
    )
    if protocol_lock != expected_protocol:
        raise RuntimeError("D123 protocol lock mismatch")
    identities = set()
    namespaces = set()
    for stage in plan["stages"]:
        scientific = {
            "kind": stage["kind"],
            "key": stage["key"],
            "dependencies": stage["dependencies"],
            "protocol_lock_sha256": stage["protocol_lock_sha256"],
            "execution": stage["execution"],
            "expected_outputs": stage["expected_outputs"],
        }
        identity = canonical_sha256(scientific)
        if stage["scientific_identity_sha256"] != identity:
            raise RuntimeError("D123 stage scientific identity mismatch")
        if identity in identities:
            raise RuntimeError("D123 stage scientific identity duplicated")
        identities.add(identity)
        expected_stage_spec = canonical_sha256(
            {
                **scientific,
                "scientific_identity_sha256": identity,
            }
        )
        if stage.get("stage_spec_sha256") != expected_stage_spec:
            raise RuntimeError("D123 stage specification hash mismatch")
        execution = stage["execution"]
        if not isinstance(execution, Mapping) or type(execution.get("gpu")) is not bool:
            raise RuntimeError("D123 stage execution contract is malformed")
        expected_outputs = stage["expected_outputs"]
        if (
            not isinstance(expected_outputs, list)
            or not expected_outputs
            or len(set(map(str, expected_outputs))) != len(expected_outputs)
            or any(Path(str(item)).is_absolute() or ".." in Path(str(item)).parts for item in expected_outputs)
        ):
            raise RuntimeError("D123 expected-output contract is malformed")
        if stage["protocol_lock_sha256"] != protocol_lock:
            raise RuntimeError("D123 stage escaped the protocol lock")
        for key in ("work_attempt_template", "output_attempt_template"):
            namespace = str(stage["namespace"][key])
            if identity not in namespace or namespace in namespaces:
                raise RuntimeError("D123 namespace is not identity-unique")
            namespaces.add(namespace)
    root = project_root.resolve()
    for relative, record in plan["locked_files"].items():
        path = (root / str(relative)).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise RuntimeError("D123 locked file escapes project") from error
        if (
            not path.is_file()
            or path.stat().st_size != int(record["size_bytes"])
            or stream_sha256(path) != str(record["sha256"])
        ):
            raise RuntimeError(f"D123 locked file validation failed: {relative}")


def load_external_acceptance(project_root: Path) -> Mapping[str, object]:
    path = (project_root / DEFAULT_ACCEPTANCE).resolve()
    acceptance = json.loads(path.read_text(encoding="utf-8"))
    if acceptance.get("schema_version") != "scaffoldseal-d123-acceptance-v1":
        raise RuntimeError("D123 acceptance schema drifted")
    return acceptance


def assert_execution_authorized(
    plan: Mapping[str, object],
    acceptance: Mapping[str, object],
    plan_path: Path,
    runner_path: Path,
) -> None:
    if not PREFIT_REVIEW_SOURCE_LOCK or not REAL_EXECUTION_SOURCE_LOCK:
        raise RuntimeError("D123 runner source locks forbid real execution")
    if any(
        acceptance.get(key) is not True
        for key in ("accepted", "execution_authorized", "gpu_training_allowed")
    ):
        raise RuntimeError("External D123 acceptance does not authorize GPU execution")
    if acceptance.get("plan_sha256") != plan.get("plan_sha256"):
        raise RuntimeError("D123 acceptance identifies another plan")
    if acceptance.get("plan_file_sha256") != stream_sha256(plan_path):
        raise RuntimeError("D123 accepted plan bytes drifted")
    if acceptance.get("runner_file_sha256") != stream_sha256(runner_path):
        raise RuntimeError("D123 accepted runner bytes drifted")


def validate_d23_delaney_source(
    plan: Mapping[str, object], project_root: Path
) -> None:
    record = plan["scientific_lock"]["d23_delaney_source"]
    baseline_root = (
        project_root.parent / "baseline_candidates/BenchmarkCycPeptMP"
    ).resolve()
    path = (baseline_root / str(record["baseline_relative_path"])).resolve()
    try:
        path.relative_to(baseline_root)
    except ValueError as error:
        raise RuntimeError("D123 Delaney source escapes the baseline root") from error
    if (
        not path.is_file()
        or path.stat().st_size != int(record["size_bytes"])
        or stream_sha256(path) != str(record["sha256"])
    ):
        raise RuntimeError("D123 Delaney source bytes drifted")


class D123Executor:
    def __init__(
        self,
        plan: Mapping[str, object],
        acceptance: Mapping[str, object],
        project_root: Path,
        plan_path: Path,
    ) -> None:
        runner_path = Path(__file__).resolve()
        validate_plan(plan, project_root)
        assert_execution_authorized(plan, acceptance, plan_path, runner_path)
        from h1_random_cv_runner import ScientificStageLedger

        self.ledger = ScientificStageLedger(
            project_root / "artifacts/d123_ledger_v1",
            project_root=project_root,
        )
        self.plan = plan
        self.acceptance = acceptance
        self.project_root = project_root
        self.plan_path = plan_path.resolve()
        self.runner_path = runner_path
        self.by_identity = {
            str(stage["scientific_identity_sha256"]): stage
            for stage in plan["stages"]
        }

    def _revalidate_before_claim(self) -> None:
        fresh_plan = json.loads(self.plan_path.read_text(encoding="utf-8"))
        validate_plan(fresh_plan, self.project_root)
        if (
            fresh_plan.get("plan_sha256") != self.plan.get("plan_sha256")
            or canonical_sha256(fresh_plan) != canonical_sha256(self.plan)
        ):
            raise RuntimeError("D123 plan changed after executor initialization")
        fresh_acceptance = load_external_acceptance(self.project_root)
        if canonical_sha256(fresh_acceptance) != canonical_sha256(self.acceptance):
            raise RuntimeError("D123 acceptance changed after executor initialization")
        assert_execution_authorized(
            fresh_plan,
            fresh_acceptance,
            self.plan_path,
            self.runner_path,
        )
        validate_d23_delaney_source(fresh_plan, self.project_root)

    def _claim_paths(self, stage, claim) -> tuple[Path, Path]:
        latest = self.ledger.latest_attempt(stage)
        if (
            latest is None
            or latest.get("attempt_id") != claim.attempt_id
            or latest.get("status") != "CLAIMED_BEFORE_NAMESPACE_OR_MODEL"
        ):
            raise RuntimeError("D123 claimed attempt changed before namespace creation")
        work = (self.project_root / str(latest["work_namespace"])).resolve()
        output = (self.project_root / str(latest["output_namespace"])).resolve()
        allowed_work = (self.project_root / "runs/d123_v1").resolve()
        allowed_output = (self.project_root / "artifacts/d123_v1").resolve()
        try:
            work.relative_to(allowed_work)
            output.relative_to(allowed_output)
        except ValueError as error:
            raise RuntimeError("D123 attempt namespace escaped its locked root") from error
        if work.exists() or output.exists():
            raise RuntimeError("D123 fresh attempt namespace already exists")
        work.mkdir(parents=True)
        output.mkdir(parents=True)
        return work, output

    @staticmethod
    def _cpu_resource_evidence():
        from h1_random_cv_runner import make_resource_evidence

        return make_resource_evidence(
            runtime_seconds=0.0,
            max_memory_reserved_bytes=0,
            gpu_stage=False,
            source="d123_non_gpu_governance_stage",
        )

    def _run_label_gate_stage(self, stage, claim, output: Path) -> None:
        from d123_runner_governance import (
            complete_claimed_stage,
            write_label_release_receipt,
        )

        variant = str(stage["key"]["variant"])
        self.ledger.mark_scientific_execution_started(claim)
        write_label_release_receipt(
            self.plan,
            variant,
            self.ledger,
            self.project_root,
            output,
        )
        evidence = self._cpu_resource_evidence()
        self.ledger.record_scientific_execution_finished(claim, evidence)
        complete_claimed_stage(stage, claim, self.ledger, output, evidence)

    def _run_metric_stage(self, stage, claim, output: Path) -> None:
        from d123_metrics import write_variant_metrics
        from d123_runner_governance import complete_claimed_stage

        variant = str(stage["key"]["variant"])
        gate_identity = str(stage["key"]["label_gate_identity"])
        if tuple(map(str, stage["dependencies"])) != (gate_identity,):
            raise RuntimeError("D123 metric stage escaped its exact label gate")
        gate_stage = self.by_identity.get(gate_identity)
        if (
            gate_stage is None
            or gate_stage["kind"] != "d123_metric_label_release_gate"
            or gate_stage["key"].get("variant") != variant
        ):
            raise RuntimeError("D123 metric stage binds another gate or variant")
        self.ledger.mark_scientific_execution_started(claim)
        write_variant_metrics(
            self.plan,
            stage,
            variant,
            self.ledger,
            self.project_root,
            output,
        )
        evidence = self._cpu_resource_evidence()
        self.ledger.record_scientific_execution_finished(claim, evidence)
        complete_claimed_stage(stage, claim, self.ledger, output, evidence)

    @staticmethod
    def _ceil_median_four(values) -> int:
        epochs = sorted(int(value) for value in values)
        if len(epochs) != 4 or any(value < 1 or value > 2000 for value in epochs):
            raise RuntimeError("D123 selection requires four valid inner best epochs")
        return int(math.ceil((epochs[1] + epochs[2]) / 2.0))

    def _accepted_output(self, stage) -> Path:
        from d123_runner_governance import accepted_output_if_valid

        output = accepted_output_if_valid(
            stage, self.ledger, self.project_root
        )
        if output is None:
            raise RuntimeError("D123 stage dependency is not accepted")
        return output

    def _run_selection_stage(self, stage, claim, output: Path) -> None:
        from d123_runner_governance import complete_claimed_stage, canonical_sha256

        variant = str(stage["key"]["variant"])
        fold = int(stage["key"]["outer_fold"])
        dependencies = [self.by_identity[str(identity)] for identity in stage["dependencies"]]
        if (
            len(dependencies) != 4
            or {item["kind"] for item in dependencies} != {"d123_pampa_inner_fit"}
            or {str(item["key"]["variant"]) for item in dependencies} != {variant}
            or {int(item["key"]["outer_fold"]) for item in dependencies} != {fold}
            or {int(item["key"]["inner_basket"]) for item in dependencies}
            != {1, 2, 3, 4}
            or {int(item["key"]["seed_index"]) for item in dependencies} != {0}
        ):
            raise RuntimeError("D123 stopping selection escaped its exact four inner fits")
        traces = []
        for dependency in dependencies:
            trace_path = self._accepted_output(dependency) / "training_trace.json"
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            if (
                trace.get("role") != "inner"
                or trace.get("variant") != variant
                or int(trace.get("outer_fold")) != fold
                or int(trace.get("inner_basket"))
                != int(dependency["key"]["inner_basket"])
                or int(trace.get("seed")) != 0
            ):
                raise RuntimeError("D123 inner trace identity drifted before selection")
            traces.append(
                {
                    "inner_identity": str(
                        dependency["scientific_identity_sha256"]
                    ),
                    "inner_basket": int(dependency["key"]["inner_basket"]),
                    "best_epoch": int(trace["best_epoch"]),
                    "checkpoint_sha256": str(trace["checkpoint_sha256"]),
                }
            )
        selected = self._ceil_median_four(
            item["best_epoch"] for item in traces
        )
        receipt = {
            "schema_version": "scaffoldseal-d123-stopping-epoch-v1",
            "variant": variant,
            "outer_fold": fold,
            "rule": "ceil_median_4",
            "selected_epoch": selected,
            "inner_receipts": sorted(traces, key=lambda item: item["inner_basket"]),
            "scientific_identity_sha256": str(
                stage["scientific_identity_sha256"]
            ),
            "stage_spec_sha256": str(stage["stage_spec_sha256"]),
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        self.ledger.mark_scientific_execution_started(claim)
        target = output / "stopping_epoch.json"
        with target.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        evidence = self._cpu_resource_evidence()
        self.ledger.record_scientific_execution_finished(claim, evidence)
        complete_claimed_stage(stage, claim, self.ledger, output, evidence)

    def _run_descriptor_pretraining_stage(
        self, stage, claim, work: Path, output: Path
    ) -> None:
        from d123_pretraining import train_descriptor_delaney_once
        from d123_runner_governance import complete_claimed_stage
        from h1_random_cv_runner import resource_evidence_from_trace

        if (
            stage["kind"] != "d23_descriptor_delaney_pretraining"
            or int(stage["key"]["global_features_size"]) != 27
            or stage["key"]["global_features_policy"] != "all_zero"
            or list(stage["key"]["shared_by"]) != ["D2", "D3"]
        ):
            raise RuntimeError("D23 descriptor-pretraining contract drifted")
        seed_index = int(stage["key"]["seed_index"])
        effective_seed = int(stage["key"]["effective_rng_seed"])
        if (
            seed_index not in range(5)
            or effective_seed != (0, 123, 492, 1107, 1968)[seed_index]
        ):
            raise RuntimeError("D23 descriptor-pretraining seed drifted")
        delaney_record = self.plan["scientific_lock"]["d23_delaney_source"]
        baseline_root = (
            self.project_root.parent
            / "baseline_candidates/BenchmarkCycPeptMP"
        ).resolve()
        delaney_path = (
            baseline_root / str(delaney_record["baseline_relative_path"])
        ).resolve()
        try:
            delaney_path.relative_to(baseline_root)
        except ValueError as error:
            raise RuntimeError("D23 Delaney source escapes the baseline root") from error
        if (
            not delaney_path.is_file()
            or delaney_path.stat().st_size != int(delaney_record["size_bytes"])
            or stream_sha256(delaney_path) != str(delaney_record["sha256"])
        ):
            raise RuntimeError("D23 Delaney source failed pre-model byte validation")
        run_dir = work / "delaney_training"
        if run_dir.exists():
            raise RuntimeError("D23 pretraining helper namespace already exists")
        self.ledger.mark_scientific_execution_started(claim)
        with self.ledger.lease_heartbeat(claim):
            trace = train_descriptor_delaney_once(
                baseline_root,
                run_dir,
                f"d23-seed-{seed_index}",
                seed_index=seed_index,
                effective_rng_seed=effective_seed,
            )
        if (
            int(trace["seed_index"]) != seed_index
            or int(trace["effective_rng_seed"]) != effective_seed
            or int(trace["global_features_size"]) != 27
            or trace["global_features_policy"]
            != "all_zero_no_pampa_descriptor_access"
            or trace.get("raw_csv_sha256") != delaney_record["sha256"]
            or trace.get("target") != delaney_record["target_column"]
        ):
            raise RuntimeError("D23 pretraining trace escaped the zero-feature contract")
        checkpoint = run_dir / "checkpoint1.pt"
        if (
            not checkpoint.is_file()
            or str(trace["checkpoint_sha256"]) != stream_sha256(checkpoint)
        ):
            raise RuntimeError("D23 pretraining checkpoint trace mismatch")
        shutil.copy2(checkpoint, output / "checkpoint1.pt")
        trace_path = output / "training_trace.json"
        with trace_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(trace, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        evidence = resource_evidence_from_trace(
            trace, trace_path, gpu_stage=True
        )
        self.ledger.record_scientific_execution_finished(claim, evidence)
        complete_claimed_stage(stage, claim, self.ledger, output, evidence)

    def _pretrained_checkpoint(self, stage) -> tuple[Path, str]:
        variant = str(stage["key"]["variant"])
        seed_index = int(stage["key"]["seed_index"])
        if variant == "D1":
            receipt = self.plan["scientific_lock"][
                "d1_accepted_checkpoint_receipts"
            ][str(seed_index)]
            if (
                stage["key"].get("d1_checkpoint_receipt_sha256")
                != receipt["receipt_sha256"]
            ):
                raise RuntimeError("D1 stage binds another accepted checkpoint receipt")
            checkpoint = (
                self.project_root / str(receipt["checkpoint"]["relative_path"])
            ).resolve()
            expected_sha = str(receipt["checkpoint"]["sha256"])
        else:
            dependencies = [
                self.by_identity.get(str(identity))
                for identity in stage["dependencies"]
                if str(identity) in self.by_identity
            ]
            pretraining = [
                item
                for item in dependencies
                if item["kind"] == "d23_descriptor_delaney_pretraining"
            ]
            if len(pretraining) != 1:
                raise RuntimeError("D2/D3 stage lacks one descriptor pretraining dependency")
            if int(pretraining[0]["key"]["seed_index"]) != seed_index:
                raise RuntimeError("D2/D3 stage binds another pretraining seed")
            checkpoint = self._accepted_output(pretraining[0]) / "checkpoint1.pt"
            expected_sha = stream_sha256(checkpoint)
        if not checkpoint.is_file() or stream_sha256(checkpoint) != expected_sha:
            raise RuntimeError("D123 pretrained checkpoint failed exact validation")
        return checkpoint, expected_sha

    def _load_adapter_and_contract(self, stage, frames):
        import pandas as pd
        from d123_locked_adapter import D123LockedAdapter
        from split_safe import OuterFoldContract

        fold = int(stage["key"]["outer_fold"])
        baskets = pd.read_csv(
            self.project_root / "artifacts/v2_r0/inner_basket_manifest.csv"
        )
        block_to_basket = {
            str(row.sealed_block_id): int(row.inner_basket)
            for row in baskets.loc[
                baskets["outer_fold"].astype(int).eq(fold)
            ].itertuples(index=False)
        }
        outer_train = frames["outer_train"]
        basket_by_id = {
            str(row.curated_id): block_to_basket[str(row.sealed_block_id)]
            for row in outer_train.itertuples(index=False)
        }
        contract = OuterFoldContract(
            fold,
            outer_train["curated_id"],
            frames["outer_test_label_free"]["curated_id"],
            basket_by_id,
        )
        group_metadata = pd.read_csv(
            self.project_root / "artifacts/d123_inputs_v1/group_metadata.csv"
        )
        raw_descriptors = pd.read_csv(
            self.project_root / "artifacts/d123_inputs_v1/raw_descriptors.csv"
        )
        source_hashes = {
            relative: str(record["sha256"])
            for relative, record in self.plan["locked_files"].items()
            if relative.startswith("src/")
        }
        adapter = D123LockedAdapter(
            baseline_root=self.project_root.parent
            / "baseline_candidates/BenchmarkCycPeptMP",
            source_hashes=source_hashes,
            variant=str(stage["key"]["variant"]),
            group_metadata=group_metadata,
            raw_descriptors=raw_descriptors,
        )
        return adapter, contract

    def _run_inner_stage(self, stage, claim, work: Path, output: Path) -> None:
        from d123_fold_data import load_fold_frames
        from d123_runner_governance import complete_claimed_stage
        from h1_random_cv_runner import resource_evidence_from_trace
        from split_safe import FitAuditTrail, SplitSafeFitExecutor

        frames = load_fold_frames(self.plan, stage, self.project_root)
        adapter, contract = self._load_adapter_and_contract(stage, frames)
        checkpoint, checkpoint_sha = self._pretrained_checkpoint(stage)
        basket = int(stage["key"]["inner_basket"])
        audit = FitAuditTrail()
        executor = SplitSafeFitExecutor(contract, audit)
        run = contract.mint_run_context(
            work / "checkpoint_root",
            config_id=f"d123_{stage['key']['variant']}",
            seed=int(stage["key"]["effective_rng_seed"]),
            inner_basket=basket,
            pretrained_checkpoint=checkpoint,
            pretrained_checkpoint_sha256=checkpoint_sha,
        )
        train = contract.inner_training_batch(frames["outer_train"], basket)
        validation = contract.inner_validation_batch(frames["outer_train"], basket)
        recorder = contract.create_inner_evaluation_recorder(
            train,
            validation,
            basket=basket,
            feature_columns=["SMILES"],
            target_column="normalized_pampa",
            metric_identity="mean_squared_error",
            run_context=run,
            transform_sha256=adapter.transform_sha256,
            model_config_sha256=adapter.model_config_sha256,
            checkpoint_sha256=checkpoint_sha,
            audit=audit,
        )
        self.ledger.mark_scientific_execution_started(claim)
        with self.ledger.lease_heartbeat(claim):
            executor.fit_inner_frame(
                adapter,
                train,
                validation,
                basket=basket,
                feature_columns=["SMILES"],
                target_column="normalized_pampa",
                run_context=run,
                recorder=recorder,
                maximum_epochs=2000,
            )
        recorder.finalize()
        checkpoint_dir = Path(run.checkpoint_dir)
        for name in ("checkpoint1.pt", "training_trace.json"):
            shutil.copy2(checkpoint_dir / name, output / name)
        if str(stage["key"]["variant"]) in ("D2", "D3"):
            shutil.copy2(
                checkpoint_dir / "descriptor_transform.json",
                output / "descriptor_transform.json",
            )
        trace_path = output / "training_trace.json"
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        if (
            trace.get("role") != "inner"
            or trace.get("variant") != stage["key"]["variant"]
            or int(trace.get("outer_fold")) != int(stage["key"]["outer_fold"])
            or int(trace.get("inner_basket")) != basket
            or int(trace.get("seed")) != int(stage["key"]["effective_rng_seed"])
        ):
            raise RuntimeError("D123 inner training trace identity drifted")
        evidence = resource_evidence_from_trace(
            trace, trace_path, gpu_stage=True
        )
        self.ledger.record_scientific_execution_finished(claim, evidence)
        complete_claimed_stage(stage, claim, self.ledger, output, evidence)

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
            or int(selection["key"]["outer_fold"]) != int(stage["key"]["outer_fold"])
        ):
            raise RuntimeError("D123 outer fit binds another stopping selection")
        selection_receipt = json.loads(
            (
                self._accepted_output(selection) / "stopping_epoch.json"
            ).read_text(encoding="utf-8")
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
                contract.outer_frame_training_batch(frames["outer_train"]),
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
            sorted(
                frames["outer_test_label_free"]["curated_id"].astype(str)
            )
        ):
            raise RuntimeError("D123 outer prediction IDs differ from the sealed test")
        seal_outer_predictions(
            stage,
            pd.DataFrame(
                {
                    "curated_id": ids,
                    "prediction_normalized": values,
                }
            ),
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
        evidence = resource_evidence_from_trace(
            trace, trace_path, gpu_stage=True
        )
        self.ledger.record_scientific_execution_finished(claim, evidence)
        complete_claimed_stage(stage, claim, self.ledger, output, evidence)

    def _failure_evidence(self, stage, claim):
        from h1_random_cv_runner import (
            MAX_GPU_RESERVED_BYTES,
            make_resource_evidence,
            validate_resource_evidence,
        )

        latest = self.ledger.latest_attempt(stage) or {}
        provisional = latest.get("provisional_resource_evidence")
        gpu_stage = bool(stage["execution"]["gpu"])
        if (
            latest.get("scientific_execution_finished") is True
            and isinstance(provisional, Mapping)
        ):
            return validate_resource_evidence(provisional, gpu_stage=gpu_stage)
        if latest.get("scientific_execution_started") is not True:
            return make_resource_evidence(
                runtime_seconds=0.0,
                max_memory_reserved_bytes=0,
                gpu_stage=gpu_stage,
                source="d123_claim_proved_pre_model",
                conservative=True,
            )
        started = latest.get("scientific_start_unix_seconds")
        if type(started) not in (int, float):
            raise RuntimeError("D123 failed attempt start time is malformed")
        return make_resource_evidence(
            runtime_seconds=max(0.0, time.time() - float(started)),
            max_memory_reserved_bytes=(
                MAX_GPU_RESERVED_BYTES if gpu_stage else 0
            ),
            gpu_stage=gpu_stage,
            source="d123_failed_wall_time_and_gpu_guard_ceiling",
            conservative=True,
        )

    def _dispatch_stage(self, stage, claim, work: Path, output: Path) -> None:
        kind = str(stage["kind"])
        if kind == "d23_descriptor_delaney_pretraining":
            self._run_descriptor_pretraining_stage(stage, claim, work, output)
        elif kind == "d123_pampa_inner_fit":
            self._run_inner_stage(stage, claim, work, output)
        elif kind == "d123_stopping_epoch_selection":
            self._run_selection_stage(stage, claim, output)
        elif kind == "d123_pampa_outer_fit_prediction":
            self._run_outer_stage(stage, claim, work, output)
        elif kind == "d123_metric_label_release_gate":
            self._run_label_gate_stage(stage, claim, output)
        elif kind == "d123_sealed_oof_metrics":
            self._run_metric_stage(stage, claim, output)
        else:
            raise RuntimeError(f"Unsupported D123 stage kind: {kind}")

    def run(self) -> None:
        from d123_runner_governance import (
            claim_next_dependency_ready_stage,
            fail_claimed_stage,
        )

        while True:
            self._revalidate_before_claim()
            selected = claim_next_dependency_ready_stage(
                self.plan,
                self.ledger,
                self.project_root,
                recovery_reason=(
                    "D123 executor restart: previous owner proved dead; "
                    "namespace and append-only evidence preserved"
                ),
            )
            if selected is None:
                return
            stage, claim = selected
            work, output = self._claim_paths(stage, claim)
            try:
                self._dispatch_stage(stage, claim, work, output)
            except BaseException as error:
                latest = self.ledger.latest_attempt(stage)
                if (
                    latest is not None
                    and latest.get("status")
                    == "CLAIMED_BEFORE_NAMESPACE_OR_MODEL"
                ):
                    fail_claimed_stage(
                        claim,
                        self.ledger,
                        reason=f"{type(error).__name__}: {error}",
                        resource_evidence=self._failure_evidence(stage, claim),
                    )
                raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--validate-plan", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    plan_path = (root / DEFAULT_PLAN).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_plan(plan, root)
    if args.validate_plan:
        print(json.dumps({"status": "PASS", "plan_sha256": plan["plan_sha256"]}))
    if args.execute:
        acceptance = load_external_acceptance(root)
        D123Executor(plan, acceptance, root, plan_path).run()


if __name__ == "__main__":
    main()
