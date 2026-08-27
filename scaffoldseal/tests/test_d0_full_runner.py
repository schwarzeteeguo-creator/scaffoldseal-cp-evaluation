import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT.parent / "baseline_candidates" / "BenchmarkCycPeptMP"
sys.path.insert(0, str(ROOT / "src"))

from d0_full_runner import (
    CORRECTED_PROJECTED_GPU_HOURS,
    D0FullExecutor,
    FROZEN_PROJECT_INPUTS,
    FULL_LABEL_RELATIVE_PATH,
    MAX_GPU_RESERVED_BYTES,
    MAX_PROJECTED_GPU_HOURS,
    PREFIT_REVIEW_ACCEPTED,
    REAL_EXECUTION_AUTHORIZED,
    ScientificStageLedger,
    StageClaimUnavailable,
    _write_stage_artifact_manifest,
    assert_real_execution_authorized,
    build_full_plan,
    build_live_status,
    canonical_json_sha256,
    effective_rng_seed,
    invoke_pretraining_helper_no_learning,
    make_resource_evidence,
    process_identity_state,
    process_start_token,
    publish_live_status,
    read_live_status_bundle,
    read_sealed_prediction_artifact,
    record_attempt_exception,
    resource_evidence_from_trace,
    stream_sha256,
    validate_plan,
    validate_resource_evidence,
    validate_stage_artifacts,
    verify_candidate_anchor,
    write_json,
    write_sealed_prediction_artifacts,
)
from d0_fold_scoped_governance import build_fold_scoped_governance


def resource(runtime=0.0, *, gpu=True, reserved=0):
    return make_resource_evidence(
        runtime_seconds=runtime,
        max_memory_reserved_bytes=reserved,
        gpu_stage=gpu,
        source="no-learning-test-evidence",
    )


def stub_stage(identity_char="a", *, gpu=True):
    return {
        "scientific_identity_sha256": identity_char * 64,
        "stage_spec_sha256": hashlib.sha256((identity_char + "-spec").encode()).hexdigest(),
        "kind": "no_learning_stub",
        "key": {},
        "dependencies": [],
        "execution": {"gpu": gpu},
        "namespace": {
            "work_attempt_template": "runs/stub/attempt_{attempt_number:04d}",
            "output_attempt_template": "artifacts/stub/attempt_{attempt_number:04d}",
        },
        "expected_outputs": ["result.json", "artifact_manifest.json"],
    }


class D0FullRunnerRepair1NoTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = build_full_plan(ROOT, BASELINE)
        cls.by_kind = {}
        for stage in cls.plan["stages"]:
            cls.by_kind.setdefault(stage["kind"], []).append(stage)

    def test_geometry_and_seed_schedule_are_unchanged(self):
        validate_plan(self.plan)
        self.assertEqual(len(self.plan["stages"]), 186)
        self.assertEqual(self.plan["counts"]["scientific_fits_total"], 167)
        self.assertEqual(self.plan["counts"]["expected_oof_prediction_rows"], 34475)
        self.assertEqual(
            {kind: len(stages) for kind, stages in self.by_kind.items()},
            {
                "delaney_pretraining": 5,
                "pampa_inner_fit": 72,
                "stopping_epoch_selection": 18,
                "pampa_outer_fit_predict": 90,
                "sealed_oof_metrics": 1,
            },
        )
        self.assertEqual([effective_rng_seed(i) for i in range(5)], [0, 123, 492, 1107, 1968])

    def test_stage_namespace_is_independently_recomputed(self):
        tampered = copy.deepcopy(self.plan)
        tampered["stages"][0]["namespace"]["work_attempt_template"] = (
            "runs/alternate/attempt_{attempt_number:04d}"
        )
        tampered.pop("plan_sha256")
        tampered["plan_sha256"] = canonical_json_sha256(tampered)
        with self.assertRaisesRegex(ValueError, "namespace"):
            validate_plan(tampered)

    def test_candidate_manifest_is_pending_and_byte_deterministic(self):
        checked = json.loads((ROOT / "d0_full_run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(checked, self.plan)
        self.assertEqual(build_full_plan(ROOT, BASELINE), self.plan)
        self.assertEqual(self.plan["status"], "CANDIDATE_PENDING_INDEPENDENT_ACCEPTANCE")
        self.assertFalse(self.plan["authorization"]["real_execution_authorized"])

    def test_hard_source_lock_and_external_acceptance_are_both_required(self):
        self.assertFalse(PREFIT_REVIEW_ACCEPTED)
        self.assertFalse(REAL_EXECUTION_AUTHORIZED)
        with self.assertRaisesRegex(RuntimeError, "pre-fit locked"):
            assert_real_execution_authorized(self.plan, {"accepted": True})
        with self.assertRaisesRegex(RuntimeError, "pre-fit locked"):
            D0FullExecutor(self.plan, ROOT, BASELINE)

    def test_pretraining_helper_owns_exact_run_directory(self):
        observed = []

        def inert_helper(baseline_root, run_dir, tag, *, seed_index):
            observed.append(run_dir.exists())
            run_dir.mkdir(parents=True)
            return {"tag": tag, "seed_index": seed_index}

        inert_helper._scaffoldseal_no_learning_stub = True
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "parent" / "attempt_0001"
            result = invoke_pretraining_helper_no_learning(
                inert_helper, BASELINE, run_dir, "stub", seed_index=3
            )
            self.assertEqual(observed, [False])
            self.assertTrue(run_dir.is_dir())
            self.assertEqual(result["seed_index"], 3)

    def test_fold_scoped_builder_never_opens_metric_only_labels(self):
        original = pd.read_csv
        opened = []

        def guarded(path, *args, **kwargs):
            normalized = str(path).replace("\\", "/")
            if normalized.endswith(FULL_LABEL_RELATIVE_PATH):
                raise AssertionError("metric-only label table was opened")
            opened.append(normalized)
            return original(path, *args, **kwargs)

        with mock.patch("d0_full_runner.pd.read_csv", side_effect=guarded):
            rebuilt = build_full_plan(ROOT, BASELINE)
        self.assertEqual(rebuilt["plan_sha256"], self.plan["plan_sha256"])
        self.assertTrue(opened)

    def test_fold_scoped_derivatives_rebuild_byte_identically(self):
        committed_root = ROOT / "artifacts/v2_r0/d0_full_runner_repair1"
        with tempfile.TemporaryDirectory() as temporary:
            rebuilt_root = Path(temporary) / "fold-views"
            manifest = build_fold_scoped_governance(
                labels_path=ROOT / FULL_LABEL_RELATIVE_PATH,
                outer_assignments_path=ROOT / "artifacts/v2_r0/outer_record_assignments.csv",
                output_root=rebuilt_root,
            )
            committed = json.loads(
                (committed_root / "fold_scoped_target_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest, committed)
            for relative in ["label_free_features.csv", *[f"outer_{fold:02d}_training_targets.csv" for fold in range(1, 19)]]:
                self.assertEqual(
                    (rebuilt_root / relative).read_bytes(),
                    (committed_root / relative).read_bytes(),
                )

    def test_runtime_fold_view_contains_no_current_heldout_targets(self):
        executor = object.__new__(D0FullExecutor)
        executor.project_root = ROOT
        original = pd.read_csv

        def guarded(path, *args, **kwargs):
            if str(path).replace("\\", "/").endswith(FULL_LABEL_RELATIVE_PATH):
                raise AssertionError("full labels opened before predictions")
            return original(path, *args, **kwargs)

        with mock.patch("d0_full_runner.pd.read_csv", side_effect=guarded):
            _, frame, contracts = executor._load_fold_scoped_frame_and_contracts(1)
        heldout = frame["outer_fold"].astype(int).eq(1)
        self.assertEqual(int(heldout.sum()), len(contracts[1].outer_test_ids))
        self.assertTrue(frame.loc[heldout, "normalized_pampa"].isna().all())
        self.assertTrue(frame.loc[~heldout, "normalized_pampa"].notna().all())

    def _copy_anchor_inputs(self, project: Path):
        for relative in FROZEN_PROJECT_INPUTS:
            source = (ROOT / relative).resolve()
            target = (project / relative).resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        runner = project / "src/d0_full_runner.py"
        runner.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "src/d0_full_runner.py", runner)
        manifest = project / "d0_full_run_manifest.json"
        write_json(manifest, self.plan)
        return manifest

    def test_status_only_rejects_alternate_manifest_before_any_status_or_ledger_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "scaffoldseal"
            self._copy_anchor_inputs(project)
            shutil.copy2(ROOT / "d0_full_run_acceptance.json", project / "d0_full_run_acceptance.json")
            alternate = copy.deepcopy(self.plan)
            alternate["stages"][0]["namespace"]["work_attempt_template"] = (
                "runs/alternate/attempt_{attempt_number:04d}"
            )
            alternate.pop("plan_sha256")
            alternate["plan_sha256"] = canonical_json_sha256(alternate)
            alternate_path = project / "alternate_manifest.json"
            write_json(alternate_path, alternate)
            result = subprocess.run(
                [
                    sys.executable,
                    str(project / "src/d0_full_runner.py"),
                    "--status-only",
                    "--project-root",
                    str(project),
                    "--baseline-root",
                    str(BASELINE),
                    "--manifest",
                    str(alternate_path),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Alternate candidate manifests are forbidden", result.stderr)
            self.assertFalse((project / "artifacts/d0_full_run/CURRENT.json").exists())
            self.assertFalse(
                (project / str(self.plan["scheduler"]["ledger_relative_path"])).exists()
            )

    def test_external_candidate_anchor_rejects_config_code_and_target_drift(self):
        for relative in (
            "config_v2.yaml",
            "src/split_safe.py",
            "artifacts/v2_r0/d0_full_runner_repair1/outer_01_training_targets.csv",
        ):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as temporary:
                project = Path(temporary) / "scaffoldseal"
                manifest = self._copy_anchor_inputs(project)
                target = project / relative
                target.write_bytes(target.read_bytes() + b"\n# drift\n")
                with self.assertRaisesRegex(RuntimeError, "drift"):
                    verify_candidate_anchor(
                        self.plan,
                        project,
                        BASELINE,
                        manifest_path=manifest,
                    )

    def test_anchor_does_not_read_or_require_metric_only_full_label_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "scaffoldseal"
            manifest = self._copy_anchor_inputs(project)
            self.assertFalse((project / FULL_LABEL_RELATIVE_PATH).exists())
            verify_candidate_anchor(self.plan, project, BASELINE, manifest_path=manifest)

    def test_false_external_acceptance_fails_before_execution(self):
        acceptance = {"accepted": False}
        with self.assertRaisesRegex(RuntimeError, "absent or false"):
            verify_candidate_anchor(
                self.plan,
                ROOT,
                BASELINE,
                manifest_path=ROOT / "d0_full_run_manifest.json",
                acceptance=acceptance,
                require_accepted=True,
            )

    def test_resource_trace_requires_finite_nonnegative_hash_bound_gpu_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            trace_path = Path(temporary) / "training_trace.json"
            cases = [
                {"runtime_seconds": 1.0},
                {"runtime_seconds": 1.0, "gpu": {}},
                {"runtime_seconds": float("nan"), "gpu": {"max_memory_reserved_bytes": 1}},
                {"runtime_seconds": -1.0, "gpu": {"max_memory_reserved_bytes": 1}},
                {
                    "runtime_seconds": 1.0,
                    "gpu": {"max_memory_reserved_bytes": MAX_GPU_RESERVED_BYTES + 1},
                },
            ]
            for trace in cases:
                with self.subTest(trace=trace):
                    trace_path.write_text(json.dumps(trace) + "\n", encoding="utf-8")
                    with self.assertRaises(RuntimeError):
                        resource_evidence_from_trace(trace, trace_path, gpu_stage=True)
            good = {"runtime_seconds": 2.5, "gpu": {"max_memory_reserved_bytes": 1024}}
            trace_path.write_text(json.dumps(good) + "\n", encoding="utf-8")
            evidence = resource_evidence_from_trace(good, trace_path, gpu_stage=True)
            self.assertEqual(validate_resource_evidence(evidence, gpu_stage=True), evidence)

    def test_raw_resource_types_reject_bool_and_fractional_gpu_reservation(self):
        with self.assertRaisesRegex(RuntimeError, "built-in int or float"):
            make_resource_evidence(
                runtime_seconds=True,
                max_memory_reserved_bytes=0,
                gpu_stage=True,
                source="invalid-bool-runtime",
            )
        for invalid in (True, 1.5):
            with self.subTest(reservation=invalid), self.assertRaisesRegex(
                RuntimeError, "built-in int"
            ):
                make_resource_evidence(
                    runtime_seconds=1,
                    max_memory_reserved_bytes=invalid,
                    gpu_stage=True,
                    source="invalid-reservation",
                )

    def test_failed_trace_rewrite_and_summary_rewrite_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            trace_path = project / "failed_trace.json"
            trace = {
                "runtime_seconds": 300000,
                "gpu": {"max_memory_reserved_bytes": 1024},
            }
            trace_path.write_text(json.dumps(trace) + "\n", encoding="utf-8")
            ledger = ScientificStageLedger(project / "ledger", project_root=project)
            stage = stub_stage("6", gpu=True)
            claim = ledger.claim(stage)
            ledger.record_failed(
                claim,
                "failed after long run",
                resource_evidence_from_trace(trace, trace_path, gpu_stage=True),
            )
            self.assertEqual(ledger.cumulative_gpu_seconds(), 300000)

            original = trace_path.read_bytes()
            trace_path.write_bytes(original.replace(b"300000", b"000000"))
            with self.assertRaisesRegex(RuntimeError, "source file hash drift"):
                ledger.cumulative_gpu_seconds()
            with self.assertRaisesRegex(RuntimeError, "source file hash drift"):
                ledger.latest_attempt(stage)

        with tempfile.TemporaryDirectory() as temporary:
            ledger = ScientificStageLedger(Path(temporary) / "ledger")
            stage = stub_stage("7", gpu=True)
            claim = ledger.claim(stage)
            ledger.record_failed(claim, "recorded", resource(300000, gpu=True))
            summary = json.loads(claim.ledger_path.read_text(encoding="utf-8"))
            evidence = summary["attempts"][-1]["details"]["resource_evidence"]
            evidence["runtime_seconds"] = 0
            evidence["evidence_sha256"] = canonical_json_sha256(
                {key: value for key, value in evidence.items() if key != "evidence_sha256"}
            )
            summary["summary_sha256"] = canonical_json_sha256(
                {key: value for key, value in summary.items() if key != "summary_sha256"}
            )
            claim.ledger_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "differs from immutable event chain"):
                ledger.cumulative_gpu_seconds()

    def test_event_files_are_exclusive_continuous_and_hash_chained(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = ScientificStageLedger(Path(temporary) / "ledger")
            stage = stub_stage("8", gpu=False)
            first = ledger.claim(stage)
            ledger.record_failed(first, "first", resource(gpu=False))
            second = ledger.claim(stage)
            ledger.record_interrupted(second, "second", resource(gpu=False))
            files = ledger._event_files(stage["scientific_identity_sha256"])
            self.assertEqual(len(files), 4)
            previous = None
            for index, path in enumerate(files, start=1):
                event = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(event["event_index"], index)
                self.assertEqual(event["prev_event_sha256"], previous)
                previous = event["event_sha256"]
            with self.assertRaises(FileExistsError):
                descriptor = os.open(
                    str(files[0]), os.O_WRONLY | os.O_CREAT | os.O_EXCL
                )
                os.close(descriptor)

    def test_failed_attempt_runtime_counts_and_exact_72_hours_blocks(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = ScientificStageLedger(Path(temporary) / "ledger")
            stage = stub_stage("b", gpu=True)
            claim = ledger.claim(stage)
            ledger.record_failed(claim, "long failed attempt", resource(72 * 3600, gpu=True))
            self.assertEqual(ledger.cumulative_gpu_seconds(), 72 * 3600)
            executor = object.__new__(D0FullExecutor)
            executor.ledger = ledger
            with self.assertRaisesRegex(RuntimeError, "72-hour"):
                executor._check_gpu_budget()

    def test_finished_live_gpu_attempt_uses_fixed_hash_bound_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = ScientificStageLedger(Path(temporary) / "ledger")
            stage = stub_stage("5", gpu=True)
            claim = ledger.claim(stage)
            ledger.mark_scientific_execution_started(claim)
            evidence = resource(17.0, gpu=True, reserved=1024)
            ledger.record_scientific_execution_finished(claim, evidence)
            self.assertEqual(ledger.cumulative_gpu_seconds(now_unix_seconds=time.time() + 1000), 17.0)
            ledger.record_interrupted(claim, "bundle did not seal", evidence)
            self.assertEqual(ledger.cumulative_gpu_seconds(), 17.0)

    def test_all_failure_and_interruption_attempts_are_preserved_and_counted(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = ScientificStageLedger(Path(temporary) / "ledger")
            stage = stub_stage("c", gpu=True)
            first = ledger.claim(stage)
            ledger.record_failed(first, "failure", resource(11.0, gpu=True))
            second = ledger.claim(stage)
            ledger.record_interrupted(second, "interrupt", resource(13.0, gpu=True))
            self.assertEqual(ledger.cumulative_gpu_seconds(), 24.0)
            third = ledger.claim(stage)
            self.assertEqual(third.attempt_number, 3)
            self.assertNotEqual(first.attempt_id, third.attempt_id)

    def test_keyboard_interrupt_is_not_mislabeled_as_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = ScientificStageLedger(Path(temporary) / "ledger")
            stage = stub_stage("d", gpu=False)
            claim = ledger.claim(stage)
            record_attempt_exception(ledger, claim, KeyboardInterrupt(), resource(gpu=False))
            self.assertEqual(ledger.latest_attempt(stage)["status"], "INTERRUPTED_RECORDED")

    def test_live_owner_cannot_be_stolen(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = ScientificStageLedger(Path(temporary) / "ledger")
            stage = stub_stage("e", gpu=False)
            ledger.claim(stage)
            token = ledger.latest_attempt(stage)["owner"]["process_start_token"]
            with self.assertRaisesRegex(StageClaimUnavailable, "Live"):
                ledger.recover_interrupted(
                    stage,
                    reason="must not steal",
                    inspector=lambda pid: token,
                )

    def test_dead_owner_recovery_preserves_namespace_then_mints_new_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = ScientificStageLedger(root / "ledger")
            stage = stub_stage("f", gpu=True)
            first = ledger.claim(stage)
            namespace = root / "preserved-attempt"
            namespace.mkdir()
            (namespace / "partial.bin").write_bytes(b"partial")
            recovered = ledger.recover_interrupted(
                stage,
                reason="owner killed",
                inspector=lambda pid: None,
            )
            self.assertEqual(recovered["status"], "INTERRUPTED_RECORDED")
            self.assertTrue((namespace / "partial.bin").is_file())
            second = ledger.claim(stage)
            self.assertEqual(second.attempt_number, 2)

    def test_pid_reuse_and_stale_lock_are_auditable(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = ScientificStageLedger(Path(temporary) / "ledger")
            stage = stub_stage("1", gpu=False)
            claim = ledger.claim(stage)
            latest = ledger.latest_attempt(stage)
            _, lock_path = ledger._paths(stage["scientific_identity_sha256"])
            lock_path.write_text(json.dumps(latest["owner"]), encoding="utf-8")
            ledger.recover_interrupted(
                stage,
                reason="PID reused",
                inspector=lambda pid: "different-process-start-token",
            )
            stale = list(ledger.root.glob("*.stale_lock.*.json"))
            self.assertEqual(len(stale), 1)
            details = ledger.latest_attempt(stage)["details"]["dead_owner_proof"]
            self.assertEqual(details["observed_process_start_token"], "different-process-start-token")

    def test_real_killed_process_claim_can_be_recovered(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger_root = Path(temporary) / "ledger"
            stage = stub_stage("2", gpu=False)
            code = (
                "import json,sys; from pathlib import Path; "
                "from d0_full_runner import ScientificStageLedger; "
                "ScientificStageLedger(Path(sys.argv[1])).claim(json.loads(sys.argv[2]))"
            )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(ROOT / "src")
            subprocess.run(
                [sys.executable, "-c", code, str(ledger_root), json.dumps(stage)],
                env=environment,
                check=True,
                timeout=30,
            )
            ledger = ScientificStageLedger(ledger_root)
            latest = ledger.latest_attempt(stage)
            self.assertIsNone(process_start_token(int(latest["owner"]["pid"])))
            ledger.recover_interrupted(stage, reason="subprocess exited")
            self.assertEqual(ledger.latest_attempt(stage)["status"], "INTERRUPTED_RECORDED")

    def test_terminated_sleeping_claimant_is_not_alive_and_attempt_two_is_minted(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            ledger_root = project / "ledger"
            ready = project / "ready"
            stage = stub_stage("9", gpu=False)
            code = (
                "import json,sys,time; from pathlib import Path; "
                "from d0_full_runner import ScientificStageLedger; "
                "ScientificStageLedger(Path(sys.argv[1])).claim(json.loads(sys.argv[2])); "
                "Path(sys.argv[3]).write_text('ready'); time.sleep(120)"
            )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(ROOT / "src")
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    code,
                    str(ledger_root),
                    json.dumps(stage),
                    str(ready),
                ],
                env=environment,
            )
            deadline = time.time() + 30
            while not ready.is_file() and time.time() < deadline:
                time.sleep(0.05)
            self.assertTrue(ready.is_file())
            process.terminate()
            process.wait(timeout=30)
            # Keep the Popen object and its process handle alive during recovery.
            state = process_identity_state(process.pid)
            self.assertTrue(state is None or state["is_running"] is False)
            ledger = ScientificStageLedger(ledger_root)
            first = ledger.latest_attempt(stage)
            namespace = project / str(first["work_namespace"])
            namespace.mkdir(parents=True)
            (namespace / "partial.bin").write_bytes(b"partial")
            ledger.recover_interrupted(stage, reason="force-terminated sleeping claimant")
            self.assertTrue((namespace / "partial.bin").is_file())
            second = ledger.claim(stage)
            self.assertEqual(second.attempt_number, 2)

    def test_process_safe_claim_has_exactly_one_winner(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger_root = Path(temporary) / "ledger"
            stage = stub_stage("3", gpu=False)
            code = (
                "import json,sys; from pathlib import Path; "
                "from d0_full_runner import ScientificStageLedger; "
                "ScientificStageLedger(Path(sys.argv[1])).claim(json.loads(sys.argv[2]))"
            )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(ROOT / "src")
            processes = [
                subprocess.Popen(
                    [sys.executable, "-c", code, str(ledger_root), json.dumps(stage)],
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                for _ in range(6)
            ]
            returns = [process.communicate(timeout=30) + (process.returncode,) for process in processes]
            self.assertEqual(sum(item[2] == 0 for item in returns), 1)

    def test_artifacts_and_exact_bit_predictions_remain_hashable_and_label_free(self):
        stage = stub_stage("4", gpu=False)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "result.json").write_text("{}\n", encoding="utf-8")
            (root / "artifact_manifest.json").write_text("{}\n", encoding="utf-8")
            self.assertEqual(len(validate_stage_artifacts(stage, root)), 2)
            values = np.asarray([0.0, -0.0, np.nextafter(1.0, 2.0)], dtype=np.float64)
            table = pd.DataFrame(
                {
                    "curated_id": ["a", "b", "c"],
                    "outer_fold": [1, 1, 1],
                    "seed": [0, 0, 0],
                    "prediction_normalized": values,
                    "prediction_log10_papp": values * 2.0 - 6.0,
                }
            )
            lossless = root / "predictions.lossless.json"
            readable = root / "predictions.csv"
            write_sealed_prediction_artifacts(table, lossless, readable)
            recovered = read_sealed_prediction_artifact(lossless)
            np.testing.assert_array_equal(
                recovered["prediction_normalized"].to_numpy().view(np.uint64),
                values.view(np.uint64),
            )
            payload = json.loads(lossless.read_text(encoding="utf-8"))
            self.assertFalse(payload["contains_observed_labels"])
            self.assertEqual(
                payload["records"][1]["prediction_normalized_ieee754_be"],
                struct.pack(">d", -0.0).hex(),
            )

    def test_complete_inner_certificate_recovers_but_partial_does_not(self):
        inner = sorted(
            [stage for stage in self.by_kind["pampa_inner_fit"] if stage["key"]["outer_fold"] == 1],
            key=lambda stage: stage["key"]["inner_basket"],
        )
        selection = next(
            stage for stage in self.by_kind["stopping_epoch_selection"] if stage["key"]["outer_fold"] == 1
        )
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            executor = object.__new__(D0FullExecutor)
            executor.project_root = project
            executor.ledger = ScientificStageLedger(project / "ledger")
            outputs = []
            evidence = {}
            for stage in [*inner, selection]:
                claim = executor.ledger.claim(stage)
                output = project / claim.ledger_path.parent.name / "unused"
                output = project / str(stage["namespace"]["output_attempt_template"]).format(attempt_number=1)
                output.mkdir(parents=True)
                outputs.append(output)
                for relative in stage["expected_outputs"]:
                    if relative not in {"artifact_manifest.json", "bundle_completion_certificate.json"}:
                        (output / relative).parent.mkdir(parents=True, exist_ok=True)
                        (output / relative).write_text("{}\n", encoding="utf-8")
                if stage["execution"]["gpu"]:
                    trace = output / "training_trace.json"
                    file_record = {
                        "relative_path": "training_trace.json",
                        "size_bytes": trace.stat().st_size,
                        "sha256": stream_sha256(trace),
                    }
                    evidence[stage["scientific_identity_sha256"]] = make_resource_evidence(
                        runtime_seconds=0,
                        max_memory_reserved_bytes=0,
                        gpu_stage=True,
                        source="certificate-test",
                        evidence_files=[file_record],
                    )
                    _write_stage_artifact_manifest(stage, output)
                else:
                    evidence[stage["scientific_identity_sha256"]] = resource(gpu=False)
            self.assertFalse(executor._recover_sealed_inner_bundle(inner, selection))
            certificate = {
                "schema_version": "scaffoldseal-inner-bundle-complete-v1",
                "stage_identities": [stage["scientific_identity_sha256"] for stage in [*inner, selection]],
                "inner_artifact_manifest_sha256": {
                    stage["scientific_identity_sha256"]: stream_sha256(output / "artifact_manifest.json")
                    for stage, output in zip(inner, outputs[:4])
                },
                "selected_epoch_sha256": stream_sha256(outputs[-1] / "selected_epoch.json"),
                "resource_evidence_by_stage": evidence,
            }
            write_json(outputs[-1] / "bundle_completion_certificate.json", certificate)
            _write_stage_artifact_manifest(selection, outputs[-1])
            self.assertTrue(executor._recover_sealed_inner_bundle(inner, selection))
            self.assertTrue(all(executor.ledger.latest_attempt(stage)["status"] == "COMPLETED_ACCEPTED" for stage in [*inner, selection]))

    def test_live_status_is_atomic_no_training_and_ledger_consistent(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            ledger = ScientificStageLedger(project / "ledger")
            stage = self.plan["stages"][0]
            claim = ledger.claim(stage)
            ledger.record_interrupted(claim, "test interruption", resource(gpu=True))
            status = publish_live_status(
                project,
                self.plan,
                ledger,
                phase="REPAIR_TEST",
                training_state="NO_TRAINING",
                external_acceptance=False,
                next_action="wait for review",
            )
            status_path = project / "artifacts/d0_full_run/LIVE_STATUS.json"
            old_bytes = status_path.read_bytes()
            parsed = json.loads(old_bytes)
            self.assertEqual(parsed["interrupted_attempts"], 1)
            self.assertEqual(parsed["training_state"], "NO_TRAINING")
            self.assertEqual(parsed["runner_lock"]["effective_state"], "LOCKED_NO_TRAINING")
            self.assertFalse(parsed["heldout_labels_visible"])
            self.assertFalse(parsed["trial_metrics_visible"])
            self.assertEqual(
                parsed["status_sha256"],
                parsed["snapshot_sha256"],
            )
            canonical, markdown = read_live_status_bundle(project)
            self.assertEqual(canonical, parsed)
            self.assertIn(parsed["generation"], markdown)
            self.assertTrue((project / "LIVE_PROGRESS.md").is_file())
            self.assertNotIn("permeability", json.dumps(status).lower())

    def test_status_generation_pointer_never_exposes_mixed_views(self):
        boundaries = (
            "after_generation_json",
            "after_generation_markdown",
            "after_bundle_manifest",
            "before_pointer_advance",
            "after_pointer_advance",
            "after_root_json_view",
            "after_root_markdown_view",
        )
        for boundary in boundaries:
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as temporary:
                project = Path(temporary)
                ledger = ScientificStageLedger(project / "ledger")
                old = publish_live_status(
                    project,
                    self.plan,
                    ledger,
                    phase="OLD",
                    training_state="NO_TRAINING",
                    external_acceptance=False,
                    next_action="old",
                )
                stage = self.plan["stages"][0]
                claim = ledger.claim(stage)
                ledger.record_interrupted(claim, "new interruption", resource(gpu=True))

                def fail_at(observed):
                    if observed == boundary:
                        raise OSError(f"injected {boundary}")

                with self.assertRaises(OSError):
                    publish_live_status(
                        project,
                        self.plan,
                        ledger,
                        phase="NEW",
                        training_state="NO_TRAINING",
                        external_acceptance=False,
                        next_action="new",
                        failure_injector=fail_at,
                    )
                visible, markdown = read_live_status_bundle(project, repair_views=True)
                self.assertIn(visible["generation"], markdown)
                self.assertEqual(visible["snapshot_sha256"], visible["status_sha256"])
                if boundary in {
                    "after_generation_json",
                    "after_generation_markdown",
                    "after_bundle_manifest",
                    "before_pointer_advance",
                }:
                    self.assertEqual(visible["generation"], old["generation"])
                    self.assertEqual(visible["interrupted_attempts"], 0)
                else:
                    self.assertNotEqual(visible["generation"], old["generation"])
                    self.assertEqual(visible["interrupted_attempts"], 1)

    def test_cli_execute_neither_trains_nor_rewrites_candidate(self):
        manifest = ROOT / "d0_full_run_manifest.json"
        before = stream_sha256(manifest)
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "src/d0_full_runner.py"),
                    "--execute",
                    "--project-root",
                    str(ROOT),
                    "--baseline-root",
                    str(BASELINE),
                    "--manifest",
                    str(manifest),
                    "--acceptance",
                    str(ROOT / "d0_full_run_acceptance.json"),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("pre-fit locked", result.stderr)
            self.assertEqual(stream_sha256(manifest), before)
            self.assertFalse(any(Path(temporary).iterdir()))


if __name__ == "__main__":
    unittest.main()
