import unittest
from pathlib import Path
import json
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from d0_pilot_governance import (
    DispatchAlreadyRecorded,
    SingleDispatchLedger,
    exact_outer_training_rows,
    inventory_attempt_roots,
    read_lossless_prediction_artifact,
    stream_sha256,
    write_lossless_prediction_artifacts,
)
from r1c0_dmpnn_pilot import (
    prove_pretraining_equivalence,
    project_joint_lobo_hours,
    require_full_lobo_prefit_review,
)


class D0PilotPolicyTests(unittest.TestCase):
    def test_pretraining_equivalence_requires_exact_epoch_and_ids(self):
        base = {
            "split_ids": {"train": "a", "validation": "b", "test": "c"},
            "split_counts": {"train": 8, "validation": 1, "test": 1},
            "best_epoch": 2,
            "stopping_epoch": 3,
            "probe_ids_sha256": "d",
            "history": [1.0, 0.5, 0.5],
            "probe_predictions": [0.1, 0.2],
        }
        self.assertTrue(prove_pretraining_equivalence(base, dict(base))["pass"])
        changed = dict(base, best_epoch=3)
        self.assertFalse(prove_pretraining_equivalence(base, changed)["pass"])

    def test_joint_projection_counts_18_inner_and_90_outer_stages(self):
        inner = [{"runtime_seconds": 10.0}] * 4
        outer = {"runtime_seconds": 20.0, "n_train": 17, "outer_fold": 1}
        assignments = pd.DataFrame(
            {"curated_id": [f"id-{fold}" for fold in range(1, 19)], "outer_fold": range(1, 19)}
        )
        self.assertEqual(exact_outer_training_rows(assignments).tolist(), [17] * 18)
        hours = project_joint_lobo_hours(inner, outer, assignments, 30.0)
        self.assertAlmostEqual(hours, (40 * 18 + 20 * 90 + 30 * 5) / 3600)

    def test_one_shot_ledger_blocks_namespace_change_and_full_runner_is_absent(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = SingleDispatchLedger(Path(temporary))
            identity = "a" * 64
            ledger.claim(identity, {"output_dir": "first"})
            with self.assertRaises(DispatchAlreadyRecorded):
                ledger.claim(identity, {"output_dir": "different"})
        with self.assertRaisesRegex(RuntimeError, "pre-fit review"):
            require_full_lobo_prefit_review()

    def test_one_shot_ledger_is_process_safe_without_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            code = (
                "from pathlib import Path; import sys; "
                "from d0_pilot_governance import SingleDispatchLedger; "
                "SingleDispatchLedger(Path(sys.argv[1])).claim('b'*64, {'stub': True})"
            )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(ROOT / "src")
            processes = [
                subprocess.Popen(
                    [sys.executable, "-c", code, temporary],
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                for _ in range(6)
            ]
            results = [process.communicate(timeout=20) + (process.returncode,) for process in processes]
            self.assertEqual(sum(result[2] == 0 for result in results), 1)
            self.assertEqual(len(list(Path(temporary).glob("*.json"))), 1)

    def test_lossless_prediction_round_trip_and_hash_without_model(self):
        values = np.array([0.0, -0.0, np.nextafter(1.0, 2.0)], dtype=np.float64)
        table = pd.DataFrame(
            {
                "curated_id": ["a", "b", "c"],
                "outer_fold": [1, 2, 3],
                "seed": [0, 1, 2],
                "observed_log10_papp": values,
                "prediction_normalized": values[::-1],
                "prediction_log10_papp": values * 2.0 - 6.0,
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            lossless = Path(temporary) / "predictions.lossless.json"
            csv_path = Path(temporary) / "predictions.csv"
            evidence = write_lossless_prediction_artifacts(table, lossless, csv_path)
            recovered = read_lossless_prediction_artifact(lossless)
            for column in table.columns[3:]:
                expected_bits = table[column].to_numpy(np.float64).view(np.uint64)
                observed_bits = recovered[column].to_numpy(np.float64).view(np.uint64)
                np.testing.assert_array_equal(expected_bits, observed_bits)
            self.assertEqual(evidence["record_count"], 3)
            self.assertTrue(csv_path.read_text(encoding="utf-8").startswith("curated_id,"))
            payload = json.loads(lossless.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["records"][1]["observed_log10_papp_ieee754_be"],
                struct.pack(">d", -0.0).hex(),
            )

    def test_attempt_inventory_hashes_every_file_but_copies_only_small_traces(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "attempt"
            archive = Path(temporary) / "archive"
            (root / "stage").mkdir(parents=True)
            trace = root / "stage" / "training_trace.json"
            metadata = root / "stage" / "tasks.json"
            checkpoint = root / "stage" / "checkpoint1.pt"
            trace.write_text('{"fit_calls": 1}\n', encoding="utf-8")
            metadata.write_text('{"tasks": []}\n', encoding="utf-8")
            checkpoint.write_bytes(b"checkpoint-evidence")
            inventory = inventory_attempt_roots({"stub": root}, archive)
            self.assertEqual(inventory["totals"]["file_count"], 3)
            self.assertEqual(inventory["totals"]["copied_trace_count"], 1)
            copied = archive / "traces" / "stub" / "stage" / "training_trace.json"
            self.assertEqual(stream_sha256(copied), stream_sha256(trace))
            self.assertFalse((archive / "traces" / "stub" / "stage" / "tasks.json").exists())
            self.assertFalse((archive / "traces" / "stub" / "stage" / "checkpoint1.pt").exists())
            records = {record["relative_path"]: record for record in inventory["files"]}
            self.assertEqual(records["stage/checkpoint1.pt"]["size_bytes"], 19)
            self.assertIn("modified_time_ns", records["stage/checkpoint1.pt"])


if __name__ == "__main__":
    unittest.main()
