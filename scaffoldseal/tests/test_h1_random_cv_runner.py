import json
import subprocess
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT.parent / "baseline_candidates" / "BenchmarkCycPeptMP"
sys.path.insert(0, str(ROOT / "src"))

from h1_random_cv_runner import (
    OUTER_FOLDS,
    PREFIT_REVIEW_ACCEPTED,
    REAL_EXECUTION_AUTHORIZED,
    build_full_plan,
    canonical_id_hash,
    scientific_runner_record,
    stream_sha256,
    validate_plan,
)


class H1RandomCvRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = build_full_plan(ROOT, BASELINE)

    def test_candidate_is_exact_locked_56_stage_graph(self):
        validate_plan(self.plan)
        self.assertFalse(PREFIT_REVIEW_ACCEPTED)
        self.assertFalse(REAL_EXECUTION_AUTHORIZED)
        self.assertEqual(self.plan["counts"]["manifest_stage_records_total"], 56)
        self.assertEqual(self.plan["counts"]["scientific_fits_total"], 50)
        self.assertEqual(self.plan["counts"]["expected_oof_prediction_rows"], 34475)
        self.assertEqual(self.plan["scientific_lock"]["cv"]["outer_folds"], list(range(1, 6)))
        self.assertEqual(
            self.plan["scientific_lock"]["cv"]["outer_seed_indices"], list(range(5))
        )

    def test_plan_is_byte_deterministic(self):
        self.assertEqual(build_full_plan(ROOT, BASELINE), self.plan)

    def test_molecules_do_not_cross_outer_folds_and_contract_hashes_match(self):
        outer = pd.read_csv(ROOT / "artifacts/h1_random_cv_r0/outer_record_assignments.csv")
        inner = pd.read_csv(ROOT / "artifacts/h1_random_cv_r0/inner_id_basket_manifest.csv")
        contracts = pd.read_csv(
            ROOT / "artifacts/h1_random_cv_r0/pre_fit_contract_manifest.csv"
        )
        self.assertEqual(len(outer), 6895)
        self.assertEqual(outer.groupby("molecule_id")["outer_fold"].nunique().max(), 1)
        all_ids = set(outer["curated_id"].astype(str))
        for fold in OUTER_FOLDS:
            test_ids = set(
                outer.loc[outer["outer_fold"].astype(int).eq(fold), "curated_id"].astype(str)
            )
            fold_inner = inner.loc[inner["outer_fold"].astype(int).eq(fold)]
            self.assertEqual(set(fold_inner["curated_id"].astype(str)), all_ids - test_ids)
            for basket in range(1, 5):
                valid_ids = set(
                    fold_inner.loc[
                        fold_inner["inner_basket"].astype(int).eq(basket), "curated_id"
                    ].astype(str)
                )
                fit_ids = all_ids - test_ids - valid_ids
                row = contracts.loc[
                    contracts["outer_fold"].astype(int).eq(fold)
                    & contracts["inner_basket"].astype(int).eq(basket)
                ].iloc[0]
                self.assertFalse(fit_ids & valid_ids)
                self.assertFalse((fit_ids | valid_ids) & test_ids)
                self.assertEqual(canonical_id_hash(fit_ids), row["fit_ids_sha256"])
                self.assertEqual(
                    canonical_id_hash(valid_ids), row["inner_validation_ids_sha256"]
                )
                self.assertEqual(canonical_id_hash(test_ids), row["outer_test_ids_sha256"])

    def test_execute_is_hard_blocked_with_pending_acceptance(self):
        runner = scientific_runner_record(ROOT)
        acceptance = {
            "schema_version": "scaffoldseal-h1-random-cv-external-acceptance-v1",
            "accepted": False,
            "execution_authorized": False,
            "candidate_manifest_relative_path": "h1_random_cv_manifest.json",
            "candidate_manifest_sha256": stream_sha256(ROOT / "h1_random_cv_manifest.json"),
            "candidate_plan_sha256": self.plan["plan_sha256"],
            "runner_scientific_sha256": runner["scientific_sha256"],
            "authorized_runner_exact": runner,
        }
        path = ROOT / "h1_random_cv_acceptance.json"
        path.write_text(json.dumps(acceptance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "src/h1_random_cv_runner.py"),
                "--project-root",
                str(ROOT),
                "--baseline-root",
                str(BASELINE),
                "--execute",
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("independent review", result.stderr.lower())
        self.assertFalse((ROOT / "artifacts/h1_random_cv_d0_ledger_v1").exists())


if __name__ == "__main__":
    unittest.main()
