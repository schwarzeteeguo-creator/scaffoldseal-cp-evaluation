import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "v2_r0"
sys.path.insert(0, str(ROOT / "src"))

from build_v2_manifests import (  # noqa: E402
    DMPNN_NO_LEARNING_SMOKE_FILE,
    DEFAULT_CONFIG,
    FIT_BOUNDARY_POLICY_FILE,
    MANIFEST_FILES,
    build,
    greedy_group_folds,
)
from split_safe import (  # noqa: E402
    FitAuditTrail,
    OuterFoldContract,
    SplitSafePreprocessor,
    SplitViolation,
    canonical_id_hash,
    contract_manifest,
    contracts_from_manifests,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class V2GovernanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = pd.read_csv(
            ROOT / "artifacts" / "split_manifest_public.csv",
            usecols=[
                "curated_id",
                "molecule_id",
                "source",
                "analogue_component_id",
                "sealed_block_id",
            ],
        )
        cls.outer_records = pd.read_csv(ARTIFACTS / "outer_record_assignments.csv")
        cls.outer = pd.read_csv(ARTIFACTS / "outer_fold_manifest.csv")
        cls.inner = pd.read_csv(ARTIFACTS / "inner_basket_manifest.csv")
        cls.comparison = pd.read_csv(ARTIFACTS / "comparison_fold_manifest.csv")

    def test_outer_fold_identity_and_one_slot_per_record(self):
        self.assertEqual(len(self.outer_records), 6895)
        self.assertEqual(self.outer_records["curated_id"].nunique(), 6895)
        self.assertEqual(set(self.outer_records["outer_fold"]), set(range(1, 19)))
        expected = {
            block: index
            for index, block in enumerate(
                sorted(self.records["sealed_block_id"].unique()), start=1
            )
        }
        actual = self.outer_records.set_index("sealed_block_id")["outer_fold"].to_dict()
        for block, fold in expected.items():
            self.assertEqual(actual[block], fold)
        self.assertTrue(
            (self.outer_records["sealed_block_id"] == self.outer_records["outer_test_block"]).all()
        )

    def test_zero_outer_source_and_component_overlap(self):
        for test_block in sorted(self.records["sealed_block_id"].unique()):
            test = self.records[self.records["sealed_block_id"] == test_block]
            train = self.records[self.records["sealed_block_id"] != test_block]
            self.assertTrue(set(test["source"]).isdisjoint(set(train["source"])), test_block)
            self.assertTrue(
                set(test["analogue_component_id"]).isdisjoint(
                    set(train["analogue_component_id"])
                ),
                test_block,
            )

    def test_each_outer_fold_has_one_test_and_seventeen_train_blocks(self):
        for _, fold in self.outer.groupby("outer_fold"):
            self.assertEqual(len(fold), 18)
            self.assertEqual((fold["role"] == "test").sum(), 1)
            self.assertEqual((fold["role"] == "train").sum(), 17)
            test = fold.loc[fold["role"] == "test"].iloc[0]
            self.assertEqual(test["sealed_block_id"], test["outer_test_block"])

    def test_inner_baskets_follow_exact_frozen_greedy_rule(self):
        block_sizes = (
            self.records.groupby("sealed_block_id")["curated_id"]
            .size()
            .rename("n_curated_rows")
            .reset_index()
        )
        for outer_fold, observed in self.inner.groupby("outer_fold"):
            test_block = observed["outer_test_block"].iloc[0]
            available = block_sizes[block_sizes["sealed_block_id"] != test_block].copy()
            expected, _ = greedy_group_folds(available, "sealed_block_id", 4)
            actual = observed.set_index("sealed_block_id")["inner_basket"].to_dict()
            self.assertEqual(actual, expected, f"outer fold {outer_fold}")
            self.assertEqual(len(actual), 17)
            self.assertEqual(set(actual.values()), {1, 2, 3, 4})

    def test_comparison_group_isolation_and_one_fold_per_record(self):
        self.assertEqual(len(self.comparison), 6895)
        self.assertEqual(self.comparison["curated_id"].nunique(), 6895)
        pairs = (
            ("molecule_id", "molecule_id_fold"),
            ("murcko_chiral", "murcko_chiral_fold"),
            ("source", "source_fold"),
            ("analogue_component_id", "analogue_component_id_fold"),
        )
        for group_column, fold_column in pairs:
            self.assertEqual(
                self.comparison.groupby(group_column)[fold_column].nunique().max(),
                1,
                group_column,
            )
            self.assertEqual(set(self.comparison[fold_column]), {1, 2, 3, 4, 5})
            self.assertFalse(self.comparison[fold_column].isna().any())

    def test_comparison_assignments_follow_exact_frozen_greedy_rule(self):
        pairs = (
            ("molecule_id", "molecule_id_fold"),
            ("murcko_chiral", "murcko_chiral_fold"),
            ("source", "source_fold"),
            ("analogue_component_id", "analogue_component_id_fold"),
        )
        for group_column, fold_column in pairs:
            sizes = (
                self.comparison.groupby(group_column, sort=True)["curated_id"]
                .size()
                .rename("n_curated_rows")
                .reset_index()
            )
            expected, _ = greedy_group_folds(sizes, group_column, 5)
            actual = (
                self.comparison[[group_column, fold_column]]
                .drop_duplicates()
                .assign(**{group_column: lambda frame: frame[group_column].astype(str)})
                .set_index(group_column)[fold_column]
                .to_dict()
            )
            self.assertEqual(actual, expected, group_column)

    def test_manifest_schemas_are_label_blind(self):
        fragments = ("permeab", "label", "target", "outcome", "papp", "replicate")
        for filename in MANIFEST_FILES:
            columns = [
                str(column).lower()
                for column in pd.read_csv(ARTIFACTS / filename, nrows=0).columns
            ]
            self.assertFalse(
                [column for column in columns if any(fragment in column for fragment in fragments)],
                filename,
            )

    def test_raw_only_analysis_table_matches_frozen_population(self):
        analysis = pd.read_csv(ARTIFACTS / "analysis_all_labels.csv")
        self.assertEqual(len(analysis), 6895)
        self.assertEqual(analysis["curated_id"].nunique(), 6895)
        self.assertIn("permeability", analysis.columns)
        summary = json.loads((ARTIFACTS / "build_summary_v2.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["accessible_former_development_rows_checked"], 5348)
        self.assertEqual(summary["accessible_former_development_median_mismatches"], 0)
        self.assertFalse(summary["retired_partition_column_parsed"])
        self.assertEqual(summary["pre_fit_contract"]["outer_contracts"], 18)
        self.assertEqual(summary["pre_fit_contract"]["contract_rows"], 72)

    def test_v2_builder_has_no_retired_outcome_store_reference(self):
        # Construct the prohibited legacy tokens so they are not themselves embedded
        # as a contiguous reference in the v2 test source.
        tokens = ["scaffoldseal" + "_sealed", "final" + "_labels_sealed"]
        paths = [ROOT / "config_v2.yaml", ROOT / "src" / "build_v2_manifests.py"]
        for path in paths:
            text = path.read_text(encoding="utf-8").lower()
            for token in tokens:
                self.assertNotIn(token, text, path.name)
        summary = json.loads((ARTIFACTS / "build_summary_v2.json").read_text(encoding="utf-8"))
        input_names = "\n".join(
            [
                *summary["full_file_sha256"],
                *summary["legacy_allowed_projection_sha256"],
            ]
        ).lower()
        for token in tokens:
            self.assertNotIn(token, input_names)

    def test_all_eighteen_actual_manifests_satisfy_prefit_guard(self):
        contracts = contracts_from_manifests(
            self.records[["curated_id", "sealed_block_id"]],
            self.outer_records,
            self.inner,
        )
        observed = pd.read_csv(ARTIFACTS / "pre_fit_contract_manifest.csv")
        pd.testing.assert_frame_equal(contract_manifest(contracts), observed)
        synthetic = pd.DataFrame(
            {
                "curated_id": self.records["curated_id"].astype(str),
                "sentinel_feature": np.arange(len(self.records), dtype=float),
            }
        )
        for fold, contract in contracts.items():
            train = contract.outer_training_batch(synthetic)
            contract.validate_fit_batch(train)
            self.assertEqual(canonical_id_hash(train.ids), canonical_id_hash(contract.outer_train_ids))
            outer = contract.outer_test_batch(synthetic)
            with self.assertRaises(SplitViolation, msg=f"outer fold {fold}"):
                contract.validate_fit_batch(outer)

    def test_sentinel_outer_changes_cannot_affect_prefit_statistics(self):
        contract = OuterFoldContract(
            outer_fold=1,
            outer_train_ids={"T1", "T2", "T3", "T4"},
            outer_test_ids={"X1", "X2"},
            inner_basket_by_id={"T1": 1, "T2": 2, "T3": 3, "T4": 4},
        )
        base = pd.DataFrame(
            {
                "curated_id": ["T1", "T2", "T3", "T4", "X1", "X2"],
                "f_keep": [1.0, 2.0, np.nan, 4.0, 10.0, 20.0],
                "f_binary": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                "f_constant": [7.0, 7.0, 7.0, 7.0, 7.0, 7.0],
                "target": [0.0, 1.0, 0.0, 1.0, -999.0, 999.0],
            }
        )
        mutated = base.copy()
        mutated.loc[mutated["curated_id"].str.startswith("X"), [
            "f_keep", "f_binary", "f_constant", "target"
        ]] = [[1e12, -1e12, 3e12, -4e12], [-9e12, 8e12, -7e12, 6e12]]

        audits = [FitAuditTrail(), FitAuditTrail()]
        processors = []
        for frame, audit in zip((base, mutated), audits):
            processor = SplitSafePreprocessor(contract, audit)
            processor.fit(
                contract.outer_training_batch(frame),
                ["f_keep", "f_binary", "f_constant"],
                target_column="target",
            )
            processors.append(processor)
        self.assertEqual(processors[0].medians_, processors[1].medians_)
        self.assertEqual(processors[0].means_, processors[1].means_)
        self.assertEqual(processors[0].scales_, processors[1].scales_)
        self.assertEqual(processors[0].kept_features_, processors[1].kept_features_)
        self.assertEqual(processors[0].kept_features_, ("f_keep", "f_binary"))
        self.assertEqual(processors[0].fit_ids_sha256_, canonical_id_hash({"T1", "T2", "T3", "T4"}))

    def test_exact_fit_batches_and_outer_inputs_raise(self):
        contract = OuterFoldContract(
            1,
            {"T1", "T2", "T3", "T4"},
            {"X1"},
            {"T1": 1, "T2": 2, "T3": 3, "T4": 4},
        )
        frame = pd.DataFrame(
            {
                "curated_id": ["T1", "T2", "T3", "T4", "X1"],
                "feature": [1.0, 2.0, 3.0, 4.0, 999.0],
                "target": [0.0, 0.0, 1.0, 1.0, -999.0],
            }
        )
        audit = FitAuditTrail()
        contract.validate_exact_fit_batch(contract.outer_training_batch(frame))
        contract.validate_exact_fit_batch(contract.inner_training_batch(frame, 1))
        with self.assertRaises(SplitViolation):
            contract.validate_exact_fit_batch(contract.outer_test_batch(frame))
        with self.assertRaises(SplitViolation):
            SplitSafePreprocessor(contract, audit).fit(
                contract.outer_test_batch(frame), ["feature"], target_column="target"
            )
        role_frame = frame.copy()
        role_frame["role"] = "outer_test"
        with self.assertRaises(SplitViolation):
            SplitSafePreprocessor(contract, audit).fit(
                contract.outer_training_batch(role_frame), ["feature"], target_column="target"
            )

    def test_outer_validation_or_cross_basket_inner_pair_raises(self):
        contract = OuterFoldContract(
            1,
            {"T1", "T2", "T3", "T4"},
            {"X1"},
            {"T1": 1, "T2": 2, "T3": 3, "T4": 4},
        )
        frame = pd.DataFrame(
            {
                "curated_id": ["T1", "T2", "T3", "T4", "X1"],
                "feature": [1.0, 2.0, 3.0, 4.0, 5.0],
                "target": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        with self.assertRaises(SplitViolation):
            contract.validate_inner_pair(
                contract.inner_training_batch(frame, 1),
                contract.outer_test_batch(frame),
                1,
            )
        with self.assertRaises(SplitViolation):
            contract.validate_inner_pair(
                contract.inner_training_batch(frame, 1),
                contract.inner_validation_batch(frame, 2),
                1,
            )

    def test_legacy_partition_mutation_is_byte_invariant(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            canonical_target = temporary / "canonical_build"
            canonical_summary = build(DEFAULT_CONFIG, canonical_target)
            overrides = {}
            sources = {
                "frozen_blocks": ROOT / "artifacts" / "source_component_blocks.csv",
                "frozen_group_mapping": ROOT / "artifacts" / "split_manifest_public.csv",
                "accessible_former_development": ROOT / "artifacts" / "development_labeled.csv",
            }
            for key, source in sources.items():
                frame = pd.read_csv(source)
                self.assertIn("partition", frame.columns)
                frame["partition"] = [
                    f"ARBITRARY_UNUSED_ROLE_{index % 7}" for index in range(len(frame))
                ]
                destination = temporary / f"{key}.csv"
                frame.to_csv(destination, index=False, lineterminator="\n")
                overrides[key] = destination
            target = temporary / "mutated_build"
            mutated_summary = build(
                DEFAULT_CONFIG,
                target,
                legacy_input_overrides=overrides,
            )
            committed_summary = json.loads(
                (ARTIFACTS / "build_summary_v2.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                mutated_summary["legacy_allowed_projection_sha256"],
                canonical_summary["legacy_allowed_projection_sha256"],
            )
            current_build_names = [
                *MANIFEST_FILES,
                FIT_BOUNDARY_POLICY_FILE,
                DMPNN_NO_LEARNING_SMOKE_FILE,
                "analysis_all_labels.csv",
                "build_summary_v2.json",
                "SHA256SUMS",
                "data_manifest_raw.sha256",
            ]
            for name in current_build_names:
                self.assertEqual(
                    sha256(target / name), sha256(canonical_target / name), name
                )

            # The committed R0 provenance summary is an immutable QA4 snapshot: its
            # full-file hashes intentionally describe the code at that gate.  R1 may
            # extend split_safe.py without rewriting R0.  Scientific and policy
            # artifacts must nevertheless stay byte-identical to the frozen gate.
            frozen_scientific_names = [
                *MANIFEST_FILES,
                FIT_BOUNDARY_POLICY_FILE,
                DMPNN_NO_LEARNING_SMOKE_FILE,
                "analysis_all_labels.csv",
                "data_manifest_raw.sha256",
            ]
            self.assertEqual(
                mutated_summary["legacy_allowed_projection_sha256"],
                committed_summary["legacy_allowed_projection_sha256"],
            )
            for name in frozen_scientific_names:
                self.assertEqual(sha256(target / name), sha256(ARTIFACTS / name), name)

    def test_deterministic_rebuild_matches_committed_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "v2_r0"
            build(DEFAULT_CONFIG, target)
            frozen_scientific_names = [
                *MANIFEST_FILES,
                FIT_BOUNDARY_POLICY_FILE,
                DMPNN_NO_LEARNING_SMOKE_FILE,
                "analysis_all_labels.csv",
                "data_manifest_raw.sha256",
            ]
            for name in frozen_scientific_names:
                self.assertEqual(sha256(target / name), sha256(ARTIFACTS / name), name)

    def test_checksum_manifest_matches_files(self):
        for line in (ARTIFACTS / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            digest, filename = line.split("  ", 1)
            self.assertEqual(digest, sha256(ARTIFACTS / filename), filename)

    def test_r0_contains_no_model_or_training_artifacts(self):
        forbidden_suffixes = {".pt", ".pth", ".ckpt", ".joblib", ".pkl", ".onnx"}
        forbidden_fragments = ("prediction", "checkpoint", "epoch_log", "training_log")
        hits = []
        for path in ARTIFACTS.rglob("*"):
            if not path.is_file():
                continue
            lowered = path.name.lower()
            if path.suffix.lower() in forbidden_suffixes or any(
                fragment in lowered for fragment in forbidden_fragments
            ):
                hits.append(path.relative_to(ROOT).as_posix())
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
