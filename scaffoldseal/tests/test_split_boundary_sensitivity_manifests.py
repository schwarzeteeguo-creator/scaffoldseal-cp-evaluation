from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_split_boundary_sensitivity_manifests import TARGET_SIZES, build, csv_bytes


class SplitBoundarySensitivityManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        comparison = ROOT / "artifacts/v2_r0/comparison_fold_manifest.csv"
        joint = ROOT / "artifacts/v2_r0/outer_record_assignments.csv"
        cls.frames, cls.summary = build(comparison, joint)

    def test_exact_matched_sizes_and_complete_oof(self) -> None:
        records = self.frames["matched_size_random_record_assignments.csv"]
        self.assertEqual(len(records), 6895)
        self.assertEqual(records["curated_id"].nunique(), 6895)
        counts = tuple(records.groupby("matched_size_random_fold").size().reindex(range(1, 19)))
        self.assertEqual(counts, TARGET_SIZES)

    def test_molecules_do_not_cross_matched_folds(self) -> None:
        records = self.frames["matched_size_random_record_assignments.csv"]
        self.assertEqual(records.groupby("molecule_id")["matched_size_random_fold"].nunique().max(), 1)

    def test_existing_group_boundaries_remain_intact(self) -> None:
        ladder = self.frames["evaluation_boundary_ladder_manifest.csv"]
        for group, fold in (
            ("molecule_id", "molecule_id_fold"),
            ("source", "source_fold"),
            ("analogue_component_id", "analogue_component_id_fold"),
            ("sealed_block_id", "joint_outer_fold"),
        ):
            self.assertEqual(ladder.groupby(group)[fold].nunique().max(), 1)

    def test_inner_baskets_exclude_outer_fold(self) -> None:
        inner = self.frames["matched_size_random_inner_baskets.csv"]
        self.assertEqual(len(inner), 18 * 17)
        self.assertTrue((inner["outer_fold"] != inner["matched_size_random_fold"]).all())
        self.assertEqual(inner.groupby("outer_fold")["inner_basket"].nunique().min(), 4)

    def test_manifests_are_label_blind(self) -> None:
        forbidden = ("permeab", "label", "target", "outcome", "papp", "replicate")
        for frame in self.frames.values():
            for column in frame.columns:
                self.assertFalse(any(fragment in column.lower() for fragment in forbidden))

    def test_rerun_is_byte_identical(self) -> None:
        comparison = ROOT / "artifacts/v2_r0/comparison_fold_manifest.csv"
        joint = ROOT / "artifacts/v2_r0/outer_record_assignments.csv"
        rerun, _ = build(comparison, joint)
        for name, frame in self.frames.items():
            self.assertEqual(csv_bytes(frame), csv_bytes(rerun[name]))


if __name__ == "__main__":
    unittest.main()

