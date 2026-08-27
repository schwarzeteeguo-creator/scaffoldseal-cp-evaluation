import json
from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_manifests import (
    OUTCOME_DERIVED_COLUMNS,
    OUTCOME_NAME_FRAGMENTS,
    PUBLIC_SCHEMA_ALLOWLISTS,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


class OutputTests(unittest.TestCase):
    def test_public_manifests_match_explicit_allowlists(self):
        for filename, expected in PUBLIC_SCHEMA_ALLOWLISTS.items():
            actual = tuple(pd.read_csv(ARTIFACTS / filename, nrows=0).columns)
            self.assertEqual(actual, expected, filename)

    def test_public_manifests_match_outcome_denylist(self):
        for filename in PUBLIC_SCHEMA_ALLOWLISTS:
            columns = {
                str(column).strip().lower()
                for column in pd.read_csv(ARTIFACTS / filename, nrows=0).columns
            }
            self.assertFalse(columns & OUTCOME_DERIVED_COLUMNS, filename)
            hits = {
                column
                for column in columns
                if any(fragment in column for fragment in OUTCOME_NAME_FRAGMENTS)
            }
            self.assertFalse(hits, filename)

    def test_final_rows_have_only_allowlisted_public_fields(self):
        split = pd.read_csv(ARTIFACTS / "split_manifest_public.csv")
        final_ids = set(split.loc[split["partition"] == "final_test", "curated_id"])
        self.assertEqual(len(final_ids), 1547)
        curated = pd.read_csv(ARTIFACTS / "curated_records_public.csv")
        final_public = curated.loc[curated["curated_id"].isin(final_ids)]
        self.assertEqual(set(final_public["curated_id"]), final_ids)
        self.assertEqual(
            tuple(final_public.columns),
            PUBLIC_SCHEMA_ALLOWLISTS["curated_records_public.csv"],
        )
        development = pd.read_csv(
            ARTIFACTS / "development_labeled.csv", usecols=["curated_id"]
        )
        self.assertTrue(final_ids.isdisjoint(set(development["curated_id"])))

    def test_repair_record_proves_mechanical_sanitization(self):
        record = json.loads(
            (ARTIFACTS / "public_artifact_repair.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            record["removed_columns"],
            ["replicate_min", "replicate_max", "replicate_spread"],
        )
        self.assertFalse(record["external_vault_read"])
        self.assertFalse(record["split_regenerated"])
        self.assertFalse(record["non_removed_values_changed"])
        self.assertEqual(record["rows_before"], record["rows_after"])

    def test_source_component_and_block_sealing(self):
        split = pd.read_csv(ARTIFACTS / "split_manifest_public.csv")
        self.assertEqual(split.groupby("source")["partition"].nunique().max(), 1)
        self.assertEqual(
            split.groupby("analogue_component_id")["partition"].nunique().max(), 1
        )
        self.assertEqual(split.groupby("sealed_block_id")["partition"].nunique().max(), 1)

    def test_development_labels_exclude_final_test(self):
        development = pd.read_csv(ARTIFACTS / "development_labeled.csv")
        self.assertIn("permeability", development.columns)
        self.assertNotIn("final_test", set(development["partition"]))

    def test_access_log_has_single_governance_incident(self):
        lines = (ROOT / "TEST_ACCESS_LOG.tsv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("timestamp\tperson_or_process\treason\t"))
        self.assertIn("DEV-M0-001", lines[1])
        self.assertIn("public-artifact", lines[1])

    def test_summary_gate_is_internally_consistent(self):
        summary = json.loads((ARTIFACTS / "build_summary.json").read_text(encoding="utf-8"))
        checks = summary["gate"]["checks"]
        leakage = summary["gate"]["leakage"]
        expected = all(checks.values()) and all(value == 0 for value in leakage.values())
        self.assertIs(summary["gate"]["admissible"], expected)
        self.assertEqual(summary["sealing_status"], "COMPROMISED_RETIRED")
        self.assertEqual(summary["test_access_log_entries"], 1)

    def test_retirement_marker_is_present(self):
        marker = (ROOT / "FINAL_TEST_RETIRED.md").read_text(encoding="utf-8")
        self.assertIn("COMPROMISED_RETIRED", marker)
        self.assertIn("must never be used", marker)


if __name__ == "__main__":
    unittest.main()
