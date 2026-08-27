from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from build_manifests import (
    CURATED_PUBLIC_COLUMNS,
    canonical_cycle,
    make_curated_public,
    token_one_edit,
)


class RuleTests(unittest.TestCase):
    def test_canonical_cycle_is_rotation_invariant(self):
        self.assertEqual(canonical_cycle(("A", "B", "C")), canonical_cycle(("B", "C", "A")))

    def test_exact_one_edit_accepts_cyclic_rotation(self):
        topology = "circle_head_to_tail|total=4|main=4|tokens=4"
        self.assertTrue(
            token_one_edit(("A", "B", "C", "D"), ("C", "D", "A", "X"), topology, topology)
        )

    def test_exact_one_edit_rejects_two_edits(self):
        topology = "circle_head_to_tail|total=4|main=4|tokens=4"
        self.assertFalse(
            token_one_edit(("A", "B", "C", "D"), ("A", "X", "C", "Y"), topology, topology)
        )

    def test_exact_one_edit_requires_same_topology(self):
        a = "circle_head_to_tail|total=4|main=4|tokens=4"
        b = "lariat|total=4|main=3|tokens=4"
        self.assertFalse(
            token_one_edit(("A", "B", "C", "D"), ("A", "B", "C", "X"), a, b)
        )

    def test_public_projection_drops_all_internal_outcome_fields(self):
        row = {column: "safe" for column in CURATED_PUBLIC_COLUMNS}
        row.update(
            {
                "permeability": -6.2,
                "replicate_min": -6.2,
                "replicate_max": -5.9,
                "replicate_spread": 0.3,
            }
        )
        public = make_curated_public(pd.DataFrame([row]))
        self.assertEqual(tuple(public.columns), CURATED_PUBLIC_COLUMNS)
        self.assertNotIn("permeability", public.columns)
        self.assertFalse(any(column.startswith("replicate_") for column in public.columns))


if __name__ == "__main__":
    unittest.main()
