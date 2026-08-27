import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nested_classical import FAMILY_COMPUTE_RANK, assemble_oof, choose_families


class NestedClassicalTests(unittest.TestCase):
    def test_selection_uses_only_inner_metrics_and_compute_tie_break(self):
        tables = {}
        for family in FAMILY_COMPUTE_RANK:
            rows = []
            for fold in range(1, 19):
                rows.append(
                    {
                        "outer_fold": fold,
                        "config_id": f"{family}_{fold}",
                        "source_macro_mae": 1.0,
                        "row_micro_mae": 0.5,
                        "selected": True,
                    }
                )
            tables[family] = pd.DataFrame(rows)
        tables["xgboost"].loc[0, "source_macro_mae"] = 0.9

        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            paths = {}
            for family, table in tables.items():
                path = Path(directory) / f"{family}.csv"
                table.to_csv(path, index=False)
                paths[family] = path
            result = choose_families(paths)

        winners = result.loc[result["selected_family"]].set_index("outer_fold")["family"]
        self.assertEqual(winners.loc[1], "xgboost")
        self.assertTrue((winners.loc[2:] == "ridge").all())
        self.assertEqual(len(winners), 18)

    def test_rejects_missing_fold(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            paths = {}
            for family in FAMILY_COMPUTE_RANK:
                path = Path(directory) / f"{family}.csv"
                pd.DataFrame(
                    {
                        "outer_fold": range(1, 18),
                        "config_id": [f"{family}_{fold}" for fold in range(1, 18)],
                        "source_macro_mae": 1.0,
                        "row_micro_mae": 0.5,
                        "selected": True,
                    }
                ).to_csv(path, index=False)
                paths[family] = path
            with self.assertRaises(ValueError):
                choose_families(paths)

    def test_rejects_population_not_exactly_6895(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = root / "frozen.csv"
            pd.DataFrame(
                {
                    "curated_id": ["id-1"],
                    "molecule_id": ["mol-1"],
                    "source": ["source-1"],
                    "analogue_component_id": ["component-1"],
                    "sealed_block_id": ["block-1"],
                    "outer_fold": [1],
                }
            ).to_csv(frozen, index=False)
            family_selection = pd.DataFrame(
                {
                    "outer_fold": range(1, 19),
                    "family": ["random_forest"] * 18,
                    "config_id": [f"rf-{fold}" for fold in range(1, 19)],
                    "selected_family": [True] * 18,
                }
            )
            predictions = []
            for seed in range(5):
                predictions.append(
                    {
                        "curated_id": "id-1",
                        "molecule_id": "mol-1",
                        "source": "source-1",
                        "analogue_component_id": "component-1",
                        "sealed_block_id": "block-1",
                        "outer_fold": 1,
                        "model": "random_forest",
                        "config_id": "rf-1",
                        "seed": seed,
                        "observed": 0.0,
                        "prediction": 0.0,
                    }
                )
            prediction_path = root / "rf.csv"
            pd.DataFrame(predictions).to_csv(prediction_path, index=False)
            with self.assertRaises(ValueError):
                assemble_oof(
                    family_selection,
                    {
                        "ridge": prediction_path,
                        "random_forest": prediction_path,
                        "xgboost": prediction_path,
                    },
                    frozen,
                )


if __name__ == "__main__":
    unittest.main()
