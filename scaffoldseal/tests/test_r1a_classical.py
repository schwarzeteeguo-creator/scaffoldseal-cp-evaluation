import ast
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

try:
    import sklearn  # noqa: F401
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("R1a classical tests require the pinned local baseline environment") from exc


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from r1a_classical import (  # noqa: E402
    ALPHAS,
    run_lobo,
    select_alpha,
    validate_fit_audit,
)
from split_safe import (  # noqa: E402
    FitAuditTrail,
    OuterFoldContract,
    SplitSafeFitExecutor,
    SplitSafeMixedPreprocessor,
    contracts_from_manifests,
)


SYNTHETIC_FEATURES = ("bit_a", "bit_b", "continuous", "continuous__missing")
SYNTHETIC_PASSTHROUGH = ("bit_a", "bit_b", "continuous__missing")


def synthetic_lobo_inputs():
    ids = [f"R{index:02d}" for index in range(1, 19)]
    blocks = [f"B{index:02d}" for index in range(1, 19)]
    feature = pd.DataFrame({"curated_id": ids})
    feature["bit_a"] = [index % 2 for index in range(18)]
    feature["bit_b"] = [index % 3 == 0 for index in range(18)]
    feature["continuous"] = np.arange(18, dtype=float)
    feature["continuous__missing"] = 0
    feature["permeability"] = np.linspace(-6.0, -4.0, 18)
    feature = feature.loc[:, ["curated_id", *SYNTHETIC_FEATURES, "permeability"]]
    metadata = pd.DataFrame(
        {
            "curated_id": ids,
            "molecule_id": [f"M{index:02d}" for index in range(1, 19)],
            "source": [f"S{index:02d}" for index in range(1, 19)],
            "analogue_component_id": [f"A{index:02d}" for index in range(1, 19)],
            "sealed_block_id": blocks,
            "outer_fold": list(range(1, 19)),
        }
    )
    records = metadata.loc[:, ["curated_id", "sealed_block_id"]]
    outer = metadata.loc[:, ["curated_id", "outer_fold"]]
    inner_rows = []
    for outer_fold, held_out in enumerate(blocks, start=1):
        available = [block for block in blocks if block != held_out]
        for position, block in enumerate(available):
            inner_rows.append(
                {
                    "outer_fold": outer_fold,
                    "sealed_block_id": block,
                    "inner_basket": position % 4 + 1,
                }
            )
    contracts = contracts_from_manifests(records, outer, pd.DataFrame(inner_rows))
    return feature, metadata, contracts


class R1AClassicalIntegrationTests(unittest.TestCase):
    def test_exact_alpha_selection_and_tie_breaks(self):
        rows = [
            {
                "alpha": alpha,
                "config_id": f"ridge_alpha_{alpha:g}",
                "source_macro_mae": 1.0,
                "row_micro_mae": 1.0,
                "compute_rank": 1,
            }
            for alpha in ALPHAS
        ]
        self.assertEqual(float(select_alpha(rows)["alpha"]), 0.01)
        rows[3]["row_micro_mae"] = 0.9
        self.assertEqual(float(select_alpha(rows)["alpha"]), 10.0)
        rows[4]["source_macro_mae"] = 0.99
        self.assertEqual(float(select_alpha(rows)["alpha"]), 100.0)

    def test_mixed_preprocessing_is_train_only_and_binary_unscaled(self):
        contract = OuterFoldContract(
            1,
            {"T1", "T2", "T3", "T4"},
            {"X1"},
            {"T1": 1, "T2": 2, "T3": 3, "T4": 4},
        )
        base = pd.DataFrame(
            {
                "curated_id": ["T1", "T2", "T3", "T4", "X1"],
                "bit": [0, 1, 0, 1, 1],
                "continuous": [1.0, 2.0, 3.0, 4.0, 5.0],
                "y": [1.0, 2.0, 3.0, 4.0, 999.0],
            }
        )
        changed = base.copy()
        changed.loc[changed.curated_id == "X1", "continuous"] = 1e12
        processors = []
        transformed_bits = []
        for frame in (base, changed):
            audit = FitAuditTrail()
            executor = SplitSafeFitExecutor(contract, audit)
            processor = SplitSafeMixedPreprocessor(
                contract, audit, passthrough_columns=["bit"]
            )
            executor.fit_preprocessor(
                processor,
                contract.outer_training_batch(frame),
                ["bit", "continuous"],
                target_column="y",
            )
            processors.append(processor)
            transformed = processor.transform(contract.outer_test_batch(frame))
            transformed_bits.append(float(transformed.loc[0, "bit"]))
        self.assertEqual(processors[0].statistics_sha256_, processors[1].statistics_sha256_)
        self.assertEqual(processors[0].means_, processors[1].means_)
        self.assertEqual(transformed_bits, [1.0, 1.0])

    def test_full_18_fold_coverage_audit_and_deterministic_rerun(self):
        feature, metadata, contracts = synthetic_lobo_inputs()
        with tempfile.TemporaryDirectory() as temporary:
            first = run_lobo(
                feature,
                metadata,
                contracts,
                Path(temporary) / "run_first",
                feature_columns=SYNTHETIC_FEATURES,
                passthrough_columns=SYNTHETIC_PASSTHROUGH,
            )
            second = run_lobo(
                feature,
                metadata,
                contracts,
                Path(temporary) / "run_second",
                feature_columns=SYNTHETIC_FEATURES,
                passthrough_columns=SYNTHETIC_PASSTHROUGH,
            )
        for result in (first, second):
            for _, group in result.oof_predictions.groupby("model"):
                self.assertEqual(len(group), 18)
                self.assertEqual(group["curated_id"].nunique(), 18)
            self.assertEqual(len(result.inner_selection), 18 * len(ALPHAS))
            self.assertEqual(int(result.inner_selection["selected"].sum()), 18)
            validate_fit_audit(result.audit_records, contracts)
        pd.testing.assert_frame_equal(first.oof_predictions, second.oof_predictions)
        pd.testing.assert_frame_equal(first.inner_selection, second.inner_selection)
        self.assertEqual(first.audit_records, second.audit_records)

    def test_runner_has_no_direct_fit_call(self):
        tree = ast.parse((ROOT / "src" / "r1a_classical.py").read_text(encoding="utf-8"))
        direct_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "fit"
        ]
        self.assertEqual(direct_calls, [])


if __name__ == "__main__":
    unittest.main()
