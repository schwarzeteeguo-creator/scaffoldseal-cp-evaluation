import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

try:
    import sklearn
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("R1b1 RF tests require the pinned local baseline environment") from exc


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from r1b1_random_forest import (  # noqa: E402
    COMPUTE_RANK,
    EXPECTED_SKLEARN,
    N_TREES,
    OUTER_SEEDS,
    RFConfig,
    RF_GRID,
    execute_inner_checkpoint,
    execute_outer_checkpoint,
    make_seed_mean,
    select_config,
)
from split_safe import OuterFoldContract  # noqa: E402


FEATURES = ("bit_a", "bit_b", "continuous", "continuous__missing")
PASSTHROUGH = ("bit_a", "bit_b", "continuous__missing")


def synthetic_inputs():
    ids = [f"R{index:02d}" for index in range(1, 20)]
    feature = pd.DataFrame(
        {
            "curated_id": ids,
            "bit_a": [index % 2 for index in range(19)],
            "bit_b": [int(index % 3 == 0) for index in range(19)],
            "continuous": np.arange(19, dtype=float),
            "continuous__missing": 0,
            "permeability": np.linspace(-6.0, -4.0, 19),
        }
    )
    metadata = pd.DataFrame(
        {
            "curated_id": ids,
            "molecule_id": [f"M{index:02d}" for index in range(1, 20)],
            "source": [f"S{index % 5}" for index in range(19)],
            "analogue_component_id": [f"A{index:02d}" for index in range(1, 20)],
            "sealed_block_id": [f"B{index:02d}" for index in range(1, 20)],
        }
    )
    outer_train = set(ids[:-1])
    inner = {record_id: index % 4 + 1 for index, record_id in enumerate(ids[:-1])}
    contract = OuterFoldContract(1, outer_train, {ids[-1]}, inner)
    return feature, metadata.set_index("curated_id"), feature.set_index("curated_id")["permeability"], contract


@unittest.skipUnless(
    sklearn.__version__ == EXPECTED_SKLEARN,
    f"requires pinned scikit-learn {EXPECTED_SKLEARN}",
)
class R1B1RandomForestIntegrationTests(unittest.TestCase):
    def test_exact_grid_compute_rank_and_selection_order(self):
        self.assertEqual(len(RF_GRID), 12)
        self.assertEqual(len({config.config_id for config in RF_GRID}), 12)
        self.assertEqual(set(COMPUTE_RANK.values()), set(range(1, 13)))
        self.assertEqual(
            min(RF_GRID, key=lambda config: COMPUTE_RANK[config.config_id]).config_id,
            "rf_mf_sqrt_leaf_5_depth_20",
        )
        rows = [
            {
                "config_id": config.config_id,
                "source_macro_mae": 1.0,
                "row_micro_mae": 1.0,
                "compute_rank": COMPUTE_RANK[config.config_id],
            }
            for config in RF_GRID
        ]
        self.assertEqual(
            select_config(rows)["config_id"], "rf_mf_sqrt_leaf_5_depth_20"
        )
        rows[-1]["row_micro_mae"] = 0.9
        self.assertEqual(select_config(rows)["config_id"], rows[-1]["config_id"])
        rows[0]["source_macro_mae"] = 0.99
        self.assertEqual(select_config(rows)["config_id"], rows[0]["config_id"])

    def test_inner_checkpoint_is_guarded_atomic_and_resumable(self):
        feature, metadata, observed, contract = synthetic_inputs()
        config = RFConfig("sqrt", 5, 20)
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            first = execute_inner_checkpoint(
                feature,
                metadata,
                observed,
                contract,
                config,
                1,
                work,
                feature_columns=FEATURES,
                passthrough_columns=PASSTHROUGH,
            )
            second = execute_inner_checkpoint(
                feature,
                metadata,
                observed,
                contract,
                config,
                1,
                work,
                feature_columns=FEATURES,
                passthrough_columns=PASSTHROUGH,
            )
            self.assertEqual(first, second)
            fresh = execute_inner_checkpoint(
                feature,
                metadata,
                observed,
                contract,
                config,
                1,
                work / "fresh_repeat",
                feature_columns=FEATURES,
                passthrough_columns=PASSTHROUGH,
            )
            self.assertEqual(first["prediction_sha256"], fresh["prediction_sha256"])
            self.assertEqual(first["predictions"], fresh["predictions"])
            self.assertEqual(first["audit"], fresh["audit"])
            checkpoints = list(work.rglob("*.json"))
            self.assertEqual(len(checkpoints), 2)
            self.assertEqual(len(first["predictions"]), 5)
            self.assertEqual(N_TREES, first["n_estimators"])
            self.assertEqual(
                [record["operation"] for record in first["audit"]],
                [
                    "mixed_preprocessor.fit",
                    "inner_estimator.fit",
                    "guarded_inner_evaluation",
                    "guarded_prediction",
                ],
            )
            checkpoint = next(
                path for path in checkpoints if "fresh_repeat" not in path.parts
            )
            wrapper = json.loads(checkpoint.read_text(encoding="utf-8"))
            wrapper["payload_sha256"] = "0" * 64
            checkpoint.write_text(json.dumps(wrapper), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                execute_inner_checkpoint(
                    feature,
                    metadata,
                    observed,
                    contract,
                    config,
                    1,
                    work,
                    feature_columns=FEATURES,
                    passthrough_columns=PASSTHROUGH,
                )

    def test_five_outer_seeds_and_explicit_seed_mean(self):
        feature, metadata, observed, contract = synthetic_inputs()
        config = RFConfig("sqrt", 5, 20)
        frames = []
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            for seed in OUTER_SEEDS:
                payload = execute_outer_checkpoint(
                    feature,
                    metadata,
                    observed,
                    contract,
                    config,
                    seed,
                    work,
                    feature_columns=FEATURES,
                    passthrough_columns=PASSTHROUGH,
                )
                self.assertEqual(len(payload["audit"]), 3)
                frame = pd.DataFrame(payload["predictions"])
                frame["outer_fold"] = 1
                frame["config_id"] = config.config_id
                frame["seed"] = seed
                frames.append(frame)
            self.assertEqual(len(list((work / "checkpoints").rglob("*.json"))), 5)
        per_seed = pd.concat(frames, ignore_index=True)
        mean = make_seed_mean(per_seed, expected_n=1)
        self.assertEqual(len(mean), 1)
        self.assertAlmostEqual(
            float(mean.loc[0, "prediction"]), float(per_seed["prediction"].mean())
        )

    def test_runner_has_no_direct_fit_call(self):
        tree = ast.parse(
            (ROOT / "src" / "r1b1_random_forest.py").read_text(encoding="utf-8")
        )
        direct_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "fit"
        ]
        self.assertEqual(direct_calls, [])

    def test_committed_output_integrity_and_coverage(self):
        output = ROOT / "artifacts" / "r1b1_random_forest"
        checksums = {}
        for line in (output / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            digest, name = line.split("  ", 1)
            checksums[name] = digest
        self.assertEqual(len(checksums), 11)
        for name, expected in checksums.items():
            actual = hashlib.sha256((output / name).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, name)

        provenance = json.loads(
            (output / "feature_provenance.json").read_text(encoding="utf-8")
        )
        r1a_provenance = json.loads(
            (ROOT / "artifacts" / "r1a_classical" / "feature_provenance.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(provenance["n_rows"], 6895)
        self.assertEqual(provenance["n_features"], 2075)
        self.assertEqual(provenance["feature_columns"], r1a_provenance["feature_columns"])
        self.assertEqual(
            provenance["feature_matrix_sha256"],
            "074927cb1ec357dea46d7a8431025c413b1f7e2565929fce68b1a7087c28134d",
        )
        self.assertEqual(
            provenance["feature_matrix_sha256"], r1a_provenance["feature_matrix_sha256"]
        )
        forbidden = {"permeability", "source", "outer_fold", "inner_basket"}
        self.assertTrue(forbidden.isdisjoint(provenance["feature_columns"]))

        per_seed = pd.read_csv(output / "oof_predictions_per_seed.csv")
        seed_mean = pd.read_csv(output / "oof_predictions_seed_mean.csv")
        self.assertEqual(len(per_seed), 5 * 6895)
        self.assertEqual(sorted(per_seed["seed"].unique().tolist()), list(OUTER_SEEDS))
        self.assertTrue(
            (per_seed.groupby("seed")["curated_id"].nunique() == 6895).all()
        )
        self.assertTrue(
            (per_seed.groupby("seed")["outer_fold"].nunique() == 18).all()
        )
        self.assertEqual(len(seed_mean), 6895)
        self.assertEqual(seed_mean["curated_id"].nunique(), 6895)
        recomputed_mean = per_seed.groupby("curated_id", sort=False)["prediction"].mean()
        reported_mean = seed_mean.set_index("curated_id")["prediction"]
        np.testing.assert_allclose(
            reported_mean.loc[recomputed_mean.index].to_numpy(),
            recomputed_mean.to_numpy(),
            rtol=0.0,
            atol=5e-15,
        )

        selection = pd.read_csv(output / "inner_selection.csv")
        self.assertEqual(len(selection), 18 * 12)
        self.assertTrue((selection.groupby("outer_fold").size() == 12).all())
        self.assertTrue((selection.groupby("outer_fold")["selected"].sum() == 1).all())
        for _, rows in selection.groupby("outer_fold"):
            expected = select_config(rows.to_dict("records"))["config_id"]
            actual = rows.loc[rows["selected"], "config_id"].item()
            self.assertEqual(actual, expected)

        audit = json.loads((output / "fit_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(len(audit), 3726)
        operations = pd.Series([row["operation"] for row in audit]).value_counts()
        self.assertEqual(int(operations["mixed_preprocessor.fit"]), 954)
        self.assertEqual(int(operations["guarded_prediction"]), 954)
        self.assertEqual(int(operations["inner_estimator.fit"]), 864)
        self.assertEqual(int(operations["guarded_inner_evaluation"]), 864)
        self.assertEqual(int(operations["outer_estimator.fit"]), 90)

        manifest = json.loads(
            (output / "checkpoint_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(manifest), 954)
        self.assertEqual(
            sum(row["operation"] == "inner_selection" for row in manifest), 864
        )
        self.assertEqual(sum(row["operation"] == "outer_refit" for row in manifest), 90)
        self.assertTrue(all(len(row["sha256"]) == 64 for row in manifest))
        self.assertTrue(all(len(row["payload_sha256"]) == 64 for row in manifest))


if __name__ == "__main__":
    unittest.main()
