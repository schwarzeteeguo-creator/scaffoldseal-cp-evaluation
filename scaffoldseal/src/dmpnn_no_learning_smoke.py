"""Pinned BenchmarkCycPeptMP/DeepChem DMPNN no-learning boundary smoke."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd

from split_safe import (
    FitAuditTrail,
    FramePredictionOutput,
    FreshTrainingState,
    OuterFoldContract,
    SplitSafeFitExecutor,
    canonical_id_hash,
)


PINNED_COMMIT = "d82aa3c5c9c849dbd584e8669132ed3d33e50a27"
PINNED_VERSIONS = {
    "python": "3.10.18",
    "deepchem": "2.7.1",
    "torch": "2.0.1+cu118",
    "rdkit": "2022.09.5",
    "numpy": "1.23.5",
    "pandas": "1.5.3",
}
SOURCE_MODULES = (
    "DeepChemModels/ModelFeatureGenerator.py",
    "DeepChemModels/CustomizedDateLoader.py",
    "DeepChemModels/ModelTrainer.py",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class NoLearningDMPNNAdapter:
    """Real pinned construction/loader/trainer path with fit replaced pre-invocation."""

    def __init__(self, baseline_root: Path, source_hashes: dict[str, str]) -> None:
        self.baseline_root = str(baseline_root.resolve())
        self.transform_sha256 = canonical_sha256(
            {key: source_hashes[key] for key in SOURCE_MODULES[:2]}
        )
        self.fitted_transform_sha256 = self.transform_sha256
        self.model_config_sha256 = canonical_sha256(
            {
                "model": "DMPNN",
                "mode": "regression",
                "n_tasks": 1,
                "batch_size": 2,
                "generator_sha256": source_hashes[SOURCE_MODULES[0]],
            }
        )
        self.create_calls = 0
        self.inner_calls = 0
        self.outer_fit_calls = 0
        self.outer_predict_calls = 0
        self.noop_fit_calls = 0
        self.noop_outer_fit_calls = 0
        self.noop_predict_calls = 0
        self.noop_evaluate_calls = 0
        self.noop_save_calls = 0
        self.noop_restore_calls = 0
        self.optimizer_update_calls = 0
        self.observed_train_ids: tuple[str, ...] = ()
        self.observed_validation_ids: tuple[str, ...] = ()
        self.observed_train_columns: tuple[str, ...] = ()
        self.observed_validation_columns: tuple[str, ...] = ()
        self.loaded_train_ids: tuple[str, ...] = ()
        self.loaded_validation_ids: tuple[str, ...] = ()
        self.observed_outer_train_ids: tuple[str, ...] = ()
        self.observed_outer_train_columns: tuple[str, ...] = ()
        self.observed_outer_predict_ids: tuple[str, ...] = ()
        self.observed_outer_predict_columns: tuple[str, ...] = ()
        self.observed_outer_predict_values: tuple[str, ...] = ()
        self.loaded_outer_predict_ids: tuple[str, ...] = ()

    def create_fresh_state(self, *, run_context):
        from ModelFeatureGenerator import generate_model_feature
        import torch

        self.create_calls += 1
        random.seed(run_context.seed)
        np.random.seed(run_context.seed)
        torch.manual_seed(run_context.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_context.seed)
        args = SimpleNamespace(
            model_dir=run_context.checkpoint_dir,
            split="scaffoldseal_no_learning_smoke",
            mode="regression",
            batch_size=2,
        )
        _, model = generate_model_feature("DMPNN", 1, args)
        return FreshTrainingState(model=model, optimizer=object(), scheduler=None)

    def fit_inner(
        self,
        *,
        state,
        train_frame,
        validation_frame,
        target_column,
        feature_columns,
        sample_weight,
        maximum_epochs,
        options,
        run_context,
        recorder,
    ):
        import deepchem as dc
        from CustomizedDateLoader import data_loader_separate
        from ModelTrainer import model_trainer

        if maximum_epochs != 1 or sample_weight is not None:
            raise AssertionError("Smoke must be one no-learning epoch without side payloads")
        self.inner_calls += 1
        self.observed_train_ids = tuple(train_frame["curated_id"].astype(str))
        self.observed_validation_ids = tuple(validation_frame["curated_id"].astype(str))
        self.observed_train_columns = tuple(map(str, train_frame.columns))
        self.observed_validation_columns = tuple(map(str, validation_frame.columns))
        checkpoint_dir = Path(run_context.checkpoint_dir)
        train_csv = checkpoint_dir / "safe_inner_train.csv"
        validation_csv = checkpoint_dir / "safe_inner_validation.csv"
        train_frame.to_csv(train_csv, index=False)
        validation_frame.to_csv(validation_csv, index=False)
        loader = dc.data.CSVLoader(
            tasks=[target_column],
            feature_field=feature_columns[0],
            id_field="curated_id",
            featurizer=dc.feat.DMPNNFeaturizer(),
        )
        datasets = data_loader_separate(
            [[str(train_csv)], [str(validation_csv)], [str(validation_csv)]], loader
        )
        self.loaded_train_ids = tuple(map(str, datasets["train"].ids))
        self.loaded_validation_ids = tuple(map(str, datasets["valid"].ids))

        def noop_fit(dataset, nb_epoch=1, checkpoint_interval=0):
            if dataset is not datasets["train"] or nb_epoch != 1 or checkpoint_interval != 0:
                raise AssertionError("Pinned trainer fit received an unexpected dataset/options")
            self.noop_fit_calls += 1
            return 0.0

        def noop_evaluate(dataset, metrics, transformers):
            if dataset is not datasets["valid"] or transformers:
                raise AssertionError("Pinned trainer evaluation received an unexpected payload")
            self.noop_evaluate_calls += 1
            return {"mae_score": 0.0}

        def noop_save_checkpoint(*, model_dir, max_checkpoints_to_keep):
            if Path(model_dir).resolve() != checkpoint_dir.resolve() or max_checkpoints_to_keep != 1:
                raise AssertionError("Pinned trainer requested an unexpected checkpoint namespace")
            self.noop_save_calls += 1

        def noop_restore(*args, **kwargs):
            self.noop_restore_calls += 1

        # Replacement happens before the real orchestration entry point is invoked.
        state.model.fit = noop_fit
        state.model.evaluate = noop_evaluate
        state.model.save_checkpoint = noop_save_checkpoint
        state.model.restore = noop_restore
        metric = dc.metrics.Metric(dc.metrics.mean_absolute_error, name="mae_score")
        trainer_args = SimpleNamespace(mode="regression", n_epoch=1, patience=0)
        model_trainer(
            state.model,
            str(checkpoint_dir),
            datasets["train"],
            datasets["valid"],
            [metric],
            "mae_score",
            [],
            "no-learning-smoke",
            trainer_args,
            n_epoch=1,
            patience=0,
        )
        recorder.evaluate_frame_epoch(1, self, state)

    def fit_outer(
        self,
        *,
        state,
        train_frame,
        target_column,
        feature_columns,
        sample_weight,
        fixed_epoch,
        options,
        run_context,
    ):
        import deepchem as dc

        if fixed_epoch != 1 or sample_weight is not None:
            raise AssertionError("Smoke outer fit must be one no-learning epoch")
        self.outer_fit_calls += 1
        self.observed_outer_train_ids = tuple(train_frame["curated_id"].astype(str))
        self.observed_outer_train_columns = tuple(map(str, train_frame.columns))
        checkpoint_dir = Path(run_context.checkpoint_dir)
        train_csv = checkpoint_dir / "safe_outer_train.csv"
        train_frame.to_csv(train_csv, index=False)
        loader = dc.data.CSVLoader(
            tasks=[target_column],
            feature_field=feature_columns[0],
            id_field="curated_id",
            featurizer=dc.feat.DMPNNFeaturizer(),
        )
        train_dataset = loader.create_dataset(str(train_csv))
        if tuple(map(str, train_dataset.ids)) != self.observed_outer_train_ids:
            raise AssertionError("Outer loader changed exact training ID order")

        def noop_outer_fit(dataset, nb_epoch=1, checkpoint_interval=0):
            if dataset is not train_dataset or nb_epoch != 1 or checkpoint_interval != 0:
                raise AssertionError("Outer no-op fit received unexpected data/options")
            self.noop_outer_fit_calls += 1
            return 0.0

        def noop_predict(dataset):
            return np.zeros((len(dataset.ids), 1), dtype=float)

        state.model.fit = noop_outer_fit
        state.model.predict = noop_predict
        for _ in range(fixed_epoch):
            state.model.fit(train_dataset, nb_epoch=1, checkpoint_interval=0)

    def predict_validation(self, *, state, validation_frame, feature_columns, run_context):
        if tuple(validation_frame.columns) != ("curated_id", *feature_columns):
            raise AssertionError("Recorder exposed columns outside ID plus declared features")
        return np.zeros(len(validation_frame), dtype=float)

    def predict_outer(
        self,
        *,
        state,
        outer_test_frame,
        feature_columns,
        run_context,
    ):
        import deepchem as dc

        expected_columns = ("curated_id", *feature_columns)
        if tuple(outer_test_frame.columns) != expected_columns:
            raise AssertionError("Outer predictor received target or undeclared metadata")
        observed_outer_predict_ids = tuple(outer_test_frame["curated_id"].astype(str))
        observed_outer_predict_columns = tuple(map(str, outer_test_frame.columns))
        observed_outer_predict_values = tuple(
            outer_test_frame[feature_columns[0]].astype(str)
        )
        if observed_outer_predict_ids != ("SEALED_OUTER",):
            raise AssertionError("Outer predictor received the wrong sealed test ID")
        if observed_outer_predict_values != ("N#N",):
            raise AssertionError("Outer predictor did not receive the sealed SMILES value")
        if run_context.seed != 0:
            raise AssertionError("Outer predictor received a changed run seed")
        # Prediction featurization is ephemeral and cannot alter the sealed fit
        # checkpoint namespace whose content hash is revalidated after prediction.
        with tempfile.TemporaryDirectory(prefix="scaffoldseal-sealed-predict-") as temporary:
            prediction_csv = Path(temporary) / "safe_outer_predict.csv"
            outer_test_frame.to_csv(prediction_csv, index=False)
            loader = dc.data.CSVLoader(
                tasks=[],
                feature_field=feature_columns[0],
                id_field="curated_id",
                featurizer=dc.feat.DMPNNFeaturizer(),
            )
            dataset = loader.create_dataset(str(prediction_csv))
            loaded_outer_predict_ids = tuple(map(str, dataset.ids))
            if loaded_outer_predict_ids != observed_outer_predict_ids:
                raise AssertionError("Outer prediction loader changed exact ID order")
            predictions = np.asarray(state.model.predict(dataset), dtype=float).reshape(-1)
        return FramePredictionOutput(
            pd.Series(
                predictions,
                index=pd.Index(loaded_outer_predict_ids, name="curated_id"),
                name="prediction",
            ),
            {
                "outer_predict_calls": 1,
                "noop_predict_calls": 1,
            },
        )


def run_smoke(baseline_root: Path) -> dict[str, object]:
    baseline_root = baseline_root.resolve()
    observed_commit = subprocess.run(
        ["git", "-C", str(baseline_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed_commit != PINNED_COMMIT:
        raise RuntimeError(f"Baseline commit mismatch: {observed_commit}")
    module_status = subprocess.run(
        ["git", "-C", str(baseline_root), "status", "--porcelain", "--", *SOURCE_MODULES],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if module_status:
        raise RuntimeError(f"Pinned source modules are locally modified: {module_status}")
    source_hashes = {relative: file_sha256(baseline_root / relative) for relative in SOURCE_MODULES}
    deepchem_modules = baseline_root / "DeepChemModels"
    sys.path.insert(0, str(deepchem_modules))

    import deepchem
    import numpy
    import pandas
    import rdkit
    import torch

    observed_versions = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "deepchem": str(deepchem.__version__),
        "torch": str(torch.__version__),
        "rdkit": str(rdkit.__version__),
        "numpy": str(numpy.__version__),
        "pandas": str(pandas.__version__),
    }
    if observed_versions != PINNED_VERSIONS:
        raise RuntimeError(f"Pinned environment mismatch: {observed_versions}")

    frame = pd.DataFrame(
        {
            "curated_id": ["T1", "T2", "T3", "T4", "SEALED_OUTER"],
            "SMILES": ["CCO", "CCN", "CCC", "CCCl", "N#N"],
            "y_safe": [0.1, 0.2, 0.3, 0.4, 99.0],
            "forbidden_metadata": ["a", "b", "c", "d", "sealed"],
        }
    )
    contract = OuterFoldContract(
        1,
        {"T1", "T2", "T3", "T4"},
        {"SEALED_OUTER"},
        {"T1": 1, "T2": 2, "T3": 3, "T4": 4},
    )
    audit = FitAuditTrail()
    executor = SplitSafeFitExecutor(contract, audit)
    adapter = NoLearningDMPNNAdapter(baseline_root, source_hashes)
    with tempfile.TemporaryDirectory(prefix="scaffoldseal-dmpnn-no-learning-") as temporary:
        root = Path(temporary)
        run = contract.mint_run_context(
            root, config_id="pinned_dmpnn_no_learning", seed=0, inner_basket=1
        )
        comparison_run = contract.mint_run_context(
            root, config_id="pinned_dmpnn_no_learning", seed=0, inner_basket=2
        )
        outer_run = contract.mint_run_context(
            root, config_id="pinned_dmpnn_no_learning", seed=0, inner_basket=None
        )
        train = contract.inner_training_batch(frame, 1)
        validation = contract.inner_validation_batch(frame, 1)
        recorder = contract.create_inner_evaluation_recorder(
            train,
            validation,
            basket=1,
            feature_columns=["SMILES"],
            target_column="y_safe",
            metric_identity="mean_absolute_error",
            run_context=run,
            transform_sha256=adapter.transform_sha256,
            model_config_sha256=adapter.model_config_sha256,
            checkpoint_sha256="0" * 64,
            audit=audit,
        )
        executor.fit_inner_frame(
            adapter,
            train,
            validation,
            basket=1,
            feature_columns=["SMILES"],
            target_column="y_safe",
            run_context=run,
            recorder=recorder,
            maximum_epochs=1,
        )
        history = recorder.finalize()
        outer_handle = executor.fit_outer_frame(
            adapter,
            contract.outer_frame_training_batch(frame),
            feature_columns=["SMILES"],
            target_column="y_safe",
            run_context=outer_run,
            fixed_epoch=1,
        )
        frame.loc[frame["curated_id"] == "SEALED_OUTER", "SMILES"] = "C"
        outer_result = executor.predict_outer_frame(outer_handle)
        outer_telemetry = dict(outer_result.adapter_telemetry)
        outer_fit_audit = dict(audit.records[-1]["outer_fit_identity"])
        outer_fit_audit["run_namespace_sha256"] = "<runtime-bound-namespace>"
        outer_fit_audit["run_context_sha256"] = "<runtime-bound-run-context>"
        outer_fit_binding_projection_sha256 = canonical_sha256(outer_fit_audit)
        namespaces = [
            str(Path(run.checkpoint_dir).relative_to(root)).replace("\\", "/"),
            str(Path(comparison_run.checkpoint_dir).relative_to(root)).replace("\\", "/"),
            str(Path(outer_run.checkpoint_dir).relative_to(root)).replace("\\", "/"),
        ]

    expected_columns = ("curated_id", "SMILES", "y_safe")
    expected_outer_predict_columns = ("curated_id", "SMILES")
    expected_outer_predict_ids = ("SEALED_OUTER",)
    expected_outer_predict_values = ("N#N",)
    assertions = {
        "actual_dmpnn_constructed": adapter.create_calls == 2,
        "actual_csv_loader_used": set(adapter.loaded_train_ids) == set(train.ids)
        and set(adapter.loaded_validation_ids) == set(validation.ids),
        "actual_model_trainer_entered": adapter.noop_fit_calls == 1
        and adapter.noop_evaluate_calls == 1,
        "fit_replaced_before_invocation": adapter.noop_fit_calls == 1,
        "optimizer_update_calls_zero": adapter.optimizer_update_calls == 0,
        "projected_columns_only": adapter.observed_train_columns == expected_columns
        and adapter.observed_validation_columns == expected_columns,
        "outer_ids_never_exposed_to_fit_or_validation":
        "SEALED_OUTER" not in adapter.observed_train_ids
        and "SEALED_OUTER" not in adapter.observed_validation_ids
        and "SEALED_OUTER" not in adapter.observed_outer_train_ids,
        "no_arbitrary_callback_or_data_argument": adapter.inner_calls == 1,
        "outer_prediction_accepts_only_sealed_handle": tuple(
            inspect.signature(executor.predict_outer_frame).parameters
        ) == ("handle",),
        "caller_mutation_cannot_change_outer_values": outer_result.predictions == (0.0,),
        "distinct_checkpoint_namespaces": len(set(namespaces)) == 3,
        "one_guarded_history_event": len(history.events) == 1,
        "guarded_outer_fit_entered": adapter.outer_fit_calls == 1
        and adapter.noop_outer_fit_calls == 1,
        "guarded_outer_prediction_entered_once": adapter.outer_predict_calls == 0
        and adapter.noop_predict_calls == 0
        and outer_telemetry["outer_predict_calls"] == 1
        and outer_telemetry["noop_predict_calls"] == 1,
        "sealed_adapter_state_is_telemetry_free": adapter.observed_outer_predict_ids == ()
        and adapter.observed_outer_predict_columns == ()
        and adapter.observed_outer_predict_values == ()
        and adapter.loaded_outer_predict_ids == (),
        "disposable_immutable_run_context_seed_preserved": outer_result.predictions
        == (0.0,),
        "guarded_outer_prediction_exact_ids": outer_result.ids == expected_outer_predict_ids,
        "guarded_outer_prediction_outcome_free": outer_result.predictions == (0.0,),
        "guarded_outer_prediction_hashed_audited": len(outer_result.prediction_sha256) == 64
        and len(outer_result.outer_fit_identity_sha256) == 64
        and audit.records[-1]["operation"] == "guarded_outer_frame_prediction",
        "raw_outer_adapter_prediction_not_called_outside_typed_path":
        outer_telemetry["outer_predict_calls"] == 1,
    }
    if not all(assertions.values()):
        raise AssertionError(assertions)
    return {
        "schema_version": "scaffoldseal-dmpnn-no-learning-smoke-v4-adapter-run-sealed",
        "status": "PASS",
        "r1_authorized": False,
        "real_model_fit_or_weight_update_performed": False,
        "baseline": {
            "repository": "baseline_candidates/BenchmarkCycPeptMP",
            "commit": observed_commit,
            "pinned_source_modules_clean": True,
            "source_module_sha256": source_hashes,
        },
        "environment": observed_versions,
        "entry_points": {
            "model_and_featurizer": "ModelFeatureGenerator.generate_model_feature('DMPNN', 1, args)",
            "loader": "CustomizedDateLoader.data_loader_separate + deepchem.data.CSVLoader",
            "orchestration": "ModelTrainer.model_trainer",
            "update_replacement": "DMPNNModel.fit replaced with no-op recorder before model_trainer",
            "outer_prediction": "SplitSafeFitExecutor.predict_outer_frame -> sealed adapter.predict_outer -> FramePredictionOutput -> DMPNNModel.predict(no-op)",
        },
        "commands": [
            "baseline_candidates/BenchmarkCycPeptMP/.venv/Scripts/python.exe src/dmpnn_no_learning_smoke.py --baseline-root ../baseline_candidates/BenchmarkCycPeptMP --output audit/dmpnn_no_learning_smoke.json"
        ],
        "safe_boundary": {
            "outer_fold": 1,
            "inner_basket": 1,
            "training_ids_sha256": canonical_id_hash(train.ids),
            "validation_ids_sha256": canonical_id_hash(validation.ids),
            "observed_train_columns": list(adapter.observed_train_columns),
            "observed_validation_columns": list(adapter.observed_validation_columns),
            "observed_train_ids": list(adapter.observed_train_ids),
            "observed_validation_ids": list(adapter.observed_validation_ids),
            "checkpoint_namespaces": namespaces,
            "outer_training_ids_sha256": canonical_id_hash(
                adapter.observed_outer_train_ids
            ),
            "outer_test_ids_sha256": outer_result.ids_sha256,
            "outer_prediction_sha256": outer_result.prediction_sha256,
            "outer_fit_binding_projection_sha256": outer_fit_binding_projection_sha256,
            "observed_outer_train_columns": list(adapter.observed_outer_train_columns),
            "observed_outer_predict_columns": list(expected_outer_predict_columns),
            "observed_outer_predict_ids": list(expected_outer_predict_ids),
            "observed_outer_predict_values": list(expected_outer_predict_values),
        },
        "no_op_recorder": {
            "fit_calls": adapter.noop_fit_calls,
            "evaluate_calls": adapter.noop_evaluate_calls,
            "save_checkpoint_calls": adapter.noop_save_calls,
            "restore_calls": adapter.noop_restore_calls,
            "optimizer_update_calls": adapter.optimizer_update_calls,
            "outer_fit_calls": adapter.noop_outer_fit_calls,
            "outer_predict_calls": outer_telemetry["noop_predict_calls"],
        },
        "assertions": assertions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run_smoke(args.baseline_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
