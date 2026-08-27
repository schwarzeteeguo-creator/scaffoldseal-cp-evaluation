"""Bounded, preregistered D0 runtime/memory pilot through split_safe.py.

This module deliberately runs only outer fold 1, seed 0.  It creates the four
authenticated inner histories required for the fixed outer epoch and performs
one handle-only outer prediction.  It cannot dispatch the full LOBO stage.  A
repository-fixed one-shot ledger retires this already executed scientific
identity and prevents relaunch under a different work/output namespace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import pandas as pd

from d0_pilot_governance import (
    DispatchAlreadyRecorded,
    SingleDispatchLedger,
    corrected_joint_lobo_projection,
    frozen_pilot_identity,
    project_joint_lobo_hours_exact,
    write_lossless_prediction_artifacts,
)

from split_safe import (
    FitAuditTrail,
    FramePredictionOutput,
    FreshTrainingState,
    SplitSafeFitExecutor,
    canonical_array_hash,
    canonical_id_hash,
    contract_manifest,
    contracts_from_manifests,
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
    "DeepChemModels/DeepChemModelsMain.py",
    "Utils.py",
)
MODEL_CONFIG = {
    "model": "DMPNN",
    "mode": "regression",
    "n_tasks": 1,
    "batch_size": 64,
    "maximum_epochs": 2000,
    "patience": 200,
    "learning_rate": 0.001,
    "seed_schedule": "123 * seed_index ** 2",
    "seed_index": 0,
    "effective_seed": 0,
    "target_transform": "(log10_papp + 6) / 2",
}
RTOL = 1e-6
ATOL = 1e-7
MAX_GPU_HOURS = 72.0
MAX_GPU_BYTES = 7 * 1024**3
FROZEN_PROTOCOL_SHA256 = "b5fc086cb2c6fb81e9cd009fbaed24cdb62405b22192899255ff82e762c91680"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DISPATCH_LEDGER_ROOT = (
    PROJECT_ROOT / "artifacts" / "r1c0_dmpnn_attempt_archive_v1" / "dispatch_ledger"
)
FULL_LOBO_DISPATCH_AUTHORIZED = False


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def reset_rng(seed_index: int) -> int:
    seed = 123 * int(seed_index) ** 2
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return seed


def global_step(model: object) -> int:
    value = getattr(model, "_global_step", 0)
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def gpu_snapshot() -> dict[str, object]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; the locked GPU pilot cannot run")
    return {
        "device_name": torch.cuda.get_device_name(0),
        "device_total_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
    }


def reset_gpu_peak() -> None:
    import torch

    torch.cuda.init()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)


def verify_environment(baseline_root: Path) -> tuple[dict[str, str], dict[str, str]]:
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
        raise RuntimeError(f"Pinned source modules are modified: {module_status}")
    deepchem_modules = baseline_root / "DeepChemModels"
    if str(deepchem_modules) not in sys.path:
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
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return observed_versions, {name: file_sha256(baseline_root / name) for name in SOURCE_MODULES}


def create_model(run_dir: Path):
    from ModelFeatureGenerator import generate_model_feature

    args = SimpleNamespace(
        model_dir=str(run_dir), split="d0_pilot", mode="regression", batch_size=64
    )
    featurizer, model = generate_model_feature("DMPNN", 1, args)
    if float(model.optimizer.learning_rate) != 0.001 or hasattr(model.optimizer, "scheduler"):
        raise RuntimeError("Pinned optimizer/learning-rate identity changed")
    return featurizer, model


def dataset_ids_hash(dataset: object) -> str:
    return canonical_id_hash(tuple(map(str, dataset.ids)))


def train_delaney_once(
    baseline_root: Path,
    run_dir: Path,
    repeat_name: str,
    *,
    seed_index: int = 0,
) -> dict[str, object]:
    import deepchem as dc

    if run_dir.exists():
        raise RuntimeError(f"Pretraining namespace already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    # The bounded pilot uses the default seed index 0.  The separately locked
    # full runner supplies 0..4 and reuses each resulting independent Delaney
    # checkpoint only within that same seed.
    reset_rng(seed_index)
    reset_gpu_peak()
    started = time.perf_counter()
    featurizer, model = create_model(run_dir)
    raw_path = baseline_root / "CSV" / "PreTrainData" / "delaney-processed.csv"
    if not raw_path.is_file():
        raise RuntimeError("Local Delaney CSV is missing")
    loader = dc.data.CSVLoader(
        tasks=["measured log solubility in mols per litre"],
        feature_field="smiles",
        featurizer=featurizer,
    )
    raw = loader.create_dataset(str(raw_path), data_dir=str(run_dir / "raw_dataset"))
    train, valid, test = dc.splits.RandomSplitter().train_valid_test_split(
        raw,
        seed=None,
        train_dir=str(run_dir / "train_untransformed"),
        valid_dir=str(run_dir / "valid_untransformed"),
        test_dir=str(run_dir / "test_untransformed"),
    )
    transformer = dc.trans.NormalizationTransformer(transform_y=True, dataset=train)
    train = transformer.transform(train, out_dir=str(run_dir / "train"))
    valid = transformer.transform(valid, out_dir=str(run_dir / "valid"))
    test = transformer.transform(test, out_dir=str(run_dir / "test"))
    metric = dc.metrics.Metric(dc.metrics.score_function.rms_score, name="rms_score")
    history: list[float] = []
    best = float("inf")
    best_epoch = 0
    non_improving = 0
    fit_calls = 0
    start_step = global_step(model)
    for epoch in range(1, MODEL_CONFIG["maximum_epochs"] + 1):
        model.fit(train, nb_epoch=1, checkpoint_interval=0, restore=False)
        fit_calls += 1
        score = float(model.evaluate(valid, [metric], [transformer])["rms_score"])
        history.append(score)
        if score < best:
            best, best_epoch, non_improving = score, epoch, 0
            model.save_checkpoint(max_checkpoints_to_keep=1, model_dir=str(run_dir))
        else:
            non_improving += 1
        if non_improving > MODEL_CONFIG["patience"]:
            break
    end_step = global_step(model)
    checkpoint = run_dir / "checkpoint1.pt"
    if not checkpoint.is_file() or best_epoch < 1:
        raise RuntimeError("Delaney pretraining did not produce a best checkpoint")
    model.restore(str(checkpoint))
    probe = np.asarray(model.predict(test, transformers=[transformer]), dtype=float).reshape(-1)
    elapsed = time.perf_counter() - started
    payload = {
        "repeat": repeat_name,
        "raw_csv_sha256": file_sha256(raw_path),
        "split_ids": {
            "train": dataset_ids_hash(train),
            "validation": dataset_ids_hash(valid),
            "test": dataset_ids_hash(test),
        },
        "split_counts": {"train": len(train), "validation": len(valid), "test": len(test)},
        "history": history,
        "history_sha256": canonical_array_hash(history),
        "best_epoch": best_epoch,
        "stopping_epoch": len(history),
        "best_validation_rmse": best,
        "fit_calls": fit_calls,
        "optimizer_updates": end_step - start_step,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "probe_ids_sha256": dataset_ids_hash(test),
        "probe_predictions": list(map(float, probe)),
        "probe_prediction_sha256": canonical_array_hash(probe),
        "runtime_seconds": elapsed,
        "gpu": gpu_snapshot(),
    }
    write_json(run_dir / "pretraining_trace.json", payload)
    return payload


def prove_pretraining_equivalence(first: dict[str, object], second: dict[str, object]) -> dict[str, object]:
    exact = {
        "split_ids": first["split_ids"] == second["split_ids"],
        "split_counts": first["split_counts"] == second["split_counts"],
        "best_epoch": first["best_epoch"] == second["best_epoch"],
        "stopping_epoch": first["stopping_epoch"] == second["stopping_epoch"],
        "probe_ids": first["probe_ids_sha256"] == second["probe_ids_sha256"],
    }
    first_history = np.asarray(first["history"], dtype=float)
    second_history = np.asarray(second["history"], dtype=float)
    first_probe = np.asarray(first["probe_predictions"], dtype=float)
    second_probe = np.asarray(second["probe_predictions"], dtype=float)
    numeric = {
        "history": first_history.shape == second_history.shape
        and bool(np.allclose(first_history, second_history, rtol=RTOL, atol=ATOL)),
        "probe_predictions": first_probe.shape == second_probe.shape
        and bool(np.allclose(first_probe, second_probe, rtol=RTOL, atol=ATOL)),
    }
    result = {"exact": exact, "numeric": numeric, "rtol": RTOL, "atol": ATOL}
    result["pass"] = all(exact.values()) and all(numeric.values())
    return result


class LockedDMPNNAdapter:
    """Stateless adapter around the maintained DeepChem DMPNN implementation."""

    def __init__(self, baseline_root: Path, source_hashes: dict[str, str]) -> None:
        self.baseline_root = str(baseline_root.resolve())
        self.source_hashes = tuple(sorted(source_hashes.items()))
        self.transform_sha256 = canonical_json_sha256(
            {"SMILES": "object", "normalized_pampa": "(log10_papp+6)/2"}
        )
        self.fitted_transform_sha256 = self.transform_sha256
        self.model_config_sha256 = canonical_json_sha256(
            {"config": MODEL_CONFIG, "sources": dict(self.source_hashes)}
        )

    def create_fresh_state(self, *, run_context):
        reset_rng(run_context.seed)
        _, model = create_model(Path(run_context.checkpoint_dir))
        if run_context.pretrained_checkpoint is None:
            raise RuntimeError("D0 requires an explicit frozen Delaney checkpoint")
        model.restore(run_context.pretrained_checkpoint)
        optimizer = getattr(model, "_pytorch_optimizer", None)
        if optimizer is None:
            raise RuntimeError("Pinned checkpoint did not restore a fresh torch optimizer")
        return FreshTrainingState(model=model, optimizer=optimizer, scheduler=None)

    @staticmethod
    def _load_frame(frame, target_column, feature_columns, run_context, role):
        import deepchem as dc

        if tuple(feature_columns) != ("SMILES",):
            raise RuntimeError("D0 pilot accepts only frozen object-string SMILES")
        if str(frame["SMILES"].dtype) != "object":
            raise RuntimeError("D0 pilot SMILES dtype changed")
        path = Path(run_context.checkpoint_dir) / f"{role}.csv"
        data_dir = Path(run_context.checkpoint_dir) / f"{role}_dataset"
        frame.to_csv(path, index=False)
        loader = dc.data.CSVLoader(
            tasks=[target_column],
            feature_field="SMILES",
            id_field="curated_id",
            featurizer=dc.feat.DMPNNFeaturizer(),
        )
        dataset = loader.create_dataset(str(path), data_dir=str(data_dir))
        expected = tuple(frame["curated_id"].astype(str))
        if tuple(map(str, dataset.ids)) != expected:
            raise RuntimeError(f"{role} loader changed exact ID order")
        return dataset

    def fit_inner(
        self, *, state, train_frame, validation_frame, target_column,
        feature_columns, sample_weight, maximum_epochs, options, run_context, recorder
    ):
        if sample_weight is not None or maximum_epochs != 2000:
            raise RuntimeError("D0 pilot inner configuration changed")
        reset_gpu_peak()
        started = time.perf_counter()
        train = self._load_frame(train_frame, target_column, feature_columns, run_context, "train")
        valid = self._load_frame(
            validation_frame, target_column, feature_columns, run_context, "validation"
        )
        state.model._scaffoldseal_validation_dataset = valid
        state.model._scaffoldseal_validation_ids = tuple(map(str, valid.ids))
        epoch_seconds: list[float] = []
        best = float("inf")
        best_epoch = 0
        non_improving = 0
        start_step = global_step(state.model)
        for epoch in range(1, maximum_epochs + 1):
            tick = time.perf_counter()
            state.model.fit(train, nb_epoch=1, checkpoint_interval=0, restore=False)
            event = recorder.evaluate_frame_epoch(epoch, self, state)
            epoch_seconds.append(time.perf_counter() - tick)
            if event.loss < best:
                best, best_epoch, non_improving = event.loss, epoch, 0
                state.model.save_checkpoint(
                    max_checkpoints_to_keep=1, model_dir=run_context.checkpoint_dir
                )
            else:
                non_improving += 1
            if non_improving > 200:
                break
        end_step = global_step(state.model)
        checkpoint = Path(run_context.checkpoint_dir) / "checkpoint1.pt"
        if not checkpoint.is_file():
            raise RuntimeError("Inner fit did not produce a best checkpoint")
        state.model.restore(str(checkpoint))
        del state.model._scaffoldseal_validation_dataset
        del state.model._scaffoldseal_validation_ids
        trace = {
            "role": "inner",
            "outer_fold": run_context.outer_fold,
            "inner_basket": run_context.inner_basket,
            "seed": run_context.seed,
            "n_train": len(train),
            "n_validation": len(valid),
            "epochs_run": len(epoch_seconds),
            "best_epoch": best_epoch,
            "best_validation_mse": best,
            "best_validation_rmse_normalized": math.sqrt(best),
            "fit_calls": len(epoch_seconds),
            "optimizer_updates": end_step - start_step,
            "epoch_seconds": epoch_seconds,
            "runtime_seconds": time.perf_counter() - started,
            "checkpoint_sha256": file_sha256(checkpoint),
            "gpu": gpu_snapshot(),
        }
        write_json(Path(run_context.checkpoint_dir) / "training_trace.json", trace)

    def predict_validation(self, *, state, validation_frame, feature_columns, run_context):
        expected = tuple(validation_frame["curated_id"].astype(str))
        if getattr(state.model, "_scaffoldseal_validation_ids", None) != expected:
            raise RuntimeError("Validation prediction IDs differ from split-local dataset")
        dataset = state.model._scaffoldseal_validation_dataset
        return np.asarray(state.model.predict(dataset), dtype=float).reshape(-1)

    def fit_outer(
        self, *, state, train_frame, target_column, feature_columns, sample_weight,
        fixed_epoch, options, run_context
    ):
        if sample_weight is not None or fixed_epoch < 1:
            raise RuntimeError("D0 pilot outer configuration changed")
        reset_gpu_peak()
        started = time.perf_counter()
        train = self._load_frame(train_frame, target_column, feature_columns, run_context, "train")
        epoch_seconds: list[float] = []
        start_step = global_step(state.model)
        for _ in range(fixed_epoch):
            tick = time.perf_counter()
            state.model.fit(train, nb_epoch=1, checkpoint_interval=0, restore=False)
            epoch_seconds.append(time.perf_counter() - tick)
        end_step = global_step(state.model)
        state.model.save_checkpoint(max_checkpoints_to_keep=1, model_dir=run_context.checkpoint_dir)
        checkpoint = Path(run_context.checkpoint_dir) / "checkpoint1.pt"
        trace = {
            "role": "outer_refit",
            "outer_fold": run_context.outer_fold,
            "seed": run_context.seed,
            "n_train": len(train),
            "fixed_epoch": fixed_epoch,
            "fit_calls": fixed_epoch,
            "optimizer_updates": end_step - start_step,
            "epoch_seconds": epoch_seconds,
            "runtime_seconds": time.perf_counter() - started,
            "checkpoint_sha256": file_sha256(checkpoint),
            "gpu": gpu_snapshot(),
        }
        write_json(Path(run_context.checkpoint_dir) / "training_trace.json", trace)

    def predict_outer(self, *, state, outer_test_frame, feature_columns, run_context):
        import deepchem as dc
        import tempfile

        started = time.perf_counter()
        expected = tuple(outer_test_frame["curated_id"].astype(str))
        with tempfile.TemporaryDirectory(prefix="scaffoldseal-d0-predict-") as temporary:
            path = Path(temporary) / "outer.csv"
            data_dir = Path(temporary) / "dataset"
            outer_test_frame.to_csv(path, index=False)
            loader = dc.data.CSVLoader(
                tasks=[], feature_field="SMILES", id_field="curated_id",
                featurizer=dc.feat.DMPNNFeaturizer()
            )
            dataset = loader.create_dataset(str(path), data_dir=str(data_dir))
            ids = tuple(map(str, dataset.ids))
            if ids != expected:
                raise RuntimeError("Outer prediction loader changed exact ID order")
            values = np.asarray(state.model.predict(dataset), dtype=float).reshape(-1)
        return FramePredictionOutput(
            pd.Series(values, index=pd.Index(ids, name="curated_id"), name="prediction"),
            {"outer_predict_calls": 1, "prediction_runtime_seconds": time.perf_counter() - started},
        )


def load_trace(checkpoint_dir: str) -> dict[str, object]:
    return json.loads((Path(checkpoint_dir) / "training_trace.json").read_text(encoding="utf-8"))


def project_joint_lobo_hours(
    inner_traces: Iterable[dict[str, object]], outer_trace: dict[str, object],
    outer_assignments: pd.DataFrame, pretrain_seconds: float
) -> float:
    """Backward-named wrapper using exact outer-training row counts."""

    return project_joint_lobo_hours_exact(
        inner_traces, outer_trace, outer_assignments, pretrain_seconds
    )


def require_full_lobo_prefit_review() -> None:
    raise RuntimeError(
        "Full D0 dispatch is absent and unauthorized; a separate pre-fit review is required"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("d0_pilot_protocol.json"))
    args = parser.parse_args()
    baseline_root = args.baseline_root.resolve()
    work_dir = args.work_dir.resolve()
    output_dir = args.output_dir.resolve()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["status"] != "FROZEN_BEFORE_FIRST_D0_WEIGHT_UPDATE":
        raise RuntimeError("Pilot protocol is not frozen")
    if file_sha256(args.protocol) != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError("Only the exact frozen D0 pilot protocol is recognized")
    scientific_identity = frozen_pilot_identity(protocol)
    ledger = SingleDispatchLedger(DISPATCH_LEDGER_ROOT)
    try:
        claim = ledger.claim(
            scientific_identity,
            {
                "work_dir": str(work_dir),
                "output_dir": str(output_dir),
                "protocol_path": str(args.protocol.resolve()),
            },
        )
    except DispatchAlreadyRecorded as error:
        raise RuntimeError(
            "The frozen D0 pilot identity has already been attempted and may not be relaunched"
        ) from error
    if work_dir.exists() or output_dir.exists():
        ledger.update_status(claim, "STOPPED_BEFORE_SCIENTIFIC_EXECUTION_NAMESPACE_EXISTS")
        raise RuntimeError("Pilot work/output namespace already exists")
    work_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    versions, source_hashes = verify_environment(baseline_root)

    pretrain_a = train_delaney_once(baseline_root, work_dir / "pretraining_a", "A")
    pretrain_b = train_delaney_once(baseline_root, work_dir / "pretraining_b", "B")
    equivalence = prove_pretraining_equivalence(pretrain_a, pretrain_b)
    write_json(output_dir / "pretraining_equivalence.json", equivalence)
    if not equivalence["pass"]:
        raise RuntimeError("Seed-0 Delaney pretraining equivalence gate failed")
    pretrained = Path(str(pretrain_a["checkpoint"]))
    pretrained_sha256 = str(pretrain_a["checkpoint_sha256"])

    records = pd.read_csv("artifacts/v2_r0/analysis_all_labels.csv")
    outer = pd.read_csv("artifacts/v2_r0/outer_record_assignments.csv")
    inner = pd.read_csv("artifacts/v2_r0/inner_basket_manifest.csv")
    contracts = contracts_from_manifests(records, outer, inner)
    manifest = contract_manifest(contracts)
    maximum = int(manifest["n_fit_ids"].max())
    selected_case = manifest.loc[manifest["n_fit_ids"].eq(maximum)].sort_values(
        ["outer_fold", "inner_basket"], kind="stable"
    ).iloc[0]
    if (int(selected_case.outer_fold), int(selected_case.inner_basket), maximum) != (1, 4, 6370):
        raise RuntimeError("Frozen high-cost pilot case no longer matches the manifests")

    frame = pd.DataFrame(
        {
            "curated_id": records["curated_id"].astype(str),
            "SMILES": records["canonical_smiles"].astype(object),
            "normalized_pampa": (pd.to_numeric(records["permeability"]) + 6.0) / 2.0,
        }
    )
    if str(frame["SMILES"].dtype) != "object" or not np.isfinite(frame["normalized_pampa"]).all():
        raise RuntimeError("Frozen D0 frame dtype/target transform failed")
    contract = contracts[1]
    audit = FitAuditTrail()
    executor = SplitSafeFitExecutor(contract, audit)
    histories = []
    inner_traces = []
    checkpoint_root = work_dir / "pampa"
    for basket in (1, 2, 3, 4):
        adapter = LockedDMPNNAdapter(baseline_root, source_hashes)
        run = contract.mint_run_context(
            checkpoint_root, config_id="d0_locked", seed=0, inner_basket=basket,
            pretrained_checkpoint=pretrained,
            pretrained_checkpoint_sha256=pretrained_sha256,
        )
        train = contract.inner_training_batch(frame, basket)
        validation = contract.inner_validation_batch(frame, basket)
        recorder = contract.create_inner_evaluation_recorder(
            train, validation, basket=basket, feature_columns=["SMILES"],
            target_column="normalized_pampa", metric_identity="mean_squared_error",
            run_context=run, transform_sha256=adapter.transform_sha256,
            model_config_sha256=adapter.model_config_sha256,
            checkpoint_sha256=pretrained_sha256, audit=audit,
        )
        executor.fit_inner_frame(
            adapter, train, validation, basket=basket, feature_columns=["SMILES"],
            target_column="normalized_pampa", run_context=run, recorder=recorder,
            maximum_epochs=2000,
        )
        histories.append(recorder.finalize())
        inner_traces.append(load_trace(run.checkpoint_dir))
    fixed_epoch = contract.select_stopping_epoch(histories, audit)

    outer_adapter = LockedDMPNNAdapter(baseline_root, source_hashes)
    outer_run = contract.mint_run_context(
        checkpoint_root, config_id="d0_locked", seed=0, inner_basket=None,
        pretrained_checkpoint=pretrained,
        pretrained_checkpoint_sha256=pretrained_sha256,
    )
    handle = executor.fit_outer_frame(
        outer_adapter, contract.outer_frame_training_batch(frame),
        feature_columns=["SMILES"], target_column="normalized_pampa",
        run_context=outer_run, fixed_epoch=fixed_epoch,
    )
    prediction = executor.predict_outer_frame(handle)
    outer_trace = load_trace(outer_run.checkpoint_dir)

    observed_ids = list(prediction.ids)
    y_by_id = records.set_index("curated_id")["permeability"]
    normalized_prediction = np.asarray(prediction.predictions, dtype=float)
    log_prediction = normalized_prediction * 2.0 - 6.0
    y = y_by_id.loc[observed_ids].to_numpy(float)
    prediction_table = pd.DataFrame(
        {
            "curated_id": observed_ids,
            "outer_fold": 1,
            "seed": 0,
            "observed_log10_papp": y,
            "prediction_normalized": normalized_prediction,
            "prediction_log10_papp": log_prediction,
        }
    )
    prediction_artifacts = write_lossless_prediction_artifacts(
        prediction_table,
        output_dir / "pilot_outer_predictions.lossless.json",
        output_dir / "pilot_outer_predictions.csv",
    )
    projected_hours = project_joint_lobo_hours(
        inner_traces, outer_trace, outer, float(pretrain_a["runtime_seconds"])
    )
    corrected_projection = corrected_joint_lobo_projection(
        inner_traces, outer_trace, outer, float(pretrain_a["runtime_seconds"])
    )
    peak_bytes = max(
        [int(pretrain_a["gpu"]["max_memory_reserved_bytes"]),
         int(pretrain_b["gpu"]["max_memory_reserved_bytes"])]
        + [int(item["gpu"]["max_memory_reserved_bytes"]) for item in inner_traces]
        + [int(outer_trace["gpu"]["max_memory_reserved_bytes"])]
    )
    status = "PASS"
    blockers = []
    if projected_hours > MAX_GPU_HOURS:
        status = "STOP"
        blockers.append("projected_joint_lobo_gpu_hours_exceeds_72")
    if peak_bytes > MAX_GPU_BYTES:
        status = "STOP"
        blockers.append("peak_process_gpu_reserved_bytes_exceeds_7_gib")
    summary = {
        "schema_version": "scaffoldseal-d0-runtime-pilot-result-v1",
        "status": status,
        "blockers": blockers,
        "baseline_commit": PINNED_COMMIT,
        "environment": versions,
        "source_module_sha256": source_hashes,
        "protocol_sha256": file_sha256(args.protocol),
        "pilot_case": {"outer_fold": 1, "seed": 0, "high_cost_inner_basket": 4},
        "pretraining": {
            "runs": 2,
            "equivalence": equivalence,
            "canonical_checkpoint_sha256": pretrained_sha256,
            "canonical_best_epoch": pretrain_a["best_epoch"],
            "canonical_stopping_epoch": pretrain_a["stopping_epoch"],
        },
        "inner_traces": inner_traces,
        "selected_fixed_epoch": fixed_epoch,
        "outer_trace": outer_trace,
        "outer_prediction": {
            "n": len(prediction.ids),
            "ids_sha256": prediction.ids_sha256,
            "prediction_sha256": prediction.prediction_sha256,
            "source_mae_log10_papp": float(np.mean(np.abs(log_prediction - y))),
            "rmse_log10_papp": float(np.sqrt(np.mean((log_prediction - y) ** 2))),
            "artifacts": prediction_artifacts,
        },
        "fit_counts": {
            "delaney_pretraining": 2,
            "pampa_inner": 4,
            "pampa_outer": 1,
            "pampa_outer_predictions": 1,
            "pampa_inner_epoch_fit_calls": sum(int(x["fit_calls"]) for x in inner_traces),
            "pampa_outer_epoch_fit_calls": int(outer_trace["fit_calls"]),
            "pampa_inner_optimizer_updates": sum(int(x["optimizer_updates"]) for x in inner_traces),
            "pampa_outer_optimizer_updates": int(outer_trace["optimizer_updates"]),
        },
        "resources": {
            "peak_process_gpu_reserved_bytes": peak_bytes,
            "projected_joint_lobo_gpu_hours": projected_hours,
            "limits": {"gpu_hours": MAX_GPU_HOURS, "gpu_bytes": MAX_GPU_BYTES},
            "projection_basis": corrected_projection,
        },
        "audit_records": len(audit.records),
        "audit_sha256": canonical_json_sha256(audit.records),
        "full_lobo_started": False,
    }
    write_json(output_dir / "pilot_summary.json", summary)
    write_json(output_dir / "fit_audit.json", audit.records)
    write_json(
        output_dir / "inner_histories.json",
        [
            {
                "outer_fold": h.outer_fold,
                "inner_basket": h.inner_basket,
                "seed": h.seed,
                "training_ids_sha256": canonical_id_hash(h.training_ids),
                "validation_ids_sha256": canonical_id_hash(h.validation_ids),
                "events": [event._asdict() if hasattr(event, "_asdict") else event.__dict__ for event in h.events],
            }
            for h in histories
        ],
    )
    ledger.update_status(
        claim,
        "COMPLETED_BOUNDED_PILOT",
        {"result_status": status, "output_dir": str(output_dir)},
    )
    print(json.dumps({"status": status, "output": str(output_dir), "projected_gpu_hours": projected_hours}))


if __name__ == "__main__":
    main()
