"""R1b1 checkpointed Random Forest joint-block LOBO runner.

Only the frozen Random Forest stage is implemented here.  Every preprocessing
fit, estimator fit, inner evaluation and outer prediction crosses split_safe.py.
Checkpoint files contain predictions and normalized audit evidence, never a
serialized estimator, and are committed atomically only after an operation is
complete.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import threading
import time
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import psutil
from rdkit import rdBase
import sklearn
from sklearn.ensemble import RandomForestRegressor

from r1a_classical import (
    CONTINUOUS_DESCRIPTORS,
    FEATURE_COLUMNS,
    ID_COLUMN,
    PASSTHROUGH_COLUMNS,
    TARGET_COLUMN,
    _new_mixed_preprocessor,
    _prediction_frame,
    build_feature_frame,
    canonical_json_hash,
    sha256_file,
    source_macro_mae,
)
from split_safe import (
    FitAuditTrail,
    OuterFoldContract,
    SplitSafeFitExecutor,
    canonical_id_hash,
    contracts_from_manifests,
)


N_TREES = 500
SELECTION_SEED = 0
OUTER_SEEDS = (0, 1, 2, 3, 4)
FIT_N_JOBS = -1
PREDICT_N_JOBS = 1
EXPECTED_SKLEARN = "1.2.2"
EXPECTED_FEATURE_HASH = "074927cb1ec357dea46d7a8431025c413b1f7e2565929fce68b1a7087c28134d"
ZERO_HASH = "0" * 64
CHECKPOINT_SCHEMA = "scaffoldseal-r1b1-rf-checkpoint-v2-serial-prediction"
OUTPUT_SCHEMA = "scaffoldseal-r1b1-rf-v1"


@dataclass(frozen=True)
class RFConfig:
    max_features: str | float
    min_samples_leaf: int
    max_depth: int | None

    @property
    def config_id(self) -> str:
        feature = "sqrt" if self.max_features == "sqrt" else "0.25"
        depth = "none" if self.max_depth is None else str(self.max_depth)
        return f"rf_mf_{feature}_leaf_{self.min_samples_leaf}_depth_{depth}"

    def payload(self) -> dict[str, object]:
        return {
            "max_features": self.max_features,
            "min_samples_leaf": self.min_samples_leaf,
            "max_depth": self.max_depth,
        }


RF_GRID = tuple(
    RFConfig(max_features, min_samples_leaf, max_depth)
    for max_features in ("sqrt", 0.25)
    for min_samples_leaf in (1, 3, 5)
    for max_depth in (None, 20)
)


def frozen_compute_key(config: RFConfig) -> tuple[int, int, int, str]:
    """Frozen proxy: fewer candidate features, bounded depth, larger leaves."""
    return (
        0 if config.max_features == "sqrt" else 1,
        0 if config.max_depth == 20 else 1,
        -int(config.min_samples_leaf),
        config.config_id,
    )


COMPUTE_RANK = {
    config.config_id: rank
    for rank, config in enumerate(sorted(RF_GRID, key=frozen_compute_key), start=1)
}


class RandomForestFactory:
    def __init__(self, config: RFConfig, seed: int) -> None:
        self.max_features = config.max_features
        self.min_samples_leaf = int(config.min_samples_leaf)
        self.max_depth = config.max_depth
        self.seed = int(seed)
        self.model_config_sha256 = canonical_json_hash(
            {
                "model": "sklearn.ensemble.RandomForestRegressor",
                "n_estimators": N_TREES,
                "max_features": self.max_features,
                "min_samples_leaf": self.min_samples_leaf,
                "max_depth": self.max_depth,
                "random_state": self.seed,
                "fit_n_jobs": FIT_N_JOBS,
                "prediction_n_jobs": PREDICT_N_JOBS,
                "criterion": "squared_error",
                "bootstrap": True,
            }
        )

    def __call__(self) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=N_TREES,
            max_features=self.max_features,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            random_state=self.seed,
            n_jobs=FIT_N_JOBS,
            criterion="squared_error",
            bootstrap=True,
        )


class PeakRSSMonitor:
    def __init__(self) -> None:
        self.process = psutil.Process()
        self.start_bytes = self._rss()
        self.peak_bytes = self.start_bytes
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def _rss(self) -> int:
        total = self.process.memory_info().rss
        for child in self.process.children(recursive=True):
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return int(total)

    def _poll(self) -> None:
        while not self._stop.wait(0.05):
            self.peak_bytes = max(self.peak_bytes, self._rss())

    def __enter__(self) -> "PeakRSSMonitor":
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.peak_bytes = max(self.peak_bytes, self._rss())
        self._stop.set()
        self._thread.join(timeout=1.0)


def _expected_resolved_max_features(value: str | float, n_features: int) -> int:
    if value == "sqrt":
        return max(1, int(math.sqrt(n_features)))
    return max(1, int(float(value) * n_features))


def verify_fitted_rf_semantics(model: RandomForestRegressor, config: RFConfig, seed: int) -> int:
    params = model.get_params(deep=True)
    expected = {
        "n_estimators": N_TREES,
        "max_features": config.max_features,
        "min_samples_leaf": config.min_samples_leaf,
        "max_depth": config.max_depth,
        "random_state": seed,
        "n_jobs": FIT_N_JOBS,
        "criterion": "squared_error",
        "bootstrap": True,
    }
    if any(params.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Fitted Random Forest parameters differ from the frozen configuration")
    resolved = _expected_resolved_max_features(config.max_features, int(model.n_features_in_))
    if len(model.estimators_) != N_TREES or any(
        int(tree.max_features_) != resolved for tree in model.estimators_
    ):
        raise RuntimeError("scikit-learn max_features semantics differ from the frozen expectation")
    return resolved


def select_config(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    expected = {config.config_id for config in RF_GRID}
    observed = {str(row["config_id"]) for row in rows}
    if len(rows) != 12 or observed != expected:
        raise ValueError("RF selection requires each of the exact 12 frozen configurations once")
    return min(
        rows,
        key=lambda row: (
            float(row["source_macro_mae"]),
            float(row["row_micro_mae"]),
            int(row["compute_rank"]),
            str(row["config_id"]),
        ),
    )


def _logical_namespace(
    outer_fold: int, config_id: str, seed: int, basket: int | None
) -> str:
    scope = "outer_refit" if basket is None else f"inner_{basket:02d}"
    return f"outer_{outer_fold:02d}/{scope}/{config_id}/seed_{seed}"


def _normalize_audit(
    records: Sequence[dict[str, object]], logical_namespace: str
) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for record in records:
        item = dict(record)
        item.pop("feature_columns", None)
        item.pop("execution_identity_sha256", None)
        if "checkpoint_dir" in item:
            item.pop("checkpoint_dir")
            item["checkpoint_namespace"] = logical_namespace
        if "run_namespace_sha256" in item:
            item.pop("run_namespace_sha256")
            item["checkpoint_namespace"] = logical_namespace
        normalized.append(item)
    return normalized


def _checkpoint_path(
    work_dir: Path, outer_fold: int, config_id: str, seed: int, basket: int | None
) -> Path:
    if basket is None:
        return work_dir / "checkpoints" / f"outer_{outer_fold:02d}" / "outer_refit" / f"{config_id}__seed_{seed}.json"
    return work_dir / "checkpoints" / f"outer_{outer_fold:02d}" / f"inner_{basket:02d}" / f"{config_id}__seed_{seed}.json"


def _canonical_payload_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def write_checkpoint_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = {"payload": payload, "payload_sha256": _canonical_payload_hash(payload)}
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(wrapper, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def read_checkpoint(path: Path, expected: dict[str, object]) -> dict[str, object] | None:
    if not path.exists():
        return None
    wrapper = json.loads(path.read_text(encoding="utf-8"))
    if set(wrapper) != {"payload", "payload_sha256"}:
        raise RuntimeError(f"Malformed RF checkpoint: {path}")
    payload = wrapper["payload"]
    if wrapper["payload_sha256"] != _canonical_payload_hash(payload):
        raise RuntimeError(f"RF checkpoint hash mismatch: {path}")
    if any(payload.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"RF checkpoint identity mismatch: {path}")
    if payload.get("status") != "COMPLETE":
        raise RuntimeError(f"RF checkpoint is not complete: {path}")
    return payload


def _attempt_root(work_dir: Path, logical_namespace: str) -> Path:
    safe = logical_namespace.replace("/", "__")
    return work_dir / "run_namespaces" / safe / f"attempt_{os.getpid()}_{time.time_ns()}"


def _base_checkpoint_identity(
    *,
    operation: str,
    outer_fold: int,
    config: RFConfig,
    seed: int,
    basket: int | None,
    fit_ids: Iterable[str],
    prediction_ids: Iterable[str],
) -> dict[str, object]:
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "status": "COMPLETE",
        "operation": operation,
        "feature_matrix_sha256": EXPECTED_FEATURE_HASH,
        "outer_fold": int(outer_fold),
        "inner_basket": basket,
        "config_id": config.config_id,
        "config": config.payload(),
        "seed": int(seed),
        "fit_ids_sha256": canonical_id_hash(fit_ids),
        "prediction_ids_sha256": canonical_id_hash(prediction_ids),
        "n_estimators": N_TREES,
        "fit_n_jobs": FIT_N_JOBS,
        "prediction_n_jobs": PREDICT_N_JOBS,
        "sklearn_version": sklearn.__version__,
    }


def execute_inner_checkpoint(
    feature_frame: pd.DataFrame,
    metadata_by_id: pd.DataFrame,
    observed_by_id: pd.Series,
    contract: OuterFoldContract,
    config: RFConfig,
    basket: int,
    work_dir: Path,
    *,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
    passthrough_columns: Sequence[str] = PASSTHROUGH_COLUMNS,
) -> dict[str, object]:
    train = contract.inner_training_batch(feature_frame, basket)
    validation = contract.inner_validation_batch(feature_frame, basket)
    expected = _base_checkpoint_identity(
        operation="inner_selection",
        outer_fold=contract.outer_fold,
        config=config,
        seed=SELECTION_SEED,
        basket=basket,
        fit_ids=train.ids,
        prediction_ids=validation.ids,
    )
    path = _checkpoint_path(
        work_dir, contract.outer_fold, config.config_id, SELECTION_SEED, basket
    )
    existing = read_checkpoint(path, expected)
    if existing is not None:
        return existing
    audit = FitAuditTrail()
    executor = SplitSafeFitExecutor(contract, audit)
    preprocessor = _new_mixed_preprocessor(contract, audit, passthrough_columns)
    logical = _logical_namespace(contract.outer_fold, config.config_id, SELECTION_SEED, basket)
    run_root = _attempt_root(work_dir, logical)
    started = time.perf_counter()
    with PeakRSSMonitor() as memory:
        executor.fit_preprocessor(
            preprocessor, train, feature_columns, target_column=TARGET_COLUMN
        )
        factory = RandomForestFactory(config, SELECTION_SEED)
        run = contract.mint_run_context(
            run_root,
            config_id=config.config_id,
            seed=SELECTION_SEED,
            inner_basket=basket,
        )
        recorder = contract.create_inner_evaluation_recorder(
            train,
            validation,
            basket=basket,
            feature_columns=feature_columns,
            target_column=TARGET_COLUMN,
            metric_identity="mean_absolute_error",
            run_context=run,
            transform_sha256=preprocessor.transform_sha256_,
            model_config_sha256=factory.model_config_sha256,
            checkpoint_sha256=ZERO_HASH,
            audit=audit,
        )
        model = executor.fit_inner_estimator(
            factory,
            train,
            validation,
            basket=basket,
            feature_columns=feature_columns,
            target_column=TARGET_COLUMN,
            run_context=run,
            recorder=recorder,
            preprocessor=preprocessor,
        )
        resolved = verify_fitted_rf_semantics(model, config, SELECTION_SEED)
        model.n_jobs = PREDICT_N_JOBS
        _, result = recorder.evaluate_estimator_predictions(1, model)
        recorder.finalize()
    elapsed = time.perf_counter() - started
    prediction_frame = _prediction_frame(result, metadata_by_id, observed_by_id)
    payload = {
        **expected,
        "predictions": prediction_frame.to_dict(orient="records"),
        "prediction_sha256": result.prediction_sha256,
        "model_config_sha256": factory.model_config_sha256,
        "preprocessor_definition_sha256": preprocessor.transform_sha256_,
        "fitted_preprocessor_sha256": preprocessor.statistics_sha256_,
        "resolved_max_features": resolved,
        "runtime_seconds": elapsed,
        "rss_start_bytes": memory.start_bytes,
        "rss_peak_bytes": memory.peak_bytes,
        "audit": _normalize_audit(audit.records, logical),
    }
    del model
    gc.collect()
    write_checkpoint_atomic(path, payload)
    return payload


def execute_outer_checkpoint(
    feature_frame: pd.DataFrame,
    metadata_by_id: pd.DataFrame,
    observed_by_id: pd.Series,
    contract: OuterFoldContract,
    config: RFConfig,
    seed: int,
    work_dir: Path,
    *,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
    passthrough_columns: Sequence[str] = PASSTHROUGH_COLUMNS,
) -> dict[str, object]:
    outer_train = contract.outer_training_batch(feature_frame)
    outer_test = contract.outer_test_batch(feature_frame)
    expected = _base_checkpoint_identity(
        operation="outer_refit",
        outer_fold=contract.outer_fold,
        config=config,
        seed=seed,
        basket=None,
        fit_ids=outer_train.ids,
        prediction_ids=outer_test.ids,
    )
    path = _checkpoint_path(work_dir, contract.outer_fold, config.config_id, seed, None)
    existing = read_checkpoint(path, expected)
    if existing is not None:
        return existing
    audit = FitAuditTrail()
    executor = SplitSafeFitExecutor(contract, audit)
    preprocessor = _new_mixed_preprocessor(contract, audit, passthrough_columns)
    logical = _logical_namespace(contract.outer_fold, config.config_id, seed, None)
    run_root = _attempt_root(work_dir, logical)
    started = time.perf_counter()
    with PeakRSSMonitor() as memory:
        executor.fit_preprocessor(
            preprocessor, outer_train, feature_columns, target_column=TARGET_COLUMN
        )
        factory = RandomForestFactory(config, seed)
        run = contract.mint_run_context(
            run_root, config_id=config.config_id, seed=seed, inner_basket=None
        )
        model = executor.fit_outer_estimator(
            factory,
            outer_train,
            feature_columns,
            TARGET_COLUMN,
            run_context=run,
            fixed_iterations=N_TREES,
            preprocessor=preprocessor,
        )
        resolved = verify_fitted_rf_semantics(model, config, seed)
        model.n_jobs = PREDICT_N_JOBS
        result = executor.predict_outer_estimator(
            model,
            outer_test,
            feature_columns=feature_columns,
            target_column=TARGET_COLUMN,
            run_context=run,
            preprocessor=preprocessor,
        )
    elapsed = time.perf_counter() - started
    prediction_frame = _prediction_frame(result, metadata_by_id, observed_by_id)
    payload = {
        **expected,
        "predictions": prediction_frame.to_dict(orient="records"),
        "prediction_sha256": result.prediction_sha256,
        "model_config_sha256": factory.model_config_sha256,
        "preprocessor_definition_sha256": preprocessor.transform_sha256_,
        "fitted_preprocessor_sha256": preprocessor.statistics_sha256_,
        "resolved_max_features": resolved,
        "runtime_seconds": elapsed,
        "rss_start_bytes": memory.start_bytes,
        "rss_peak_bytes": memory.peak_bytes,
        "audit": _normalize_audit(audit.records, logical),
    }
    del model
    gc.collect()
    write_checkpoint_atomic(path, payload)
    return payload


def inner_selection_for_fold(
    feature_frame: pd.DataFrame,
    metadata_by_id: pd.DataFrame,
    observed_by_id: pd.Series,
    contract: OuterFoldContract,
    work_dir: Path,
) -> tuple[list[dict[str, object]], RFConfig, list[dict[str, object]]]:
    predictions: dict[str, list[pd.DataFrame]] = {
        config.config_id: [] for config in RF_GRID
    }
    audit: list[dict[str, object]] = []
    for basket in range(1, 5):
        for config in RF_GRID:
            payload = execute_inner_checkpoint(
                feature_frame,
                metadata_by_id,
                observed_by_id,
                contract,
                config,
                basket,
                work_dir,
            )
            predictions[config.config_id].append(pd.DataFrame(payload["predictions"]))
            audit.extend(payload["audit"])
    rows: list[dict[str, object]] = []
    for config in RF_GRID:
        frame = pd.concat(predictions[config.config_id], ignore_index=True)
        if (
            len(frame) != len(contract.outer_train_ids)
            or set(frame[ID_COLUMN].astype(str)) != set(contract.outer_train_ids)
            or frame[ID_COLUMN].astype(str).duplicated().any()
        ):
            raise RuntimeError("RF inner predictions do not cover exact outer training IDs")
        rows.append(
            {
                "outer_fold": contract.outer_fold,
                "config_id": config.config_id,
                **config.payload(),
                "source_macro_mae": source_macro_mae(frame),
                "row_micro_mae": float(
                    np.mean(np.abs(frame["prediction"] - frame["observed"]))
                ),
                "compute_rank": COMPUTE_RANK[config.config_id],
            }
        )
    selected_row = select_config(rows)
    for row in rows:
        row["selected"] = row["config_id"] == selected_row["config_id"]
    selected = next(config for config in RF_GRID if config.config_id == selected_row["config_id"])
    return rows, selected, audit


def run_fold(
    feature_frame: pd.DataFrame,
    metadata_by_id: pd.DataFrame,
    observed_by_id: pd.Series,
    contract: OuterFoldContract,
    work_dir: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    selection_rows, selected, audit = inner_selection_for_fold(
        feature_frame, metadata_by_id, observed_by_id, contract, work_dir
    )
    outer_payloads = []
    for seed in OUTER_SEEDS:
        payload = execute_outer_checkpoint(
            feature_frame,
            metadata_by_id,
            observed_by_id,
            contract,
            selected,
            seed,
            work_dir,
        )
        outer_payloads.append(payload)
        audit.extend(payload["audit"])
    return {
        "outer_fold": contract.outer_fold,
        "selected_config_id": selected.config_id,
        "selection_rows": selection_rows,
        "outer_payloads": outer_payloads,
        "audit": audit,
        "wall_seconds_this_invocation": time.perf_counter() - started,
    }


def _prediction_metrics(frame: pd.DataFrame) -> dict[str, float | int | list[float]]:
    work = frame.assign(error=frame["prediction"] - frame["observed"])
    work["absolute_error"] = work["error"].abs()
    work["squared_error"] = work["error"] ** 2
    source_mae = work.groupby("source", sort=True)["absolute_error"].mean()
    source_rmse = np.sqrt(work.groupby("source", sort=True)["squared_error"].mean())
    block_mae = work.groupby("sealed_block_id", sort=True)["absolute_error"].mean()
    return {
        "n": len(work),
        "source_macro_mae": float(source_mae.mean()),
        "source_macro_rmse": float(source_rmse.mean()),
        "row_micro_mae": float(work["absolute_error"].mean()),
        "row_micro_rmse": float(np.sqrt(work["squared_error"].mean())),
        "block_median_mae": float(block_mae.median()),
        "block_mae_iqr": [
            float(block_mae.quantile(0.25)),
            float(block_mae.quantile(0.75)),
        ],
    }


def make_seed_mean(per_seed: pd.DataFrame, *, expected_n: int = 6895) -> pd.DataFrame:
    key = [
        ID_COLUMN,
        "molecule_id",
        "source",
        "analogue_component_id",
        "sealed_block_id",
        "outer_fold",
        "config_id",
        "observed",
    ]
    counts = per_seed.groupby(ID_COLUMN, sort=False)["seed"].nunique()
    if len(counts) != expected_n or not (counts == len(OUTER_SEEDS)).all():
        raise RuntimeError("Seed-mean prediction requires all five seeds for every record")
    mean = per_seed.groupby(key, sort=False, as_index=False)["prediction"].mean()
    mean["model"] = "random_forest_seed_mean"
    return mean.loc[:, [*key[:-2], "model", "config_id", "observed", "prediction"]]


def _per_group_metrics(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    work = frame.assign(error=frame["prediction"] - frame["observed"])
    work["absolute_error"] = work["error"].abs()
    work["squared_error"] = work["error"] ** 2
    grouped = (
        work.groupby(["prediction_type", "seed", group_column], sort=True, dropna=False)
        .agg(n=(ID_COLUMN, "size"), mae=("absolute_error", "mean"), mse=("squared_error", "mean"))
        .reset_index()
    )
    grouped["rmse"] = np.sqrt(grouped.pop("mse"))
    return grouped


def validate_audit(audit: Sequence[dict[str, object]], contracts: dict[int, OuterFoldContract]) -> None:
    for record in audit:
        if not str(record.get("operation", "")).endswith(".fit"):
            continue
        contract = contracts[int(record["outer_fold"])]
        basket = record.get("inner_basket")
        expected = (
            canonical_id_hash(contract.outer_train_ids)
            if basket is None
            else canonical_id_hash(contract.expected_inner_ids(int(basket))[0])
        )
        if record.get("fit_ids_sha256") != expected:
            raise RuntimeError("RF fit audit differs from exact authorized training IDs")
        if expected == canonical_id_hash(contract.outer_test_ids):
            raise RuntimeError("Outer-test IDs reached an RF fit")


def checkpoint_manifest(work_dir: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted((work_dir / "checkpoints").rglob("*.json")):
        wrapper = json.loads(path.read_text(encoding="utf-8"))
        payload = wrapper["payload"]
        rows.append(
            {
                "path": path.relative_to(work_dir).as_posix(),
                "sha256": sha256_file(path),
                "payload_sha256": wrapper["payload_sha256"],
                "operation": payload["operation"],
                "outer_fold": payload["outer_fold"],
                "inner_basket": payload["inner_basket"],
                "config_id": payload["config_id"],
                "seed": payload["seed"],
                "runtime_seconds": payload["runtime_seconds"],
                "rss_peak_bytes": payload["rss_peak_bytes"],
            }
        )
    return rows


def write_outputs(
    output_dir: Path,
    feature_provenance: dict[str, object],
    contracts: dict[int, OuterFoldContract],
    fold_results: Sequence[dict[str, object]],
    work_dir: Path,
    command: str,
    total_runtime_seconds: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    selection = pd.DataFrame(
        [row for fold in fold_results for row in fold["selection_rows"]]
    ).sort_values(["outer_fold", "config_id"], kind="stable")
    if len(selection) != 216 or selection.groupby("outer_fold")["selected"].sum().ne(1).any():
        raise RuntimeError("RF selection table lacks exact 18-fold by 12-config coverage")
    per_seed_rows: list[pd.DataFrame] = []
    audit = []
    for fold in fold_results:
        audit.extend(fold["audit"])
        for payload in fold["outer_payloads"]:
            frame = pd.DataFrame(payload["predictions"])
            frame["outer_fold"] = int(fold["outer_fold"])
            frame["model"] = "random_forest"
            frame["config_id"] = payload["config_id"]
            frame["seed"] = int(payload["seed"])
            per_seed_rows.append(frame)
    per_seed = pd.concat(per_seed_rows, ignore_index=True)
    columns = [
        ID_COLUMN,
        "molecule_id",
        "source",
        "analogue_component_id",
        "sealed_block_id",
        "outer_fold",
        "model",
        "config_id",
        "seed",
        "observed",
        "prediction",
    ]
    per_seed = per_seed.loc[:, columns].sort_values(["seed", ID_COLUMN], kind="stable")
    for seed, group in per_seed.groupby("seed"):
        if len(group) != 6895 or group[ID_COLUMN].nunique() != 6895:
            raise RuntimeError(f"RF seed {seed} lacks exact OOF coverage")
    seed_mean = make_seed_mean(per_seed).sort_values(ID_COLUMN, kind="stable")
    validate_audit(audit, contracts)
    seed_metrics = []
    for seed, group in per_seed.groupby("seed", sort=True):
        seed_metrics.append({"seed": int(seed), **_prediction_metrics(group)})
    metric_names = [
        "source_macro_mae",
        "source_macro_rmse",
        "row_micro_mae",
        "row_micro_rmse",
        "block_median_mae",
    ]
    metric_summary = {
        name: {
            "mean_across_seed_metrics": float(np.mean([row[name] for row in seed_metrics])),
            "sample_sd_across_seed_metrics": float(np.std([row[name] for row in seed_metrics], ddof=1)),
        }
        for name in metric_names
    }
    seed_mean_metrics = _prediction_metrics(seed_mean)
    combined_groups = pd.concat(
        [
            per_seed.assign(prediction_type="per_seed"),
            seed_mean.assign(seed=np.nan, prediction_type="five_seed_mean"),
        ],
        ignore_index=True,
    )
    per_source = _per_group_metrics(combined_groups, "source")
    per_block = _per_group_metrics(combined_groups, "sealed_block_id")
    manifest = checkpoint_manifest(work_dir)
    if len(manifest) != 954:
        raise RuntimeError("Expected 864 inner and 90 outer complete RF checkpoints")
    semantics = {
        "sklearn_version": sklearn.__version__,
        "fit_n_jobs": FIT_N_JOBS,
        "prediction_n_jobs": PREDICT_N_JOBS,
        "max_features": {
            "sqrt": "max(1, int(sqrt(n_features_in_)))",
            "0.25": "max(1, int(0.25 * n_features_in_))",
        },
        "verified_on_every_fitted_tree": True,
    }
    accepted_r1a = json.loads(
        (Path(__file__).resolve().parents[1] / "artifacts" / "r1a_classical" / "metrics_summary.json").read_text(encoding="utf-8")
    )
    summary = {
        "status": "PROVISIONAL_PENDING_INDEPENDENT_VERIFICATION",
        "per_seed_metrics": seed_metrics,
        "seed_metric_mean_and_sd": metric_summary,
        "metrics_of_explicit_five_seed_mean_prediction": seed_mean_metrics,
        "accepted_r1a_descriptive_comparators": accepted_r1a["models"],
        "interpretation_scope": "descriptive_only_no_H2_verdict",
    }
    per_seed.to_csv(output_dir / "oof_predictions_per_seed.csv", index=False, lineterminator="\n")
    seed_mean.to_csv(output_dir / "oof_predictions_seed_mean.csv", index=False, lineterminator="\n")
    selection.to_csv(output_dir / "inner_selection.csv", index=False, lineterminator="\n")
    per_source.to_csv(output_dir / "per_source_metrics.csv", index=False, lineterminator="\n")
    per_block.to_csv(output_dir / "per_block_metrics.csv", index=False, lineterminator="\n")
    for name, payload in (
        ("metrics_summary.json", summary),
        ("fit_audit.json", audit),
        ("checkpoint_manifest.json", manifest),
        ("sklearn_semantics.json", semantics),
        (
            "feature_provenance.json",
            {
                **feature_provenance,
                "schema_version": OUTPUT_SCHEMA,
                "r1a_feature_hash_required": EXPECTED_FEATURE_HASH,
                "r1a_feature_provenance_sha256": sha256_file(
                    Path(__file__).resolve().parents[1] / "artifacts" / "r1a_classical" / "feature_provenance.json"
                ),
            },
        ),
        (
            "run_metadata.json",
            {
                "status": "PROVISIONAL_PENDING_INDEPENDENT_VERIFICATION",
                "command": command,
                "runtime_seconds": total_runtime_seconds,
                "grid": [config.payload() for config in RF_GRID],
                "n_estimators": N_TREES,
                "selection_seed": SELECTION_SEED,
                "outer_seeds": list(OUTER_SEEDS),
                "fit_n_jobs": FIT_N_JOBS,
                "prediction_n_jobs": PREDICT_N_JOBS,
                "selection_order": [
                    "concatenated_inner_source_macro_mae",
                    "row_micro_mae",
                    "frozen_lower_compute_rank",
                    "lexical_config_id",
                ],
                "environment": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                    "scikit_learn": sklearn.__version__,
                    "rdkit": rdBase.rdkitVersion,
                    "logical_cpus": psutil.cpu_count(logical=True),
                    "physical_memory_bytes": psutil.virtual_memory().total,
                },
                "checkpoint_resume_policy": "atomic complete operation payloads only; partial attempts never loaded",
                "scientific_models_run": ["random_forest"],
                "not_run": ["xgboost", "dmpnn"],
            },
        ),
    ):
        (output_dir / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    checksums = {
        path.name: sha256_file(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "SHA256SUMS"
    }
    (output_dir / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items())),
        encoding="utf-8",
        newline="\n",
    )


def load_inputs(root: Path):
    if sklearn.__version__ != EXPECTED_SKLEARN:
        raise RuntimeError(
            f"R1b1 production requires scikit-learn {EXPECTED_SKLEARN}, observed {sklearn.__version__}"
        )
    analysis_path = root / "artifacts" / "v2_r0" / "analysis_all_labels.csv"
    feature_frame, metadata, provenance = build_feature_frame(
        analysis_path, root / "artifacts" / "curated_records_public.csv"
    )
    if provenance["feature_matrix_sha256"] != EXPECTED_FEATURE_HASH:
        raise RuntimeError("RF feature matrix differs from accepted R1a")
    accepted = json.loads(
        (root / "artifacts" / "r1a_classical" / "feature_provenance.json").read_text(encoding="utf-8")
    )
    for key in ("feature_matrix_sha256", "feature_columns", "passthrough_columns", "continuous_columns", "definitions"):
        if provenance[key] != accepted[key]:
            raise RuntimeError(f"RF feature provenance differs from accepted R1a: {key}")
    records = pd.read_csv(analysis_path, usecols=[ID_COLUMN, "sealed_block_id"])
    contracts = contracts_from_manifests(
        records,
        pd.read_csv(root / "artifacts" / "v2_r0" / "outer_record_assignments.csv"),
        pd.read_csv(root / "artifacts" / "v2_r0" / "inner_basket_manifest.csv"),
    )
    return feature_frame, metadata, provenance, contracts


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--pilot-inner", action="store_true")
    parser.add_argument("--outer-fold", type=int, action="append")
    parser.add_argument("--fold-only", action="store_true")
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    feature_frame, metadata, provenance, contracts = load_inputs(root)
    metadata_by_id = metadata.set_index(ID_COLUMN)
    observed_by_id = feature_frame.set_index(ID_COLUMN)[TARGET_COLUMN]
    if args.pilot_inner:
        # Deliberately high-cost grid member gives a conservative one-fit estimate.
        config = RFConfig(0.25, 1, None)
        payload = execute_inner_checkpoint(
            feature_frame,
            metadata_by_id,
            observed_by_id,
            contracts[1],
            config,
            1,
            args.work_dir,
        )
        print(
            json.dumps(
                {
                    "pilot": "outer_01_inner_01_high_cost_config",
                    "config_id": config.config_id,
                    "operation_runtime_seconds": payload["runtime_seconds"],
                    "rss_peak_bytes": payload["rss_peak_bytes"],
                    "total_invocation_seconds": time.perf_counter() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return
    selected_folds = sorted(set(args.outer_fold or contracts.keys()))
    fold_results = []
    for outer_fold in selected_folds:
        result = run_fold(
            feature_frame,
            metadata_by_id,
            observed_by_id,
            contracts[outer_fold],
            args.work_dir,
        )
        fold_results.append(result)
        print(
            json.dumps(
                {
                    "outer_fold": int(outer_fold),
                    "selected_config_id": result["selected_config_id"],
                    "wall_seconds_this_invocation": result["wall_seconds_this_invocation"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if args.fold_only:
        return
    if selected_folds != list(range(1, 19)) or args.output_dir is None:
        raise RuntimeError("Final output assembly requires all 18 folds and --output-dir")
    command = (
        f"python src/r1b1_random_forest.py --work-dir {args.work_dir.as_posix()} "
        f"--output-dir {args.output_dir.as_posix()}"
    )
    write_outputs(
        args.output_dir,
        provenance,
        contracts,
        fold_results,
        args.work_dir,
        command,
        time.perf_counter() - started,
    )


if __name__ == "__main__":
    main()
