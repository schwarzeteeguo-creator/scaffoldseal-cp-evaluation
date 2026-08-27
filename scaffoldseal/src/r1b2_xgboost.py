"""R1b2 checkpointed XGBoost joint-block LOBO runner.

Only the frozen XGBoost stage is implemented.  Every preprocessing fit,
estimator fit, inner evaluation and prediction crosses split_safe.py.  Atomic
checkpoints contain predictions and normalized audit evidence, never models.
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
import time
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import psutil
from rdkit import rdBase
import sklearn
import xgboost
from xgboost import XGBRegressor

from r1a_classical import (
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
from r1b1_random_forest import PeakRSSMonitor
from split_safe import (
    FitAuditTrail,
    OuterFoldContract,
    SplitSafeFitExecutor,
    canonical_id_hash,
    contracts_from_manifests,
)


SELECTION_SEED = 0
OUTER_SEEDS = (0, 1, 2, 3, 4)
FIT_N_JOBS = 8
PREDICT_N_JOBS = 1
TREE_METHOD = "hist"
DEVICE = "cpu"
MISSING_SENTINEL = -999999.0
EXPECTED_SKLEARN = "1.2.2"
EXPECTED_XGBOOST = "2.1.1"
EXPECTED_FEATURE_HASH = "074927cb1ec357dea46d7a8431025c413b1f7e2565929fce68b1a7087c28134d"
ZERO_HASH = "0" * 64
CHECKPOINT_SCHEMA = "scaffoldseal-r1b2-xgb-checkpoint-v1-cpu-hist-serial-prediction"
OUTPUT_SCHEMA = "scaffoldseal-r1b2-xgb-v1"


@dataclass(frozen=True)
class XGBConfig:
    n_estimators: int
    max_depth: int
    learning_rate: float
    reg_lambda: int

    @property
    def config_id(self) -> str:
        lr = format(self.learning_rate, ".2f")
        return (
            f"xgb_n_{self.n_estimators}_depth_{self.max_depth}_"
            f"lr_{lr}_lambda_{self.reg_lambda}"
        )

    def payload(self) -> dict[str, object]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "reg_lambda": self.reg_lambda,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }


XGB_GRID = tuple(
    XGBConfig(n_estimators, max_depth, learning_rate, reg_lambda)
    for n_estimators in (300, 800)
    for max_depth in (4, 8)
    for learning_rate in (0.03, 0.10)
    for reg_lambda in (1, 10)
)


def frozen_compute_key(config: XGBConfig) -> tuple[int, int]:
    """Frozen before results: fewer boosting rounds, then shallower trees."""
    return config.n_estimators, config.max_depth


_COMPUTE_TIERS = {
    key: rank
    for rank, key in enumerate(sorted({frozen_compute_key(c) for c in XGB_GRID}), start=1)
}
COMPUTE_RANK = {
    config.config_id: _COMPUTE_TIERS[frozen_compute_key(config)] for config in XGB_GRID
}


class XGBoostFactory:
    def __init__(self, config: XGBConfig, seed: int) -> None:
        self.n_estimators = int(config.n_estimators)
        self.max_depth = int(config.max_depth)
        self.learning_rate = float(config.learning_rate)
        self.reg_lambda = int(config.reg_lambda)
        self.seed = int(seed)
        self.model_config_sha256 = canonical_json_hash(
            {
                "model": "xgboost.XGBRegressor",
                **config.payload(),
                "objective": "reg:squarederror",
                "booster": "gbtree",
                "tree_method": TREE_METHOD,
                "device": DEVICE,
                "random_state": self.seed,
                "fit_n_jobs": FIT_N_JOBS,
                "prediction_n_jobs": PREDICT_N_JOBS,
                "eval_metric": "rmse",
                "validate_parameters": True,
                "verbosity": 0,
                "missing": MISSING_SENTINEL,
                "callbacks": None,
                "early_stopping_rounds": None,
            }
        )

    def __call__(self) -> XGBRegressor:
        return XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            reg_lambda=self.reg_lambda,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            booster="gbtree",
            tree_method=TREE_METHOD,
            device=DEVICE,
            random_state=self.seed,
            n_jobs=FIT_N_JOBS,
            eval_metric="rmse",
            validate_parameters=True,
            verbosity=0,
            missing=MISSING_SENTINEL,
            callbacks=None,
            early_stopping_rounds=None,
        )


def _safe_json_value(value: object) -> object:
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def constructor_parameter_policy() -> dict[str, object]:
    explicit = {
        "n_estimators",
        "max_depth",
        "learning_rate",
        "reg_lambda",
        "subsample",
        "colsample_bytree",
        "objective",
        "booster",
        "tree_method",
        "device",
        "random_state",
        "n_jobs",
        "eval_metric",
        "validate_parameters",
        "verbosity",
        "missing",
        "callbacks",
        "early_stopping_rounds",
    }
    defaults = XGBRegressor().get_params(deep=True)
    return {
        "explicit_grid_parameters": {
            "n_estimators": [300, 800],
            "max_depth": [4, 8],
            "learning_rate": [0.03, 0.10],
            "reg_lambda": [1, 10],
        },
        "explicit_fixed_constructor_parameters": {
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "booster": "gbtree",
            "tree_method": TREE_METHOD,
            "device": DEVICE,
            "random_state": "scheduled_seed",
            "n_jobs": FIT_N_JOBS,
            "eval_metric": "rmse",
            "validate_parameters": True,
            "verbosity": 0,
            "missing": MISSING_SENTINEL,
            "callbacks": None,
            "early_stopping_rounds": None,
        },
        "otherwise_xgboost_2_1_1_defaults": {
            key: _safe_json_value(value)
            for key, value in sorted(defaults.items())
            if key not in explicit
        },
        "fit_call_policy": {
            "eval_set": None,
            "sample_weight": None,
            "base_margin": None,
            "xgb_model": None,
            "sample_weight_eval_set": None,
            "base_margin_eval_set": None,
            "feature_weights": None,
        },
    }


def verify_fitted_xgb_semantics(
    model: XGBRegressor, config: XGBConfig, seed: int
) -> dict[str, object]:
    params = model.get_params(deep=True)
    expected = {
        "n_estimators": config.n_estimators,
        "max_depth": config.max_depth,
        "learning_rate": config.learning_rate,
        "reg_lambda": config.reg_lambda,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "booster": "gbtree",
        "tree_method": TREE_METHOD,
        "device": DEVICE,
        "random_state": seed,
        "n_jobs": FIT_N_JOBS,
        "eval_metric": "rmse",
        "validate_parameters": True,
        "verbosity": 0,
        "missing": MISSING_SENTINEL,
        "callbacks": None,
        "early_stopping_rounds": None,
    }
    if any(params.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Fitted XGBoost parameters differ from the frozen configuration")
    booster = model.get_booster()
    config_json = booster.save_config()
    saved = json.loads(config_json)
    learner = saved["learner"]
    generic = learner["generic_param"]
    gradient = learner["gradient_booster"]
    train = gradient["gbtree_train_param"]
    tree = gradient["tree_train_param"]
    if (
        booster.num_boosted_rounds() != config.n_estimators
        or learner["learner_train_param"]["objective"] != "reg:squarederror"
        or gradient["name"] != "gbtree"
        or train["tree_method"] != TREE_METHOD
        or generic["device"] != DEVICE
        or int(generic["nthread"]) != FIT_N_JOBS
        or int(generic["seed"]) != seed
        or int(tree["max_depth"]) != config.max_depth
        or not math.isclose(float(tree["learning_rate"]), config.learning_rate, rel_tol=1e-6)
        or not math.isclose(float(tree["reg_lambda"]), config.reg_lambda, rel_tol=1e-6)
        or not math.isclose(float(tree["subsample"]), 0.8, rel_tol=1e-6)
        or not math.isclose(float(tree["colsample_bytree"]), 0.8, rel_tol=1e-6)
    ):
        raise RuntimeError("XGBoost booster semantics differ from the frozen expectation")
    return {
        "booster_config_sha256_before_serial_prediction": hashlib.sha256(
            config_json.encode("utf-8")
        ).hexdigest(),
        "num_boosted_rounds": booster.num_boosted_rounds(),
        "tree_method": train["tree_method"],
        "updater": train["updater"],
        "device": generic["device"],
        "fit_nthread": int(generic["nthread"]),
        "base_score": learner["learner_model_param"]["base_score"],
        "boost_from_average": learner["learner_model_param"]["boost_from_average"],
        "num_feature": int(learner["learner_model_param"]["num_feature"]),
    }


def set_and_verify_serial_prediction(model: XGBRegressor) -> None:
    model.set_params(n_jobs=PREDICT_N_JOBS)
    saved = json.loads(model.get_booster().save_config())
    if (
        model.get_params(deep=True).get("n_jobs") != PREDICT_N_JOBS
        or int(saved["learner"]["generic_param"]["nthread"]) != PREDICT_N_JOBS
    ):
        raise RuntimeError("XGBoost serial prediction setting did not reach the Booster")


def select_config(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    expected = {config.config_id for config in XGB_GRID}
    observed = {str(row["config_id"]) for row in rows}
    if len(rows) != 16 or observed != expected:
        raise ValueError("XGBoost selection requires each frozen configuration once")
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
    normalized = []
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
    scope = "outer_refit" if basket is None else f"inner_{basket:02d}"
    return (
        work_dir
        / "checkpoints"
        / f"outer_{outer_fold:02d}"
        / scope
        / f"{config_id}__seed_{seed}.json"
    )


def _canonical_payload_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
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


def read_checkpoint(
    path: Path, expected: dict[str, object]
) -> dict[str, object] | None:
    if not path.exists():
        return None
    wrapper = json.loads(path.read_text(encoding="utf-8"))
    if set(wrapper) != {"payload", "payload_sha256"}:
        raise RuntimeError(f"Malformed XGBoost checkpoint: {path}")
    payload = wrapper["payload"]
    if wrapper["payload_sha256"] != _canonical_payload_hash(payload):
        raise RuntimeError(f"XGBoost checkpoint hash mismatch: {path}")
    if any(payload.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"XGBoost checkpoint identity mismatch: {path}")
    if payload.get("status") != "COMPLETE":
        raise RuntimeError(f"XGBoost checkpoint is not complete: {path}")
    return payload


def _attempt_root(work_dir: Path, logical_namespace: str) -> Path:
    safe = logical_namespace.replace("/", "__")
    return work_dir / "run_namespaces" / safe / f"attempt_{os.getpid()}_{time.time_ns()}"


def _base_checkpoint_identity(
    *,
    operation: str,
    outer_fold: int,
    config: XGBConfig,
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
        "fit_n_jobs": FIT_N_JOBS,
        "prediction_n_jobs": PREDICT_N_JOBS,
        "tree_method": TREE_METHOD,
        "device": DEVICE,
        "xgboost_version": xgboost.__version__,
        "sklearn_version": sklearn.__version__,
    }


def execute_inner_checkpoint(
    feature_frame: pd.DataFrame,
    metadata_by_id: pd.DataFrame,
    observed_by_id: pd.Series,
    contract: OuterFoldContract,
    config: XGBConfig,
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
        factory = XGBoostFactory(config, SELECTION_SEED)
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
            use_internal_eval_set=False,
            preprocessor=preprocessor,
        )
        semantics = verify_fitted_xgb_semantics(model, config, SELECTION_SEED)
        set_and_verify_serial_prediction(model)
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
        "booster_semantics": semantics,
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
    config: XGBConfig,
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
        factory = XGBoostFactory(config, seed)
        run = contract.mint_run_context(
            run_root, config_id=config.config_id, seed=seed, inner_basket=None
        )
        model = executor.fit_outer_estimator(
            factory,
            outer_train,
            feature_columns,
            TARGET_COLUMN,
            run_context=run,
            fixed_iterations=config.n_estimators,
            preprocessor=preprocessor,
        )
        semantics = verify_fitted_xgb_semantics(model, config, seed)
        set_and_verify_serial_prediction(model)
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
        "booster_semantics": semantics,
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
) -> tuple[list[dict[str, object]], XGBConfig, list[dict[str, object]]]:
    predictions = {config.config_id: [] for config in XGB_GRID}
    audit: list[dict[str, object]] = []
    for basket in range(1, 5):
        for config in XGB_GRID:
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
    rows = []
    for config in XGB_GRID:
        frame = pd.concat(predictions[config.config_id], ignore_index=True)
        if (
            len(frame) != len(contract.outer_train_ids)
            or set(frame[ID_COLUMN].astype(str)) != set(contract.outer_train_ids)
            or frame[ID_COLUMN].astype(str).duplicated().any()
        ):
            raise RuntimeError("XGBoost inner predictions lack exact outer-training coverage")
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
    selected = next(c for c in XGB_GRID if c.config_id == selected_row["config_id"])
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


def _prediction_metrics(frame: pd.DataFrame) -> dict[str, object]:
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
        "block_mae_iqr": [float(block_mae.quantile(0.25)), float(block_mae.quantile(0.75))],
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
        raise RuntimeError("Seed mean requires all five XGBoost seeds for every record")
    mean = per_seed.groupby(key, sort=False, as_index=False)["prediction"].mean()
    mean["model"] = "xgboost_seed_mean"
    return mean.loc[:, [*key[:-2], "model", "config_id", "observed", "prediction"]]


def _per_group_metrics(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    work = frame.assign(error=frame["prediction"] - frame["observed"])
    work["absolute_error"] = work["error"].abs()
    work["squared_error"] = work["error"] ** 2
    grouped = (
        work.groupby(["prediction_type", "seed", group_column], sort=True, dropna=False)
        .agg(
            n=(ID_COLUMN, "size"),
            mae=("absolute_error", "mean"),
            mse=("squared_error", "mean"),
        )
        .reset_index()
    )
    grouped["rmse"] = np.sqrt(grouped.pop("mse"))
    return grouped


def validate_audit(
    audit: Sequence[dict[str, object]], contracts: dict[int, OuterFoldContract]
) -> None:
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
            raise RuntimeError("XGBoost fit audit differs from authorized training IDs")
        if expected == canonical_id_hash(contract.outer_test_ids):
            raise RuntimeError("Outer-test IDs reached an XGBoost fit")


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
    full_compute_wall_seconds: float,
    output_assembly_runtime_seconds: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    selection = pd.DataFrame(
        [row for fold in fold_results for row in fold["selection_rows"]]
    ).sort_values(["outer_fold", "config_id"], kind="stable")
    if len(selection) != 288 or selection.groupby("outer_fold")["selected"].sum().ne(1).any():
        raise RuntimeError("XGBoost selection lacks 18-fold by 16-config coverage")
    per_seed_rows = []
    audit = []
    for fold in fold_results:
        audit.extend(fold["audit"])
        for payload in fold["outer_payloads"]:
            frame = pd.DataFrame(payload["predictions"])
            frame["outer_fold"] = int(fold["outer_fold"])
            frame["model"] = "xgboost"
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
            raise RuntimeError(f"XGBoost seed {seed} lacks exact OOF coverage")
    seed_mean = make_seed_mean(per_seed).sort_values(ID_COLUMN, kind="stable")
    validate_audit(audit, contracts)
    seed_metrics = [
        {"seed": int(seed), **_prediction_metrics(group)}
        for seed, group in per_seed.groupby("seed", sort=True)
    ]
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
            "sample_sd_across_seed_metrics": float(
                np.std([row[name] for row in seed_metrics], ddof=1)
            ),
        }
        for name in metric_names
    }
    seed_mean_metrics = _prediction_metrics(seed_mean)
    combined = pd.concat(
        [
            per_seed.assign(prediction_type="per_seed"),
            seed_mean.assign(seed=np.nan, prediction_type="five_seed_mean"),
        ],
        ignore_index=True,
    )
    per_source = _per_group_metrics(combined, "source")
    per_block = _per_group_metrics(combined, "sealed_block_id")
    manifest = checkpoint_manifest(work_dir)
    if len(manifest) != 1242:
        raise RuntimeError("Expected 1,152 inner and 90 outer XGBoost checkpoints")
    accepted_r1a = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "artifacts"
            / "r1a_classical"
            / "metrics_summary.json"
        ).read_text(encoding="utf-8")
    )
    accepted_rf = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "artifacts"
            / "r1b1_random_forest"
            / "metrics_summary.json"
        ).read_text(encoding="utf-8")
    )
    summary = {
        "status": "PROVISIONAL_PENDING_INDEPENDENT_VERIFICATION",
        "per_seed_metrics": seed_metrics,
        "seed_metric_mean_and_sd": metric_summary,
        "metrics_of_explicit_five_seed_mean_prediction": seed_mean_metrics,
        "accepted_r1a_descriptive_comparators": accepted_r1a["models"],
        "accepted_rf_descriptive_comparator": accepted_rf[
            "metrics_of_explicit_five_seed_mean_prediction"
        ],
        "interpretation_scope": "descriptive_only_no_H1_or_H2_verdict",
    }
    per_seed.to_csv(output_dir / "oof_predictions_per_seed.csv", index=False, lineterminator="\n")
    seed_mean.to_csv(
        output_dir / "oof_predictions_seed_mean.csv", index=False, lineterminator="\n"
    )
    selection.to_csv(output_dir / "inner_selection.csv", index=False, lineterminator="\n")
    per_source.to_csv(output_dir / "per_source_metrics.csv", index=False, lineterminator="\n")
    per_block.to_csv(output_dir / "per_block_metrics.csv", index=False, lineterminator="\n")
    semantics = {
        "xgboost_version": xgboost.__version__,
        "scikit_learn_version": sklearn.__version__,
        "fit_n_jobs": FIT_N_JOBS,
        "prediction_n_jobs": PREDICT_N_JOBS,
        "tree_method": TREE_METHOD,
        "device": DEVICE,
        "objective": "reg:squarederror",
        "booster": "gbtree",
        "eval_metric": "rmse",
        "eval_set": None,
        "callbacks": None,
        "early_stopping_rounds": None,
        "missing_sentinel": MISSING_SENTINEL,
        "missing_sentinel_present_in_fitted_features": False,
        "constructor_parameter_policy": constructor_parameter_policy(),
        "verified_on_every_fitted_booster": True,
    }
    payloads = (
        ("metrics_summary.json", summary),
        ("fit_audit.json", audit),
        ("checkpoint_manifest.json", manifest),
        ("xgboost_semantics.json", semantics),
        (
            "feature_provenance.json",
            {
                **feature_provenance,
                "schema_version": OUTPUT_SCHEMA,
                "r1a_feature_hash_required": EXPECTED_FEATURE_HASH,
                "r1a_feature_provenance_sha256": sha256_file(
                    Path(__file__).resolve().parents[1]
                    / "artifacts"
                    / "r1a_classical"
                    / "feature_provenance.json"
                ),
            },
        ),
        (
            "run_metadata.json",
            {
                "status": "PROVISIONAL_PENDING_INDEPENDENT_VERIFICATION",
                "command": command,
                "runtime_seconds": full_compute_wall_seconds,
                "output_assembly_runtime_seconds": output_assembly_runtime_seconds,
                "checkpoint_operation_sum_seconds": float(
                    sum(row["runtime_seconds"] for row in manifest)
                ),
                "checkpoint_peak_rss_bytes": int(
                    max(row["rss_peak_bytes"] for row in manifest)
                ),
                "grid": [config.payload() for config in XGB_GRID],
                "selection_seed": SELECTION_SEED,
                "outer_seeds": list(OUTER_SEEDS),
                "fit_n_jobs": FIT_N_JOBS,
                "prediction_n_jobs": PREDICT_N_JOBS,
                "tree_method": TREE_METHOD,
                "device": DEVICE,
                "selection_order": [
                    "concatenated_inner_source_macro_mae",
                    "row_micro_mae",
                    "frozen_lower_compute_rank",
                    "lexical_config_id",
                ],
                "compute_rank_policy_frozen_before_results": [
                    "fewer_n_estimators",
                    "shallower_max_depth",
                    "learning_rate_and_reg_lambda_same_compute_tier",
                ],
                "environment": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                    "scikit_learn": sklearn.__version__,
                    "xgboost": xgboost.__version__,
                    "rdkit": rdBase.rdkitVersion,
                    "logical_cpus": psutil.cpu_count(logical=True),
                    "physical_memory_bytes": psutil.virtual_memory().total,
                },
                "checkpoint_resume_policy": (
                    "atomic complete operation payloads only; partial attempts never loaded"
                ),
                "full_compute_orchestration": {
                    "concurrent_fold_workers": 4,
                    "fold_groups": [
                        [1, 2, 3, 4, 5],
                        [6, 7, 8, 9],
                        [10, 11, 12, 13],
                        [14, 15, 16, 17, 18],
                    ],
                    "workers_used_disjoint_checkpoint_directories": True,
                    "assembly_loaded_complete_checkpoints_only": True,
                },
                "scientific_models_run": ["xgboost"],
                "not_run": ["dmpnn"],
            },
        ),
    )
    for name, payload in payloads:
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
            f"R1b2 requires scikit-learn {EXPECTED_SKLEARN}, observed {sklearn.__version__}"
        )
    if xgboost.__version__ != EXPECTED_XGBOOST:
        raise RuntimeError(
            f"R1b2 requires XGBoost {EXPECTED_XGBOOST}, observed {xgboost.__version__}"
        )
    analysis_path = root / "artifacts" / "v2_r0" / "analysis_all_labels.csv"
    feature_frame, metadata, provenance = build_feature_frame(
        analysis_path, root / "artifacts" / "curated_records_public.csv"
    )
    if provenance["feature_matrix_sha256"] != EXPECTED_FEATURE_HASH:
        raise RuntimeError("XGBoost feature matrix differs from accepted R1a")
    accepted = json.loads(
        (root / "artifacts" / "r1a_classical" / "feature_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    for key in (
        "feature_matrix_sha256",
        "feature_columns",
        "passthrough_columns",
        "continuous_columns",
        "definitions",
    ):
        if provenance[key] != accepted[key]:
            raise RuntimeError(f"XGBoost feature provenance differs from R1a: {key}")
    if (feature_frame.loc[:, FEATURE_COLUMNS] == MISSING_SENTINEL).any().any():
        raise RuntimeError("Finite XGBoost missing sentinel collides with a predictor value")
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
    parser.add_argument("--full-compute-wall-seconds", type=float)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    feature_frame, metadata, provenance, contracts = load_inputs(root)
    metadata_by_id = metadata.set_index(ID_COLUMN)
    observed_by_id = feature_frame.set_index(ID_COLUMN)[TARGET_COLUMN]
    if args.pilot_inner:
        config = XGBConfig(800, 8, 0.03, 1)
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
        f"python src/r1b2_xgboost.py --work-dir {args.work_dir.as_posix()} "
        f"--output-dir {args.output_dir.as_posix()} "
        f"--full-compute-wall-seconds {args.full_compute_wall_seconds}"
    )
    assembly_runtime = time.perf_counter() - started
    write_outputs(
        args.output_dir,
        provenance,
        contracts,
        fold_results,
        args.work_dir,
        command,
        (
            float(args.full_compute_wall_seconds)
            if args.full_compute_wall_seconds is not None
            else assembly_runtime
        ),
        assembly_runtime,
    )


if __name__ == "__main__":
    main()
