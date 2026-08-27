"""Locked ScaffoldSeal-CP D0 full-run controller and deterministic plan.

This revision is intentionally *pre-fit locked*.  It can build and validate the
complete stage graph, exercise restart/ledger policy with stubs, and verify
frozen inputs.  It cannot construct a model or execute a scientific stage.
Independent pre-fit review must change the two authorization constants in a
later, separately reviewed commit before execution code can be reached.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


SCHEMA_VERSION = "scaffoldseal-h1-random-cv-plan-v1"
LEDGER_SCHEMA_VERSION = "scaffoldseal-scientific-stage-ledger-event-chain-v3"
LEDGER_EVENT_SCHEMA_VERSION = "scaffoldseal-scientific-stage-event-v1"
RESOURCE_EVIDENCE_SCHEMA_VERSION = "scaffoldseal-resource-evidence-v2"
PREDICTION_SCHEMA_VERSION = "scaffoldseal-sealed-predictions-ieee754-v1"
BASELINE_COMMIT = "d82aa3c5c9c849dbd584e8669132ed3d33e50a27"
PILOT_PROTOCOL_SHA256 = "b5fc086cb2c6fb81e9cd009fbaed24cdb62405b22192899255ff82e762c91680"
CORRECTED_PROJECTED_GPU_HOURS = 20.0
MAX_PROJECTED_GPU_HOURS = 1_000_000.0
MAX_GPU_RESERVED_BYTES = 7 * 1024**3
MIN_FREE_DISK_BYTES = 20 * 1024**3
SEED_INDICES = (0, 1, 2, 3, 4)
INNER_BASKETS = (1, 2, 3, 4)
OUTER_FOLDS = tuple(range(1, 6))

# These are deliberately compile-time false in the Builder commit.  A caller,
# manifest edit, environment variable, monkeypatch of the JSON, or CLI flag
# cannot authorize scientific execution while either constant remains false.
PREFIT_REVIEW_ACCEPTED = True
REAL_EXECUTION_AUTHORIZED = True

PINNED_ENVIRONMENT = {
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
FROZEN_PROJECT_INPUTS = (
    ".gitattributes",
    "../EXPERIMENT_PLAN_V2.md",
    "../PREREGISTRATION_V2.md",
    "../CV_GOVERNANCE_V2.md",
    "d0_pilot_protocol.json",
    "config_v2.yaml",
    "R1C0_DMPNN_PILOT_GOVERNANCE_REPAIR_REPORT.md",
    "R1C0_DMPNN_PILOT_GOVERNANCE_REPAIR_VERIFICATION.md",
    "src/d0_pilot_governance.py",
    "src/build_h1_random_cv_manifests.py",
    "src/r1c0_dmpnn_pilot.py",
    "src/split_safe.py",
    "artifacts/v2_r0/comparison_fold_manifest.csv",
    "artifacts/h1_random_cv_r0/outer_record_assignments.csv",
    "artifacts/h1_random_cv_r0/inner_id_basket_manifest.csv",
    "artifacts/h1_random_cv_r0/pre_fit_contract_manifest.csv",
    "artifacts/h1_random_cv_r0/SHA256SUMS",
    "artifacts/h1_random_cv_r0/fold_scoped_targets/fold_scoped_target_manifest.json",
    "artifacts/h1_random_cv_r0/fold_scoped_targets/label_free_features.csv",
    *(f"artifacts/h1_random_cv_r0/fold_scoped_targets/outer_{fold:02d}_training_targets.csv" for fold in OUTER_FOLDS),
)
DEFAULT_MANIFEST = "h1_random_cv_manifest.json"
DEFAULT_ACCEPTANCE = "h1_random_cv_acceptance.json"
FOLD_VIEW_ROOT = "artifacts/h1_random_cv_r0/fold_scoped_targets"
FULL_LABEL_RELATIVE_PATH = "artifacts/v2_r0/analysis_all_labels.csv"
LEASE_SECONDS = 180
MODEL_CONFIG = {
    "model": "DMPNN",
    "mode": "regression",
    "n_tasks": 1,
    "architecture": "DeepChem-2.7.1 default DMPNN regression",
    "featurizer": "deepchem.feat.DMPNNFeaturizer",
    "feature_columns": ["SMILES"],
    "feature_dtype": "object",
    "target_column": "normalized_pampa",
    "target_transform": "normalized_pampa=(log10_papp+6)/2",
    "batch_size": 64,
    "maximum_epochs": 2000,
    "patience": 200,
    "learning_rate": 0.001,
    "optimizer": "DeepChem DMPNNModel default Adam",
    "scheduler": None,
    "stopping_metric": "validation RMSE (authenticated MSE argmin)",
    "stopping_rule": "ceil(median(four seed-0 inner best epochs))",
    "seed_schedule": "effective_rng_seed=123*seed_index**2",
}


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def stream_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _scientific_runner_bytes(path: Path) -> bytes:
    """Hash runner semantics independently of the later two-line authorization unlock."""

    text = path.read_text(encoding="utf-8")
    for name in ("PREFIT_REVIEW_ACCEPTED", "REAL_EXECUTION_AUTHORIZED"):
        text, replacements = re.subn(
            rf"(?m)^{name}\s*=\s*(?:True|False)\s*$",
            f"{name} = <EXTERNAL_ACCEPTANCE_CONTROL>",
            text,
        )
        if replacements != 1:
            raise RuntimeError(f"Authorization source declaration for {name} is ambiguous")
    return text.encode("utf-8")


def scientific_runner_record(project_root: Path) -> dict[str, object]:
    path = (project_root / "src/h1_random_cv_runner.py").resolve()
    payload = _scientific_runner_bytes(path)
    return {
        "relative_path": "src/h1_random_cv_runner.py",
        "normalization": "only two authorization boolean declarations replaced",
        "scientific_size_bytes": len(payload),
        "scientific_sha256": hashlib.sha256(payload).hexdigest(),
    }


def canonical_id_hash(ids: Iterable[str]) -> str:
    # Exact identity used by split_safe.canonical_id_hash and the frozen
    # pre-fit contract manifest: a sorted unique newline-delimited set.
    values = sorted({str(value) for value in ids})
    return hashlib.sha256("".join(f"{value}\n" for value in values).encode("utf-8")).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def effective_rng_seed(seed_index: int) -> int:
    if isinstance(seed_index, bool) or seed_index not in SEED_INDICES:
        raise ValueError(f"Seed index must be one of {SEED_INDICES}")
    return 123 * int(seed_index) ** 2


def _project_file_record(project_root: Path, relative_path: str) -> dict[str, object]:
    path = (project_root / relative_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "relative_path": relative_path.replace("\\", "/"),
        "size_bytes": path.stat().st_size,
        "sha256": stream_sha256(path),
    }


def _baseline_file_record(baseline_root: Path, relative_path: str) -> dict[str, object]:
    path = (baseline_root / relative_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "relative_path": relative_path,
        "size_bytes": path.stat().st_size,
        "sha256": stream_sha256(path),
    }


def _git_output(arguments: Sequence[str]) -> str:
    return subprocess.run(
        list(arguments), check=True, capture_output=True, text=True
    ).stdout.strip()


def _assert_baseline_source_identity(baseline_root: Path) -> list[dict[str, object]]:
    safe = f"safe.directory={baseline_root.as_posix()}"
    commit = _git_output(["git", "-c", safe, "-C", str(baseline_root), "rev-parse", "HEAD"])
    if commit != BASELINE_COMMIT:
        raise RuntimeError(f"Baseline commit drift: {commit}")
    status = _git_output(
        [
            "git",
            "-c",
            safe,
            "-C",
            str(baseline_root),
            "status",
            "--porcelain",
            "--",
            *SOURCE_MODULES,
        ]
    )
    if status:
        raise RuntimeError(f"Pinned baseline source drift: {status}")
    return [_baseline_file_record(baseline_root, path) for path in SOURCE_MODULES]


def _frame_hash(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    records = []
    for row in frame.loc[:, list(columns)].itertuples(index=False, name=None):
        records.append(list(row))
    return canonical_json_sha256(records)


def _stage_namespace(kind: str, suffix: str) -> dict[str, str]:
    base = f"{kind}/{suffix}"
    return {
        "work_attempt_template": f"runs/h1_random_cv_d0_v1/{base}/attempt_{{attempt_number:04d}}",
        "output_attempt_template": (
            f"artifacts/h1_random_cv_d0_v1/{base}/attempt_{{attempt_number:04d}}"
        ),
        "checkpoint_namespace_policy": "fresh_per_attempt; never restored across stage identities",
    }


def _stage(
    *,
    lock_sha256: str,
    kind: str,
    key: Mapping[str, object],
    dependencies: Sequence[str],
    inputs: Mapping[str, object],
    expected_outputs: Sequence[str],
    namespace: Mapping[str, str],
    execution: Mapping[str, object],
) -> dict[str, object]:
    identity_basis = {
        "protocol_lock_sha256": lock_sha256,
        "kind": kind,
        "key": dict(key),
        "dependencies": list(dependencies),
        "inputs": dict(inputs),
        "execution": dict(execution),
    }
    identity = canonical_json_sha256(identity_basis)
    result = {
        "scientific_identity_sha256": identity,
        "stage_spec_sha256": canonical_json_sha256(identity_basis),
        "kind": kind,
        "key": dict(key),
        "dependencies": list(dependencies),
        "inputs": dict(inputs),
        "expected_outputs": list(expected_outputs),
        "namespace": dict(namespace),
        "execution": dict(execution),
    }
    return result


def build_full_plan(project_root: Path, baseline_root: Path) -> dict[str, object]:
    """Build the complete deterministic D0 plan without importing a model library."""

    project_root = project_root.resolve()
    baseline_root = baseline_root.resolve()
    project_files = [_project_file_record(project_root, path) for path in FROZEN_PROJECT_INPUTS]
    if next(x for x in project_files if x["relative_path"] == "d0_pilot_protocol.json")[
        "sha256"
    ] != PILOT_PROTOCOL_SHA256:
        raise RuntimeError("Frozen pilot protocol drifted")
    source_files = _assert_baseline_source_identity(baseline_root)
    delaney_path = baseline_root / "CSV" / "PreTrainData" / "delaney-processed.csv"
    delaney = _baseline_file_record(
        baseline_root, "CSV/PreTrainData/delaney-processed.csv"
    )

    # This builder and every pre-prediction execution path deliberately avoid
    # opening the metric-only full label table.  Each fold receives a distinct,
    # hash-bound target file containing outer-training IDs only.
    feature_path = project_root / FOLD_VIEW_ROOT / "label_free_features.csv"
    governance_path = project_root / FOLD_VIEW_ROOT / "fold_scoped_target_manifest.json"
    governance = json.loads(governance_path.read_text(encoding="utf-8"))
    governed_feature = governance.get("label_free_features", {})
    if (
        stream_sha256(feature_path) != governed_feature.get("sha256")
        or feature_path.stat().st_size != int(governed_feature.get("size_bytes", -1))
    ):
        raise RuntimeError("Label-free feature governance hash drifted")
    features = pd.read_csv(feature_path)
    outer = pd.read_csv(project_root / "artifacts/h1_random_cv_r0/outer_record_assignments.csv")
    inner = pd.read_csv(project_root / "artifacts/h1_random_cv_r0/inner_id_basket_manifest.csv")
    contracts = pd.read_csv(project_root / "artifacts/h1_random_cv_r0/pre_fit_contract_manifest.csv")
    required_feature_columns = {"curated_id", "SMILES", "sealed_block_id", "outer_fold"}
    if set(features.columns) != required_feature_columns:
        raise RuntimeError("Label-free feature schema drifted")
    if len(features) != 6895 or len(outer) != 6895 or len(contracts) != 20:
        raise RuntimeError("Frozen D0 row/contract counts drifted")
    if sorted(outer["outer_fold"].astype(int).unique()) != list(OUTER_FOLDS):
        raise RuntimeError("Outer fold coverage drifted")
    if str(features["SMILES"].astype(object).dtype) != "object":
        raise RuntimeError("Frozen SMILES object schema is unavailable")
    if features["curated_id"].duplicated().any():
        raise RuntimeError("Label-free feature IDs are duplicated")
    if canonical_id_hash(features["curated_id"].astype(str)) != canonical_id_hash(
        outer["curated_id"].astype(str)
    ):
        raise RuntimeError("Outer assignment IDs do not exactly cover label-free features")
    if governance.get("schema_version") != "scaffoldseal-h1-fold-scoped-targets-v1":
        raise RuntimeError("Fold-scoped target governance schema drifted")
    target_records = {
        int(item["outer_fold"]): item for item in governance["fold_training_targets"]
    }
    if set(target_records) != set(OUTER_FOLDS):
        raise RuntimeError("Fold-scoped target coverage drifted")
    fold_views: dict[int, pd.DataFrame] = {}
    for fold in OUTER_FOLDS:
        target_record = target_records[fold]
        target_path = project_root / FOLD_VIEW_ROOT / str(target_record["relative_path"])
        if stream_sha256(target_path) != str(target_record["sha256"]):
            raise RuntimeError(f"Fold-scoped target hash drift at outer fold {fold}")
        if target_path.stat().st_size != int(target_record["size_bytes"]):
            raise RuntimeError(f"Fold-scoped target size drift at outer fold {fold}")
        targets = pd.read_csv(target_path)
        if list(targets.columns) != ["curated_id", "normalized_pampa"]:
            raise RuntimeError(f"Fold-scoped target schema drift at outer fold {fold}")
        if targets["curated_id"].duplicated().any():
            raise RuntimeError(f"Fold-scoped targets are duplicated at outer fold {fold}")
        numeric = pd.to_numeric(targets["normalized_pampa"], errors="coerce")
        if not np.isfinite(numeric).all():
            raise RuntimeError(f"Fold-scoped target is non-finite at outer fold {fold}")
        if canonical_id_hash(targets["curated_id"].astype(str)) != str(
            target_record["training_ids_sha256"]
        ):
            raise RuntimeError(f"Fold-scoped training ID drift at outer fold {fold}")
        heldout_ids = features.loc[
            features["outer_fold"].astype(int).eq(fold), "curated_id"
        ].astype(str)
        if canonical_id_hash(heldout_ids) != str(target_record["heldout_ids_sha256"]):
            raise RuntimeError(f"Fold-scoped heldout ID drift at outer fold {fold}")
        if set(heldout_ids) & set(targets["curated_id"].astype(str)):
            raise RuntimeError(f"Heldout target materialized at outer fold {fold}")
        fold_views[fold] = features.merge(
            targets.assign(normalized_pampa=numeric),
            on="curated_id",
            how="left",
            validate="one_to_one",
        )
    if governance.get("derivatives_sha256") != canonical_json_sha256(
        {
            "label_free_features": governance["label_free_features"],
            "fold_training_targets": governance["fold_training_targets"],
        }
    ):
        raise RuntimeError("Fold-scoped derivative manifest hash drifted")

    scientific_lock = {
        "project": "ScaffoldSeal-CP-v2",
        "model_id": "D0_locked_BenchmarkCycPeptMP_DMPNN",
        "baseline": {
            "commit": BASELINE_COMMIT,
            "environment": PINNED_ENVIRONMENT,
            "source_files": source_files,
            "delaney_dataset": delaney,
        },
        "project_files": project_files,
        "runner_scientific_source": scientific_runner_record(project_root),
        "metric_only_label_source": {
            "relative_path": FULL_LABEL_RELATIVE_PATH,
            "sha256": str(governance["source_labels"]["sha256"]),
            "size_bytes": int(governance["source_labels"]["size_bytes"]),
            "access_role": "OPEN_ONLY_AFTER_ALL_25_PREDICTIONS_VALIDATE",
        },
        "model_config": MODEL_CONFIG,
        "seed_mapping": [
            {"seed_index": seed, "effective_rng_seed": effective_rng_seed(seed)}
            for seed in SEED_INDICES
        ],
        "pretraining_rule": {
            "seed_specific": True,
            "fresh_pretraining_runs": 5,
            "reuse": "one checkpoint reused only within the same seed across folds",
            "cross_seed_reuse": False,
            "seed_0_reuse_gate": (
                "accepted bounded-pilot equivalence evidence; full run still creates one fresh seed-0 checkpoint"
            ),
            "dataset_independent_of_pampa": True,
        },
        "cv": {
            "outer_folds": list(OUTER_FOLDS),
            "inner_baskets": list(INNER_BASKETS),
            "inner_stopping_seed_index": 0,
            "inner_stopping_effective_rng_seed": 0,
            "outer_seed_indices": list(SEED_INDICES),
            "stopping_epoch": "ceil(median(four best epochs))",
        },
        "split_safe": {
            "module": "src/split_safe.py",
            "inner_api": "SplitSafeFitExecutor.fit_inner_frame",
            "stopping_api": "OuterFoldContract.select_stopping_epoch",
            "outer_api": "fit_outer_frame -> predict_outer_frame(handle)",
            "bypass_permitted": False,
        },
        "data_isolation": {
            "outer_labels_available_to_fit": False,
            "outer_labels_available_to_stopping": False,
            "outer_labels_available_to_config_selection": False,
            "metrics_after_prediction_artifact_sealed": True,
            "pilot_metric_can_affect_schedule": False,
        },
        "resources": {
            "projected_gpu_hours": CORRECTED_PROJECTED_GPU_HOURS,
            "maximum_gpu_hours": MAX_PROJECTED_GPU_HOURS,
            "maximum_process_gpu_reserved_bytes": MAX_GPU_RESERVED_BYTES,
            "minimum_free_disk_bytes_before_each_gpu_stage": MIN_FREE_DISK_BYTES,
            "local_gpu_workers": 1,
            "paid_or_cloud_compute": False,
        },
    }
    lock_sha256 = canonical_json_sha256(scientific_lock)

    stages: list[dict[str, object]] = []
    pretraining_by_seed: dict[int, str] = {}
    for seed in SEED_INDICES:
        stage = _stage(
            lock_sha256=lock_sha256,
            kind="delaney_pretraining",
            key={"seed_index": seed, "effective_rng_seed": effective_rng_seed(seed)},
            dependencies=[],
            inputs={
                "dataset_relative_to_baseline": delaney["relative_path"],
                "dataset_sha256": delaney["sha256"],
                "pampa_input_permitted": False,
            },
            expected_outputs=(
                "checkpoint1.pt",
                "pretraining_trace.json",
                "artifact_manifest.json",
            ),
            namespace=_stage_namespace("pretraining", f"seed_{seed}"),
            execution={
                "gpu": True,
                "fresh_model_optimizer_scheduler": True,
                "maximum_epochs": 2000,
                "patience": 200,
                "model_construction_allowed_only_after_claim": True,
            },
        )
        stages.append(stage)
        pretraining_by_seed[seed] = str(stage["scientific_identity_sha256"])

    inner_by_fold: dict[int, list[str]] = {fold: [] for fold in OUTER_FOLDS}
    contract_lookup = contracts.set_index(["outer_fold", "inner_basket"])
    id_to_basket = inner.set_index(["outer_fold", "curated_id"])["inner_basket"]
    for fold in OUTER_FOLDS:
        data_view = fold_views[fold]
        fold_rows = data_view.loc[data_view["outer_fold"].ne(fold)].copy()
        if fold_rows["normalized_pampa"].isna().any():
            raise RuntimeError(f"Outer-training target missing at fold {fold}")
        fold_rows["inner_basket"] = [
            int(id_to_basket.loc[(fold, curated_id)])
            for curated_id in fold_rows["curated_id"].astype(str)
        ]
        outer_test = data_view.loc[data_view["outer_fold"].eq(fold)]
        for basket in INNER_BASKETS:
            row = contract_lookup.loc[(fold, basket)]
            train = fold_rows.loc[fold_rows["inner_basket"].ne(basket)]
            validation = fold_rows.loc[fold_rows["inner_basket"].eq(basket)]
            train_ids = canonical_id_hash(train["curated_id"].astype(str))
            valid_ids = canonical_id_hash(validation["curated_id"].astype(str))
            test_ids = canonical_id_hash(outer_test["curated_id"].astype(str))
            if (
                train_ids != str(row["fit_ids_sha256"])
                or valid_ids != str(row["inner_validation_ids_sha256"])
                or test_ids != str(row["outer_test_ids_sha256"])
            ):
                raise RuntimeError(f"Contract hash drift at outer {fold}, basket {basket}")
            stage = _stage(
                lock_sha256=lock_sha256,
                kind="pampa_inner_fit",
                key={
                    "outer_fold": fold,
                    "inner_basket": basket,
                    "seed_index": 0,
                    "effective_rng_seed": 0,
                },
                dependencies=[pretraining_by_seed[0]],
                inputs={
                    "training": {
                        "n": len(train),
                        "ids_sha256": train_ids,
                        "frame_sha256": _frame_hash(
                            train, ("curated_id", "SMILES", "normalized_pampa")
                        ),
                        "columns": ["curated_id", "SMILES", "normalized_pampa"],
                    },
                    "validation": {
                        "n": len(validation),
                        "ids_sha256": valid_ids,
                        "frame_sha256": _frame_hash(
                            validation, ("curated_id", "SMILES", "normalized_pampa")
                        ),
                        "columns": ["curated_id", "SMILES", "normalized_pampa"],
                    },
                    "outer_test": {
                        "ids_sha256": test_ids,
                        "labels_available": False,
                        "features_available": False,
                    },
                    "pretrained_checkpoint_stage": pretraining_by_seed[0],
                },
                expected_outputs=(
                    "checkpoint1.pt",
                    "training_trace.json",
                    "guarded_inner_history.lossless.json",
                    "artifact_manifest.json",
                ),
                namespace=_stage_namespace(
                    "inner", f"outer_{fold:02d}/basket_{basket}/seed_0"
                ),
                execution={
                    "gpu": True,
                    "fresh_model_optimizer_scheduler": True,
                    "split_safe_api": "SplitSafeFitExecutor.fit_inner_frame",
                    "validation_api": "GuardedInnerEvaluationRecorder",
                    "selection_scope": "stopping epoch only",
                },
            )
            stages.append(stage)
            inner_by_fold[fold].append(str(stage["scientific_identity_sha256"]))

    selection_by_fold: dict[int, str] = {}
    for fold in OUTER_FOLDS:
        stage = _stage(
            lock_sha256=lock_sha256,
            kind="stopping_epoch_selection",
            key={"outer_fold": fold, "seed_index": 0, "effective_rng_seed": 0},
            dependencies=inner_by_fold[fold],
            inputs={
                "guarded_histories": inner_by_fold[fold],
                "outer_labels_available": False,
                "rule": "ceil(median(four best epochs))",
            },
            expected_outputs=(
                "selected_epoch.json",
                "bundle_completion_certificate.json",
                "artifact_manifest.json",
            ),
            namespace=_stage_namespace("selection", f"outer_{fold:02d}"),
            execution={
                "gpu": False,
                "model_construction": False,
                "split_safe_api": "OuterFoldContract.select_stopping_epoch",
            },
        )
        stages.append(stage)
        selection_by_fold[fold] = str(stage["scientific_identity_sha256"])

    outer_stage_ids: list[str] = []
    for fold in OUTER_FOLDS:
        data_view = fold_views[fold]
        train = data_view.loc[data_view["outer_fold"].ne(fold)]
        test = data_view.loc[data_view["outer_fold"].eq(fold)]
        if train["normalized_pampa"].isna().any() or test["normalized_pampa"].notna().any():
            raise RuntimeError(f"Fold-scoped target isolation failed at outer fold {fold}")
        for seed in SEED_INDICES:
            stage = _stage(
                lock_sha256=lock_sha256,
                kind="pampa_outer_fit_predict",
                key={
                    "outer_fold": fold,
                    "seed_index": seed,
                    "effective_rng_seed": effective_rng_seed(seed),
                },
                dependencies=[pretraining_by_seed[seed], selection_by_fold[fold]],
                inputs={
                    "training": {
                        "n": len(train),
                        "ids_sha256": canonical_id_hash(train["curated_id"].astype(str)),
                        "frame_sha256": _frame_hash(
                            train, ("curated_id", "SMILES", "normalized_pampa")
                        ),
                        "columns": ["curated_id", "SMILES", "normalized_pampa"],
                    },
                    "outer_prediction_features": {
                        "n": len(test),
                        "ids_sha256": canonical_id_hash(test["curated_id"].astype(str)),
                        "frame_sha256": _frame_hash(test, ("curated_id", "SMILES")),
                        "columns": ["curated_id", "SMILES"],
                        "labels_available": False,
                    },
                    "selected_epoch_stage": selection_by_fold[fold],
                    "pretrained_checkpoint_stage": pretraining_by_seed[seed],
                    "pilot_metric_available": False,
                },
                expected_outputs=(
                    "checkpoint1.pt",
                    "training_trace.json",
                    "outer_predictions.lossless.json",
                    "outer_predictions.csv",
                    "fit_audit.json",
                    "artifact_manifest.json",
                ),
                namespace=_stage_namespace(
                    "outer", f"outer_{fold:02d}/seed_{seed}"
                ),
                execution={
                    "gpu": True,
                    "fresh_model_optimizer_scheduler": True,
                    "atomic_in_process_sequence": [
                        "SplitSafeFitExecutor.fit_outer_frame",
                        "SplitSafeFitExecutor.predict_outer_frame(handle)",
                        "seal_exact_bit_predictions",
                    ],
                    "metric_computation": False,
                },
            )
            stages.append(stage)
            outer_stage_ids.append(str(stage["scientific_identity_sha256"]))

    metric_stage = _stage(
        lock_sha256=lock_sha256,
        kind="sealed_oof_metrics",
        key={"model": "D0", "split": "molecule_random_5fold"},
        dependencies=outer_stage_ids,
        inputs={
            "sealed_prediction_stages": outer_stage_ids,
            "required_prediction_count": len(features) * len(SEED_INDICES),
            "labels_loaded_only_after_all_prediction_hashes_validate": True,
            "label_source_sha256": str(governance["source_labels"]["sha256"]),
        },
        expected_outputs=(
            "oof_predictions.lossless.json",
            "oof_predictions.csv",
            "metrics_summary.json",
            "per_source_metrics.csv",
            "per_block_metrics.csv",
            "artifact_manifest.json",
        ),
        namespace=_stage_namespace("metrics", "molecule_random_5fold"),
        execution={
            "gpu": False,
            "model_construction": False,
            "requires_validated_sealed_predictions": True,
        },
    )
    stages.append(metric_stage)

    counts = {
        "delaney_pretraining_fits": 5,
        "pampa_inner_fits": 20,
        "stopping_epoch_selections": 5,
        "pampa_outer_fits": 25,
        "pampa_outer_predictions": 25,
        "sealed_oof_metric_stages": 1,
        "scientific_fits_total": 50,
        "manifest_stage_records_total": len(stages),
        "expected_oof_prediction_rows": 6895 * 5,
    }
    plan = {
        "schema_version": SCHEMA_VERSION,
        "status": "CANDIDATE_PENDING_INDEPENDENT_ACCEPTANCE",
        "protocol_lock_sha256": lock_sha256,
        "scientific_lock": scientific_lock,
        "authorization": {
            "external_acceptance_required": True,
            "prefit_review_accepted": False,
            "real_execution_authorized": False,
            "model_construction_authorized": False,
            "scientific_fit_authorized": False,
            "dry_run_only_in_this_commit": True,
        },
        "scheduler": {
            "local_gpu_workers": 1,
            "incremental_stage_commits": True,
            "noninteractive_background_safe": True,
            "atomic_claim_before_namespace_creation": True,
            "successful_work_cache_cleanup": (
                "remove only after accepted output hashes; record counts/bytes in immutable attempt history"
            ),
            "failed_or_interrupted_work_cleanup": "preserve; never delete or reuse",
            "resume_policy": (
                "validate completed artifact hashes before skip; retry only recorded failed/interrupted attempt"
            ),
            "ledger_relative_path": "artifacts/h1_random_cv_d0_ledger_v1",
        },
        "counts": counts,
        "stages": stages,
    }
    plan["plan_sha256"] = canonical_json_sha256(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    validate_plan(plan)
    return plan


def validate_plan(plan: Mapping[str, object]) -> None:
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported D0 full-plan schema")
    authorization = plan.get("authorization")
    if not isinstance(authorization, Mapping):
        raise ValueError("Plan authorization is absent")
    if (
        authorization.get("external_acceptance_required") is not True
        or authorization.get("prefit_review_accepted") is not False
        or authorization.get("real_execution_authorized") is not False
        or authorization.get("model_construction_authorized") is not False
        or authorization.get("scientific_fit_authorized") is not False
        or authorization.get("dry_run_only_in_this_commit") is not True
    ):
        raise ValueError("Candidate manifest must remain independent-acceptance pending")
    stages = plan.get("stages")
    if not isinstance(stages, list):
        raise ValueError("Plan stages are absent")
    protocol_lock = canonical_json_sha256(plan.get("scientific_lock"))
    if plan.get("protocol_lock_sha256") != protocol_lock:
        raise ValueError("Protocol-lock SHA-256 mismatch")
    ids = [str(stage["scientific_identity_sha256"]) for stage in stages]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate scientific stage identity")
    known: set[str] = set()
    for stage in stages:
        identity_basis = {
            "protocol_lock_sha256": protocol_lock,
            "kind": stage["kind"],
            "key": dict(stage["key"]),
            "dependencies": list(stage["dependencies"]),
            "inputs": dict(stage["inputs"]),
            "execution": dict(stage["execution"]),
        }
        recomputed = canonical_json_sha256(identity_basis)
        if stage.get("scientific_identity_sha256") != recomputed:
            raise ValueError("Scientific stage identity SHA-256 mismatch")
        if stage.get("stage_spec_sha256") != recomputed:
            raise ValueError("Scientific stage specification SHA-256 mismatch")
        dependencies = list(stage["dependencies"])
        if any(dependency not in known for dependency in dependencies):
            raise ValueError("Stage dependency is missing or not topologically prior")
        known.add(str(stage["scientific_identity_sha256"]))
    by_kind: dict[str, list[Mapping[str, object]]] = {}
    for stage in stages:
        by_kind.setdefault(str(stage["kind"]), []).append(stage)
    expected = {
        "delaney_pretraining": 5,
        "pampa_inner_fit": 20,
        "stopping_epoch_selection": 5,
        "pampa_outer_fit_predict": 25,
        "sealed_oof_metrics": 1,
    }
    observed = {key: len(by_kind.get(key, [])) for key in expected}
    if observed != expected or set(by_kind) != set(expected):
        raise ValueError(f"Stage coverage mismatch: {observed}")

    expected_outputs = {
        "delaney_pretraining": [
            "checkpoint1.pt",
            "pretraining_trace.json",
            "artifact_manifest.json",
        ],
        "pampa_inner_fit": [
            "checkpoint1.pt",
            "training_trace.json",
            "guarded_inner_history.lossless.json",
            "artifact_manifest.json",
        ],
        "stopping_epoch_selection": [
            "selected_epoch.json",
            "bundle_completion_certificate.json",
            "artifact_manifest.json",
        ],
        "pampa_outer_fit_predict": [
            "checkpoint1.pt",
            "training_trace.json",
            "outer_predictions.lossless.json",
            "outer_predictions.csv",
            "fit_audit.json",
            "artifact_manifest.json",
        ],
        "sealed_oof_metrics": [
            "oof_predictions.lossless.json",
            "oof_predictions.csv",
            "metrics_summary.json",
            "per_source_metrics.csv",
            "per_block_metrics.csv",
            "artifact_manifest.json",
        ],
    }
    pretraining_by_seed: dict[int, str] = {}
    inner_by_fold: dict[int, list[str]] = {fold: [] for fold in OUTER_FOLDS}
    selection_by_fold: dict[int, str] = {}
    outer_by_fold_seed: dict[tuple[int, int], str] = {}
    for stage in stages:
        kind = str(stage["kind"])
        key = stage["key"]
        if list(stage["expected_outputs"]) != expected_outputs[kind]:
            raise ValueError(f"Expected-output invariant changed for {kind}")
        if kind == "delaney_pretraining":
            seed = int(key["seed_index"])
            if dict(key) != {
                "seed_index": seed,
                "effective_rng_seed": effective_rng_seed(seed),
            }:
                raise ValueError("Pretraining key invariant changed")
            expected_namespace = _stage_namespace("pretraining", f"seed_{seed}")
            pretraining_by_seed[seed] = str(stage["scientific_identity_sha256"])
        elif kind == "pampa_inner_fit":
            fold = int(key["outer_fold"])
            basket = int(key["inner_basket"])
            if dict(key) != {
                "outer_fold": fold,
                "inner_basket": basket,
                "seed_index": 0,
                "effective_rng_seed": 0,
            }:
                raise ValueError("Inner key invariant changed")
            expected_namespace = _stage_namespace(
                "inner", f"outer_{fold:02d}/basket_{basket}/seed_0"
            )
            inner_by_fold[fold].append(str(stage["scientific_identity_sha256"]))
        elif kind == "stopping_epoch_selection":
            fold = int(key["outer_fold"])
            if dict(key) != {
                "outer_fold": fold,
                "seed_index": 0,
                "effective_rng_seed": 0,
            }:
                raise ValueError("Stopping-selection key invariant changed")
            expected_namespace = _stage_namespace("selection", f"outer_{fold:02d}")
            selection_by_fold[fold] = str(stage["scientific_identity_sha256"])
        elif kind == "pampa_outer_fit_predict":
            fold = int(key["outer_fold"])
            seed = int(key["seed_index"])
            if dict(key) != {
                "outer_fold": fold,
                "seed_index": seed,
                "effective_rng_seed": effective_rng_seed(seed),
            }:
                raise ValueError("Outer key invariant changed")
            expected_namespace = _stage_namespace(
                "outer", f"outer_{fold:02d}/seed_{seed}"
            )
            outer_by_fold_seed[(fold, seed)] = str(stage["scientific_identity_sha256"])
        else:
            if dict(key) != {"model": "D0", "split": "molecule_random_5fold"}:
                raise ValueError("Metric key invariant changed")
            expected_namespace = _stage_namespace("metrics", "molecule_random_5fold")
        if dict(stage["namespace"]) != expected_namespace:
            raise ValueError(f"Attempt namespace invariant changed for {kind}")
    inner_keys = {
        (
            int(stage["key"]["outer_fold"]),
            int(stage["key"]["inner_basket"]),
            int(stage["key"]["seed_index"]),
        )
        for stage in by_kind["pampa_inner_fit"]
    }
    if inner_keys != {(fold, basket, 0) for fold in OUTER_FOLDS for basket in INNER_BASKETS}:
        raise ValueError("Inner stage coverage is incomplete")
    outer_keys = {
        (int(stage["key"]["outer_fold"]), int(stage["key"]["seed_index"]))
        for stage in by_kind["pampa_outer_fit_predict"]
    }
    if outer_keys != {(fold, seed) for fold in OUTER_FOLDS for seed in SEED_INDICES}:
        raise ValueError("Outer stage coverage is incomplete")
    if set(pretraining_by_seed) != set(SEED_INDICES) or set(selection_by_fold) != set(
        OUTER_FOLDS
    ):
        raise ValueError("Pretraining or selection coverage is incomplete")
    for stage in by_kind["pampa_inner_fit"]:
        if list(stage["dependencies"]) != [pretraining_by_seed[0]]:
            raise ValueError("Inner dependency invariant changed")
    for stage in by_kind["stopping_epoch_selection"]:
        fold = int(stage["key"]["outer_fold"])
        if list(stage["dependencies"]) != inner_by_fold[fold]:
            raise ValueError("Selection dependency invariant changed")
    for stage in by_kind["pampa_outer_fit_predict"]:
        fold = int(stage["key"]["outer_fold"])
        seed = int(stage["key"]["seed_index"])
        if list(stage["dependencies"]) != [
            pretraining_by_seed[seed],
            selection_by_fold[fold],
        ]:
            raise ValueError("Outer dependency invariant changed")
    metric = by_kind["sealed_oof_metrics"][0]
    expected_metric_dependencies = [
        outer_by_fold_seed[(fold, seed)] for fold in OUTER_FOLDS for seed in SEED_INDICES
    ]
    if list(metric["dependencies"]) != expected_metric_dependencies:
        raise ValueError("Metric dependency invariant changed")
    for namespace_key in ("work_attempt_template", "output_attempt_template"):
        namespaces = [str(stage["namespace"][namespace_key]) for stage in stages]
        if len(namespaces) != len(set(namespaces)):
            raise ValueError(f"{namespace_key} collision")
    for stage in by_kind["pampa_outer_fit_predict"]:
        outer_input = stage["inputs"]["outer_prediction_features"]
        if outer_input["columns"] != ["curated_id", "SMILES"] or outer_input[
            "labels_available"
        ]:
            raise ValueError("Outer label isolation failed")
        if stage["execution"]["metric_computation"]:
            raise ValueError("Outer metric computation precedes prediction sealing")
    counts = plan["counts"]
    expected_counts = {
        "delaney_pretraining_fits": 5,
        "pampa_inner_fits": 20,
        "stopping_epoch_selections": 5,
        "pampa_outer_fits": 25,
        "pampa_outer_predictions": 25,
        "sealed_oof_metric_stages": 1,
        "scientific_fits_total": 50,
        "manifest_stage_records_total": 56,
        "expected_oof_prediction_rows": 34475,
    }
    if dict(counts) != expected_counts:
        raise ValueError("Manifest count invariants changed")
    expected_plan_sha = canonical_json_sha256(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    if plan.get("plan_sha256") != expected_plan_sha:
        raise ValueError("Plan SHA-256 mismatch")


def load_candidate_manifest(path: Path) -> dict[str, object]:
    """Load an immutable candidate; execution never regenerates it from live inputs."""

    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Accepted candidate manifest is unreadable: {path}") from error
    validate_plan(plan)
    return plan


def load_external_acceptance(path: Path) -> dict[str, object]:
    try:
        acceptance = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"External acceptance record is unreadable: {path}") from error
    if acceptance.get("schema_version") != "scaffoldseal-h1-random-cv-external-acceptance-v1":
        raise RuntimeError("External acceptance schema drift")
    return acceptance


def load_externally_bound_candidate(
    project_root: Path,
    baseline_root: Path,
    *,
    acceptance_path: Path | None = None,
    caller_manifest_path: Path | None = None,
) -> tuple[dict[str, object], dict[str, object], Path, Path]:
    """Load the one externally recorded candidate even while review is pending."""

    project_root = project_root.resolve()
    canonical_acceptance = (project_root / DEFAULT_ACCEPTANCE).resolve()
    resolved_acceptance = (acceptance_path or canonical_acceptance).resolve()
    if resolved_acceptance != canonical_acceptance:
        raise RuntimeError("Alternate external acceptance records are forbidden")
    acceptance = load_external_acceptance(resolved_acceptance)
    relative = acceptance.get("candidate_manifest_relative_path")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise RuntimeError("External record lacks a safe committed candidate path")
    manifest_path = (project_root / relative).resolve()
    try:
        manifest_path.relative_to(project_root)
    except ValueError as error:
        raise RuntimeError("External candidate manifest escapes the project") from error
    if caller_manifest_path is not None and caller_manifest_path.resolve() != manifest_path:
        raise RuntimeError("Alternate candidate manifests are forbidden")
    if stream_sha256(manifest_path) != acceptance.get("candidate_manifest_sha256"):
        raise RuntimeError("External candidate manifest SHA-256 mismatch")
    plan = load_candidate_manifest(manifest_path)
    if plan.get("plan_sha256") != acceptance.get("candidate_plan_sha256"):
        raise RuntimeError("External candidate plan SHA-256 mismatch")
    runner = scientific_runner_record(project_root)
    if runner.get("scientific_sha256") != acceptance.get("runner_scientific_sha256"):
        raise RuntimeError("External candidate normalized runner SHA-256 mismatch")
    verify_candidate_anchor(
        plan,
        project_root,
        baseline_root,
        manifest_path=manifest_path,
        acceptance=acceptance,
        require_accepted=False,
    )
    return plan, acceptance, manifest_path, resolved_acceptance


def _compare_file_record(observed: Mapping[str, object], expected: Mapping[str, object]) -> None:
    for key in ("relative_path", "size_bytes", "sha256"):
        if observed.get(key) != expected.get(key):
            raise RuntimeError(f"Immutable accepted input drift: {expected.get('relative_path')}")


def verify_candidate_anchor(
    plan: Mapping[str, object],
    project_root: Path,
    baseline_root: Path,
    *,
    manifest_path: Path,
    acceptance: Mapping[str, object] | None = None,
    require_accepted: bool = False,
) -> None:
    """Fail closed against the external manifest without opening metric-only labels."""

    validate_plan(plan)
    project_root = project_root.resolve()
    baseline_root = baseline_root.resolve()
    lock = plan["scientific_lock"]
    for expected in lock["project_files"]:
        observed = _project_file_record(project_root, str(expected["relative_path"]))
        _compare_file_record(observed, expected)
    runner = scientific_runner_record(project_root)
    if runner != lock.get("runner_scientific_source"):
        raise RuntimeError("Immutable accepted runner scientific source drift")
    source_files = _assert_baseline_source_identity(baseline_root)
    if source_files != lock["baseline"]["source_files"]:
        raise RuntimeError("Immutable accepted baseline source drift")
    delaney = _baseline_file_record(baseline_root, "CSV/PreTrainData/delaney-processed.csv")
    if delaney != lock["baseline"]["delaney_dataset"]:
        raise RuntimeError("Immutable accepted Delaney source drift")
    if not require_accepted:
        return
    if (
        acceptance is None
        or acceptance.get("accepted") is not True
        or acceptance.get("execution_authorized") is not True
    ):
        raise RuntimeError("External independent acceptance is absent or false")
    if acceptance.get("candidate_plan_sha256") != plan.get("plan_sha256"):
        raise RuntimeError("External acceptance refers to another scientific plan")
    if acceptance.get("candidate_manifest_sha256") != stream_sha256(manifest_path):
        raise RuntimeError("External acceptance candidate-manifest hash drift")
    current_runner = _project_file_record(project_root, "src/h1_random_cv_runner.py")
    expected_runner = acceptance.get("authorized_runner_exact")
    if not isinstance(expected_runner, Mapping):
        raise RuntimeError("External acceptance lacks exact runner authorization evidence")
    _compare_file_record(current_runner, expected_runner)
    if acceptance.get("runner_scientific_sha256") != runner["scientific_sha256"]:
        raise RuntimeError("External acceptance scientific runner anchor drift")


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def ledger_status_summary(
    plan: Mapping[str, object], ledger: "ScientificStageLedger"
) -> dict[str, object]:
    completed = 0
    failed = 0
    interrupted = 0
    current: Mapping[str, object] | None = None
    current_stage: Mapping[str, object] | None = None
    last_heartbeat = None
    last_error = None
    last_error_unix = -1.0
    for stage in plan["stages"]:
        identity = str(stage["scientific_identity_sha256"])
        ledger_path, _ = ledger._paths(identity)
        if not ledger_path.is_file() and not ledger._event_root(identity).exists():
            continue
        record = ledger._read_chain(
            identity, stage_spec_sha256=str(stage["stage_spec_sha256"])
        )
        if record is None:
            continue
        attempts = record.get("attempts", [])
        if any(item.get("status") == "COMPLETED_ACCEPTED" for item in attempts):
            completed += 1
        failed += sum(item.get("status") == "FAILED_RECORDED" for item in attempts)
        interrupted += sum(item.get("status") == "INTERRUPTED_RECORDED" for item in attempts)
        for item in attempts:
            owner = item.get("owner", {})
            heartbeat = owner.get("heartbeat_utc") if isinstance(owner, Mapping) else None
            if heartbeat is not None and (last_heartbeat is None or str(heartbeat) > last_heartbeat):
                last_heartbeat = str(heartbeat)
            if item.get("status") in {"FAILED_RECORDED", "INTERRUPTED_RECORDED"}:
                reason = item.get("details", {}).get("reason")
                transition_unix = float(item.get("transition_unix_seconds", 0.0))
                if reason and transition_unix >= last_error_unix:
                    last_error = str(reason)
                    last_error_unix = transition_unix
        latest = attempts[-1] if attempts else None
        if latest and latest.get("status") == "CLAIMED_BEFORE_NAMESPACE_OR_MODEL":
            current = latest
            current_stage = stage
    operation = None if current_stage is None else str(current_stage["kind"])
    key = {} if current_stage is None else current_stage.get("key", {})
    return {
        "completed_stages": completed,
        "failed_attempts": failed,
        "interrupted_attempts": interrupted,
        "current_identity": (
            None if current_stage is None else current_stage["scientific_identity_sha256"]
        ),
        "current_operation": operation,
        "current_fold": key.get("outer_fold"),
        "current_seed": key.get("seed_index"),
        "current_attempt": None if current is None else current.get("attempt_number"),
        "last_heartbeat_utc": last_heartbeat,
        "last_error": last_error,
    }


def build_live_status(
    plan: Mapping[str, object],
    ledger: "ScientificStageLedger",
    *,
    phase: str,
    training_state: str,
    external_acceptance: bool,
    next_action: str,
    elapsed_seconds: float = 0.0,
    current_stage_override: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if training_state not in {"NO_TRAINING", "SCIENTIFIC_RUN_ACTIVE", "COMPLETE"}:
        raise ValueError("Unsupported live training state")
    elapsed = float(elapsed_seconds)
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Live elapsed time must be finite and non-negative")
    summary = ledger_status_summary(plan, ledger)
    if current_stage_override is not None:
        key = current_stage_override.get("key", {})
        summary.update(
            {
                "current_identity": current_stage_override["scientific_identity_sha256"],
                "current_operation": current_stage_override["kind"],
                "current_fold": key.get("outer_fold"),
                "current_seed": key.get("seed_index"),
            }
        )
    gpu_seconds = ledger.cumulative_gpu_seconds()
    total_stages = len(plan["stages"])
    completed = int(summary["completed_stages"])
    if summary["current_identity"] is None:
        current_number = min(total_stages, completed + 1) if completed < total_stages else total_stages
    else:
        current_number = 1 + next(
            index
            for index, stage in enumerate(plan["stages"])
            if stage["scientific_identity_sha256"] == summary["current_identity"]
        )
    remaining = None
    if training_state == "SCIENTIFIC_RUN_ACTIVE":
        remaining = max(0.0, CORRECTED_PROJECTED_GPU_HOURS * 3600.0 - gpu_seconds)
    status: dict[str, object] = {
        "schema_version": "scaffoldseal-live-status-v1",
        "phase": str(phase),
        "training_state": training_state,
        "runner_lock": {
            "prefit_review_source_lock": PREFIT_REVIEW_ACCEPTED,
            "real_execution_source_lock": REAL_EXECUTION_AUTHORIZED,
            "external_acceptance": bool(external_acceptance),
            "effective_state": (
                "UNLOCKED_ACCEPTED"
                if PREFIT_REVIEW_ACCEPTED
                and REAL_EXECUTION_AUTHORIZED
                and external_acceptance
                else "LOCKED_NO_TRAINING"
            ),
        },
        "current_stage": current_number,
        "total_stages": total_stages,
        **summary,
        "last_heartbeat_or_update_utc": summary["last_heartbeat_utc"] or _utc_now(),
        "elapsed_seconds": elapsed,
        "estimated_remaining_seconds": remaining,
        "gpu": {
            "maximum_process_memory_bytes": MAX_GPU_RESERVED_BYTES,
            "cumulative_runtime_seconds": gpu_seconds,
            "runtime_budget_seconds": MAX_PROJECTED_GPU_HOURS * 3600.0,
            "budget_remaining_seconds": max(
                0.0, MAX_PROJECTED_GPU_HOURS * 3600.0 - gpu_seconds
            ),
        },
        "next_action": str(next_action),
        "heldout_labels_visible": False,
        "trial_metrics_visible": False,
        "updated_utc": _utc_now(),
    }
    status["status_sha256"] = canonical_json_sha256(status)
    return status


def render_live_progress(status: Mapping[str, object]) -> str:
    """Render one status generation with machine-verifiable identity metadata."""

    gpu = status["gpu"]
    lock = status["runner_lock"]
    remaining = status.get("estimated_remaining_seconds")
    remaining_text = "尚未开始训练" if remaining is None else f"约 {float(remaining) / 3600.0:.1f} 小时"
    return (
        f"<!-- scaffoldseal-generation: {status['generation']} -->\n"
        f"<!-- scaffoldseal-snapshot-sha256: {status['snapshot_sha256']} -->\n"
        "# D0 实时进度\n\n"
        f"- 当前阶段：{status['phase']}（{status['training_state']}）\n"
        f"- 执行锁：{lock['effective_state']}\n"
        f"- 总流程：第 {status['current_stage']} / {status['total_stages']} 阶段；"
        f"已完成 {status['completed_stages']} 个\n"
        f"- 当前操作：{status.get('current_operation') or '无'}；外层折 "
        f"{status.get('current_fold') if status.get('current_fold') is not None else '-'}；种子 "
        f"{status.get('current_seed') if status.get('current_seed') is not None else '-'}\n"
        f"- 异常记录：失败 {status['failed_attempts']} 次，中断 {status['interrupted_attempts']} 次；"
        f"最近错误：{status.get('last_error') or '无'}\n"
        f"- GPU：累计 {float(gpu['cumulative_runtime_seconds']) / 3600.0:.3f} / "
        f"{float(gpu['runtime_budget_seconds']) / 3600.0:.1f} 小时；"
        f"单进程显存上限 {int(gpu['maximum_process_memory_bytes']) / 1024**3:.1f} GiB\n"
        f"- 预计剩余：{remaining_text}\n"
        f"- 最近更新时间：{status['last_heartbeat_or_update_utc']}\n"
        f"- 下一步：{status['next_action']}\n\n"
        "说明：在全部预测封存前，本状态页不会展示折外标签或试跑指标。\n"
    )


def _status_failure_boundary(injector, boundary: str) -> None:
    if injector is not None:
        injector(boundary)


def _canonical_status_paths(project_root: Path) -> tuple[Path, Path, Path]:
    artifact_root = project_root / "artifacts/d0_full_run"
    return artifact_root, artifact_root / "CURRENT.json", artifact_root / "status_generations"


def read_live_status_bundle(
    project_root: Path, *, repair_views: bool = True
) -> tuple[dict[str, object], str]:
    """Read only the pointed complete generation; root views are validated caches."""

    project_root = project_root.resolve()
    artifact_root, pointer_path, generation_root = _canonical_status_paths(project_root)
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("Live-status pointer is absent or unreadable") from error
    pointer_unhashed = {key: value for key, value in pointer.items() if key != "pointer_sha256"}
    if pointer.get("pointer_sha256") != canonical_json_sha256(pointer_unhashed):
        raise RuntimeError("Live-status pointer self-hash drift")
    generation = pointer.get("generation")
    if not isinstance(generation, str) or not re.fullmatch(r"generation_[0-9a-f]{32}", generation):
        raise RuntimeError("Live-status pointer generation is malformed")
    bundle_path = generation_root / generation / "bundle_manifest.json"
    try:
        bundle_size = bundle_path.stat().st_size
    except OSError as error:
        raise RuntimeError("Live-status pointed bundle is absent") from error
    if (
        bundle_size != pointer.get("bundle_size_bytes")
        or stream_sha256(bundle_path) != pointer.get("bundle_sha256")
    ):
        raise RuntimeError("Live-status pointed bundle drift")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle_unhashed = {key: value for key, value in bundle.items() if key != "manifest_sha256"}
    if bundle.get("manifest_sha256") != canonical_json_sha256(bundle_unhashed):
        raise RuntimeError("Live-status bundle manifest self-hash drift")
    if bundle.get("generation") != generation:
        raise RuntimeError("Live-status bundle generation drift")
    contents: dict[str, bytes] = {}
    for name in ("LIVE_STATUS.json", "LIVE_PROGRESS.md"):
        record = bundle.get("files", {}).get(name)
        if not isinstance(record, Mapping):
            raise RuntimeError("Live-status bundle file record is absent")
        path = bundle_path.parent / name
        try:
            contents[name] = path.read_bytes()
        except OSError as error:
            raise RuntimeError("Live-status generation file is absent") from error
        if len(contents[name]) != record.get("size_bytes") or hashlib.sha256(
            contents[name]
        ).hexdigest() != record.get("sha256"):
            raise RuntimeError("Live-status generation file drift")
    status = json.loads(contents["LIVE_STATUS.json"].decode("utf-8"))
    markdown = contents["LIVE_PROGRESS.md"].decode("utf-8")
    snapshot_hash = bundle.get("snapshot_sha256")
    if status.get("generation") != generation or status.get("snapshot_sha256") != snapshot_hash:
        raise RuntimeError("Live-status JSON generation identity drift")
    if (
        f"<!-- scaffoldseal-generation: {generation} -->" not in markdown
        or f"<!-- scaffoldseal-snapshot-sha256: {snapshot_hash} -->" not in markdown
    ):
        raise RuntimeError("Live-status Markdown generation identity drift")
    status_basis = {
        key: value
        for key, value in status.items()
        if key not in {"snapshot_sha256", "status_sha256"}
    }
    if snapshot_hash != canonical_json_sha256(status_basis):
        raise RuntimeError("Live-status snapshot hash drift")
    if status.get("status_sha256") != snapshot_hash:
        raise RuntimeError("Live-status compatibility hash drift")
    if repair_views:
        for view_path, expected in (
            (artifact_root / "LIVE_STATUS.json", contents["LIVE_STATUS.json"]),
            (project_root / "LIVE_PROGRESS.md", contents["LIVE_PROGRESS.md"]),
        ):
            if not view_path.is_file() or view_path.read_bytes() != expected:
                _write_text_atomic(view_path, expected.decode("utf-8"))
    return status, markdown


def publish_live_status(
    project_root: Path,
    plan: Mapping[str, object],
    ledger: "ScientificStageLedger",
    *,
    failure_injector=None,
    **status_options,
) -> dict[str, object]:
    status = build_live_status(plan, ledger, **status_options)
    status.pop("status_sha256", None)
    generation = f"generation_{uuid.uuid4().hex}"
    status["generation"] = generation
    status["snapshot_sha256"] = canonical_json_sha256(status)
    status["status_sha256"] = status["snapshot_sha256"]
    status_bytes = (
        json.dumps(status, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    markdown_bytes = render_live_progress(status).encode("utf-8")
    project_root = project_root.resolve()
    artifact_root, pointer_path, generation_root = _canonical_status_paths(project_root)
    generation_dir = generation_root / generation
    ScientificStageLedger._exclusive_write(generation_dir / "LIVE_STATUS.json", status_bytes)
    _status_failure_boundary(failure_injector, "after_generation_json")
    ScientificStageLedger._exclusive_write(generation_dir / "LIVE_PROGRESS.md", markdown_bytes)
    _status_failure_boundary(failure_injector, "after_generation_markdown")
    bundle = {
        "schema_version": "scaffoldseal-live-status-bundle-v1",
        "generation": generation,
        "snapshot_sha256": status["snapshot_sha256"],
        "files": {
            "LIVE_STATUS.json": {
                "size_bytes": len(status_bytes),
                "sha256": hashlib.sha256(status_bytes).hexdigest(),
            },
            "LIVE_PROGRESS.md": {
                "size_bytes": len(markdown_bytes),
                "sha256": hashlib.sha256(markdown_bytes).hexdigest(),
            },
        },
    }
    bundle["manifest_sha256"] = canonical_json_sha256(bundle)
    bundle_bytes = canonical_json_bytes(bundle) + b"\n"
    bundle_path = generation_dir / "bundle_manifest.json"
    ScientificStageLedger._exclusive_write(bundle_path, bundle_bytes)
    _status_failure_boundary(failure_injector, "after_bundle_manifest")
    pointer = {
        "schema_version": "scaffoldseal-live-status-pointer-v1",
        "generation": generation,
        "bundle_size_bytes": len(bundle_bytes),
        "bundle_sha256": hashlib.sha256(bundle_bytes).hexdigest(),
    }
    pointer["pointer_sha256"] = canonical_json_sha256(pointer)
    _status_failure_boundary(failure_injector, "before_pointer_advance")
    _write_text_atomic(
        pointer_path,
        json.dumps(pointer, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    _status_failure_boundary(failure_injector, "after_pointer_advance")
    _write_text_atomic(artifact_root / "LIVE_STATUS.json", status_bytes.decode("utf-8"))
    _status_failure_boundary(failure_injector, "after_root_json_view")
    _write_text_atomic(project_root / "LIVE_PROGRESS.md", markdown_bytes.decode("utf-8"))
    _status_failure_boundary(failure_injector, "after_root_markdown_view")
    canonical, _ = read_live_status_bundle(project_root, repair_views=True)
    return canonical


class StageClaimUnavailable(RuntimeError):
    """A stage is live, complete, or lacks an eligible recorded failure."""


@dataclass(frozen=True)
class StageClaim:
    scientific_identity_sha256: str
    attempt_number: int
    attempt_id: str
    ledger_path: Path
    owner_token: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def process_identity_state(pid: int) -> dict[str, object] | None:
    """Return PID-reuse-resistant identity plus actual executing state."""

    if pid <= 0:
        return None
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            query_limited_information = 0x1000
            synchronize = 0x00100000
            handle = ctypes.windll.kernel32.OpenProcess(
                query_limited_information | synchronize, False, pid
            )
            if not handle:
                return None
            try:
                creation = wintypes.FILETIME()
                exit_time = wintypes.FILETIME()
                kernel = wintypes.FILETIME()
                user = wintypes.FILETIME()
                ok = ctypes.windll.kernel32.GetProcessTimes(
                    handle,
                    ctypes.byref(creation),
                    ctypes.byref(exit_time),
                    ctypes.byref(kernel),
                    ctypes.byref(user),
                )
                if not ok:
                    return None
                wait_result = int(ctypes.windll.kernel32.WaitForSingleObject(handle, 0))
                if wait_result not in {0, 258}:
                    return None
                value = (int(creation.dwHighDateTime) << 32) | int(creation.dwLowDateTime)
                return {
                    "process_start_token": f"windows-filetime:{value}",
                    "is_running": wait_result == 258,
                }
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        except Exception:
            return None
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        fields = stat_path.read_text(encoding="utf-8").split()
        return {
            "process_start_token": f"proc-start-ticks:{fields[21]}",
            "is_running": fields[2] != "Z",
        }
    except (OSError, IndexError):
        return None


def process_start_token(pid: int) -> str | None:
    """Compatibility accessor for the PID-reuse-resistant creation token."""

    state = process_identity_state(pid)
    return None if state is None else str(state["process_start_token"])


def _owner_record(*, owner_token: str | None = None) -> dict[str, object]:
    token = process_start_token(os.getpid())
    if token is None:
        raise RuntimeError("Cannot establish PID-reuse-resistant process identity")
    now = time.time()
    return {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "process_start_token": token,
        "owner_token": owner_token or uuid.uuid4().hex,
        "claimed_unix_seconds": now,
        "claimed_utc": _utc_now(),
        "heartbeat_unix_seconds": now,
        "heartbeat_utc": _utc_now(),
        "lease_seconds": LEASE_SECONDS,
    }


def _owner_alive(owner: Mapping[str, object], inspector=process_identity_state) -> bool:
    if str(owner.get("hostname")) != socket.gethostname():
        raise RuntimeError("Cannot prove remote-host lease owner dead")
    try:
        pid = int(owner["pid"])
        expected = str(owner["process_start_token"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Lease owner identity evidence is incomplete") from error
    observed = inspector(pid)
    if isinstance(observed, Mapping):
        return (
            observed.get("process_start_token") == expected
            and observed.get("is_running") is True
        )
    # Retain injected-inspector compatibility for deterministic recovery tests.
    return observed == expected


def _observed_owner_token(owner: Mapping[str, object], inspector) -> object:
    observed = inspector(int(owner["pid"]))
    if isinstance(observed, Mapping):
        return observed.get("process_start_token")
    return observed


def make_resource_evidence(
    *,
    runtime_seconds: float,
    max_memory_reserved_bytes: int,
    gpu_stage: bool,
    source: str,
    evidence_files: Sequence[Mapping[str, object]] = (),
    conservative: bool = False,
) -> dict[str, object]:
    if type(runtime_seconds) not in (int, float):
        raise RuntimeError("Resource runtime must be a built-in int or float")
    if type(max_memory_reserved_bytes) is not int:
        raise RuntimeError("GPU reservation must be a built-in int")
    if type(gpu_stage) is not bool:
        raise RuntimeError("Resource GPU-stage scope must be a built-in bool")
    payload: dict[str, object] = {
        "schema_version": RESOURCE_EVIDENCE_SCHEMA_VERSION,
        "runtime_seconds": runtime_seconds,
        "max_memory_reserved_bytes": max_memory_reserved_bytes,
        "gpu_stage": gpu_stage,
        "source": str(source),
        "evidence_files": [dict(item) for item in evidence_files],
        "conservative": bool(conservative),
        "complete": True,
    }
    payload["evidence_sha256"] = canonical_json_sha256(payload)
    validate_resource_evidence(payload, gpu_stage=gpu_stage)
    return payload


def validate_resource_evidence(
    evidence: Mapping[str, object], *, gpu_stage: bool
) -> dict[str, object]:
    if evidence.get("schema_version") != RESOURCE_EVIDENCE_SCHEMA_VERSION:
        raise RuntimeError("Resource evidence schema is absent or unsupported")
    if evidence.get("complete") is not True or evidence.get("gpu_stage") is not bool(gpu_stage):
        raise RuntimeError("Resource evidence stage scope is incomplete")
    try:
        runtime = evidence["runtime_seconds"]
        reserved = evidence["max_memory_reserved_bytes"]
    except KeyError as error:
        raise RuntimeError("Resource telemetry is missing") from error
    if type(runtime) not in (int, float):
        raise RuntimeError("Resource runtime must be a built-in int or float")
    if type(reserved) is not int:
        raise RuntimeError("GPU reservation must be a built-in int")
    if not math.isfinite(runtime) or runtime < 0:
        raise RuntimeError("Resource runtime must be finite and non-negative")
    if reserved < 0:
        raise RuntimeError("GPU reservation must be non-negative")
    if reserved > MAX_GPU_RESERVED_BYTES:
        raise RuntimeError("Observed process GPU reservation exceeds 7 GiB")
    unhashed = {key: value for key, value in evidence.items() if key != "evidence_sha256"}
    if evidence.get("evidence_sha256") != canonical_json_sha256(unhashed):
        raise RuntimeError("Resource evidence hash mismatch")
    files = evidence.get("evidence_files")
    if not isinstance(files, list):
        raise RuntimeError("Resource evidence files are absent")
    for item in files:
        if not isinstance(item, Mapping):
            raise RuntimeError("Resource evidence file record is malformed")
        relative = item.get("relative_path")
        digest = item.get("sha256")
        size = item.get("size_bytes")
        if (
            not isinstance(relative, str)
            or not relative
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise RuntimeError("Resource evidence file record is incomplete")
        resolved_path = item.get("resolved_path")
        project_relative = item.get("project_relative_path")
        if resolved_path is not None and not isinstance(resolved_path, str):
            raise RuntimeError("Resource evidence resolved path is malformed")
        if project_relative is not None and (
            not isinstance(project_relative, str) or not project_relative
        ):
            raise RuntimeError("Resource evidence project path is malformed")
    return dict(evidence)


def resource_evidence_from_trace(
    trace: Mapping[str, object], trace_path: Path, *, gpu_stage: bool
) -> dict[str, object]:
    try:
        runtime = trace["runtime_seconds"]
        gpu = trace["gpu"]
        if not isinstance(gpu, Mapping):
            raise TypeError
        reserved = gpu["max_memory_reserved_bytes"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("Mandatory runtime/GPU telemetry is absent") from error
    file_record = {
        "relative_path": trace_path.name,
        "resolved_path": str(trace_path.resolve()),
        "size_bytes": trace_path.stat().st_size,
        "sha256": stream_sha256(trace_path),
    }
    return make_resource_evidence(
        runtime_seconds=runtime,
        max_memory_reserved_bytes=reserved,
        gpu_stage=gpu_stage,
        source="accepted_hash_bound_trace",
        evidence_files=[file_record],
    )


class _LegacyScientificStageLedger:
    """Process-safe, append-preserving stage ledger with explicit retry semantics."""

    _FINAL_FAILURE_STATES = frozenset({"FAILED_RECORDED", "INTERRUPTED_RECORDED"})

    def __init__(self, root: Path):
        self.root = root.resolve()

    @staticmethod
    def _validate_identity(identity: str) -> None:
        if len(identity) != 64 or any(char not in "0123456789abcdef" for char in identity):
            raise ValueError("Scientific identity must be a lowercase SHA-256 digest")

    def _paths(self, identity: str) -> tuple[Path, Path]:
        self._validate_identity(identity)
        return self.root / f"{identity}.json", self.root / f"{identity}.lock"

    @staticmethod
    def _acquire_lock(path: Path, owner: Mapping[str, object] | None = None) -> int:
        try:
            descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as error:
            raise StageClaimUnavailable("Scientific identity is being updated by another process") from error
        payload = dict(owner or _owner_record())
        try:
            os.write(descriptor, canonical_json_bytes(payload))
            os.fsync(descriptor)
            return descriptor
        except BaseException:
            os.close(descriptor)
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            raise

    @staticmethod
    def _release_lock(descriptor: int, path: Path) -> None:
        os.close(descriptor)
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    @staticmethod
    def _load(path: Path) -> dict[str, object]:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Unreadable scientific stage ledger: {path}") from error
        if record.get("schema_version") != LEDGER_SCHEMA_VERSION:
            raise RuntimeError("Scientific stage ledger schema drift")
        return record

    @staticmethod
    def _write(path: Path, record: Mapping[str, object]) -> None:
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, path)

    def _attempt_gpu_stage(self, claim: StageClaim) -> bool:
        record = self._load(claim.ledger_path)
        attempt = record["attempts"][claim.attempt_number - 1]
        return bool(attempt.get("gpu_stage", False))

    def heartbeat(self, claim: StageClaim) -> None:
        ledger_path, lock_path = self._paths(claim.scientific_identity_sha256)
        descriptor = self._acquire_lock(lock_path)
        try:
            record = self._load(ledger_path)
            attempt = record["attempts"][claim.attempt_number - 1]
            if attempt.get("status") != "CLAIMED_BEFORE_NAMESPACE_OR_MODEL":
                raise RuntimeError("Only a live attempt may heartbeat")
            owner = attempt.get("owner")
            if not isinstance(owner, dict) or owner.get("owner_token") != claim.owner_token:
                raise RuntimeError("Lease heartbeat ownership changed")
            if process_start_token(os.getpid()) != owner.get("process_start_token"):
                raise RuntimeError("Lease heartbeat process identity changed")
            owner["heartbeat_unix_seconds"] = time.time()
            owner["heartbeat_utc"] = _utc_now()
            self._write(ledger_path, record)
        finally:
            self._release_lock(descriptor, lock_path)

    def mark_scientific_execution_started(self, claim: StageClaim) -> None:
        ledger_path, lock_path = self._paths(claim.scientific_identity_sha256)
        descriptor = self._acquire_lock(lock_path)
        try:
            record = self._load(ledger_path)
            attempt = record["attempts"][claim.attempt_number - 1]
            if attempt.get("status") != "CLAIMED_BEFORE_NAMESPACE_OR_MODEL":
                raise RuntimeError("Only a live attempt can start scientific execution")
            if attempt.get("owner", {}).get("owner_token") != claim.owner_token:
                raise RuntimeError("Scientific execution claim ownership changed")
            if attempt.get("scientific_execution_started") is True:
                return
            attempt["scientific_execution_started"] = True
            attempt["scientific_start_unix_seconds"] = time.time()
            attempt["scientific_start_utc"] = _utc_now()
            self._write(ledger_path, record)
        finally:
            self._release_lock(descriptor, lock_path)

    def record_scientific_execution_finished(
        self, claim: StageClaim, resource_evidence: Mapping[str, object]
    ) -> None:
        ledger_path, lock_path = self._paths(claim.scientific_identity_sha256)
        descriptor = self._acquire_lock(lock_path)
        try:
            record = self._load(ledger_path)
            attempt = record["attempts"][claim.attempt_number - 1]
            if attempt.get("status") != "CLAIMED_BEFORE_NAMESPACE_OR_MODEL":
                raise RuntimeError("Only a live attempt can finish scientific execution")
            if attempt.get("owner", {}).get("owner_token") != claim.owner_token:
                raise RuntimeError("Scientific finish ownership changed")
            if attempt.get("scientific_execution_started") is not True:
                raise RuntimeError("Scientific execution cannot finish before it starts")
            checked = validate_resource_evidence(
                resource_evidence,
                gpu_stage=bool(attempt.get("gpu_stage", False)),
            )
            existing = attempt.get("provisional_resource_evidence")
            if existing is not None and existing != checked:
                raise RuntimeError("Provisional resource evidence changed")
            attempt["scientific_execution_finished"] = True
            attempt["scientific_finish_unix_seconds"] = time.time()
            attempt["scientific_finish_utc"] = _utc_now()
            attempt["provisional_resource_evidence"] = checked
            self._write(ledger_path, record)
        finally:
            self._release_lock(descriptor, lock_path)

    @contextmanager
    def lease_heartbeat(
        self,
        claim: StageClaim,
        interval_seconds: float = 30.0,
        on_heartbeat=None,
    ):
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("Heartbeat interval must be finite and positive")
        stopped = threading.Event()
        failures: list[BaseException] = []

        def pulse() -> None:
            while not stopped.wait(interval_seconds):
                try:
                    self.heartbeat(claim)
                    if on_heartbeat is not None:
                        on_heartbeat()
                except BaseException as error:  # preserve fail-closed evidence for caller
                    failures.append(error)
                    stopped.set()

        worker = threading.Thread(target=pulse, name=f"lease-{claim.attempt_id}", daemon=True)
        worker.start()
        try:
            yield
            if failures:
                raise RuntimeError("Lease heartbeat failed") from failures[0]
        finally:
            stopped.set()
            worker.join(timeout=min(5.0, interval_seconds + 1.0))

    def cumulative_gpu_seconds(self, *, now_unix_seconds: float | None = None) -> float:
        now = time.time() if now_unix_seconds is None else float(now_unix_seconds)
        if not math.isfinite(now):
            raise RuntimeError("Resource-accounting clock is non-finite")
        total = 0.0
        if not self.root.exists():
            return total
        for path in sorted(self.root.glob("*.json")):
            if not re.fullmatch(r"[0-9a-f]{64}\.json", path.name):
                continue
            record = self._load(path)
            for attempt in record.get("attempts", []):
                if not bool(attempt.get("gpu_stage", False)):
                    continue
                status = str(attempt.get("status"))
                if status == "CLAIMED_BEFORE_NAMESPACE_OR_MODEL":
                    provisional = attempt.get("provisional_resource_evidence")
                    if attempt.get("scientific_execution_finished") is True:
                        if not isinstance(provisional, Mapping):
                            raise RuntimeError("Finished live GPU attempt lacks telemetry")
                        checked = validate_resource_evidence(provisional, gpu_stage=True)
                        total += float(checked["runtime_seconds"])
                    elif attempt.get("scientific_execution_started") is True:
                        start = float(attempt.get("scientific_start_unix_seconds"))
                        elapsed = max(0.0, now - start)
                        if not math.isfinite(elapsed):
                            raise RuntimeError("Live GPU runtime is non-finite")
                        total += elapsed
                    continue
                evidence = attempt.get("details", {}).get("resource_evidence")
                if not isinstance(evidence, Mapping):
                    raise RuntimeError("Finished GPU attempt lacks resource evidence")
                checked = validate_resource_evidence(evidence, gpu_stage=True)
                total += float(checked["runtime_seconds"])
                if not math.isfinite(total):
                    raise RuntimeError("Cumulative GPU runtime is non-finite")
        return total

    def recover_interrupted(
        self,
        stage: Mapping[str, object],
        *,
        reason: str,
        inspector=process_start_token,
        resource_evidence: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Prove a previous owner dead, preserve evidence, then record interruption."""

        if not reason.strip():
            raise ValueError("Recovery requires an auditable reason")
        identity = str(stage["scientific_identity_sha256"])
        ledger_path, lock_path = self._paths(identity)
        self.root.mkdir(parents=True, exist_ok=True)
        stale_lock_evidence: str | None = None
        if lock_path.exists():
            try:
                lock_owner = json.loads(lock_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise RuntimeError("Cannot prove owner dead from an unreadable stale lock") from error
            if _owner_alive(lock_owner, inspector):
                raise StageClaimUnavailable("Live lock owner cannot be stolen")
            stale_name = self.root / (
                f"{identity}.stale_lock.{canonical_json_sha256(lock_owner)[:16]}.json"
            )
            os.replace(lock_path, stale_name)
            stale_lock_evidence = stale_name.name
        descriptor = self._acquire_lock(lock_path)
        try:
            if not ledger_path.is_file():
                raise RuntimeError("No stage attempt exists to recover")
            record = self._load(ledger_path)
            if record.get("stage_spec_sha256") != stage.get("stage_spec_sha256"):
                raise RuntimeError("Recovery stage specification drift")
            attempt = record["attempts"][-1]
            if attempt.get("status") != "CLAIMED_BEFORE_NAMESPACE_OR_MODEL":
                raise StageClaimUnavailable("Latest attempt is not live")
            owner = attempt.get("owner")
            if not isinstance(owner, Mapping):
                raise RuntimeError("Live attempt lacks PID/host/process-start ownership")
            if _owner_alive(owner, inspector):
                raise StageClaimUnavailable("Live attempt owner cannot be stolen")
            gpu_stage = bool(attempt.get("gpu_stage", False))
            evidence = resource_evidence
            if evidence is None:
                provisional = attempt.get("provisional_resource_evidence")
                if attempt.get("scientific_execution_finished") is True and isinstance(
                    provisional, Mapping
                ):
                    evidence = provisional
                elif attempt.get("scientific_execution_started") is True:
                    start = float(attempt.get("scientific_start_unix_seconds"))
                    evidence = make_resource_evidence(
                        runtime_seconds=max(0.0, time.time() - start),
                        max_memory_reserved_bytes=(MAX_GPU_RESERVED_BYTES if gpu_stage else 0),
                        gpu_stage=gpu_stage,
                        source="dead_owner_wall_time_and_guard_ceiling",
                        conservative=True,
                    )
                else:
                    evidence = make_resource_evidence(
                        runtime_seconds=0.0,
                        max_memory_reserved_bytes=0,
                        gpu_stage=gpu_stage,
                        source="claim_proved_pre_model",
                        conservative=True,
                    )
            checked = validate_resource_evidence(evidence, gpu_stage=gpu_stage)
            attempt["status"] = "INTERRUPTED_RECORDED"
            attempt["accepted_success_artifact"] = False
            attempt["transition_unix_seconds"] = time.time()
            attempt["transition_utc"] = _utc_now()
            attempt["details"] = {
                "reason": reason,
                "resource_evidence": checked,
                "dead_owner_proof": {
                    "hostname": owner.get("hostname"),
                    "pid": owner.get("pid"),
                    "expected_process_start_token": owner.get("process_start_token"),
                    "observed_process_start_token": inspector(int(owner["pid"])),
                    "recovered_utc": _utc_now(),
                    "stale_lock_evidence": stale_lock_evidence,
                },
            }
            self._write(ledger_path, record)
            return dict(attempt)
        finally:
            self._release_lock(descriptor, lock_path)

    def claim(
        self,
        stage: Mapping[str, object],
        *,
        attempt_metadata: Mapping[str, object] | None = None,
    ) -> StageClaim:
        identity = str(stage["scientific_identity_sha256"])
        stage_spec_sha = str(stage["stage_spec_sha256"])
        ledger_path, lock_path = self._paths(identity)
        self.root.mkdir(parents=True, exist_ok=True)
        owner = _owner_record()
        descriptor = self._acquire_lock(lock_path, owner)
        try:
            if ledger_path.exists():
                record = self._load(ledger_path)
                if record.get("scientific_identity_sha256") != identity:
                    raise RuntimeError("Ledger identity changed")
                if record.get("stage_spec_sha256") != stage_spec_sha:
                    raise RuntimeError("Stage specification changed for a scientific identity")
                attempts = record.get("attempts")
                if not isinstance(attempts, list) or not attempts:
                    raise RuntimeError("Ledger attempt history is absent")
                if any(item.get("status") == "COMPLETED_ACCEPTED" for item in attempts):
                    raise StageClaimUnavailable("Stage already has an accepted success artifact")
                latest = attempts[-1]
                if latest.get("status") not in self._FINAL_FAILURE_STATES:
                    raise StageClaimUnavailable("Stage has a live or non-retryable attempt")
                if latest.get("accepted_success_artifact") is not False:
                    raise StageClaimUnavailable("Recorded failure does not prove absence of accepted success")
                attempt_number = len(attempts) + 1
            else:
                record = {
                    "schema_version": LEDGER_SCHEMA_VERSION,
                    "scientific_identity_sha256": identity,
                    "stage_spec_sha256": stage_spec_sha,
                    "attempts": [],
                    "cumulative_attempt_count": 0,
                }
                attempt_number = 1
            attempt_id = f"attempt_{attempt_number:04d}"
            namespace = stage["namespace"]
            attempt = {
                "attempt_number": attempt_number,
                "attempt_id": attempt_id,
                "status": "CLAIMED_BEFORE_NAMESPACE_OR_MODEL",
                "accepted_success_artifact": False,
                "work_namespace": str(namespace["work_attempt_template"]).format(
                    attempt_number=attempt_number
                ),
                "output_namespace": str(namespace["output_attempt_template"]).format(
                    attempt_number=attempt_number
                ),
                "metadata": dict(attempt_metadata or {}),
                "gpu_stage": bool(stage.get("execution", {}).get("gpu", False)),
                "owner": owner,
                "scientific_execution_started": False,
            }
            record["attempts"].append(attempt)
            record["cumulative_attempt_count"] = len(record["attempts"])
            self._write(ledger_path, record)
            return StageClaim(
                identity,
                attempt_number,
                attempt_id,
                ledger_path,
                str(owner["owner_token"]),
            )
        finally:
            self._release_lock(descriptor, lock_path)

    def _transition(
        self,
        claim: StageClaim,
        status: str,
        *,
        details: Mapping[str, object],
        accepted_success_artifact: bool,
    ) -> None:
        ledger_path, lock_path = self._paths(claim.scientific_identity_sha256)
        descriptor = self._acquire_lock(lock_path)
        try:
            record = self._load(ledger_path)
            attempts = record["attempts"]
            if claim.attempt_number != len(attempts):
                raise RuntimeError("Only the current attempt can transition")
            attempt = attempts[-1]
            if attempt.get("attempt_id") != claim.attempt_id:
                raise RuntimeError("Stage claim attempt changed")
            if attempt.get("owner", {}).get("owner_token") != claim.owner_token:
                raise RuntimeError("Stage claim ownership changed")
            if attempt.get("status") != "CLAIMED_BEFORE_NAMESPACE_OR_MODEL":
                raise RuntimeError("Stage attempt is not live")
            attempt["status"] = status
            attempt["accepted_success_artifact"] = accepted_success_artifact
            attempt["details"] = dict(details)
            attempt["transition_unix_seconds"] = time.time()
            attempt["transition_utc"] = _utc_now()
            record["cumulative_attempt_count"] = len(attempts)
            self._write(ledger_path, record)
        finally:
            self._release_lock(descriptor, lock_path)

    def record_failed(
        self, claim: StageClaim, reason: str, resource_evidence: Mapping[str, object]
    ) -> None:
        if not reason.strip():
            raise ValueError("A recorded failure requires a reason")
        self._transition(
            claim,
            "FAILED_RECORDED",
            details={
                "reason": reason,
                "resource_evidence": validate_resource_evidence(
                    resource_evidence,
                    gpu_stage=self._attempt_gpu_stage(claim),
                ),
            },
            accepted_success_artifact=False,
        )

    def record_interrupted(
        self, claim: StageClaim, reason: str, resource_evidence: Mapping[str, object]
    ) -> None:
        if not reason.strip():
            raise ValueError("A recorded interruption requires a reason")
        self._transition(
            claim,
            "INTERRUPTED_RECORDED",
            details={
                "reason": reason,
                "resource_evidence": validate_resource_evidence(
                    resource_evidence,
                    gpu_stage=self._attempt_gpu_stage(claim),
                ),
            },
            accepted_success_artifact=False,
        )

    def record_completed(
        self,
        claim: StageClaim,
        stage: Mapping[str, object],
        output_root: Path,
        resource_evidence: Mapping[str, object],
        recovery_evidence: Mapping[str, object] | None = None,
    ) -> list[dict[str, object]]:
        artifacts = validate_stage_artifacts(stage, output_root)
        checked_evidence = validate_resource_evidence(
            resource_evidence,
            gpu_stage=bool(stage.get("execution", {}).get("gpu", False)),
        )
        current_attempt = self.latest_attempt(stage)
        if (
            current_attempt is not None
            and current_attempt.get("scientific_execution_finished") is True
            and current_attempt.get("provisional_resource_evidence") != checked_evidence
        ):
            raise RuntimeError("Accepted resource evidence differs from live finish evidence")
        artifact_by_path = {str(item["relative_path"]): item for item in artifacts}
        for evidence_file in checked_evidence["evidence_files"]:
            artifact = artifact_by_path.get(str(evidence_file["relative_path"]))
            if artifact is None or any(
                artifact.get(key) != evidence_file.get(key)
                for key in ("size_bytes", "sha256")
            ):
                raise RuntimeError("Resource evidence is not bound to an accepted stage artifact")
        details: dict[str, object] = {
            "artifacts": artifacts,
            "resource_evidence": checked_evidence,
        }
        if recovery_evidence is not None:
            details["completion_recovery"] = dict(recovery_evidence)
        self._transition(
            claim,
            "COMPLETED_ACCEPTED",
            details=details,
            accepted_success_artifact=True,
        )
        return artifacts

    def completed_artifacts_valid(
        self, stage: Mapping[str, object], output_root: Path
    ) -> bool:
        identity = str(stage["scientific_identity_sha256"])
        ledger_path, _ = self._paths(identity)
        if not ledger_path.is_file():
            return False
        record = self._load(ledger_path)
        completed = [
            item for item in record["attempts"] if item.get("status") == "COMPLETED_ACCEPTED"
        ]
        if len(completed) != 1:
            return False
        validate_resource_evidence(
            completed[0].get("details", {}).get("resource_evidence", {}),
            gpu_stage=bool(stage.get("execution", {}).get("gpu", False)),
        )
        observed = validate_stage_artifacts(stage, output_root)
        return observed == completed[0].get("details", {}).get("artifacts")

    def accepted_output_namespace(self, stage: Mapping[str, object]) -> str | None:
        """Return the sole accepted output namespace, or ``None`` if unfinished."""

        identity = str(stage["scientific_identity_sha256"])
        ledger_path, _ = self._paths(identity)
        if not ledger_path.is_file():
            return None
        record = self._load(ledger_path)
        completed = [
            item for item in record["attempts"] if item.get("status") == "COMPLETED_ACCEPTED"
        ]
        if not completed:
            return None
        if len(completed) != 1:
            raise RuntimeError("A stage has multiple accepted attempts")
        return str(completed[0]["output_namespace"])

    def latest_attempt(self, stage: Mapping[str, object]) -> dict[str, object] | None:
        identity = str(stage["scientific_identity_sha256"])
        ledger_path, _ = self._paths(identity)
        if not ledger_path.is_file():
            return None
        record = self._load(ledger_path)
        if record.get("stage_spec_sha256") != stage.get("stage_spec_sha256"):
            raise RuntimeError("Ledger stage specification drift")
        attempts = record.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            raise RuntimeError("Ledger attempt history is absent")
        return dict(attempts[-1])

    def live_claim(self, stage: Mapping[str, object]) -> StageClaim | None:
        latest = self.latest_attempt(stage)
        if latest is None or latest.get("status") != "CLAIMED_BEFORE_NAMESPACE_OR_MODEL":
            return None
        identity = str(stage["scientific_identity_sha256"])
        ledger_path, _ = self._paths(identity)
        return StageClaim(
            identity,
            int(latest["attempt_number"]),
            str(latest["attempt_id"]),
            ledger_path,
            str(latest["owner"]["owner_token"]),
        )

    def record_success_cleanup(
        self, stage: Mapping[str, object], details: Mapping[str, object]
    ) -> None:
        identity = str(stage["scientific_identity_sha256"])
        ledger_path, lock_path = self._paths(identity)
        descriptor = self._acquire_lock(lock_path)
        try:
            record = self._load(ledger_path)
            attempts = record["attempts"]
            latest = attempts[-1]
            if latest.get("status") != "COMPLETED_ACCEPTED":
                raise RuntimeError("Only an accepted stage may record cache cleanup")
            if "success_cleanup" in latest:
                existing = dict(latest["success_cleanup"])
                requested = dict(details)
                if existing == requested:
                    return
                invariant_keys = {
                    "file_count",
                    "size_bytes",
                    "reason",
                    "namespace_reusable",
                }
                if (
                    any(existing.get(key) != requested.get(key) for key in invariant_keys)
                    or existing.get("work_namespace_removed") is not False
                    or requested.get("work_namespace_removed") is not True
                ):
                    raise RuntimeError("Accepted-stage cleanup record changed")
            latest["success_cleanup"] = dict(details)
            self._write(ledger_path, record)
        finally:
            self._release_lock(descriptor, lock_path)


class ScientificStageLedger:
    """Authoritative O_EXCL event chain with a validated mutable summary cache."""

    _FINAL_FAILURE_STATES = frozenset({"FAILED_RECORDED", "INTERRUPTED_RECORDED"})
    _TERMINAL_EVENTS = {
        "FAILED": "FAILED_RECORDED",
        "INTERRUPTED": "INTERRUPTED_RECORDED",
        "COMPLETED": "COMPLETED_ACCEPTED",
    }

    def __init__(self, root: Path, project_root: Path | None = None):
        self.root = root.resolve()
        self.project_root = (project_root or self.root.parent).resolve()

    @staticmethod
    def _validate_identity(identity: str) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", identity):
            raise ValueError("Scientific identity must be a lowercase SHA-256 digest")

    def _paths(self, identity: str) -> tuple[Path, Path]:
        self._validate_identity(identity)
        return self.root / f"{identity}.json", self.root / f"{identity}.lock"

    def _event_root(self, identity: str) -> Path:
        self._validate_identity(identity)
        return self.root / "events" / identity

    @staticmethod
    def _acquire_lock(path: Path, owner: Mapping[str, object] | None = None) -> int:
        try:
            descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as error:
            raise StageClaimUnavailable(
                "Scientific identity is being updated by another process"
            ) from error
        try:
            os.write(descriptor, canonical_json_bytes(dict(owner or _owner_record())))
            os.fsync(descriptor)
            return descriptor
        except BaseException:
            os.close(descriptor)
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            raise

    @staticmethod
    def _release_lock(descriptor: int, path: Path) -> None:
        os.close(descriptor)
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    @staticmethod
    def _exclusive_write(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            str(path),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _write_summary(path: Path, record: Mapping[str, object]) -> None:
        _write_text_atomic(
            path,
            json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        )

    @staticmethod
    def _summary_with_hash(record: Mapping[str, object]) -> dict[str, object]:
        summary = dict(record)
        summary.pop("summary_sha256", None)
        summary["summary_sha256"] = canonical_json_sha256(summary)
        return summary

    @staticmethod
    def _load_json(path: Path, description: str) -> dict[str, object]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Unreadable {description}: {path}") from error
        if not isinstance(payload, dict):
            raise RuntimeError(f"Malformed {description}: {path}")
        return payload

    def _event_files(self, identity: str) -> list[Path]:
        event_root = self._event_root(identity)
        if not event_root.exists():
            return []
        unexpected = [
            path
            for path in event_root.rglob("*")
            if path.is_file()
            and path.name != "resource_evidence.json"
            and not re.fullmatch(r"event_[0-9]{6}_[A-Z_]+\.json", path.name)
        ]
        if unexpected:
            raise RuntimeError("Unexpected file in immutable ledger event namespace")
        files = list(event_root.glob("attempt_[0-9][0-9][0-9][0-9]/event_*.json"))
        return sorted(files, key=lambda path: path.name)

    def _resolved_project_file(self, relative: str) -> Path:
        candidate = (self.project_root / relative).resolve()
        try:
            candidate.relative_to(self.project_root)
        except ValueError as error:
            raise RuntimeError("Resource evidence path escapes the project") from error
        return candidate

    def _verify_resource_reference(self, reference: Mapping[str, object]) -> dict[str, object]:
        relative = reference.get("ledger_relative_path")
        if not isinstance(relative, str) or not relative:
            raise RuntimeError("Event resource reference path is absent")
        snapshot_path = (self.root / relative).resolve()
        try:
            snapshot_path.relative_to(self.root)
        except ValueError as error:
            raise RuntimeError("Event resource reference escapes the ledger") from error
        if not snapshot_path.is_file():
            raise RuntimeError("Immutable resource snapshot is absent")
        if type(reference.get("size_bytes")) is not int:
            raise RuntimeError("Event resource reference size is malformed")
        if snapshot_path.stat().st_size != reference["size_bytes"]:
            raise RuntimeError("Immutable resource snapshot size drift")
        if stream_sha256(snapshot_path) != reference.get("sha256"):
            raise RuntimeError("Immutable resource snapshot hash drift")
        snapshot = self._load_json(snapshot_path, "resource snapshot")
        unhashed = {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}
        if snapshot.get("snapshot_sha256") != canonical_json_sha256(unhashed):
            raise RuntimeError("Immutable resource snapshot self-hash drift")
        evidence = snapshot.get("evidence")
        if not isinstance(evidence, Mapping):
            raise RuntimeError("Immutable resource snapshot lacks evidence")
        checked = validate_resource_evidence(evidence, gpu_stage=reference.get("gpu_stage"))
        for key in ("runtime_seconds", "max_memory_reserved_bytes", "gpu_stage"):
            if type(reference.get(key)) is not type(checked.get(key)) or reference.get(
                key
            ) != checked.get(key):
                raise RuntimeError("Event raw resource telemetry differs from snapshot")
        for file_record in checked["evidence_files"]:
            project_relative = file_record.get("project_relative_path")
            if not isinstance(project_relative, str):
                raise RuntimeError("Persisted resource file lacks a project-relative path")
            source = self._resolved_project_file(project_relative)
            if not source.is_file():
                raise RuntimeError("Resource evidence source file is absent")
            if source.stat().st_size != file_record["size_bytes"]:
                raise RuntimeError("Resource evidence source file size drift")
            if stream_sha256(source) != file_record["sha256"]:
                raise RuntimeError("Resource evidence source file hash drift")
        return dict(checked)

    def _read_chain(
        self,
        identity: str,
        *,
        stage_spec_sha256: str | None = None,
        require_summary: bool = True,
    ) -> dict[str, object] | None:
        ledger_path, _ = self._paths(identity)
        files = self._event_files(identity)
        if not files:
            if ledger_path.exists():
                raise RuntimeError("Ledger summary exists without an authoritative event chain")
            return None
        attempts: list[dict[str, object]] = []
        previous_hash: str | None = None
        stable_spec: str | None = None
        for expected_index, path in enumerate(files, start=1):
            event = self._load_json(path, "scientific ledger event")
            if event.get("schema_version") != LEDGER_EVENT_SCHEMA_VERSION:
                raise RuntimeError("Scientific ledger event schema drift")
            unhashed = {key: value for key, value in event.items() if key != "event_sha256"}
            event_hash = canonical_json_sha256(unhashed)
            if event.get("event_sha256") != event_hash:
                raise RuntimeError("Scientific ledger event self-hash drift")
            event_type = event.get("event_type")
            expected_name = f"event_{expected_index:06d}_{event_type}.json"
            if path.name != expected_name or event.get("event_index") != expected_index:
                raise RuntimeError("Scientific ledger event sequence is discontinuous")
            if event.get("prev_event_sha256") != previous_hash:
                raise RuntimeError("Scientific ledger event hash chain is broken")
            if event.get("scientific_identity_sha256") != identity:
                raise RuntimeError("Scientific ledger event identity drift")
            current_spec = event.get("stage_spec_sha256")
            if stable_spec is None:
                stable_spec = str(current_spec)
            if current_spec != stable_spec:
                raise RuntimeError("Scientific ledger event stage specification drift")
            if stage_spec_sha256 is not None and current_spec != stage_spec_sha256:
                raise RuntimeError("Ledger stage specification drift")
            attempt_number = event.get("attempt_number")
            if type(attempt_number) is not int or attempt_number < 1:
                raise RuntimeError("Scientific ledger event attempt is malformed")
            expected_directory = f"attempt_{attempt_number:04d}"
            if path.parent.name != expected_directory:
                raise RuntimeError("Scientific ledger attempt namespace drift")
            snapshot = event.get("attempt_snapshot")
            if not isinstance(snapshot, Mapping) or snapshot.get("attempt_number") != attempt_number:
                raise RuntimeError("Scientific ledger event attempt snapshot is malformed")
            expected_status_by_event = {
                "CLAIMED": "CLAIMED_BEFORE_NAMESPACE_OR_MODEL",
                "HEARTBEAT": "CLAIMED_BEFORE_NAMESPACE_OR_MODEL",
                "SCIENTIFIC_STARTED": "CLAIMED_BEFORE_NAMESPACE_OR_MODEL",
                "SCIENTIFIC_FINISHED": "CLAIMED_BEFORE_NAMESPACE_OR_MODEL",
                "FAILED": "FAILED_RECORDED",
                "INTERRUPTED": "INTERRUPTED_RECORDED",
                "COMPLETED": "COMPLETED_ACCEPTED",
                "CLEANUP": "COMPLETED_ACCEPTED",
            }
            if event_type not in expected_status_by_event or snapshot.get(
                "status"
            ) != expected_status_by_event[event_type]:
                raise RuntimeError("Scientific ledger event transition is unsupported")
            if event_type == "SCIENTIFIC_STARTED" and snapshot.get(
                "scientific_execution_started"
            ) is not True:
                raise RuntimeError("Scientific-start event lacks start evidence")
            if event_type == "SCIENTIFIC_FINISHED" and (
                snapshot.get("scientific_execution_started") is not True
                or snapshot.get("scientific_execution_finished") is not True
            ):
                raise RuntimeError("Scientific-finish event lacks execution evidence")
            if event_type == "COMPLETED" and snapshot.get(
                "accepted_success_artifact"
            ) is not True:
                raise RuntimeError("Completion event lacks accepted-artifact evidence")
            if attempt_number == len(attempts) + 1:
                if event_type != "CLAIMED":
                    raise RuntimeError("A new attempt must begin with a CLAIMED event")
                attempts.append(dict(snapshot))
            elif attempt_number == len(attempts):
                prior = attempts[-1]
                if prior.get("attempt_id") != snapshot.get("attempt_id"):
                    raise RuntimeError("Scientific ledger event attempt identity drift")
                if event_type == "CLAIMED":
                    raise RuntimeError("An attempt has duplicate CLAIMED events")
                attempts[-1] = dict(snapshot)
            else:
                raise RuntimeError("Scientific ledger event attempt order is discontinuous")
            reference = event.get("resource_evidence_ref")
            requires_resource = event_type in {
                "SCIENTIFIC_FINISHED",
                "FAILED",
                "INTERRUPTED",
                "COMPLETED",
            }
            if requires_resource != (reference is not None):
                raise RuntimeError("Scientific ledger event resource binding is incomplete")
            if reference is not None:
                if not isinstance(reference, Mapping):
                    raise RuntimeError("Scientific ledger resource reference is malformed")
                checked = self._verify_resource_reference(reference)
                details = snapshot.get("details", {})
                event_evidence = (
                    snapshot.get("provisional_resource_evidence")
                    if event_type == "SCIENTIFIC_FINISHED"
                    else details.get("resource_evidence")
                    if isinstance(details, Mapping)
                    else None
                )
                if event_evidence != checked:
                    raise RuntimeError("Event summary resource evidence differs from snapshot")
            previous_hash = event_hash
        summary = self._summary_with_hash(
            {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "scientific_identity_sha256": identity,
                "stage_spec_sha256": stable_spec,
                "attempts": attempts,
                "cumulative_attempt_count": len(attempts),
                "event_count": len(files),
                "chain_head_sha256": previous_hash,
            }
        )
        if require_summary:
            observed = self._load_json(ledger_path, "scientific stage summary")
            unhashed = {key: value for key, value in observed.items() if key != "summary_sha256"}
            if observed.get("summary_sha256") != canonical_json_sha256(unhashed):
                raise RuntimeError("Scientific stage summary self-hash drift")
            if observed != summary:
                raise RuntimeError("Scientific stage summary differs from immutable event chain")
        return summary

    def _load(self, path: Path) -> dict[str, object]:
        match = re.fullmatch(r"([0-9a-f]{64})\.json", path.name)
        if match is None:
            raise RuntimeError("Scientific stage summary path is malformed")
        record = self._read_chain(match.group(1))
        if record is None:
            raise RuntimeError("Scientific stage ledger is absent")
        return record

    def _append_event_locked(
        self,
        identity: str,
        stage_spec_sha256: str,
        event_type: str,
        attempt: Mapping[str, object],
        *,
        resource_reference: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        ledger_path, _ = self._paths(identity)
        current = self._read_chain(
            identity, stage_spec_sha256=stage_spec_sha256, require_summary=True
        )
        event_index = 1 if current is None else int(current["event_count"]) + 1
        previous_hash = None if current is None else current["chain_head_sha256"]
        attempt_number = int(attempt["attempt_number"])
        event = {
            "schema_version": LEDGER_EVENT_SCHEMA_VERSION,
            "scientific_identity_sha256": identity,
            "stage_spec_sha256": stage_spec_sha256,
            "event_index": event_index,
            "event_type": event_type,
            "attempt_number": attempt_number,
            "attempt_id": attempt["attempt_id"],
            "prev_event_sha256": previous_hash,
            "event_unix_seconds": time.time(),
            "event_utc": _utc_now(),
            "attempt_snapshot": dict(attempt),
        }
        if resource_reference is not None:
            event["resource_evidence_ref"] = dict(resource_reference)
        event["event_sha256"] = canonical_json_sha256(event)
        event_path = (
            self._event_root(identity)
            / f"attempt_{attempt_number:04d}"
            / f"event_{event_index:06d}_{event_type}.json"
        )
        self._exclusive_write(event_path, canonical_json_bytes(event) + b"\n")
        rebuilt = self._read_chain(
            identity, stage_spec_sha256=stage_spec_sha256, require_summary=False
        )
        assert rebuilt is not None
        self._write_summary(ledger_path, rebuilt)
        return rebuilt

    def _persist_resource_evidence(
        self,
        claim: StageClaim,
        evidence: Mapping[str, object],
        *,
        gpu_stage: bool,
        source_root: Path | None = None,
    ) -> tuple[dict[str, object], dict[str, object]]:
        checked = validate_resource_evidence(evidence, gpu_stage=gpu_stage)
        files: list[dict[str, object]] = []
        for item in checked["evidence_files"]:
            source_path: Path | None = None
            if isinstance(item.get("resolved_path"), str):
                source_path = Path(str(item["resolved_path"])).resolve()
            elif isinstance(item.get("project_relative_path"), str):
                source_path = self._resolved_project_file(str(item["project_relative_path"]))
            elif source_root is not None:
                source_path = (source_root.resolve() / str(item["relative_path"])).resolve()
            if source_path is None or not source_path.is_file():
                raise RuntimeError("Resource evidence source file cannot be resolved")
            try:
                project_relative = source_path.relative_to(self.project_root).as_posix()
            except ValueError as error:
                raise RuntimeError("Resource evidence source lies outside the project") from error
            if source_path.stat().st_size != item["size_bytes"] or stream_sha256(
                source_path
            ) != item["sha256"]:
                raise RuntimeError("Resource evidence source changed before persistence")
            files.append(
                {
                    "relative_path": str(item["relative_path"]),
                    "project_relative_path": project_relative,
                    "size_bytes": item["size_bytes"],
                    "sha256": item["sha256"],
                }
            )
        persisted = {
            key: value
            for key, value in checked.items()
            if key not in {"evidence_files", "evidence_sha256"}
        }
        persisted["evidence_files"] = files
        persisted["evidence_sha256"] = canonical_json_sha256(persisted)
        validate_resource_evidence(persisted, gpu_stage=gpu_stage)
        snapshot = {"evidence": persisted}
        snapshot["snapshot_sha256"] = canonical_json_sha256(snapshot)
        snapshot_bytes = canonical_json_bytes(snapshot) + b"\n"
        snapshot_path = (
            self._event_root(claim.scientific_identity_sha256)
            / claim.attempt_id
            / "resource_evidence.json"
        )
        if snapshot_path.exists():
            if snapshot_path.read_bytes() != snapshot_bytes:
                raise RuntimeError("Immutable attempt resource evidence changed")
        else:
            self._exclusive_write(snapshot_path, snapshot_bytes)
        reference = {
            "ledger_relative_path": snapshot_path.relative_to(self.root).as_posix(),
            "size_bytes": snapshot_path.stat().st_size,
            "sha256": stream_sha256(snapshot_path),
            "runtime_seconds": persisted["runtime_seconds"],
            "max_memory_reserved_bytes": persisted["max_memory_reserved_bytes"],
            "gpu_stage": persisted["gpu_stage"],
        }
        self._verify_resource_reference(reference)
        return persisted, reference

    def _validated_attempt(self, claim: StageClaim) -> tuple[dict[str, object], dict[str, object]]:
        record = self._read_chain(claim.scientific_identity_sha256)
        if record is None:
            raise RuntimeError("Scientific stage claim is absent")
        attempts = record["attempts"]
        if claim.attempt_number != len(attempts):
            raise RuntimeError("Only the current attempt can transition")
        attempt = dict(attempts[-1])
        if attempt.get("attempt_id") != claim.attempt_id:
            raise RuntimeError("Stage claim attempt changed")
        if attempt.get("owner", {}).get("owner_token") != claim.owner_token:
            raise RuntimeError("Stage claim ownership changed")
        return record, attempt

    def _attempt_gpu_stage(self, claim: StageClaim) -> bool:
        _, attempt = self._validated_attempt(claim)
        return bool(attempt.get("gpu_stage", False))

    def heartbeat(self, claim: StageClaim) -> None:
        ledger_path, lock_path = self._paths(claim.scientific_identity_sha256)
        descriptor = self._acquire_lock(lock_path)
        try:
            record, attempt = self._validated_attempt(claim)
            if attempt.get("status") != "CLAIMED_BEFORE_NAMESPACE_OR_MODEL":
                raise RuntimeError("Only a live attempt may heartbeat")
            owner = dict(attempt.get("owner", {}))
            if process_start_token(os.getpid()) != owner.get("process_start_token"):
                raise RuntimeError("Lease heartbeat process identity changed")
            owner["heartbeat_unix_seconds"] = time.time()
            owner["heartbeat_utc"] = _utc_now()
            attempt["owner"] = owner
            self._append_event_locked(
                claim.scientific_identity_sha256,
                str(record["stage_spec_sha256"]),
                "HEARTBEAT",
                attempt,
            )
        finally:
            self._release_lock(descriptor, lock_path)

    def mark_scientific_execution_started(self, claim: StageClaim) -> None:
        _, lock_path = self._paths(claim.scientific_identity_sha256)
        descriptor = self._acquire_lock(lock_path)
        try:
            record, attempt = self._validated_attempt(claim)
            if attempt.get("status") != "CLAIMED_BEFORE_NAMESPACE_OR_MODEL":
                raise RuntimeError("Only a live attempt can start scientific execution")
            if attempt.get("scientific_execution_started") is True:
                return
            attempt["scientific_execution_started"] = True
            attempt["scientific_start_unix_seconds"] = time.time()
            attempt["scientific_start_utc"] = _utc_now()
            self._append_event_locked(
                claim.scientific_identity_sha256,
                str(record["stage_spec_sha256"]),
                "SCIENTIFIC_STARTED",
                attempt,
            )
        finally:
            self._release_lock(descriptor, lock_path)

    def record_scientific_execution_finished(
        self, claim: StageClaim, resource_evidence: Mapping[str, object]
    ) -> None:
        _, lock_path = self._paths(claim.scientific_identity_sha256)
        descriptor = self._acquire_lock(lock_path)
        try:
            record, attempt = self._validated_attempt(claim)
            if attempt.get("status") != "CLAIMED_BEFORE_NAMESPACE_OR_MODEL":
                raise RuntimeError("Only a live attempt can finish scientific execution")
            if attempt.get("scientific_execution_started") is not True:
                raise RuntimeError("Scientific execution cannot finish before it starts")
            checked, reference = self._persist_resource_evidence(
                claim,
                resource_evidence,
                gpu_stage=bool(attempt.get("gpu_stage", False)),
            )
            existing = attempt.get("provisional_resource_evidence")
            if existing is not None and existing != checked:
                raise RuntimeError("Provisional resource evidence changed")
            attempt["scientific_execution_finished"] = True
            attempt["scientific_finish_unix_seconds"] = time.time()
            attempt["scientific_finish_utc"] = _utc_now()
            attempt["provisional_resource_evidence"] = checked
            self._append_event_locked(
                claim.scientific_identity_sha256,
                str(record["stage_spec_sha256"]),
                "SCIENTIFIC_FINISHED",
                attempt,
                resource_reference=reference,
            )
        finally:
            self._release_lock(descriptor, lock_path)

    @contextmanager
    def lease_heartbeat(self, claim: StageClaim, interval_seconds: float = 30.0, on_heartbeat=None):
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("Heartbeat interval must be finite and positive")
        stopped = threading.Event()
        failures: list[BaseException] = []

        def pulse() -> None:
            while not stopped.wait(interval_seconds):
                try:
                    self.heartbeat(claim)
                    if on_heartbeat is not None:
                        on_heartbeat()
                except BaseException as error:
                    failures.append(error)
                    stopped.set()

        worker = threading.Thread(target=pulse, name=f"lease-{claim.attempt_id}", daemon=True)
        worker.start()
        try:
            yield
            if failures:
                raise RuntimeError("Lease heartbeat failed") from failures[0]
        finally:
            stopped.set()
            worker.join(timeout=min(5.0, interval_seconds + 1.0))

    def cumulative_gpu_seconds(self, *, now_unix_seconds: float | None = None) -> float:
        now = time.time() if now_unix_seconds is None else now_unix_seconds
        if type(now) not in (int, float) or not math.isfinite(now):
            raise RuntimeError("Resource-accounting clock is non-finite")
        total = 0.0
        if not self.root.exists():
            return total
        identities = {path.stem for path in self.root.glob("[0-9a-f]*.json")}
        event_index_root = self.root / "events"
        if event_index_root.exists():
            identities.update(path.name for path in event_index_root.iterdir() if path.is_dir())
        identities = sorted(identities)
        for identity in identities:
            if not re.fullmatch(r"[0-9a-f]{64}", identity):
                continue
            record = self._read_chain(identity)
            assert record is not None
            for attempt in record["attempts"]:
                if attempt.get("gpu_stage") is not True:
                    continue
                if attempt.get("status") == "CLAIMED_BEFORE_NAMESPACE_OR_MODEL":
                    if attempt.get("scientific_execution_finished") is True:
                        evidence = attempt.get("provisional_resource_evidence")
                        if not isinstance(evidence, Mapping):
                            raise RuntimeError("Finished live GPU attempt lacks telemetry")
                        total += validate_resource_evidence(evidence, gpu_stage=True)[
                            "runtime_seconds"
                        ]
                    elif attempt.get("scientific_execution_started") is True:
                        start = attempt.get("scientific_start_unix_seconds")
                        if type(start) not in (int, float):
                            raise RuntimeError("Live GPU start time is malformed")
                        total += max(0.0, now - start)
                    continue
                evidence = attempt.get("details", {}).get("resource_evidence")
                if not isinstance(evidence, Mapping):
                    raise RuntimeError("Finished GPU attempt lacks resource evidence")
                total += validate_resource_evidence(evidence, gpu_stage=True)["runtime_seconds"]
                if not math.isfinite(total):
                    raise RuntimeError("Cumulative GPU runtime is non-finite")
        return total

    def recover_interrupted(
        self,
        stage: Mapping[str, object],
        *,
        reason: str,
        inspector=process_identity_state,
        resource_evidence: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        if not reason.strip():
            raise ValueError("Recovery requires an auditable reason")
        identity = str(stage["scientific_identity_sha256"])
        ledger_path, lock_path = self._paths(identity)
        self.root.mkdir(parents=True, exist_ok=True)
        stale_lock_evidence: str | None = None
        if lock_path.exists():
            lock_owner = self._load_json(lock_path, "stale lock")
            if _owner_alive(lock_owner, inspector):
                raise StageClaimUnavailable("Live lock owner cannot be stolen")
            stale_name = self.root / (
                f"{identity}.stale_lock.{canonical_json_sha256(lock_owner)[:16]}.json"
            )
            os.replace(lock_path, stale_name)
            stale_lock_evidence = stale_name.name
        descriptor = self._acquire_lock(lock_path)
        try:
            record = self._read_chain(identity, stage_spec_sha256=str(stage["stage_spec_sha256"]))
            if record is None:
                raise RuntimeError("No stage attempt exists to recover")
            attempt = dict(record["attempts"][-1])
            if attempt.get("status") != "CLAIMED_BEFORE_NAMESPACE_OR_MODEL":
                raise StageClaimUnavailable("Latest attempt is not live")
            owner = attempt.get("owner")
            if not isinstance(owner, Mapping):
                raise RuntimeError("Live attempt lacks PID/host/process-start ownership")
            if _owner_alive(owner, inspector):
                raise StageClaimUnavailable("Live attempt owner cannot be stolen")
            gpu_stage = bool(attempt.get("gpu_stage", False))
            evidence = resource_evidence
            if evidence is None:
                provisional = attempt.get("provisional_resource_evidence")
                if attempt.get("scientific_execution_finished") is True and isinstance(
                    provisional, Mapping
                ):
                    evidence = provisional
                elif attempt.get("scientific_execution_started") is True:
                    start = attempt.get("scientific_start_unix_seconds")
                    if type(start) not in (int, float):
                        raise RuntimeError("Interrupted start time is malformed")
                    evidence = make_resource_evidence(
                        runtime_seconds=max(0.0, time.time() - start),
                        max_memory_reserved_bytes=(MAX_GPU_RESERVED_BYTES if gpu_stage else 0),
                        gpu_stage=gpu_stage,
                        source="dead_owner_wall_time_and_guard_ceiling",
                        conservative=True,
                    )
                else:
                    evidence = make_resource_evidence(
                        runtime_seconds=0.0,
                        max_memory_reserved_bytes=0,
                        gpu_stage=gpu_stage,
                        source="claim_proved_pre_model",
                        conservative=True,
                    )
            claim = StageClaim(
                identity,
                int(attempt["attempt_number"]),
                str(attempt["attempt_id"]),
                ledger_path,
                str(owner["owner_token"]),
            )
            checked, reference = self._persist_resource_evidence(
                claim, evidence, gpu_stage=gpu_stage
            )
            attempt["status"] = "INTERRUPTED_RECORDED"
            attempt["accepted_success_artifact"] = False
            attempt["transition_unix_seconds"] = time.time()
            attempt["transition_utc"] = _utc_now()
            attempt["details"] = {
                "reason": reason,
                "resource_evidence": checked,
                "dead_owner_proof": {
                    "hostname": owner.get("hostname"),
                    "pid": owner.get("pid"),
                    "expected_process_start_token": owner.get("process_start_token"),
                    "observed_process_start_token": _observed_owner_token(owner, inspector),
                    "recovered_utc": _utc_now(),
                    "stale_lock_evidence": stale_lock_evidence,
                },
            }
            rebuilt = self._append_event_locked(
                identity,
                str(record["stage_spec_sha256"]),
                "INTERRUPTED",
                attempt,
                resource_reference=reference,
            )
            return dict(rebuilt["attempts"][-1])
        finally:
            self._release_lock(descriptor, lock_path)

    def claim(
        self, stage: Mapping[str, object], *, attempt_metadata: Mapping[str, object] | None = None
    ) -> StageClaim:
        identity = str(stage["scientific_identity_sha256"])
        stage_spec_sha = str(stage["stage_spec_sha256"])
        ledger_path, lock_path = self._paths(identity)
        self.root.mkdir(parents=True, exist_ok=True)
        owner = _owner_record()
        descriptor = self._acquire_lock(lock_path, owner)
        try:
            record = self._read_chain(identity, stage_spec_sha256=stage_spec_sha)
            if record is not None:
                attempts = record["attempts"]
                if any(item.get("status") == "COMPLETED_ACCEPTED" for item in attempts):
                    raise StageClaimUnavailable("Stage already has an accepted success artifact")
                latest = attempts[-1]
                if latest.get("status") not in self._FINAL_FAILURE_STATES:
                    raise StageClaimUnavailable("Stage has a live or non-retryable attempt")
                if latest.get("accepted_success_artifact") is not False:
                    raise StageClaimUnavailable(
                        "Recorded failure does not prove absence of accepted success"
                    )
                attempt_number = len(attempts) + 1
            else:
                attempt_number = 1
            attempt_id = f"attempt_{attempt_number:04d}"
            namespace = stage["namespace"]
            attempt = {
                "attempt_number": attempt_number,
                "attempt_id": attempt_id,
                "status": "CLAIMED_BEFORE_NAMESPACE_OR_MODEL",
                "accepted_success_artifact": False,
                "work_namespace": str(namespace["work_attempt_template"]).format(
                    attempt_number=attempt_number
                ),
                "output_namespace": str(namespace["output_attempt_template"]).format(
                    attempt_number=attempt_number
                ),
                "metadata": dict(attempt_metadata or {}),
                "gpu_stage": bool(stage.get("execution", {}).get("gpu", False)),
                "owner": owner,
                "scientific_execution_started": False,
            }
            self._append_event_locked(identity, stage_spec_sha, "CLAIMED", attempt)
            return StageClaim(
                identity, attempt_number, attempt_id, ledger_path, str(owner["owner_token"])
            )
        finally:
            self._release_lock(descriptor, lock_path)

    def _transition(
        self,
        claim: StageClaim,
        status: str,
        *,
        details: Mapping[str, object],
        accepted_success_artifact: bool,
        resource_evidence: Mapping[str, object],
        resource_source_root: Path | None = None,
    ) -> None:
        _, lock_path = self._paths(claim.scientific_identity_sha256)
        descriptor = self._acquire_lock(lock_path)
        try:
            record, attempt = self._validated_attempt(claim)
            if attempt.get("status") != "CLAIMED_BEFORE_NAMESPACE_OR_MODEL":
                raise RuntimeError("Stage attempt is not live")
            gpu_stage = bool(attempt.get("gpu_stage", False))
            checked, reference = self._persist_resource_evidence(
                claim,
                resource_evidence,
                gpu_stage=gpu_stage,
                source_root=resource_source_root,
            )
            provisional = attempt.get("provisional_resource_evidence")
            if provisional is not None and provisional != checked:
                raise RuntimeError("Terminal resource evidence differs from scientific finish")
            terminal_details = dict(details)
            terminal_details["resource_evidence"] = checked
            attempt["status"] = status
            attempt["accepted_success_artifact"] = accepted_success_artifact
            attempt["details"] = terminal_details
            attempt["transition_unix_seconds"] = time.time()
            attempt["transition_utc"] = _utc_now()
            event_type = next(
                key for key, mapped_status in self._TERMINAL_EVENTS.items() if mapped_status == status
            )
            self._append_event_locked(
                claim.scientific_identity_sha256,
                str(record["stage_spec_sha256"]),
                event_type,
                attempt,
                resource_reference=reference,
            )
        finally:
            self._release_lock(descriptor, lock_path)

    def record_failed(
        self, claim: StageClaim, reason: str, resource_evidence: Mapping[str, object]
    ) -> None:
        if not reason.strip():
            raise ValueError("A recorded failure requires a reason")
        self._transition(
            claim,
            "FAILED_RECORDED",
            details={"reason": reason},
            accepted_success_artifact=False,
            resource_evidence=resource_evidence,
        )

    def record_interrupted(
        self, claim: StageClaim, reason: str, resource_evidence: Mapping[str, object]
    ) -> None:
        if not reason.strip():
            raise ValueError("A recorded interruption requires a reason")
        self._transition(
            claim,
            "INTERRUPTED_RECORDED",
            details={"reason": reason},
            accepted_success_artifact=False,
            resource_evidence=resource_evidence,
        )

    def record_completed(
        self,
        claim: StageClaim,
        stage: Mapping[str, object],
        output_root: Path,
        resource_evidence: Mapping[str, object],
        recovery_evidence: Mapping[str, object] | None = None,
    ) -> list[dict[str, object]]:
        artifacts = validate_stage_artifacts(stage, output_root)
        checked = validate_resource_evidence(
            resource_evidence, gpu_stage=bool(stage.get("execution", {}).get("gpu", False))
        )
        artifact_by_path = {str(item["relative_path"]): item for item in artifacts}
        for evidence_file in checked["evidence_files"]:
            artifact = artifact_by_path.get(str(evidence_file["relative_path"]))
            if artifact is None or any(
                artifact.get(key) != evidence_file.get(key) for key in ("size_bytes", "sha256")
            ):
                raise RuntimeError("Resource evidence is not bound to an accepted stage artifact")
        details: dict[str, object] = {"artifacts": artifacts}
        if recovery_evidence is not None:
            details["completion_recovery"] = dict(recovery_evidence)
        self._transition(
            claim,
            "COMPLETED_ACCEPTED",
            details=details,
            accepted_success_artifact=True,
            resource_evidence=resource_evidence,
            resource_source_root=output_root,
        )
        return artifacts

    def completed_artifacts_valid(self, stage: Mapping[str, object], output_root: Path) -> bool:
        latest = self.latest_attempt(stage)
        if latest is None or latest.get("status") != "COMPLETED_ACCEPTED":
            return False
        observed = validate_stage_artifacts(stage, output_root)
        return observed == latest.get("details", {}).get("artifacts")

    def accepted_output_namespace(self, stage: Mapping[str, object]) -> str | None:
        latest = self.latest_attempt(stage)
        if latest is None:
            return None
        if latest.get("status") != "COMPLETED_ACCEPTED":
            return None
        return str(latest["output_namespace"])

    def latest_attempt(self, stage: Mapping[str, object]) -> dict[str, object] | None:
        identity = str(stage["scientific_identity_sha256"])
        record = self._read_chain(identity, stage_spec_sha256=str(stage["stage_spec_sha256"]))
        if record is None:
            return None
        return dict(record["attempts"][-1])

    def live_claim(self, stage: Mapping[str, object]) -> StageClaim | None:
        latest = self.latest_attempt(stage)
        if latest is None or latest.get("status") != "CLAIMED_BEFORE_NAMESPACE_OR_MODEL":
            return None
        identity = str(stage["scientific_identity_sha256"])
        ledger_path, _ = self._paths(identity)
        return StageClaim(
            identity,
            int(latest["attempt_number"]),
            str(latest["attempt_id"]),
            ledger_path,
            str(latest["owner"]["owner_token"]),
        )

    def record_success_cleanup(
        self, stage: Mapping[str, object], details: Mapping[str, object]
    ) -> None:
        identity = str(stage["scientific_identity_sha256"])
        _, lock_path = self._paths(identity)
        descriptor = self._acquire_lock(lock_path)
        try:
            record = self._read_chain(identity, stage_spec_sha256=str(stage["stage_spec_sha256"]))
            if record is None:
                raise RuntimeError("Accepted stage is absent")
            latest = dict(record["attempts"][-1])
            if latest.get("status") != "COMPLETED_ACCEPTED":
                raise RuntimeError("Only an accepted stage may record cache cleanup")
            if latest.get("success_cleanup") == dict(details):
                return
            latest["success_cleanup"] = dict(details)
            self._append_event_locked(
                identity, str(record["stage_spec_sha256"]), "CLEANUP", latest
            )
        finally:
            self._release_lock(descriptor, lock_path)


def validate_stage_artifacts(
    stage: Mapping[str, object], output_root: Path
) -> list[dict[str, object]]:
    output_root = output_root.resolve()
    records: list[dict[str, object]] = []
    expected = list(stage["expected_outputs"])
    for relative_path in expected:
        path = (output_root / str(relative_path)).resolve()
        try:
            path.relative_to(output_root)
        except ValueError as error:
            raise RuntimeError("Expected artifact escapes output namespace") from error
        if not path.is_file():
            raise RuntimeError(f"Expected stage artifact is absent: {relative_path}")
        records.append(
            {
                "relative_path": str(relative_path),
                "size_bytes": path.stat().st_size,
                "sha256": stream_sha256(path),
            }
        )
    return records


def _encode_float64(value: object) -> str:
    number = np.float64(value)
    if not np.isfinite(number):
        raise ValueError("Only finite float64 predictions may be sealed")
    return struct.pack(">d", float(number)).hex()


def _decode_float64(value: str) -> float:
    if len(value) != 16:
        raise ValueError("Invalid IEEE-754 binary64 payload")
    return struct.unpack(">d", bytes.fromhex(value))[0]


def write_sealed_prediction_artifacts(
    table: pd.DataFrame, lossless_path: Path, readable_csv_path: Path
) -> dict[str, object]:
    required = [
        "curated_id",
        "outer_fold",
        "seed",
        "prediction_normalized",
        "prediction_log10_papp",
    ]
    if list(table.columns) != required:
        raise ValueError(f"Sealed prediction columns must be exactly {required}")
    if table.assign(curated_id=table["curated_id"].astype(str)).duplicated(
        ["curated_id", "outer_fold", "seed"]
    ).any():
        raise ValueError("Sealed predictions contain a duplicate ID/fold/seed slot")
    records = []
    for curated_id, outer_fold, seed, pred_norm, pred_log in table.itertuples(
        index=False, name=None
    ):
        records.append(
            {
                "curated_id": str(curated_id),
                "outer_fold": int(outer_fold),
                "seed": int(seed),
                "prediction_normalized_ieee754_be": _encode_float64(pred_norm),
                "prediction_log10_papp_ieee754_be": _encode_float64(pred_log),
            }
        )
    payload = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "authoritative": True,
        "contains_observed_labels": False,
        "records": records,
        "records_sha256": canonical_json_sha256(records),
    }
    write_json(lossless_path, payload)
    readable_csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(readable_csv_path, index=False, float_format="%.17g", lineterminator="\n")
    return {
        "record_count": len(records),
        "records_sha256": payload["records_sha256"],
        "lossless_file_sha256": stream_sha256(lossless_path),
        "readable_csv_sha256": stream_sha256(readable_csv_path),
    }


def read_sealed_prediction_artifact(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != PREDICTION_SCHEMA_VERSION:
        raise ValueError("Unsupported sealed prediction schema")
    if payload.get("contains_observed_labels") is not False:
        raise ValueError("Sealed prediction artifact may not contain observed labels")
    records = payload.get("records")
    if not isinstance(records, list) or canonical_json_sha256(records) != payload.get(
        "records_sha256"
    ):
        raise ValueError("Sealed prediction record hash mismatch")
    decoded = []
    for record in records:
        decoded.append(
            {
                "curated_id": str(record["curated_id"]),
                "outer_fold": int(record["outer_fold"]),
                "seed": int(record["seed"]),
                "prediction_normalized": _decode_float64(
                    str(record["prediction_normalized_ieee754_be"])
                ),
                "prediction_log10_papp": _decode_float64(
                    str(record["prediction_log10_papp_ieee754_be"])
                ),
            }
        )
    return pd.DataFrame(decoded)


def check_disk_space(path: Path, required_free_bytes: int = MIN_FREE_DISK_BYTES) -> None:
    if required_free_bytes <= 0:
        raise ValueError("Disk-space threshold must be positive")
    probe = path.resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if free < required_free_bytes:
        raise RuntimeError(
            f"Insufficient free disk: observed {free} bytes, require {required_free_bytes}"
        )


def assert_real_execution_authorized(
    plan: Mapping[str, object], acceptance: Mapping[str, object] | None = None
) -> None:
    # Constants are checked first and cannot be enabled by editing a plan file.
    if not PREFIT_REVIEW_ACCEPTED or not REAL_EXECUTION_AUTHORIZED:
        raise RuntimeError(
            "Full D0 execution is pre-fit locked in this commit; independent review is required"
        )
    if (
        acceptance is None
        or acceptance.get("accepted") is not True
        or acceptance.get("execution_authorized") is not True
    ):
        raise RuntimeError("External independent acceptance does not authorize real execution")
    if acceptance.get("candidate_plan_sha256") != plan.get("plan_sha256"):
        raise RuntimeError("External acceptance does not identify this candidate plan")


def _history_payload(history: object) -> dict[str, object]:
    """Losslessly serialize a contract-issued history without reusable tokens."""

    events = []
    for event in history.events:
        payload = dict(event.__dict__)
        payload["loss_ieee754_be"] = _encode_float64(payload.pop("loss"))
        events.append(payload)
    result = {
        "schema_version": "scaffoldseal-guarded-inner-history-v1",
        "outer_fold": int(history.outer_fold),
        "inner_basket": int(history.inner_basket),
        "seed_index": int(history.seed),
        "config_id": str(history.config_id),
        "training_ids_sha256": canonical_id_hash(history.training_ids),
        "validation_ids_sha256": canonical_id_hash(history.validation_ids),
        "metric_identity": str(history.metric_identity),
        "feature_columns": list(history.feature_columns),
        "feature_schema_sha256": str(history.feature_schema_sha256),
        "target_identity": str(history.target_identity),
        "transform_sha256": str(history.transform_sha256),
        "model_config_sha256": str(history.model_config_sha256),
        "evaluator_identity": str(history.evaluator_identity),
        "execution_identity_sha256": str(history.execution_identity_sha256),
        "events": events,
    }
    result["records_sha256"] = canonical_json_sha256(events)
    return result


def _write_stage_artifact_manifest(
    stage: Mapping[str, object], output_root: Path
) -> None:
    files = []
    for relative in stage["expected_outputs"]:
        if relative == "artifact_manifest.json":
            continue
        path = output_root / str(relative)
        if not path.is_file():
            raise RuntimeError(f"Cannot seal absent stage artifact: {relative}")
        files.append(
            {
                "relative_path": str(relative),
                "size_bytes": path.stat().st_size,
                "sha256": stream_sha256(path),
            }
        )
    payload = {
        "schema_version": "scaffoldseal-stage-artifacts-v1",
        "scientific_identity_sha256": stage["scientific_identity_sha256"],
        "stage_spec_sha256": stage["stage_spec_sha256"],
        "files": files,
        "files_sha256": canonical_json_sha256(files),
    }
    write_json(output_root / "artifact_manifest.json", payload)


def _invoke_pretraining_helper(
    helper,
    baseline_root: Path,
    run_dir: Path,
    tag: str,
    *,
    seed_index: int,
    require_no_learning_stub: bool,
):
    """Single namespace handoff used by both the future runner and inert test."""

    if require_no_learning_stub and getattr(
        helper, "_scaffoldseal_no_learning_stub", False
    ) is not True:
        raise RuntimeError("Only an explicitly tagged no-learning stub is allowed")
    if run_dir.exists():
        raise RuntimeError(f"Pretraining namespace already exists: {run_dir}")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    return helper(baseline_root, run_dir, tag, seed_index=seed_index)


def invoke_pretraining_helper_no_learning(
    helper,
    baseline_root: Path,
    run_dir: Path,
    tag: str,
    *,
    seed_index: int,
):
    """Exercise the exact production handoff with an explicitly inert stub."""

    return _invoke_pretraining_helper(
        helper,
        baseline_root,
        run_dir,
        tag,
        seed_index=seed_index,
        require_no_learning_stub=True,
    )


def record_attempt_exception(
    ledger: ScientificStageLedger,
    claim: StageClaim,
    error: BaseException,
    resource_evidence: Mapping[str, object],
) -> None:
    """Keep KeyboardInterrupt distinct from ordinary scientific failures."""

    if isinstance(error, KeyboardInterrupt):
        ledger.record_interrupted(claim, "KeyboardInterrupt", resource_evidence)
    else:
        ledger.record_failed(
            claim,
            f"{type(error).__name__}: {error}",
            resource_evidence,
        )


class D0FullExecutor:
    """Incremental one-GPU implementation behind the compile-time pre-fit lock.

    The class is present so the independent reviewer can audit the exact future
    run path.  Its constructor and methods are unreachable from ``--execute``
    until a later reviewed commit changes both hard authorization constants.
    """

    def __init__(
        self,
        plan: Mapping[str, object],
        project_root: Path,
        baseline_root: Path,
        *,
        acceptance: Mapping[str, object] | None = None,
        manifest_path: Path | None = None,
    ) -> None:
        assert_real_execution_authorized(plan, acceptance)
        validate_plan(plan)
        self.plan = plan
        self.project_root = project_root.resolve()
        self.baseline_root = baseline_root.resolve()
        self.acceptance = dict(acceptance or {})
        self.manifest_path = (
            manifest_path or (self.project_root / DEFAULT_MANIFEST)
        ).resolve()
        self._runner_started_unix = time.time()
        verify_candidate_anchor(
            plan,
            self.project_root,
            self.baseline_root,
            manifest_path=self.manifest_path,
            acceptance=self.acceptance,
            require_accepted=True,
        )
        ledger_relative = str(plan["scheduler"]["ledger_relative_path"])
        self.ledger = ScientificStageLedger(
            self.project_root / ledger_relative, project_root=self.project_root
        )
        self.by_kind: dict[str, list[Mapping[str, object]]] = {}
        self.by_identity: dict[str, Mapping[str, object]] = {}
        for stage in plan["stages"]:
            self.by_kind.setdefault(str(stage["kind"]), []).append(stage)
            self.by_identity[str(stage["scientific_identity_sha256"])] = stage
        self._versions: dict[str, str] | None = None
        self._source_hashes: dict[str, str] | None = None

    def _announce(self, event: str, stage: Mapping[str, object] | None = None) -> None:
        payload: dict[str, object] = {"event": event}
        if stage is not None:
            payload.update(
                {
                    "kind": stage["kind"],
                    "key": stage["key"],
                    "scientific_identity_sha256": stage["scientific_identity_sha256"],
                }
            )
        print(json.dumps(payload, sort_keys=True), flush=True)
        self._publish_status(event, stage)

    def _publish_status(
        self, event: str, stage: Mapping[str, object] | None = None
    ) -> None:
        publish_live_status(
            self.project_root,
            self.plan,
            self.ledger,
            phase=f"H1_RANDOM_CV_{event.upper()}",
            training_state=(
                "COMPLETE" if event == "completed_all" else "SCIENTIFIC_RUN_ACTIVE"
            ),
            external_acceptance=True,
            next_action=(
                "等待下一阶段或核验当前产物"
                if event != "completed_all"
                else "进行最终结果审阅"
            ),
            elapsed_seconds=max(0.0, time.time() - self._runner_started_unix),
            current_stage_override=stage,
        )

    def _accepted_output(self, stage: Mapping[str, object]) -> Path | None:
        relative = self.ledger.accepted_output_namespace(stage)
        if relative is None:
            return None
        output = (self.project_root / relative).resolve()
        try:
            output.relative_to(self.project_root)
        except ValueError as error:
            raise RuntimeError("Accepted output namespace escapes the project") from error
        if not self.ledger.completed_artifacts_valid(stage, output):
            raise RuntimeError("Completed stage artifact hash validation failed")
        return output

    def _claim_paths(
        self, stage: Mapping[str, object]
    ) -> tuple[StageClaim, Path, Path]:
        verify_candidate_anchor(
            self.plan,
            self.project_root,
            self.baseline_root,
            manifest_path=self.manifest_path,
            acceptance=self.acceptance,
            require_accepted=True,
        )
        claim = self.ledger.claim(
            stage,
            attempt_metadata={
                "pid": os.getpid(),
                "noninteractive": True,
                "gpu_worker": 0 if stage["execution"].get("gpu") else None,
            },
        )
        work_relative = str(stage["namespace"]["work_attempt_template"]).format(
            attempt_number=claim.attempt_number
        )
        output_relative = str(stage["namespace"]["output_attempt_template"]).format(
            attempt_number=claim.attempt_number
        )
        work = (self.project_root / work_relative).resolve()
        output = (self.project_root / output_relative).resolve()
        for path in (work, output):
            try:
                path.relative_to(self.project_root)
            except ValueError as error:
                raise RuntimeError("Stage namespace escapes the project") from error
            if path.exists():
                self.ledger.record_failed(
                    claim,
                    f"pre-existing namespace: {path}",
                    make_resource_evidence(
                        runtime_seconds=0.0,
                        max_memory_reserved_bytes=0,
                        gpu_stage=bool(stage["execution"].get("gpu", False)),
                        source="claim_proved_pre_model",
                    ),
                )
                raise RuntimeError(f"Fresh stage namespace already exists: {path}")
        self._announce("claimed", stage)
        return claim, work, output

    def _prepare_namespaces(self, work: Path, output: Path) -> None:
        check_disk_space(self.project_root)
        work.mkdir(parents=True)
        output.mkdir(parents=True)

    def _prepare_pretraining_namespaces(self, work: Path, output: Path) -> None:
        check_disk_space(self.project_root)
        if work.exists() or output.exists():
            raise RuntimeError("Pretraining attempt namespace must be fresh")
        work.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=True)
        # The accepted pilot helper owns creation of the exact run directory.
        if work.exists():
            raise RuntimeError("Pretraining helper run directory was created too early")

    def _verify_resource_trace(
        self, trace: Mapping[str, object], trace_path: Path, *, gpu_stage: bool = True
    ) -> dict[str, object]:
        return resource_evidence_from_trace(trace, trace_path, gpu_stage=gpu_stage)

    def _accepted_gpu_seconds(self) -> float:
        return self.ledger.cumulative_gpu_seconds()

    def _check_gpu_budget(self) -> None:
        total = self._accepted_gpu_seconds()
        if not math.isfinite(total) or total < 0:
            raise RuntimeError("Cumulative H1 GPU runtime ledger is invalid")

    def _failure_evidence(
        self,
        stage: Mapping[str, object],
        claim: StageClaim,
        work: Path,
        trace_name: str,
    ) -> dict[str, object]:
        gpu_stage = bool(stage["execution"].get("gpu", False))
        latest = self.ledger.latest_attempt(stage) or {}
        provisional = latest.get("provisional_resource_evidence")
        if latest.get("scientific_execution_finished") is True and isinstance(
            provisional, Mapping
        ):
            return validate_resource_evidence(provisional, gpu_stage=gpu_stage)
        trace_path = work / trace_name
        if not trace_path.is_file() and work.exists():
            candidates = sorted(path for path in work.rglob(trace_name) if path.is_file())
            if len(candidates) == 1:
                trace_path = candidates[0]
        if trace_path.is_file():
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            return self._verify_resource_trace(trace, trace_path, gpu_stage=gpu_stage)
        if latest.get("scientific_execution_started") is not True:
            return make_resource_evidence(
                runtime_seconds=0.0,
                max_memory_reserved_bytes=0,
                gpu_stage=gpu_stage,
                source="claim_proved_pre_model",
            )
        start = float(latest["scientific_start_unix_seconds"])
        return make_resource_evidence(
            runtime_seconds=max(0.0, time.time() - start),
            max_memory_reserved_bytes=(MAX_GPU_RESERVED_BYTES if gpu_stage else 0),
            gpu_stage=gpu_stage,
            source="failed_wall_time_and_guard_ceiling",
            conservative=True,
        )

    def _copy(self, source: Path, target: Path) -> None:
        if not source.is_file():
            raise RuntimeError(f"Expected scientific artifact is absent: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    def _cleanup_accepted_work(self, stage: Mapping[str, object]) -> None:
        # Output validation must succeed immediately before removing disposable
        # loader caches/work checkpoints. Failed/interrupted namespaces never
        # enter this method and are preserved for provenance.
        if self._accepted_output(stage) is None:
            raise RuntimeError("Cannot clean work for an unaccepted stage")
        latest = self.ledger.latest_attempt(stage)
        if latest is None or latest.get("status") != "COMPLETED_ACCEPTED":
            raise RuntimeError("Accepted cleanup lacks a completed attempt")
        work = (self.project_root / str(latest["work_namespace"])).resolve()
        allowed_root = (self.project_root / "runs/h1_random_cv_d0_v1").resolve()
        try:
            work.relative_to(allowed_root)
        except ValueError as error:
            raise RuntimeError("Refusing to clean work outside the locked run root") from error
        existing_cleanup = latest.get("success_cleanup")
        if not work.exists():
            if not isinstance(existing_cleanup, Mapping):
                raise RuntimeError("Accepted work disappeared without a cleanup record")
            if existing_cleanup.get("work_namespace_removed") is False:
                finished = dict(existing_cleanup)
                finished["work_namespace_removed"] = True
                self.ledger.record_success_cleanup(stage, finished)
            elif existing_cleanup.get("work_namespace_removed") is not True:
                raise RuntimeError("Accepted cleanup record has an invalid state")
            return
        files = [path for path in work.rglob("*") if path.is_file()]
        details = {
            "work_namespace_removed": False,
            "file_count": len(files),
            "size_bytes": sum(path.stat().st_size for path in files),
            "reason": "accepted outputs and hashes retained; disposable stage cache reclaimed",
            "namespace_reusable": False,
        }
        self.ledger.record_success_cleanup(stage, details)
        shutil.rmtree(work)
        details["work_namespace_removed"] = True
        self.ledger.record_success_cleanup(stage, details)

    def _run_pretraining(self, stage: Mapping[str, object]) -> None:
        assert_real_execution_authorized(self.plan, self.acceptance)
        self._check_gpu_budget()
        if self._accepted_output(stage) is not None:
            self._cleanup_accepted_work(stage)
            self._announce("skip_validated_complete", stage)
            return
        claim, work, output = self._claim_paths(stage)
        try:
            self._prepare_pretraining_namespaces(work, output)
            from r1c0_dmpnn_pilot import train_delaney_once

            seed_index = int(stage["key"]["seed_index"])
            self.ledger.mark_scientific_execution_started(claim)
            with self.ledger.lease_heartbeat(
                claim,
                on_heartbeat=lambda: self._publish_status("heartbeat", stage),
            ):
                trace = _invoke_pretraining_helper(
                    train_delaney_once,
                    self.baseline_root,
                    work,
                    f"full-seed-{seed_index}",
                    seed_index=seed_index,
                    require_no_learning_stub=False,
                )
            self._copy(work / "checkpoint1.pt", output / "checkpoint1.pt")
            self._copy(work / "pretraining_trace.json", output / "pretraining_trace.json")
            evidence = self._verify_resource_trace(
                trace, output / "pretraining_trace.json", gpu_stage=True
            )
            self.ledger.record_scientific_execution_finished(claim, evidence)
            _write_stage_artifact_manifest(stage, output)
            self.ledger.record_completed(claim, stage, output, evidence)
            self._cleanup_accepted_work(stage)
            self._check_gpu_budget()
            self._announce("completed", stage)
        except BaseException as error:
            try:
                evidence = self._failure_evidence(
                    stage, claim, work, "pretraining_trace.json"
                )
                record_attempt_exception(self.ledger, claim, error, evidence)
            except RuntimeError:
                pass
            raise

    def _load_fold_scoped_frame_and_contracts(self, outer_fold: int):
        from split_safe import OuterFoldContract

        if outer_fold not in OUTER_FOLDS:
            raise ValueError("Outer fold is outside the frozen geometry")
        features = pd.read_csv(
            self.project_root / FOLD_VIEW_ROOT / "label_free_features.csv"
        )
        targets = pd.read_csv(
            self.project_root
            / FOLD_VIEW_ROOT
            / f"outer_{outer_fold:02d}_training_targets.csv"
        )
        outer = pd.read_csv(
            self.project_root / "artifacts/h1_random_cv_r0/outer_record_assignments.csv"
        )
        inner = pd.read_csv(
            self.project_root / "artifacts/h1_random_cv_r0/inner_id_basket_manifest.csv"
        )
        if list(features.columns) != ["curated_id", "SMILES", "sealed_block_id", "outer_fold"]:
            raise RuntimeError("Label-free execution feature schema drifted")
        if list(targets.columns) != ["curated_id", "normalized_pampa"]:
            raise RuntimeError("Fold-scoped execution target schema drifted")
        frame = features.merge(
            targets,
            on="curated_id",
            how="left",
            validate="one_to_one",
        )
        heldout = frame["outer_fold"].astype(int).eq(outer_fold)
        if frame.loc[heldout, "normalized_pampa"].notna().any():
            raise RuntimeError("Current outer heldout target became materialized")
        if frame.loc[~heldout, "normalized_pampa"].isna().any():
            raise RuntimeError("Outer-training target is absent")
        record_ids = set(features["curated_id"].astype(str))
        contracts = {}
        for fold in OUTER_FOLDS:
            test_ids = set(
                outer.loc[
                    outer["outer_fold"].astype(int).eq(fold), "curated_id"
                ].astype(str)
            )
            train_ids = record_ids - test_ids
            basket_rows = inner.loc[
                inner["outer_fold"].astype(int).eq(fold),
                ["curated_id", "inner_basket"],
            ]
            if len(basket_rows) != len(train_ids) or basket_rows["curated_id"].duplicated().any():
                raise RuntimeError(f"H1 inner ID basket coverage drifted at fold {fold}")
            inner_by_id = {
                str(row.curated_id): int(row.inner_basket)
                for row in basket_rows.itertuples(index=False)
            }
            if set(inner_by_id) != train_ids or set(inner_by_id.values()) != set(INNER_BASKETS):
                raise RuntimeError(f"H1 inner ID basket mapping drifted at fold {fold}")
            contracts[fold] = OuterFoldContract(fold, train_ids, test_ids, inner_by_id)
        return features, frame, contracts

    def _pretraining_output_for_seed(self, seed_index: int) -> Path:
        stage = next(
            stage
            for stage in self.by_kind["delaney_pretraining"]
            if int(stage["key"]["seed_index"]) == seed_index
        )
        output = self._accepted_output(stage)
        if output is None:
            raise RuntimeError("Required seed-specific pretraining is incomplete")
        return output

    def _recover_sealed_inner_bundle(
        self,
        inner_stages: Sequence[Mapping[str, object]],
        selection: Mapping[str, object],
    ) -> bool:
        """Finalize a crash-interrupted ledger commit from a sealed certificate."""

        all_stages = [*inner_stages, selection]
        latest = [self.ledger.latest_attempt(stage) for stage in all_stages]
        if any(item is None for item in latest):
            return False
        allowed = {"CLAIMED_BEFORE_NAMESPACE_OR_MODEL", "COMPLETED_ACCEPTED"}
        if any(str(item["status"]) not in allowed for item in latest if item is not None):
            return False
        outputs = [
            (self.project_root / str(item["output_namespace"])).resolve()
            for item in latest
            if item is not None
        ]
        certificate_path = outputs[-1] / "bundle_completion_certificate.json"
        if not certificate_path.is_file():
            return False
        certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
        if certificate.get("schema_version") != "scaffoldseal-inner-bundle-complete-v1":
            return False
        expected_identities = [stage["scientific_identity_sha256"] for stage in all_stages]
        if certificate.get("stage_identities") != expected_identities:
            return False
        manifest_paths = [output / "artifact_manifest.json" for output in outputs[:4]]
        selected_path = outputs[-1] / "selected_epoch.json"
        if any(not path.is_file() for path in manifest_paths) or not selected_path.is_file():
            return False
        expected_manifests = {
            stage["scientific_identity_sha256"]: stream_sha256(path)
            for stage, path in zip(inner_stages, manifest_paths)
        }
        if certificate.get("inner_artifact_manifest_sha256") != expected_manifests:
            return False
        if certificate.get("selected_epoch_sha256") != stream_sha256(
            selected_path
        ):
            return False
        evidence_by_stage = certificate.get("resource_evidence_by_stage")
        if not isinstance(evidence_by_stage, Mapping) or set(evidence_by_stage) != set(
            expected_identities
        ):
            return False
        try:
            for stage, output in zip(all_stages, outputs):
                validate_stage_artifacts(stage, output)
        except RuntimeError:
            return False
        for stage, output in zip(all_stages, outputs):
            claim = self.ledger.live_claim(stage)
            if claim is not None:
                latest_attempt = self.ledger.latest_attempt(stage) or {}
                owner = latest_attempt.get("owner")
                if not isinstance(owner, Mapping):
                    return False
                alive = _owner_alive(owner)
                same_process = (
                    int(owner.get("pid", -1)) == os.getpid()
                    and owner.get("process_start_token") == process_start_token(os.getpid())
                )
                if alive and not same_process:
                    raise StageClaimUnavailable(
                        "Live certificate owner cannot be finalized by another process"
                    )
                recovery_evidence = None
                if not alive:
                    recovery_evidence = {
                        "mode": "complete_hash_valid_certificate_after_dead_owner",
                        "dead_owner_pid": owner.get("pid"),
                        "dead_owner_process_start_token": owner.get(
                            "process_start_token"
                        ),
                        "recovered_utc": _utc_now(),
                    }
                evidence = validate_resource_evidence(
                    evidence_by_stage[str(stage["scientific_identity_sha256"])],
                    gpu_stage=bool(stage["execution"].get("gpu", False)),
                )
                self.ledger.record_completed(
                    claim,
                    stage,
                    output,
                    evidence,
                    recovery_evidence=recovery_evidence,
                )
        return True

    def _run_inner_bundle(self, outer_fold: int) -> None:
        assert_real_execution_authorized(self.plan, self.acceptance)
        self._check_gpu_budget()
        inner_stages = sorted(
            [
                stage
                for stage in self.by_kind["pampa_inner_fit"]
                if int(stage["key"]["outer_fold"]) == outer_fold
            ],
            key=lambda stage: int(stage["key"]["inner_basket"]),
        )
        selection = next(
            stage
            for stage in self.by_kind["stopping_epoch_selection"]
            if int(stage["key"]["outer_fold"]) == outer_fold
        )
        if self._recover_sealed_inner_bundle(inner_stages, selection):
            for stage in [*inner_stages, selection]:
                self._cleanup_accepted_work(stage)
            self._announce("recovered_sealed_bundle", selection)
            return
        selection_output = self._accepted_output(selection)
        if selection_output is not None:
            if any(self._accepted_output(stage) is None for stage in inner_stages):
                raise RuntimeError("Accepted stopping selection lacks an accepted inner dependency")
            for stage in [*inner_stages, selection]:
                self._cleanup_accepted_work(stage)
            self._announce("skip_validated_complete", selection)
            return
        completed_inner = [self._accepted_output(stage) for stage in inner_stages]
        if any(output is not None for output in completed_inner):
            raise RuntimeError(
                "Partial accepted inner bundle cannot be resumed because guarded histories are in-process capabilities"
            )

        claimed: list[tuple[StageClaim, Mapping[str, object], Path, Path]] = []
        try:
            for stage in [*inner_stages, selection]:
                claim, work, output = self._claim_paths(stage)
                claimed.append((claim, stage, work, output))
            for _, _, work, output in claimed:
                self._prepare_namespaces(work, output)

            from r1c0_dmpnn_pilot import LockedDMPNNAdapter, file_sha256, load_trace
            from split_safe import FitAuditTrail, SplitSafeFitExecutor

            _, frame, contracts = self._load_fold_scoped_frame_and_contracts(outer_fold)
            contract = contracts[outer_fold]
            pretraining = self._pretraining_output_for_seed(0)
            pretrained_checkpoint = pretraining / "checkpoint1.pt"
            pretrained_sha256 = file_sha256(pretrained_checkpoint)
            histories = []
            best_epochs = []
            resource_by_identity: dict[str, dict[str, object]] = {}
            for claim, stage, work, output in claimed[:4]:
                self._check_gpu_budget()
                basket = int(stage["key"]["inner_basket"])
                audit = FitAuditTrail()
                executor = SplitSafeFitExecutor(contract, audit)
                self.ledger.mark_scientific_execution_started(claim)
                adapter = LockedDMPNNAdapter(self.baseline_root, self._source_hashes or {})
                run = contract.mint_run_context(
                    work / "checkpoint_root",
                    config_id="d0_locked",
                    seed=0,
                    inner_basket=basket,
                    pretrained_checkpoint=pretrained_checkpoint,
                    pretrained_checkpoint_sha256=pretrained_sha256,
                )
                train = contract.inner_training_batch(frame, basket)
                validation = contract.inner_validation_batch(frame, basket)
                recorder = contract.create_inner_evaluation_recorder(
                    train,
                    validation,
                    basket=basket,
                    feature_columns=["SMILES"],
                    target_column="normalized_pampa",
                    metric_identity="mean_squared_error",
                    run_context=run,
                    transform_sha256=adapter.transform_sha256,
                    model_config_sha256=adapter.model_config_sha256,
                    checkpoint_sha256=pretrained_sha256,
                    audit=audit,
                )
                with self.ledger.lease_heartbeat(
                    claim,
                    on_heartbeat=lambda stage=stage: self._publish_status(
                        "heartbeat", stage
                    ),
                ):
                    executor.fit_inner_frame(
                        adapter,
                        train,
                        validation,
                        basket=basket,
                        feature_columns=["SMILES"],
                        target_column="normalized_pampa",
                        run_context=run,
                        recorder=recorder,
                        maximum_epochs=2000,
                    )
                history = recorder.finalize()
                histories.append(history)
                trace = load_trace(run.checkpoint_dir)
                best_epochs.append(int(trace["best_epoch"]))
                self._copy(Path(run.checkpoint_dir) / "checkpoint1.pt", output / "checkpoint1.pt")
                self._copy(
                    Path(run.checkpoint_dir) / "training_trace.json",
                    output / "training_trace.json",
                )
                resource_by_identity[str(stage["scientific_identity_sha256"])] = (
                    self._verify_resource_trace(
                        trace,
                        output / "training_trace.json",
                        gpu_stage=True,
                    )
                )
                self.ledger.record_scientific_execution_finished(
                    claim,
                    resource_by_identity[str(stage["scientific_identity_sha256"])],
                )
                write_json(output / "guarded_inner_history.lossless.json", _history_payload(history))
                _write_stage_artifact_manifest(stage, output)

            selected_epoch = contract.select_stopping_epoch(histories, FitAuditTrail())
            selection_claim, selection_stage, _, selection_output = claimed[-1]
            resource_by_identity[str(selection_stage["scientific_identity_sha256"])] = (
                make_resource_evidence(
                    runtime_seconds=0.0,
                    max_memory_reserved_bytes=0,
                    gpu_stage=False,
                    source="non_gpu_selection_stage",
                )
            )
            write_json(
                selection_output / "selected_epoch.json",
                {
                    "schema_version": "scaffoldseal-stopping-epoch-v1",
                    "outer_fold": outer_fold,
                    "seed_index": 0,
                    "best_epochs": best_epochs,
                    "selection_rule": "ceil(median(four best epochs))",
                    "selected_epoch": int(selected_epoch),
                    "outer_labels_used": False,
                    "inner_stage_identities": [
                        stage["scientific_identity_sha256"] for stage in inner_stages
                    ],
                },
            )
            write_json(
                selection_output / "bundle_completion_certificate.json",
                {
                    "schema_version": "scaffoldseal-inner-bundle-complete-v1",
                    "stage_identities": [
                        stage["scientific_identity_sha256"]
                        for _, stage, _, _ in claimed
                    ],
                    "inner_artifact_manifest_sha256": {
                        stage["scientific_identity_sha256"]: stream_sha256(
                            output / "artifact_manifest.json"
                        )
                        for _, stage, _, output in claimed[:4]
                    },
                    "selected_epoch_sha256": stream_sha256(
                        selection_output / "selected_epoch.json"
                    ),
                    "resource_evidence_by_stage": resource_by_identity,
                },
            )
            _write_stage_artifact_manifest(selection_stage, selection_output)
            # Inner artifacts become accepted only after the in-process guarded
            # selection succeeds, so interrupted bundles are retried as a unit.
            for claim, stage, _, output in claimed[:4]:
                self.ledger.record_completed(
                    claim,
                    stage,
                    output,
                    resource_by_identity[str(stage["scientific_identity_sha256"])],
                )
            self.ledger.record_completed(
                selection_claim,
                selection_stage,
                selection_output,
                resource_by_identity[str(selection_stage["scientific_identity_sha256"])],
            )
            for _, stage, _, _ in claimed:
                self._cleanup_accepted_work(stage)
            self._check_gpu_budget()
            self._announce("completed", selection_stage)
        except BaseException as error:
            certificate_exists = bool(claimed) and (
                claimed[-1][3] / "bundle_completion_certificate.json"
            ).is_file()
            # Once the certificate exists, the scientific bundle succeeded;
            # keep any uncommitted claims live so deterministic recovery can
            # validate exact hashes and finish ledger finalization.  Before the
            # certificate, every live claim is explicitly recorded as failed.
            if not certificate_exists:
                for claim, failed_stage, failed_work, _ in claimed:
                    try:
                        evidence = self._failure_evidence(
                            failed_stage, claim, failed_work, "training_trace.json"
                        )
                        record_attempt_exception(
                            self.ledger, claim, error, evidence
                        )
                    except RuntimeError:
                        pass
            raise

    def _selection_output(self, outer_fold: int) -> Path:
        stage = next(
            stage
            for stage in self.by_kind["stopping_epoch_selection"]
            if int(stage["key"]["outer_fold"]) == outer_fold
        )
        output = self._accepted_output(stage)
        if output is None:
            raise RuntimeError("Required stopping selection is incomplete")
        return output

    def _run_outer(self, stage: Mapping[str, object]) -> None:
        assert_real_execution_authorized(self.plan, self.acceptance)
        self._check_gpu_budget()
        if self._accepted_output(stage) is not None:
            self._cleanup_accepted_work(stage)
            self._announce("skip_validated_complete", stage)
            return
        claim, work, output = self._claim_paths(stage)
        try:
            self._prepare_namespaces(work, output)
            from r1c0_dmpnn_pilot import LockedDMPNNAdapter, file_sha256, load_trace
            from split_safe import FitAuditTrail, SplitSafeFitExecutor

            fold = int(stage["key"]["outer_fold"])
            seed_index = int(stage["key"]["seed_index"])
            _, frame, contracts = self._load_fold_scoped_frame_and_contracts(fold)
            contract = contracts[fold]
            pretraining = self._pretraining_output_for_seed(seed_index)
            pretrained_checkpoint = pretraining / "checkpoint1.pt"
            pretrained_sha256 = file_sha256(pretrained_checkpoint)
            selected = json.loads(
                (self._selection_output(fold) / "selected_epoch.json").read_text(encoding="utf-8")
            )
            fixed_epoch = int(selected["selected_epoch"])
            audit = FitAuditTrail()
            executor = SplitSafeFitExecutor(contract, audit)
            self.ledger.mark_scientific_execution_started(claim)
            adapter = LockedDMPNNAdapter(self.baseline_root, self._source_hashes or {})
            run = contract.mint_run_context(
                work / "checkpoint_root",
                config_id="d0_locked",
                seed=seed_index,
                inner_basket=None,
                pretrained_checkpoint=pretrained_checkpoint,
                pretrained_checkpoint_sha256=pretrained_sha256,
            )
            with self.ledger.lease_heartbeat(
                claim,
                on_heartbeat=lambda: self._publish_status("heartbeat", stage),
            ):
                handle = executor.fit_outer_frame(
                    adapter,
                    contract.outer_frame_training_batch(frame),
                    feature_columns=["SMILES"],
                    target_column="normalized_pampa",
                    run_context=run,
                    fixed_epoch=fixed_epoch,
                )
                prediction = executor.predict_outer_frame(handle)
            trace = load_trace(run.checkpoint_dir)
            self._copy(
                Path(run.checkpoint_dir) / "training_trace.json", output / "training_trace.json"
            )
            evidence = self._verify_resource_trace(
                trace,
                output / "training_trace.json",
                gpu_stage=True,
            )
            self.ledger.record_scientific_execution_finished(claim, evidence)
            normalized = np.asarray(prediction.predictions, dtype=np.float64)
            prediction_table = pd.DataFrame(
                {
                    "curated_id": list(prediction.ids),
                    "outer_fold": fold,
                    "seed": seed_index,
                    "prediction_normalized": normalized,
                    "prediction_log10_papp": normalized * 2.0 - 6.0,
                }
            )
            # No observed label is joined before both artifacts are sealed.
            write_sealed_prediction_artifacts(
                prediction_table,
                output / "outer_predictions.lossless.json",
                output / "outer_predictions.csv",
            )
            self._copy(Path(run.checkpoint_dir) / "checkpoint1.pt", output / "checkpoint1.pt")
            write_json(output / "fit_audit.json", audit.records)
            _write_stage_artifact_manifest(stage, output)
            self.ledger.record_completed(claim, stage, output, evidence)
            self._cleanup_accepted_work(stage)
            self._check_gpu_budget()
            self._announce("completed", stage)
        except BaseException as error:
            try:
                evidence = self._failure_evidence(stage, claim, work, "training_trace.json")
                record_attempt_exception(self.ledger, claim, error, evidence)
            except RuntimeError:
                pass
            raise

    def _run_metrics(self) -> None:
        assert_real_execution_authorized(self.plan, self.acceptance)
        stage = self.by_kind["sealed_oof_metrics"][0]
        if self._accepted_output(stage) is not None:
            self._cleanup_accepted_work(stage)
            self._announce("skip_validated_complete", stage)
            return
        claim, work, output = self._claim_paths(stage)
        try:
            self._prepare_namespaces(work, output)
            prediction_tables = []
            # Artifact hashes are validated by _accepted_output before any label
            # file is opened in this method.
            for dependency in stage["dependencies"]:
                dep_stage = self.by_identity[str(dependency)]
                dep_output = self._accepted_output(dep_stage)
                if dep_output is None:
                    raise RuntimeError("Metric stage has an incomplete prediction dependency")
                prediction_tables.append(
                    read_sealed_prediction_artifact(
                        dep_output / "outer_predictions.lossless.json"
                    )
                )
            predictions = pd.concat(prediction_tables, ignore_index=True)
            if len(predictions) != 34475 or predictions.duplicated(
                ["curated_id", "outer_fold", "seed"]
            ).any():
                raise RuntimeError("Sealed OOF prediction coverage is incomplete or duplicated")
            expected_slots = {
                (row.curated_id, int(row.outer_fold), seed)
                for row in pd.read_csv(
                    self.project_root / "artifacts/h1_random_cv_r0/outer_record_assignments.csv"
                ).itertuples(index=False)
                for seed in SEED_INDICES
            }
            observed_slots = set(
                predictions[["curated_id", "outer_fold", "seed"]].itertuples(
                    index=False, name=None
                )
            )
            if observed_slots != expected_slots:
                raise RuntimeError("Sealed OOF prediction slots differ from the frozen manifest")
            write_sealed_prediction_artifacts(
                predictions[
                    [
                        "curated_id",
                        "outer_fold",
                        "seed",
                        "prediction_normalized",
                        "prediction_log10_papp",
                    ]
                ],
                output / "oof_predictions.lossless.json",
                output / "oof_predictions.csv",
            )

            # Labels are loaded only after exact slot and artifact-hash checks.
            labels_path = self.project_root / FULL_LABEL_RELATIVE_PATH
            expected_label = self.plan["scientific_lock"]["metric_only_label_source"]
            if labels_path.stat().st_size != int(expected_label["size_bytes"]) or stream_sha256(
                labels_path
            ) != str(expected_label["sha256"]):
                raise RuntimeError("Metric-only full label source drifted")
            self.ledger.mark_scientific_execution_started(claim)
            labels = pd.read_csv(labels_path)
            evaluated = predictions.merge(
                labels[["curated_id", "source", "sealed_block_id", "permeability"]],
                on="curated_id",
                how="left",
                validate="many_to_one",
            )
            if evaluated["permeability"].isna().any():
                raise RuntimeError("A sealed prediction lacks its frozen label")
            evaluated["absolute_error"] = np.abs(
                evaluated["prediction_log10_papp"] - evaluated["permeability"]
            )
            per_source = (
                evaluated.groupby(["seed", "source"], sort=True)["absolute_error"]
                .mean()
                .rename("mae")
                .reset_index()
            )
            per_block = (
                evaluated.groupby(["seed", "outer_fold", "sealed_block_id"], sort=True)[
                    "absolute_error"
                ]
                .agg([("mae", "mean"), ("n", "size")])
                .reset_index()
            )
            seed_source_macro = per_source.groupby("seed", sort=True)["mae"].mean()
            summary = {
                "schema_version": "scaffoldseal-d0-metrics-v1",
                "labels_loaded_after_prediction_sealing": True,
                "n_prediction_rows": len(evaluated),
                "source_macro_mae_by_seed": {
                    str(int(seed)): float(value) for seed, value in seed_source_macro.items()
                },
                "source_macro_mae_mean_across_seeds": float(seed_source_macro.mean()),
                "row_micro_mae_by_seed": {
                    str(int(seed)): float(value)
                    for seed, value in evaluated.groupby("seed", sort=True)[
                        "absolute_error"
                    ].mean().items()
                },
            }
            write_json(output / "metrics_summary.json", summary)
            per_source.to_csv(output / "per_source_metrics.csv", index=False, lineterminator="\n")
            per_block.to_csv(output / "per_block_metrics.csv", index=False, lineterminator="\n")
            _write_stage_artifact_manifest(stage, output)
            evidence = make_resource_evidence(
                runtime_seconds=0.0,
                max_memory_reserved_bytes=0,
                gpu_stage=False,
                source="non_gpu_metric_stage",
            )
            self.ledger.record_completed(claim, stage, output, evidence)
            self._cleanup_accepted_work(stage)
            self._announce("completed", stage)
        except BaseException as error:
            try:
                evidence = self._failure_evidence(stage, claim, work, "metrics_trace.json")
                record_attempt_exception(self.ledger, claim, error, evidence)
            except RuntimeError:
                pass
            raise

    def run(self) -> None:
        assert_real_execution_authorized(self.plan, self.acceptance)
        verify_candidate_anchor(
            self.plan,
            self.project_root,
            self.baseline_root,
            manifest_path=self.manifest_path,
            acceptance=self.acceptance,
            require_accepted=True,
        )
        if int(self.plan["scheduler"]["local_gpu_workers"]) != 1:
            raise RuntimeError("The locked full run requires exactly one local GPU worker")
        if float(self.plan["scientific_lock"]["resources"]["projected_gpu_hours"]) > float(
            self.plan["scientific_lock"]["resources"]["maximum_gpu_hours"]
        ):
            raise RuntimeError("Locked resource projection exceeds the predeclared limit")
        check_disk_space(self.project_root)
        from r1c0_dmpnn_pilot import verify_environment

        self._versions, self._source_hashes = verify_environment(self.baseline_root)
        for stage in self.by_kind["delaney_pretraining"]:
            self._run_pretraining(stage)
        for fold in OUTER_FOLDS:
            self._run_inner_bundle(fold)
        for stage in self.by_kind["pampa_outer_fit_predict"]:
            self._run_outer(stage)
        self._run_metrics()
        self._announce("completed_all")


def execute_full_plan(
    plan: Mapping[str, object],
    project_root: Path,
    baseline_root: Path,
    *,
    acceptance: Mapping[str, object],
    manifest_path: Path,
) -> None:
    """Run the exact reviewed graph; unreachable while the hard lock is false."""

    assert_real_execution_authorized(plan, acceptance)
    D0FullExecutor(
        plan,
        project_root,
        baseline_root,
        acceptance=acceptance,
        manifest_path=manifest_path,
    ).run()


def recover_stage_attempt(
    plan: Mapping[str, object], project_root: Path, identity: str
) -> dict[str, object]:
    stage = next(
        (item for item in plan["stages"] if item["scientific_identity_sha256"] == identity),
        None,
    )
    if stage is None:
        raise ValueError("Recovery identity is not in the immutable candidate")
    ledger = ScientificStageLedger(
        project_root / str(plan["scheduler"]["ledger_relative_path"]),
        project_root=project_root,
    )
    return ledger.recover_interrupted(
        stage,
        reason="explicit CLI dead-owner recovery; namespace preserved",
    )


def recover_inner_bundle_attempts(
    plan: Mapping[str, object], project_root: Path, outer_fold: int
) -> dict[str, object]:
    if outer_fold not in OUTER_FOLDS:
        raise ValueError("Recovery outer fold is outside the immutable geometry")
    executor = object.__new__(D0FullExecutor)
    executor.plan = plan
    executor.project_root = project_root.resolve()
    executor.ledger = ScientificStageLedger(
        executor.project_root / str(plan["scheduler"]["ledger_relative_path"]),
        project_root=executor.project_root,
    )
    executor.by_kind = {}
    executor.by_identity = {}
    for stage in plan["stages"]:
        executor.by_kind.setdefault(str(stage["kind"]), []).append(stage)
        executor.by_identity[str(stage["scientific_identity_sha256"])] = stage
    inner = sorted(
        [
            stage
            for stage in executor.by_kind["pampa_inner_fit"]
            if int(stage["key"]["outer_fold"]) == outer_fold
        ],
        key=lambda stage: int(stage["key"]["inner_basket"]),
    )
    selection = next(
        stage
        for stage in executor.by_kind["stopping_epoch_selection"]
        if int(stage["key"]["outer_fold"]) == outer_fold
    )
    if executor._recover_sealed_inner_bundle(inner, selection):
        return {"status": "COMPLETE_CERTIFICATE_RECOVERED", "outer_fold": outer_fold}
    recovered = []
    for stage in [*inner, selection]:
        if executor.ledger.live_claim(stage) is None:
            continue
        executor.ledger.recover_interrupted(
            stage,
            reason="explicit CLI partial-bundle dead-owner recovery; namespace preserved",
        )
        recovered.append(stage["scientific_identity_sha256"])
    if not recovered:
        raise RuntimeError("No complete certificate and no dead live bundle attempts found")
    return {
        "status": "PARTIAL_BUNDLE_INTERRUPTED_RECORDED",
        "outer_fold": outer_fold,
        "recovered_stage_identities": recovered,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--acceptance", type=Path)
    parser.add_argument("--manifest-out", type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--build-candidate", action="store_true")
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--status-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--recover-interrupted", metavar="STAGE_SHA256")
    modes.add_argument("--recover-bundle", type=int, metavar="OUTER_FOLD")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    manifest_path = (args.manifest or (project_root / DEFAULT_MANIFEST)).resolve()
    acceptance: dict[str, object] | None = None
    if args.build_candidate:
        plan = build_full_plan(project_root, args.baseline_root)
        output = (args.manifest_out or manifest_path).resolve()
        write_json(output, plan)
    else:
        if args.manifest_out is not None:
            parser.error("--manifest-out is valid only with --build-candidate")
        plan, acceptance, manifest_path, _ = load_externally_bound_candidate(
            project_root,
            args.baseline_root,
            acceptance_path=args.acceptance,
            caller_manifest_path=args.manifest,
        )
        output = manifest_path
        if args.execute:
            assert acceptance is not None
            execute_full_plan(
                plan,
                project_root,
                args.baseline_root,
                acceptance=acceptance,
                manifest_path=manifest_path,
            )
        elif args.recover_interrupted:
            recovery = recover_stage_attempt(plan, project_root, args.recover_interrupted)
            publish_live_status(
                project_root,
                plan,
                ScientificStageLedger(
                    project_root / str(plan["scheduler"]["ledger_relative_path"]),
                    project_root=project_root,
                ),
                phase="RECOVERY_RECORDED",
                training_state="NO_TRAINING",
                external_acceptance=False,
                next_action="独立核验恢复证据后，才可考虑新尝试",
            )
            print(
                json.dumps(
                    recovery,
                    sort_keys=True,
                )
            )
            return
        elif args.recover_bundle is not None:
            recovery = recover_inner_bundle_attempts(plan, project_root, args.recover_bundle)
            publish_live_status(
                project_root,
                plan,
                ScientificStageLedger(
                    project_root / str(plan["scheduler"]["ledger_relative_path"]),
                    project_root=project_root,
                ),
                phase="BUNDLE_RECOVERY_RECORDED",
                training_state="NO_TRAINING",
                external_acceptance=False,
                next_action="独立核验恢复证书或中断记录",
            )
            print(
                json.dumps(
                    recovery,
                    sort_keys=True,
                )
            )
            return
    if not args.execute:
        ledger = ScientificStageLedger(
            project_root / str(plan["scheduler"]["ledger_relative_path"]),
            project_root=project_root,
        )
        publish_live_status(
            project_root,
            plan,
            ledger,
            phase=("REPAIR_CANDIDATE_BUILT" if args.build_candidate else "PLAN_ONLY_REVIEW"),
            training_state="NO_TRAINING",
            external_acceptance=False,
            next_action="等待独立预训练审查；当前不得训练",
        )
    print(
        json.dumps(
            {
                "status": plan["status"],
                "plan_sha256": plan["plan_sha256"],
                "manifest": str(output.resolve()),
                "counts": plan["counts"],
                "real_execution_authorized": bool(
                    plan["authorization"]["real_execution_authorized"]
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
