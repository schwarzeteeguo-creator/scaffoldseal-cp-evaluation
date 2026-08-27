"""R1a deterministic training-mean and nested ridge joint-block LOBO runner."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
import sklearn
from sklearn.linear_model import Ridge

from split_safe import (
    FitAuditTrail,
    GuardedPredictionResult,
    OuterFoldContract,
    SplitSafeFitExecutor,
    SplitSafeMixedPreprocessor,
    canonical_id_hash,
    contracts_from_manifests,
)


ID_COLUMN = "curated_id"
TARGET_COLUMN = "permeability"
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
TOPOLOGY_CATEGORIES = ("circle_head_to_tail", "circle_other", "lariat")
STANDARD_RESIDUES = frozenset("ARNDCEQGHILKMFPSTWYV")
CONTINUOUS_DESCRIPTORS = (
    "ring_size",
    "molecular_weight",
    "clogp",
    "tpsa",
    "hbd",
    "hba",
    "rotatable_bonds",
    "formal_charge",
    "fraction_sp3",
    "stereocenter_count",
    "n_methyl_count",
    "noncanonical_count",
)
MISSING_FLAGS = tuple(f"{name}__missing" for name in CONTINUOUS_DESCRIPTORS)
FINGERPRINT_COLUMNS = tuple(f"ecfp4_bit_{index:04d}" for index in range(2048))
TOPOLOGY_COLUMNS = tuple(f"topology__{name}" for name in TOPOLOGY_CATEGORIES)
PASSTHROUGH_COLUMNS = (*FINGERPRINT_COLUMNS, *TOPOLOGY_COLUMNS, *MISSING_FLAGS)
FEATURE_COLUMNS = (*FINGERPRINT_COLUMNS, *TOPOLOGY_COLUMNS, *CONTINUOUS_DESCRIPTORS, *MISSING_FLAGS)
ZERO_HASH = "0" * 64


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def alpha_config_id(alpha: float) -> str:
    return f"ridge_alpha_{format(float(alpha), '.12g')}"


class TrainingMeanRegressor:
    def __init__(self) -> None:
        self.mean_: float | None = None

    def fit(self, X, y):
        values = pd.to_numeric(y, errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError("Training mean requires finite outcomes")
        self.mean_ = float(np.mean(values))
        return self

    def predict(self, X):
        if self.mean_ is None:
            raise ValueError("Training mean is not fitted")
        return np.full(len(X), self.mean_, dtype=float)


class TrainingMeanFactory:
    def __init__(self) -> None:
        self.model_config_sha256 = canonical_json_hash({"model": "outer_training_mean"})

    def __call__(self) -> TrainingMeanRegressor:
        return TrainingMeanRegressor()


class RidgeFactory:
    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)
        self.model_config_sha256 = canonical_json_hash(
            {
                "model": "sklearn.linear_model.Ridge",
                "alpha": self.alpha,
                "fit_intercept": True,
                "solver": "lsqr",
                "tol": 1e-4,
            }
        )

    def __call__(self) -> Ridge:
        return Ridge(
            alpha=self.alpha,
            fit_intercept=True,
            solver="lsqr",
            tol=1e-4,
        )


def _is_n_methyl_token(token: str) -> bool:
    normalized = str(token).strip().lower()
    return normalized == "sar" or normalized.startswith("me")


def _is_noncanonical_token(token: str) -> bool:
    return str(token).strip() not in STANDARD_RESIDUES


def descriptor_definitions() -> dict[str, object]:
    return {
        "ecfp4": {
            "implementation": "RDKit MorganGenerator",
            "radius": 2,
            "fp_size": 2048,
            "include_chirality": True,
            "scaling": "none",
        },
        "topology_class": {
            "definition": "prefix before first '|' in frozen topology_signature",
            "encoding": "fixed one-hot",
            "categories": list(TOPOLOGY_CATEGORIES),
            "scaling": "none",
        },
        "ring_size": "frozen curated_records_public.ring_size",
        "molecular_weight": "rdkit.Chem.Descriptors.MolWt",
        "clogp": "rdkit.Chem.Crippen.MolLogP",
        "tpsa": "rdkit.Chem.rdMolDescriptors.CalcTPSA",
        "hbd": "rdkit.Chem.Lipinski.NumHDonors",
        "hba": "rdkit.Chem.Lipinski.NumHAcceptors",
        "rotatable_bonds": "rdkit.Chem.Lipinski.NumRotatableBonds",
        "formal_charge": "sum(atom.GetFormalCharge())",
        "fraction_sp3": "rdkit.Chem.rdMolDescriptors.CalcFractionCSP3",
        "stereocenter_count": (
            "len(Chem.FindMolChiralCenters(includeUnassigned=True, "
            "useLegacyImplementation=False))"
        ),
        "n_methyl_count": (
            "count frozen canonical_sequence tokens whose lowercase token is 'sar' "
            "or begins with 'me'; side-chain '(Me)' alone is not counted"
        ),
        "noncanonical_count": (
            "count frozen canonical_sequence tokens not exactly one of the 20 standard "
            "L-amino-acid one-letter codes ARNDCEQGHILKMFPSTWYV"
        ),
        "missingness": {
            "flags": list(MISSING_FLAGS),
            "definition": "1 iff corresponding pre-imputation descriptor is missing/non-finite",
            "imputation": "inner/outer training median only",
        },
        "continuous_preprocessing": (
            "training-only median imputation, zero-variance filtering, centering and unit-variance scaling"
        ),
        "binary_preprocessing": "finite passthrough without scaling",
    }


def build_feature_frame(
    analysis_path: Path,
    curated_public_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    analysis = pd.read_csv(analysis_path)
    public = pd.read_csv(
        curated_public_path,
        usecols=[ID_COLUMN, "canonical_sequence", "ring_size", "topology_signature"],
    )
    if analysis[ID_COLUMN].duplicated().any() or public[ID_COLUMN].duplicated().any():
        raise ValueError("R1a inputs require unique curated IDs")
    merged = analysis.merge(public, on=ID_COLUMN, how="left", validate="one_to_one", suffixes=("", "_public"))
    if len(merged) != 6895 or merged["canonical_sequence"].isna().any():
        raise ValueError("R1a feature join must cover exactly 6,895 curated records")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=True
    )
    bits = np.zeros((len(merged), 2048), dtype=np.uint8)
    descriptor_rows: list[dict[str, float]] = []
    topology_values: list[str] = []
    for row_index, row in enumerate(merged.itertuples(index=False)):
        mol = Chem.MolFromSmiles(str(row.canonical_smiles))
        if mol is None:
            raise ValueError(f"Unparseable frozen canonical SMILES: {row.curated_id}")
        fingerprint = generator.GetFingerprint(mol)
        DataStructs.ConvertToNumpyArray(fingerprint, bits[row_index])
        tokens = json.loads(str(row.canonical_sequence))
        topology = str(row.topology_signature).split("|", 1)[0]
        if topology not in TOPOLOGY_CATEGORIES:
            raise ValueError(f"Unrecognized frozen topology class: {topology}")
        topology_values.append(topology)
        descriptor_rows.append(
            {
                "ring_size": float(row.ring_size),
                "molecular_weight": float(Descriptors.MolWt(mol)),
                "clogp": float(Crippen.MolLogP(mol)),
                "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
                "hbd": float(Lipinski.NumHDonors(mol)),
                "hba": float(Lipinski.NumHAcceptors(mol)),
                "rotatable_bonds": float(Lipinski.NumRotatableBonds(mol)),
                "formal_charge": float(sum(atom.GetFormalCharge() for atom in mol.GetAtoms())),
                "fraction_sp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
                "stereocenter_count": float(
                    len(
                        Chem.FindMolChiralCenters(
                            mol,
                            includeUnassigned=True,
                            useLegacyImplementation=False,
                        )
                    )
                ),
                "n_methyl_count": float(sum(_is_n_methyl_token(token) for token in tokens)),
                "noncanonical_count": float(
                    sum(_is_noncanonical_token(token) for token in tokens)
                ),
            }
        )
    feature_frame = pd.DataFrame({ID_COLUMN: merged[ID_COLUMN].astype(str)})
    feature_frame = pd.concat(
        [feature_frame, pd.DataFrame(bits, columns=FINGERPRINT_COLUMNS)], axis=1
    )
    for category, column in zip(TOPOLOGY_CATEGORIES, TOPOLOGY_COLUMNS):
        feature_frame[column] = (np.asarray(topology_values) == category).astype(np.uint8)
    descriptors = pd.DataFrame(descriptor_rows, columns=CONTINUOUS_DESCRIPTORS)
    for name in CONTINUOUS_DESCRIPTORS:
        values = pd.to_numeric(descriptors[name], errors="coerce").to_numpy(float)
        feature_frame[name] = values
        feature_frame[f"{name}__missing"] = (~np.isfinite(values)).astype(np.uint8)
    feature_frame[TARGET_COLUMN] = pd.to_numeric(merged[TARGET_COLUMN], errors="raise").to_numpy(float)
    feature_frame = feature_frame.loc[:, [ID_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]]
    metadata = merged.loc[
        :,
        [
            ID_COLUMN,
            "molecule_id",
            "source",
            "analogue_component_id",
            "sealed_block_id",
        ],
    ].copy()
    values = feature_frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype="<f8", copy=True)
    feature_hash = hashlib.sha256()
    feature_hash.update(json.dumps(list(FEATURE_COLUMNS), separators=(",", ":")).encode("utf-8"))
    feature_hash.update("\n".join(feature_frame[ID_COLUMN].astype(str)).encode("utf-8"))
    feature_hash.update(values.tobytes(order="C"))
    provenance = {
        "schema_version": "scaffoldseal-r1a-features-v1",
        "analysis_input_sha256": sha256_file(analysis_path),
        "curated_public_input_sha256": sha256_file(curated_public_path),
        "n_rows": len(feature_frame),
        "n_features": len(FEATURE_COLUMNS),
        "feature_columns": list(FEATURE_COLUMNS),
        "passthrough_columns": list(PASSTHROUGH_COLUMNS),
        "continuous_columns": list(CONTINUOUS_DESCRIPTORS),
        "definitions": descriptor_definitions(),
        "feature_matrix_sha256": feature_hash.hexdigest(),
        "rdkit": rdBase.rdkitVersion,
    }
    return feature_frame, metadata, provenance


def source_macro_mae(predictions: pd.DataFrame) -> float:
    per_source = predictions.assign(
        absolute_error=(predictions["prediction"] - predictions["observed"]).abs()
    ).groupby("source", sort=True)["absolute_error"].mean()
    return float(per_source.mean())


def select_alpha(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    if {float(row["alpha"]) for row in rows} != set(ALPHAS):
        raise ValueError("Alpha selection requires the exact frozen five-value grid")
    return min(
        rows,
        key=lambda row: (
            float(row["source_macro_mae"]),
            float(row["row_micro_mae"]),
            int(row["compute_rank"]),
            str(row["config_id"]),
        ),
    )


def _prediction_frame(
    result: GuardedPredictionResult,
    metadata_by_id: pd.DataFrame,
    observed_by_id: pd.Series,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            ID_COLUMN: result.ids,
            "prediction": result.predictions,
            "observed": [float(observed_by_id.loc[value]) for value in result.ids],
        }
    )
    return frame.merge(metadata_by_id.reset_index(), on=ID_COLUMN, how="left", validate="one_to_one")


def _new_mixed_preprocessor(
    contract: OuterFoldContract,
    audit: FitAuditTrail,
    passthrough_columns: Sequence[str],
) -> SplitSafeMixedPreprocessor:
    return SplitSafeMixedPreprocessor(
        contract,
        audit,
        passthrough_columns=passthrough_columns,
        max_missing_fraction=0.50,
        min_variance=0.0,
    )


@dataclass
class LoboResult:
    oof_predictions: pd.DataFrame
    inner_selection: pd.DataFrame
    audit_records: list[dict[str, object]]
    run_namespace_map: dict[str, str]


def run_lobo(
    feature_frame: pd.DataFrame,
    metadata: pd.DataFrame,
    contracts: dict[int, OuterFoldContract],
    run_root: Path,
    *,
    outer_folds: Iterable[int] | None = None,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
    passthrough_columns: Sequence[str] = PASSTHROUGH_COLUMNS,
    progress: bool = False,
) -> LoboResult:
    run_root = run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    selected_folds = tuple(sorted(contracts)) if outer_folds is None else tuple(sorted(map(int, outer_folds)))
    observed_by_id = feature_frame.set_index(ID_COLUMN)[TARGET_COLUMN]
    metadata_by_id = metadata.set_index(ID_COLUMN)
    all_oof: list[pd.DataFrame] = []
    all_selection: list[dict[str, object]] = []
    all_audit: list[dict[str, object]] = []
    namespace_map: dict[str, str] = {}
    for outer_fold in selected_folds:
        contract = contracts[outer_fold]
        audit = FitAuditTrail()
        executor = SplitSafeFitExecutor(contract, audit)
        per_alpha_predictions: dict[float, list[pd.DataFrame]] = {alpha: [] for alpha in ALPHAS}
        for basket in range(1, 5):
            train = contract.inner_training_batch(feature_frame, basket)
            validation = contract.inner_validation_batch(feature_frame, basket)
            preprocessor = _new_mixed_preprocessor(contract, audit, passthrough_columns)
            executor.fit_preprocessor(
                preprocessor,
                train,
                feature_columns,
                target_column=TARGET_COLUMN,
            )
            for alpha in ALPHAS:
                factory = RidgeFactory(alpha)
                run = contract.mint_run_context(
                    run_root,
                    config_id=alpha_config_id(alpha),
                    seed=0,
                    inner_basket=basket,
                )
                relative_namespace = str(Path(run.checkpoint_dir).relative_to(run_root)).replace("\\", "/")
                namespace_map[
                    hashlib.sha256(str(Path(run.checkpoint_dir).resolve()).encode("utf-8")).hexdigest()
                ] = relative_namespace
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
                _, prediction_result = recorder.evaluate_estimator_predictions(1, model)
                recorder.finalize()
                per_alpha_predictions[alpha].append(
                    _prediction_frame(prediction_result, metadata_by_id, observed_by_id)
                )
        fold_selection_rows: list[dict[str, object]] = []
        for alpha in ALPHAS:
            inner = pd.concat(per_alpha_predictions[alpha], ignore_index=True)
            if set(inner[ID_COLUMN]) != set(contract.outer_train_ids) or inner[ID_COLUMN].duplicated().any():
                raise RuntimeError("Inner predictions must cover every outer-training ID exactly once")
            row = {
                "outer_fold": outer_fold,
                "config_id": alpha_config_id(alpha),
                "alpha": float(alpha),
                "source_macro_mae": source_macro_mae(inner),
                "row_micro_mae": float(np.mean(np.abs(inner["prediction"] - inner["observed"]))),
                "compute_rank": 1,
            }
            fold_selection_rows.append(row)
        selected = select_alpha(fold_selection_rows)
        for row in fold_selection_rows:
            row["selected"] = bool(row["config_id"] == selected["config_id"])
            all_selection.append(row)

        outer_train = contract.outer_training_batch(feature_frame)
        outer_test = contract.outer_test_batch(feature_frame)
        mean_factory = TrainingMeanFactory()
        mean_run = contract.mint_run_context(
            run_root, config_id="training_mean", seed=0, inner_basket=None
        )
        namespace_map[
            hashlib.sha256(str(Path(mean_run.checkpoint_dir).resolve()).encode("utf-8")).hexdigest()
        ] = str(Path(mean_run.checkpoint_dir).relative_to(run_root)).replace("\\", "/")
        mean_model = executor.fit_outer_estimator(
            mean_factory,
            outer_train,
            feature_columns,
            TARGET_COLUMN,
            run_context=mean_run,
        )
        mean_result = executor.predict_outer_estimator(
            mean_model,
            outer_test,
            feature_columns=feature_columns,
            target_column=TARGET_COLUMN,
            run_context=mean_run,
        )
        mean_predictions = _prediction_frame(mean_result, metadata_by_id, observed_by_id)
        mean_predictions["model"] = "training_mean"
        mean_predictions["config_id"] = "training_mean"
        mean_predictions["alpha"] = np.nan
        mean_predictions["outer_fold"] = outer_fold
        all_oof.append(mean_predictions)

        outer_preprocessor = _new_mixed_preprocessor(contract, audit, passthrough_columns)
        executor.fit_preprocessor(
            outer_preprocessor,
            outer_train,
            feature_columns,
            target_column=TARGET_COLUMN,
        )
        ridge_factory = RidgeFactory(float(selected["alpha"]))
        ridge_run = contract.mint_run_context(
            run_root,
            config_id=str(selected["config_id"]),
            seed=0,
            inner_basket=None,
        )
        namespace_map[
            hashlib.sha256(str(Path(ridge_run.checkpoint_dir).resolve()).encode("utf-8")).hexdigest()
        ] = str(Path(ridge_run.checkpoint_dir).relative_to(run_root)).replace("\\", "/")
        ridge_model = executor.fit_outer_estimator(
            ridge_factory,
            outer_train,
            feature_columns,
            TARGET_COLUMN,
            run_context=ridge_run,
            preprocessor=outer_preprocessor,
        )
        ridge_result = executor.predict_outer_estimator(
            ridge_model,
            outer_test,
            feature_columns=feature_columns,
            target_column=TARGET_COLUMN,
            run_context=ridge_run,
            preprocessor=outer_preprocessor,
        )
        ridge_predictions = _prediction_frame(ridge_result, metadata_by_id, observed_by_id)
        ridge_predictions["model"] = "ridge"
        ridge_predictions["config_id"] = str(selected["config_id"])
        ridge_predictions["alpha"] = float(selected["alpha"])
        ridge_predictions["outer_fold"] = outer_fold
        all_oof.append(ridge_predictions)

        for record in audit.records:
            compact = dict(record)
            compact.pop("feature_columns", None)
            if "checkpoint_dir" in compact:
                absolute = Path(str(compact.pop("checkpoint_dir"))).resolve()
                compact["checkpoint_namespace"] = str(absolute.relative_to(run_root)).replace("\\", "/")
            if "run_namespace_sha256" in compact:
                observed_hash = str(compact.pop("run_namespace_sha256"))
                compact["checkpoint_namespace"] = namespace_map[observed_hash]
            compact.pop("execution_identity_sha256", None)
            all_audit.append(compact)
        if progress:
            print(
                json.dumps(
                    {
                        "outer_fold": int(outer_fold),
                        "selected_alpha": float(selected["alpha"]),
                        "n_outer_test": len(outer_test.ids),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    oof = pd.concat(all_oof, ignore_index=True)
    oof = oof.loc[
        :,
        [
            ID_COLUMN,
            "molecule_id",
            "source",
            "analogue_component_id",
            "sealed_block_id",
            "outer_fold",
            "model",
            "config_id",
            "alpha",
            "observed",
            "prediction",
        ],
    ].sort_values(["model", ID_COLUMN], kind="stable").reset_index(drop=True)
    selection = pd.DataFrame(all_selection).sort_values(
        ["outer_fold", "config_id"], kind="stable"
    ).reset_index(drop=True)
    return LoboResult(oof, selection, all_audit, namespace_map)


def validate_fit_audit(
    audit_records: Sequence[dict[str, object]],
    contracts: dict[int, OuterFoldContract],
) -> None:
    for record in audit_records:
        if not str(record.get("operation", "")).endswith(".fit"):
            continue
        contract = contracts[int(record["outer_fold"])]
        basket = record.get("inner_basket")
        if basket is None:
            expected = canonical_id_hash(contract.outer_train_ids)
        else:
            expected = canonical_id_hash(contract.expected_inner_ids(int(basket))[0])
        if record.get("fit_ids_sha256") != expected:
            raise RuntimeError("Fit audit hash differs from the exact authorized training IDs")
        if record.get("fit_ids_sha256") == canonical_id_hash(contract.outer_test_ids):
            raise RuntimeError("Outer-test IDs reached a fit audit record")


def metrics_tables(oof: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    work = oof.assign(
        error=oof["prediction"] - oof["observed"],
    )
    work["absolute_error"] = work["error"].abs()
    work["squared_error"] = work["error"] ** 2
    per_source = (
        work.groupby(["model", "source"], sort=True)
        .agg(
            n=(ID_COLUMN, "size"),
            mae=("absolute_error", "mean"),
            mse=("squared_error", "mean"),
        )
        .reset_index()
    )
    per_source["rmse"] = np.sqrt(per_source.pop("mse"))
    per_block = (
        work.groupby(["model", "sealed_block_id", "outer_fold"], sort=True)
        .agg(
            n=(ID_COLUMN, "size"),
            mae=("absolute_error", "mean"),
            mse=("squared_error", "mean"),
        )
        .reset_index()
    )
    per_block["rmse"] = np.sqrt(per_block.pop("mse"))
    summary_rows: list[dict[str, object]] = []
    for model, group in work.groupby("model", sort=True):
        source_group = per_source.loc[per_source["model"] == model]
        block_group = per_block.loc[per_block["model"] == model]
        summary_rows.append(
            {
                "model": model,
                "n": len(group),
                "source_macro_mae": float(source_group["mae"].mean()),
                "source_macro_rmse": float(source_group["rmse"].mean()),
                "row_micro_mae": float(group["absolute_error"].mean()),
                "row_micro_rmse": float(np.sqrt(group["squared_error"].mean())),
                "block_median_mae": float(block_group["mae"].median()),
                "block_mae_iqr": [
                    float(block_group["mae"].quantile(0.25)),
                    float(block_group["mae"].quantile(0.75)),
                ],
            }
        )
    return {"status": "PROVISIONAL_PENDING_VERIFIER", "models": summary_rows}, per_source, per_block


def write_outputs(
    output_dir: Path,
    result: LoboResult,
    feature_provenance: dict[str, object],
    contracts: dict[int, OuterFoldContract],
    runtime_seconds: float,
    command: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    validate_fit_audit(result.audit_records, contracts)
    expected_ids = set().union(*(contract.outer_test_ids for contract in contracts.values()))
    for model, group in result.oof_predictions.groupby("model"):
        if len(group) != len(expected_ids) or set(group[ID_COLUMN]) != expected_ids or group[ID_COLUMN].duplicated().any():
            raise RuntimeError(f"{model} does not have exactly one OOF prediction per record")
    metrics, per_source, per_block = metrics_tables(result.oof_predictions)
    result.oof_predictions.to_csv(output_dir / "oof_predictions.csv", index=False, lineterminator="\n")
    result.inner_selection.to_csv(output_dir / "inner_selection.csv", index=False, lineterminator="\n")
    per_source.to_csv(output_dir / "per_source_metrics.csv", index=False, lineterminator="\n")
    per_block.to_csv(output_dir / "per_block_metrics.csv", index=False, lineterminator="\n")
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    (output_dir / "feature_provenance.json").write_text(
        json.dumps(feature_provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output_dir / "fit_audit.json").write_text(
        json.dumps(result.audit_records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    metadata = {
        "status": "PROVISIONAL_PENDING_VERIFIER",
        "command": command,
        "runtime_seconds": runtime_seconds,
        "alpha_grid": list(ALPHAS),
        "selection_order": [
            "concatenated_inner_source_macro_mae",
            "row_micro_mae",
            "lower_compute_rank",
            "lexical_config_id",
        ],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "scientific_models_run": ["outer_training_mean", "ridge"],
        "not_run": ["random_forest", "xgboost", "dmpnn"],
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
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


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, action="append")
    args = parser.parse_args()
    analysis_path = root / "artifacts" / "v2_r0" / "analysis_all_labels.csv"
    curated_public_path = root / "artifacts" / "curated_records_public.csv"
    outer_path = root / "artifacts" / "v2_r0" / "outer_record_assignments.csv"
    inner_path = root / "artifacts" / "v2_r0" / "inner_basket_manifest.csv"
    started = time.perf_counter()
    feature_frame, metadata, provenance = build_feature_frame(analysis_path, curated_public_path)
    records = pd.read_csv(analysis_path, usecols=[ID_COLUMN, "sealed_block_id"])
    outer = pd.read_csv(outer_path)
    inner = pd.read_csv(inner_path)
    contracts = contracts_from_manifests(records, outer, inner)
    selected_contracts = contracts
    if args.outer_fold:
        selected_contracts = {fold: contracts[fold] for fold in sorted(set(args.outer_fold))}
    command = (
        "python src/r1a_classical.py --output-dir "
        f"{args.output_dir.as_posix()} --run-root {args.run_root.as_posix()}"
    )
    result = run_lobo(
        feature_frame,
        metadata,
        selected_contracts,
        args.run_root,
        progress=True,
    )
    write_outputs(
        args.output_dir,
        result,
        provenance,
        selected_contracts,
        time.perf_counter() - started,
        command,
    )


if __name__ == "__main__":
    main()
