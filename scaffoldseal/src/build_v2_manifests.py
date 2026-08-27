"""Build ScaffoldSeal-CP v2 R0 cross-validation governance artifacts.

This module never creates, reads, or imports a retrospective final-test label
boundary. It reconstructs all 6,895 labels from the public raw PAMPA table by
calling the frozen curation implementation, then joins only the frozen public
curation/group mappings. Fold construction is outcome-blind.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold

from build_manifests import curate
from split_safe import contract_manifest, contracts_from_manifests, fit_boundary_policy


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config_v2.yaml"
MANIFEST_FILES = (
    "outer_record_assignments.csv",
    "outer_fold_manifest.csv",
    "inner_basket_manifest.csv",
    "comparison_fold_manifest.csv",
    "pre_fit_contract_manifest.csv",
)
FIT_BOUNDARY_POLICY_FILE = "fit_boundary_policy.json"
DMPNN_NO_LEARNING_SMOKE_FILE = "dmpnn_no_learning_smoke.json"
LABEL_COLUMN = "permeability"
EXPECTED_CURATED_ROWS = 6895
EXPECTED_MOLECULES = 6862
EXPECTED_BLOCKS = 18
EXPECTED_COMPONENTS = 305
LEGACY_PROJECTION_COLUMNS = {
    "frozen_curated_public": (
        "curated_id",
        "molecule_id",
        "canonical_smiles",
        "source",
        "topology_signature",
        "ring_size",
    ),
    "frozen_components": ("molecule_id", "analogue_component_id"),
    "frozen_blocks": ("sealed_block_id",),
    "frozen_group_mapping": (
        "curated_id",
        "molecule_id",
        "source",
        "analogue_component_id",
        "sealed_block_id",
    ),
    "accessible_former_development": ("curated_id", LABEL_COLUMN),
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def write_csv(frame: pd.DataFrame, path: Path) -> str:
    payload = csv_bytes(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return sha256_bytes(payload)


def read_legacy_projection(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    """Read and canonically order only v2-allowed legacy columns."""
    frame = pd.read_csv(path, usecols=list(columns))
    return frame.loc[:, list(columns)].copy()


def projection_sha256(frame: pd.DataFrame) -> str:
    return sha256_bytes(csv_bytes(frame))


def resolve_input(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def find_single_raw(repo_root: Path, pattern: str) -> Path:
    matches = sorted(repo_root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one raw PAMPA file, found {len(matches)}")
    return matches[0].resolve()


def greedy_group_folds(
    group_sizes: pd.DataFrame,
    group_column: str,
    n_folds: int,
) -> tuple[dict[str, int], list[int]]:
    """Assign intact groups by the frozen descending-size greedy rule."""
    required = {group_column, "n_curated_rows"}
    if set(group_sizes.columns) != required:
        raise ValueError(f"Expected exactly {sorted(required)}")
    if group_sizes[group_column].duplicated().any():
        raise ValueError("Group identifiers must be unique")
    ordered = group_sizes.assign(
        **{group_column: group_sizes[group_column].astype(str)}
    ).sort_values(
        ["n_curated_rows", group_column],
        ascending=[False, True],
        kind="stable",
    )
    totals = [0] * n_folds
    assignment: dict[str, int] = {}
    for row in ordered.itertuples(index=False):
        group_id = str(getattr(row, group_column))
        size = int(row.n_curated_rows)
        fold_index = min(range(n_folds), key=lambda index: (totals[index], index))
        assignment[group_id] = fold_index + 1
        totals[fold_index] += size
    return assignment, totals


def chiral_murcko(canonical_smiles: str, molecule_id: str) -> str:
    mol = Chem.MolFromSmiles(str(canonical_smiles))
    if mol is None:
        raise ValueError(f"Frozen canonical structure cannot be parsed: {molecule_id}")
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=True
    )
    return scaffold if scaffold else f"NO_SCAFFOLD:{molecule_id}"


def reconstruct_labels(raw_path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    raw = pd.read_csv(raw_path, low_memory=False)
    result = curate(raw)
    labels = result.curated[["curated_id", LABEL_COLUMN]].copy()
    labels = labels.sort_values("curated_id", kind="stable").reset_index(drop=True)
    if len(labels) != EXPECTED_CURATED_ROWS or labels["curated_id"].duplicated().any():
        raise RuntimeError("Raw-only curation did not reconstruct 6,895 unique curated IDs")
    return labels, result.flow


def load_frozen_records(
    repo_root: Path,
    config: dict,
    overrides: dict[str, Path] | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    inputs = config["inputs"]
    overrides = overrides or {}

    def selected_path(key: str) -> Path:
        return Path(overrides[key]).resolve() if key in overrides else resolve_input(
            repo_root, inputs[key]
        )

    projections = {
        key: read_legacy_projection(selected_path(key), columns)
        for key, columns in LEGACY_PROJECTION_COLUMNS.items()
        if key != "accessible_former_development"
    }
    projection_hashes = {
        key: projection_sha256(frame) for key, frame in projections.items()
    }
    curated = projections["frozen_curated_public"]
    components = projections["frozen_components"]
    blocks = projections["frozen_blocks"]
    group_mapping = projections["frozen_group_mapping"]
    if len(curated) != EXPECTED_CURATED_ROWS:
        raise RuntimeError("Frozen curated record count changed")
    if curated["molecule_id"].nunique() != EXPECTED_MOLECULES:
        raise RuntimeError("Frozen molecule count changed")
    if components["analogue_component_id"].nunique() != EXPECTED_COMPONENTS:
        raise RuntimeError("Frozen component count changed")
    if len(blocks) != EXPECTED_BLOCKS or blocks["sealed_block_id"].nunique() != EXPECTED_BLOCKS:
        raise RuntimeError("Frozen block count changed")
    if group_mapping["curated_id"].duplicated().any():
        raise RuntimeError("Frozen group mapping contains duplicate curated IDs")

    records = curated.merge(
        group_mapping,
        on=["curated_id", "molecule_id", "source"],
        how="left",
        validate="one_to_one",
    )
    if records[["analogue_component_id", "sealed_block_id"]].isna().any().any():
        raise RuntimeError("Frozen group mapping is incomplete")
    component_check = records[["molecule_id", "analogue_component_id"]].drop_duplicates()
    expected_components = components[["molecule_id", "analogue_component_id"]]
    merged = component_check.merge(
        expected_components,
        on=["molecule_id", "analogue_component_id"],
        how="outer",
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        raise RuntimeError("Frozen molecule/component mapping mismatch")
    return records, projection_hashes


def build_outer(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    blocks = sorted(records["sealed_block_id"].astype(str).unique())
    if len(blocks) != EXPECTED_BLOCKS:
        raise RuntimeError("Expected exactly 18 outer blocks")
    fold_of_block = {block: index + 1 for index, block in enumerate(blocks)}
    outer_records = records[
        ["curated_id", "molecule_id", "source", "analogue_component_id", "sealed_block_id"]
    ].copy()
    outer_records["outer_fold"] = outer_records["sealed_block_id"].map(fold_of_block)
    outer_records["outer_test_block"] = outer_records["sealed_block_id"]
    outer_records = outer_records.sort_values("curated_id", kind="stable").reset_index(drop=True)

    block_stats = (
        records.groupby("sealed_block_id", sort=True)
        .agg(
            n_curated_rows=("curated_id", "size"),
            n_sources=("source", "nunique"),
            n_analogue_components=("analogue_component_id", "nunique"),
        )
        .reset_index()
    )
    fold_rows = []
    inner_rows = []
    for outer_fold, test_block in enumerate(blocks, start=1):
        for block in blocks:
            stat = block_stats.loc[block_stats["sealed_block_id"] == block].iloc[0]
            fold_rows.append(
                {
                    "outer_fold": outer_fold,
                    "outer_test_block": test_block,
                    "sealed_block_id": block,
                    "role": "test" if block == test_block else "train",
                    "n_curated_rows": int(stat["n_curated_rows"]),
                    "n_sources": int(stat["n_sources"]),
                    "n_analogue_components": int(stat["n_analogue_components"]),
                }
            )
        available = block_stats.loc[block_stats["sealed_block_id"] != test_block, [
            "sealed_block_id", "n_curated_rows"
        ]].copy()
        assignment, _ = greedy_group_folds(available, "sealed_block_id", 4)
        for row in available.sort_values("sealed_block_id", kind="stable").itertuples(index=False):
            inner_rows.append(
                {
                    "outer_fold": outer_fold,
                    "outer_test_block": test_block,
                    "sealed_block_id": row.sealed_block_id,
                    "inner_basket": assignment[str(row.sealed_block_id)],
                    "n_curated_rows": int(row.n_curated_rows),
                }
            )
    outer_manifest = pd.DataFrame(fold_rows).sort_values(
        ["outer_fold", "sealed_block_id"], kind="stable"
    ).reset_index(drop=True)
    inner_manifest = pd.DataFrame(inner_rows).sort_values(
        ["outer_fold", "inner_basket", "sealed_block_id"], kind="stable"
    ).reset_index(drop=True)
    return outer_records, outer_manifest, inner_manifest


def assign_group_column(records: pd.DataFrame, group_column: str, n_folds: int = 5) -> pd.Series:
    sizes = (
        records.groupby(group_column, sort=True)["curated_id"]
        .size()
        .rename("n_curated_rows")
        .reset_index()
    )
    assignment, _ = greedy_group_folds(sizes, group_column, n_folds)
    return records[group_column].astype(str).map(assignment).astype(int)


def build_comparison(records: pd.DataFrame) -> pd.DataFrame:
    work = records[
        ["curated_id", "molecule_id", "canonical_smiles", "source", "analogue_component_id"]
    ].copy()
    molecule_scaffolds = (
        work[["molecule_id", "canonical_smiles"]]
        .drop_duplicates()
        .sort_values("molecule_id", kind="stable")
    )
    molecule_scaffolds["murcko_chiral"] = [
        chiral_murcko(smiles, molecule_id)
        for molecule_id, smiles in molecule_scaffolds[["molecule_id", "canonical_smiles"]].itertuples(index=False)
    ]
    work = work.merge(
        molecule_scaffolds[["molecule_id", "murcko_chiral"]],
        on="molecule_id",
        how="left",
        validate="many_to_one",
    )
    work["molecule_id_fold"] = assign_group_column(work, "molecule_id")
    work["murcko_chiral_fold"] = assign_group_column(work, "murcko_chiral")
    work["source_fold"] = assign_group_column(work, "source")
    work["analogue_component_id_fold"] = assign_group_column(work, "analogue_component_id")
    return work[
        [
            "curated_id",
            "molecule_id",
            "murcko_chiral",
            "source",
            "analogue_component_id",
            "molecule_id_fold",
            "murcko_chiral_fold",
            "source_fold",
            "analogue_component_id_fold",
        ]
    ].sort_values("curated_id", kind="stable").reset_index(drop=True)


def assert_label_blind_manifests(frames: Iterable[pd.DataFrame]) -> None:
    forbidden = {LABEL_COLUMN, "label", "target", "outcome", "papp"}
    fragments = ("permeab", "label", "target", "outcome", "papp", "replicate")
    for frame in frames:
        lowered = {str(column).lower() for column in frame.columns}
        if lowered & forbidden:
            raise RuntimeError("Outcome column present in a fold manifest")
        if any(any(fragment in column for fragment in fragments) for column in lowered):
            raise RuntimeError("Outcome-derived column name present in a fold manifest")


def verify_outer_isolation(records: pd.DataFrame) -> dict[str, int]:
    source_overlap = 0
    component_overlap = 0
    for test_block in sorted(records["sealed_block_id"].unique()):
        test = records.loc[records["sealed_block_id"] == test_block]
        train = records.loc[records["sealed_block_id"] != test_block]
        source_overlap += len(set(test["source"]) & set(train["source"]))
        component_overlap += len(
            set(test["analogue_component_id"]) & set(train["analogue_component_id"])
        )
    return {
        "outer_source_overlap_total": int(source_overlap),
        "outer_component_overlap_total": int(component_overlap),
    }


def build(
    config_path: Path,
    output_override: Path | None = None,
    legacy_input_overrides: dict[str, Path] | None = None,
) -> dict[str, object]:
    config_path = config_path.resolve()
    repo_root = config_path.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw_path = find_single_raw(repo_root, config["inputs"]["raw_pampa_glob"])
    records, legacy_projection_hashes = load_frozen_records(
        repo_root, config, legacy_input_overrides
    )
    labels, curation_flow = reconstruct_labels(raw_path)
    if set(records["curated_id"]) != set(labels["curated_id"]):
        raise RuntimeError("Raw-only labels do not match the 6,895 frozen curated IDs")
    analysis = records[
        [
            "curated_id", "molecule_id", "source", "analogue_component_id",
            "sealed_block_id", "canonical_smiles", "topology_signature", "ring_size",
        ]
    ].merge(labels, on="curated_id", validate="one_to_one")
    analysis = analysis.sort_values("curated_id", kind="stable").reset_index(drop=True)

    if legacy_input_overrides and "accessible_former_development" in legacy_input_overrides:
        development_path = Path(
            legacy_input_overrides["accessible_former_development"]
        ).resolve()
    else:
        development_path = resolve_input(
            repo_root, config["inputs"]["accessible_former_development"]
        )
    former = read_legacy_projection(
        development_path,
        LEGACY_PROJECTION_COLUMNS["accessible_former_development"],
    )
    legacy_projection_hashes["accessible_former_development"] = projection_sha256(
        former
    )
    comparison = labels.merge(
        former,
        on="curated_id",
        how="inner",
        suffixes=("_rebuilt", "_frozen"),
        validate="one_to_one",
    )
    delta = np.abs(
        comparison[f"{LABEL_COLUMN}_rebuilt"].to_numpy(float)
        - comparison[f"{LABEL_COLUMN}_frozen"].to_numpy(float)
    )
    mismatch_count = int(np.count_nonzero(~np.isclose(delta, 0.0, atol=1e-12, rtol=0.0)))
    if mismatch_count:
        raise RuntimeError("Raw-only medians differ from accessible former development labels")

    outer_records, outer_manifest, inner_manifest = build_outer(records)
    comparison_manifest = build_comparison(records)
    contracts = contracts_from_manifests(
        records[["curated_id", "sealed_block_id"]],
        outer_records,
        inner_manifest,
    )
    pre_fit_manifest = contract_manifest(contracts)
    assert_label_blind_manifests(
        [
            outer_records,
            outer_manifest,
            inner_manifest,
            comparison_manifest,
            pre_fit_manifest,
        ]
    )
    isolation = verify_outer_isolation(records)
    if any(isolation.values()):
        raise RuntimeError("Frozen joint blocks fail outer source/component isolation")

    output_dir = (
        output_override.resolve()
        if output_override is not None
        else resolve_input(repo_root, config["outputs"]["directory"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "outer_record_assignments.csv": outer_records,
        "outer_fold_manifest.csv": outer_manifest,
        "inner_basket_manifest.csv": inner_manifest,
        "comparison_fold_manifest.csv": comparison_manifest,
        "pre_fit_contract_manifest.csv": pre_fit_manifest,
        "analysis_all_labels.csv": analysis,
    }
    hashes = {name: write_csv(frame, output_dir / name) for name, frame in outputs.items()}
    policy = fit_boundary_policy()
    policy_payload = (json.dumps(policy, indent=2, sort_keys=True) + "\n").encode("utf-8")
    (output_dir / FIT_BOUNDARY_POLICY_FILE).write_bytes(policy_payload)
    hashes[FIT_BOUNDARY_POLICY_FILE] = sha256_bytes(policy_payload)
    smoke_source = (repo_root / "audit" / DMPNN_NO_LEARNING_SMOKE_FILE).resolve()
    smoke_payload = smoke_source.read_bytes()
    smoke = json.loads(smoke_payload)
    if (
        smoke.get("status") != "PASS"
        or smoke.get("r1_authorized") is not False
        or smoke.get("real_model_fit_or_weight_update_performed") is not False
        or not all(smoke.get("assertions", {}).values())
    ):
        raise RuntimeError("Pinned DMPNN no-learning smoke evidence is missing or invalid")
    (output_dir / DMPNN_NO_LEARNING_SMOKE_FILE).write_bytes(smoke_payload)
    hashes[DMPNN_NO_LEARNING_SMOKE_FILE] = sha256_bytes(smoke_payload)
    raw_manifest = f"{sha256_file(raw_path)}  {raw_path.as_posix()}\n".encode("utf-8")
    (output_dir / "data_manifest_raw.sha256").write_bytes(raw_manifest)
    hashes["data_manifest_raw.sha256"] = sha256_bytes(raw_manifest)

    full_file_hashes = {
        "raw_pampa_csv": sha256_file(raw_path),
        "config_v2.yaml": sha256_file(config_path),
        "src/build_v2_manifests.py": sha256_file(Path(__file__).resolve()),
        "src/split_safe.py": sha256_file((repo_root / "src" / "split_safe.py").resolve()),
        "src/dmpnn_no_learning_smoke.py": sha256_file(
            (repo_root / "src" / "dmpnn_no_learning_smoke.py").resolve()
        ),
        "audit/dmpnn_no_learning_smoke.json": sha256_file(smoke_source),
    }
    summary = {
        "project": config["project"],
        "version": config["version"],
        "status": "BUILDER_REPAIR7_PASS_REVERIFICATION_REQUIRED",
        "previous_verifier_verdict": "FAIL_REPAIR6_ADAPTER_AND_RUN_MUTATION",
        "r1_authorized": False,
        "label_reconstruction_source": "public_raw_pampa_plus_frozen_curation_rules",
        "retired_partition_column_parsed": False,
        "curated_rows": int(len(records)),
        "unique_molecules": int(records["molecule_id"].nunique()),
        "sources": int(records["source"].nunique()),
        "analogue_components": int(records["analogue_component_id"].nunique()),
        "joint_blocks": int(records["sealed_block_id"].nunique()),
        "outer_folds": int(outer_records["outer_fold"].nunique()),
        "inner_baskets_per_outer_fold": 4,
        "comparison_folds": 5,
        "accessible_former_development_rows_checked": int(len(comparison)),
        "accessible_former_development_median_mismatches": mismatch_count,
        "accessible_former_development_max_absolute_delta": float(delta.max()) if len(delta) else None,
        "curation_flow": {key: int(value) for key, value in curation_flow.items()},
        "leakage_checks": isolation,
        "manifest_schemas_label_blind": True,
        "pre_fit_contract": {
            "outer_contracts": len(contracts),
            "contract_rows": len(pre_fit_manifest),
            "fit_id_hashes_recorded": True,
            "actual_manifest_guard_pass": True,
            "typed_fit_boundary_policy": FIT_BOUNDARY_POLICY_FILE,
            "generic_fit_kwargs": False,
            "caller_supplied_loss_curves": False,
            "outcome_feature_denylist_enforced": True,
            "fresh_run_state_required": True,
            "inner_fit_execution_identity_bound": True,
            "four_history_identity_consistency_required": True,
            "outer_frame_prediction_identity_bound": True,
            "outer_frame_prediction_single_use": True,
            "outer_frame_prediction_handle_only": True,
            "outer_frame_authoritative_values_dtypes_order_index_hashed": True,
            "outer_frame_fitted_state_executor_private": True,
            "outer_frame_model_checkpoint_content_verified_pre_post_prediction": True,
            "outer_frame_adapter_config_runtime_verified_pre_post_prediction": True,
            "outer_frame_run_context_immutable_snapshot_verified_pre_post_prediction": True,
            "outer_frame_prediction_telemetry_detached": True,
            "outer_frame_prediction_telemetry_scalar_only": True,
            "outer_frame_prediction_final_post_output_state_verification": True,
            "outer_frame_prediction_failure_audited": True,
            "pinned_dmpnn_no_learning_smoke": DMPNN_NO_LEARNING_SMOKE_FILE,
        },
        "full_file_sha256": dict(sorted(full_file_hashes.items())),
        "legacy_allowed_projection_sha256": dict(
            sorted(legacy_projection_hashes.items())
        ),
        "outputs_sha256": dict(sorted(hashes.items())),
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
            "platform": platform.platform(),
        },
    }
    summary_payload = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8")
    (output_dir / "build_summary_v2.json").write_bytes(summary_payload)
    hashes["build_summary_v2.json"] = sha256_bytes(summary_payload)
    checksum_payload = "".join(
        f"{digest}  {name}\n" for name, digest in sorted(hashes.items())
    ).encode("utf-8")
    (output_dir / "SHA256SUMS").write_bytes(checksum_payload)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build(args.config, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
