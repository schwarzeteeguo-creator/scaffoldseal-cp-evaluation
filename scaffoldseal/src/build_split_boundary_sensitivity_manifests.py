"""Build outcome-blind manifests for post-confirmatory split sensitivities."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "artifacts/v2_r0/comparison_fold_manifest.csv"
DEFAULT_JOINT = ROOT / "artifacts/v2_r0/outer_record_assignments.csv"
DEFAULT_OUTPUT = ROOT / "artifacts/split_boundary_sensitivity_v1"
ASSIGNMENT_SEED = 20260810
TARGET_SIZES = (16, 16, 1518, 11, 4010, 3, 842, 4, 36, 18, 17, 105, 10, 8, 249, 18, 7, 7)
FORBIDDEN_FRAGMENTS = ("permeab", "label", "target", "outcome", "papp", "replicate")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def write_csv(frame: pd.DataFrame, path: Path) -> dict[str, object]:
    payload = csv_bytes(frame)
    path.write_bytes(payload)
    return {"relative_path": path.name, "size_bytes": len(payload), "sha256": sha256_bytes(payload)}


def assert_label_blind(frame: pd.DataFrame) -> None:
    lowered = [str(column).lower() for column in frame.columns]
    if any(fragment in column for column in lowered for fragment in FORBIDDEN_FRAGMENTS):
        raise RuntimeError(f"Outcome-related column in manifest: {frame.columns.tolist()}")


def greedy_inner_baskets(bin_sizes: pd.DataFrame) -> dict[int, int]:
    ordered = bin_sizes.sort_values(
        ["n_curated_rows", "matched_size_random_fold"],
        ascending=[False, True],
        kind="stable",
    )
    totals = [0, 0, 0, 0]
    assignment: dict[int, int] = {}
    for row in ordered.itertuples(index=False):
        basket_index = min(range(4), key=lambda index: (totals[index], index))
        fold = int(row.matched_size_random_fold)
        assignment[fold] = basket_index + 1
        totals[basket_index] += int(row.n_curated_rows)
    return assignment


def matched_size_assignment(comparison: pd.DataFrame) -> dict[str, int]:
    groups = (
        comparison.groupby("molecule_id", sort=True)["curated_id"]
        .size()
        .rename("n_curated_rows")
        .reset_index()
        .sort_values("molecule_id", kind="stable")
        .reset_index(drop=True)
    )
    rng = np.random.Generator(np.random.PCG64(ASSIGNMENT_SEED))
    first_order = rng.permutation(len(groups))
    groups = groups.iloc[first_order].reset_index(drop=True)
    multi = groups.loc[groups["n_curated_rows"] > 1].reset_index(drop=True)
    singleton_ids = groups.loc[groups["n_curated_rows"] == 1, "molecule_id"].to_numpy()

    remaining = np.asarray(TARGET_SIZES, dtype=int)
    assignment: dict[str, int] = {}
    for row in multi.itertuples(index=False):
        size = int(row.n_curated_rows)
        eligible = np.flatnonzero(remaining >= size)
        if not len(eligible):
            raise RuntimeError(f"No target bin can hold molecule group of size {size}")
        probabilities = remaining[eligible].astype(float)
        probabilities /= probabilities.sum()
        selected = int(rng.choice(eligible, p=probabilities))
        assignment[str(row.molecule_id)] = selected + 1
        remaining[selected] -= size

    singleton_ids = singleton_ids[rng.permutation(len(singleton_ids))]
    slots = np.concatenate(
        [np.repeat(index + 1, int(capacity)) for index, capacity in enumerate(remaining)]
    )
    slots = slots[rng.permutation(len(slots))]
    if len(slots) != len(singleton_ids):
        raise RuntimeError("Residual capacity does not equal singleton molecule count")
    for molecule_id, fold in zip(singleton_ids, slots, strict=True):
        assignment[str(molecule_id)] = int(fold)

    if len(assignment) != groups["molecule_id"].nunique():
        raise RuntimeError("Not every molecule received one matched-size fold")
    return assignment


def build(comparison_path: Path, joint_path: Path) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    comparison = pd.read_csv(comparison_path)
    joint = pd.read_csv(joint_path)
    required_comparison = {
        "curated_id",
        "molecule_id",
        "source",
        "analogue_component_id",
        "molecule_id_fold",
        "source_fold",
        "analogue_component_id_fold",
    }
    required_joint = {"curated_id", "sealed_block_id", "outer_fold"}
    if not required_comparison.issubset(comparison.columns):
        raise RuntimeError("Comparison manifest is missing required columns")
    if not required_joint.issubset(joint.columns):
        raise RuntimeError("Joint assignment manifest is missing required columns")
    if len(comparison) != 6895 or comparison["curated_id"].nunique() != 6895:
        raise RuntimeError("Comparison manifest must contain 6,895 unique curated records")

    assignment = matched_size_assignment(comparison)
    matched = comparison[["curated_id", "molecule_id"]].copy()
    matched["matched_size_random_fold"] = matched["molecule_id"].astype(str).map(assignment)
    matched = matched.sort_values("curated_id", kind="stable").reset_index(drop=True)
    if matched["matched_size_random_fold"].isna().any():
        raise RuntimeError("Matched-size assignment is incomplete")
    matched["matched_size_random_fold"] = matched["matched_size_random_fold"].astype(int)
    if matched.groupby("molecule_id")["matched_size_random_fold"].nunique().max() != 1:
        raise RuntimeError("A molecule crosses matched-size folds")
    observed_sizes = tuple(
        int(matched.loc[matched["matched_size_random_fold"] == fold].shape[0])
        for fold in range(1, 19)
    )
    if observed_sizes != TARGET_SIZES:
        raise RuntimeError(f"Matched-size targets failed: {observed_sizes}")

    outer_rows: list[dict[str, object]] = []
    inner_rows: list[dict[str, object]] = []
    size_frame = pd.DataFrame(
        {
            "matched_size_random_fold": range(1, 19),
            "n_curated_rows": TARGET_SIZES,
        }
    )
    for outer_fold in range(1, 19):
        for row in size_frame.itertuples(index=False):
            fold = int(row.matched_size_random_fold)
            outer_rows.append(
                {
                    "outer_fold": outer_fold,
                    "matched_size_random_fold": fold,
                    "role": "test" if fold == outer_fold else "train",
                    "n_curated_rows": int(row.n_curated_rows),
                }
            )
        available = size_frame.loc[size_frame["matched_size_random_fold"] != outer_fold]
        baskets = greedy_inner_baskets(available)
        for row in available.sort_values("matched_size_random_fold", kind="stable").itertuples(index=False):
            fold = int(row.matched_size_random_fold)
            inner_rows.append(
                {
                    "outer_fold": outer_fold,
                    "matched_size_random_fold": fold,
                    "inner_basket": baskets[fold],
                    "n_curated_rows": int(row.n_curated_rows),
                }
            )

    outer_manifest = pd.DataFrame(outer_rows)
    inner_manifest = pd.DataFrame(inner_rows).sort_values(
        ["outer_fold", "inner_basket", "matched_size_random_fold"], kind="stable"
    ).reset_index(drop=True)
    ladder = comparison[
        [
            "curated_id",
            "molecule_id",
            "source",
            "analogue_component_id",
            "molecule_id_fold",
            "source_fold",
            "analogue_component_id_fold",
        ]
    ].merge(
        matched[["curated_id", "matched_size_random_fold"]], on="curated_id", validate="one_to_one"
    ).merge(
        joint[["curated_id", "sealed_block_id", "outer_fold"]].rename(
            columns={"outer_fold": "joint_outer_fold"}
        ),
        on="curated_id",
        validate="one_to_one",
    )
    ladder = ladder.sort_values("curated_id", kind="stable").reset_index(drop=True)

    for group_column, fold_column in (
        ("molecule_id", "molecule_id_fold"),
        ("source", "source_fold"),
        ("analogue_component_id", "analogue_component_id_fold"),
        ("molecule_id", "matched_size_random_fold"),
        ("sealed_block_id", "joint_outer_fold"),
    ):
        if ladder.groupby(group_column)[fold_column].nunique().max() != 1:
            raise RuntimeError(f"Group crosses frozen boundary: {group_column} -> {fold_column}")

    frames = {
        "matched_size_random_record_assignments.csv": matched,
        "matched_size_random_outer_manifest.csv": outer_manifest,
        "matched_size_random_inner_baskets.csv": inner_manifest,
        "evaluation_boundary_ladder_manifest.csv": ladder,
    }
    for frame in frames.values():
        assert_label_blind(frame)
    summary: dict[str, object] = {
        "schema_version": "scaffoldseal-split-boundary-sensitivity-manifests-v1",
        "analysis_status": "post-confirmatory outcome-blind manifest freeze",
        "training_or_tuning_performed": False,
        "assignment_rng": "numpy.PCG64",
        "assignment_seed": ASSIGNMENT_SEED,
        "n_records": 6895,
        "n_molecules": int(comparison["molecule_id"].nunique()),
        "matched_size_target_counts": list(TARGET_SIZES),
        "existing_boundary_fold_counts": {
            column: {str(int(k)): int(v) for k, v in comparison.groupby(column).size().items()}
            for column in ("molecule_id_fold", "source_fold", "analogue_component_id_fold")
        },
        "input_sha256": {
            "comparison_fold_manifest.csv": hashlib.sha256(comparison_path.read_bytes()).hexdigest(),
            "outer_record_assignments.csv": hashlib.sha256(joint_path.read_bytes()).hexdigest(),
        },
    }
    return frames, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--joint", type=Path, default=DEFAULT_JOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frames, summary = build(args.comparison.resolve(), args.joint.resolve())
    file_records = [write_csv(frame, args.output / name) for name, frame in frames.items()]
    summary["files"] = file_records
    payload = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8")
    (args.output / "manifest_summary.json").write_bytes(payload)
    print(payload.decode("utf-8"))


if __name__ == "__main__":
    main()

