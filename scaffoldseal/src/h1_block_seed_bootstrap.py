from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd


N_REPLICATES = 10_000
N_BLOCKS = 18
N_SEEDS = 5
THRESHOLD = 0.10
SCHEMA = "scaffoldseal-h1-block-seed-bootstrap-v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def file_record(path: Path, root: Path) -> dict[str, object]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def verify_stage_manifest(directory: Path) -> None:
    manifest = json.loads((directory / "artifact_manifest.json").read_text())
    for record in manifest["files"]:
        path = directory / record["relative_path"]
        if path.stat().st_size != int(record["size_bytes"]) or sha256(path) != record["sha256"]:
            raise RuntimeError(f"Frozen artifact mismatch: {path}")


def input_paths(runtime_root: Path) -> dict[str, Path]:
    return {
        "labels": runtime_root / "artifacts/v2_r0/analysis_all_labels.csv",
        "assignments": runtime_root / "artifacts/v2_r0/outer_record_assignments.csv",
        "lobo_predictions": runtime_root
        / "artifacts/r1c0_d0_full_v1/metrics/joint_block_lobo/attempt_0001/oof_predictions.csv",
        "lobo_artifact_manifest": runtime_root
        / "artifacts/r1c0_d0_full_v1/metrics/joint_block_lobo/attempt_0001/artifact_manifest.json",
        "random_predictions": runtime_root
        / "artifacts/h1_random_cv_d0_v1/metrics/molecule_random_5fold/attempt_0001/oof_predictions.csv",
        "random_artifact_manifest": runtime_root
        / "artifacts/h1_random_cv_d0_v1/metrics/molecule_random_5fold/attempt_0001/artifact_manifest.json",
    }


def build_plan(runtime_root: Path, project_root: Path) -> dict[str, object]:
    paths = input_paths(runtime_root)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    verify_stage_manifest(paths["lobo_predictions"].parent)
    verify_stage_manifest(paths["random_predictions"].parent)
    records = {name: file_record(path, runtime_root) for name, path in paths.items()}
    seed_material = {
        "domain": "ScaffoldSeal-CP/H1/outer-block-seed-bootstrap/v1",
        "inputs": records,
        "n_replicates": N_REPLICATES,
        "n_blocks": N_BLOCKS,
        "n_seeds": N_SEEDS,
    }
    rng_seed = int(canonical_sha(seed_material)[:16], 16)
    plan = {
        "schema_version": SCHEMA,
        "scientific_rule": {
            "statistic": "LOBO source-macro MAE minus molecule-random source-macro MAE",
            "resampling": "sample 18 outer blocks and 5 paired model seeds with replacement; use their Cartesian product",
            "block_integrity": "all rows and sources in each sampled block remain together; repeated block copies remain distinct",
            "replicates": N_REPLICATES,
            "interval": "two-sided percentile 95%",
            "support_point_threshold": THRESHOLD,
            "support_ci_rule": "lower bound strictly above zero",
        },
        "rng": {
            "algorithm": "numpy.PCG64",
            "seed_derivation": "first 64 bits of SHA256(canonical seed material)",
            "seed": rng_seed,
            "seed_material_sha256": canonical_sha(seed_material),
        },
        "inputs": records,
        "runner": file_record(project_root / "src/h1_block_seed_bootstrap.py", project_root),
    }
    plan["plan_sha256"] = canonical_sha(plan)
    return plan


def prepare_tables(runtime_root: Path) -> tuple[pd.DataFrame, list[str], list[int]]:
    paths = input_paths(runtime_root)
    labels = pd.read_csv(
        paths["labels"],
        usecols=["curated_id", "source", "sealed_block_id", "permeability"],
    )
    assignments = pd.read_csv(
        paths["assignments"],
        usecols=["curated_id", "sealed_block_id", "outer_fold"],
    )
    if len(labels) != 6895 or labels["curated_id"].nunique() != 6895:
        raise RuntimeError("Frozen label population is not exactly 6,895 unique records")
    if labels["sealed_block_id"].nunique() != N_BLOCKS:
        raise RuntimeError("Frozen population does not contain exactly 18 blocks")
    if assignments["curated_id"].nunique() != 6895:
        raise RuntimeError("Outer assignments do not cover the frozen population")
    check = labels.merge(assignments, on="curated_id", suffixes=("_label", "_assignment"))
    if not (check["sealed_block_id_label"] == check["sealed_block_id_assignment"]).all():
        raise RuntimeError("Block mapping mismatch between frozen inputs")

    arms: list[pd.DataFrame] = []
    for arm, path in (
        ("lobo", paths["lobo_predictions"]),
        ("random", paths["random_predictions"]),
    ):
        pred = pd.read_csv(path)
        expected = {"curated_id", "outer_fold", "seed", "prediction_log10_papp"}
        if set(pred.columns) != expected | {"prediction_normalized"}:
            raise RuntimeError(f"Unexpected prediction columns for {arm}")
        if len(pred) != 34475:
            raise RuntimeError(f"{arm} OOF row count mismatch")
        joined = pred.merge(labels, on="curated_id", validate="many_to_one")
        if joined[["source", "sealed_block_id", "permeability"]].isna().any().any():
            raise RuntimeError(f"{arm} label join is incomplete")
        joined["absolute_error"] = (
            joined["prediction_log10_papp"].astype(float) - joined["permeability"].astype(float)
        ).abs()
        per_source = (
            joined.groupby(["seed", "sealed_block_id", "source"], sort=True)["absolute_error"]
            .mean()
            .reset_index(name="source_mae")
        )
        per_source["arm"] = arm
        arms.append(per_source)
    table = pd.concat(arms, ignore_index=True)
    blocks = sorted(labels["sealed_block_id"].astype(str).unique())
    seeds = sorted(table["seed"].astype(int).unique())
    if len(blocks) != N_BLOCKS or seeds != list(range(N_SEEDS)):
        raise RuntimeError("Block or seed universe mismatch")
    if table.groupby(["arm", "seed"])["source"].nunique().min() != labels["source"].nunique():
        raise RuntimeError("A model arm lacks source coverage")
    return table, blocks, seeds


def point_estimates(table: pd.DataFrame) -> dict[str, float]:
    by_arm_seed = (
        table.groupby(["arm", "seed"], sort=True)["source_mae"].mean().reset_index()
    )
    values = by_arm_seed.groupby("arm")["source_mae"].mean()
    return {
        "lobo_source_macro_mae": float(values["lobo"]),
        "random_source_macro_mae": float(values["random"]),
        "gap_lobo_minus_random": float(values["lobo"] - values["random"]),
    }


def bootstrap(
    table: pd.DataFrame,
    blocks: list[str],
    seeds: list[int],
    rng_seed: int,
    n_replicates: int = N_REPLICATES,
) -> pd.DataFrame:
    grouped: dict[tuple[str, int, str], np.ndarray] = {}
    for (arm, seed, block), group in table.groupby(
        ["arm", "seed", "sealed_block_id"], sort=True
    ):
        grouped[(str(arm), int(seed), str(block))] = group["source_mae"].to_numpy(float)
    rng = np.random.Generator(np.random.PCG64(rng_seed))
    output: list[dict[str, float | int]] = []
    for replicate in range(n_replicates):
        sampled_blocks = rng.choice(blocks, size=len(blocks), replace=True)
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        arm_values: dict[str, float] = {}
        for arm in ("lobo", "random"):
            source_copy_errors: list[np.ndarray] = []
            for block in sampled_blocks:
                for seed in sampled_seeds:
                    source_copy_errors.append(grouped[(arm, int(seed), str(block))])
            arm_values[arm] = float(np.concatenate(source_copy_errors).mean())
        output.append(
            {
                "replicate": replicate,
                "lobo_source_macro_mae": arm_values["lobo"],
                "random_source_macro_mae": arm_values["random"],
                "gap_lobo_minus_random": arm_values["lobo"] - arm_values["random"],
            }
        )
    return pd.DataFrame(output)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fsync_file(path: Path) -> None:
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def atomic_publish(output_dir: Path, builder) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise FileExistsError(f"Canonical output already exists: {output_dir}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.nonfinal-attempt-", dir=output_dir.parent)
    )
    try:
        builder(temporary)
        required = {
            "bootstrap_replicates.csv",
            "bootstrap_summary.json",
            "artifact_manifest.json",
        }
        actual = {path.name for path in temporary.iterdir() if path.is_file()}
        if actual != required:
            raise RuntimeError(f"Atomic output file set mismatch: {sorted(actual)}")
        manifest = json.loads((temporary / "artifact_manifest.json").read_text(encoding="utf-8"))
        for record in manifest["files"]:
            path = temporary / str(record["relative_path"])
            if path.stat().st_size != int(record["size_bytes"]) or sha256(path) != record["sha256"]:
                raise RuntimeError(f"New artifact failed pre-publication verification: {path}")
        for path in sorted(temporary.iterdir()):
            if path.is_file():
                fsync_file(path)
        if output_dir.exists():
            raise FileExistsError(f"Canonical output appeared during build: {output_dir}")
        os.replace(temporary, output_dir)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def execute(runtime_root: Path, project_root: Path, output_dir: Path) -> None:
    plan_path = project_root / "h1_bootstrap_plan.json"
    acceptance_path = project_root / "h1_bootstrap_acceptance.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    rebuilt = build_plan(runtime_root, project_root)
    if rebuilt != plan:
        raise RuntimeError("Frozen bootstrap plan or its inputs changed")
    if not acceptance.get("accepted") or not acceptance.get("execution_authorized"):
        raise RuntimeError("Bootstrap execution is not independently authorized")
    if acceptance.get("plan_sha256") != plan["plan_sha256"]:
        raise RuntimeError("Acceptance is not bound to this plan")

    table, blocks, seeds = prepare_tables(runtime_root)
    point = point_estimates(table)
    replicates = bootstrap(table, blocks, seeds, int(plan["rng"]["seed"]))
    lower, upper = np.quantile(
        replicates["gap_lobo_minus_random"].to_numpy(float), [0.025, 0.975]
    )
    summary = {
        "schema_version": SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "n_replicates": N_REPLICATES,
        "point_estimates": point,
        "gap_percentile_95_ci": {"lower": float(lower), "upper": float(upper)},
        "support_rules": {
            "point_gap_at_least_0_10": bool(point["gap_lobo_minus_random"] >= THRESHOLD),
            "ci_entirely_above_zero": bool(lower > 0.0),
        },
    }
    summary["h1_supported"] = bool(all(summary["support_rules"].values()))
    def build(temporary: Path) -> None:
        replicates.to_csv(
            temporary / "bootstrap_replicates.csv", index=False, lineterminator="\n"
        )
        write_json(temporary / "bootstrap_summary.json", summary)
        files = [
            file_record(temporary / name, temporary)
            for name in ("bootstrap_replicates.csv", "bootstrap_summary.json")
        ]
        manifest = {
            "schema_version": "scaffoldseal-bootstrap-artifacts-v1",
            "plan_sha256": plan["plan_sha256"],
            "files": files,
        }
        manifest["files_sha256"] = canonical_sha(files)
        write_json(temporary / "artifact_manifest.json", manifest)

    atomic_publish(output_dir, build)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.prepare == args.execute:
        raise SystemExit("Choose exactly one of --prepare or --execute")
    if args.prepare:
        write_json(args.project_root / "h1_bootstrap_plan.json", build_plan(args.runtime_root, args.project_root))
        return
    if args.output_dir is None:
        raise SystemExit("--output-dir is required for execution")
    execute(args.runtime_root, args.project_root, args.output_dir)


if __name__ == "__main__":
    main()
