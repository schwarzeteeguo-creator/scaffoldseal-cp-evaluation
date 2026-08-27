from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from rdkit import rdBase

from d123_features import (
    CONTINUOUS_DESCRIPTORS,
    DESCRIPTOR_COLUMNS,
    MAX_MISSING_FRACTION,
    MIN_VARIANCE,
    N_GLOBAL_FEATURES,
    TOPOLOGY_CATEGORIES,
    build_raw_descriptor_frame,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def record(path: Path, root: Path) -> dict[str, object]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError(f"D123 input output already exists: {output}")
    analysis_path = root / "artifacts/v2_r0/analysis_all_labels.csv"
    public_path = root / "artifacts/curated_records_public.csv"
    descriptors = build_raw_descriptor_frame(analysis_path, public_path)
    analysis = pd.read_csv(
        analysis_path,
        usecols=[
            "curated_id",
            "source",
            "analogue_component_id",
            "sealed_block_id",
        ],
    )
    if len(analysis) != 6895 or analysis["curated_id"].nunique() != 6895:
        raise RuntimeError("D123 group metadata population drifted")
    output.mkdir(parents=True)
    descriptor_path = output / "raw_descriptors.csv"
    metadata_path = output / "group_metadata.csv"
    descriptors.to_csv(descriptor_path, index=False, lineterminator="\n")
    analysis.to_csv(metadata_path, index=False, lineterminator="\n")
    provenance = {
        "schema_version": "scaffoldseal-d123-inputs-v1",
        "n_rows": 6895,
        "descriptor_concepts": 13,
        "global_feature_width": N_GLOBAL_FEATURES,
        "descriptor_columns": list(DESCRIPTOR_COLUMNS),
        "continuous_columns": list(CONTINUOUS_DESCRIPTORS),
        "topology_categories": list(TOPOLOGY_CATEGORIES),
        "max_missing_fraction": MAX_MISSING_FRACTION,
        "min_variance": MIN_VARIANCE,
        "analysis_input": record(analysis_path, root),
        "curated_public_input": record(public_path, root),
        "rdkit": rdBase.rdkitVersion,
    }
    provenance_path = output / "provenance.json"
    write_json(provenance_path, provenance)
    files = [
        record(descriptor_path, output),
        record(metadata_path, output),
        record(provenance_path, output),
    ]
    manifest = {
        "schema_version": "scaffoldseal-d123-input-manifest-v1",
        "files": files,
        "files_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
