"""Outcome-independent curation, censoring, and source-metadata audit.

This script does not fit a model or use predictions. It reconstructs the frozen
curation from the public PAMPA table, summarizes censoring by source, checks
which assay-protocol fields are available as structured columns, and writes
release-safe manifests without structures or permeability labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "scaffoldseal" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from build_manifests import curate  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_detection_limit(value_1: object, value_2: object) -> str:
    text = " ".join(
        str(value)
        for value in (value_1, value_2)
        if pd.notna(value) and str(value).strip()
    ).lower()
    if not text:
        return "not_censored"
    if "solubility" in text or "not tested" in text:
        return "solubility_or_not_tested"
    if "no value" in text or "n/a" in text or "%t>100" in text:
        return "no_reportable_value"
    if "not detected" in text or "undetected" in text or "below lod" in text or "blod" in text:
        return "not_detected_or_below_lod"
    if re.search(r"(^|\s)<\s*[-+]?\d", text):
        return "explicit_upper_limit"
    if "set to" in text or "value described" in text:
        return "database_assigned_or_reported_floor"
    return "other_detection_limit_note"


def structured_protocol_audit(columns: list[str]) -> dict[str, object]:
    patterns = {
        "assay_pH": [r"(^|_)ph($|_)"],
        "membrane_composition": [r"membrane", r"lipid"],
        "incubation_time": [r"incubation", r"duration", r"time"],
        "temperature": [r"temperature", r"temp"],
        "donor_acceptor_conditions": [
            r"donor_(well|solution|buffer|compartment|condition)",
            r"acceptor_(well|solution|buffer|compartment|condition)",
            r"(well|solution|buffer|compartment|condition)_donor",
            r"(well|solution|buffer|compartment|condition)_acceptor",
        ],
        "detection_limit_notes": [r"detection_limit"],
    }
    result: dict[str, object] = {}
    for field, field_patterns in patterns.items():
        matches = [
            column
            for column in columns
            if any(re.search(pattern, column.lower()) for pattern in field_patterns)
        ]
        result[field] = {
            "available_as_structured_column": bool(matches),
            "matching_columns": matches,
        }
    return result


def json_ready(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    raw_path = args.raw.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(raw_path, low_memory=False)
    result = curate(raw)
    work = raw.copy()
    work["is_censored"] = work["Detection_Limit_1"].notna() | work["Detection_Limit_2"].notna()
    work["censoring_note_class"] = [
        classify_detection_limit(value_1, value_2)
        for value_1, value_2 in zip(work["Detection_Limit_1"], work["Detection_Limit_2"])
    ]
    work["numeric_permeability"] = pd.to_numeric(work["Permeability"], errors="coerce")

    curated_by_source = (
        result.curated.groupby("source", sort=True)
        .agg(
            curated_records=("curated_id", "size"),
            curated_endpoint_median=("permeability", "median"),
            curated_endpoint_q1=("permeability", lambda values: values.quantile(0.25)),
            curated_endpoint_q3=("permeability", lambda values: values.quantile(0.75)),
            curated_endpoint_min=("permeability", "min"),
            curated_endpoint_max=("permeability", "max"),
        )
        .reset_index()
        .rename(columns={"source": "Source"})
    )
    source_summary = (
        work.groupby("Source", sort=True)
        .agg(
            raw_rows=("ID", "size"),
            censored_rows=("is_censored", "sum"),
            numeric_endpoint_rows=("numeric_permeability", "count"),
        )
        .reset_index()
    )
    source_summary["uncensored_rows"] = source_summary["raw_rows"] - source_summary["censored_rows"]
    source_summary["censored_fraction"] = source_summary["censored_rows"] / source_summary["raw_rows"]
    source_summary = source_summary.merge(curated_by_source, on="Source", how="left", validate="one_to_one")
    source_summary = source_summary.sort_values("Source", kind="stable").reset_index(drop=True)
    source_summary.to_csv(output_dir / "source_censoring_and_endpoint_summary.csv", index=False, lineterminator="\n")

    class_summary = (
        work.loc[work["is_censored"]]
        .groupby("censoring_note_class", sort=True)
        .agg(rows=("ID", "size"), sources=("Source", "nunique"))
        .reset_index()
    )
    class_summary["fraction_of_censored_rows"] = class_summary["rows"] / int(work["is_censored"].sum())
    class_summary.to_csv(output_dir / "censoring_note_class_summary.csv", index=False, lineterminator="\n")

    sentinel_summary = (
        work.loc[work["is_censored"]]
        .groupby("numeric_permeability", dropna=False, sort=True)
        .agg(rows=("ID", "size"), sources=("Source", "nunique"))
        .reset_index()
        .rename(columns={"numeric_permeability": "database_numeric_value"})
    )
    sentinel_summary.to_csv(output_dir / "censored_database_numeric_values.csv", index=False, lineterminator="\n")

    release_curation = result.curation_manifest[
        [
            "raw_id",
            "raw_group_id",
            "source",
            "year",
            "version",
            "is_censored",
            "curation_status",
            "exclusion_or_link_reason",
            "curated_id",
        ]
    ].copy()
    release_curation.to_csv(output_dir / "curation_manifest_release_safe.csv", index=False, lineterminator="\n")

    release_groups = result.curated[
        [
            "curated_id",
            "molecule_id",
            "source",
            "year",
            "version",
            "topology_signature",
            "ring_size",
            "main_chain_length",
            "raw_ids_all",
            "raw_ids_used",
            "n_raw_rows",
            "n_uncensored_used",
        ]
    ].copy()
    release_groups.to_csv(output_dir / "curated_group_manifest_release_safe.csv", index=False, lineterminator="\n")

    censored_manifest = result.curation_manifest.loc[result.curation_manifest["is_censored"]]
    source_with_censoring = source_summary.loc[source_summary["censored_rows"] > 0].copy()
    max_row = source_with_censoring.sort_values(
        ["censored_fraction", "censored_rows", "Source"], ascending=[False, False, True]
    ).iloc[0]
    summary = {
        "schema_version": "scaffoldseal-curation-source-audit-v1",
        "raw_file_sha256": sha256_file(raw_path),
        "flow": result.flow,
        "censoring": {
            "sources_with_censored_rows": int(source_with_censoring["Source"].nunique()),
            "median_source_censored_fraction_among_all_sources": float(source_summary["censored_fraction"].median()),
            "maximum_source_censored_fraction": float(max_row["censored_fraction"]),
            "maximum_fraction_source": str(max_row["Source"]),
            "maximum_fraction_source_raw_rows": int(max_row["raw_rows"]),
            "maximum_fraction_source_censored_rows": int(max_row["censored_rows"]),
            "censored_rows_linked_to_retained_groups": int(
                (censored_manifest["curation_status"] == "excluded_linked_to_curated_group").sum()
            ),
            "censored_rows_without_retained_group": int(
                (censored_manifest["curation_status"] == "excluded").sum()
            ),
            "note_classes": {
                str(row.censoring_note_class): int(row.rows)
                for row in class_summary.itertuples(index=False)
            },
        },
        "structured_protocol_fields": structured_protocol_audit([str(column) for column in raw.columns]),
        "interpretation_boundary": (
            "The source table does not provide structured PAMPA pH, membrane-composition, "
            "incubation-time, temperature, or donor/acceptor-condition fields. Detection-limit "
            "notes are heterogeneous and include database-assigned floors, explicit upper limits, "
            "non-detection, missing reportable values, and solubility or non-testing failures. "
            "The database numeric values for these rows are therefore not treated as interchangeable "
            "quantitative censoring bounds."
        ),
    }
    (output_dir / "curation_source_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=json_ready) + "\n",
        encoding="utf-8",
    )

    output_hashes = {
        path.name: sha256_file(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "SHA256SUMS"
    }
    (output_dir / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in output_hashes.items()),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=json_ready))


if __name__ == "__main__":
    main()
