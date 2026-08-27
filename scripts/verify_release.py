"""Check that the candidate contains only the intended public release material."""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "README.md",
    "DATA_ACCESS.md",
    "RELEASE_CHECKLIST.md",
    "LICENSE",
    "CITATION.cff",
    ".zenodo.json",
    "NOTICE.md",
    "requirements-reporting.txt",
    "paper/manuscript_v0.7.md",
    "paper/supplementary/supporting_information_v0.7.md",
    "paper/pdf/ScaffoldSeal_CP_manuscript_submission_draft_v0.7.pdf",
    "paper/pdf/ScaffoldSeal_CP_supporting_information_v0.7.pdf",
    "paper/pdf/ScaffoldSeal_CP_complete_review_package_v0.7.pdf",
    "docs/PUBLIC_PROTOCOL_TIMELINE.md",
    "docs/SPLIT_BOUNDARY_SENSITIVITY_PREREGISTRATION.md",
    "docs/SPLIT_BOUNDARY_MANIFEST_FREEZE.md",
    "paper/references.bib",
    "paper/figures/make_main_figures.py",
    "paper/figures/output/figure1_study_workflow.png",
    "paper/figures/output/figure2_evidence_geometry.png",
    "paper/figures/output/figure3_main_results.png",
    "paper/figures/output/figure4_failure_heterogeneity.png",
    "paper/supplementary/analyze_curation_source_audit.py",
    "paper/supplementary/analyze_d3_coverage_stratification.py",
    "paper/supplementary/source_data/curation_source_audit_v1/curation_manifest_release_safe.csv",
    "paper/supplementary/source_data/curation_source_audit_v1/curated_group_manifest_release_safe.csv",
    "paper/supplementary/source_data/d3_coverage_stratification_v1/d3_coverage_aggregation_sensitivity.csv",
    "scaffoldseal/artifacts/v2_r0/outer_record_assignments.csv",
    "scaffoldseal/artifacts/h1_random_cv_r0/outer_record_assignments.csv",
    "scaffoldseal/artifacts/split_boundary_sensitivity_v1/evaluation_boundary_ladder_manifest.csv",
    "data/README.md",
    "docs/RELEASE_INVENTORY_V0.7.md",
    "scaffoldseal/config_v2.yaml",
    "docs/BASELINE_ENVIRONMENT.txt",
}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".npy", ".npz", ".gzip", ".log", ".pyc", ".tif", ".tiff"}
FORBIDDEN_NAME_FRAGMENTS = {
    "final_labels",
    "development_labeled",
    "analysis_all_labels",
    "curated_records_public.csv",
    "split_manifest_public.csv",
    "oof_predictions",
    "residuals",
    "interval_rows.csv",
}
RISKY_CSV_FIELDS = {
    "canonical_smiles",
    "smiles",
    "permeability",
    "label",
    "target",
    "prediction",
    "residual",
    "raw_ids",
}
TEXT_SUFFIXES = {".md", ".txt", ".yaml", ".yml", ".json", ".csv", ".py", ".bib", ".cff", ".template", ".gitignore"}
LEAK_PATTERNS = {
    "Windows user path": re.compile(r"[A-Za-z]:[\\/]Users[\\/][^\\/\s]+", re.IGNORECASE),
    "Unix home path": re.compile(r"/home/[^/\s]+/"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "sealed-label vault path": re.compile(r"final_labels_sealed", re.IGNORECASE),
}


def all_files() -> list[Path]:
    return sorted(
        (
            path
            for path in ROOT.rglob("*")
            if path.is_file() and ".git" not in path.relative_to(ROOT).parts
        ),
        key=lambda item: item.relative_to(ROOT).as_posix(),
    )


def main() -> int:
    errors: list[str] = []
    files = all_files()

    for relative in sorted(REQUIRED):
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")

    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        lower_name = path.name.lower()

        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden artifact type: {relative}")
        if any(fragment in lower_name for fragment in FORBIDDEN_NAME_FRAGMENTS):
            errors.append(f"forbidden row-level/internal filename: {relative}")
        if path.stat().st_size > 10 * 1024 * 1024:
            errors.append(f"file exceeds 10 MiB release limit: {relative}")

        if (path.suffix.lower() in TEXT_SUFFIXES or path.name == ".gitignore") and path != Path(__file__).resolve():
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                errors.append(f"text file is not UTF-8: {relative}")
                continue
            for label, pattern in LEAK_PATTERNS.items():
                if pattern.search(text):
                    errors.append(f"possible {label} in {relative}")

        if path.suffix.lower() == ".csv":
            try:
                with path.open("r", encoding="utf-8-sig", newline="") as handle:
                    header = next(csv.reader(handle), [])
            except (OSError, UnicodeDecodeError, csv.Error) as exc:
                errors.append(f"cannot inspect CSV header {relative}: {exc}")
                continue
            risky = sorted({field.strip().lower() for field in header} & RISKY_CSV_FIELDS)
            if risky:
                errors.append(f"risky row-level CSV fields in {relative}: {', '.join(risky)}")

    total_bytes = sum(path.stat().st_size for path in files)
    if errors:
        print("RELEASE CHECK FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"PASS: {len(files)} files, {total_bytes:,} bytes")
    print("No forbidden artifact types, risky CSV fields, or recognized private paths found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
