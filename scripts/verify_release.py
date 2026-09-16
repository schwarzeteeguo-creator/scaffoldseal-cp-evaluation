"""Audit the public v0.8.1 release candidate."""

from __future__ import annotations

import csv
import json
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
    "requirements.txt",
    "requirements-reporting.txt",
    "environment.yml",
    "code/README.md",
    "code/curation/README.md",
    "code/splitting/README.md",
    "code/evaluation/README.md",
    "code/analysis/README.md",
    "code/figures/README.md",
    "configs/README.md",
    "manifests/README.md",
    "results/README.md",
    "docs/REPRODUCIBILITY.md",
    "docs/RELEASE_INVENTORY_V0.8.md",
    "docs/RELEASE_NOTES_v0.8.1.md",
    "docs/PUBLIC_PROTOCOL_TIMELINE.md",
    "docs/SPLIT_BOUNDARY_SENSITIVITY_PREREGISTRATION.md",
    "docs/SPLIT_BOUNDARY_MANIFEST_FREEZE.md",
    "paper/manuscript/manuscript_jcheminform_v1.1_DOI.tex",
    "paper/manuscript/ScaffoldSeal_CP_JCheminform_manuscript_v1.1_DOI.pdf",
    "paper/manuscript/references.bib",
    "paper/supporting_information/supporting_information_jcheminform_v1.1_DOI.tex",
    "paper/supporting_information/ScaffoldSeal_CP_JCheminform_supporting_information_v1.1_DOI.pdf",
    "paper/complete_review/ScaffoldSeal_CP_JCheminform_complete_review_package_v1.1_DOI.pdf",
    "paper/figures/make_main_figures.py",
    "paper/figures/output/figure1_study_workflow.png",
    "paper/figures/output/figure2_evidence_geometry.png",
    "paper/figures/output/figure3_main_results.png",
    "paper/figures/output/figure4_failure_heterogeneity.png",
    "paper/supplementary/analyze_curation_source_audit.py",
    "paper/supplementary/source_data/curation_source_audit_v1/curation_manifest_release_safe.csv",
    "paper/supplementary/source_data/curation_source_audit_v1/curated_group_manifest_release_safe.csv",
    "scaffoldseal/artifacts/v2_r0/outer_record_assignments.csv",
    "scaffoldseal/artifacts/h1_random_cv_r0/outer_record_assignments.csv",
    "scaffoldseal/artifacts/split_boundary_sensitivity_v1/evaluation_boundary_ladder_manifest.csv",
    "data/README.md",
    "scaffoldseal/config_v2.yaml",
    "docs/BASELINE_ENVIRONMENT.txt",
}
FORBIDDEN_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".npy",
    ".npz",
    ".gzip",
    ".log",
    ".pyc",
    ".tif",
    ".tiff",
}
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
TEXT_SUFFIXES = {
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
    ".csv",
    ".py",
    ".bib",
    ".cff",
    ".template",
    ".gitignore",
    ".tex",
}
LEAK_PATTERNS = {
    "Windows user path": re.compile(r"[A-Za-z]:[\\/]Users[\\/][^\\/\s]+", re.IGNORECASE),
    "Unix home path": re.compile(r"/home/[^/\s]+/"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "sealed-label vault path": re.compile(r"final_labels_sealed", re.IGNORECASE),
}
EXPECTED_CREATORS = [
    "Guo, Yutao",
    "Jiang, Xujing",
    "Zhang, Zihan",
    "Zhao, Xuezhou",
    "Chen, Mengxi",
    "Zhang, Langzhe",
    "Meng, Xiangyu",
    "Xiong, Guojun",
    "Wu, Dan",
]


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
    warnings: list[str] = []
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

        if (
            path.suffix.lower() in TEXT_SUFFIXES or path.name == ".gitignore"
        ) and path != Path(__file__).resolve():
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
                errors.append(
                    f"risky row-level CSV fields in {relative}: {', '.join(risky)}"
                )

    try:
        zenodo = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"invalid .zenodo.json: {exc}")
    else:
        creators = [item.get("name") for item in zenodo.get("creators", [])]
        if creators != EXPECTED_CREATORS:
            errors.append(f"Zenodo creator order mismatch: {creators}")
        if zenodo.get("version") != "0.8.1":
            errors.append("Zenodo version must be 0.8.1")

    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    if not re.search(r'^version:\s*["\']?0\.8\.1["\']?\s*$', citation, re.MULTILINE):
        errors.append("CITATION.cff version must be 0.8.1")
    if 'doi: "10.5281/zenodo.22126299"' not in citation:
        errors.append("CITATION.cff must record the stable Concept DOI before archiving")
    citation_positions = [citation.find(name.split(", ")[0]) for name in EXPECTED_CREATORS]
    if any(position < 0 for position in citation_positions):
        errors.append("CITATION.cff is missing one or more family names")

    manuscript_release_text = "\n".join(
        [
            (ROOT / "paper/manuscript/manuscript_jcheminform_v1.1_DOI.tex").read_text(
                encoding="utf-8"
            ),
            (ROOT / "paper/manuscript/references.bib").read_text(encoding="utf-8"),
            (
                ROOT
                / "paper/supporting_information/supporting_information_jcheminform_v1.1_DOI.tex"
            ).read_text(encoding="utf-8"),
        ]
    )
    if "10.5281/zenodo.22126300" in manuscript_release_text:
        errors.append("The packaged manuscript still cites the previous v0.7.7 DOI")
    if "10.5281/zenodo.22732455" not in manuscript_release_text:
        errors.append("The packaged manuscript does not cite the preceding archived release DOI")

    total_bytes = sum(path.stat().st_size for path in files)
    if errors:
        print("RELEASE CHECK FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"PASS: {len(files)} files, {total_bytes:,} bytes")
    print("No forbidden artifact types, risky CSV fields, or recognized private paths found.")
    for warning in warnings:
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
