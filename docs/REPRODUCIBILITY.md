# Reproducibility guide

## Scope

This release supports inspection of the reported aggregate findings, deterministic reconstruction of the release-safe curation and split outputs after lawful acquisition of the upstream data, regeneration of manuscript figures, and audit of the frozen evaluation boundary.

It does not provide third-party row-level structures or permeability labels, per-record predictions, residuals, model weights, private caches or credentials.

## 1. Obtain the upstream data

Obtain CycPeptMPDB v1.2 PAMPA records from the official database download route documented in `../DATA_ACCESS.md`. Comply with the upstream terms.

Verify that the exact input used for the frozen analysis contains 7,298 rows and has SHA-256:

```text
02da1cfc18a92b3ae6e70152445b23c05ce6bb0b6ed10fc7c9e141fbd9462fde
```

Do not place that file under version control.

## 2. Create the release-safe environment

```bash
conda env create -f environment.yml
conda activate scaffoldseal-cp
```

The exact upstream DMPNN reproduction used the distinct environment recorded in `BASELINE_ENVIRONMENT.txt`.

## 3. Reconstruct the curation audit

```bash
python paper/supplementary/analyze_curation_source_audit.py --raw "/path/to/CycPeptMPDB_Peptide_Assay_PAMPA.csv" --output-dir paper/supplementary/source_data/curation_source_audit_v1
```

The expected summary is:

- 7,298 raw rows.
- 372 detection-limit rows excluded from continuous regression.
- 6,926 uncensored usable rows.
- 31 compatible duplicate/source-structure groups collapsed.
- 6,895 curated source-structure records.
- 6,862 unique molecules.
- 41 retained sources.

## 4. Inspect the evaluation boundaries

Use `../manifests/README.md` to locate the joint, molecule-grouped, source-only, analogue-only and matched-size manifest families. Governance and freeze records are under `../docs/`.

## 5. Regenerate release-safe figures

```bash
python paper/figures/make_main_figures.py
```

The script consumes released aggregate source tables under `paper/figures/source_data/` and writes rendered figures under `paper/figures/output/`.

## 6. Run public-package checks

```bash
python scripts/verify_release.py
python scripts/generate_checksums.py --check
```

## 7. Understand the remaining boundary

The public archive permits independent inspection of aggregate results, evaluation geometry, curation counts and file provenance. Full rerunning of model fitting additionally requires lawful acquisition of the exact upstream table and the separately documented baseline environment. Scripts that reconstruct robustness statistics from excluded per-record predictions require an authorized local runtime root.
