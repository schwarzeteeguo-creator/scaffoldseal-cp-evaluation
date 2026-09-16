# ScaffoldSeal-CP Evaluation

Release-ready repository for the manuscript *DMPNN Performance Estimates Diverge across Evaluation Boundaries in a Cyclic-Peptide Permeability Benchmark*, prepared for submission to the *Journal of Cheminformatics*.

This directory is intended to be copied **as the repository root** for release `v0.8.1`. Its contents are directly browsable; the repository is not distributed only as a nested ZIP archive.

## Repository map

- `code/`: task-oriented map of the public curation, splitting, evaluation, analysis and figure code.
- `configs/`: map of frozen configuration and dependency records.
- `manifests/`: map of release-safe curation, fold-assignment and evaluation-boundary manifests.
- `results/`: map of aggregate results and figure source data.
- `docs/`: preregistration, governance, provenance, release inventory and reproducibility instructions.
- `paper/manuscript/`: Journal of Cheminformatics main-manuscript PDF and editable LaTeX source.
- `paper/supporting_information/`: Supporting Information PDF, LaTeX source and supplementary figures.
- `paper/complete_review/`: combined manuscript-plus-SI PDF for internal review.
- `scaffoldseal/src/`: canonical executable analysis and evaluation modules.
- `scaffoldseal/tests/`: automated tests.
- `scaffoldseal/artifacts/`: selected release-safe manifests and aggregate outputs.
- `paper/figures/` and `paper/supplementary/`: canonical reporting scripts, source tables and figures.

The `code/`, `configs/`, `manifests/` and `results/` directories are navigation layers. Canonical files remain in their validated locations so that existing commands and imports are not broken.

## Data and redistribution boundary

The study reused CycPeptMPDB v1.2 PAMPA records. This repository does not redistribute upstream molecular structures, row-level permeability labels, per-record predictions, residuals, model weights, credentials, caches or private run directories because a separate redistribution licence for the upstream records was not identified.

The exact locally used upstream file contained 7,298 rows and had SHA-256:

```text
02da1cfc18a92b3ae6e70152445b23c05ce6bb0b6ed10fc7c9e141fbd9462fde
```

See `DATA_ACCESS.md` and `data/README.md` for the access route, curation counts, released column dictionary and reconstruction boundary.

## Publicly included material

- Deterministic curation and split code.
- Frozen governance and protocol records.
- Release-safe record and curated-group manifests without structures or endpoint values.
- Joint, molecule-grouped, source-only, analogue-only and matched-size split manifests.
- Aggregate benchmark, robustness, calibration and uncertainty outputs.
- Figure source-data tables and deterministic figure-generation code.
- Journal of Cheminformatics manuscript and Supporting Information files.
- File-level SHA-256 checksums.

## Environment

For release-safe curation, reporting and tests:

```bash
conda env create -f environment.yml
conda activate scaffoldseal-cp
```

Alternatively:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

The reproduced upstream DMPNN baseline used a separate pinned environment documented in `docs/BASELINE_ENVIRONMENT.txt`.

## Verify the public package

```bash
python scripts/verify_release.py
python scripts/generate_checksums.py --check
```

## Rebuild the release-safe figures

```bash
python paper/figures/make_main_figures.py
```

Some post-confirmatory reconstruction scripts require excluded, locally reconstructed out-of-fold artifacts. Those scripts identify the required authorized runtime input and do not imply that private row-level predictions are part of this release.

## Citation and archive status

- Repository: https://github.com/schwarzeteeguo-creator/scaffoldseal-cp-evaluation
- All-version Zenodo Concept DOI: https://doi.org/10.5281/zenodo.22126299
- Previous immutable `v0.7.7` record: https://doi.org/10.5281/zenodo.22126300
- Exact `v0.8.0` DOI: https://doi.org/10.5281/zenodo.22732455
- Exact `v0.8.1` DOI: https://doi.org/10.5281/zenodo.22787442

The `v0.8.0` and `v0.8.1` GitHub releases are archived at their version-specific DOIs above. The `v0.8.1` release adds the revised submission package and complete author metadata. Do not reuse, delete or move either archived tag.

## Licence

The authors' original code, documentation, figures, aggregate tables and release manifests are provided under the MIT License. This licence does not apply to third-party CycPeptMPDB records or external software. See `LICENSE` and `NOTICE.md`.
