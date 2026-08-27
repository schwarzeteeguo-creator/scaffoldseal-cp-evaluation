# ScaffoldSeal-CP Evaluation

This folder is the cleaned **v0.7 GitHub release package** for the manuscript *DMPNN Performance Estimates Diverge across Evaluation Boundaries in a Cyclic-Peptide Permeability Benchmark*. It is prepared for formal GitHub Release publication and subsequent Zenodo archiving.

Suggested repository name: `scaffoldseal-cp-evaluation`.

## Included

- `paper/manuscript_v0.7.md` and `paper/supplementary/supporting_information_v0.7.md`.
- Separate manuscript and Supporting Information PDFs plus a combined review PDF under `paper/pdf/`.
- Main and supplementary figure code, release-safe source tables, and PDF/PNG/SVG assets. This includes the ACS Table-of-Contents graphic under `paper/figures/output/`. Large TIFF submission assets are generated locally and deliberately excluded from Git.
- Frozen version-2 governance files, protocol timeline, split-boundary sensitivity preregistration, manifest-freeze record, and release inventory under `docs/`.
- Split-safe evaluation/training code, tests, selected aggregate artifacts, and outcome-blind split manifests under `scaffoldseal/`.
- Post-confirmatory H1 robustness scripts, the release-safe curation/source audit, D3 coverage stratification, and machine-readable aggregate/source-level outputs.
- Exact release-safe joint and molecule-grouped fold assignments plus the outcome-blind source-only, analogue-only and matched-size manifests.

## Not included

This candidate does not redistribute the upstream row-level CycPeptMPDB table, molecular structures, permeability labels, per-record predictions or residuals, model weights, caches, private vaults, credentials, or machine-specific runtime directories. See `DATA_ACCESS.md`.

## Licence and citation

The authors' original repository content is released under the MIT licence; its scope and third-party exclusions are stated in `LICENSE` and `NOTICE.md`. Citation metadata is provided in `CITATION.cff`. The repository-level Zenodo metadata is in `.zenodo.json`.

## Scientific status

The accepted H1 and H2 results are frozen. The v0.7 revision makes the training-size confound explicit, audits heterogeneous censoring and absent structured assay-condition fields, stratifies D3 coverage without treating seed slots as independent, records uncompleted protocol arms, and narrows the title and causal wording.

The size-profile-matched random, source-only, and analogue-only D0 analyses are post-confirmatory. Their scientific plan and outcome-blind manifests are frozen, but model results are not yet included and cannot revise H1 or H2.

## Reproduce release-safe figures and checks

```bash
python -m venv .venv
python -m pip install -r requirements-reporting.txt
python paper/figures/make_main_figures.py
python paper/figures/make_toc_graphic.py
python scripts/verify_release.py
python scripts/generate_checksums.py --check
```

The H1 robustness and D3 coverage reconstruction scripts require excluded, locally reconstructed OOF artifacts. Set `SCAFFOLDSEAL_RUNTIME_ROOT` to an authorized runtime root before running those scripts. Aggregate figure inputs and released audit outputs do not require private row-level prediction files. The pinned training environment and upstream commit are documented in `docs/BASELINE_ENVIRONMENT.txt`; `requirements-reporting.txt` covers only public reporting figures.

## Public-release sequence

1. The public repository is [schwarzeteeguo-creator/scaffoldseal-cp-evaluation](https://github.com/schwarzeteeguo-creator/scaffoldseal-cp-evaluation).
2. Confirm upstream redistribution terms and retain the current exclusion boundary unless permission is explicit.
3. Create a tagged GitHub release from the uploaded commit.
4. Archive that exact release with Zenodo or an equivalent repository.
5. Add the Zenodo DOI to the manuscript and `CITATION.cff` only after it resolves publicly.

Until the archival release is created, local hashes are described as internal freeze evidence, not as independently time-stamped public preregistration.
