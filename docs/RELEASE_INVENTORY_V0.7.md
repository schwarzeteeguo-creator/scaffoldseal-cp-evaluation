# ScaffoldSeal-CP release inventory v0.7

This inventory maps manuscript and Supporting Information claims to the cleaned GitHub upload candidate. “Included” means present locally; it does not mean that a public DOI has been minted.

| Release path/category | Required content | v0.7 status | SI correspondence |
|---|---|---|---|
| `paper/` | v0.7 manuscript, SI, references and three review PDFs | Included after PDF build | Entire article and SI |
| `docs/` | version-2 protocol, recovery map, CV governance, experiment plan, split-boundary freeze and public timeline | Included; internal freeze is distinguished from public preregistration | Tables S7 and S10 |
| `paper/supplementary/source_data/curation_source_audit_v1/` | 7,298-row curation accounting, 31 collapsed groups, censoring/source summaries and hashes | Included as release-safe files; structures and labels excluded | Tables S1 and S9 |
| `scaffoldseal/artifacts/v2_r0/` | Exact joint outer assignments, outer manifest and inner baskets | Included | Joint boundary methods |
| `scaffoldseal/artifacts/h1_random_cv_r0/` | Exact molecule-grouped outer assignments and inner contracts | Included | H1 comparison |
| `scaffoldseal/artifacts/split_boundary_sensitivity_v1/` | Source-only, analogue-only and matched-size outcome-blind manifests | Included; corresponding model results pending | Table S10 |
| `paper/supplementary/source_data/` | H1 robustness, source calibration, threshold geometry, prior work and D3 coverage stratification | Included | Tables S2-S6 and S11; Figs. S1-S3 |
| `paper/figures/` | Deterministic figure code, source tables and PDF/PNG/SVG assets | Included; TIFF generated locally and excluded from Git | Figs. 1-4 |
| `scaffoldseal/src/`, `tests/`, configs | Split-safe runners, governance checks and fixed configuration | Included | Model and integrity methods |
| `docs/BASELINE_ENVIRONMENT.txt` | Upstream repository, exact commit and training environment | Included | Reproducibility methods |
| `data/README.md`, `DATA_ACCESS.md` | Exact raw hash, official acquisition route, redistribution boundary and reconstruction instructions | Included; upstream row-level reuse licence not identified, so raw records remain excluded | Data Availability and Table S9 |
| `SHA256SUMS` | File-level hashes for the cleaned candidate | Regenerated after final PDF build | Release integrity |
| `CITATION.cff`, `LICENSE`, `NOTICE.md` | Author list, citation metadata and reuse terms | Included; the archive DOI is added only after Zenodo publication | Archive metadata |

## Deliberately excluded

- The upstream row-level CycPeptMPDB export, molecular structures and permeability labels.
- Per-record predictions, residuals, reconstructed interval rows and model weights.
- Private vaults, credentials, caches, machine-specific paths and failed-run working directories.
- TIFF submission assets and historical manuscript/PDF drafts. Local legacy copies are explicitly ignored by `.gitignore` and excluded from `SHA256SUMS` until they can be removed from the working candidate.

## Scientific work still pending

1. Size-profile-matched molecule-random, source-only and analogue-only D0 model fits. Their manifests are frozen, but results are post-confirmatory and absent from v0.7.
2. The originally planned chirality-aware Murcko D0 arm, which remains uncompleted.
3. A prospective matched-protocol PAMPA evaluation.

## Public-release blockers

1. GitHub release URL and archival DOI.
2. Completion of all authors' confirmation of names, affiliations, CRediT, funding and conflicts.
3. Upstream redistribution licence remains unidentified; raw records must remain excluded unless explicit permission is obtained.
