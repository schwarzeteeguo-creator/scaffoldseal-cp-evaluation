# Release inventory v0.8.0

## Metadata and governance

| Material | Location | Public status |
|---|---|---|
| Repository overview | `README.md` | Included |
| Citation metadata | `CITATION.cff` | Included; v0.8.0 DOI recorded |
| Zenodo metadata | `.zenodo.json` | Included |
| Licence and third-party notice | `LICENSE`, `NOTICE.md` | Included |
| Data-access boundary | `DATA_ACCESS.md` | Included |
| Protocol timeline and freeze records | `docs/` | Included |
| File checksums | `SHA256SUMS` | Regenerated for final package |

## Code and environments

| Material | Location | Public status |
|---|---|---|
| Canonical analysis/evaluation code | `scaffoldseal/src/` | Included |
| Tests | `scaffoldseal/tests/` | Included |
| Curation and sensitivity scripts | `paper/supplementary/` | Included |
| Figure-generation scripts | `paper/figures/` | Included |
| Release-safe environment | `environment.yml`, `requirements.txt` | Included |
| Pinned baseline environment | `docs/BASELINE_ENVIRONMENT.txt` | Included |

## Manifests and results

| Material | Location | Public status |
|---|---|---|
| Release-safe curation manifests | `paper/supplementary/source_data/curation_source_audit_v1/` | Included |
| Joint-block manifests | `scaffoldseal/artifacts/v2_r0/` | Included |
| Molecule-grouped manifests | `scaffoldseal/artifacts/h1_random_cv_r0/` | Included |
| Boundary-sensitivity manifests | `scaffoldseal/artifacts/split_boundary_sensitivity_v1/` | Included |
| Aggregate evaluation summaries | `scaffoldseal/artifacts/` | Included |
| Figure source tables | `paper/figures/source_data/` | Included |
| SI robustness/source tables | `paper/supplementary/source_data/` | Included |

## Manuscript materials

| Material | Location | Public status |
|---|---|---|
| Journal of Cheminformatics main manuscript | `paper/manuscript/` | Included |
| Supporting Information | `paper/supporting_information/` | Included |
| Combined internal review PDF | `paper/complete_review/` | Included |
| Editable LaTeX source archive | `paper/ScaffoldSeal_CP_JCheminform_LaTeX_source_v1.1_DOI.zip` | Included |

## Deliberately excluded

- Third-party row-level CycPeptMPDB exports.
- Molecular structures and permeability labels.
- Per-record predictions, residuals and interval rows.
- Model weights and training caches.
- Credentials, private paths and machine-specific runtime directories.
- Live D4 training logs and incomplete D4 outputs.

The exact v0.8.0 archive is available at https://doi.org/10.5281/zenodo.22732455.
