# Released data dictionary and reconstruction boundary

The upstream PAMPA table is not redistributed while its reuse terms remain unresolved. This directory documents how the release-safe manifests relate to that table.

## Exact upstream input

- Local analysis filename: `CycPeptMPDB_Peptide_Assay_PAMPA (5).csv`
- Rows: 7,298
- SHA-256: `02da1cfc18a92b3ae6e70152445b23c05ce6bb0b6ed10fc7c9e141fbd9462fde`
- Source publication: https://doi.org/10.1021/acs.jcim.2c01573
- Official acquisition URL: http://cycpeptmpdb.com/download/

The source publication identifies this page as the database download route. A separate reuse or redistribution licence for the row-level records was not identified, so the records are not redistributed in this release.

## Release-safe curation manifest

`paper/supplementary/source_data/curation_source_audit_v1/curation_manifest_release_safe.csv` contains one row per upstream raw identifier.

| Field | Meaning |
|---|---|
| `raw_id` | Stable identifier assigned during deterministic curation |
| `raw_group_id` | Source-structure grouping identifier; no structure is exposed |
| `source`, `year`, `version` | Upstream provenance metadata |
| `is_censored` | Whether a detection-limit field triggered the frozen exclusion rule |
| `curation_status` | Included, excluded, or excluded but linked to a retained group |
| `exclusion_or_link_reason` | Rule accounting for the row |
| `curated_id` | Final analytical-record identifier when applicable |

## Release-safe curated-group manifest

`curated_group_manifest_release_safe.csv` contains one row per analytical source-structure group. `molecule_id` is a hash-derived grouping identifier, and the topology/count fields are aggregate non-structural descriptors. `raw_ids_all` and `raw_ids_used` support accounting for the 31 compatible multi-row groups. The file contains no molecular structures and no permeability endpoint values.

## Split manifests

Joint and molecule-grouped outer assignments are under `scaffoldseal/artifacts/v2_r0/` and `scaffoldseal/artifacts/h1_random_cv_r0/`. The outcome-blind source-only, analogue-only and matched-size fold columns are in `scaffoldseal/artifacts/split_boundary_sensitivity_v1/evaluation_boundary_ladder_manifest.csv`. Their presence does not imply that the pending model fits were completed.
