# Data access and reproducibility boundary

## Reused third-party data

This study uses PAMPA permeability records from CycPeptMPDB v1.2. The source publication is:

> Li et al. CycPeptMPDB: A Comprehensive Database of Membrane Permeability of Cyclic Peptides. *Journal of Chemical Information and Modeling* (2023). https://doi.org/10.1021/acs.jcim.2c01573

The article DOI identifies the source publication; it is not treated as a dataset DOI. The source article identifies the official download page as http://cycpeptmpdb.com/download/.

No separate redistribution licence for the downloaded row-level records was identified in the available source documentation. Users must obtain the records through the official route and comply with the applicable terms. This release therefore does not redistribute the upstream structures or permeability labels.

The exact local input used for the frozen analysis:

- Local filename: `CycPeptMPDB_Peptide_Assay_PAMPA (5).csv`
- Rows: 7,298
- SHA-256: `02da1cfc18a92b3ae6e70152445b23c05ce6bb0b6ed10fc7c9e141fbd9462fde`

The parenthetical filename suffix reflects a local browser-download name and is not a database accession.

## Materials included

- Frozen selection, exclusion and evaluation rules.
- Release-safe curation and group manifests without structures or endpoint values.
- Hash-based and aggregate split/governance manifests.
- Aggregate benchmark, robustness, calibration and uncertainty metrics.
- Figure source-data tables and main-results tables.
- Reporting, curation, splitting and evaluation code.
- Journal of Cheminformatics manuscript and Supporting Information materials.
- Provenance, environment and checksum records.

## Deterministic curation reconstruction

After obtaining the exact upstream table and verifying its SHA-256, run from the repository root:

```bash
python paper/supplementary/analyze_curation_source_audit.py --raw "/path/to/CycPeptMPDB_Peptide_Assay_PAMPA.csv" --output-dir paper/supplementary/source_data/curation_source_audit_v1
```

The expected summary reports:

- 7,298 raw rows.
- 372 detection-limit rows excluded from continuous regression.
- 6,926 uncensored usable rows.
- 31 compatible collapsed groups.
- 6,895 curated source-structure records.
- 6,862 unique molecules.
- 41 retained sources.

See `data/README.md` for the released column dictionary.

## Materials not included

- Upstream molecular structures or raw database exports.
- Row-level permeability labels or development-label tables.
- Per-record predictions, residuals, intervals or raw descriptors.
- Model weights, training caches and internal run directories.
- Credentials, private vault paths and machine-specific configuration.

## Archive identifiers

- Repository: https://github.com/schwarzeteeguo-creator/scaffoldseal-cp-evaluation
- Stable all-version Concept DOI: https://doi.org/10.5281/zenodo.22126299
- Previous immutable v0.7.7 DOI: https://doi.org/10.5281/zenodo.22126300
- Exact v0.8.0 DOI: https://doi.org/10.5281/zenodo.22732455

## Submission-ready availability wording

> This study reused CycPeptMPDB v1.2 PAMPA records from the database download page (http://cycpeptmpdb.com/download/), as described in the source publication (https://doi.org/10.1021/acs.jcim.2c01573). The exact 7,298-row input had SHA-256 02da1cfc18a92b3ae6e70152445b23c05ce6bb0b6ed10fc7c9e141fbd9462fde. The authors do not redistribute upstream structures or permeability labels because a separate redistribution licence was not identified. The release-safe archive provides deterministic reconstruction instructions, curation and split manifests, aggregate analysis outputs, figure source data, reporting code, provenance records and file-level checksums at https://github.com/schwarzeteeguo-creator/scaffoldseal-cp-evaluation and https://doi.org/10.5281/zenodo.22732455.

If the upstream licence is later confirmed to permit redistribution, document the exact licence and source version before adding any row-level material.
