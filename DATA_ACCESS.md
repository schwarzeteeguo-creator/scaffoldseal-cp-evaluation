# Data access and reproducibility boundary

## Reused third-party data

The study uses PAMPA permeability records from CycPeptMPDB v1.2. The source article is:

> Li et al. CycPeptMPDB: A Comprehensive Database of Membrane Permeability of Cyclic Peptides. *Journal of Chemical Information and Modeling* (2023). https://doi.org/10.1021/acs.jcim.2c01573

That article DOI identifies the source publication; it is not asserted here to be a dataset DOI or accession. The source article identifies the official download page as http://cycpeptmpdb.com/download/. Neither the source article nor the database catalogue located a separate reuse or redistribution licence for the downloaded records. Obtain the records from the official page and comply with its terms. This release does not redistribute the row-level records.

The file used for the frozen analysis was locally named `CycPeptMPDB_Peptide_Assay_PAMPA (5).csv`, contained 7,298 rows, and had SHA-256 `02da1cfc18a92b3ae6e70152445b23c05ce6bb0b6ed10fc7c9e141fbd9462fde`. The parenthetical suffix reflects a local browser-download name and is not a database accession. The official download page is recorded above; do not substitute the article DOI as if it were the dataset URL.

## Materials included here

- Frozen selection and evaluation rules.
- Hash-based and aggregate split/governance manifests.
- Aggregate benchmark and uncertainty metrics.
- Figure source-data tables and main-results tables.
- Reporting code and rendered manuscript figures.
- Provenance and environment records needed to interpret the reported results.
- Release-safe row and curated-group manifests containing identifiers, source provenance, curation status and fold membership, but no structures or endpoint values.

These materials support independent checking of the reported aggregate claims without exposing the excluded row-level corpus.

## Deterministic reconstruction

After obtaining the exact upstream table and verifying its SHA-256, an authorized user can regenerate the release-safe curation audit from the repository root:

```bash
python paper/supplementary/analyze_curation_source_audit.py \
  --raw "/path/to/CycPeptMPDB_Peptide_Assay_PAMPA.csv" \
  --output-dir paper/supplementary/source_data/curation_source_audit_v1
```

The resulting summary must report 7,298 raw rows, 372 excluded detection-limit rows, 6,926 uncensored usable rows, 31 compatible collapsed groups, 6,895 curated source-structure records, 6,862 unique molecules and 41 retained sources. See `data/README.md` for the released column dictionary.

## Materials not included

- Upstream molecular structures or raw database exports.
- Row-level permeability labels or development-label tables.
- Per-record predictions, residuals, confidence intervals, or raw descriptors.
- Model weights, training caches, and internal run directories.
- Credentials, private vault paths, and machine-specific configuration.

## Manuscript-ready statement after public archiving

Replace the bracketed fields only after the release exists:

> This study reused CycPeptMPDB v1.2 PAMPA records from http://cycpeptmpdb.com/download/, as described in the source publication (https://doi.org/10.1021/acs.jcim.2c01573). The exact 7,298-row input had SHA-256 02da1cfc18a92b3ae6e70152445b23c05ce6bb0b6ed10fc7c9e141fbd9462fde. The authors do not redistribute upstream structures or permeability labels because a redistribution licence has not been identified. The release-safe archive provides deterministic reconstruction instructions, curation and split manifests, aggregate analysis outputs, figure source data and reporting code at [GITHUB RELEASE URL] and [ZENODO DOI].

If the upstream licence is later confirmed to permit redistribution, document the licence and exact source version before adding any row-level data.
