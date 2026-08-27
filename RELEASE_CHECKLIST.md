# Public release checklist

Do not make this repository public until the blocking items below are resolved.

## Blocking author decisions

- [x] Public repository created: `schwarzeteeguo-creator/scaffoldseal-cp-evaluation`.
- [x] Author names and affiliations have been entered in citation and Zenodo metadata. Add ORCID identifiers later if available.
- [ ] Confirm that every included source file can be publicly released.
- [ ] Confirm the upstream CycPeptMPDB redistribution terms; keep row-level records excluded unless permission is explicit.
- [x] The authors' original repository content is licensed under MIT; third-party data remain excluded (see `NOTICE.md`).
- [x] `CITATION.cff` and `.zenodo.json` contain release-ready metadata without unresolvable placeholder fields.
- [ ] Replace manuscript repository/DOI placeholders only after the links resolve.

## Technical checks

- [ ] Run `python scripts/verify_release.py` and resolve all failures.
- [ ] Run `python scripts/generate_checksums.py` after the final edit.
- [ ] Run `python scripts/generate_checksums.py --check` and confirm all hashes pass.
- [ ] Reproduce the figures from a clean environment using `requirements-reporting.txt`.
- [ ] Review `git status` and confirm that no raw data, weights, caches, secrets, or TIFF files are staged.
- [ ] Confirm that `paper/manuscript_v0.7.md` and the matching SI/PDF files are the intended public manuscript version.
- [ ] Confirm a stable official acquisition URL or author-supplied access route for the exact upstream PAMPA table and verify its SHA-256.
- [ ] Confirm that the release-safe curation manifests contain no structures or endpoint values.

## Publication sequence

- [x] Create the GitHub repository and commit this folder's contents as the repository root.
- [ ] Connect the repository to Zenodo before creating the archival release.
- [ ] Create an initial tagged GitHub release, for example `v0.1.0`.
- [ ] Confirm that Zenodo archived the release and issued a DOI.
- [ ] Add the final release URL and DOI to the manuscript, README, and citation metadata.
- [ ] Regenerate checksums if any archived content changes, then issue a new release rather than rewriting an existing archival record.
