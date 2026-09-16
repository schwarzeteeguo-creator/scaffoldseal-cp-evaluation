# Public release checklist for v0.8.1

## Metadata

- [x] Repository URL recorded.
- [x] Release version set to 0.8.1.
- [x] Xujing Jiang added as the second author.
- [x] Guojun Xiong added as the penultimate author.
- [x] Dan Wu retained as final and corresponding author.
- [x] Author order aligned between `CITATION.cff` and `.zenodo.json`.
- [x] MIT licence applied only to the authors' original materials.
- [ ] Add ORCID identifiers if the authors choose to provide them.

## Public-data boundary

- [x] Upstream CycPeptMPDB structures and permeability labels excluded.
- [x] Per-record predictions and residuals excluded.
- [x] Model weights, caches, logs, credentials and private runtime paths excluded.
- [x] Release-safe manifest columns checked for structures and endpoint values.
- [ ] Author confirms that every included original source file may be publicly released.
- [ ] Author confirms the current upstream access route immediately before submission.

## Technical checks

- [x] Run `python scripts/verify_release.py`.
- [x] Run `python scripts/generate_checksums.py`.
- [x] Run `python scripts/generate_checksums.py --check`.
- [x] Confirm `.zenodo.json` parses as valid JSON.
- [x] Confirm `CITATION.cff` parses as valid YAML/CFF.
- [x] Confirm the manuscript and SI PDFs open correctly.
- [x] Confirm repository files are directly browsable after push.

## Publication sequence

- [x] GitHub repository exists.
- [x] Zenodo integration was previously enabled.
- [x] Commit and push the release package to `main`.
- [x] Create the immutable GitHub release `v0.8.1`.
- [x] Confirm Zenodo successfully archives v0.8.1.
- [x] Copy the new version-specific DOI: `10.5281/zenodo.22787442`.
- [x] Replace the v0.8.0 DOI in the final manuscript, SI references and bibliography.
- [x] Regenerate the final submission PDFs and source ZIP.
- [x] Verify that the new DOI resolves publicly.

Do not overwrite or delete the historical v0.7.7 or v0.8.0 Zenodo records.
