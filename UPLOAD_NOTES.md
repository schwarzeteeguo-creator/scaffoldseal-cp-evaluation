# Upload notes for v0.8.0

## What to upload

Use the **contents** of `github_upload_scaffoldseal_cp_v0.8.0` as the root of:

https://github.com/schwarzeteeguo-creator/scaffoldseal-cp-evaluation

Do not upload the outer folder as a nested directory, and do not replace the browsable repository with a single ZIP.

## Before committing

1. Confirm that `README.md`, `.zenodo.json`, `CITATION.cff`, `LICENSE`, `code/`, `configs/`, `manifests/`, `results/`, `docs/`, `paper/` and `scaffoldseal/` appear at the repository root.
2. Confirm the author order: Yutao Guo; Xujing Jiang; Zihan Zhang; Xuezhou Zhao; Mengxi Chen; Langzhe Zhang; Xiangyu Meng; Dan Wu.
3. Confirm that no third-party row-level structures or permeability labels are present.
4. Run `python scripts/verify_release.py`.
5. Regenerate and verify `SHA256SUMS`.

## Release sequence

1. Commit and push the v0.8.0 contents to `main`.
2. Confirm the Zenodo integration toggle is ON.
3. Create a new immutable GitHub tag and release named `v0.8.0`.
4. Do not mark it as a prerelease.
5. Wait for Zenodo to archive it and issue a new version-specific DOI.
6. Insert that DOI into the submitted manuscript and its bibliography.
7. Do not move or reuse the `v0.8.0` tag.

The manuscript source currently records the previous public archive because the new DOI does not exist yet. The final submitted manuscript must be regenerated after the v0.8.0 DOI is minted.
