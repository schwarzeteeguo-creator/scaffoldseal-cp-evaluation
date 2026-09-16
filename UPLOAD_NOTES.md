# Upload notes for v0.8.0

## What to upload

Use the **contents** of `github_upload_scaffoldseal_cp_v0.8.0` as the root of:

https://github.com/schwarzeteeguo-creator/scaffoldseal-cp-evaluation

Do not upload the outer folder as a nested directory, and do not replace the browsable repository with a single ZIP.

## Before committing

1. Confirm that `README.md`, `.zenodo.json`, `CITATION.cff`, `LICENSE`, `code/`, `configs/`, `manifests/`, `results/`, `docs/`, `paper/` and `scaffoldseal/` appear at the repository root.
2. Confirm the author order: Yutao Guo; Xujing Jiang; Zihan Zhang; Xuezhou Zhao; Mengxi Chen; Langzhe Zhang; Xiangyu Meng; Guojun Xiong; Dan Wu.
3. Confirm that no third-party row-level structures or permeability labels are present.
4. Run `python scripts/verify_release.py`.
5. Regenerate and verify `SHA256SUMS`.

## Release status

- The immutable GitHub release `v0.8.0` has been archived by Zenodo.
- Version DOI: https://doi.org/10.5281/zenodo.22732455
- Concept DOI: https://doi.org/10.5281/zenodo.22126299
- Do not move or reuse the `v0.8.0` tag.

The DOI metadata and final Journal of Cheminformatics submission files may be committed to the default branch after release, but the archived `v0.8.0` tag must remain unchanged.
