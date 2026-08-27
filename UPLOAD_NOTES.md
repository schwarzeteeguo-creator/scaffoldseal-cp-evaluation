# Upload notes

- Use this folder, `github_upload_scaffoldseal_cp_v0.7`, as the GitHub repository root.
- Do not upload the larger working candidate folder with TIFFs and historical draft copies.
- Local v0.5/v0.6 copies in this folder are explicitly ignored by `.gitignore` and excluded from `SHA256SUMS`; do not force-add them.
- The expected public payload excludes all `*.tif`, `*.tiff`, caches, model weights, raw labels and row-level predictions.
- `CITATION.cff`, `LICENSE`, `NOTICE.md` and `.zenodo.json` are ready for publication. The GitHub URL and Zenodo DOI must be added only after their public records exist.
- Submit `paper/figures/output/toc_graphic_evaluation_boundary.tiff` separately as the ACS Table-of-Contents graphic; the same graphic is also appended to the manuscript PDF as a page labeled “For Table of Contents Only”.
- The post-confirmatory matched-size/source-only/analogue-only manifests are included; their model-performance results are not.
- The chirality-aware Murcko arm remains uncompleted and is not represented as a reported result.
