# Figure QA record

Date: 2026-08-21

## Contract and logic

- PASS: each figure has a one-sentence conclusion, declared archetype, panel map, evidence hierarchy and reviewer-risk check in `FIGURE_CONTRACTS_V0.4.md`.
- PASS: Fig. 1 is an author-supplied, reviewed vector overview that integrates outcome-blind curation, source/analogue joint blocks, nested evaluation, the D0 boundary comparison and the fixed-D3 summary.
- PASS: Fig. 2 shows every frozen joint block and quantifies both concentration and source/analogue connectivity.
- PASS: Fig. 3 retains H1 as the hero evidence; fixed-candidate performance and calibration remain subordinate panels.
- PASS: Fig. 4 shows all five seeds, all 41 sources and all 18 blocks, avoiding selective display of only adverse regimes.
- PASS: D3 is explicitly marked as the fixed candidate; D1/D2 are not presented as post hoc replacements.

## Backend and exports

- Backend: Python with matplotlib for Figs. 2-4; Fig. 1 is an author-supplied vector PDF, retained as the authoritative source and rendered to PNG only for manuscript preview.
- Final width: 183 mm double-column design.
- Exports for Figs. 2-4: editable SVG, PDF, 600 dpi TIFF and 300 dpi PNG preview. Fig. 1: author-supplied vector PDF plus a high-resolution PNG preview; the previous generated SVG is archived and is not the current Figure 1 source.
- Text: Figs. 2-4 use editable SVG/PDF text; Fig. 1 remains editable in the supplied vector-PDF authoring workflow.
- Palette: restrained blue/orange/neutral/green/red family; meaning is also carried by position, labels and sign.

## Statistics and source data

- Fig. 1: 7,298 raw rows, 372 censored exclusions, 6,926 uncensored usable rows, 31 collapsed compatible groups, 6,895 curated records, 6,862 unique molecules, 41 sources, 141,425 analogue edges, 305 analogue components and 18 joint blocks. It also displays D0 source-macro MAEs of 0.4672 and 1.0046, their 0.5373 gap (95% interval 0.3244-0.7530), and fixed-D3 summary values already reported elsewhere in the manuscript.
- Fig. 2: all 18 frozen joint blocks; record, source and analogue-component counts only.
- Fig. 3a: source-macro MAE; five-seed evaluation; 10,000 paired outer-block/seed bootstrap replicates for the gap confidence interval.
- Fig. 3b: mean-across-five-seed source-macro MAE; lower is better.
- Fig. 3c: 34,475 frozen out-of-fold prediction slots; fold-local empirical intervals; nominal and observed coverage shown directly.
- Fig. 4a: all five scheduled seeds. Fig. 4b: all 41 sources. Fig. 4c: all 18 outer blocks. Panels b and c use descriptive five-seed means.
- Clean aggregate CSV source files are stored under `paper/figures/source_data/`.
- Plotting script: `paper/figures/make_main_figures.py` regenerates Figs. 2-4 only and deliberately protects the supplied Fig. 1 from overwrite.

## Visual inspection

- PASS: Fig. 1 was rendered from the supplied vector PDF and inspected at manuscript placement; all four numbered stages, the D0 comparison, D3 summary and caveat are visible without clipping.
- PASS: Fig. 2 uses a log axis explicitly, labels the three largest blocks without overlap, and states the point-area mapping.
- PASS: Fig. 3a uses a single-line gap/CI annotation and short manual bracket caps; Fig. 3b uses a dot plot rather than truncated bars; Fig. 3c uses matched axes and marks the prespecified 90% acceptance interval.
- PASS: Fig. 4 source labels are contained in a compact inset, and fold callouts do not obscure the plotted points.
- PASS: axes and units are defined; legends are not repeated unnecessarily.
- PASS: PNG previews inspected at original resolution before manuscript integration.

## Integrity

- Vector-native charts and schematic; Fig. 1 is a supplied vector graphic rather than a generated raster illustration. No photographic or microscopy manipulation was used.
- No simulated or placeholder performance values.
- All plotted values trace to frozen manifests, released aggregate metric payloads, the frozen reporting table or the interval-coverage artifact.
- No row-level prediction data are included in the figure source-data package.
- No training, tuning, wet-lab work, paid compute or external data transfer was performed for figure creation.
