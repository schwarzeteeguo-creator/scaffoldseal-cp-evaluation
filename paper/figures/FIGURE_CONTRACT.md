# Figure contract

## Main results figure

- **Core conclusion:** DMPNN performance estimates diverge between molecule-grouped pseudorandom and joint source/analogue-block procedures; the completed contrast does not isolate training-size reduction from dependence removal, and the fixed D3 modification neither improves joint-block performance nor provides adequately calibrated empirical intervals.
- **Archetype:** quantitative grid with a dominant H1 panel.
- **Target/output:** JCIM-oriented double-column figure; 183 mm wide; editable SVG/PDF plus 600 dpi TIFF and PNG preview.
- **Backend:** Python (matplotlib) exclusively.
- **Panel map:**
  - **a:** H1 source-macro MAE under molecule-random and joint source/analogue-block evaluation, with the prespecified gap and bootstrap CI.
  - **b:** joint-block source-macro MAE for D0, nested-selected classical, D1, D2, and fixed candidate D3.
  - **c:** nominal versus observed D3 empirical interval coverage at 50%, 80%, and 90%, including the 90% acceptance band.
- **Evidence hierarchy:** panel a is the hero evidence; panel b establishes that H2 did not close the gap; panel c establishes under-coverage.
- **Statistics:** five-seed source-macro means; H1 10,000-replicate block/seed bootstrap CI; interval coverage across 34,475 frozen OOF prediction slots.
- **Source data:** frozen main-results CSV and coverage summary JSON.
- **Image integrity:** vector-native chart; no raster image manipulation.
- **Reviewer risks:** avoid implying universality or a separated causal effect; distinguish D3 as the internally prespecified candidate; state that the H1 CI is for the gap rather than individual bars; do not display D1 as a post hoc replacement; identify coverage slots as repeated seed realizations rather than independent samples.

## Dependence-aware evaluation overview

- **Core conclusion:** outcome-blind curation and joint source/analogue blocking yield a more demanding evaluation boundary, under which the completed D0 estimate diverges from molecule-grouped pseudorandom evaluation; the contrast is a protocol divergence, not an isolated causal effect of dependence removal.
- **Archetype:** schematic-led composite with a compact quantitative readout.
- **Panel map:** 1, curation from 7,298 raw rows to 6,895 records; 2, analogue/source graph to 18 joint blocks; 3, nested leave-one-block-out evaluation; 4, D0 boundary comparison and fixed-D3 summary.
- **Target/output:** JCIM-oriented double-column vector PDF supplied by the authors, with a rendered PNG preview. The prior editable SVG workflow is archived rather than presented as the current Figure 1 source.
- **Evidence hierarchy:** panel 4 is the quantitative readout; panels 1-3 justify why its two procedures target different boundaries.
- **Reviewer risks:** do not imply that every molecule-grouped pair is a duplicate, that joint blocking creates a balanced population, or that the displayed D0 gap isolates dependence removal from training-size differences. D3 remains a fixed candidate rather than a post hoc replacement.
