# Supplementary figure contracts

## Supplementary Figure S1 — H1 influence and estimand sensitivity

Core conclusion: The large H1 gap between joint-block and molecule-random evaluation is not removed by omitting any single frozen joint block, although its magnitude varies with block identity and aggregation estimand.

Figure archetype: quantitative three-panel grid.

Target journal/output: double-column supplementary figure; editable SVG/PDF plus 600 dpi TIFF and 300 dpi PNG.

Backend: Python (matplotlib) only.

Final size: 183 mm wide, approximately 86 mm high.

Panel map:

- a: Leave-one-block-out source-macro H1 gap for all 18 possible single-block omissions, ordered by the resulting gap. The full-population gap and the preregistered 0.10 support threshold are shown as references.
- b: Within-block source-macro H1 gap versus frozen block record count for all 18 blocks, with the three largest blocks labelled.
- c: Random-split and joint-block MAE under row-micro, source-macro, and equal-block-macro aggregation; the connecting segment represents the H1 gap under each estimand.

Evidence hierarchy: panel a is the decisive influence analysis; panel b localizes heterogeneity; panel c tests whether the qualitative contrast depends on weighting records, sources, or blocks.

Statistics needed: all 6,895 frozen records, all 41 sources, all 18 frozen blocks, both D0 evaluation arms, and all five scheduled seeds. No refitting, new randomization, hypothesis test, or model selection is performed.

Source data needed: verified frozen OOF predictions, frozen labels and assignments, and the frozen aggregate block manifest.

Image-integrity notes: every block is shown; no jitter, simulated observations, or omitted outliers; point area in panel b is fixed rather than data-encoded.

Reviewer risk: these diagnostics were specified after the confirmatory H1 result was known. They are descriptive robustness analyses, not a new confirmatory test. Omitting blocks changes the represented target population, and the three aggregation schemes estimate different quantities.

## Supplementary Figure S2 — Source-stratified error and point-prediction calibration

Core conclusion: Random-split optimism is distributed across most sources, while joint source/analogue shift produces substantially broader source-level signed bias and poorer point-prediction calibration.

Figure archetype: quantitative three-panel grid.

Target journal/output: double-column supplementary figure; editable SVG/PDF plus 600 dpi TIFF and 300 dpi PNG.

Backend: Python (matplotlib) only.

Final size: 183 mm wide, approximately 82 mm high.

Panel map:

- a: Ranked source-level MAE gap for all 41 sources, with positive and negative gaps encoded by both colour and marker shape. Label the five largest positive gaps and all negative gaps.
- b: Source-level mean signed error under molecule-random versus joint-block evaluation, with the identity line and zero-bias axes. Label the five sources with the largest absolute joint-block bias.
- c: Arm-specific prediction-decile calibration curves after averaging the five frozen OOF predictions for each record. Show the identity line and descriptive calibration slopes.

Evidence hierarchy: panel a establishes breadth across sources; panel b localizes systematic over- and underprediction; panel c provides the population-level calibration diagnosis.

Statistics needed: all 6,895 frozen records, all 41 sources, both D0 evaluation arms and five scheduled seeds. Source summaries are means across seeds. Calibration curves use one five-seed mean prediction per record and ten equal-frequency, arm-specific prediction bins. No hypothesis test or recalibration is performed.

Source data needed: verified frozen OOF predictions and frozen record/source/block assignments.

Image-integrity notes: all 41 sources and all ten bins per arm are shown; no jitter, smoothing or simulated observations; source markers have fixed area.

Reviewer risk: source sizes are highly unequal, several extreme source estimates are based on few records, binning is descriptive, and calibration is evaluated on already released OOF predictions. These results must not be used to tune or recalibrate the models.

## Supplementary Figure S3 — Analogue-threshold sensitivity of evidence geometry

Core conclusion: The number of molecular analogue components is threshold-sensitive, but source/component closure leaves the joint-block evidence geometry highly concentrated across the prespecified 0.70, 0.80 and 0.90 similarity thresholds.

Figure archetype: quantitative three-panel grid.

Target journal/output: double-column supplementary figure; editable SVG/PDF plus 600 dpi TIFF and 300 dpi PNG.

Backend: Python (matplotlib) only.

Final size: 183 mm wide, approximately 76 mm high.

Panel map:

- a: Number of analogue components and source/component joint blocks versus threshold on a logarithmic axis, with the constant 41 source groups as a reference.
- b: Largest-block and three-largest-block record shares versus threshold.
- c: Ranked joint-block record counts for each threshold on a logarithmic axis, with the frozen primary 0.80 definition emphasized.

Evidence hierarchy: panel a separates molecular sensitivity from provenance closure; panel b provides the key concentration result; panel c shows that the full size distribution, not only one summary, remains similarly skewed.

Statistics needed: deterministic outcome-blind graph reconstruction at thresholds 0.70, 0.80 and 0.90 while holding chiral ECFP4 radius/bit length, molecular-weight ratio, exact one-token cyclic-edit rule, source provenance and all curated records fixed. Descriptive counts only.

Source data needed: frozen public curated structures/source fields, frozen configuration and the original graph-building code.

Image-integrity notes: every joint block is shown; no model performance, outcome value or simulated observation is plotted; primary 0.80 is visually emphasized without suppressing alternatives.

Reviewer risk: the alternative definitions were reconstructed after confirmatory outcomes existed. The analysis describes partition geometry only. Existing OOF predictions are not rescored under alternative partitions because their training sets do not correspond to those partitions; valid performance comparison would require complete newly preregistered nested retraining.
