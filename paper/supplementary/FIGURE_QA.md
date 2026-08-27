# Supplementary figure QA

## Supplementary Figure S1

- Contract: quantitative three-panel grid; double-column width (183 mm); Python/matplotlib backend.
- Data coverage: 6,895 records, 41 sources, 18 joint blocks, two frozen D0 evaluation arms and five seeds.
- Integrity: both prediction directories passed their frozen artifact-manifest size and SHA-256 checks before analysis; the calculated block counts were cross-checked against the frozen aggregate block manifest.
- Numerical checks: the source-macro headline values reproduce the confirmatory H1 result to an absolute tolerance of 1e-12; all 18 single-block omissions are present; all three aggregation estimands are present.
- Visual checks: all panels, labels and reference lines are legible at the exported double-column size; the sole negative within-block gap and the three largest blocks are labelled; the panel-c legend is placed outside the data region.
- Exports: editable SVG, PDF, 600 dpi TIFF and 300 dpi PNG.
- Interpretation boundary: explicitly labelled as post-confirmatory descriptive analysis; no replacement confidence interval or new confirmatory claim is made.

## Supplementary Figure S2

- Contract: quantitative three-panel grid; double-column width (183 mm); Python/matplotlib backend.
- Data coverage: all 6,895 records, 41 sources, 18 joint blocks, two frozen D0 evaluation arms and five seeds. Calibration uses one five-seed mean prediction per record and ten arm-specific prediction deciles.
- Integrity: the same manifest-verified frozen inputs used for Supplementary Figure S1 were reloaded; no fitted or recalibrated values were introduced.
- Numerical checks: source table contains exactly 41 rows, block table exactly 18 rows and calibration source data exactly ten bins per arm. Positive source gaps total 38/41; 28/41 are positive in all five seeds.
- Visual checks: all 41 sources are plotted; the five largest positive source gaps, all negative gaps and the five largest absolute joint-block biases are labelled without overlap; signed-bias axes use identical limits; calibration axes share an identity line and equal scale.
- Exports: editable SVG, PDF, 600 dpi TIFF and 300 dpi PNG.
- Interpretation boundary: post-confirmatory descriptive diagnosis only. Small-source extremes, arm-specific binning and the prohibition on post-hoc recalibration are stated in the legend/report.

## Supplementary Figure S3

- Contract: quantitative three-panel grid; double-column width (183 mm); Python/matplotlib backend.
- Data coverage: all 6,895 public curated records, 6,862 unique molecules and 41 sources; deterministic graph reconstruction at the prespecified descriptive thresholds 0.70, 0.80 and 0.90.
- Integrity: the graph input contains no permeability/outcome field. Chiral ECFP4 radius/length, molecular-weight ratio, exact one-token edit rule and source closure were held fixed. The primary 0.80 reconstruction exactly matches all frozen component and block identifiers and the 141,425-edge/305-component/18-block headline counts.
- Numerical checks: threshold summary has three rows; ranked source data show all 15, 18 and 20 joint blocks; record assignments contain exactly 6,895 rows per threshold. Partition-comparison metrics are calculated directly from record/molecule contingency tables.
- Visual checks: logarithmic scaling is explicit; all block-size profiles are shown; primary 0.80 is highlighted but alternatives remain visible; labels and legends do not cover the data.
- Exports: editable SVG, PDF, 600 dpi TIFF and 300 dpi PNG.
- Interpretation boundary: geometry only. The legend/report explicitly state that frozen OOF predictions cannot be rescored as valid alternative-threshold evaluations and that valid performance comparisons would require new nested retraining.
