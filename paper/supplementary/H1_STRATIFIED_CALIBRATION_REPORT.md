# H1 source-stratified error and point-calibration report

## Scope

This is a post-confirmatory, zero-training diagnosis of the frozen H1 OOF predictions. It did not fit, tune, select or recalibrate any model. Source and block summaries are descriptive five-seed means; calibration curves use one five-seed mean OOF prediction per frozen record.

## Breadth of the random-split optimism gap

The joint-block minus molecule-random MAE gap was positive for 38/41 sources on average. It was positive in all five seeds for 28/41 sources and in at least four seeds for 36/41 sources. The result is therefore widespread rather than confined to one or two sources, although the magnitude is highly heterogeneous.

The five largest source-level gaps were:

- 2022_Lee: n=21, gap 2.503, joint bias 2.745.
- 2018_Lee: n=6, gap 2.408, joint bias -2.676.
- 2019_Ono: n=8, gap 1.521, joint bias -1.758.
- 2018_García-Pindado: n=4, gap 1.518, joint bias -1.837.
- 2018_Buckton: n=2, gap 1.198, joint bias -2.430.

Several extremes have small source sample sizes, so their magnitudes should not be interpreted as stable population estimates. They are localization diagnostics, not independent confirmatory effects.

## Source and block signed bias

The median absolute source-level signed bias increased from 0.078 under molecule-random evaluation to 0.578 under joint-block evaluation. Absolute bias exceeded 0.5 log10 Papp units for 23/41 joint-block source summaries versus 3/41 molecule-random summaries.

The three blocks with the largest absolute joint-block row-weighted bias were:

- Fold 14: n=8, joint bias -2.615, random bias -0.461.
- Fold 8: n=4, joint bias -1.837, random bias 0.045.
- Fold 11: n=17, joint bias -0.958, random bias 0.049.

Positive bias denotes permeability predictions that are too high (less negative log10 Papp); negative bias denotes predictions that are too low.

## Point-prediction calibration

After averaging the five OOF predictions for each record, the observed-on-predicted calibration slope was 0.964 for molecule-random evaluation and 0.595 for joint-block evaluation. The decile-weighted absolute calibration error was 0.027 and 0.181, respectively. Global signed bias was -0.001 under random splitting and -0.175 under joint-block shift.

The joint-block slope below one and the compressed prediction-decile curve indicate regression toward the training-domain mean under source/analogue shift. This is a point-prediction calibration diagnosis and is separate from the D3 empirical interval-coverage result.

## Interpretation boundary

- These analyses explain the released H1 contrast; they do not constitute external validation.
- Prediction bins are arm-specific and descriptive, and no post-hoc recalibration was fitted.
- Sources are highly unequal in size; all 41 are retained, but small-source extremes are labelled as such in the source data.
- The original H1 block/seed bootstrap remains the inferential analysis.
