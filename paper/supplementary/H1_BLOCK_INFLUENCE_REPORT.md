# H1 block-influence robustness report

## Scope

This is a descriptive, zero-training analysis of the already frozen H1 out-of-fold predictions. It was specified after the confirmatory H1 result was known and is therefore not treated as a new confirmatory test. No model was fitted, selected, calibrated, or tuned.

## Main result

The full source-macro gap was reproduced exactly: joint-block MAE 1.004559, molecule-random MAE 0.467218, and gap 0.537341 log10 Papp units.

All 18 leave-one-block-out recalculations remained above the preregistered 0.10 point-gap threshold. The smallest gap was 0.472 after omitting fold 14 (8 records; 2 source(s)), and the largest was 0.552 after omitting fold 17 (7 records; 1 source(s)). Thus, no single frozen joint block alone explains the qualitative H1 contrast.

Simultaneously removing the three largest blocks left only 525 records, 19 sources, and 15 blocks. In that heavily altered target population, the source-macro gap was 0.533. This stress test is reported only as a sensitivity description because the represented population changes substantially.

## Aggregation estimands

- row_micro: joint 0.643, random 0.293, gap 0.350.
- source_macro: joint 1.005, random 0.467, gap 0.537.
- block_macro: joint 0.905, random 0.439, gap 0.466.

The H1 direction is unchanged whether records, sources, or blocks receive equal weight. The magnitude changes because these estimands answer different questions in a highly imbalanced evidence geometry.

## Heterogeneity

1 of 18 individual blocks had a negative within-block gap. A negative within-block value does not contradict the population-level H1 result; it identifies regimes where the two split strategies behaved differently and motivates the next stratified diagnostic.

## Interpretation boundary

- The analysis supports robustness to deleting any one existing block; it does not demonstrate external validity to a new laboratory or chemical regime.
- Leave-one-block-out deletion changes the represented target population and does not yield a replacement confidence interval.
- The original 10,000-replicate block/seed bootstrap remains the inferential analysis for H1.
