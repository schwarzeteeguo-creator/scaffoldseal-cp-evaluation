# Analogue-threshold sensitivity of block geometry

## Scope and integrity boundary

This post-confirmatory sensitivity analysis rebuilt the outcome-blind analogue and source/component graphs at the prespecified descriptive thresholds 0.70, 0.80 and 0.90. Chiral ECFP4 radius/length, the 0.80–1.25 molecular-weight ratio, the exact one-token cyclic-edit rule, source provenance and all 6,895 public curated records were held fixed. No permeability value, fitted model or prediction was used to form any graph.

The threshold-0.80 reconstruction exactly reproduced the frozen 141,425 edges, 305 analogue components, 18 joint blocks and every record-level primary component/block identifier.

## Results

- Threshold 0.70: 648,106 analogue edges, 91 analogue components, 15 joint blocks, largest share 58.80%, top-three share 93.02%, effective block count 2.435.
- Threshold 0.80: 141,425 analogue edges, 305 analogue components, 18 joint blocks, largest share 58.16%, top-three share 92.39%, effective block count 2.480.
- Threshold 0.90: 38,787 analogue edges, 745 analogue components, 20 joint blocks, largest share 57.84%, top-three share 92.07%, effective block count 2.504.

Changing the similarity threshold strongly altered the molecular graph: analogue components ranged from 91 at 0.70 to 745 at 0.90. Source/component closure absorbed much of this variation, leaving only 15–20 joint blocks. The largest block remained 57.84–58.80% of all records, the three largest remained 92.07–93.02%, and the inverse-Simpson effective number of blocks remained 2.435–2.504.

Thus, the manuscript's central evidence-geometry limitation—many molecular rows but very few highly concentrated independent source/analogue regimes—is not an artefact of choosing exactly 0.80. The exact identities of smaller components and blocks do change, as quantified in the partition-comparison source data.

## Interpretation boundary

- This result concerns graph and partition geometry only; it does not show that H1 performance is numerically invariant to the threshold.
- The frozen OOF predictions cannot be validly rescored as if they came from the 0.70 or 0.90 partitions because the corresponding training/test boundaries differ.
- A valid threshold-specific performance comparison would require complete nested retraining under separately frozen alternative partitions. Because confirmatory outcomes are already known, such runs would be exploratory and cannot replace the primary 0.80 result.
- The current analysis therefore strengthens the evidence-geometry argument while preserving the original confirmatory boundary.
