# Split-boundary manifest freeze

**Date:** 2026-08-10 (Asia/Shanghai)  
**Status:** outcome-blind manifest construction completed after the scientific plan freeze and before any new model fit.

The builder read only `comparison_fold_manifest.csv` and `outer_record_assignments.csv`; neither input contains permeability values. Six standard-library verification tests passed: exact target sizes and OOF completeness, molecule isolation, preservation of existing source/analogue/joint grouping, outer-fold exclusion from inner baskets, forbidden outcome-column rejection, and byte-identical rebuilding.

| Artifact | SHA-256 | Bytes |
|---|---|---:|
| `matched_size_random_record_assignments.csv` | `7dfcc9dfd01759fe6a5cb80161fb1706b77031fe5ee44aa164c0348a74e4845d` | 310,762 |
| `matched_size_random_outer_manifest.csv` | `d19e581523d0b62a22860fe68d35dc32a15900b0e9c1e5bec95f34bd00d0d33e` | 4,610 |
| `matched_size_random_inner_baskets.csv` | `0a8a05da03ef14f463721d4d931471353ecdc89653a6e770adc823e2bff7be16` | 3,158 |
| `evaluation_boundary_ladder_manifest.csv` | `48d5f732e99f203c7a8a4f62c33c14e7f3ade552fd2734afa10d195a6456b785` | 751,659 |

The matched-size record counts are exactly `16, 16, 1518, 11, 4010, 3, 842, 4, 36, 18, 17, 105, 10, 8, 249, 18, 7, 7`, paired to joint folds 1–18. Every one of the 6,895 curated records appears once, and each of the 6,862 molecule identifiers occurs in exactly one matched-size fold.

No training, tuning, selection, prediction, metric calculation, held-out-label access, or D4 access occurred during this freeze.

