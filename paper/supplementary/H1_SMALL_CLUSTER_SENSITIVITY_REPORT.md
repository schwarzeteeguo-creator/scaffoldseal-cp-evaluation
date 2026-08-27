# H1 small-cluster sensitivity report

Status: post-confirmatory, zero-training sensitivity analysis of frozen H1 out-of-fold predictions.

## Result

- Frozen source-macro gap (joint minus molecule-random MAE): 0.5373.
- CR1 sealed-block-clustered interval with t(17) critical value: 0.3777 to 0.6970 (SE 0.0757).
- Delete-one-block jackknife interval with t(17) critical value: 0.3779 to 0.6968 (SE 0.0756).
- Delete-one-block estimates ranged from 0.4724 to 0.5523.
- Exhaustive two-sided sealed-block sign-flip test: p = 0.00001526 (4 of 262144 assignments).
- Positive block-specific mean gaps: 17 of 18.

## Interpretation boundary

These analyses address finite-cluster and leverage sensitivity; they do not replace the preregistered 10,000-replicate block/seed bootstrap. The CR1 and jackknife intervals rely on treating the 18 sealed blocks as independent clusters. The exact sign-flip result additionally relies on joint sign symmetry under the null. Unequal source and record counts remain part of the target benchmark rather than being removed by reweighting.
