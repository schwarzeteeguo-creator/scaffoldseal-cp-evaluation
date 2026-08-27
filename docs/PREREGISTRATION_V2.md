# PREREGISTRATION_V2.md

> Option-A recovery preregistration. Version 1.0, frozen on 2026-07-17 after explicit Recovery Plan Gate approval and before any v2 model fit. `PREREGISTRATION.md` v1.0 remains the immutable failed protocol. Post-freeze changes require a dated deviation.

- **Primary venue:** *Journal of Chemical Information and Modeling*
- **Retrospective endpoint:** PAMPA `log10(Papp in cm/s)`
- **Confirmatory computational population:** 6,895 curated source-structure groups in 18 verified joint source-component blocks
- **Future prospective population:** one separately frozen, newly purchased head-to-tail cyclic-peptide scaffold series

## 1. Immutable data and groups

The primary curation, 0.80 chiral ECFP4 plus one-token cyclic-edit analogue rule, 305 analogue components and 18 joint blocks are inherited unchanged from the independently verified Milestone 0 artifacts. The retired v1 partition and label vault have no analytical role.

All 18 blocks are included in deterministic leave-one-joint-block-out evaluation. The remaining 17 blocks form four label-blind inner baskets as specified in `CV_GOVERNANCE_V2.md`. No permeability value is used to create a fold.

## 2. Confirmatory models

- **D0:** locked reproduced DMPNN.
- **D3:** D0 plus fixed source/component-balanced loss and the frozen 13-item chemistry/topology descriptor vector.
- **Classical comparator:** ridge, Random Forest or XGBoost selected within each outer fold from the closed grids in `EXPERIMENT_PLAN_V2.md`.
- **Required ablations:** D1 (balance only) and D2 (descriptors only).

Seeds are `0–4`. Mean and ridge predictors are deterministic. DMPNN stopping epoch is the rounded-up median inner best epoch; architecture and optimizer family are not tuned.

## 3. H1 — evaluation optimism

**Primary statistic:** D0 equal-source macro MAE under joint-block LOBO minus D0 equal-source macro MAE under molecule-random five-fold CV, computed on the same curated population and seeds.

**Support rule:** H1 is supported only if:

1. the gap is at least `0.10 log10(Papp)`; and
2. the two-sided 95% confidence interval from 10,000 outer-block/seed bootstrap replicates is entirely above zero.

Murcko, source-only and analogue-component splits; nearest-training similarity; source/scaffold overlap; and the historical reproduced anchors are secondary mechanism/context analyses. They cannot replace the primary comparison.

## 4. H2 — usable prediction under joint shift

**Primary metric:** equal-source macro MAE across all 41 sources using joint-block out-of-fold predictions.

**Fixed candidate:** D3.

**Support rule:** H2 is supported only if all conditions hold:

1. D3 improves source-macro MAE by at least `0.03` versus D0;
2. D3 improves by at least `0.05` versus the nested-selected classical comparator;
3. the 95% paired outer-block bootstrap interval for D3's improvement versus the stronger comparator excludes zero;
4. pooled 90% cross-fitted empirical interval coverage is 85–95%; and
5. D3 source-macro Spearman is no more than 0.03 below the best comparator.

D1, D2, interval-width, risk-coverage, row-micro metrics and per-block analyses are secondary. They do not rescue a failed H2.

## 5. H3 — future prospective enrichment

H3 is a provisional claim and requires a second freeze after assay qualification and before candidate-group disclosure.

The primary comparison will be predicted-good versus matched-random within one fixed scaffold, matched by medicinal-chemistry design stratum. Provisional support requires at least 16 QC-passing compounds per primary group, a good-minus-random difference of at least `0.50 log10(Papp)`, one-sided stratum-permutation `p < 0.05`, a 95% stratum-bootstrap interval above zero, no more than a 10-percentage-point solubility/recovery disadvantage and no more than a 15-percentage-point differential attrition unless the prespecified attrition sensitivity analysis agrees.

Pilot compounds cannot support H3. The exact donor concentration, solubility floor, recovery threshold, positive/negative controls, exclusion rules and replicate model must be frozen after assay qualification and before candidate selection.

## 6. Secondary outcomes

- row-micro MAE/RMSE and block-macro median MAE;
- source-macro and eligible-stratum Spearman;
- interval coverage/width at 50%, 80% and 90%;
- calibration error and risk-coverage;
- performance versus nearest-training similarity, block size, topology, ring size, charge, N-methylation and noncanonical-residue count;
- top-quartile enrichment where a held-out source has enough rows;
- matched-edit analyses as descriptive, source-specific diagnostics only.

Correlations are not reported for strata with fewer than 10 rows. Absolute errors remain included.

## 7. Bootstrap and multiplicity

- Use 10,000 deterministic bootstrap replicates with the 18 outer blocks as the resampling unit; all rows and sources inside a sampled block remain together.
- For seed-sensitive models, seed is resampled within model. H1 resamples the corresponding random-CV result as well as outer blocks.
- H1 and H2 are sequential, separately reported claims. H1 is primary. A failed H2 does not negate H1 and cannot be hidden.
- Non-primary pairwise model comparisons use Holm correction within analysis family.

## 8. Missingness, censoring and failures

- The primary continuous retrospective population retains the frozen v1 censoring/exclusion rule; no exclusions depend on prediction error.
- Preprocessing is fitted inside outer training only.
- Every scheduled seed and block prediction is reported. Hardware/software failures are documented before metric review and rerun from the frozen configuration.
- Tiny blocks remain in absolute-error aggregation. Poorly performing sources, topologies or compounds are not removed post hoc.
- No primary prospective outcome is imputed. Synthesis, purification, incoming-QC and assay failures remain in the attrition flow.

## 9. Model nomination after retrospective evaluation

The prospective model is nominated only after all prespecified retrospective OOF files are locked: eligible coverage 85–95%, then lowest source-macro MAE, then narrower interval within a 0.01 MAE tie, then lower compute. If no model meets coverage, choose lowest source-macro MAE and explicitly record calibration failure.

This nomination may use retrospective results because the prospective outcomes do not yet exist. It must be frozen before the candidate universe is scored for group assignment.

## 10. Falsification and stop rules

- H1 fails if either H1 support condition fails.
- H2 fails if any H2 support condition fails.
- H3 fails if any final prospective support condition fails.
- Any group-overlap or outcome-leakage test failure stops affected training.
- The primary analogue threshold, blocks, metrics, seeds and thresholds are not changed after results are viewed.
- No fixed retrospective final test will be created after seeing LOBO errors.

## 11. Authorization boundary

Recovery Plan Gate approval authorizes only the v2 local computational work and its independent verification. It does not authorize vendor contact, quotes, purchasing, synthesis, assays, paid compute or external data transfer.

## 12. Deviation log after freeze

| Date | Change | Reason | Before/after affected results? | Consequence |
|---|---|---|---|---|
| — | None at freeze | — | — | — |
