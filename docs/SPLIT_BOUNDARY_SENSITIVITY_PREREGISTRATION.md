# Split-boundary sensitivity preregistration

**Version:** 1.0  
**Freeze date:** 2026-08-10 (Asia/Shanghai)  
**Status at freeze:** post-confirmatory exploratory plan; no fit under this plan had started and no result from these new evaluation arms existed.  
**Confirmatory boundary:** this plan cannot alter, rescue, or replace the frozen H1 or H2 decisions in `PREREGISTRATION_V2.md`.

## 1. Questions

The already reported H1 contrast changes both the dependence boundary and the amount of training data available in a fold. This plan adds two analyses:

1. a molecule-random control whose 18 test-set sizes exactly match the 18 joint-block test-set sizes, to separate the observed joint-boundary contrast from the fold-specific reduction in training-set size; and
2. a descriptive evaluation ladder that holds out sources only, transitive analogue components only, or their joint connected components.

All new analyses use the same frozen 6,895 records, continuous log10(Papp) endpoint, D0 architecture, Delaney initialization procedure, five seed indices, batch size, optimizer, maximum epoch, patience, prediction format, and source-macro primary metric as the accepted H1 analysis.

## 2. Frozen analytical population

- Records: 6,895 curated source–structure groups.
- Unique stereochemical molecules: 6,862.
- Sources: 41.
- Analogue components: 305 under the frozen primary relation.
- Joint source/analogue blocks: 18.
- No record, endpoint, source identity, analogue edge, component, or joint block may be changed under this plan.
- All fold creation is outcome-blind. Permeability values are forbidden from fold manifests and assignment code.

## 3. Size-profile-matched molecule-random control

### 3.1 Target sizes

The 18 test sets must contain exactly 16, 16, 1,518, 11, 4,010, 3, 842, 4, 36, 18, 17, 105, 10, 8, 249, 18, 7, and 7 records, respectively, matching joint outer folds 1–18 in order. Consequently, each matched-random outer fold has exactly the same test and training record count as its paired joint fold.

### 3.2 Assignment rule

- The indivisible assignment unit is `molecule_id`; repeated records of the same stereochemical molecule cannot cross training and test sets.
- Molecule groups are ordered by a fixed pseudorandom permutation generated from NumPy `PCG64` with integer seed `20260810`.
- Groups containing more than one record are assigned first. For each group, the eligible target bins are those with remaining capacity at least as large as the group. One eligible bin is sampled with probability proportional to its remaining capacity.
- Singleton molecule groups are then pseudorandomly permuted with the same generator and used to fill all residual capacities exactly.
- The builder must fail unless all 18 target counts are exact, every curated record appears once, every molecule appears in exactly one fold, and rerunning the builder is byte-identical.
- The assignment file and its SHA-256 digest are frozen before the first matched-random model fit.

### 3.3 Nested development

For each matched-random outer fold, the other 17 matched-random bins are assigned intact to four outcome-blind inner baskets by the same frozen decreasing-size greedy rule used for the joint analysis. Seed index 0 is used for inner stopping. The rounded-up median of the four authenticated best epochs determines the validation-free outer refit epoch. Outer refits are run for seed indices 0–4. The accepted, seed-specific Delaney checkpoints may be reused because they are label-independent and already frozen; no fine-tuned checkpoint may initialize another fold.

### 3.4 Estimand and comparison

The primary new statistic is:

`joint-block source-macro MAE − size-profile-matched molecule-random source-macro MAE`.

Both arms contain one OOF prediction for every record and seed. The point estimate is the difference between five-seed mean source-macro MAEs. A 10,000-replicate deterministic paired block/seed bootstrap will use the original 18 joint blocks as the resampling unit. The interval is descriptive and post-confirmatory. The existing five-fold molecule-random result remains reported because it estimates the performance of the published random-CV procedure; the matched control addresses the narrower training-size question.

## 4. Evaluation-boundary ladder

### 4.1 Source-only arm

Use the already frozen `source_fold` column in `comparison_fold_manifest.csv`. Each complete source belongs to one of five deterministic folds with record counts 2,866, 1,518, 842, 835, and 834. For each outer fold, the other four source folds are the four inner baskets. No source may cross an outer or inner boundary.

### 4.2 Analogue-only arm

Use the already frozen `analogue_component_id_fold` column. Each complete analogue component belongs to one of five deterministic folds with record counts 3,716, 1,218, 654, 654, and 653. For each outer fold, the other four component folds are the four inner baskets. No analogue component may cross an outer or inner boundary.

### 4.3 Joint arm

Reuse the accepted 18-fold D0 joint-block OOF predictions without retraining or rescoring under a different boundary.

### 4.4 Reporting rule

Report molecule-random five-fold, size-profile-matched random, source-only, analogue-only, and joint source/analogue results in one table. For each arm report fold count, test-size range, five-seed source-macro MAE, row-micro MAE, and OOF completeness. Source-only and analogue-only results are post-confirmatory descriptive comparisons. Because fold count and training-size profiles differ, they must not be interpreted as an additive causal decomposition of provenance and chemical dependence.

## 5. Integrity and stop rules

- Every fit and prediction must use the existing split-safe contract, stage-specific empty namespace, append-only execution record, and immutable artifact manifest.
- Outer outcomes may be read only after predictions for that fold and seed are sealed.
- All scheduled predictions are retained; a poor result is not a reason to rerun, alter, or omit an arm.
- Hardware or software failures may be retried only under the identical scientific identity and must preserve failed-attempt evidence.
- Any change to data, endpoint, model, seeds, assignment seed, target sizes, metric, stopping rule, or reporting definition requires a dated amendment before the affected fit.
- These analyses cannot be used to nominate a replacement confirmatory model or revise D3.

## 6. Compute disclosure

The matched-random arm schedules 72 inner and 90 outer PAMPA fits. The source-only and analogue-only arms schedule 20 inner and 25 outer fits each. Delaney pretraining is reused from accepted seed-specific checkpoints. The total planned incremental workload is 252 PAMPA fits. Execution may be staged, paused at an accepted-stage boundary, or moved to another machine without changing the scientific identities.

