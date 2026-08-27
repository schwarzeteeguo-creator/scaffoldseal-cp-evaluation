# CV_GOVERNANCE_V2.md

> Version 1.0, frozen on 2026-07-17 after explicit Recovery Plan Gate approval. Post-freeze changes require a dated deviation.

## 1. Immutable input graph

The v2 study reuses the independently verified v1 curation and analogue graph without changing its primary definition:

- molecule nodes: 6,862 unique curated stereochemical structures;
- chiral ECFP4 radius 2 edge when Tanimoto is at least 0.80 and molecular-weight ratio is 0.80–1.25;
- union with an exact one-token cyclic edit edge at unchanged topology and ring length;
- connected analogue components joined to all contributing sources in a bipartite graph;
- connected components of that bipartite graph are the 18 indivisible joint blocks.

The retired v1 `partition` column is ignored. It has no role in v2.

## 2. Outer evaluation

Use deterministic leave-one-joint-block-out (LOBO) evaluation:

1. sort blocks by `sealed_block_id` for stable fold identifiers;
2. hold one complete joint block out;
3. use the other 17 blocks as the outer-training population;
4. fit all transforms and models without the held-out block;
5. predict the held-out block once;
6. repeat until all 18 blocks have out-of-fold predictions.

No outer-block label may affect feature selection, preprocessing, early stopping, hyperparameters, model choice, residual quantiles or candidate replacement.

## 3. Inner baskets

Within each outer-training population, construct four inner baskets using no outcomes:

1. order the 17 available blocks by descending row count, breaking ties by `sealed_block_id`;
2. place each block into the basket with the smallest current row count, breaking basket ties by basket number;
3. keep every joint block intact;
4. serialize the assignments and hash them before training.

The four baskets are used as grouped inner folds. The imbalance and exact composition of every outer/inner fold are reported.

## 4. Roles of inner folds

- Classical hyperparameters are selected by mean inner source-macro MAE; row-micro MAE is the first tie-breaker and lower compute is the second.
- DMPNN architecture, featurizer, optimizer family and learning-rate schedule remain inherited from the locked baseline. Inner folds determine only the stopping epoch: use the median best epoch, rounded up, then refit on all 17 outer-training blocks.
- M1 and M2 are fixed switches, not result-driven additions.
- For empirical intervals, train the already fixed point model four times, each time omitting one inner basket; pool absolute out-of-fold residuals and use their finite-sample quantiles. Refit the point model on all outer-training blocks before predicting the outer block. These are described as cross-fitted empirical intervals, not exact split-conformal guarantees.

## 5. Seeds

- Primary stochastic seeds: `0, 1, 2, 3, 4`.
- Ridge and mean predictors are deterministic.
- Hyperparameter/stopping selection is performed with seed `0`; the selected setting/epoch is then evaluated with all five seeds.
- Every scheduled seed is reported. A run may be excluded only for a documented implementation or hardware failure identified without using its favorability.

## 6. Comparison splits

For H1, the locked DMPNN is also evaluated with the same five seeds on:

1. molecule-random five-fold CV;
2. chirality-aware Bemis–Murcko group five-fold CV;
3. source-group CV with deterministic five-basket balancing;
4. analogue-component group five-fold CV;
5. joint-block LOBO.

Five-fold assignments use the same descending-size greedy rule with seed-independent tie-breaking. Exact-scaffold overlap, source overlap and nearest-training ECFP4 similarity are saved for every held-out prediction.

The historical published/random and scaffold reproductions remain exposed context. They are not the v2 test and are not used in the primary H1 statistic.

## 7. Leakage tests

Before any model fit, automated tests must establish:

- zero joint-block overlap across every outer train/test pair;
- zero source and analogue-component overlap in joint-block LOBO;
- zero group overlap for each grouped comparison split;
- exactly one outer out-of-fold prediction slot per curated row and model/seed;
- training-only fitting of imputation, scaling, feature filtering and stopping decisions;
- no permeability/outcome-derived column in public feature or manifest tables;
- hashes for input curation, block graph, fold manifests and configuration files.

Any failure stops training until repaired and independently rechecked.

## 8. Aggregation under block imbalance

The 18 blocks are all scientifically relevant but not equally precise. Therefore:

- H1 primary metric is source-macro MAE, matching the new-source generalization claim and preventing row-rich sources from defining the result.
- H2 primary metric is source-macro MAE, giving each of the 41 sources equal weight.
- full per-block and per-source MAE tables are mandatory;
- block-macro median MAE and pooled row-micro MAE are secondary;
- Pearson, Spearman and R-squared are suppressed for any reporting stratum with fewer than 10 rows, but its absolute errors remain included;
- uncertainty uses 10,000 outer-block bootstrap replicates, with all rows and sources from a sampled block retained together. Seed is resampled within model where relevant.

This dual reporting prevents the 4,010-row block from silently defining every conclusion while also avoiding unstable equal-weight correlations from three-row blocks.

## 9. No retrospective final-test vocabulary

There is no fixed retrospective v2 final test. The 18 outer test blocks jointly form an out-of-fold benchmark. The only future untouched confirmation is the separately frozen prospective chemical panel. The retired v1 partition and label vault remain unusable.
