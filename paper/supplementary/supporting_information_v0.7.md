# Supporting Information

## DMPNN Performance Estimates Diverge across Evaluation Boundaries in a Cyclic-Peptide Permeability Benchmark

**Authors:** Yutao Guo<sup>1,2</sup>, Zihan Zhang<sup>1,2</sup>, Xuezhou Zhao<sup>1,2</sup>, Mengxi Chen<sup>1,2</sup>, Dan Wu<sup>1,3,*</sup>

**Affiliations:** <sup>1</sup>International College of Pharmaceutical Innovation, Soochow University, Suzhou 215123, China; <sup>2</sup>School of Pharmacy and Biomolecular Sciences, RCSI, 123 St Stephen's Green, Dublin 2, D02 VY51, Ireland; <sup>3</sup>Department of Chemistry, RCSI, 123 St Stephen's Green, Dublin 2, D02 YN77, Ireland. <sup>*</sup>Corresponding author: danwu@suda.edu.cn

### Scope and analysis boundary

This Supporting Information reports curation and split details, post-confirmatory robustness and sensitivity analyses for the frozen H1 comparison, a primary-source comparison with prior evaluation designs, the protocol timeline, deviations from the original split ladder, and the public-release inventory. Completed sensitivity analyses used the already accepted D0 out-of-fold predictions, checkpoint-only D3 residual reconstruction, or outcome-blind graph and curation inputs. No model was fitted, tuned, selected, replaced or recalibrated for those analyses. The internally prespecified 10,000-replicate block/seed bootstrap remains the inferential analysis for H1.

The H1 comparison used 6,895 curated records, 41 data sources, 18 frozen joint source/analogue blocks and five scheduled seeds. Joint-block and molecule-random prediction artifacts were verified against their frozen size and SHA-256 manifests before calculation.

The original version-2 split ladder listed molecule-grouped pseudorandom, chirality-aware Murcko, source-only, analogue-only and joint boundaries. Only the molecule-grouped and joint D0 arms were completed before outcome review. On 10 August 2026, source-only and analogue-only manifests were re-frozen operationally, and a newly added size-profile-matched molecule-random control was frozen, before any fit in those arms. Their pending results will be post-confirmatory and cannot alter the original H1 or H2 decisions. The Murcko arm remains uncompleted.

### Curation accounting and fold assignment

The raw PAMPA table contained 7,298 rows. A row was classified as irrecoverably censored when either source detection-limit field was populated; 372 rows met this frozen rule. The remaining 6,926 rows had parseable structures and sequences. They were grouped by source and isomeric canonical SMILES. Multi-row groups were collapsed only when year, database version, topology signature and canonical sequence agreed across uncensored rows; the analytical endpoint was their median. Exactly 31 compatible groups contained more than one eligible row, explaining the arithmetic difference between 6,926 usable raw rows and 6,895 curated records. There were no incompatible groups and no invalid structure/sequence rows in the frozen population.

| Curation stage | Rows or groups | Rule |
|---|---:|---|
| Raw PAMPA rows | 7,298 | Starting public table |
| Irrecoverably censored rows | 372 | Either detection-limit field populated |
| Uncensored usable rows | 6,926 | Parseable structure and sequence, numeric endpoint |
| Compatible multi-row groups collapsed | 31 | Same source and isomeric structure; compatible metadata; median endpoint |
| Curated source–structure records | 6,895 | Primary analytical population |
| Unique stereochemical molecules | 6,862 | `molecule_id` assignment units |

**Table S1 | Raw-to-curated accounting.** The row-level curation manifest retains raw identifiers, censoring flags, inclusion or linked-exclusion status, and the final curated identifier.

For the molecule-grouped pseudorandom comparison, all records sharing `molecule_id` remained in one fold. Molecule groups were sorted by decreasing size with stable structure-derived identifiers breaking ties, then placed into the currently smallest of five folds. Each fold contained 1,379 records. Complete sources and analogue components were allowed to cross this comparison boundary. In the joint analysis, neither a source nor an analogue component could cross the outer boundary.

### H1 block influence and aggregation sensitivity

Each frozen joint block was omitted in turn, and source-macro MAE was recalculated across the remaining sources for both evaluation arms. The difference was defined as joint-block MAE minus molecule-random MAE. This deletion analysis measures sensitivity to individual observed blocks; it does not define a new target population or confidence interval.

Alternative descriptive estimands assigned equal weight to records (row micro), sources (source macro) or blocks (block macro). For block-macro aggregation, source-specific MAEs were averaged within each block, blocks were weighted equally within each seed, and the five seed summaries were averaged.

| Analysis | Joint-block MAE | Molecule-random MAE | Gap |
|---|---:|---:|---:|
| Row micro | 0.6430 | 0.2928 | 0.3502 |
| Source macro | 1.0046 | 0.4672 | 0.5373 |
| Equal-block macro | 0.9048 | 0.4386 | 0.4662 |

**Table S2 | H1 aggregation sensitivity.** Values are means across five scheduled seeds. The three weighting schemes estimate different target quantities and are reported descriptively.

[[FIGURE_S1]]

### Small-cluster and high-leverage sensitivity

The five accepted seeds were first averaged within each source, giving 41 paired source-level joint-minus-random MAE differences nested in 18 sealed blocks. Three calculations then examined sensitivity to the small and unequal cluster set. An intercept-only CR1 variance estimator clustered the paired source differences by sealed block and used a t critical value with 17 degrees of freedom. A delete-one-block jackknife used the same t reference. Finally, all 2^18 assignments obtained by independently changing the sign of each block contribution were enumerated.

| Sensitivity | Estimate | Standard error | 95% interval or p value |
|---|---:|---:|---:|
| CR1 block-clustered t interval | 0.5373 | 0.0757 | 0.3777–0.6970 |
| Delete-one-block jackknife t interval | 0.5373 | 0.0756 | 0.3779–0.6968 |
| Exhaustive block sign flip | 0.5373 | — | p = 0.00001526 |

**Table S3 | Small-cluster sensitivity of the H1 source-macro gap.** Delete-one-block point estimates ranged from 0.4724 to 0.5523. Seventeen of 18 block-specific mean gaps were positive, and 4 of 262,144 sign assignments were at least as extreme as observed. These analyses are post-confirmatory. CR1 and jackknife calculations treat the 18 blocks as independent clusters; the sign-flip test additionally assumes joint sign symmetry under the null. None replaces the internally prespecified block/seed bootstrap or creates additional independent regimes.

### Source-stratified error and point-prediction calibration

Absolute and signed errors were summarized within each source and seed, then averaged across the five seeds. Positive signed error denotes predicted permeability that is too high (less negative log10 Papp), whereas negative signed error denotes predicted permeability that is too low. Sources were retained regardless of size; extreme estimates from small sources are localization diagnostics rather than stable independent effects.

For point-prediction calibration, the five OOF predictions for each record were averaged first. Records were divided into ten equal-frequency prediction bins separately within each evaluation arm. Mean observed permeability was compared with mean predicted permeability in each bin. Calibration slope was obtained from a descriptive least-squares regression of observed values on mean predictions; no recalibration function was fitted or applied.

| Calibration summary | Joint-block | Molecule-random |
|---|---:|---:|
| Calibration slope | 0.5950 | 0.9637 |
| Calibration intercept | -2.2192 | -0.2075 |
| Pearson correlation | 0.3252 | 0.8729 |
| Global signed bias | -0.1746 | -0.0010 |
| Decile-weighted absolute calibration error | 0.1808 | 0.0269 |
| Median absolute source bias | 0.5779 | 0.0784 |
| Sources with absolute bias above 0.5 | 23 of 41 | 3 of 41 |

**Table S4 | Source-level bias and point-prediction calibration.** Calibration used one five-seed mean OOF prediction per record. Prediction bins were arm-specific and contained all 6,895 records.

[[FIGURE_S2]]

### Analogue-threshold sensitivity of evidence geometry

The outcome-blind analogue graph was reconstructed at the prespecified descriptive chiral ECFP4 Tanimoto thresholds of 0.70, 0.80 and 0.90. Fingerprint radius and length, chirality, the molecular-weight-ratio constraint, the exact one-token cyclic-edit relation, source provenance and all public curated records were held fixed. The threshold-0.80 reconstruction exactly reproduced every frozen primary component and block identifier.

Alternative thresholds were used only to compare graph and partition geometry. Existing OOF predictions were not rescored under these partitions because their training/test boundaries differ. A valid alternative-threshold performance comparison would require complete newly frozen nested retraining and would be exploratory after release of the primary outcomes.

| Tanimoto threshold | Analogue edges | Analogue components | Joint blocks | Largest block share | Top-three share | Effective blocks |
|---:|---:|---:|---:|---:|---:|---:|
| 0.70 | 648,106 | 91 | 15 | 58.80% | 93.02% | 2.435 |
| 0.80 | 141,425 | 305 | 18 | 58.16% | 92.39% | 2.480 |
| 0.90 | 38,787 | 745 | 20 | 57.84% | 92.07% | 2.504 |

**Table S5 | Outcome-blind analogue-threshold sensitivity.** The effective block count is the inverse Simpson concentration index calculated from record shares.

[[FIGURE_S3]]

### Prior-work comparison of evaluation boundaries

The comparison below was updated from primary reports through 10 August 2026. It is intended to delimit the present contribution, not to claim novelty for source holdout, chemical clustering, scaffold splitting, nested validation, or similarity-aware splitting individually.

| Study | Reported evaluation boundary | Complete source held out? | Transitive analogue component held out? | Same boundary throughout nested development? |
|---|---|---:|---:|---:|
| Geylan et al. (2024) | Four large external sources; separate source-group and canonical-group scenarios | Yes, in source arms | No | Partly; grouping scenarios were separate |
| CycPeptMP (2024) | Kennard–Stone holdout plus cross-validation | No | No | No |
| PeptideCLM (2025) | Six embedding/PCA clusters, leave one cluster out | No | No | Development within retained clusters, not joint blocks |
| MultiCycPermea (2025) | Random, Murcko/Jaccard-cluster OOD and permeability-cliff settings | No | No | No joint source–analogue nesting |
| 13-method benchmark (2025) | Random and Murcko-scaffold splits; external set | No | No | No joint source–analogue nesting |
| DataSAIL (2025) | Generic similarity-aware one- and two-dimensional constrained splits | User-defined | User-defined | Depends on the user workflow |
| Present study | Molecule-grouped pseudorandom and joint source–analogue connected-component LOBO | Yes | Yes | Yes for the accepted joint analysis |

**Table S6 | Primary-source comparison of evaluation boundaries.** Geylan et al. is the closest provenance-aware precedent located. The present distinction is simultaneous closure of alternating source–analogue paths throughout model development. This is a search-limited comparison, not proof that no unindexed or unpublished study used a similar design. A post-confirmatory source-only, analogue-only and size-profile-matched random ladder has been frozen but is not yet reported.

Primary report identifiers: Geylan et al., DOI 10.1039/D4DD00056K; CycPeptMP, DOI 10.1093/bib/bbae417; PeptideCLM, DOI 10.1021/acs.jcim.4c01441; MultiCycPermea, DOI 10.1186/s12915-025-02166-2; 13-method benchmark, DOI 10.1186/s13321-025-01083-4; DataSAIL, DOI 10.1038/s41467-025-58606-8.

### Protocol timeline and public identifiers

| Date | Protocol event | Public-release evidence |
|---|---|---|
| 2026-07-17 | Original v1 feasibility gate failed; compromised v1 final partition and vault retired | Failed protocol, decisions D-018/D-019 and repair audit retained |
| 2026-07-17 | Version-2 recovery map, CV governance and experiment plan frozen internally before any v2 fit | File hashes listed in `PUBLIC_PROTOCOL_TIMELINE.md`; no independent public timestamp |
| 2026-07-17 to 2026-07-19 | Pre-fit split, preprocessing, stopping and prediction controls reviewed | Test and audit records; training remained blocked until acceptance |
| 2026-07-26 to 2026-07-27 | H1 random-comparison plan accepted and executed | Accepted stage manifests, sealed OOF predictions and bootstrap outputs |
| 2026-08-09 | D123 sealed audit and one controlled metric release | Decisions D-052 to D-054 and H2 verdict report |
| 2026-08-10 | Zero-training sensitivity analyses and split-boundary completion plan frozen | Reporting scripts, input hashes and post-confirmatory internal freeze |

**Table S7 | Protocol and release timeline.** The complete chronology, file hashes and distinction between an internal freeze and an independently time-stamped public preregistration are provided in `PUBLIC_PROTOCOL_TIMELINE.md`. The public repository is **https://github.com/schwarzeteeguo-creator/scaffoldseal-cp-evaluation**; the remaining unresolved archival field is **[Zenodo DOI pending]**. Local hashes alone are not described as independent public preregistration.

### Itemized Supporting Information and release mapping

| SI item | Local/release files | Status in this version |
|---|---|---|
| Curation accounting | `curation_manifest_release_safe.csv`; `curated_group_manifest_release_safe.csv`; audit JSON; curation code and hashes | Included as release-safe manifests; structures and endpoint labels excluded |
| Primary joint split | outer record assignments, outer-fold manifest and inner-basket manifest | Complete and frozen |
| Comparison splits | molecule-grouped assignments; source-only, analogue-only and matched-size manifests | Outcome-blind manifests complete; corresponding model results pending |
| H1 headline | accepted joint/random metric files, OOF manifests and bootstrap output | Complete and frozen |
| H1 influence and aggregation | `analyze_h1_block_influence.py`; three CSVs; JSON summary; Fig. S1 | Complete |
| H1 source calibration | `analyze_h1_stratified_calibration.py`; four CSVs; JSON summary; Fig. S2 | Complete |
| Analogue-threshold geometry | `analyze_block_definition_sensitivity.py`; four CSVs; JSON summary; Fig. S3 | Complete |
| H1 small-cluster sensitivity | `analyze_h1_small_cluster_sensitivity.py`; paired-source, cluster, deletion CSVs; JSON summary | Complete |
| Censoring and source metadata | `analyze_curation_source_audit.py`; release-safe manifests; source and note summaries; JSON | Complete; no raw structures or labels redistributed |
| D3 coverage aggregation | `analyze_d3_coverage_stratification.py`; per-seed, per-fold and per-source CSVs; JSON | Complete; checkpoint-only predictions, no fitting |
| Prior-work boundary comparison | `PRIOR_WORK_COMPARISON_NOTES.md`; `prior_work_evaluation_comparison.csv` | Complete through search date |
| Protocol and frozen files | v2 internal protocol, research map, CV governance, experiment plan, deviation table and public timeline | Complete locally; external timestamp absent |
| Public archive metadata | release checksum, `CITATION.cff`, license, GitHub URL and Zenodo DOI | Pending author/license/deposition actions |

**Table S8 | Mapping of SI statements to actual files.** The authoritative package-level checklist is `RELEASE_INVENTORY_V0.7.md`. A manuscript statement must be removed or marked pending if its corresponding released file is absent.

### Censoring and source-metadata audit

The frozen curation rule excluded every row for which either detection-limit field was populated. A zero-training audit classified the two free-text fields conservatively and checked whether an excluded source-structure group was otherwise represented by an uncensored replicate. The classes below describe database annotations; they are not assumed to be mutually commensurate assay bounds.

| Audit item | Rows or sources |
|---|---:|
| Excluded rows carrying a detection-limit annotation | 372 |
| Raw-data sources containing at least one excluded row | 23 |
| Database-assigned or reported floor | 259 |
| Explicit upper limit | 40 |
| No reportable value | 19 |
| Not detected or below limit of detection | 9 |
| Solubility failure or not tested | 7 |
| Other detection-limit note | 38 |
| Excluded rows linked to a retained uncensored group | 2 |
| Excluded rows without a retained continuous-endpoint group | 370 |
| Completely censored source | 1 source, 10 rows |

**Table S9 | Censoring semantics and retained-group linkage.** Note classes sum to 372. Numeric database entries among the excluded rows were heterogeneous: 270 were recorded as -10, 45 as -7, 15 as -4, and the remainder used other values. The raw table provided no harmonized structured columns for PAMPA pH, membrane composition, incubation time, temperature, or donor/acceptor conditions. Accordingly, the analysis neither substituted sentinel values as ordinary endpoints nor imposed a universal quantitative censoring bound. Source identity is treated as a combined provenance and protocol proxy.

### Protocol completion and deviations

| Planned element | Status before outcome review | Status in this version | Evidential role |
|---|---|---|---|
| Joint source/analogue D0 | Completed | Reported | Primary internally prespecified evaluation |
| Molecule-grouped pseudorandom D0 | Completed | Reported | Internally prespecified H1 comparison |
| Chirality-aware Murcko D0 | Not completed | Pending | Originally planned secondary boundary; no result implied |
| Source-only D0 | Not completed | Outcome-blind manifest re-frozen 10 August | Post-confirmatory when run |
| Analogue-only D0 | Not completed | Outcome-blind manifest re-frozen 10 August | Post-confirmatory when run |
| Size-profile-matched molecule-random D0 | Not in original ladder | Newly frozen 10 August; pending | Post-confirmatory training-size control |
| D1-D3 joint-block ladder | Completed | Reported | Internally prespecified H2 tests and ablations |
| Prospective matched-assay H3 | Not performed | Pending external data | Future prospective confirmation |

**Table S10 | Completion and evidential status of planned and added analyses.** Source-only and analogue-only boundaries were listed in the original plan but were not completed before the joint and molecule-grouped outcomes were opened. Their operational re-freeze does not restore confirmatory status. The matched-size control was added only after recognizing that the completed H1 comparison changed both dependence and training-size profiles.

### D3 coverage aggregation sensitivity

Each curated record contributed five algorithmic seed realizations to the primary slot-pooled coverage denominator. These realizations are useful for measuring execution variability but are not independent experimental observations. We therefore report the seed range and equal-source and equal-block descriptive estimands without a binomial confidence interval.

| Nominal coverage | Slot-pooled | Per-seed range | Equal-source macro | Equal-block macro | Block range |
|---:|---:|---:|---:|---:|---:|
| 50% | 44.1% | 41.9-46.9% | 31.9% | 28.4% | 0.0-80.0% |
| 80% | 67.2% | 65.4-68.7% | 58.6% | 59.5% | 0.0-100.0% |
| 90% | 77.0% | 75.8-78.1% | 71.4% | 75.0% | 0.0-100.0% |

**Table S11 | Sensitivity of D3 empirical-interval coverage to aggregation.** Slot pooling weights every record-seed pair equally; source and block macro estimates weight the 41 sources and 18 outer blocks equally, respectively. The wide block range reflects highly unequal and sometimes very small omitted regimes. All aggregations show under-coverage at the higher nominal levels.

### Source-data files

The influence package contains `h1_leave_one_block_out.csv`, `h1_aggregation_sensitivity.csv` and `h1_per_block_effect.csv`.

The stratified-calibration package contains `h1_source_stratified_metrics.csv`, `h1_block_stratified_metrics.csv`, `h1_prediction_decile_calibration.csv` and `h1_calibration_summary.csv`.

The threshold-sensitivity package contains `block_definition_threshold_summary.csv`, `block_definition_ranked_sizes.csv`, `block_definition_partition_comparison.csv` and `block_definition_record_assignments.csv`.

The small-cluster package contains `h1_paired_source_effects.csv`, `h1_cluster_contributions.csv`, `h1_small_cluster_delete_one.csv` and `h1_small_cluster_sensitivity_summary.json`.

The literature package contains `prior_work_evaluation_comparison.csv` and `PRIOR_WORK_COMPARISON_NOTES.md`.

The curation/source-audit package contains `source_censoring_and_endpoint_summary.csv`, `censoring_note_class_summary.csv`, `censored_database_numeric_values.csv`, `curation_manifest_release_safe.csv`, `curated_group_manifest_release_safe.csv`, `curation_source_audit_summary.json` and `SHA256SUMS`.

The D3 coverage-stratification package contains `d3_coverage_by_seed.csv`, `d3_coverage_by_outer_fold.csv`, `d3_coverage_by_source.csv`, `d3_coverage_aggregation_sensitivity.csv`, `d3_coverage_stratification_summary.json` and `SHA256SUMS`.
