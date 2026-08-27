# EXPERIMENT_PLAN_V2.md

> Option-A recovery plan, version 1.0, frozen on 2026-07-17 after explicit Recovery Plan Gate approval. Computational BUILD is authorized; wet-lab work, procurement and paid compute remain unauthorized. The v1 plan remains preserved as a failed protocol.

## 1. Objective and publication logic

The study will quantify how cyclic-peptide PAMPA prediction degrades when both complete data sources and connected analogue series are excluded, then test whether a lightweight chemistry-aware model and empirical uncertainty remain useful under that shift. A later, separately approved prospective single-scaffold panel will test whether the historically selected model produces real medicinal-chemistry enrichment.

The paper is viable if H1 is supported and the benchmark is executed rigorously, even if the proposed DMPNN modification does not win. A strong H2 and prospective H3 would raise the work from a benchmark/audit paper toward a stronger JCIM paper; a chemically interpretable prospective series could support a *Journal of Medicinal Chemistry* stretch submission.

## 2. Locked data facts

- 7,298 raw PAMPA rows.
- 372 censored rows excluded from the primary continuous endpoint under the v1 frozen rule.
- 6,895 curated source-structure groups, 6,862 unique molecules and 41 sources.
- 141,425 frozen analogue edges, 305 analogue components and 18 joint source-component blocks.
- Largest block: 4,010 rows, 3,978 molecules, 162 analogue components and 20 sources.
- The previous four-way minimum of 950 components was impossible; no v1 model was trained.
- The v1 final partition and vault are permanently retired and will not be reopened, relabeled or rehashed for v2.

## 3. Retrospective validation design

`CV_GOVERNANCE_V2.md` is normative. In summary:

- outer evaluation: 18-fold leave-one-joint-block-out;
- inner evaluation: four deterministic label-blind baskets over the remaining 17 blocks;
- all transforms, stopping rules, hyperparameters and residual quantiles fit within outer training;
- exactly one joint-block out-of-fold prediction per curated record/model/seed;
- no fixed retrospective final test;
- all fold manifests and hashes are frozen before the first fit.

The locked DMPNN is additionally evaluated under molecule-random, chirality-aware Murcko, source-only and analogue-component group splits. This split ladder isolates how increasingly strict independence changes apparent accuracy.

## 4. Model ladder

### 4.1 Trivial and classical

1. outer-training mean;
2. ridge on 2,048-bit radius-2 ECFP plus fixed descriptors, `alpha` in `{0.01, 0.1, 1, 10, 100}`;
3. Random Forest with 500 trees, `max_features` in `{sqrt, 0.25}`, `min_samples_leaf` in `{1, 3, 5}` and `max_depth` in `{None, 20}`;
4. XGBoost with `n_estimators` in `{300, 800}`, `max_depth` in `{4, 8}`, `learning_rate` in `{0.03, 0.10}`, `reg_lambda` in `{1, 10}`, `subsample=0.8` and `colsample_bytree=0.8`.

Classical hyperparameters are chosen inside each outer fold using its four inner baskets. The strongest classical comparator is the nested inner-selected algorithm, not a model selected after viewing outer errors.

### 4.2 Deep models

- **D0:** locked BenchmarkCycPeptMP DMPNN at maintained commit `d82aa3c5c9c849dbd584e8669132ed3d33e50a27`, preserving the original featurizer, architecture, Delaney pretraining, batch size 64, optimizer/scheduler, maximum 2,000 epochs and patience 200.
- **D1 / M1:** D0 with group-balanced loss. Each source receives equal total weight; within source, each analogue component receives equal total weight; within component, records share that weight.
- **D2 / M2:** D0 with a frozen descriptor vector concatenated only at the readout.
- **D3 / M1+M2:** both modifications together. D3 is the prespecified H2 candidate.

The M2 descriptor vector is ring size, topology class, molecular weight, cLogP, TPSA, H-bond donors, H-bond acceptors, rotatable bonds, formal charge, fraction sp3, stereocenter count, N-methyl count and noncanonical-residue count. Descriptor computation and missingness flags are fixed before outcomes are modeled. Continuous descriptors use outer-training statistics only.

Inner folds set the DMPNN stopping epoch; there is no open-ended deep hyperparameter search. Primary seeds are `0–4`. D0 may also be shown against the historical ten-seed reproduction, but that historical comparison is contextual.

### 4.3 Uncertainty

For D0, D3 and the nested-selected classical procedure, construct 50%, 80% and 90% cross-fitted empirical intervals from inner out-of-fold absolute residuals. Report coverage, width, calibration error, risk-coverage and coverage by outer block and nearest-training similarity. Do not call these intervals distribution-free guarantees.

## 5. Hypotheses and analyses

### H1 — evaluation optimism

- Fixed model: D0.
- Primary statistic: joint-block-LOBO source-macro MAE minus molecule-random five-fold source-macro MAE on the same curated records and seeds.
- Support: gap at least `0.10 log10(Papp)` and the two-sided 95% outer-block/seed bootstrap interval lies entirely above zero.
- Mechanism diagnostics: Murcko, source-only and analogue-only errors; exact-scaffold/source overlap; nearest-training ECFP4 similarity; block size and topology.

### H2 — useful model under shift

- Fixed candidate: D3.
- Comparators: D0 and the nested-selected classical procedure.
- Primary statistic: 41-source macro MAE from joint-block out-of-fold predictions.
- Support requires all of: D3 improves by at least `0.03` versus D0; improves by at least `0.05` versus the classical comparator; the 95% paired outer-block bootstrap interval versus the stronger comparator excludes zero; pooled 90% interval coverage is 85–95%; and source-macro Spearman is no more than 0.03 below the best comparator.
- D1 and D2 are mandatory ablations. They cannot replace D3 for the confirmatory H2 verdict.

### H3 — future prospective enrichment

H3 is only provisionally specified here. It must be frozen again after assay qualification and before candidate identities or predictions are released to experimental staff.

- Population: one head-to-tail cyclic-peptide scaffold and a frozen enumerated analogue universe.
- Primary comparison: predicted-good versus matched-random.
- Matching unit: predeclared medicinal-chemistry design stratum, not scaffold, because all compounds share one scaffold.
- Provisional support: at least 16 QC-passing compounds per primary group; adjusted good-minus-random difference at least `0.50 log10(Papp)`; one-sided design-stratum permutation `p < 0.05`; 95% design-stratum bootstrap interval excludes zero; solubility/recovery pass rate not worse by more than 10 percentage points; group attrition difference no more than 15 percentage points or a prespecified sensitivity analysis agrees.

Pilot compounds establish feasibility only and cannot support H3.

## 6. Model nomination for the prospective panel

After all retrospective out-of-fold predictions are locked, select one model by the following outcome-independent ordering rule applied to the prespecified metrics:

1. eligible models must have pooled 90% empirical coverage between 85% and 95%;
2. among eligible models, choose lowest source-macro MAE;
3. if within 0.01 MAE, choose narrower 90% intervals;
4. if still tied, choose the lower-compute model;
5. if no model is coverage-eligible, select lowest source-macro MAE and state that uncertainty qualification failed.

Fit the nominated model to the full retrospective population only after its configuration is frozen. Prospective outcomes remain untouched and are the external confirmation.

## 7. Prospective single-scaffold panel

### 7.1 Scaffold and universe selection

Before prediction-based group assignment, the chemistry team selects one scaffold using only:

- a mature, supplier-accepted head-to-tail synthesis/cyclization route;
- an unambiguous stereochemical structure and tractable analytical method;
- a feasible enumerated universe of at least 120 analogues from commercially available or supplier-approved monomers;
- variation at at least three edit positions while holding cyclization topology fixed;
- no selection based on retrospective model rank or unpublished prospective PAMPA values.

Freeze the scaffold, enumerated universe, feasibility flags, diversity rules and ordered replacement list. Predictions are then generated once.

### 7.2 Groups and size

- predicted-good;
- predicted-poor;
- high-uncertainty/OOD;
- matched-random.

Pilot: commission 16–24 starts to obtain 12–16 QC-passing compounds. Publication panel: approximately 80–100 starts to target 64 QC-passing compounds, 16 per group. Expansion requires a separate user approval after pilot delivery and assay feasibility.

### 7.3 Chemistry, QC and assay

- Preferred route: qualified custom peptide synthesis; internal synthesis is a fallback.
- Supplier specification: full stereochemistry, residue modification state, cyclization linkage, salt/counterion, requested amount, minimum purity, HPLC trace and LC-HRMS.
- Incoming study QC: identity/purity confirmation, concentration/content assignment, vehicle stability and precipitation check; new active compounds require publication-appropriate HPLC, HRMS and NMR evidence.
- Primary endpoint: one matched PAMPA `log10(Papp)` protocol with donor and acceptor quantitation.
- Mandatory counter-screens: mass balance/recovery, precipitation, carryover and aqueous/kinetic solubility in the assay buffer context.
- Operators remain blinded to prediction group until outcomes, exclusions and analysis code are locked.
- At least three independent assay runs for the publication panel and blind remeasurement/resynthesis of 10–20%.

Wet-lab execution, vendor contact and spending are not authorized at this gate.

## 8. Required outputs

- immutable fold manifests and hashes;
- leakage-unit tests and independent verification;
- one row-level OOF prediction table per model/seed/split;
- complete run configuration, environment and elapsed-time log;
- pooled, source-macro, block and failure-mode tables;
- calibration/risk-coverage analysis;
- negative results and failed runs;
- a prospective model card and candidate-selection manifest before H3 begins.

## 9. Milestones and compute

| Milestone | Output | Estimated local resource |
|---|---|---:|
| R0 | Freeze v2 documents, folds, hashes and leakage tests | CPU, 1–2 days |
| R1 | Mean/classical ladder and D0 split ladder | <20 CPU-h; 25–50 GPU-h |
| R2 | D1/D2/D3 joint-block LOBO, seeds 0–4 | 90–180 GPU-h |
| R3 | Cross-fitted intervals, bootstrap and independent audit | <20 CPU-h; 20–40 GPU-h |
| R4 | Freeze prospective model card and chemical-universe protocol | CPU/planning only |
| R5 | Separately approved pilot and later publication panel | quote and assay budget required |

The estimated confirmatory computation is roughly 135–270 local GPU-hours, performed in checkpointed stages on the existing RTX 4060 8 GB. No rented compute is requested. Exact 17-fold inner LOBO was considered because it would maximize inner granularity, but it would require 306 inner fits per configuration before seeds and is not proportionate to the local hardware. The deterministic four-basket inner design preserves complete source/analogue blocking while making the closed experiment executable. If runtime exceeds the upper estimate, the user is shown actual throughput before any scope reduction; scheduled models/seeds are not silently dropped.

## 10. Stop and deviation rules

- Stop if any fold-manifest or outcome-leakage test fails.
- Stop if a code defect changes group membership or held-out labels after training begins; repair and restart affected runs with a dated deviation.
- Do not remove the 4,010-row block, the tiny blocks or difficult topologies because of poor performance.
- Do not change the 0.80 analogue threshold for the primary analysis. Thresholds 0.70 and 0.90 are descriptive sensitivity analyses only and do not regenerate confirmatory folds.
- Any change after freezing is entered in `DECISIONS.md` and the v2 deviation log before affected results are interpreted.
