# RESEARCH_MAP_V2.md

> Option-A recovery map, version 1.0, frozen on 2026-07-17 after explicit Recovery Plan Gate approval. `RESEARCH_MAP.md` remains the historical Gate-0 record. Post-freeze changes require a dated deviation.

## Working title

**ScaffoldSeal-CP v2: how cyclic-peptide permeability benchmarks fail under source and analogue-series shift, with prospective single-scaffold confirmation**

## Research question

How much of the apparent accuracy of cyclic-peptide PAMPA models survives when every publication-linked and close-analogue-linked block is held out in turn? Can a small, chemistry-aware model provide useful predictions and calibrated uncertainty under that shift, and can the selected model enrich permeability within a newly purchased single-scaffold medicinal-chemistry series measured under one matched assay?

## Why the scope changed

The frozen v1 graph contains 6,895 curated source-structure groups, 6,862 unique molecules, 305 analogue components and only 18 indivisible source-component blocks. The v1 four-way protocol required 950 independent components and therefore failed before model training. The initial final partition was also permanently retired after a public outcome-derived field incident. Neither problem is repaired by lowering thresholds after seeing the data.

Version 2 makes the verified dependence structure the subject of the paper rather than pretending it is absent.

## Confirmatory claims

- **H1 — evaluation optimism:** molecule-random and ordinary scaffold evaluation materially understate error relative to joint source-plus-analogue leave-one-block-out prediction.
- **H2 — usable modeling under shift:** the prespecified group-balanced, chemistry-augmented DMPNN improves joint-block out-of-fold performance over the locked DMPNN and a nested-selected classical comparator while maintaining useful empirical interval coverage.
- **H3 — prospective enrichment:** after a separate prospective freeze, the historically selected model enriches PAMPA permeability within one new, synthetically coherent cyclic-peptide scaffold relative to matched random controls without a material solubility/recovery penalty.

H1 is the paper's primary computational claim. H2 may fail without invalidating H1. H3 requires a separate resource and prospective-protocol approval and cannot be claimed from the retrospective data.

## Defensible novelty

The novelty is the combination of:

1. a provenance-preserving cyclic-peptide PAMPA curation with stereochemistry and topology retained;
2. a peptide-aware analogue graph joined to publication/source provenance;
3. complete out-of-block predictions for all 18 verified joint blocks, with the same model also evaluated on optimistic comparison splits;
4. explicit quantification of how error changes with source overlap, analogue similarity and block size;
5. cross-fitted empirical uncertainty under the same blocked design; and
6. a separately blinded, single-scaffold medicinal-chemistry confirmation under one matched PAMPA and solubility/recovery protocol.

The project does **not** claim a novel neural architecture, universal effects of N-methylation or stereochemical inversion, oral exposure, cellular uptake or exact finite-sample conformal guarantees.

The focused collision check through 2026-07-17 found no study combining all four elements above. The closest prospective collision is PEGASUS, which already couples multimodal models, a large proxy-permeability assay and designed cell-permeable cyclic peptides. Therefore the safe novelty wording is "to our knowledge, the first joint source-and-analogue connected-component cyclic-peptide permeability benchmark paired with a preregistered prospective single-scaffold PAMPA test," not "the first AI-designed permeable cyclic peptides." See `RECOVERY_LITERATURE_CHECK.md`.

## Data and feasibility

- Primary retrospective endpoint: uncensored `log10(Papp in cm/s)` from PAMPA only.
- Curated analysis population: 6,895 source-structure groups, 6,862 unique molecules and 41 sources.
- Dependence structure: 141,425 frozen analogue edges, 305 components and 18 joint source-component blocks.
- Block sizes are highly imbalanced: 4,010; 1,518; 842; 249; 105; 36; 18; 18; 17; 16; 16; 11; 10; 8; 7; 7; 4; and 3 rows.
- All 18 blocks are retained. Tiny blocks are reported transparently and are not used alone for stable correlation estimates.

## Evaluation architecture

Each joint block is held out once as the outer test block. All fitting, descriptor scaling, early-stopping decisions, hyperparameter selection and residual calibration use only the other 17 blocks. This produces exactly one joint-block out-of-fold prediction per curated record for each prespecified model and seed.

For each outer fold, the remaining blocks are assigned to four deterministic inner baskets by a label-blind greedy row-balancing rule. Inner baskets are used for classical hyperparameter selection, DMPNN stopping-epoch determination and cross-fitted residual construction. Joint blocks are never split.

The primary comparison splits are molecule-random five-fold, Murcko five-fold, source-only leave-one-source/group-fold, analogue-component five-fold and joint-block leave-one-block-out. Comparison assignments use structure/provenance only and are frozen before labels enter modeling code.

## Model scope

- training-mean predictor;
- ridge, Random Forest and XGBoost on ECFP4 plus a fixed medicinal-chemistry descriptor panel;
- locked reproduced DMPNN;
- DMPNN + group-balanced optimization (M1);
- DMPNN + fixed chemistry descriptors (M2);
- DMPNN + M1 + M2;
- cross-fitted empirical prediction intervals for the prespecified point models.

No transformer, generative model, full conformer regeneration or open-ended architecture search is confirmatory.

## Prospective chemistry concept

The prospective study uses one head-to-tail cyclic-peptide scaffold with a mature custom-synthesis route and a frozen, enumerated analogue universe. Candidate groups are predicted-good, predicted-poor, high-uncertainty and matched-random. Matching is performed within design strata defined by edit position, ring size, molecular-weight band, formal charge, cLogP band, N-methyl/noncanonical count and supplier-rated difficulty.

The staged target is 12–16 QC-passing pilot compounds followed, only after feasibility approval, by approximately 64 QC-passing compounds. Every delivered compound requires incoming identity/purity and concentration checks. The primary assay is one matched PAMPA protocol with donor/acceptor quantitation; mass balance, recovery, precipitation and aqueous/kinetic solubility are mandatory counter-screens.

No vendor contact, quote request, purchase, synthesis or assay execution is authorized by this v2 computational recovery plan.

## Target venue

- Primary: *Journal of Chemical Information and Modeling*.
- Stretch: *Journal of Medicinal Chemistry* only if the prospective series provides a strong, chemically interpretable structure-property story and complete characterization.
- Fallbacks: *Journal of Cheminformatics*, *Digital Discovery* or *Molecular Pharmaceutics* depending on whether the final emphasis is benchmarking or ADME experimentation.

## Falsifiability

- H1 fails if the prespecified joint-block versus molecule-random MAE gap is below 0.10 log units or its blocked uncertainty interval includes zero.
- H2 fails if the fixed M1+M2 candidate does not meet every prespecified improvement and coverage condition.
- H3 fails if prospective enrichment, solubility/recovery, QC or attrition conditions are not met.
- No failed claim may be rescued by changing the analogue threshold, excluding difficult blocks, selecting a favorable seed or relabeling an exploratory analysis as confirmatory.
