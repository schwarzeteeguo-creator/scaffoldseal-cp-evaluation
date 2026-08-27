# Results traceability register

Status: frozen reporting input, 2026-08-09.

## Primary claim

The primary evidence is H1: the same DMPNN evaluated by molecule-random splitting has source-macro MAE `0.46721829369813345`, whereas the leakage-resistant joint source/analogue-block evaluation has MAE `1.00455900469197`. The prespecified gap is `0.5373407109938364`, with a 10,000-replicate block/seed bootstrap 95% CI `[0.32436739522417257, 0.7530427419165402]`. Source: `scaffoldseal/artifacts/h1_bootstrap_v1/bootstrap_summary.json`.

Permitted claim: conventional random splitting substantially overestimates out-of-block generalization for this dataset and model under the frozen evaluation protocol.

Forbidden claim: universal performance inflation across all cyclic-peptide datasets, all architectures, or prospective experiments.

## Fixed-candidate hypothesis

H2 is not supported. D3 source-macro MAE is `0.9968783240`; it improves over D0 by only `0.0076806806` (required at least `0.03`) and is worse than the nested-selected classical procedure by `0.0840534990` (required improvement at least `0.05`). Source: `scaffoldseal/artifacts/d123_metrics_release_and_h2_verdict.md`.

The paired block/seed bootstrap 95% CI for classical-minus-D3 MAE is `[-0.23160502863057889, -0.01688417519446687]`, supporting degradation rather than improvement. Source: `scaffoldseal/artifacts/h2_condition3_condition5_zero_training.json`.

The 90% empirical interval covers `0.7702102973168963` of frozen D3 OOF prediction slots, below the preregistered `[0.85, 0.95]` range. Source: `scaffoldseal/artifacts/d3_interval_coverage/coverage_summary.json`.

D3 source-macro Spearman is `0.1978296815142659`, below the best comparator value `0.2744423428668366` by `0.07661266135257072`, exceeding the allowed deficit `0.03`. Source: `scaffoldseal/artifacts/h2_condition3_condition5_zero_training.json`.

Permitted claim: the tested weighting-plus-descriptor modification did not improve leakage-resistant generalization and remained miscalibrated.

Forbidden claim: descriptor augmentation is intrinsically harmful in all models or datasets.

## Supporting and secondary evidence

- D1 source-macro MAE: `0.9049673130`.
- D2 source-macro MAE: `0.9708264821`.
- D3 source-macro MAE: `0.9968783240`.
- Nested-selected classical source-macro MAE: `0.9128248250270173`.
- D1/D2 are secondary ablations and cannot replace the preregistered D3 candidate after results were observed.

## Uncertainty reconstruction provenance

The interval analysis restored 72/72 accepted D3 inner checkpoints without fitting, reconstructed 117,215 fold-local validation residual slots, and verified unchanged model-state hashes before/after all 72 inference passes. Machine-readable sources:

- `scaffoldseal/artifacts/d3_checkpoint_only_reconstruction/reconstruction_summary.json`
- `scaffoldseal/artifacts/d3_interval_coverage/coverage_summary.json`

## Limitations that must appear in the manuscript

1. The 18 joint blocks are highly imbalanced; the largest contains 58.16% of records and the three largest contain 92.39%.
2. The work estimates generalization under the observed source/analogue geometry, not a balanced target population.
3. No prospective wet-lab validation was performed; the current paper makes an evaluation-method claim, not a prospective discovery claim.
4. H2 is a negative confirmatory result and must be reported prominently.
5. The empirical intervals are fold-local residual intervals, not an exact split-conformal guarantee.
6. Findings are based on one cyclic-peptide PAMPA compilation and a defined model/comparator ladder.

## Integrity status

Every headline number above maps to a preserved machine-readable or audited artifact. No new fitting, post-release tuning, wet-lab experiment, paid compute, or external data transfer was used to create this register.
