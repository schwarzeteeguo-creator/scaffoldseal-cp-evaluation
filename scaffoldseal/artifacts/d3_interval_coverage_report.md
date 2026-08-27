# D3 checkpoint-only empirical interval coverage

Date: 2026-08-09

## Verdict

H2 condition 4 fails. The preregistered 90% cross-fitted empirical interval achieved pooled coverage `0.7702102973`, below the required inclusive range `[0.85, 0.95]`. D3 is under-covered and therefore overconfident under the frozen joint-block evaluation.

This completes all five H2 conditions. Conditions 1, 2, 3, 4, and 5 all fail; the previously established H2 verdict remains falsified/not supported.

## Coverage and width

| Nominal coverage | Observed coverage | Calibration error | Mean full width (log10 Papp) |
|---:|---:|---:|---:|
| 50% | 0.4413923133 | -0.0586076867 | 1.0228164564 |
| 80% | 0.6715300943 | -0.1284699057 | 1.9868498413 |
| 90% | 0.7702102973 | -0.1297897027 | 2.6137662758 |

Coverage was evaluated over 34,475 frozen D3 OOF prediction slots (6,895 records × 5 seeds). Each outer fold used only its own reconstructed outer-training residual pool. Finite-sample empirical half-widths used the upper order statistic `k=min(n,ceil((n+1)*level))`.

## Zero-training provenance

- Frozen plan SHA-256: `6b73097ee83c000a48ec4dfc646ddbcad3a561d3276c33e3661d3bf8b9d8f5c5`.
- 72/72 accepted D3 inner checkpoints restored; 72/72 model-state hashes were unchanged before/after inference.
- 117,215 inner-validation residual slots reconstructed with exact fold-local outer-training coverage.
- Exactly one accepted `attempt_002` was selected (outer fold 7, basket 3); the other 71 used accepted `attempt_001`.
- Reconstruction summary SHA-256: `ACE820855C3870283125EB52CE52A6FC84C97BDE4236FDDE569B29C48442C7A7`.
- Coverage summary SHA-256: `00F78029B0ECC6B96DCC9E33CE363503F0BCA9940239997B0DED772166BF52CB`.

No fitting, optimizer step, checkpoint write, tuning, model selection, split change, threshold change, or wet-lab action occurred. A timed-out parent command left its checkpoint-only child alive; a later duplicate read-only inference pass completed all 72 checks but refused to overwrite the already completed output. No duplicate training or accepted scientific identity was created.

Machine-readable outputs are in `scaffoldseal/artifacts/d3_checkpoint_only_reconstruction/` and `scaffoldseal/artifacts/d3_interval_coverage/`.
