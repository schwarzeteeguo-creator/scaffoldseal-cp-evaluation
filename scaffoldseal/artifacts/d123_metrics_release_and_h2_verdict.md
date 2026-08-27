# D123 controlled metrics release and preregistered H2 verdict

Date: 2026-08-09  
Frozen plan SHA-256: `6b73097ee83c000a48ec4dfc646ddbcad3a561d3276c33e3661d3bf8b9d8f5c5`

## Release boundary

The user authorized one controlled release after the sealed post-training audit passed. Access was recorded before opening in `D123_METRICS_ACCESS_LOG.tsv`. The three ledger-bound metric files were hash-verified before reading. The permanently retired v1 vault was not accessed, and no retrospective final test was created.

## Released results

| Model | Source-macro MAE | Row-micro MAE |
|---|---:|---:|
| D1 | 0.9049673130 | 0.6477564386 |
| D2 | 0.9708264821 | 0.7458933901 |
| D3 (fixed H2 candidate) | 0.9968783240 | 0.8596534960 |
| D0 | 1.0045590047 | 0.6430483419 |
| Nested-selected classical procedure | 0.9128248250 | 0.6476913194 |

All D1/D2/D3 payloads state `labels_loaded_after_prediction_sealing=true` and contain 34,475 prediction rows.

## H2 decision

`PREREGISTRATION_V2.md` requires every support condition to hold.

1. D3 improvement over D0: `0.0076806806`; required at least `0.03`. **FAIL**.
2. D3 improvement over the nested-selected classical comparator: `-0.0840534990`; required at least `0.05`. D3 is worse. **FAIL**.

Therefore **H2 is falsified / not supported**. The paired-bootstrap, coverage and Spearman conditions cannot rescue the all-conditions rule. D1 and D2 are secondary ablations and cannot replace the fixed candidate after release. D1 is descriptively the best D123 variant on the primary metric, but this is not a confirmatory D3 success.

## Evidence hashes

- D1: `cc42326c402b64d8ec4a15a53c66bfc322b9d0913ab3d1ef4cab7f1ffba2901`
- D2: `2542c9ba6cfeb6f4e348fbbb5060e0f935a7bb6da5727f195cdf6b496bfe0aa8`
- D3: `8067a1fbce9737c921819fbc37eb33d9daf6dc578e8526b014c801cbfabf89de`
- D0 summary: `8a958b47d18d105dc323e0dc8b97a294d37b61c4119016b289bbe658761b7261`
- Nested classical summary: `cd7c279344101dc24c88fc3d94b57db78182598547425af2075e953b6e3fe941`

## Next boundary

The revealed joint-block OOF benchmark must not become a tuning target. Next work is prespecified failure diagnosis and complete secondary reporting. Any new optimization claim requires a separately labeled exploratory preregistration and genuinely fresh untouched outcome evidence.
