# Public protocol timeline and freeze record

This document is a repository-ready chronology of the ScaffoldSeal-CP retrospective study. It distinguishes an internal, hash-verifiable protocol freeze from an independently time-stamped public preregistration. Until a public repository release and archive DOI exist, the project must not describe the local files alone as independent public preregistration.

| Date (Asia/Shanghai) | Event | Outcome status | Frozen evidence |
|---|---|---|---|
| 2026-07-17 | Original four-way partition feasibility gate | Failed before model fitting; v1 final partition and vault permanently retired after a public-artifact leakage incident | `DECISIONS.md`, D-018 and D-019; immutable failed `PREREGISTRATION.md` retained |
| 2026-07-17 | Option-A recovery plan approved | Version-2 research map, CV governance, experiment plan and hypothesis document frozen internally before any v2 fit | `DECISIONS.md`, D-020 to D-022; hashes below; no independent public timestamp |
| 2026-07-17 to 2026-07-19 | Pre-fit split and execution controls reviewed | Training held until outcome isolation, preprocessing, stopping and prediction contracts passed | `DECISIONS.md`, D-023 to D-034; test and audit records retained |
| 2026-07-18 onward | Frozen classical and D0 ladder executed | Negative comparators retained; no grid expansion after results | `EXPERIMENT_LOG.md`; D-028 to D-030 and later D0 records |
| 2026-07-26 | H1 random-comparison plan and acceptance frozen | Five-fold molecule-grouped random D0 comparison authorized only after pre-fit review | H1 plan/acceptance artifacts and `PROJECT_STATE.md` |
| 2026-07-27 | H1 execution completed | H1 supported under the prespecified rule; all 56 stages accepted | H1 artifact manifests, OOF predictions and bootstrap outputs |
| 2026-07-29 | D123 executor pre-fit review completed | Scientific plan accepted but training remained paused until explicit GPU authorization | `DECISIONS.md`, D-051 |
| 2026-08-09 | D123 post-training sealed audit | All 551 scientific identities accepted before metric release | `DECISIONS.md`, D-052 |
| 2026-08-09 | Controlled D1–D3 metric release | H2 failed all five conditions; benchmark-directed rescue prohibited | `DECISIONS.md`, D-053 and D-054; controlled-release report |
| 2026-08-10 | Zero-training H1 robustness analyses | Influence, calibration, analogue-threshold geometry and small-cluster analyses labeled post-confirmatory | Reporting scripts, input hashes and machine-readable summaries |
| 2026-08-10 | Split-boundary completion plan frozen internally | Source-only and analogue-only arms from the original ladder were re-operationalized; matched-size random was newly added. All pending fits are post-confirmatory and cannot revise H1/H2 | `SPLIT_BOUNDARY_SENSITIVITY_PREREGISTRATION.md`; SI Table S10 |

## Version-2 protocol hashes at the manuscript freeze

These hashes identify the retained local files. They should be cross-checked after packaging and replaced or supplemented by the immutable release tag and archive record at deposition.

| File | SHA-256 |
|---|---|
| `PREREGISTRATION_V2.md` | `3917a120e17622faf15a5d21c6b5cf3379bca9caf1067fc5b1af96adadb4b12` |
| `RESEARCH_MAP_V2.md` | `5e1e3e2da3ae7c08cfcfbafa4171469984cad6bcbec6ae850609cc86ebbed089` |
| `CV_GOVERNANCE_V2.md` | `a2df7b68271f96e74531cb412784bf5d05a9af083bf5f5a9a3b16d495889b48b` |
| `EXPERIMENT_PLAN_V2.md` | `35019506bb47bb66a3b0ba46aa1316a06a4c5083f5e45a95553d4f58eb3fa9d4` |
| `SPLIT_BOUNDARY_SENSITIVITY_PREREGISTRATION.md` | `d9220cadd5b184a3cc504254e618cdaa4b1cebcdc67cadfaed7535538a8d9d82` |

## Public identifiers

| Resource | Status | DOI/URL field for manuscript |
|---|---|---|
| Source-code repository | Release candidate prepared locally; not yet public | `[GitHub release URL pending]` |
| Versioned archival snapshot | Deposit after author/license review | `[Zenodo DOI pending]` |
| Upstream CycPeptMPDB data | Source publication identified; stable table-acquisition route and redistribution terms require verification | Article DOI `https://doi.org/10.1021/acs.jcim.2c01573`; exact raw SHA-256 in `DATA_ACCESS.md` |
| Maintained baseline code | Pinned upstream repository/commit documented in manuscript | `[upstream repository URL and commit]` |

Submission should remain blocked until the repository URL resolves publicly, the archival DOI resolves to the exact tagged release, and the manuscript/SI identifiers are updated from the same release manifest.
