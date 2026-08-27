# ScaffoldSeal-CP H1/H2 scientific synthesis

Date: 2026-08-09  
Phase: Gate-3 evidence synthesis after a supported primary H1 and falsified H2

## Executive conclusion

The project establishes a strong evaluation finding but not a successful new predictive model. The preregistered primary H1 is supported: molecule-random evaluation materially understates error under joint source-plus-analogue shift. The confirmatory H2 is falsified: fixed candidate D3 does not achieve the required gains over D0 or the nested-selected classical procedure.

## What is supported

### 1. Random splits are substantially optimistic

- Molecule-random D0 source-macro MAE: `0.4672182937`.
- Joint-block LOBO D0 source-macro MAE: `1.0045590047`.
- Gap: `0.5373407110 log10(Papp)`.
- Preregistered 10,000-replicate 95% interval: `[0.3243673952, 0.7530427419]`.

The gap exceeds the fixed `0.10` threshold, and the entire interval is above zero. H1 is formally supported. This is the project's strongest confirmatory result and was designated primary before outcome review.

### 2. Joint source-plus-analogue transfer is genuinely difficult

Under the hard joint-block evaluation, the outer-training mean has source-macro MAE `0.7768779452`, better than the nested-selected classical procedure (`0.9128248250`), D1 (`0.9049673130`), D2 (`0.9708264821`), D3 (`0.9968783240`) and D0 (`1.0045590047`). Greater model complexity did not reliably overcome the shift.

### 3. D1 contains a real descriptive signal, but not a confirmatory model win

D1 improves over D0 by `0.0995916917` source-macro MAE and is slightly better than the nested-selected classical procedure by `0.0078575120`. Its gain is heterogeneous and concentrated in particular sources. D1 was a secondary ablation, so it cannot replace fixed candidate D3 after release or be presented as H2 support.

## What is not supported

- D3 is not a usable confirmatory predictor under the frozen H2 rule.
- The descriptor-side modification D2 does not improve the D0/D1 representation under this evaluation.
- Combining D1 and D2 does not produce additive or synergistic improvement.
- No evidence supports claiming superiority over the simple outer-training mean.
- No retrospective replacement candidate or revised threshold is permitted.

## Mechanistic interpretation

The failure structure is consistent with regime-dependent negative interaction from the descriptor-side modification. D2 and D3 are worse than D1 at all five seeds. D3 degradation is concentrated in particular sources and two large blocks, while the largest block does not collapse. This makes random seed noise or universal undertraining less plausible, but it does not prove the exact causal mechanism.

## Honest research contribution

The defensible contribution is an **evaluation and generalization result**:

1. a leakage-resistant joint source-plus-analogue blocking framework;
2. quantitative evidence that molecule-random evaluation is severely optimistic for this cyclic-peptide permeability setting;
3. a complete baseline ladder showing that advanced representations and descriptor fusion do not automatically improve cross-source/cross-analogue transfer;
4. a reproducible negative result that localizes where descriptor fusion fails.

This is not a claim of a new state-of-the-art predictor. The supported story is about evaluation realism, generalization limits and the danger of trusting random-split performance.

## Recommended disposition

Proceed toward an H1-centered, methods/evaluation manuscript only after confirming that the target venue accepts rigorous benchmark/generalization studies with a negative modeling result. H2 should be reported prominently as falsified, not hidden. No further confirmatory tuning should use the revealed v2 OOF benchmark. A future model-improvement project should use a new exploratory preregistration and fresh untouched prospective or otherwise independent outcomes.

This synthesis reaches a Gate-3 decision point: the evidence supports an honest contribution reframe already anchored in the preregistered primary H1, but manuscript drafting remains a separate user/accountability gate.
