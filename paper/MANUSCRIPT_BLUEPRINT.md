# Manuscript blueprint

## Working title

**Random splits overstate cyclic-peptide permeability prediction: a source- and analogue-blocked evaluation study**

## Target and fallback

- Primary target: *Journal of Chemical Information and Modeling*.
- Fallback: *Journal of Cheminformatics*.

Final formatting and policy checks must use current official author guidance before submission.

## One-sentence contribution

A preregistered, leakage-resistant evaluation of cyclic-peptide PAMPA prediction shows that molecule-random cross-validation substantially overstates generalization, while a fixed weighting-plus-descriptor DMPNN modification fails to close the resulting gap and remains under-covered.

## Narrative structure

1. **Problem:** cyclic-peptide permeability datasets contain source and close-analogue dependencies that random splitting can distribute across train and test.
2. **Methodological response:** freeze joint source/analogue blocks, nested selection, five seeds, immutable execution ledgers, and prespecified decisions.
3. **Primary result:** random-split source-macro MAE `0.4672` versus joint-block `1.0046`; gap `0.5373`, 95% CI `[0.3244, 0.7530]`.
4. **Model-improvement test:** the fixed D3 modification fails every H2 condition and cannot be promoted post hoc.
5. **Implication:** evaluation design, not a new state-of-the-art model, is the defensible contribution.
6. **Limitations:** block imbalance, single compiled dataset, retrospective evidence, no prospective assay, empirical rather than exact conformal intervals.

## Planned main display items

- **Figure 1:** dataset and evaluation schematic—molecule-random versus joint source/analogue blocking.
- **Figure 2a:** H1 random-versus-joint performance with the block/seed bootstrap interval.
- **Figure 2b:** D0, classical, D1, D2, and D3 source-macro MAE; visibly label D3 as the preregistered candidate.
- **Figure 2c:** nominal versus observed D3 interval coverage at 50%, 80%, and 90%, with the prespecified 85%–95% acceptance interval for the 90% interval.
- **Table 1:** dataset, block, seed, split, and evaluation geometry.
- **Table 2:** complete H1/H2 decision table, including all five failed H2 conditions.

## Planned supplementary items

- Per-seed and per-source metrics.
- Per-block error and coverage tables.
- Frozen identity, ledger, checkpoint, and environment provenance.
- D3 failure-structure diagnostics.
- Zero-training interval reconstruction audit.

## Claim discipline

Use “substantially overestimated under the frozen joint-block protocol,” not “invalidates all prior permeability models.” Use “the tested D3 modification failed,” not “descriptors do not work.” Do not describe D1 as a successful replacement because it was not the fixed confirmatory candidate.

## Wet-lab position

Wet-lab work is not required for this evaluation-centered manuscript. A future prospective matched-protocol PAMPA study would be a separate H3 project requiring new preregistration, partner qualification, budget approval, procurement, and assay governance.
