# Terminology ledger

| Canonical term | First-use definition | Decision |
|---|---|---|
| cyclic-peptide permeability | passive permeability of cyclic peptides measured by PAMPA | Do not generalize to oral bioavailability or cellular uptake. |
| PAMPA | parallel artificial membrane permeability assay (PAMPA) | Primary endpoint only; keep separate from Caco-2, MDCK, and proxy assays. |
| log10 Papp | log10-transformed apparent permeability (Papp), with Papp expressed in cm s−1 | Use consistently for prediction units; do not imply that the logarithm is applied to a dimensional quantity without first fixing the unit. |
| molecule-grouped pseudorandom evaluation | five-fold pseudorandom cross-validation with all records for one `molecule_id` kept in one fold | Never imply row-level independence; use “molecule-grouped” at first mention. |
| joint source/analogue-block evaluation | leave-one-joint-block-out evaluation over connected source and analogue components | Canonical hard-shift protocol. |
| joint block | connected component of the bipartite source–analogue-component graph | Do not call it a scaffold. |
| analogue component | connected component induced by the frozen peptide-aware analogue relation | Distinct from Murcko scaffold. |
| source-macro MAE | mean of per-source mean absolute errors | Primary error metric; lower is better. |
| DMPNN | directed message-passing neural network (DMPNN) | Define once. |
| D0 | locked reproduced DMPNN baseline | Use D0 consistently. |
| D1 | D0 with source/component-balanced loss | Secondary ablation. |
| D2 | D0 with fixed chemistry/topology descriptors | Secondary ablation. |
| D3 | D0 with both balanced loss and fixed descriptors | Internally prespecified H2 candidate; never replace post hoc. |
| nested-selected classical procedure | ridge, Random Forest, or XGBoost selected within each outer fold | Treat as a procedure, not a single globally selected model. |
| cross-fitted empirical interval | fold-local interval based on inner-validation absolute residual quantiles | Not an exact split-conformal guarantee. |
| prospectively frozen internal protocol | versioned local protocol and hashes fixed before the corresponding model fits | Do not call this an independently timestamped public preregistration. |
| slot-pooled coverage | coverage across record-by-seed prediction slots | Each record appears under five algorithmic seeds; slots are not independent experimental units. |
| OOF | out-of-fold (OOF) | Define once; each record receives one prediction per model/seed. |
