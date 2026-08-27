# Prior-work evaluation comparison

**Search update:** 2026-08-10.  
**Scope:** primary reports most directly relevant to cyclic-peptide permeability splitting, plus DataSAIL as a general leakage-aware splitting framework.  
**Claim boundary:** this is a search-limited comparison, not proof that no unpublished or unindexed study used a similar design.

Primary-source checks support the distinctions encoded in `source_data/prior_work_evaluation_comparison.csv`:

- Geylan et al. held out each of four large data sources as an external test and separately investigated source-group and stereochemistry-agnostic canonical-group splits. Those are important precedents, but the provenance and chemical grouping rules were evaluated as separate scenarios rather than as connected source–analogue blocks.
- PeptideCLM evaluated six representation-derived clusters in a leave-one-cluster-out design with fivefold development inside retained clusters. The clusters were not source–analogue connected components.
- MultiCycPermea reported random, Murcko/Jaccard hierarchical-cluster OOD and permeability-cliff settings. It did not hold out complete sources jointly with transitive chemical components.
- The 13-method benchmark compared random and Murcko-scaffold splits and supplied the DMPNN architecture reproduced as D0 here. Its split units did not jointly close provenance and analogue paths.
- DataSAIL provides generic one- and two-dimensional similarity-aware splitting through clustering and constrained optimization. It supports the principle that the intended dependency must determine the split, but the source–analogue graph used here remains a study-specific representation.

The safe novelty statement is therefore narrow: within the primary literature located by the search date, no checked cyclic-peptide permeability benchmark applied complete-source and transitive-analogue connected-component exclusion throughout nested model development. The individual ideas of source holdout, chemical clustering, scaffold splitting, nested validation and similarity-aware splitting are not claimed as new.

