# Curation and Source-Metadata Audit

This zero-training audit reconciles the downloaded CycPeptMPDB PAMPA table with the frozen analytical population and documents what the source table can and cannot support.

The raw file contained 7,298 rows. The frozen rule excluded 372 rows with a populated detection-limit field, leaving 6,926 uncensored usable rows. Median collapse of 31 compatible multi-row source-structure groups produced 6,895 analytical records, 6,862 unique molecules, and 41 retained sources. No invalid structure/sequence rows or incompatible groups occurred.

The 372 annotations were heterogeneous: 259 database-assigned or reported floors, 40 explicit upper limits, 19 no-reportable-value notes, 9 non-detection or below-LOD notes, 7 solubility/not-tested cases, and 38 other notes. Only two excluded rows were linked to a retained group with an uncensored replicate. Numeric database entries among excluded rows were likewise heterogeneous, so they are not treated as interchangeable quantitative censoring bounds.

The raw table does not provide harmonized structured fields for PAMPA pH, membrane composition, incubation time, temperature, or donor/acceptor conditions. Source identity is therefore a combined provenance and protocol proxy. Blocking by source protects the evaluation boundary against dependencies associated with that proxy, but it cannot attribute a model-error difference to a particular assay factor.

Release-safe manifests omit structures and endpoint values pending confirmation of upstream redistribution permissions. The raw-file SHA-256 and deterministic reconstruction outputs are recorded in the accompanying JSON and checksum files.
