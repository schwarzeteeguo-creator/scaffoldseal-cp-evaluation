# Reference metadata audit

Date: 2026-08-09

## Scope and method

All 22 DOI-bearing entries in `references.bib` were resolved against CrossRef registry metadata. The audit compared DOI, article title, journal, publication year, volume, issue, and pages or article number when those fields were available in both records. Raw CrossRef-derived BibTeX records are preserved in `paper/reference_audit/`.

## Outcome

- Verified without substantive metadata conflict: 21 entries.
- Corrected: 1 entry.
- Not found: 0 entries.
- Manual verification required: 0 entries.

## Correction

The entry `kalinina2025datasail` used the title “DataSAIL: Data Splitting against Information Leakage.” CrossRef metadata for DOI `10.1038/s41467-025-58606-8` gives “Data splitting to avoid information leakage with DataSAIL.” The BibTeX title was corrected while preserving the existing citation key.

## Boundary

This audit verifies registry metadata, not whether each cited paper supports every sentence in the manuscript. Claim-level citation support remains an author responsibility.
