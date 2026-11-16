# Analysis Component Design Notes

- Component: `evidence_analysis` (stage `A3` of TRACE)
- Input contract: normalized Evidence envelope described in [`INTERFACE_CONTRACT.md`](INTERFACE_CONTRACT.md)

## Architectural Decisions

1. Analysis topics are expanded exclusively from `analysis_profile_structured.json`: prespecified primary outcomes, secondary outcomes, subgroup dimensions and sensitivity analyses.
2. Runtime source facets are derived from structured Evidence fields, filtering out only source-quoted contradictions while retaining ambiguous records for review.
3. A `Planner` schedules declared topics for local review based on typed multigraph inventory counts.
4. A topic `Reviewer` selects eligible candidate IDs within its local graph context and may trigger up to one expansion pass over the complete source graph.
5. Study-identity dependence governance, data-family validation and statistical pooling are executed by deterministic `R/metafor` calculators after evidence selection.
6. Leave-one-out, fixed-versus-random, Baujat, small-study diagnostics and forest plots are generated post-pooling without altering primary evidence membership.
