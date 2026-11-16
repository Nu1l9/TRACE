# Stage-A3 Analysis Component (`evidence_analysis`)

This module implements stage `A3` of **TRACE**: graph-constrained candidate retrieval, two-stage protocol review (`Planner` + `Reviewer`), study-identity duplicate governance and deterministic `R/metafor` synthesis.

## Core Capabilities

- **Evidence multigraph & retrieval:** Typed study–article–evidence multigraph with semantic context ranking (`text-embedding-3-small`) and bounded full-graph expansion.
- **Input contract:** Strict four-file JSON handoff specified in [`INTERFACE_CONTRACT.md`](INTERFACE_CONTRACT.md).
- **Deterministic calculators:** Family-specific Python validation and bundled `R/metafor` scripts for primary/subgroup/sensitivity pooling, robustness diagnostics and small-study diagnostics.

## Standalone Execution

From the root repository, use `python trace_cli.py run` or `python trace_cli.py retry-analysis` (see the root [`README.md`](../../../../README.md)). To validate or run an isolated `A3` input directory directly:

```bash
python run_analysis.py --input-dir /path/to/analysis_input --output-dir /path/to/run --check
python run_analysis.py --input-dir /path/to/analysis_input --output-dir /path/to/run
```

## Outputs

- `analysis_result.json`: complete typed analysis results, selected records, diagnostics and trace.
- `analysis_summary.json`: concise summary of selected evidence and pooled estimates.
- `graph.json`, `runtime_facets.json`, `retrieval_expansion_audit.json`, `embedding_retrieval.json`: multigraph and retrieval provenance artefacts.
- `forest_plots/`, `reporting_bias/`: diagnostic figures when sample-size gates are met.
- `run_metadata.json`: resolved model, endpoint, statistical policy and call-count metadata.
