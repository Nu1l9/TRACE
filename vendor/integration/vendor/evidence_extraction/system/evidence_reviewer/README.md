# Read-Only Candidate Evidence Reviewer

The candidate-evidence review layer operates immediately after `A2` typed extraction:

```text
clean source blocks -> locate regions -> typed extract -> organize -> batched candidate review
```

In each batched LLM call over candidates from a single article, the reviewer evaluates PICO alignment, protocol extraction constraints and verbatim source quotations without modifying extracted numbers or computing pooled effects. Each candidate receives an explicit role:

- `main_analysis_candidate`
- `supplementary_or_sensitivity`
- `source_only`
- `uncertain`
- `none` (mapped to `discard_clear_mismatch` at handoff)

Candidates sharing an identical `source_quote` reference a single deduplicated source block in the prompt (default batch size `12`), preserving full table and footnote context while keeping structured responses compact (`prediction_id`, `disposition`, `evidence_role` and rationale). Both machine-readable JSON traces and `review_summary.md` summaries are persisted under `03_review/` via the root TRACE pipeline (`python trace_cli.py run`).
