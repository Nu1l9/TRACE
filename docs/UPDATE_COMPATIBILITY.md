# Incremental & Cross-Release Updates

[Home](../README.md)

`trace_cli.py update` (and `python -m extensions update`) enables living evidence synthesis: starting from a completed or stopped baseline run, TRACE writes an updated run to a new directory while reusing verified upstream artefacts for unchanged compatible studies.

```bash
# Preview reusable vs. recomputed stages without API calls
python trace_cli.py update \
  --base-run /path/to/previous-run \
  --request /path/to/updated-request.json \
  --output /path/to/new-run \
  --plan-only
```

Remove `--plan-only` and supply `--env-file .env --workers 2` to execute the update. Structured document additions, revisions and removals can also be passed via `--delta /path/to/update.delta.json` instead of a full `--request`.

---

## Stage Reuse & Recomputation Rules

TRACE evaluates baseline artefact integrity and per-stage reuse eligibility independently:

1. **Baseline integrity check:** Completed artefacts in `--base-run` must match their recorded SHA-256 digests before any stage can be inherited. The baseline directory itself is never modified.
2. **Conditional upstream inheritance:** When a document's source content, protocol target definitions, model/provider configuration and stage implementation bindings match, completed screening decisions, RoB assessments, typed extractions and candidate-review statuses are copied into the new run directory with parent-lineage metadata. Modifying extraction parameters also invalidates downstream candidate-review reuse for affected records.
3. **Conservative recomputation on change:** Newly added documents, revised source files at the same path (detected against saved per-document bindings) or stages affected by protocol/component updates are scheduled for fresh execution.
4. **Fresh synthesis over the updated pool:** Removed documents are excluded from the handoff pool, and `A3` re-executes evidence selection, duplicate governance and `R/metafor` synthesis across the combined inherited + newly extracted candidate set, recording version-to-version comparisons of selected study counts, pooled estimates, confidence intervals and heterogeneity.

The generated update plan (`baseline_compatibility`, `rerun_reasons`, per-stage `compatible` flags and `reusable_articles`) is re-verified immediately before seeding each stage. Use `--resume` to continue an interrupted run in place with identical inputs, and `update` in a new directory whenever adding studies, revising protocols or upgrading components.
