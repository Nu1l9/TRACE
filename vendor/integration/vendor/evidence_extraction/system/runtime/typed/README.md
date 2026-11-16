# Typed Extraction Runtime (`extraction_runtime: "typed"`)

This module implements TRACE's stage-`A2` layout-aware typed extraction runtime:

```text
clean source blocks -> locate target regions -> typed Pydantic transcription -> deterministic source checks
```

Target source-data families are validated through family-specific Pydantic schemas, with explicit canonical aliases where applicable:

```text
two_group_event_counts          -> binary_arm_counts
binary_exposure_outcome_counts  -> binary_arm_counts
adjusted_time_to_event_effect   -> adjusted_ratio_effect
```

Additional supported families—including genotype/allele counts, diagnostic `2 × 2` contingency tables, single-arm event proportions, person-time counts and continuous group or change-score summaries—preserve their native typed schemas alongside exact source block identifiers and verbatim quotes. Invoke via the root launcher (`python trace_cli.py run` with `options.extraction_runtime = "typed"`).
