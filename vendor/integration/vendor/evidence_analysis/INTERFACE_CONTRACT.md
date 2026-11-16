# Analysis Input Contract

## Boundary

The analysis component receives only the output of the prior extraction stage
and a task protocol. It does not receive primary PDFs, review forest plots,
Gold labels, inclusion labels, `quantitative_usability`, precomputed pooling
methods, or a list of Evidence that should be selected.

The upstream system writes these four JSON files into one `input-dir`:

```text
input-dir/
  extraction_protocol.json
  analysis_task.json
  analysis_profile_structured.json
  analysis_evidence_candidates.json
```

Benchmark-style paths (`inputs/...` and `tasks/analysis_main_outcomes.json`)
are accepted for compatibility, but integration should use the flat layout.

## 1. `extraction_protocol.json`

Defines the PICO and source-level scope. It must include `protocol_id`,
`case_id`, and a `pico` object. Use `intervention_or_exposure` plus
`comparison`, `population`, and `outcome` descriptions. The protocol may hold
one or more `targets` with stable `target_id` values.

## 2. `analysis_task.json`

Defines the task identity only. Required: `task_id`. Recommended fields:
`case_id`, `task_type: "analysis"`, and a human-readable task title.

## 3. `analysis_profile_structured.json`

Defines what analyses are allowed, without selecting any Evidence. Required:

- `profile_version`: begins with `2`.
- `primary_analysis_spec`: one declared primary topic.
- `primary_analysis_spec.analysis_id`, `target_id`, `outcome_group`,
  `eligible_data_families`, `accepted_effect_types`.

Optional `secondary_outcomes` create declared additional topics.
Optional `candidate_subgroup_dimensions` and
`candidate_sensitivity_dimensions` describe allowed explorations. Each level
is expanded into an executable topic by the runtime plan resolver. They carry
terms and rules, never candidate IDs.

Method-neutral profiles are supported: the component maps data family and
effect representation to a calculator internally, then the method selector
chooses a defensible model, heterogeneity estimator and interval configuration
after Evidence selection. Those internal routes are not sent to the reviewer.

## 4. `analysis_evidence_candidates.json`

```json
{
  "candidates": [
    {
      "candidate_id": "opaque_evidence_id",
      "article_id": "article_id",
      "study_id": "independent_study_id",
      "target_id": "protocol_target_id",
      "exposure": "source-supported exposure or intervention",
      "comparison": "source-supported comparator",
      "outcome": {"label": "source outcome", "code": "optional_source_code"},
      "population_or_subgroup": "overall population or named subgroup",
      "timepoint": "source follow-up",
      "data_family": "binary_arm_counts",
      "estimand": "risk_ratio",
      "raw_data": {},
      "effect": {
        "effect_type": "RR",
        "value": 1.0,
        "ci_lower": 0.8,
        "ci_upper": 1.2,
        "direction": "optional_source_defined_direction"
      },
      "source_location": {"document_id": "...", "page": 4, "table": "Table 2"}
    }
  ]
}
```

Required top-level candidate fields are `candidate_id`, `article_id`,
`study_id`, `target_id`, `outcome`, `data_family`, and `raw_data`.
`candidate_id` is opaque and unique. `study_id` is the unit used by dependence
handling and must be consistent across all reports of one independent study.

The stable Evidence envelope is exactly:

```text
candidate_id, article_id, study_id, target_id,
exposure, comparison, outcome, population_or_subgroup, timepoint,
data_family, estimand, raw_data, effect, source_location
```

Fields may be missing or empty when extraction cannot support them. Do not add
an upstream label saying that a record is poolable, selected, Gold-matched, or
best. `raw_data` contains only source-extracted numeric material needed by the
relevant data-family calculator, such as arm counts, means/SDs, person-time,
diagnostic four-cell counts or reported effects with confidence intervals.

## Output hand-off to the Brain

The Brain receives `analysis_summary.json` plus a path to
`analysis_result.json`. It should present primary/secondary/subgroup outcomes
by `analysis_id`, selected candidate IDs, pooled results, warnings and status.
It should retain the detailed result, graph and diagnostics as audit evidence;
it must not reconstruct or overwrite a pooled estimate itself.
