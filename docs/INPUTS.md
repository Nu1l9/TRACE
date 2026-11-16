# Input Request Contract

[Home](../README.md)

A TRACE request JSON defines the clinical question, prespecified research protocol, analysis plan and candidate documents for a synthesis run.

## Top-Level Structure

| Field | Required | Description |
| --- | --- | --- |
| `schema_version` | Yes | Request contract identifier (e.g. `"meta_pipeline/1.0"`) |
| `case_id` | Yes | Stable case identifier used across outputs, dashboards and incremental updates |
| `topic` | Yes | Clinical research question |
| `protocol` | Yes | PICO eligibility criteria, allowed study designs, `statistical_method_policy` and structured `targets` with allowed data families |
| `analysis_profile` | Yes | Prespecified primary, secondary, subgroup and sensitivity analyses, accepted effect representations, follow-up windows, selection preferences and study-dependence rules |
| `documents` | For `run` | Candidate documents with `article_id`, `study_id` and `paths` (`pdf`, `markdown` and optional `content_list`). Omitted or empty in open-corpus `review` requests |
| `options` | Optional | Runtime options, including `extraction_runtime: "typed"` and `route_rob: true` |

Relative document paths resolve against the directory containing `request.json`. When multiple publications report on the same underlying cohort or clinical trial, assign them the same `study_id` so `A3` can govern duplicate reports according to the analysis plan.

## Examples & Companion Tasks

- **Minimal schema fixture:** [`vendor/integration/examples/request.json`](../vendor/integration/examples/request.json) illustrates the request structure on small local Markdown documents.
- **Runnable clinical example:** [Open-Access Quickstart](../QUICKSTART.md) generates a complete `request.json` and JATS-derived Markdown inputs from PubMed Central trials.
- **Drafting from partial fields:** Use `python trace_cli.py prepare` (see [CLI Guide](CLI.md)) to assemble and validate a `request.json` interactively.
- **Benchmark cases:** The 22 clinical benchmark requests and blind task exports are maintained in [**EviSynth-Bench**](https://github.com/Nu1l9/EviSynth-Bench).
