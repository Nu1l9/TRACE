# Statistical Method Interfaces

[Home](../README.md)

TRACE configures run-level statistical synthesis behavior in `A3` via `protocol.statistical_method_policy` (defined in [`schemas/statistical_method_policy.schema.json`](../schemas/statistical_method_policy.schema.json)):

| Policy value | Interface | Synthesis behavior |
| --- | --- | --- |
| `"fixed"` *(default)* | Prespecified statistical policy | Applies REML random-effects pooling with 95% normal-approximation intervals by default, while honoring explicitly declared fixed-effect or clustered routes in the `AnalysisProfile` and saving separate Hartung–Knapp sensitivity intervals for eligible comparative random-effects outcomes. *(Note: `"fixed"` refers to the fixed policy rather than a fixed-effect model.)* |
| `"model_selected"` | Source-aware method selector | Evaluates the clinical question, selected source records, effect summaries, study-dependence metadata and calculator capabilities via the configured analysis text model, records a structured rationale, and dispatches the chosen specification to the deterministic `R/metafor` calculator. |

If omitted, `protocol.statistical_method_policy` resolves to `"fixed"`. Merge the desired policy fragment (see [`config/test_protocol_parameters.json`](../config/test_protocol_parameters.json) and [`config/model_selected_protocol_parameters.json`](../config/model_selected_protocol_parameters.json)) into your request's `protocol` object:

```json
{
  "protocol": {
    "statistical_method_policy": "fixed"
  }
}
```

---

## Supported `model_selected` Configurations

When `"model_selected"` is enabled, the selector operates strictly after evidence selection—using the already chosen study set without altering evidence membership or trial-running alternative pooled estimates—and selects among supported `R/metafor` configurations:

- **Fixed-effect inverse-variance pooling** with 95% normal-approximation intervals.
- **Random-effects pooling** with DerSimonian–Laird (`DL`), restricted maximum likelihood (`REML`) or Paule–Mandel (`PM`) heterogeneity estimation, paired with 95% normal-approximation or Hartung–Knapp (`HK`) intervals.
- **Declared clustered routes**, which retain their dedicated clustered calculators.

In `model_selected` mode, the selected confidence-interval method serves directly as the primary interval. Invalid configurations undergo at most one schema/capability repair pass; unresolved configuration errors block the affected topic with an explicit status rather than silently falling back to `"fixed"`.

---

## Provenance & Run Binding

Every `A3` output directory records `statistical_method_policy.json`, `statistical_policy.json` and the resolved policy metadata in `run_metadata.json` (`model_selected` rationales are written under `method_selection/` and numerical validation checks under `calculation_audit/`).

Because the statistical policy is bound into the analysis input fingerprint, switching between `"fixed"` and `"model_selected"` requires a new output directory.
