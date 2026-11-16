# System Architecture

[Home](../README.md)

TRACE separates protocol orchestration, stage-specific clinical reasoning and deterministic statistical calculation into modular layers connected by explicit JSON contracts and content-bound checkpoints (`Fig. 1a,b`).

## Module Mapping

| Architectural layer | Repository location | Responsibilities |
| --- | --- | --- |
| **Unified CLI & Protocol Orchestrator** | `trace_cli.py`, `run_pipeline.py`, `extensions/` | Normalizes requests (`protocol` + `AnalysisProfile`), enforces cross-stage handoff invariants, binds checkpoints to source bytes and contracts, manages `resume` / `update` lineage, and executes study-design RoB assessment |
| **Document I/O & Retrieval** | `document_io/`, `literature_retrieval/` | Converts primary PDFs via MinerU (`document_io`); builds local `BM25` + `Dense` (`BAAI/bge-small-en-v1.5`) indexes, performs weighted RRF fusion and hands frozen candidate sets to `A1` (`literature_retrieval`) |
| **`A1` Screening & RoB Agent** | `vendor/integration/vendor/miners`, `extensions/screening_bias.py` | Two-step PICO screening (title/abstract → full text, with human adjudication pause for unresolved studies) and deterministic study-design routing across 6 RoB tools (`RoB 2`, `ROBINS-I`, `ROBINS-E`, `QUADAS-3`, `QUIPS`, `PROBAST+AI`) |
| **`A2` Extraction & Verification Agent** | `vendor/integration/vendor/evidence_extraction` | Cleans and indexes document layout blocks; locates target regions; performs typed Pydantic transcription across 9 data/estimand families; runs deterministic source/numerical checks and read-only batched candidate review (`main_analysis_candidate`, `supplementary_or_sensitivity`, `source_only`, `uncertain`, `discard_clear_mismatch`) |
| **`A3` Selection & Synthesis Agent** | `vendor/integration/vendor/evidence_analysis` (`analysis_core`, `analysis_engine`) | Constructs typed study–article–evidence multigraphs; runs two-stage `Planner` + `Reviewer` scheduling with semantic ranking (`text-embedding-3-small`) and up to one full-graph expansion; governs `study_id` duplicates; executes `R/metafor` pooling, robustness diagnostics (leave-one-out, FE vs. RE, Baujat) and small-study diagnostics (Egger, Begg, trim-and-fill, contour funnel plots) |
| **Post-Synthesis Presentation & Q&A** | `scripts/render_run_dashboard.py`, `expert_qa/` | Renders interactive HTML dashboards; serves read-only tool-calling Q&A scoped to a single frozen case/run/analysis snapshot |

## Cross-Stage Handoff & Provenance Invariants

1. **Decoupled clinical vs. statistical gates:** `A1` evaluates clinical PICO eligibility independently of whether an article reports immediately poolable summary statistics; `A2` transcribes all protocol-relevant source records without prematurely discarding alternative adjustment models or follow-up windows; `A3` selects target-concordant records per `AnalysisProfile` and enforces numerical poolability and minimum-study gates.
2. **Selective triage over silent dropping:** Clearly mismatched extractions (`discard_clear_mismatch`) are excluded at the `A2` → `A3` boundary, whereas uncertain or boundary records retain explicit review flags (`?` / `human_review`) and surface in the quarantined pending-review pool on the dashboard.
3. **Read-only post-run layer:** `expert_qa` attaches after pipeline termination and accesses saved screening, RoB, extraction, review and `R/metafor` outputs exclusively through read-only tools with citation and verbatim-quote verification.
4. **Benchmark separation:** All benchmark task inputs, held-out Gold annotations and `T1`–`T4` evaluators live independently in [**EviSynth-Bench**](https://github.com/Nu1l9/EviSynth-Bench).
