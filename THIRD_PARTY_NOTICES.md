# Component Provenance & Third-Party Notices

[Home](README.md)

This repository packages the **TRACE** multi-agent meta-analysis framework while keeping third-party dependencies, primary clinical publications and external assessment instruments cleanly separated:

| Component / Resource | Location | Distribution & Upstream Terms |
| --- | --- | --- |
| Protocol orchestration, cross-stage handoffs, updates & dashboards | `trace_cli.py`, `run_pipeline.py`, `extensions/`, `scripts/` | TRACE core source |
| Screening & RoB routing (`miners`) | `vendor/integration/vendor/miners` | Embedded TRACE stage-`A1` module |
| Layout-grounded typed extraction & candidate review | `vendor/integration/vendor/evidence_extraction` | Embedded TRACE stage-`A2` module |
| Graph-constrained selection & deterministic synthesis | `vendor/integration/vendor/evidence_analysis` | Embedded TRACE stage-`A3` module |
| Open-corpus hybrid retrieval (`BM25` + `Dense` + RRF) | `literature_retrieval/` | Embedded TRACE retrieval adapter |
| Read-only provenance-linked results Q&A | `expert_qa/` | Embedded TRACE post-synthesis workspace |
| MinerU HTTP conversion client | `document_io/mineru.py` | Client wrapper only; MinerU service/library terms apply separately |
| Python & R runtime libraries (`metafor`, `jsonlite`, etc.) | Installed via `pip` / `Rscript` | Not bundled; governed by their respective upstream open-source licenses |
| Risk-of-bias (RoB) instrument texts & compact prompts | Catalog only (`resources/rob/catalog.json`) | Not redistributed; install authorized local copies via [resources/rob/README.md](resources/rob/README.md) |
| Primary clinical study PDFs & benchmark Gold annotations | External / [**EviSynth-Bench**](https://github.com/Nu1l9/EviSynth-Bench) | Primary PDFs obtained from publishers / PMC / Europe PMC; benchmark data governed by EviSynth-Bench (`MIT` / `CC BY 4.0`) |
