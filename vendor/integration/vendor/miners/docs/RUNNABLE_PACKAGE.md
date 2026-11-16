# MINERS Stage-A1 Package Guide

`miners` implements stage `A1` of **TRACE**—supporting open-corpus hybrid retrieval (`BM25` + `Dense` + weighted RRF), two-step PICO screening (title/abstract → full text for `Unsure` records) and study-design risk-of-bias (RoB) routing across `RoB 2`, `ROBINS-I`, `ROBINS-E`, `QUADAS-3`, `QUIPS` and `PROBAST+AI`.

```text
Local literature corpus
  -> SQLite document store
  -> BM25 + dense retrieval + weighted RRF fusion
  -> top-k candidate pool
  -> PICO title/abstract screening
  -> full-text adjudication for Unsure records
  -> final Include set
  -> study-design RoB tool routing
  -> JSON / Markdown / CSV audit artifacts
```

In end-to-end TRACE workflows, invoke `python trace_cli.py run` or `python trace_cli.py review` from the repository root (see [`docs/RETRIEVAL.md`](../../../../../docs/RETRIEVAL.md)). For standalone `miners` CLI usage:

## 1. Install & Preflight

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[all]"
miners doctor --api-key-env OPENAI_API_KEY --env-file .env
miners doctor --check-dense
```

## 2. Build Hybrid Retrieval Indexes

```bash
miners prepare --corpus data/corpus.json --store cache/miners/documents.sqlite
miners build-bm25 --store cache/miners/documents.sqlite --index-dir cache/miners/bm25
miners build-dense --store cache/miners/documents.sqlite --index-dir cache/miners/dense --model BAAI/bge-small-en-v1.5
```

## 3. Run Retrieval, Screening & RoB Routing

```bash
miners review \
  --paper-id 13 \
  --mode hybrid \
  --per-query-k 1000 \
  --fusion-k 500 \
  --final-k 200 \
  --screen \
  --fulltext \
  --route-rob \
  --model gpt-5.6-luna \
  --api-key-env OPENAI_API_KEY \
  --base-url https://api.openai.com/v1 \
  --env-file .env \
  --output-dir runs/miners
```

Outputs are written under `runs/miners/hybrid/paper_<ID>/` (`results.json`, `report.md`, `manual_review.csv`, `metrics.json` and `risk_of_bias_recommendations.json`).
