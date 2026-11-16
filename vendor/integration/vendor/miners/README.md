# MINERS

MINERS is an auditable evidence-synthesis system. Its main workflow starts with
Hybrid literature retrieval, then carries candidates through PICO screening,
full-text adjudication, and risk-of-bias tool routing.

```text
Literature corpus
  -> BM25 + Dense + RRF Hybrid retrieval
  -> PICO title/abstract screening
  -> full-text adjudication of Unsure records
  -> final Include set
  -> risk-of-bias tool recommendation
  -> JSON, Markdown, and CSV audit artifacts
```

The package supports both open-corpus Hybrid retrieval (`review`) and supplied-candidate PICO screening with full-text escalation and study-design RoB routing (`case-review`).

## Runtime Structure

```text
miners/
  workflow.py              # Main Hybrid -> PICO -> full text -> RoB orchestration
  dataset.py               # Corpus metadata store and full-text access
  indexes.py                # BM25 and dense embedding indexes
  retrieval.py              # Multi-query planning, RRF fusion, optional reranking
  screening.py              # PICO screening and Unsure full-text resolution
  evidence.py               # Full-text evidence retrieval for screening
  paper_node_adapter.py     # OpenAI-compatible LLM client and Pydantic parsing
  rob_routing.py            # RoB 2 / ROBINS-I / ROBINS-E / QUADAS-3 / QUIPS / PROBAST+AI routing
  case_screening_rob.py     # Supplied-candidate PICO -> full-text -> RoB workflow
  reporting.py              # JSON, Markdown, CSV audit artifacts
  evaluation.py             # Retrieval and screening evaluation
  cli.py                    # `miners` command
```

The package has no runtime import of the legacy experiment code. The published
wheel contains this `miners/` implementation only.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[all]"
```

Create `.env` from `.env.example` before using an LLM-backed stage.

## Run

```powershell
miners doctor --check-dense --env-file .env --api-key-env OPENAI_API_KEY

miners review `
  --paper-id 13 `
  --mode hybrid `
  --screen `
  --fulltext `
  --route-rob `
  --model gpt-5.6-luna `
  --api-key-env OPENAI_API_KEY `
  --base-url https://api.openai.com/v1 `
  --env-file .env
```

The full command reference, corpus preparation steps, and case workflow are in
[docs/RUNNABLE_PACKAGE.md](docs/RUNNABLE_PACKAGE.md).
