# TRACE Unified CLI Guide (`trace_cli.py`)

[Home](../README.md)

`trace_cli.py` provides the unified command-line entry to TRACE's protocol-guided orchestrator (`Fig. 1a`). It coordinates protocol drafting, input validation, open-corpus retrieval, `A1`–`A3` stage execution, checkpoint recovery, incremental updates and the read-only `expert_qa` assistant through one consistent interface.

---

## 1. Environment Setup

Activate your Python virtual environment from the repository root and ensure `R` (`metafor`, `jsonlite`), `config/provider.json`, `.env` and any required RoB guidance files are configured as described in [Installation & Configuration](../README.md#installation--configuration).

```bash
python trace_cli.py --help
python trace_cli.py run --help
python trace_cli.py qa --help
```

`trace_cli.py` and `run_pipeline.py` automatically append a timestamp to newly created `run`, `review` and executed `update` directories and print the exact path to stdout. `python -m extensions` writes directly to the supplied `--output` path without adding a timestamp. Use the exact directory path created at launch when resuming or inspecting a run.

---

## 2. Draft and Export a Protocol (`prepare`)

If you already have a complete `request.json`, proceed directly to `check`. Otherwise, supply partial known fields in a JSON file using the request schema's top-level keys:

```bash
python trace_cli.py prepare \
  --topic "Your clinical research question" \
  --known /path/to/known_fields.json \
  --output work/protocol_draft.json
```

Without `--live`, `prepare` runs deterministically with zero API calls, listing missing protocol fields and open items to refine. Add `--live --env-file .env` to draft missing protocol fields with the configured text model (or `--completion /path/to/proposal.json` to apply a saved proposal); user-supplied fields in `--known` always take precedence.

Once all required fields pass contract validation, export the runnable request:

```bash
python trace_cli.py prepare \
  --topic "Your clinical research question" \
  --known /path/to/known_fields.json \
  --output work/protocol_draft.json \
  --request-output work/request.json
```

Relative document paths are resolved against the `--known` file location upon export. See [Input Contract](INPUTS.md) and [Statistical Method Interfaces](statistical_method_policy.md).

---

## 3. Validate and Execute (`check` / `run`)

```bash
python trace_cli.py check --request work/request.json --env-file .env
python trace_cli.py run --request work/request.json --env-file .env --output runs/my_review --workers 2
```

- `check` verifies request schemas, local document availability and runtime components without calling remote APIs. (`python -m extensions provider-check --env-file .env --request work/request.json` can be used separately to test live endpoint connectivity.)
- `run` executes document preparation, `A1` screening & RoB routing, `A2` layout-grounded typed extraction & candidate review, and `A3` graph-guided selection & `R/metafor` synthesis, rendering `dashboard.html` as stages complete.

| Option | Description |
| --- | --- |
| `--request` | Path to request JSON; relative document paths resolve against this file |
| `--output` | Output directory prefix for new runs, or exact timestamped directory for `--resume` |
| `--env-file` | Path to local `.env` file containing provider credentials |
| `--workers` | Stage worker concurrency (default `4`; `2` is recommended for standard API rate limits) |
| `--extensions` | Optional extension-settings JSON (keep identical across `--resume`) |
| `--retry-deferred` | Retry articles previously deferred after transient upstream timeouts |

---

## 4. Inspect Status, Resume & Run Open-Corpus Retrieval (`status` / `resume` / `review`)

```bash
python trace_cli.py status --run runs/EXACT_TIMESTAMPED_DIRECTORY
python trace_cli.py resume --request work/request.json --env-file .env \
  --output runs/EXACT_TIMESTAMPED_DIRECTORY --workers 2
```

- `status` reads `pipeline_manifest.json` and reports persisted stage states and generated dashboard paths without model calls or file writes.
- `resume` (alias for `run --resume`) reuses verified checkpoints when source fingerprints, protocol contracts and settings match.

For open-corpus workflows starting from a bibliographic SQLite store and `BM25` / `Dense` indexes, use `review`:

```bash
# Freeze retrieved candidates first without LLM screening calls
python trace_cli.py review --request /path/to/literature_request.json \
  --store cache/literature.sqlite --bm25-index cache/bm25 \
  --mode bm25 --output runs/corpus_review --prepare-only
```

To execute screening, extraction and synthesis on the frozen candidate pool, rerun `trace_cli.py review` with the printed timestamped directory, `--resume` and `--env-file .env`. See [Literature Retrieval](RETRIEVAL.md).

---

## 5. Incremental Updates & Stage Recovery (`update` / `retry-analysis` / `rob-retry`)

```bash
# Preview reusable vs. recomputed stages without API calls
python trace_cli.py update --base-run runs/OLD_DIRECTORY \
  --request /path/to/updated_request.json \
  --output runs/updated_review --plan-only

# Execute the incremental update in a new timestamped directory
python trace_cli.py update --base-run runs/OLD_DIRECTORY \
  --request /path/to/updated_request.json \
  --output runs/updated_review --env-file .env --workers 2
```

Supply `--delta /path/to/update.delta.json` instead of `--request` for structured document additions/revisions/removals. Baseline runs remain unmodified; see [Incremental Update Compatibility](UPDATE_COMPATIBILITY.md).

Targeted stage recovery commands:

```bash
python trace_cli.py retry-analysis --output runs/EXACT_TIMESTAMPED_DIRECTORY --env-file .env --workers 2
python trace_cli.py rob-retry --run-dir runs/EXACT_TIMESTAMPED_DIRECTORY --env-file .env
```

- `retry-analysis` archives a completed or blocked terminal analysis and reruns `A3` from reviewed `A2` evidence.
- `rob-retry` retries unresolved RoB assessments (for example, after installing missing RoB instrument resources).
- `python trace_cli.py expert --help` exposes interactive protocol-alignment and expert-review utilities.

---

## 6. Provenance-Linked Results Q&A (`qa`)

```bash
python trace_cli.py qa inspect --run runs/EXACT_TIMESTAMPED_DIRECTORY
python trace_cli.py qa open --run runs/EXACT_TIMESTAMPED_DIRECTORY --env-file .env
```

- `qa inspect` validates and summarizes the run's frozen evidence snapshot locally.
- `qa open` (POSIX) registers the run with the local research library and opens the integrated workspace (`http://127.0.0.1:8770/`).
- For Windows or foreground execution, launch `qa serve` directly:

```bash
python trace_cli.py qa serve --run runs/EXACT_TIMESTAMPED_DIRECTORY \
  --provider-config config/provider.json --env-file .env \
  --port 8765 --log-dir work/qa-sessions
```

Model calls occur only when a question is submitted; see [Results Q&A Documentation](../expert_qa/README.md) and manuscript Fig. 5 for workflow details.

---

## 7. Exit Codes & Analysis States

- **`0` — Handled completion:** Pipeline finished cleanly. Check `analysis_summary.json` or `dashboard.html` for per-analysis outcomes (`pooled` vs. explicit `Not pooled` / `blocked` when minimum-study or poolability gates apply).
- **`1` — Validation or runtime error:** Schema mismatch, missing local file or unhandled execution failure.
- **`2` — Actionable pause or CLI usage error:** Returned when a stage pauses for input—such as `needs_screening_review` (unresolved full-text eligibility awaiting human adjudication), `no_included_studies`, `no_extracted_evidence` or `draft`—or on invalid CLI flags.
