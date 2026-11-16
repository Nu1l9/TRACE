# Provenance-Linked Results Q&A (`expert_qa`)

[Home](../README.md)

`expert_qa` provides a read-only, tool-calling assistant and local web workspace over frozen TRACE runs (`Fig. 5`). Each conversation is scoped to a single saved case, run and analysis snapshot, allowing investigators to query study inclusion/exclusion rationales, risk-of-bias judgements, extracted source records, fixed-versus-random model comparisons, leave-one-out sensitivity diagnostics and source-grounded hypotheses for follow-up clinical research.

Every response is assembled from read-only tool calls over persisted run artefacts and primary Markdown/`content_list` passages, with programmatic verification of citation IDs and verbatim quoted excerpts. `expert_qa` never modifies run files, alters study selection or reruns statistical pooling; exploratory explanations and trial-design suggestions are presented explicitly alongside their supporting evidence and proposed verification steps.

---

## 1. Launch on a Single Saved Run

From the repository root with your Python environment activated:

```bash
python -m pip install -r requirements-qa.txt
python trace_cli.py qa inspect --run /path/to/saved-run
python trace_cli.py qa serve \
  --run /path/to/saved-run \
  --provider-config config/provider.json \
  --env-file /path/to/provider.env \
  --log-dir /path/to/qa-sessions
```

Open the printed localhost URL (default `http://127.0.0.1:8765/`). Inspecting a snapshot and loading the web UI make zero model calls; API requests occur only when a question is submitted. Store `--log-dir` outside the scientific run directory, and keep source Markdown / `content_list` files at their declared paths when relocating runs.

---

## 2. Multi-Run Research Library

To browse and query multiple completed runs from one unified workspace (`http://127.0.0.1:8770/` by default):

```bash
python -m expert_qa register --run /path/to/saved-run --env-file /path/to/provider.env --data-dir /path/to/qa-library
python -m expert_qa library --data-dir /path/to/qa-library
```

On POSIX systems, `python trace_cli.py qa open --run /path/to/saved-run --env-file .env` registers the run and launches the library workspace automatically. Set `META_PIPELINE_QA_DATA_DIR` or pass `--data-dir` to customize the library path (defaults to the OS user-data directory under `meta_pipeline/result_guide`). Passing `--install-entries` to `register` adds workspace navigation links to saved dashboards while backing up the original HTML and leaving all JSON artefacts untouched.

---

## 3. Provider Configuration

`expert_qa` reads `text_provider` from `config/provider.json` (`https://api.openai.com/v1` by default; no embedding endpoint is needed) and authenticates with `OPENAI_API_KEY` from `--env-file` or the environment. Optional QA-specific overrides (`EXPERT_QA_MODEL`, `EXPERT_QA_BASE_URL`, `EXPERT_QA_API_KEY`) take precedence when set.

---

## 4. Unit Tests

```bash
python -m unittest discover -s expert_qa/tests -v
```
