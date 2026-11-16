# Open-Corpus Literature Retrieval

[Home](../README.md)

The `review` command extends TRACE to start directly from a bibliographic corpus, discovering candidate studies via hybrid lexical + dense retrieval before handing the frozen candidate set to `A1`–`A3`:

```text
Research protocol → BM25 + Dense → weighted RRF → optional cross-encoder reranking
  → frozen candidate set → A1 screening & RoB → A2 extraction & review
  → A3 selection & synthesis → dashboards & results Q&A
```

Whereas `run --request ...` operates on a pre-supplied `documents` list, a `review` request omits `documents` (or sets `documents: []`) while retaining the full `protocol` and `AnalysisProfile`.

---

## 1. Corpus Format & Document Mapping

Supply a JSON array or JSONL file where each record provides an integer `ID`, `title`, `abstract` and optional `year`. Retrieval indexes `title` and `abstract`. For downstream full-text adjudication and extraction, each retrieved candidate also requires:

- `study_id`: independent study identifier (shared across multiple reports of the same trial or cohort).
- Full-text source paths (`paths.markdown`, `paths.pdf` and optional `paths.content_list`) or structured `sections`.

Source paths in corpus records resolve relative to the corpus file, or can be supplied separately via `--document-map /path/to/map.json` keyed by string corpus ID:

```json
{
  "123": {
    "study_id": "trial_A",
    "paths": {
      "markdown": "sources/report_A.md",
      "pdf": "sources/report_A.pdf"
    }
  }
}
```

Any retrieved candidate lacking full-text paths or `study_id` is recorded explicitly in `00_retrieval/retrieval.json` so missing materials can be supplied before downstream synthesis.

---

## 2. Build the Corpus Store & Indexes

```bash
python trace_cli.py prepare-corpus \
  --corpus /path/to/corpus.json --store cache/literature.sqlite

python trace_cli.py build-bm25 \
  --store cache/literature.sqlite --index-dir cache/bm25

# Optional local dense embeddings & reranking dependencies:
pip install -r requirements-retrieval.txt
python trace_cli.py build-dense \
  --store cache/literature.sqlite --index-dir cache/dense \
  --model BAAI/bge-small-en-v1.5 --batch-size 32
```

 Pass `--rebuild` whenever the underlying corpus or index parameters change.

### Hybrid retrieval formulation

From the research question and structured PICO fields (`P`, `I/E`, `C`, `O`), TRACE deterministically constructs up to four deduplicated queries—`question` (`w = 1.0`), `question + PICO` (`w = 1.2`), `P + I/E + O` (`w = 1.0`) and `I/E + O` (`w = 0.7`).

- **BM25:** $k_1 = 1.5$, $b = 0.75$ over lowercase Unicode-normalized unigrams (`len >= 2`, `min_df = 2`, vocabulary cap `300,000`).
- **Dense:** L2-normalized local embeddings (`BAAI/bge-small-en-v1.5` by default) with exact block-wise inner-product search.
- **Weighted RRF:** Each query retrieves up to `--per-query-k` (default `1,000`) candidates per retriever and accumulates `weight / (60 + rank)`, retaining the top `--fusion-k` (`500`) fused candidates and outputting the top `--final-k` (`200`) screening candidates.

---

## 3. Execute Retrieval & Downstream Synthesis

```bash
python trace_cli.py review \
  --request /path/to/research_request.json \
  --store cache/literature.sqlite \
  --bm25-index cache/bm25 --dense-index cache/dense \
  --mode hybrid --per-query-k 1000 --fusion-k 500 --final-k 200 \
  --env-file .env --output runs/literature_review --workers 2
```

- Use `--mode bm25` for lexical-only retrieval without local dense embeddings.
- Add `--reranker-model MODEL` for optional CrossEncoder reranking, or `--max-year YEAR` to filter out publications after a cutoff year (records with unknown year are retained).
- Add `--prepare-only` to write the frozen candidate set (`00_retrieval/request.json` and `00_retrieval/retrieval.json`) without launching LLM screening. Repeat the command with the printed timestamped directory, `--resume` and `--env-file .env` to continue from the frozen candidate set.

---

## 4. Minimal Local Example

The bundled synthetic corpus runs locally with zero model downloads or API calls:

```bash
python trace_cli.py prepare-corpus \
  --corpus examples/literature/corpus.json --store cache/demo.sqlite
python trace_cli.py build-bm25 \
  --store cache/demo.sqlite --index-dir cache/demo_bm25 --min-df 1
python trace_cli.py review \
  --request examples/literature/request.json \
  --store cache/demo.sqlite --bm25-index cache/demo_bm25 \
  --mode bm25 --output runs/retrieval_demo --prepare-only
```
