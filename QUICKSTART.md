# Quickstart: Two Open-Access Influenza Trials

[Home](README.md)

This walkthrough runs **TRACE** end to end on primary clinical trial reports from the PubMed Central (PMC) Open Access collection, derived from **`ma_0025`** (*immune plasma or hyperimmune immunoglobulin for hospitalized influenza*, evaluating all-cause mortality as a binary outcome). By default, the preparation script downloads two independent randomized trials (`PMC5828518` and `PMC6868512`); an optional third publicly accessible author manuscript (`PMC6941345`) can be added to demonstrate incremental cross-version updates.

---

## 1. Install and Configure

From the repository root, set up a Python 3.12+ environment (tested with Python 3.13) and install the R synthesis dependencies:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-qa.txt -c constraints-tested.txt
Rscript -e 'install.packages(c("metafor", "jsonlite"), repos="https://cloud.r-project.org")'
```

Copy `config/provider.example.json` to `config/provider.json`, set your OpenAI model ID (`https://api.openai.com/v1`), and create a local `provider.env`:

```dotenv
OPENAI_API_KEY=your_openai_api_key
```

To include design-routed risk-of-bias (RoB) drafts during `A1`, install authorized local RoB guidance files beforehand (see [RoB resource setup](resources/rob/README.md)):

```bash
python scripts/install_rob_resources.py --source /path/to/authorized-rob
```

---

## 2. Download and Prepare Primary Trial Reports

```bash
python scripts/prepare_quickstart.py --output work/influenza-input
```

The standard-library preparation script fetches pinned JATS XML versions via the [PMC Cloud Service](https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/), verifies SHA-256 checksums and article identities, and generates:

- `request.json`: structured protocol and `AnalysisProfile` (`protocol.statistical_method_policy = "fixed"`, REML random-effects with 95% normal intervals, comparator covering standard care alone).
- `sources/*.xml`: original JATS XML files.
- `documents/*.md` and `*.content_list.json`: main text, structured HTML tables, table footnotes and figure captions (reference lists excluded from extraction input).
- `SOURCE_MANIFEST.json`: document identities, license metadata and SHA-256 digests for all generated inputs.

| Trial report | Trial registration | PMC access metadata |
| --- | --- | --- |
| [Beigel 2017 — immune plasma](https://pmc.ncbi.nlm.nih.gov/articles/PMC5828518/) | `NCT01052480` | PMC OA (`CC BY-NC-ND 4.0` / TDM) |
| [Davey 2019 — FLU-IVIG](https://pmc.ncbi.nlm.nih.gov/articles/PMC6868512/) | `NCT02287467` | PMC OA (`CC BY-NC-ND 4.0`) |
| [Beigel 2019 — phase 3 immune plasma](https://pmc.ncbi.nlm.nih.gov/articles/PMC6941345/) *(optional update)* | `NCT02572817` | Public author manuscript (TDM terms) |

Once downloaded, the verified XML cache can also prepare inputs offline:

```bash
python scripts/prepare_quickstart.py --output work/influenza-offline --cache work/influenza-input/sources --offline
```

*(Note: If a local macOS Python installation lacks a default CA bundle, export `SSL_CERT_FILE=/etc/ssl/cert.pem` before running the downloader.)*

---

## 3. Validate and Run the Pipeline

The commands below invoke `python -m extensions` to write directly to `runs/influenza` (`python trace_cli.py run` is also supported and appends a timestamp to the output directory name):

```bash
python -m extensions check --request work/influenza-input/request.json --env-file provider.env
python -m extensions run --request work/influenza-input/request.json --output runs/influenza --env-file provider.env --workers 2
```

- `check` validates schemas, document paths and runtime components locally with zero API calls.
- `run` executes PICO screening and RoB routing (`A1`), layout-grounded typed extraction and candidate review (`A2`), and graph-guided evidence selection with `R/metafor` synthesis (`A3`). Append `--resume` to continue an interrupted run with unchanged inputs.

---

## 4. Inspect the Dashboard and Provenance-Linked Q&A

```bash
python scripts/render_run_dashboard.py --run-dir runs/influenza --language en
python -m expert_qa inspect --run runs/influenza
python -m expert_qa serve --run runs/influenza --provider-config config/provider.json --env-file provider.env --log-dir work/influenza-qa
```

Open `runs/influenza/dashboard.html` to inspect screening rationales, RoB drafts, extracted trial counts, pending-review flags and forest plots. Open the localhost URL printed by `expert_qa serve` to ask provenance-linked questions—such as which reports contributed to the mortality estimate, where each arm count appears in the JATS source, or why leave-one-out diagnostics require at least three included studies. See [Results Q&A Documentation](expert_qa/README.md).

---

## 5. Incremental Update with the Third Trial Report

Prepare the three-trial variant (which writes `update.delta.json` containing only `PMC6941345` while keeping the original two reports at their registered paths), inspect the reuse plan, and execute the incremental update in a separate directory:

```bash
python scripts/prepare_quickstart.py --output work/influenza-three --cache work/influenza-input/sources --include-author-manuscript
python -m extensions update --base-run runs/influenza --delta work/influenza-three/update.delta.json --output runs/influenza-update --plan-only
python -m extensions update --base-run runs/influenza --delta work/influenza-three/update.delta.json --output runs/influenza-update --env-file provider.env
```

Compatible screening, RoB, extraction and review artefacts from `runs/influenza` are reused for the first two trials while `PMC6941345` is processed and `A3` recomputes the pooled mortality synthesis. See [Incremental Update Compatibility](docs/UPDATE_COMPATIBILITY.md).

---

## Offline Unit Tests

```bash
python -m unittest discover -s tests -p test_quickstart.py -v
python -m unittest discover -s expert_qa/tests -v
```
