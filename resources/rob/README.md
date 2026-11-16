# Risk-of-Bias (RoB) Instrument Resource Setup

[Home](../../README.md)

During stage `A1`, TRACE routes included studies by study design across six established risk-of-bias instruments (seven versioned routes). Because upstream RoB instruments carry separate copyright and licensing terms (such as `CC BY-NC-ND 4.0`), this repository distributes the routing code, schema validators, checksum catalog ([`catalog.json`](catalog.json)) and local installer rather than bundling third-party instrument texts.

---

## 1. Supported Instruments & Required Files

All relative paths below are resolved under `resources/rob/` (and must follow the identical relative directory layout inside your local source folder, e.g. `/path/to/authorized-rob/`). Each route pairs an instrument guide text with a compact structured-assessment prompt file, verified by SHA-256 and byte length against [`catalog.json`](catalog.json):

| Installer ID / Instrument | Pinned version | Guide path | Compact prompt path | Official instrument page |
| --- | --- | --- | --- | --- |
| `rob2` / RoB 2 | `2019-08-22` | `text/RoB 2/03_RoB2_cribsheet_2019.txt` | `prompts/rob2.txt` | [riskofbias.info](https://www.riskofbias.info/welcome/rob-2-0-tool) |
| `robins-i` / ROBINS-I | `V2-2025-11-20-draft` | `text/ROBINS-I V2/ROBINS-I_V2_cribsheet_2025-11-20.txt` | `prompts/robins-i.txt` | [riskofbias.info](https://www.riskofbias.info/welcome/robins-i-tool) |
| `robins-i-2016` / ROBINS-I | `2016` | `text/ROBINS-I/04_ROBINS-I_score_sheet_template_2016.txt` | `prompts/robins-i-2016.txt` | [riskofbias.info](https://www.riskofbias.info/welcome/robins-i-tool) |
| `robins-e` / ROBINS-E | `2024-03-24` | `text/ROBINS-E/03_ROBINS-E_score_sheet_template_24March2024.txt` | `prompts/robins-e.txt` | [riskofbias.info](https://www.riskofbias.info/welcome/robins-e-tool) |
| `quadas3` / QUADAS-3 | `1.2 (2026)` | `text/QUADAS 3/02_QUADAS3_latest_version_1.2_score_sheet.txt` | `prompts/quadas3.txt` | [University of Bristol](https://www.bristol.ac.uk/population-health-sciences/projects/quadas/quadas-3/) |
| `quips` / QUIPS | `2013` | `quips_rubric.txt` | `prompts/quips.txt` | [Cochrane Prognosis](https://methods.cochrane.org/prognosis/tools) |
| `probast-ai` / PROBAST+AI | `2025` | `text/PROBAST/05_PROBAST_AI_latest_score_sheet_table3.txt` | `prompts/probast-ai.txt` | [probast.org](https://www.probast.org/) |

Installing all seven routes requires 14 text files; individual routes can also be installed selectively with `--tool <id>`. ([`required_files.json`](required_files.json) retains historical archive checksums for legacy checkpoint compatibility and is not needed for new runs.)

If you hold an authorized legacy archive containing the compact prompt constants, extract them deterministically (via AST literal parsing without code execution) using:

```bash
python scripts/prepare_rob_prompts.py \
  --legacy-code /path/to/authorized-historical/extensions/screening_bias.py \
  --output /path/to/authorized-rob
```

---

## 2. Install & Verify

Run from the repository root once your local `/path/to/authorized-rob` directory is prepared:

```bash
# Inspect all routes, versions, target paths and official URLs
python scripts/install_rob_resources.py --list

# Validate all 14 files in the source directory without copying
python scripts/install_rob_resources.py --source /path/to/authorized-rob --check

# Install all seven routes and verify the installed files
python scripts/install_rob_resources.py --source /path/to/authorized-rob
python scripts/install_rob_resources.py --check

# Or install/verify specific routes only (repeat --tool as needed)
python scripts/install_rob_resources.py --source /path/to/authorized-rob --tool rob2
python scripts/install_rob_resources.py --check --tool rob2
```

Installed files under `resources/rob/text/` and `resources/rob/prompts/` remain Git-ignored. At runtime, TRACE verifies file hashes before invoking RoB assessment; if a route's resources are not installed, the RoB status is recorded as unresolved (`needs-assessment`) without blocking independent extraction and synthesis stages. See [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md) for attribution details.
