# Contributing

Run changes in an isolated Python virtual environment using synthetic or local test fixtures. Before submitting a pull request, verify that all repository boundary checks and offline test suites pass:

```bash
python scripts/check_repository.py --tracked
python scripts/test_portable.py
```

- **Preserve stage contracts and checkpoints:** Keep cross-stage JSON contracts, `component_provenance.json` and vendored module import paths stable so checkpoint recovery and cross-version `update` workflows remain compatible.
- **Keep private and third-party artefacts out of Git:** Never commit API keys, `.env` or `config/provider.json` files, third-party primary study PDFs, external RoB guide texts, runtime `runs/` outputs, Q&A session logs or held-out benchmark Gold files. Use synthetic fixtures under `tests/` for regression coverage.
