from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import run_benchmark_extraction as portable_adapter


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "benchmark"
    / "meta_analysis_bench"
    / "case_config.json"
)
CONTROLLER_RUNNER = PROJECT_ROOT / "locator_extractor_agent" / "run.py"
LEGACY_FIND_MARKDOWN = portable_adapter.find_markdown


def _manifest_markdown(case_dir: Path, article_id: str) -> Path | None:
    paths_file = case_dir / "inputs" / "extraction_document_paths.json"
    if not paths_file.is_file():
        return None
    payload = json.loads(paths_file.read_text(encoding="utf-8"))
    documents = payload.get("documents", [])
    if isinstance(documents, dict):
        document = documents.get(article_id)
    else:
        document = next(
            (
                item
                for item in documents
                if item.get("article_id") == article_id
                or item.get("document_id") == article_id
            ),
            None,
        )
    if not document:
        return None
    raw_path = document.get("paths", {}).get("markdown")
    if not raw_path:
        return None
    case_root = case_dir.resolve()
    candidate = (case_root / raw_path).resolve()
    if not candidate.is_relative_to(case_root):
        raise ValueError(f"Markdown path escapes portable case: {raw_path}")
    if not candidate.is_file():
        raise FileNotFoundError(f"Manifest Markdown does not exist: {candidate}")
    return candidate


def _find_markdown(
    case_dir: Path, article_id: str, search_roots: list[str]
) -> Path:
    return _manifest_markdown(case_dir, article_id) or LEGACY_FIND_MARKDOWN(
        case_dir, article_id, search_roots
    )


def _finalize_staging_config(config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest_path = config_path.parent / "inputs" / "extraction_document_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config["inputs"]["study_ids"] = {
        article_id: document.get("study_id")
        for article_id, document in manifest["documents"].items()
    }
    config["workflow"] = {
        "stages": ["inspect", "static_clean", "locate", "extract", "organize"],
        "screening": "skip_because_included_studies_are_supplied",
        "extractor": "locator_extractor_tools",
        "planner": "disabled",
        "critic": "disabled",
    }
    portable_adapter.write_json(config_path, config)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adapt and run a portable case with the locator-to-extractor workflow."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--case-id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-staging", action="store_true")
    parser.add_argument("--max-retries", type=int)
    parser.add_argument("--retry-delay-seconds", type=float)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    adapter_config = portable_adapter.load_json(args.config.resolve())
    portable_root = portable_adapter.resolve_project_path(
        adapter_config["portable_root"]
    )
    case_id = args.case_id or adapter_config.get("default_case_id")
    if not case_id:
        raise ValueError("Specify --case-id or configure default_case_id")
    case_dir = portable_root / "cases" / case_id
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Unknown portable benchmark case: {case_dir}")
    if not CONTROLLER_RUNNER.is_file():
        raise FileNotFoundError(f"Locator runner not found: {CONTROLLER_RUNNER}")

    portable_adapter.validate_case(portable_root, case_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    staging_dir = (
        portable_adapter.resolve_project_path(adapter_config["staging_root"])
        / case_id
        / stamp
    )
    portable_adapter.find_markdown = _find_markdown
    config_path, article_ids = portable_adapter.build_staging(
        case_dir, staging_dir, adapter_config
    )
    _finalize_staging_config(config_path)
    runs_dir = portable_adapter.resolve_project_path(adapter_config["runs_root"])
    max_retries = (
        args.max_retries
        if args.max_retries is not None
        else int(adapter_config["max_retries"])
    )
    retry_delay = (
        args.retry_delay_seconds
        if args.retry_delay_seconds is not None
        else float(adapter_config["retry_delay_seconds"])
    )
    command = [
        sys.executable,
        str(CONTROLLER_RUNNER),
        "--case-config",
        str(config_path),
        "--runs-dir",
        str(runs_dir),
        "--max-retries",
        str(max_retries),
        "--retry-delay-seconds",
        str(retry_delay),
    ]
    if args.dry_run:
        command.append("--dry-run")

    print(f"Case: {case_id}", flush=True)
    print(f"Articles configured: {len(article_ids)}", flush=True)
    print("Workflow: static clean -> locate -> extract -> organize", flush=True)
    print("Planner: disabled", flush=True)
    print("Critic: disabled", flush=True)
    print(f"Staging inputs: {staging_dir}", flush=True)
    print(f"Runs root: {runs_dir}", flush=True)
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)

    keep_staging = args.keep_staging or bool(adapter_config.get("keep_staging"))
    if result.returncode == 0 and not keep_staging:
        shutil.rmtree(staging_dir, ignore_errors=True)
    elif keep_staging:
        print(f"Staging retained: {staging_dir}", flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
