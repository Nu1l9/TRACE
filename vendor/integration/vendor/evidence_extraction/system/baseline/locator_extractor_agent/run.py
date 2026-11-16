from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SITE_PACKAGES = PROJECT_ROOT / ".codex_runtime" / "site-packages"
if RUNTIME_SITE_PACKAGES.is_dir():
    sys.path.insert(0, str(RUNTIME_SITE_PACKAGES))
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

FALLBACK_ENV_FILE = PROJECT_ROOT.parent / ".env"
PROJECT_ENV_FILE = PROJECT_ROOT / ".env"
if FALLBACK_ENV_FILE.is_file():
    load_dotenv(FALLBACK_ENV_FILE, override=False)
if PROJECT_ENV_FILE.is_file():
    load_dotenv(PROJECT_ENV_FILE, override=True)
ENV_FILE = PROJECT_ENV_FILE if PROJECT_ENV_FILE.is_file() else FALLBACK_ENV_FILE

from constrained_meta_agent.loader import load_case_config
from locator_extractor_agent.controller import run_article


DEFAULT_RUNS_DIR = (
    PROJECT_ROOT
    / "benchmark"
    / "meta_analysis_bench"
    / "runs"
    / "locator-extractor-cases"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the isolated locator-to-extractor workflow."
    )
    parser.add_argument("--case-config", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-retries", type=int, default=15)
    parser.add_argument("--retry-delay-seconds", type=float, default=5.0)
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def _submission_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    projected = copy.deepcopy(prediction)
    if projected.get("quantitative_usability") == "needs_extraction":
        projected["internal_quantitative_usability"] = "needs_extraction"
        projected["quantitative_usability"] = "source_only"
    return projected


def main() -> int:
    args = _parse_args()
    if args.max_retries < 0 or args.retry_delay_seconds < 0:
        raise ValueError("Retry counts and delays must be non-negative")
    if not args.dry_run and not any(
        os.getenv(name) for name in ("OPENAI_API_KEY", "TEXT_PROVIDER_API_KEY", "PIPELINE_LLM_API_KEY")
    ):
        raise RuntimeError(f"No supported API key was loaded from {ENV_FILE}")

    config_path = args.case_config.resolve()
    runs_dir = args.runs_dir.resolve()
    task = load_case_config(config_path)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    started_at = datetime.now(timezone.utc).isoformat()
    case_run_dir = runs_dir / task.case_id / "extraction" / "locator-run" / run_id
    article_runs: list[dict[str, Any]] = []
    failed_articles: list[str] = []
    case_predictions: list[dict[str, Any]] = []

    for article_id in task.included_article_ids:
        run_dir = case_run_dir / article_id
        attempts = args.max_retries + 1
        for attempt in range(1, attempts + 1):
            attempt_name = f"attempt_{attempt:02d}"
            try:
                result = run_article(task, article_id, run_dir, args.dry_run)
                _write_json(run_dir / "run_manifest.json", result["manifest"])
                _write_json(run_dir / "controller_state.json", result["controller_state"])
                if not args.dry_run:
                    projected_submission = copy.deepcopy(
                        result["extraction_submission"]
                    )
                    projected_submission["predictions"] = [
                        _submission_prediction(prediction)
                        for prediction in projected_submission["predictions"]
                    ]
                    _write_json(
                        run_dir / "raw_system_output.json", result["raw_system_output"]
                    )
                    _write_json(
                        run_dir / "extraction_submission.json",
                        projected_submission,
                    )
                    _write_json(run_dir / "issues.json", result["issues"])
                    case_predictions.extend(projected_submission["predictions"])
                article_runs.append(
                    {
                        "article_id": article_id,
                        "run_dir": str(run_dir),
                        "status": "completed",
                        "attempts": attempt,
                    }
                )
                break
            except Exception as exc:
                error_dir = (
                    case_run_dir / "failed_attempts" / article_id / attempt_name
                )
                error_dir.parent.mkdir(parents=True, exist_ok=True)
                if run_dir.exists():
                    shutil.move(str(run_dir), str(error_dir))
                else:
                    error_dir.mkdir(parents=True, exist_ok=True)
                _write_json(
                    error_dir / "failure.json",
                    {
                        "article_id": article_id,
                        "attempt": attempt,
                        "provider": os.getenv("META_AGENT_LLM_PROVIDER", os.getenv("TEXT_PROVIDER_NAME", "official_openai")),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "failed_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                if attempt == attempts:
                    failed_articles.append(article_id)
                    article_runs.append(
                        {
                            "article_id": article_id,
                            "status": "failed",
                            "attempts": attempt,
                            "failure_dir": str(error_dir.parent),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                    break
                print(
                    f"[{article_id}] attempt {attempt} failed ({type(exc).__name__}); "
                    f"retrying in {args.retry_delay_seconds:g}s.",
                    flush=True,
                )
                time.sleep(args.retry_delay_seconds)

    case_run_dir.mkdir(parents=True, exist_ok=True)
    case_submission_path = case_run_dir / "case_extraction_submission.json"
    if not args.dry_run:
        _write_json(
            case_submission_path,
            {
                "case_id": task.case_id,
                "task_id": task.task_id,
                "task_type": "extraction",
                "submission_status": (
                    "scoreable" if case_predictions and not failed_articles
                    else "requires_human_review"
                ),
                "predictions": case_predictions,
            },
        )
    _write_json(
        case_run_dir / "case_run_manifest.json",
        {
            "case_id": task.case_id,
            "task_id": task.task_id,
            "workflow": "locator_extractor_tools",
            "config_path": str(config_path),
            "started_at": started_at,
            "dry_run": args.dry_run,
            "max_retries": args.max_retries,
            "included_article_ids": task.included_article_ids,
            "article_runs": article_runs,
            "failed_article_ids": failed_articles,
            "prediction_count": len(case_predictions),
            "case_submission": (
                str(case_submission_path) if not args.dry_run else None
            ),
        },
    )
    print(case_run_dir)
    return 1 if failed_articles else 0


if __name__ == "__main__":
    raise SystemExit(main())
