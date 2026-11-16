from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SNAPSHOT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SNAPSHOT_ROOT.parents[1]
RUNTIME_SITE_PACKAGES = PROJECT_ROOT / ".codex_runtime" / "site-packages"
if RUNTIME_SITE_PACKAGES.is_dir():
    sys.path.insert(0, str(RUNTIME_SITE_PACKAGES))
# Put the nested frozen package ahead of the live package. The project root is
# still available for shared dependencies such as constrained_meta_agent.
sys.path.insert(0, str(SNAPSHOT_ROOT))
sys.path.insert(1, str(PROJECT_ROOT))

from dotenv import load_dotenv

FALLBACK_ENV_FILE = PROJECT_ROOT.parent / ".env"
PROJECT_ENV_FILE = PROJECT_ROOT / ".env"
# Typed follows the project policy that credentials live in E:\miners\.env.
# The inner project file may supply defaults but cannot override outer keys.
if PROJECT_ENV_FILE.is_file():
    load_dotenv(PROJECT_ENV_FILE, override=False)
if FALLBACK_ENV_FILE.is_file():
    load_dotenv(FALLBACK_ENV_FILE, override=True)
ENV_FILE = FALLBACK_ENV_FILE if FALLBACK_ENV_FILE.is_file() else PROJECT_ENV_FILE

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
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--review-structured",
        action="store_true",
        help="Run the read-only batched candidate evidence review after extraction.",
    )
    parser.add_argument("--review-structured-batch-size", type=int, default=12)
    parser.add_argument("--review-structured-workers", type=int, default=4)
    parser.add_argument("--review-structured-max-retries", type=int, default=3)
    parser.add_argument("--review-structured-retry-delay-seconds", type=float, default=5.0)
    parser.add_argument(
        "--review-structured-provider",
        choices=("official_openai", "openai_compatible"),
        help="Provider for the post-extraction review only.",
    )
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


def _run_article_with_retries(
    task: Any,
    article_id: str,
    case_run_dir: Path,
    dry_run: bool,
    max_retries: int,
    retry_delay: float,
) -> dict[str, Any]:
    """Run one article serially; multiple articles are scheduled by the caller."""
    run_dir = case_run_dir / article_id
    attempts = max_retries + 1
    provider = os.getenv("META_AGENT_LLM_PROVIDER", os.getenv("TEXT_PROVIDER_NAME", "official_openai"))
    for attempt in range(1, attempts + 1):
        attempt_name = f"attempt_{attempt:02d}"
        try:
            result = run_article(
                task,
                article_id,
                run_dir,
                dry_run,
                provider=provider,
            )
            _write_json(run_dir / "run_manifest.json", result["manifest"])
            _write_json(run_dir / "controller_state.json", result["controller_state"])
            predictions: list[dict[str, Any]] = []
            if not dry_run:
                projected_submission = copy.deepcopy(
                    result["extraction_submission"]
                )
                projected_submission["predictions"] = [
                    _submission_prediction(prediction)
                    for prediction in projected_submission["predictions"]
                ]
                _write_json(run_dir / "raw_system_output.json", result["raw_system_output"])
                _write_json(
                    run_dir / "extraction_submission.json",
                    projected_submission,
                )
                _write_json(run_dir / "issues.json", result["issues"])
                predictions = projected_submission["predictions"]
            return {
                "article_run": {
                    "article_id": article_id,
                    "run_dir": str(run_dir),
                    "status": "completed",
                    "attempts": attempt,
                    "provider": provider,
                },
                "predictions": predictions,
            }
        except Exception as exc:
            error_dir = case_run_dir / "failed_attempts" / article_id / attempt_name
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
                    "provider": provider,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            if attempt == attempts:
                return {
                    "article_run": {
                        "article_id": article_id,
                        "status": "failed",
                        "attempts": attempt,
                        "failure_dir": str(error_dir.parent),
                        "provider": provider,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    "predictions": [],
                }
            print(
                f"[{article_id}] attempt {attempt} failed ({type(exc).__name__}); "
                f"retrying in {retry_delay:g}s.",
                flush=True,
            )
            time.sleep(retry_delay)
    raise RuntimeError(f"Retry loop ended unexpectedly for {article_id}")


def main() -> int:
    args = _parse_args()
    if args.max_retries < 0 or args.retry_delay_seconds < 0:
        raise ValueError("Retry counts and delays must be non-negative")
    if args.max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if args.review_structured_batch_size < 2:
        raise ValueError("review-structured-batch-size must be at least 2")
    if args.review_structured_workers < 1:
        raise ValueError("review-structured-workers must be at least 1")
    if args.review_structured_max_retries < 0 or args.review_structured_retry_delay_seconds < 0:
        raise ValueError("review-structured retry counts and delays must be non-negative")
    if not args.dry_run and not any(
        os.getenv(name) for name in ("OPENAI_API_KEY", "TEXT_PROVIDER_API_KEY", "COMPATIBLE_API_KEY")
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

    worker_count = min(args.max_workers, len(task.included_article_ids))
    print(f"Article workers: {worker_count}", flush=True)
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="locator-article",
    ) as executor:
        futures = {
            article_id: executor.submit(
                _run_article_with_retries,
                task,
                article_id,
                case_run_dir,
                args.dry_run,
                args.max_retries,
                args.retry_delay_seconds,
            )
            for article_id in task.included_article_ids
        }
        for article_id in task.included_article_ids:
            result = futures[article_id].result()
            article_runs.append(result["article_run"])
            case_predictions.extend(result["predictions"])
            if result["article_run"]["status"] == "failed":
                failed_articles.append(article_id)

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
    case_manifest = {
        "case_id": task.case_id,
        "task_id": task.task_id,
        "workflow": "locator_extractor_tools",
        "config_path": str(config_path),
        "started_at": started_at,
        "dry_run": args.dry_run,
        "max_retries": args.max_retries,
        "max_workers": worker_count,
        "parallel_articles": True,
        "included_article_ids": task.included_article_ids,
        "article_runs": article_runs,
        "failed_article_ids": failed_articles,
        "prediction_count": len(case_predictions),
        "case_submission": str(case_submission_path) if not args.dry_run else None,
        "candidate_review_structured": {
            "requested": args.review_structured,
            "status": "not_requested" if not args.review_structured else "pending",
            "provider": args.review_structured_provider if args.review_structured else None,
            "batch_size": args.review_structured_batch_size if args.review_structured else None,
            "max_workers": args.review_structured_workers if args.review_structured else None,
            "max_retries": args.review_structured_max_retries if args.review_structured else None,
            "retry_delay_seconds": (
                args.review_structured_retry_delay_seconds if args.review_structured else None
            ),
            "output_path": None,
            "summary_path": None,
        },
    }
    _write_json(case_run_dir / "case_run_manifest.json", case_manifest)
    if args.review_structured and not args.dry_run and not failed_articles:
        from evidence_reviewer.reviewer import (
            review_submission,
            write_review,
            write_review_summary,
        )

        previous_provider = os.environ.get("META_AGENT_LLM_PROVIDER")
        if args.review_structured_provider:
            os.environ["META_AGENT_LLM_PROVIDER"] = args.review_structured_provider
        try:
            provider = os.environ.get("META_AGENT_LLM_PROVIDER", "official_openai")
            print(f"Candidate review structured: starting (provider={provider})", flush=True)
            review = review_submission(
                task,
                {
                    "case_id": task.case_id,
                    "task_id": task.task_id,
                    "predictions": case_predictions,
                },
                batch_size=args.review_structured_batch_size,
                max_workers=args.review_structured_workers,
                max_retries=args.review_structured_max_retries,
                retry_delay_seconds=args.review_structured_retry_delay_seconds,
            )
        finally:
            if args.review_structured_provider:
                if previous_provider is None:
                    os.environ.pop("META_AGENT_LLM_PROVIDER", None)
                else:
                    os.environ["META_AGENT_LLM_PROVIDER"] = previous_provider
        review_dir = (
            case_run_dir
            / "candidate-review-structured"
            / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        review_path = review_dir / "candidate_evidence_review_structured.json"
        write_review(review_path, review)
        summary_path = review_dir / "review_summary.md"
        write_review_summary(summary_path, review)
        case_manifest["candidate_review_structured"].update(
            {
                "status": "completed",
                "output_path": str(review_path),
                "summary_path": str(summary_path),
            }
        )
        _write_json(case_run_dir / "case_run_manifest.json", case_manifest)
        print(f"Candidate review structured: {review_path}", flush=True)
    elif args.review_structured and failed_articles:
        case_manifest["candidate_review_structured"]["status"] = "skipped_failed_extraction"
        _write_json(case_run_dir / "case_run_manifest.json", case_manifest)
        print(
            "Candidate review structured skipped because one or more article extractions failed.",
            flush=True,
        )
    print(case_run_dir)
    return 1 if failed_articles else 0


if __name__ == "__main__":
    raise SystemExit(main())
