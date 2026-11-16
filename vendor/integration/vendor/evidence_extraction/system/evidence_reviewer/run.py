from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SITE_PACKAGES = PROJECT_ROOT / ".codex_runtime" / "site-packages"
if RUNTIME_SITE_PACKAGES.is_dir():
    sys.path.insert(0, str(RUNTIME_SITE_PACKAGES))
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT.parent / ".env", override=False)
load_dotenv(PROJECT_ROOT / ".env", override=True)

from constrained_meta_agent.loader import load_case_config
from evidence_reviewer.reviewer import (
    review_submission,
    write_review,
    write_review_summary,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run workflow structured read-only batched candidate evidence review."
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--submission", type=Path)
    parser.add_argument("--case-config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=5.0)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def _resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    run_dir = args.run_dir.resolve() if args.run_dir else None
    submission = args.submission
    case_config = args.case_config
    if run_dir is not None:
        submission = submission or run_dir / "case_extraction_submission.json"
        manifest_path = run_dir / "case_run_manifest.json"
        if case_config is None and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("config_path"):
                case_config = Path(manifest["config_path"])
    if submission is None or case_config is None:
        raise ValueError("Provide --run-dir or both --submission and --case-config")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (
        run_dir / "candidate-review-structured" / timestamp
        if run_dir is not None
        else submission.resolve().parent / "candidate-review-structured" / timestamp
    )
    return submission.resolve(), case_config.resolve(), output_dir.resolve()


def main() -> int:
    args = _parse_args()
    submission_path, case_config_path, output_dir = _resolve_inputs(args)
    if not submission_path.is_file() or not case_config_path.is_file():
        raise FileNotFoundError("Submission or case config does not exist")
    if args.limit is not None and args.limit < 2:
        raise ValueError("--limit must be at least 2 for batched review")
    if args.max_retries < 0 or args.retry_delay_seconds < 0:
        raise ValueError("Retry counts and delays must be non-negative")
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    if args.limit is not None:
        submission = dict(submission)
        submission["predictions"] = submission.get("predictions", [])[: args.limit]
    task = load_case_config(case_config_path)
    print(f"Case: {task.case_id}", flush=True)
    print("Workflow version: structured", flush=True)
    print(f"Candidates: {len(submission.get('predictions', []))}", flush=True)
    print(f"Batch size: {args.batch_size}", flush=True)
    print(
        f"Review retries: initial attempt plus {args.max_retries} retries "
        f"({args.retry_delay_seconds:g}s delay)",
        flush=True,
    )
    print(
        "Boundary: review only; no calculation, selection, tool execution, or mutation",
        flush=True,
    )
    review = review_submission(
        task,
        submission,
        configured_model=args.model,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        retry_delay_seconds=args.retry_delay_seconds,
    )
    output_path = output_dir / "candidate_evidence_review_structured.json"
    write_review(output_path, review)
    summary_path = output_dir / "review_summary.md"
    write_review_summary(summary_path, review)
    print(output_path, flush=True)
    print(summary_path, flush=True)
    return 1 if review["failed_review_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
