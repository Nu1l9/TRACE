from __future__ import annotations

import json
import os
import shutil
import sys
import time
import traceback
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SITE_PACKAGES = PROJECT_ROOT / ".codex_runtime" / "site-packages"


def _configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass


_configure_console_encoding()

if RUNTIME_SITE_PACKAGES.is_dir():
    sys.path.insert(0, str(RUNTIME_SITE_PACKAGES))

from dotenv import load_dotenv

FALLBACK_ENV_FILE = PROJECT_ROOT.parent / ".env"
PROJECT_ENV_FILE = PROJECT_ROOT / ".env"
if FALLBACK_ENV_FILE.is_file():
    load_dotenv(FALLBACK_ENV_FILE, override=False)
if PROJECT_ENV_FILE.is_file():
    load_dotenv(PROJECT_ENV_FILE, override=True)
ENV_FILE = PROJECT_ENV_FILE if PROJECT_ENV_FILE.is_file() else FALLBACK_ENV_FILE

if __package__ in {None, ""}:
    sys.path.insert(0, str(PROJECT_ROOT))
    from constrained_meta_agent.controller import run_article
    from constrained_meta_agent.loader import load_case_config
else:
    from .controller import run_article
    from .loader import load_case_config

DEFAULT_CASE_CONFIG = PROJECT_ROOT / "benchmark" / "meta_analysis_bench" / "cases" / "ma_0001" / "case_config.json"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "benchmark" / "meta_analysis_bench" / "runs"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a constrained meta-analysis case without gold scoring."
    )
    parser.add_argument(
        "--case-config",
        type=Path,
        default=DEFAULT_CASE_CONFIG,
        help="Path to case_config.json.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Directory where run artifacts will be written.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and write manifests without calling the LLM.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=15,
        help="Retries for each failed article after its initial attempt.",
    )
    parser.add_argument(
        "--retry-delay-seconds",
        type=float,
        default=5.0,
        help="Seconds to wait between retries for a failed article.",
    )
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def main() -> int:
    args = _parse_args()
    config_path = args.case_config.resolve()
    runs_dir = args.runs_dir.resolve()
    dry_run = args.dry_run
    if args.max_retries < 0:
        raise ValueError("--max-retries must be zero or greater")
    if args.retry_delay_seconds < 0:
        raise ValueError("--retry-delay-seconds must be zero or greater")

    if not dry_run and not any(
        os.getenv(name) for name in ("OPENAI_API_KEY", "TEXT_PROVIDER_API_KEY", "PIPELINE_LLM_API_KEY")
    ):
        raise RuntimeError(
            f"TEXT_PROVIDER_API_KEY or PIPELINE_LLM_API_KEY was not loaded from {ENV_FILE}"
        )

    task = load_case_config(config_path)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    case_started_at = datetime.now(timezone.utc).isoformat()
    case_run_dir = runs_dir / task.case_id / "extraction" / "config-run" / run_id
    article_runs = []
    failed_articles = []
    for article_id in task.included_article_ids:
        run_dir = case_run_dir / article_id
        completed = False
        attempts = args.max_retries + 1
        for attempt in range(1, attempts + 1):
            attempt_name = f"attempt_{attempt:02d}"
            try:
                result = run_article(task, article_id, run_dir, dry_run)
                _write_json(run_dir / "run_manifest.json", result["manifest"])
                _write_json(run_dir / "controller_state.json", result["controller_state"])
                if not dry_run:
                    _write_json(run_dir / "raw_system_output.json", result["raw_system_output"])
                    _write_json(run_dir / "extraction_submission.json", result["extraction_submission"])
                    _write_json(run_dir / "issues.json", result.get("issues", result["adapter_issues"]))
                    if task.raw["workflow"]["extractor"] == "legacy_graph_compatibility":
                        _write_json(run_dir / "adapter_issues.json", result["adapter_issues"])
                article_runs.append(
                    {
                        "article_id": article_id,
                        "run_dir": str(run_dir),
                        "status": "completed",
                        "attempts": attempt,
                    }
                )
                completed = True
                break
            except Exception as exc:
                error_dir = case_run_dir / "failed_attempts" / article_id / attempt_name
                if run_dir.exists():
                    error_dir.parent.mkdir(parents=True, exist_ok=True)
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
                    print(
                        f"[{article_id}] failed after {attempts} attempts on the selected provider; "
                        "continuing with the next article.",
                        flush=True,
                    )
                    article_runs.append(
                        {
                            "article_id": article_id,
                            "status": "failed",
                            "attempts": attempt,
                            "provider": os.getenv("META_AGENT_LLM_PROVIDER", "official_openai"),
                            "failure_dir": str(error_dir.parent),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                    failed_articles.append(article_id)
                    break
                print(
                    f"[{article_id}] attempt {attempt} failed ({type(exc).__name__}); "
                    f"retrying in {args.retry_delay_seconds:g}s "
                    f"({args.max_retries - attempt + 1} retries remaining).",
                    flush=True,
                )
                time.sleep(args.retry_delay_seconds)

        if not completed:
            continue

    _write_json(
        case_run_dir / "case_run_manifest.json",
        {
            "case_id": task.case_id,
            "task_id": task.task_id,
            "config_path": str(config_path.resolve()),
            "started_at": case_started_at,
            "dry_run": dry_run,
            "max_retries": args.max_retries,
            "retry_delay_seconds": args.retry_delay_seconds,
            "included_article_ids": task.included_article_ids,
            "article_runs": article_runs,
            "failed_article_ids": failed_articles,
        },
    )
    print(case_run_dir)
    return 1 if failed_articles else 0


if __name__ == "__main__":
    raise SystemExit(main())
