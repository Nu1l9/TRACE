"""Command line interface for the runnable MINERS package."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

from .dataset import ReviewDataset
from .evaluation import aggregate_metrics, evaluate_result, load_result

from . import __version__
from .case_workflow import CandidateCaseConfig, run_candidate_case
from .workflow import (
    HybridReviewConfig,
    HybridReviewWorkflow,
    CorpusPaths,
    build_bm25_index,
    build_dense_index,
    prepare_document_store,
)


DEFAULT_PATHS = CorpusPaths()


def _load_env_file(path: str | Path | None) -> None:
    if path is None:
        return
    env_path = Path(path)
    if not env_path.is_file():
        raise FileNotFoundError(f"Environment file does not exist: {env_path}")
    load_dotenv(env_path, override=False)


def _add_corpus_path_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--papers", type=Path, default=DEFAULT_PATHS.papers)
    parser.add_argument("--test-ids", type=Path, default=DEFAULT_PATHS.test_ids)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_PATHS.corpus)
    parser.add_argument("--store", type=Path, default=DEFAULT_PATHS.store)
    parser.add_argument("--bm25-index", type=Path, default=DEFAULT_PATHS.bm25_index)
    parser.add_argument("--dense-index", type=Path, default=DEFAULT_PATHS.dense_index)


def _corpus_paths_from_args(args: argparse.Namespace) -> CorpusPaths:
    return CorpusPaths(
        papers=args.papers,
        test_ids=args.test_ids,
        corpus=args.corpus,
        store=args.store,
        bm25_index=args.bm25_index,
        dense_index=args.dense_index,
    )


def _command_prepare(args: argparse.Namespace) -> None:
    store = prepare_document_store(
        args.corpus,
        args.store,
        limit=args.limit,
        rebuild=args.rebuild,
    )
    print(json.dumps({"store": str(store.db_path), "documents": store.count()}, indent=2))


def _command_build_bm25(args: argparse.Namespace) -> None:
    build_bm25_index(
        args.store,
        args.index_dir,
        max_features=args.max_features,
        min_df=args.min_df,
        rebuild=args.rebuild,
    )
    print((args.index_dir / "config.json").read_text(encoding="utf-8"))


def _command_build_dense(args: argparse.Namespace) -> None:
    build_dense_index(
        args.store,
        args.index_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        rebuild=args.rebuild,
    )
    print((args.index_dir / "config.json").read_text(encoding="utf-8"))


def _command_review(args: argparse.Namespace) -> None:
    _load_env_file(args.env_file)
    config = HybridReviewConfig(
        paths=_corpus_paths_from_args(args),
        output_dir=args.output_dir,
        mode=args.mode,
        per_query_k=args.per_query_k,
        fusion_k=args.fusion_k,
        final_k=args.final_k,
        date_filter=args.date_filter,
        reranker_model=args.reranker_model,
        screen=args.screen,
        fulltext=args.fulltext,
        fulltext_policy=args.fulltext_policy,
        protocol_mode=args.protocol_mode,
        unsure_policy=args.unsure_policy,
        generate_report=args.generate_report,
        route_rob=args.route_rob,
        rob_review_type=args.rob_review_type,
        expected_designs=tuple(args.expected_design or ()),
        max_rob_fulltext_chars=args.max_rob_fulltext_chars,
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
    )
    results = HybridReviewWorkflow(config).run(args.paper_id)
    payload = {
        "runs": [
            {
                "paper_id": result.paper_id,
                "results": str(result.results_path),
                "report": str(result.report_path),
                "manual_review": str(result.manual_review_path),
                "risk_of_bias": str(result.risk_of_bias_path) if result.risk_of_bias_path else None,
                "metrics": result.metrics,
            }
            for result in results
        ]
    }
    if len(results) > 1:
        payload["batch_aggregate"] = aggregate_metrics([item.metrics for item in results])
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _command_case_review(args: argparse.Namespace) -> None:
    _load_env_file(args.env_file)
    summary = run_candidate_case(
        CandidateCaseConfig(
            specs_zip=args.specs_zip,
            positives_zip=args.positives_zip,
            negatives_zip=args.negatives_zip,
            case_id=args.case_id,
            output_dir=args.output_dir,
            model=args.model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            review_type=args.review_type,
            expected_designs=tuple(args.expected_design or ()),
            max_study_pages=args.max_study_pages,
            max_chars_per_page=args.max_chars_per_page,
            max_screening_fulltext_chars=args.max_screening_fulltext_chars,
            max_fulltext_adjudication_attempts=args.max_fulltext_adjudication_attempts,
            max_rob_fulltext_chars=args.max_rob_fulltext_chars,
            resume_unsure=args.resume_unsure,
            dry_run=args.dry_run,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _command_evaluate(args: argparse.Namespace) -> None:
    dataset = ReviewDataset(args.papers, args.test_ids)
    rows = []
    for path in args.results_dir.rglob("results.json"):
        result = load_result(path)
        paper_id = int(result.get("paper_id", -1))
        if args.test_only and paper_id not in dataset.test_ids:
            continue
        rows.append(evaluate_result(dataset.get(paper_id), result))
    output = {"per_paper": rows, "aggregated": aggregate_metrics(rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["aggregated"], ensure_ascii=False, indent=2))


def _command_doctor(args: argparse.Namespace) -> None:
    _load_env_file(args.env_file)
    paths = _corpus_paths_from_args(args)
    checks = {
        "papers": paths.papers.is_file(),
        "test_ids": paths.test_ids.is_file(),
        "corpus": paths.corpus.is_file(),
        "document_store": paths.store.is_file(),
        "bm25_index": (paths.bm25_index / "config.json").is_file(),
        "dense_index": (paths.dense_index / "config.json").is_file(),
    }
    dense_runtime = None
    if args.check_dense:
        try:
            importlib.import_module("sentence_transformers")
            dense_runtime = {"ok": True}
        except Exception as exc:  # Diagnostic command must report, not raise.
            dense_runtime = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(
        json.dumps(
            {
                "package": "miners",
                "version": __version__,
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "api_key_configured": bool(os.getenv(args.api_key_env)),
                "api_key_env": args.api_key_env,
                "paths": {key: str(value) for key, value in vars(paths).items()},
                "checks": checks,
                "dense_runtime": dense_runtime,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="miners",
        description="MINERS: auditable Hybrid retrieval, PICO screening, full-text adjudication, and RoB routing.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="Build the SQLite store from a local literature corpus.")
    prepare.add_argument("--corpus", type=Path, default=DEFAULT_PATHS.corpus)
    prepare.add_argument("--store", type=Path, default=DEFAULT_PATHS.store)
    prepare.add_argument("--limit", type=int)
    prepare.add_argument("--rebuild", action="store_true")
    prepare.set_defaults(func=_command_prepare)

    bm25 = commands.add_parser("build-bm25", help="Build the sparse index for Hybrid retrieval.")
    bm25.add_argument("--store", type=Path, default=DEFAULT_PATHS.store)
    bm25.add_argument("--index-dir", type=Path, default=DEFAULT_PATHS.bm25_index)
    bm25.add_argument("--max-features", type=int, default=300_000)
    bm25.add_argument("--min-df", type=int, default=2)
    bm25.add_argument("--rebuild", action="store_true")
    bm25.set_defaults(func=_command_build_bm25)

    dense = commands.add_parser("build-dense", help="Build the dense index for Hybrid retrieval.")
    dense.add_argument("--store", type=Path, default=DEFAULT_PATHS.store)
    dense.add_argument("--index-dir", type=Path, default=DEFAULT_PATHS.dense_index)
    dense.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    dense.add_argument("--batch-size", type=int, default=64)
    dense.add_argument("--rebuild", action="store_true")
    dense.set_defaults(func=_command_build_dense)

    review = commands.add_parser(
        "review",
        aliases=["run"],
        help="Run the full Hybrid retrieval -> PICO screening -> full-text workflow.",
    )
    _add_corpus_path_arguments(review)
    review.add_argument("--paper-id", type=int, nargs="+", required=True)
    review.add_argument("--output-dir", type=Path, default=Path("runs/miners"))
    review.add_argument("--mode", choices=("bm25", "dense", "hybrid"), default="hybrid")
    review.add_argument("--per-query-k", type=int, default=1000)
    review.add_argument("--fusion-k", type=int, default=500)
    review.add_argument("--final-k", type=int, default=200)
    review.add_argument("--date-filter", action=argparse.BooleanOptionalAction, default=True)
    review.add_argument("--reranker-model")
    review.add_argument("--screen", action="store_true", help="Enable PICO title/abstract screening.")
    review.add_argument("--fulltext", action="store_true", help="Resolve abstract-Unsure records with local full text.")
    review.add_argument("--fulltext-policy", choices=("unsure", "all-available"), default="unsure")
    review.add_argument("--protocol-mode", choices=("strict", "oracle"), default="strict")
    review.add_argument("--unsure-policy", choices=("include", "exclude"), default="exclude")
    review.add_argument("--generate-report", action="store_true")
    review.add_argument(
        "--route-rob",
        action="store_true",
        help="Route final Include records with local full text to a risk-of-bias tool.",
    )
    review.add_argument(
        "--rob-review-type",
        choices=("auto", "intervention_effect", "exposure_effect", "diagnostic_accuracy", "prognostic_factor", "prediction_model"),
        default="auto",
        help="Optional routing constraint; auto lets the study full text determine the question type.",
    )
    review.add_argument(
        "--expected-design",
        action="append",
        choices=("individual_rct", "cluster_rct", "crossover_rct", "nonrandomized_intervention", "cohort", "case_control", "cross_sectional", "diagnostic_accuracy", "prognostic_factor_study", "prediction_model_study", "other"),
    )
    review.add_argument("--max-rob-fulltext-chars", type=int, default=120_000)
    review.add_argument("--model", default="gpt-4o-mini")
    review.add_argument("--api-key-env", default="OPENAI_API_KEY")
    review.add_argument("--base-url")
    review.add_argument("--env-file", type=Path)
    review.set_defaults(func=_command_review)

    case_review = commands.add_parser(
        "case-review",
        help="Screen a supplied mixed candidate set, then route final Includes to an RoB tool.",
    )
    case_review.add_argument("--specs-zip", type=Path, required=True)
    case_review.add_argument("--positives-zip", type=Path, required=True)
    case_review.add_argument("--negatives-zip", type=Path, required=True)
    case_review.add_argument("--case-id", required=True)
    case_review.add_argument("--output-dir", type=Path, default=Path("runs/case_review"))
    case_review.add_argument("--dry-run", action="store_true")
    case_review.add_argument("--resume-unsure", action="store_true")
    case_review.add_argument(
        "--review-type",
        choices=("auto", "intervention_effect", "exposure_effect", "diagnostic_accuracy", "prognostic_factor", "prediction_model"),
        default="auto",
    )
    case_review.add_argument(
        "--expected-design",
        action="append",
        choices=("individual_rct", "cluster_rct", "crossover_rct", "nonrandomized_intervention", "cohort", "case_control", "cross_sectional", "diagnostic_accuracy", "prognostic_factor_study", "prediction_model_study", "other"),
    )
    case_review.add_argument("--model", default="gpt-5.6-luna")
    case_review.add_argument("--api-key-env", default="OPENAI_API_KEY")
    case_review.add_argument("--base-url", default="https://api.openai.com/v1")
    case_review.add_argument("--env-file", type=Path)
    case_review.add_argument("--max-study-pages", type=int, default=50)
    case_review.add_argument("--max-chars-per-page", type=int, default=12_000)
    case_review.add_argument("--max-screening-fulltext-chars", type=int, default=120_000)
    case_review.add_argument("--max-fulltext-adjudication-attempts", type=int, default=3)
    case_review.add_argument("--max-rob-fulltext-chars", type=int, default=120_000)
    case_review.set_defaults(func=_command_case_review)

    evaluate = commands.add_parser("evaluate", help="Evaluate saved MINERS retrieval/screening runs.")
    evaluate.add_argument("--papers", type=Path, default=DEFAULT_PATHS.papers)
    evaluate.add_argument("--test-ids", type=Path, default=DEFAULT_PATHS.test_ids)
    evaluate.add_argument("--results-dir", type=Path, default=Path("runs/miners"))
    evaluate.add_argument("--test-only", action=argparse.BooleanOptionalAction, default=True)
    evaluate.add_argument("--output", type=Path, default=Path("runs/miners/evaluation.json"))
    evaluate.set_defaults(func=_command_evaluate)

    doctor = commands.add_parser("doctor", help="Check configured data, indexes, and API-key presence without an LLM call.")
    _add_corpus_path_arguments(doctor)
    doctor.add_argument("--env-file", type=Path)
    doctor.add_argument("--api-key-env", default="OPENAI_API_KEY")
    doctor.add_argument(
        "--check-dense",
        action="store_true",
        help="Import sentence-transformers to detect broken optional binary dependencies.",
    )
    doctor.set_defaults(func=_command_doctor)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
