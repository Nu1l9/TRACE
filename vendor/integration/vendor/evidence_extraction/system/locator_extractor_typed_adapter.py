"""Run the remaining structured-shaped cases through the independent typed runtime.

Typed reads the already-normalized structured input package and writes only a transient
controller staging directory plus typed run artifacts.  It does not alter the structured
input package, the frozen baseline snapshots, or historical runs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "benchmark"
    / "meta_analysis_bench"
    / "locator_extractor_typed_data_family_config.json"
)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def project_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("description") or value.get("label") or "")
    return str(value or "")


def extraction_task_path(case_dir: Path) -> Path:
    for name in ("extraction_all_reports.json", "extraction_all_archived_reports.json"):
        candidate = case_dir / "tasks" / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No extraction task under {case_dir / 'tasks'}")


def document_records(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    documents = manifest.get("documents") or {}
    if isinstance(documents, list):
        return {
            str(item.get("article_id") or item.get("document_id")): item
            for item in documents
            if isinstance(item, dict) and (item.get("article_id") or item.get("document_id"))
        }
    if isinstance(documents, dict):
        return {
            str(article_id): {"article_id": article_id, **item}
            for article_id, item in documents.items()
            if isinstance(item, dict)
        }
    raise ValueError("Document manifest has no usable documents collection")


def path_records(paths: dict[str, Any]) -> dict[str, dict[str, Any]]:
    documents = paths.get("documents") or {}
    if not isinstance(documents, dict):
        raise ValueError("Structured-shaped document paths must be keyed by article ID")
    return {str(article_id): item for article_id, item in documents.items() if isinstance(item, dict)}


def resolve_representation(case_dir: Path, path_record: dict[str, Any], name: str) -> Path:
    raw_path = path_record.get(name) or (path_record.get("paths") or {}).get(name)
    if not raw_path:
        raise FileNotFoundError(f"No {name} path declared")
    candidate = Path(str(raw_path))
    resolved = candidate.resolve() if candidate.is_absolute() else (case_dir / candidate).resolve()
    root = case_dir.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Representation path escapes structured case: {raw_path}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing declared {name}: {resolved}")
    return resolved


def controller_target(target: dict[str, Any], pico: dict[str, Any]) -> dict[str, Any]:
    outcome = target.get("outcome") or {}
    if not text(outcome):
        outcome = {
            "description": text(pico.get("outcome"))
            or "Source-reported outcome summary",
            "aliases": [],
        }
    elif isinstance(outcome, str):
        outcome = {"description": outcome, "aliases": []}
    return {
        "target_id": target.get("target_id"),
        "outcome": outcome,
        "accepted_data_families": list(target.get("accepted_data_families") or []),
        "preferred_effect_measures": list(target.get("preferred_effect_measures") or []),
        "estimands": list(target.get("estimands") or []),
        "effect_selection_rule": target.get("effect_selection_rule"),
        "data_contract": target.get("data_contract") or {},
        "quantitative_usability": target.get("quantitative_usability"),
        "extraction_notes": list(target.get("extraction_notes") or []),
        "candidate_capture_policy": target.get("candidate_capture_policy") or {},
        "quantitative_use_policy": target.get("quantitative_use_policy"),
    }


def build_controller_config(
    case: dict[str, Any],
    task: dict[str, Any],
    protocol: dict[str, Any],
    article_ids: list[str],
    study_ids: dict[str, str],
    models: dict[str, str],
    execution_policy: dict[str, Any],
) -> dict[str, Any]:
    pico = protocol.get("pico") or {}
    targets = [controller_target(target, pico) for target in protocol.get("targets") or []]
    if not targets:
        raise ValueError("Typed requires at least one normalized extraction target")
    outcomes = list(
        dict.fromkeys(
            label
            for target in targets
            for label in [text(target["outcome"]), *target["outcome"].get("aliases", [])]
            if label
        )
    )
    allowed_effect_types = list(
        dict.fromkeys(
            str(effect_type)
            for target in targets
            for effect_type in target["preferred_effect_measures"]
            if str(effect_type).strip()
        )
    )
    estimands = list(
        dict.fromkeys(
            str(estimand)
            for target in targets
            for estimand in target["estimands"]
            if str(estimand).strip()
        )
    )
    return {
        "schema_version": "meta_analysis_case/baseline",
        "case_id": case["case_id"],
        "task_id": task["task_id"],
        "task_type": "extraction",
        "topic": str(
            protocol.get("review_question")
            or protocol.get("record_policy")
            or case.get("review_question")
            or protocol.get("protocol_id")
            or "Source-grounded extraction"
        ),
        "pico": {
            "population": text(pico.get("population")),
            "exposure": text(pico.get("intervention_or_exposure") or pico.get("exposure")),
            "comparator": text(pico.get("comparator")),
            "outcomes": outcomes,
        },
        "inputs": {
            "document_index": "inputs/extraction_documents.json",
            "included_article_ids": article_ids,
            "study_ids": study_ids,
            "representations": ["markdown", "content_list"],
            "forbidden_document_ids": list(
                (task.get("allowed_inputs") or {}).get("forbidden_document_ids") or []
            ),
        },
        "extraction": {
            "allowed_effect_types": allowed_effect_types,
            "estimands": estimands,
            "estimand_policy": "preserve_source_reported",
            "targets": targets,
            "model_selection_policy": "Preserve every source-reported candidate; no study or effect selection occurs in extraction.",
            "exclusion_concepts": " ".join(
                note for target in targets for note in target["extraction_notes"]
            ),
            "evidence_requirements": (
                "Transcribe only source-supported data, preserve exact evidence, and do not "
                "calculate, pool, choose an analysis record, or infer missing values."
            ),
        },
        "workflow": {
            "stages": ["inspect", "static_clean", "locate", "extract", "organize"],
            "screening": "skip_because_included_studies_are_supplied",
            "extractor": "locator_extractor_tools",
            "planner": "disabled",
            "critic": "disabled",
            "candidate_review_structured": "disabled",
        },
        "system_partition": "locator_extractor_typed_data_family",
        "models": models,
        "execution_policy": execution_policy,
    }


def build_staging(
    case_dir: Path, staging_dir: Path, config: dict[str, Any]
) -> tuple[Path, list[str]]:
    case = read_json(case_dir / "case.json")
    task = read_json(extraction_task_path(case_dir))
    protocol = read_json(case_dir / "inputs" / "extraction_protocol.json")
    manifest = read_json(case_dir / "inputs" / "extraction_document_manifest.json")
    paths = read_json(case_dir / "inputs" / "extraction_document_paths.json")
    documents = document_records(manifest)
    document_paths = path_records(paths)
    article_ids = list((task.get("allowed_inputs") or {}).get("document_ids") or [])
    if not article_ids:
        article_ids = list((read_json(case_dir / "inputs" / "extraction_documents.json")).get("article_ids") or [])
    if not article_ids or len(article_ids) != len(set(article_ids)):
        raise ValueError("Typed case must declare a non-empty, unique article list")

    input_dir = staging_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=False)
    staged_manifest: dict[str, Any] = {}
    staged_paths: dict[str, Any] = {}
    study_ids: dict[str, str] = {}
    for article_id in article_ids:
        document = documents.get(article_id)
        declared_paths = document_paths.get(article_id)
        if document is None or declared_paths is None:
            raise ValueError(f"Typed source input is missing manifest/path data for {article_id}")
        markdown = resolve_representation(case_dir, declared_paths, "markdown")
        content_list = resolve_representation(case_dir, declared_paths, "content_list")
        study_id = str(document.get("study_id") or f"s_{article_id}")
        study_ids[article_id] = study_id
        staged_manifest[article_id] = {
            "document_id": str(document.get("document_id") or article_id),
            "article_id": article_id,
            "study_id": study_id,
            "role": "candidate_article",
            "access_policy": "task_input",
            "available_representations": ["markdown", "content_list"],
        }
        staged_paths[article_id] = {
            "markdown": str(markdown),
            "content_list": str(content_list),
        }

    write_json(input_dir / "extraction_document_manifest.json", {"case_id": case_dir.name, "documents": staged_manifest})
    write_json(input_dir / "extraction_document_paths.json", {"case_id": case_dir.name, "documents": staged_paths})
    write_json(
        input_dir / "extraction_documents.json",
        {
            "case_id": case_dir.name,
            "task_id": task.get("task_id"),
            "article_ids": article_ids,
            "manifest_file": "extraction_document_manifest.json",
            "paths_file": "extraction_document_paths.json",
        },
    )
    write_json(
        staging_dir / "case_config.json",
        build_controller_config(
            {**case, "case_id": case_dir.name},
            task,
            protocol,
            article_ids,
            study_ids,
            config["models"],
            config["execution_policy"],
        ),
    )
    return staging_dir / "case_config.json", article_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--case-id")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-staging", action="store_true")
    parser.add_argument("--max-retries", type=int)
    parser.add_argument("--retry-delay-seconds", type=float)
    parser.add_argument("--max-workers", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = read_json(args.config.resolve())
    source_root = project_path(config["source_root"])
    supported_cases = list(config["supported_cases"])
    if args.list_cases:
        for case_id in supported_cases:
            case_dir = source_root / "cases" / case_id
            task = read_json(extraction_task_path(case_dir))
            protocol = read_json(case_dir / "inputs" / "extraction_protocol.json")
            article_ids = list(
                (task.get("allowed_inputs") or {}).get("document_ids") or []
            )
            if not article_ids:
                article_ids = list(
                    (read_json(case_dir / "inputs" / "extraction_documents.json"))
                    .get("article_ids")
                    or []
                )
            families = sorted(
                {
                    str(family)
                    for target in protocol.get("targets") or []
                    for family in target.get("accepted_data_families") or []
                    if str(family).strip()
                }
            )
            print(
                f"{case_id}\tarticles={len(article_ids)}\t"
                f"families={','.join(families)}"
            )
        return 0
    case_id = args.case_id or config.get("default_case_id")
    if case_id not in supported_cases:
        raise ValueError(f"Unsupported typed case {case_id!r}. Use --list-cases.")
    case_dir = source_root / "cases" / case_id
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Typed input case is missing: {case_dir}")
    runner = project_path(config["controller_runner"])
    if not runner.is_file():
        raise FileNotFoundError(f"Typed runtime runner is missing: {runner}")
    if not args.dry_run and not config.get("execution_policy"):
        raise ValueError("Typed execution policy is missing")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    staging_dir = project_path(config["staging_root"]) / case_id / stamp
    config_path, article_ids = build_staging(case_dir, staging_dir, config)
    max_retries = args.max_retries if args.max_retries is not None else int(config["max_retries"])
    retry_delay = (
        args.retry_delay_seconds
        if args.retry_delay_seconds is not None
        else float(config["retry_delay_seconds"])
    )
    max_workers = args.max_workers if args.max_workers is not None else int(config["max_workers"])
    if max_retries < 0 or retry_delay < 0 or max_workers < 1:
        raise ValueError("Retry counts/delay and worker count must be non-negative/positive")
    command = [
        sys.executable,
        str(runner),
        "--case-config",
        str(config_path),
        "--runs-dir",
        str(project_path(config["runs_root"])),
        "--max-retries",
        str(max_retries),
        "--retry-delay-seconds",
        str(retry_delay),
        "--max-workers",
        str(max_workers),
    ]
    if args.dry_run:
        command.append("--dry-run")

    print(f"Case: {case_id}", flush=True)
    print("System version: locator_extractor_typed_data_family", flush=True)
    print("Workflow: static clean -> locate -> extract -> organize", flush=True)
    print("Planner: disabled", flush=True)
    print("Critic: disabled", flush=True)
    print("Candidate review structured: disabled", flush=True)
    print(f"Articles configured: {len(article_ids)}", flush=True)
    print(f"Structured source inputs: {case_dir}", flush=True)
    print(f"Typed runtime: {runner.parent}", flush=True)
    print(f"Staging inputs: {staging_dir}", flush=True)
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    keep_staging = args.keep_staging or bool(config.get("keep_staging"))
    if result.returncode == 0 and not keep_staging:
        shutil.rmtree(staging_dir, ignore_errors=True)
    else:
        print(f"Staging retained: {staging_dir}", flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
