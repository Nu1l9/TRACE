from __future__ import annotations

import argparse
from copy import deepcopy
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


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
LEGACY_BUILD_CONTROLLER_CONFIG = portable_adapter.build_controller_config

STRUCTURED_RAW_DATA_FAMILIES = {
    "binary_arm_outcome",
    "binary_arm_event_counts",
    "two_group_event_counts",
    "binary_exposure_outcome_counts",
    "diagnostic_four_cell_counts",
    "single_arm_event_counts",
    "independent_continuous_arms",
    "single_arm_continuous_summary",
}


def _pico_text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("description") or value.get("label") or "")
    return str(value or "")


def _structured_source_data_targets(case_id: str) -> list[dict[str, Any]]:
    if case_id == "ma_0010":
        return [
            {
                "target_id": "rs763780_source_distribution",
                "outcome": {
                    "description": "Rheumatoid arthritis case status",
                    "aliases": ["RA case status"],
                },
                "accepted_data_families": [
                    "complete_genotype_case_control_counts",
                    "allele_frequency_summary_only",
                ],
                "preferred_effect_measures": [],
                "estimands": [],
                "data_contract": {
                    "template": "rs763780 genotype counts or allele-frequency summary",
                    "calculation": "forbidden_in_extraction",
                },
                "extraction_notes": [
                    "Copy source genotype counts without selecting a contrast or calculating an odds ratio.",
                    "Keep allele-frequency-only reports separate; never infer genotype counts.",
                ],
            }
        ]
    if case_id == "ma_0041":
        return [
            {
                "target_id": "glenoid_positioning_summary",
                "outcome": {
                    "description": "Source-reported glenoid positioning summary",
                    "aliases": ["glenoid version", "glenoid inclination", "glenoid offset"],
                },
                "accepted_data_families": [
                    "independent_continuous_arms",
                    "single_arm_continuous_summary",
                ],
                "preferred_effect_measures": [],
                "estimands": [],
                "data_contract": {
                    "template": "glenoid positioning continuous summaries",
                    "calculation": "forbidden_in_extraction",
                },
                "extraction_notes": [
                    "Preserve outcome axis, measurement semantics, unit, arm labels, and signed values.",
                    "Do not create a comparator for single-arm summaries or calculate MD/SMD.",
                ],
            }
        ]
    raise ValueError(f"No source-data target adapter is defined for {case_id}")


def _normalize_structured_protocol(protocol: dict[str, Any], case_id: str) -> dict[str, Any]:
    """Normalize the structured benchmark contract to the unchanged locator runtime shape."""
    normalized = deepcopy(protocol)
    pico = normalized.get("pico") or {}
    normalized["review_question"] = str(
        normalized.get("review_question")
        or normalized.get("record_policy")
        or normalized.get("protocol_id")
        or "Source-grounded extraction"
    )
    normalized["pico"] = {
        "population": {"description": _pico_text(pico.get("population"))},
        "intervention_or_exposure": {
            "description": _pico_text(
                pico.get("intervention_or_exposure") or pico.get("exposure")
            )
        },
        "comparator": {"description": _pico_text(pico.get("comparator"))},
    }
    targets = list(normalized.get("targets") or [])
    if case_id in {"ma_0010", "ma_0041"}:
        targets = _structured_source_data_targets(case_id)
    if not targets:
        raise ValueError(f"No extraction targets found for {case_id}")
    for target in targets:
        outcome = target.get("outcome")
        if isinstance(outcome, str):
            target["outcome"] = {"description": outcome, "aliases": []}
        target.setdefault("outcome", {"description": normalized["review_question"], "aliases": []})
        target.setdefault("accepted_data_families", [])
        target.setdefault("preferred_effect_measures", [])
        target.setdefault("estimands", [])
        target.setdefault("extraction_notes", [])
    normalized["targets"] = targets
    return normalized


def _find_structured_prepared_file(
    prepared_root: Path, case_id: str, article_id: str, pattern: str
) -> Path:
    result_dir = prepared_root / case_id / article_id / "result"
    candidates = sorted(result_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"MinerU cached {pattern} is missing for {case_id}/{article_id}: {result_dir}"
        )
    return candidates[0]


def _build_structured_staging(
    case_dir: Path,
    staging_dir: Path,
    adapter_config: dict[str, Any],
) -> tuple[Path, list[str]]:
    """Stage a structured portable case for the pre-existing locator-extractor runner."""
    case = portable_adapter.load_json(case_dir / "case.json")
    task_path = case_dir / "tasks" / "extraction_all_reports.json"
    if not task_path.is_file():
        task_path = case_dir / "tasks" / "extraction_all_archived_reports.json"
    task = portable_adapter.load_json(task_path)
    protocol = _normalize_structured_protocol(
        portable_adapter.load_json(case_dir / "inputs" / "extraction_protocol.json"),
        case_dir.name,
    )
    manifest_path = case_dir / "inputs" / "extraction_document_manifest.json"
    if not manifest_path.is_file():
        manifest_path = case_dir / "documents" / "manifest.json"
    manifest = portable_adapter.load_json(manifest_path)
    raw_documents = manifest.get("documents") or []
    if isinstance(raw_documents, dict):
        raw_documents = [
            {"document_id": document_id, **document}
            for document_id, document in raw_documents.items()
        ]
    documents_by_id = {
        document.get("article_id") or document.get("document_id"): document
        for document in raw_documents
        if document.get("article_id") or document.get("document_id")
    }
    article_ids = list(task.get("allowed_inputs", {}).get("document_ids") or [])
    if not article_ids:
        forbidden = set(task.get("allowed_inputs", {}).get("forbidden_document_ids") or [])
        article_ids = [
            article_id
            for article_id, document in documents_by_id.items()
            if article_id not in forbidden and not document.get("gold_only")
        ]
    missing = sorted(set(article_ids) - set(documents_by_id))
    if missing:
        raise ValueError(f"Task documents missing from manifest: {missing}")
    prepared_root = portable_adapter.resolve_project_path(adapter_config["prepared_root"])
    input_dir = staging_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=False)
    staged_manifest: dict[str, Any] = {}
    staged_paths: dict[str, Any] = {}
    for article_id in article_ids:
        document = documents_by_id[article_id]
        markdown = _find_structured_prepared_file(
            prepared_root, case_dir.name, article_id, "full.md"
        )
        try:
            # The original MinerU list is already flat and matches the old
            # locator's source-item contract. The structured list is page-nested.
            source_content_list = _find_structured_prepared_file(
                prepared_root, case_dir.name, article_id, "*_content_list.json"
            )
        except FileNotFoundError:
            source_content_list = _find_structured_prepared_file(
                prepared_root, case_dir.name, article_id, "*_content_list_v2.json"
            )
        content_list = portable_adapter.load_json(source_content_list)
        if not isinstance(content_list, list):
            raise ValueError(f"MinerU content list is not an array: {source_content_list}")
        staged_content_list = input_dir / f"{article_id}_content_list.json"
        portable_adapter.write_json(staged_content_list, content_list)
        staged_manifest[article_id] = {
            "document_id": document.get("document_id") or article_id,
            "article_id": article_id,
            "study_id": document.get("study_id") or article_id,
            "role": "included_candidate_article",
            "access_policy": "allowed_system_input",
            "available_representations": ["markdown", "content_list"],
        }
        staged_paths[article_id] = {
            "markdown": str(markdown.resolve()),
            "content_list": str(staged_content_list.resolve()),
        }
    portable_adapter.write_json(
        input_dir / "extraction_document_manifest.json",
        {"case_id": case_dir.name, "documents": staged_manifest},
    )
    portable_adapter.write_json(
        input_dir / "extraction_document_paths.json",
        {"case_id": case_dir.name, "documents": staged_paths},
    )
    portable_adapter.write_json(
        input_dir / "extraction_documents.json",
        {
            "case_id": case_dir.name,
            "task_id": task.get("task_id"),
            "article_ids": article_ids,
            "manifest_file": "extraction_document_manifest.json",
            "paths_file": "extraction_document_paths.json",
        },
    )
    controller_config = _build_controller_config(
        {**case, "case_id": case_dir.name},
        task,
        protocol,
        article_ids,
        adapter_config["models"],
        adapter_config["execution_policy"],
    )
    controller_config["inputs"]["study_ids"] = {
        article_id: staged_manifest[article_id]["study_id"] for article_id in article_ids
    }
    controller_config["workflow"] = {
        "stages": ["inspect", "static_clean", "locate", "extract", "organize"],
        "screening": "skip_because_included_studies_are_supplied",
        "extractor": "locator_extractor_tools",
        "planner": "disabled",
        "critic": "disabled",
    }
    controller_config["system_partition"] = "locator_extractor_structured_23"
    for target in controller_config["extraction"]["targets"]:
        if any(
            family in STRUCTURED_RAW_DATA_FAMILIES
            for family in target.get("accepted_data_families", [])
        ):
            target["quantitative_usability"] = None
    config_path = staging_dir / "case_config.json"
    portable_adapter.write_json(config_path, controller_config)
    return config_path, article_ids


def _build_controller_config(
    case: dict,
    task: dict,
    protocol: dict,
    article_ids: list[str],
    models: dict[str, str],
    execution_policy: dict,
    pico_override: dict | None = None,
) -> dict:
    """Allow extraction-only protocols to preserve source-defined estimands."""
    raw_families = {
        family
        for target in protocol.get("targets", [])
        for family in target.get("accepted_data_families", [])
    }
    source_only_families = {
        "complete_genotype_case_control_counts",
        "allele_frequency_summary_only",
        "independent_continuous_arms",
        "single_arm_continuous_summary",
    }
    if raw_families and raw_families.issubset(source_only_families):
        pico = protocol.get("pico") or {}
        outcomes = [
            target.get("outcome", {}).get("description", "")
            for target in protocol.get("targets", [])
        ]
        normalized_targets = []
        for target in protocol.get("targets", []):
            normalized_targets.append(
                {
                    "target_id": target.get("target_id"),
                    "outcome": target.get("outcome"),
                    "accepted_data_families": target.get("accepted_data_families", []),
                    "preferred_effect_measures": [],
                    "estimands": [],
                    "effect_selection_rule": None,
                    "data_contract": target.get("data_contract"),
                    "quantitative_usability": None,
                    "extraction_notes": target.get("extraction_notes", []),
                }
            )
        return {
            "schema_version": "meta_analysis_case/baseline",
            "case_id": case["case_id"],
            "task_id": task["task_id"],
            "task_type": "extraction",
            "topic": protocol.get("review_question", "Source-grounded extraction"),
            "pico": {
                "population": _pico_text(pico.get("population")),
                "exposure": _pico_text(
                    pico.get("intervention_or_exposure") or pico.get("exposure")
                ),
                "comparator": _pico_text(pico.get("comparator")),
                "outcomes": [outcome for outcome in outcomes if outcome],
            },
            "inputs": {
                "document_index": "inputs/extraction_documents.json",
                "included_article_ids": article_ids,
                "representations": ["markdown", "content_list"],
                "forbidden_document_ids": task.get("allowed_inputs", {}).get(
                    "forbidden_document_ids", []
                ),
            },
            "extraction": {
                "allowed_effect_types": [],
                "estimands": [],
                "targets": normalized_targets,
                "model_selection_policy": "No effect selection in source-data extraction.",
                "exclusion_concepts": " ".join(
                    note
                    for target in protocol.get("targets", [])
                    for note in target.get("extraction_notes", [])
                ),
                "evidence_requirements": (
                    "Every row must preserve exact source evidence and raw source data. "
                    "Do not calculate, select, or convert an effect."
                ),
                "estimand_policy": "preserve_source_reported",
            },
            "workflow": {
                "stages": ["inspect", "static_clean", "locate", "extract", "organize"],
                "screening": "skip_because_included_studies_are_supplied",
                "extractor": "locator_extractor_tools",
                "planner": "disabled",
                "critic": "disabled",
            },
            "models": models,
            "execution_policy": execution_policy,
        }
    declared_estimands = [
        estimand
        for target in protocol.get("targets", [])
        for estimand in target.get("estimands", [])
        if str(estimand).strip()
    ]
    if declared_estimands:
        return LEGACY_BUILD_CONTROLLER_CONFIG(
            case,
            task,
            protocol,
            article_ids,
            models,
            execution_policy,
            pico_override,
        )

    compatible_protocol = deepcopy(protocol)
    for target in compatible_protocol.get("targets", []):
        target["estimands"] = ["source_reported"]
    config = LEGACY_BUILD_CONTROLLER_CONFIG(
        case,
        task,
        compatible_protocol,
        article_ids,
        models,
        execution_policy,
        pico_override,
    )
    config["extraction"]["estimands"] = []
    config["extraction"]["estimand_policy"] = "preserve_source_reported"
    for original, staged in zip(
        protocol.get("targets", []), config["extraction"].get("targets", [])
    ):
        staged["estimands"] = original.get("estimands", [])
    return config


portable_adapter.build_controller_config = _build_controller_config


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


def _find_manifest_markdown_strict(
    case_dir: Path, article_id: str, search_roots: list[str]
) -> Path:
    """Require the complete Markdown explicitly declared by the benchmark."""
    del search_roots
    markdown = _manifest_markdown(case_dir, article_id)
    if markdown is None:
        raise FileNotFoundError(
            f"No manifest Markdown is declared for {article_id} under {case_dir}"
        )
    return markdown


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
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--review-structured", action="store_true")
    parser.add_argument("--review-structured-batch-size", type=int, default=12)
    parser.add_argument("--review-structured-workers", type=int, default=4)
    parser.add_argument("--review-structured-max-retries", type=int, default=3)
    parser.add_argument("--review-structured-retry-delay-seconds", type=float, default=5.0)
    parser.add_argument(
        "--review-structured-provider",
        choices=("official_openai", "openai_compatible"),
        help="Provider for the attached post-extraction review only.",
    )
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

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    staging_dir = (
        portable_adapter.resolve_project_path(adapter_config["staging_root"])
        / case_id
        / stamp
    )
    input_format = adapter_config.get("input_format", "portable_baseline")
    if input_format == "portable_structured_mineru":
        input_source_policy = "mineru_cached_content_list"
        config_path, article_ids = _build_structured_staging(
            case_dir, staging_dir, adapter_config
        )
    elif input_format == "portable_baseline":
        portable_adapter.validate_case(portable_root, case_dir)
        input_source_policy = adapter_config.get(
            "input_source_policy", "manifest_or_search"
        )
        if input_source_policy == "manifest_complete_markdown":
            portable_adapter.find_markdown = _find_manifest_markdown_strict
        elif input_source_policy == "manifest_or_search":
            portable_adapter.find_markdown = _find_markdown
        else:
            raise ValueError(f"Unsupported input_source_policy: {input_source_policy}")
        config_path, article_ids = portable_adapter.build_staging(
            case_dir, staging_dir, adapter_config
        )
        _finalize_staging_config(config_path)
    else:
        raise ValueError(f"Unsupported input_format: {input_format}")
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
    max_workers = (
        args.max_workers
        if args.max_workers is not None
        else int(adapter_config.get("max_workers", 4))
    )
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if args.review_structured_batch_size < 2:
        raise ValueError("review-structured-batch-size must be at least 2")
    if args.review_structured_workers < 1:
        raise ValueError("review-structured-workers must be at least 1")
    if args.review_structured_max_retries < 0 or args.review_structured_retry_delay_seconds < 0:
        raise ValueError("review-structured retry counts and delays must be non-negative")
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
        "--max-workers",
        str(max_workers),
    ]
    if args.review_structured:
        command.extend(
            [
                "--review-structured",
                "--review-structured-batch-size",
                str(args.review_structured_batch_size),
                "--review-structured-workers",
                str(args.review_structured_workers),
                "--review-structured-max-retries",
                str(args.review_structured_max_retries),
                "--review-structured-retry-delay-seconds",
                str(args.review_structured_retry_delay_seconds),
            ]
        )
        if args.review_structured_provider:
            command.extend(["--review-structured-provider", args.review_structured_provider])
    if args.dry_run:
        command.append("--dry-run")

    print(f"Case: {case_id}", flush=True)
    print(f"Articles configured: {len(article_ids)}", flush=True)
    print("Workflow: static clean -> locate -> extract -> organize", flush=True)
    print("Planner: disabled", flush=True)
    print("Critic: disabled", flush=True)
    print(f"Input source policy: {input_source_policy}", flush=True)
    print(f"Staging inputs: {staging_dir}", flush=True)
    print(f"Runs root: {runs_dir}", flush=True)
    print(f"Article workers: {max_workers}", flush=True)
    if args.review_structured:
        print(
            "Candidate review structured: enabled "
            f"(batch size {args.review_structured_batch_size}, workers {args.review_structured_workers}, "
            f"retries {args.review_structured_max_retries}, "
            f"provider {args.review_structured_provider or 'same as extraction'})",
            flush=True,
        )
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)

    keep_staging = args.keep_staging or bool(adapter_config.get("keep_staging"))
    if result.returncode == 0 and not keep_staging:
        shutil.rmtree(staging_dir, ignore_errors=True)
    elif keep_staging:
        print(f"Staging retained: {staging_dir}", flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
