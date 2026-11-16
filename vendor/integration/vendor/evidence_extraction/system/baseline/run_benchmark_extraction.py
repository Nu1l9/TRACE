"""Run a portable Meta-Analysis Bench extraction case with the constrained agent."""

from __future__ import annotations

import argparse
import json
import re
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
    / "portable_extraction_config.json"
)
CONTROLLER_RUNNER = PROJECT_ROOT / "constrained_meta_agent" / "run.py"
PAGE_HEADER = re.compile(r"(?m)^## PDF page\s+(\d+)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate, adapt, and run a portable benchmark extraction case."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--case-id")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-staging", action="store_true")
    parser.add_argument("--max-retries", type=int)
    parser.add_argument("--retry-delay-seconds", type=float)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def resolve_project_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = value.strip()
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
    return result


def list_cases(portable_root: Path) -> int:
    for case_dir in sorted((portable_root / "cases").glob("ma_*")):
        case_file = case_dir / "case.json"
        if not case_file.is_file():
            continue
        case = load_json(case_file)
        print(
            f"{case['case_id']}\t{case.get('case_status', 'runnable')}\t"
            f"{case.get('title', '')}"
        )
    return 0


def validate_case(portable_root: Path, case_dir: Path) -> None:
    runner = portable_root / "runner.py"
    result = subprocess.run(
        [
            sys.executable,
            str(runner),
            "validate",
            str(case_dir),
            "--check-local-paths",
        ],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Portable benchmark validation failed for {case_dir.name}")


def find_markdown(case_dir: Path, article_id: str, search_roots: list[str]) -> Path:
    for raw_root in search_roots:
        root = case_dir / raw_root / article_id
        if not root.is_dir():
            continue
        exact = root / "full.md"
        if exact.is_file():
            return exact
        candidates = sorted(root.glob("*.md"))
        if candidates:
            return candidates[0]
        nested = sorted(root.rglob("full.md"))
        if nested:
            return nested[0]
    raise FileNotFoundError(
        f"No prepared Markdown found for {article_id} under {case_dir}"
    )


def chunk_text(text: str, size: int) -> list[str]:
    if size < 1000:
        raise ValueError("chunk_chars must be at least 1000")
    return [text[start : start + size] for start in range(0, len(text), size)] or [""]


def markdown_to_content_list(text: str, chunk_chars: int) -> list[dict[str, Any]]:
    matches = list(PAGE_HEADER.finditer(text))
    if not matches:
        return [
            {"type": "text", "text": chunk, "page_idx": None, "bbox": None}
            for chunk in chunk_text(text, chunk_chars)
        ]

    items: list[dict[str, Any]] = []
    preamble = text[: matches[0].start()].strip()
    for index, match in enumerate(matches):
        page = int(match.group(1))
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        if index == 0 and preamble:
            body = f"{preamble}\n\n{body}"
        for chunk in chunk_text(body, chunk_chars):
            items.append(
                {
                    "type": "text",
                    "text": chunk,
                    "page_idx": page - 1,
                    "bbox": None,
                }
            )
    return items


def quantitative_usability(policy: str | None) -> str:
    normalized = (policy or "").casefold()
    if "source_only" in normalized:
        return "source_only"
    if "needs_extraction" in normalized:
        return "needs_extraction"
    return "poolable"


def controller_targets(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    targets = []
    for target in protocol["targets"]:
        targets.append(
            {
                "target_id": target["target_id"],
                "outcome": target["outcome"],
                "accepted_data_families": target.get("accepted_data_families", []),
                "preferred_effect_measures": target.get("preferred_effect_measures", []),
                "estimands": target.get("estimands", []),
                "effect_selection_rule": target.get("effect_selection_rule"),
                "data_contract": target.get("data_contract"),
                "quantitative_use_policy": target.get("quantitative_use_policy"),
                "quantitative_usability": quantitative_usability(
                    target.get("quantitative_use_policy")
                ),
                "extraction_notes": target.get("extraction_notes", []),
            }
        )
    return targets


def build_controller_config(
    case: dict[str, Any],
    task: dict[str, Any],
    protocol: dict[str, Any],
    article_ids: list[str],
    models: dict[str, str],
    execution_policy: dict[str, Any],
    pico_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    targets = controller_targets(protocol)
    outcomes = unique_strings(
        [
            label
            for target in protocol["targets"]
            for label in [
                target["outcome"]["description"],
                *target["outcome"].get("aliases", []),
            ]
        ]
    )
    effect_types = unique_strings(
        [
            effect_type
            for target in protocol["targets"]
            for effect_type in target.get("preferred_effect_measures", [])
        ]
    )
    estimands = unique_strings(
        [
            estimand
            for target in protocol["targets"]
            for estimand in target.get("estimands", [])
        ]
    )
    if not effect_types or not estimands:
        raise ValueError("Extraction protocol must declare effect measures and estimands")

    selection_rules = unique_strings(
        [
            target.get("effect_selection_rule") or ""
            for target in protocol["targets"]
        ]
    )
    extraction_notes = unique_strings(
        [
            note
            for target in protocol["targets"]
            for note in target.get("extraction_notes", [])
        ]
    )
    pico = protocol["pico"]
    effective_pico_source = pico_override or {
        "population": pico["population"]["description"],
        "exposure": pico["intervention_or_exposure"]["description"],
        "comparator": pico["comparator"]["description"],
        "outcomes": outcomes,
    }
    required_pico_fields = {"population", "exposure", "comparator", "outcomes"}
    missing_pico_fields = required_pico_fields - set(effective_pico_source)
    if missing_pico_fields:
        raise ValueError(f"pico_override is missing fields: {sorted(missing_pico_fields)}")
    effective_pico = {
        field: effective_pico_source[field]
        for field in ("population", "exposure", "comparator", "outcomes")
    }
    return {
        "schema_version": "meta_analysis_case/baseline",
        "case_id": case["case_id"],
        "task_id": task["task_id"],
        "task_type": "extraction",
        "topic": (pico_override or {}).get("topic", protocol["review_question"]),
        "pico": effective_pico,
        "inputs": {
            "document_index": "inputs/extraction_documents.json",
            "included_article_ids": article_ids,
            "representations": ["markdown", "content_list"],
            "forbidden_document_ids": task.get("allowed_inputs", {}).get(
                "forbidden_document_ids", []
            ),
        },
        "extraction": {
            "allowed_effect_types": effect_types,
            "estimands": estimands,
            "targets": targets,
            "model_selection_policy": " ".join(selection_rules),
            "exclusion_concepts": " ".join(extraction_notes),
            "evidence_requirements": (
                "Every extracted record must retain exact source evidence, page, "
                "source-defined effect scale, exposure unit, adjustment set, and missingness."
            ),
        },
        "workflow": {
            "stages": ["inspect", "plan", "locate", "extract", "verify"],
            "screening": "skip_because_included_studies_are_supplied",
            "extractor": "native_tools",
        },
        "models": models,
        "execution_policy": execution_policy,
    }


def build_staging(
    case_dir: Path,
    staging_dir: Path,
    adapter_config: dict[str, Any],
) -> tuple[Path, list[str]]:
    case = load_json(case_dir / "case.json")
    task = load_json(case_dir / "tasks" / "extraction_all_reports.json")
    protocol = load_json(case_dir / "inputs" / "extraction_protocol.json")
    document_manifest = load_json(
        case_dir / "inputs" / "extraction_document_manifest.json"
    )
    documents_by_id = {
        document["article_id"]: document
        for document in document_manifest["documents"]
    }
    article_ids = task["allowed_inputs"]["document_ids"]
    if not article_ids or len(article_ids) != len(set(article_ids)):
        raise ValueError("Extraction task requires unique document IDs")
    missing = sorted(set(article_ids) - set(documents_by_id))
    if missing:
        raise ValueError(f"Task documents missing from manifest: {missing}")

    input_dir = staging_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=False)
    staged_manifest: dict[str, Any] = {}
    staged_paths: dict[str, Any] = {}
    for article_id in article_ids:
        document = documents_by_id[article_id]
        markdown = find_markdown(
            case_dir,
            article_id,
            adapter_config["prepared_markdown_search_roots"],
        )
        content_list_path = input_dir / f"{article_id}_content_list.json"
        content_list = markdown_to_content_list(
            markdown.read_text(encoding="utf-8"),
            int(adapter_config["chunk_chars"]),
        )
        write_json(content_list_path, content_list)
        staged_manifest[article_id] = {
            "document_id": document["document_id"],
            "article_id": article_id,
            "study_id": document.get("study_id"),
            "role": "included_candidate_article",
            "access_policy": "allowed_system_input",
            "available_representations": ["markdown", "content_list"],
        }
        staged_paths[article_id] = {
            "markdown": str(markdown.resolve()),
            "content_list": str(content_list_path.resolve()),
        }

    write_json(
        input_dir / "extraction_document_manifest.json",
        {"case_id": case["case_id"], "documents": staged_manifest},
    )
    write_json(
        input_dir / "extraction_document_paths.json",
        {"case_id": case["case_id"], "documents": staged_paths},
    )
    write_json(
        input_dir / "extraction_documents.json",
        {
            "case_id": case["case_id"],
            "task_id": task["task_id"],
            "article_ids": article_ids,
            "manifest_file": "extraction_document_manifest.json",
            "paths_file": "extraction_document_paths.json",
        },
    )
    controller_config = build_controller_config(
        case,
        task,
        protocol,
        article_ids,
        adapter_config["models"],
        adapter_config["execution_policy"],
        adapter_config.get("pico_override"),
    )
    config_path = staging_dir / "case_config.json"
    write_json(config_path, controller_config)
    return config_path, article_ids


def main() -> int:
    args = parse_args()
    adapter_config = load_json(args.config.resolve())
    portable_root = resolve_project_path(adapter_config["portable_root"])
    if args.list_cases:
        return list_cases(portable_root)

    case_id = args.case_id or adapter_config.get("default_case_id")
    if not case_id:
        raise ValueError("Specify --case-id or set default_case_id in the config")
    case_dir = portable_root / "cases" / case_id
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Unknown portable benchmark case: {case_dir}")
    if not CONTROLLER_RUNNER.is_file():
        raise FileNotFoundError(f"Controller runner not found: {CONTROLLER_RUNNER}")

    validate_case(portable_root, case_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    staging_dir = (
        resolve_project_path(adapter_config["staging_root"]) / case_id / stamp
    )
    config_path, article_ids = build_staging(
        case_dir,
        staging_dir,
        adapter_config,
    )
    runs_dir = resolve_project_path(adapter_config["runs_root"])
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
    print(f"Mode: {'dry-run' if args.dry_run else 'full extraction'}", flush=True)
    print(f"Portable case: {case_dir}", flush=True)
    print(f"Staging inputs: {staging_dir}", flush=True)
    print(f"Runs root: {runs_dir}", flush=True)
    print(f"Configured-provider retries per article: {max_retries} plus initial attempt", flush=True)
    print("Requests remain on the configured provider after retries.", flush=True)
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    keep_staging = args.keep_staging or bool(adapter_config.get("keep_staging"))
    if result.returncode == 0 and not keep_staging:
        shutil.rmtree(staging_dir, ignore_errors=True)
    elif keep_staging:
        print(f"Staging retained: {staging_dir}", flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
