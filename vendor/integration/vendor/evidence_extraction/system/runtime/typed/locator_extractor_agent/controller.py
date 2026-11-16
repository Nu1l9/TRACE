from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from constrained_meta_agent.agent_tools import inspect_document
from constrained_meta_agent.models import LoadedTask

from .tools import (
    extract_from_locations,
    locate_relevant_sources,
    organize_extractions,
    static_clean_document,
)


def _progress(article_id: str, message: str) -> None:
    print(f"[{article_id}] {message}", flush=True)


def run_article(
    task: LoadedTask,
    article_id: str,
    run_dir: Path,
    dry_run: bool,
    provider: str = "official_openai",
) -> dict[str, Any]:
    document = task.documents[article_id]
    run_dir.mkdir(parents=True, exist_ok=False)
    state: dict[str, Any] = {"stage": "prepare", "history": []}

    _progress(article_id, "inspect_document")
    inspection = inspect_document(document)
    state["history"].append(
        {"action": "inspect_document", "status": "completed", "result": inspection}
    )

    manifest = {
        "case_id": task.case_id,
        "task_id": task.task_id,
        "article_id": article_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "mode": "dry-run" if dry_run else "locator_extractor_tools",
        "workflow": [
            "static_clean_document",
            "locate_relevant_sources",
            "extract_from_locations",
            "organize_extractions",
        ],
        "controller_policy": (
            "static cleanup, high-recall location, direct extraction, deterministic "
            "organization; no planner, no critic, no Gold access"
        ),
        "allowed_representations": task.allowed_representations,
        "included_article_ids": task.included_article_ids,
        "pico": task.pico,
        "extraction_contract": {
            "allowed_effect_types": task.allowed_effect_types,
            "estimands": task.estimands,
            "targets": task.raw.get("extraction", {}).get("targets", []),
            "model_selection_policy": task.model_selection_policy,
            "exclusion_concepts": task.exclusion_concepts,
        },
        "inspection": inspection,
    }

    if dry_run:
        state["stage"] = "completed"
        return {"manifest": manifest, "controller_state": state}

    state["stage"] = "clean"
    _progress(article_id, "static_clean_document")
    catalog, cleanup_audit = static_clean_document(
        document,
        remove_reference_items_only=(
            task.raw.get("system_partition")
            in {"locator_extractor_structured_23", "locator_extractor_typed_data_family"}
        ),
    )
    state["history"].append(
        {
            "action": "static_clean_document",
            "status": "completed",
            **cleanup_audit,
        }
    )
    if not catalog:
        raise ValueError(f"Static cleanup retained no source regions for {article_id}")

    state["stage"] = "locate"
    _progress(article_id, "locate_relevant_sources")
    locations, located_sources, locator_issues = locate_relevant_sources(
        task, catalog, provider
    )
    state["history"].append(
        {
            "action": "locate_relevant_sources",
            "status": "completed" if located_sources else "needs_review",
            "source_ids": locations.source_ids,
            "source_count": len(located_sources),
            "unresolved_location_terms": locations.unresolved_location_terms,
        }
    )

    state["stage"] = "extract"
    extraction = None
    extraction_issues: list[dict[str, Any]] = []
    if located_sources:
        feedback = ""
        max_retries = int(task.execution_policy.get("max_article_retries", 1))
        for attempt in range(max_retries + 1):
            _progress(article_id, f"extract_from_locations attempt={attempt + 1}")
            extraction = extract_from_locations(
                task, located_sources, feedback, provider
            )
            state["history"].append(
                {
                    "action": "extract_from_locations",
                    "attempt": attempt + 1,
                    "status": "completed" if extraction.rows else "retry",
                    "row_count": len(extraction.rows),
                    "unresolved_items": extraction.unresolved_items,
                }
            )
            if extraction.rows:
                break
            feedback = "; ".join(extraction.unresolved_items) or (
                "No rows were returned. Reinspect every located source and transcribe "
                "all source-supported candidate data without making eligibility decisions."
            )
        if extraction is not None and not extraction.rows:
            extraction_issues.append(
                {
                    "stage": "extract",
                    "issues": ["no_effect_rows_extracted", *extraction.unresolved_items],
                }
            )
        elif extraction is not None and extraction.unresolved_items:
            extraction_issues.append(
                {
                    "stage": "extract",
                    "issues": ["partially_unresolved_extraction", *extraction.unresolved_items],
                }
            )

    state["stage"] = "organize"
    _progress(article_id, "organize_extractions")
    if extraction is None:
        predictions: list[dict[str, Any]] = []
        verification_issues: list[dict[str, Any]] = []
    else:
        predictions, verification_issues = organize_extractions(
            task, extraction, catalog, article_id
        )
    all_issues = [*locator_issues, *extraction_issues, *verification_issues]
    if not predictions:
        all_issues.append({"stage": "organize", "issues": ["no_predictions_produced"]})
    state["history"].append(
        {
            "action": "organize_extractions",
            "status": "completed",
            "prediction_count": len(predictions),
            "issue_count": len(all_issues),
        }
    )
    state["stage"] = "completed"
    _progress(
        article_id,
        f"completed predictions={len(predictions)} issues={len(all_issues)}",
    )

    submission = {
        "case_id": task.case_id,
        "task_id": task.task_id,
        "task_type": "extraction",
        "submission_status": (
            "scoreable" if predictions and not all_issues else "requires_human_review"
        ),
        "predictions": predictions,
    }
    raw_output = {
        "cleanup_audit": cleanup_audit,
        "source_catalog": [
            {key: value for key, value in source.items() if key != "content"}
            for source in catalog
        ],
        "locations": locations.model_dump(),
        "located_sources": [source["source_id"] for source in located_sources],
        "extraction": extraction.model_dump() if extraction is not None else None,
    }
    return {
        "manifest": manifest,
        "controller_state": state,
        "raw_system_output": raw_output,
        "extraction_submission": submission,
        "issues": all_issues,
    }
