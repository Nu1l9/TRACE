from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import LoadedTask
from .agent_tools import inspect_document, legacy_graph_extract
from .native_tools import (
    build_source_catalog,
    extract_effect_rows,
    locate_candidate_evidence,
    normalize_and_verify_effects,
    plan_extraction_targets,
)


def _progress(article_id: str, message: str) -> None:
    print(f"[{article_id}] {message}", flush=True)


def _legacy_submission(task: LoadedTask, article_id: str, result: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions = []
    issues = []
    for index, item in enumerate(result["structured_data"], start=1):
        # The legacy graph has already discarded HR vs OR in StructuredResult.
        # Preserve that limitation rather than infer a type in the adapter.
        predictions.append(
            {
                "prediction_id": f"legacy_{article_id}_{index:03d}",
                "article_id": article_id,
                "outcome": item.get("outcome"),
                "population_or_subgroup": item.get("row_label"),
                "estimand": None,
                "timepoint": None,
                "used_effect": {
                    "effect_type": None,
                    "value": item.get("effect_size"),
                    "ci_lower": item.get("ci_lower"),
                    "ci_upper": item.get("ci_upper"),
                },
                "source_location": {
                    "document_id": task.documents[article_id].document_id,
                    "page": None,
                    "source_type": "unknown",
                    "source_id": item.get("section_id"),
                    "locator_note": item.get("source_snippet"),
                },
                "adapter_status": "incomplete_legacy_projection",
            }
        )
        issues.append(
            {
                "prediction_id": f"legacy_{article_id}_{index:03d}",
                "missing_fields": ["effect_type", "estimand", "timepoint", "page", "source_type"],
                "reason": "The legacy StructuredResult does not preserve these fields.",
            }
        )
    return (
        {
            "case_id": task.case_id,
            "task_id": task.task_id,
            "task_type": "extraction",
            "submission_status": "not_scoreable_until_legacy_output_is_enriched",
            "predictions": predictions,
        },
        issues,
    )


def run_article(task: LoadedTask, article_id: str, run_dir: Path, dry_run: bool) -> dict[str, Any]:
    document = task.documents[article_id]
    run_dir.mkdir(parents=True, exist_ok=False)
    state: dict[str, Any] = {"stage": "prepare", "history": []}

    _progress(article_id, "inspect_document")
    inspection = inspect_document(document)
    state["history"].append({"action": "inspect_document", "status": "completed", "result": inspection})
    state["stage"] = "inspect"

    extractor_mode = task.raw["workflow"]["extractor"]
    manifest = {
        "case_id": task.case_id,
        "task_id": task.task_id,
        "article_id": article_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "mode": "dry-run" if dry_run else extractor_mode,
        "allowed_representations": task.allowed_representations,
        "included_article_ids": task.included_article_ids,
        "pico": task.pico,
        "extraction_contract": {
            "allowed_effect_types": task.allowed_effect_types,
            "estimands": task.estimands,
            "model_selection_policy": task.model_selection_policy,
            "exclusion_concepts": task.exclusion_concepts,
        },
        "controller_policy": "config-driven constrained stages; no Gold access",
        "inspection": inspection,
    }

    if dry_run:
        state["stage"] = "completed"
        return {"manifest": manifest, "controller_state": state}

    if extractor_mode == "legacy_graph_compatibility":
        state["stage"] = "extract"
        _progress(article_id, "legacy_graph_extract")
        result = legacy_graph_extract(document, task)
        state["history"].append(
            {"action": "legacy_graph_extract", "status": "completed", "extracted_rows": len(result["structured_data"])}
        )
        submission, adapter_issues = _legacy_submission(task, article_id, result)
        state["stage"] = "verify"
        state["history"].append(
            {"action": "project_legacy_output", "status": "completed", "adapter_issues": len(adapter_issues)}
        )
        state["stage"] = "completed"
        return {
            "manifest": manifest,
            "controller_state": state,
            "raw_system_output": result,
            "extraction_submission": submission,
            "adapter_issues": adapter_issues,
        }

    if extractor_mode != "native_tools":
        raise ValueError(f"Unsupported workflow extractor: {extractor_mode}")

    state["stage"] = "plan"
    _progress(article_id, "build_source_catalog")
    catalog = build_source_catalog(document, task)
    state["history"].append(
        {"action": "build_source_catalog", "status": "completed", "candidate_sources": len(catalog)}
    )
    if not catalog:
        raise ValueError(f"No candidate evidence sources found for {article_id}")

    _progress(article_id, "plan_extraction_targets")
    plan = plan_extraction_targets(task, catalog)
    located, locator_issues = locate_candidate_evidence(plan, catalog)
    state["history"].append(
        {
            "action": "plan_extraction_targets",
            "status": "completed" if plan.tasks else "no_tasks",
            "task_count": len(plan.tasks),
        }
    )
    state["stage"] = "locate"
    _progress(article_id, f"locate_candidate_evidence tasks={len(plan.tasks)}")
    state["history"].append(
        {
            "action": "locate_candidate_evidence",
            "status": "completed" if located else "needs_review",
            "located_tasks": len(located),
            "issues": locator_issues,
        }
    )

    state["stage"] = "extract"
    extractions = []
    extraction_issues: list[dict[str, Any]] = []
    max_extract_retries = task.execution_policy.get("max_article_retries", 1)
    for located_task in located:
        task_id = located_task["task"].task_id
        extracted = None
        extraction_feedback = ""
        for attempt in range(max_extract_retries + 1):
            _progress(article_id, f"extract_effect_rows task={task_id} attempt={attempt + 1}")
            extracted = extract_effect_rows(task, located_task, extraction_feedback)
            state["history"].append(
                {
                    "action": "extract_effect_rows",
                    "task_id": task_id,
                    "attempt": attempt + 1,
                    "status": "completed" if extracted.rows else "retry",
                    "row_count": len(extracted.rows),
                    "unresolved_items": extracted.unresolved_items,
                }
            )
            if extracted.rows:
                break
            extraction_feedback = "; ".join(extracted.unresolved_items) or "No effect rows were returned. Re-check the supplied evidence and task."
        if extracted is not None:
            extractions.append(extracted)
            if not extracted.rows:
                extraction_issues.append(
                    {"task_id": task_id, "issues": ["no_effect_rows_extracted", *extracted.unresolved_items]}
                )
            elif extracted.unresolved_items:
                extraction_issues.append(
                    {"task_id": task_id, "issues": ["partially_unresolved_extraction", *extracted.unresolved_items]}
                )

    state["stage"] = "verify"
    _progress(article_id, "normalize_and_verify_effects")
    predictions, verification_issues = normalize_and_verify_effects(task, extractions, catalog, article_id)
    all_issues = [*locator_issues, *extraction_issues, *verification_issues]
    if not predictions:
        all_issues.append({"stage": "verify", "issues": ["no_predictions_produced"]})
    state["history"].append(
        {
            "action": "normalize_and_verify_effects",
            "status": "completed",
            "prediction_count": len(predictions),
            "issue_count": len(all_issues),
        }
    )
    state["stage"] = "completed"
    _progress(article_id, f"completed predictions={len(predictions)} issues={len(all_issues)}")

    submission_status = "scoreable" if predictions and not all_issues else "requires_human_review"
    submission = {
        "case_id": task.case_id,
        "task_id": task.task_id,
        "task_type": "extraction",
        "submission_status": submission_status,
        "predictions": predictions,
    }
    raw_output = {
        "source_catalog": [
            {key: value for key, value in source.items() if key != "content"}
            for source in catalog
        ],
        "plan": plan.model_dump(),
        "extractions": [extraction.model_dump() for extraction in extractions],
    }
    return {
        "manifest": manifest,
        "controller_state": state,
        "raw_system_output": raw_output,
        "extraction_submission": submission,
        "issues": all_issues,
        "adapter_issues": all_issues,
    }
