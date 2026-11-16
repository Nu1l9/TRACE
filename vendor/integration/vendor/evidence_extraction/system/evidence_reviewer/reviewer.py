from __future__ import annotations

import copy
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from constrained_meta_agent.models import LoadedTask
from llm_endpoint import chat_model

from .models import CandidateReviewBatch


REVIEW_SYSTEM_PROMPT = """You are the read-only candidate evidence review stage of extraction workflow structured. Review each supplied candidate against the PICO, explicit extraction requirements, and its source quote.

Your task is to place evidence in the correct layer of the evidence pool. You do not choose a preferred estimate and you do not modify extraction output.

Your boundary is strict:
- Do not calculate, convert, reconstruct, correct, rank, select, or merge any result.
- Check the supplied raw_data numbers against source evidence, including the arm, denominator, outcome and timepoint. Empty used_effect fields are valid when analysis must later calculate from source counts or means.
- Never treat a valid schema or an extraction usability hint as proof of numeric support. Percentages or Kaplan-Meier estimates alone do not support reconstructed integer event counts. If essential numeric fields lack source support, assign source_only or uncertain and identify the unsupported fields in the reason.
- Do not repeat an effect value, CI, count, percentage, model, or source quote in your response.
- Do not use Gold, review selections, or pooled results.
- A candidate not eligible for the main analysis is not automatically invalid. Preserve source-grounded, task-relevant evidence for later analysis.

Assign exactly one evidence_role:
- `main_analysis_candidate`: source support and stated requirements make it eligible for a later main-analysis assessment. This is eligibility only, not a final selection.
- `supplementary_or_sensitivity`: source-grounded and within the task's non-relaxable PICO boundaries, but better treated as a supplementary, sensitivity, subgroup, alternate, overlapping, or otherwise non-primary candidate. It may differ in analytical role, but not in the core task object.
- `source_only`: source-grounded and within the task's non-relaxable PICO boundaries, but it supplies contextual or descriptive evidence rather than a usable quantitative effect for the target analysis.
- `uncertain`: missing, conflicting, or insufficient evidence prevents a defensible role assignment.
- `none`: the source and task contract establish an irreconcilable contradiction with a non-relaxable task boundary, so the candidate cannot serve any task-related evidence role. Mere non-primary status, overlap, alternate specification, incomplete adjustment information, or a different but related reporting form is not enough.

Set disposition strictly from evidence_role:
- `main_analysis_candidate`, `supplementary_or_sensitivity`, or `source_only` -> `keep_as_valid_candidate`.
- `uncertain` -> `needs_further_review`.
- `none` -> `discard_clear_mismatch`.

`discard_clear_mismatch` is therefore a general semantic exclusion label, not a shortcut for records that fail to enter the main analysis. It requires affirmative evidence of an irreconcilable task contradiction.

Return exactly one item for each prediction_id. Each item must contain only prediction_id, disposition, evidence_role, and one concise English reason of at most 120 characters. Do not add fields or restate source data."""


def _compact_pico_dimension(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"description": value, "aliases": []}
    return {
        "description": value.get("description"),
        "aliases": value.get("aliases", [])[:12],
    }


def _compact_target(target: Any) -> dict[str, Any]:
    target = target if isinstance(target, dict) else {}
    outcome = target.get("outcome")
    outcome = outcome if isinstance(outcome, dict) else {"description": outcome}
    return {
        "target_id": target.get("target_id"),
        "outcome": _compact_pico_dimension(outcome),
        "comparison": target.get("comparison"),
        "population_or_subgroup": target.get("population_or_subgroup"),
        "accepted_data_families": target.get("accepted_data_families", []),
        "preferred_effect_measures": target.get("preferred_effect_measures", []),
        "estimands": target.get("estimands", []),
        "extraction_notes": target.get("extraction_notes", []),
        "timepoint_policy": target.get("timepoint_policy"),
        "model_selection_policy": target.get("model_selection_policy"),
    }


def _review_contract(task: LoadedTask) -> dict[str, Any]:
    """Keep the repeated per-batch task context semantic and bounded."""
    raw_extraction = task.raw.get("extraction", {})
    pico = task.pico if isinstance(task.pico, dict) else {}
    exposure = pico.get("exposure", pico.get("intervention_or_exposure"))
    return {
        "pico": {
            "population": _compact_pico_dimension(pico.get("population")),
            "exposure": _compact_pico_dimension(exposure),
            "comparator": _compact_pico_dimension(pico.get("comparator")),
        },
        "targets": [
            _compact_target(target) for target in raw_extraction.get("targets", [])
        ],
        "allowed_effect_types": task.allowed_effect_types,
        "estimands": task.estimands,
        "estimand_policy": raw_extraction.get("estimand_policy"),
        "exclusion_concepts": task.exclusion_concepts,
    }


def _model(task: LoadedTask, configured_model: str | None):
    models = task.raw.get("models", {})
    model_name = configured_model or models.get("reviewer") or models.get("default")
    timeout = float(task.execution_policy.get("llm_timeout_seconds", 110.0))
    return chat_model(model_name, timeout=timeout, max_retries=0)


def _article_batches(
    predictions: list[dict[str, Any]], batch_size: int
) -> list[list[dict[str, Any]]]:
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2 for batched review")
    by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    article_order: list[str] = []
    for prediction in predictions:
        article_id = str(prediction.get("article_id", ""))
        if article_id not in by_article:
            article_order.append(article_id)
        by_article[article_id].append(prediction)
    batches: list[list[dict[str, Any]]] = []
    for article_id in article_order:
        rows = by_article[article_id]
        for start in range(0, len(rows), batch_size):
            batches.append(rows[start : start + batch_size])
    return batches


def _compact_candidate(
    prediction: dict[str, Any], source_evidence_id: str
) -> dict[str, Any]:
    """Candidate fields reference a separately transmitted source-evidence block."""
    effect = prediction.get("used_effect") or {}
    return {
        "prediction_id": prediction.get("prediction_id"),
        "source_evidence_id": source_evidence_id,
        "target_id": prediction.get("target_id"),
        "outcome": prediction.get("outcome"),
        "exposure": prediction.get("exposure"),
        "comparator": prediction.get("comparator"),
        "population_or_subgroup": prediction.get("population_or_subgroup"),
        "estimand": prediction.get("estimand"),
        "data_family": prediction.get("data_family"),
        "raw_data": copy.deepcopy(prediction.get("raw_data") or {}),
        "extraction_issues": copy.deepcopy(prediction.get("issues") or prediction.get("organizer_issues") or []),
        "effect_type": effect.get("effect_type"),
        "effect_value": effect.get("value"),
        "ci_lower": effect.get("ci_lower"),
        "ci_upper": effect.get("ci_upper"),
        "model_name": prediction.get("model_name"),
        "adjustment": prediction.get("adjustment"),
        "timepoint": prediction.get("timepoint"),
    }


def _compact_batch_payload(
    predictions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Share actual source blocks across rows, including differing source quotes."""
    evidence_ids: dict[str, str] = {}
    source_evidence: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    def register(block: dict[str, Any], *, missing_key: str = "") -> str:
        key = json.dumps(block, sort_keys=True, ensure_ascii=False) + missing_key
        if key not in evidence_ids:
            evidence_ids[key] = f"source_{len(source_evidence) + 1}"
            source_evidence.append({"source_evidence_id": evidence_ids[key], **block})
        return evidence_ids[key]

    for prediction in predictions:
        location = prediction.get("source_location") or {}
        quote = str(location.get("source_quote") or "")
        source_evidence_id = register({
            "source_type": location.get("source_type"),
            "source_id": location.get("source_id"),
            "source_quote": quote,
        }, missing_key="" if quote else str(prediction.get("prediction_id")))
        supporting_ids = []
        for block in location.get("source_evidence") or []:
            if not isinstance(block, dict) or not block.get("source_id"):
                continue
            text = str(block.get("text") or block.get("source_quote") or "")
            if not text.strip():
                continue
            supporting_ids.append(register({
                "source_type": block.get("source_type"),
                "source_id": block["source_id"],
                "source_text": text,
                "context_only": bool(block.get("context_only")),
            }))
        candidate = _compact_candidate(prediction, source_evidence_id)
        candidate["supporting_source_evidence_ids"] = list(dict.fromkeys(supporting_ids))
        candidates.append(candidate)
    return source_evidence, candidates


def review_batch(
    task: LoadedTask,
    predictions: list[dict[str, Any]],
    configured_model: str | None = None,
) -> CandidateReviewBatch:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REVIEW_SYSTEM_PROMPT),
            (
                "user",
                "PICO and extraction requirements:\n{contract}\n\n"
                "Source evidence blocks (each may support multiple candidates):\n"
                "{source_evidence}\n\n"
                "Candidate records from one article; use source_evidence_id to inspect "
                "their evidence (including supporting_source_evidence_ids):\n{candidates}\n\n"
                "Only return candidate ID, disposition, evidence_role, and concise reason.",
            ),
        ]
    )
    chain = prompt | _model(task, configured_model).with_structured_output(
        CandidateReviewBatch
    )
    source_evidence, candidates = _compact_batch_payload(predictions)
    result = chain.invoke(
        {
            "contract": json.dumps(_review_contract(task), ensure_ascii=False),
            "source_evidence": json.dumps(source_evidence, ensure_ascii=False),
            "candidates": json.dumps(candidates, ensure_ascii=False),
        }
    )
    expected_ids = [str(prediction["prediction_id"]) for prediction in predictions]
    returned_ids = [review.prediction_id for review in result.reviews]
    if len(returned_ids) != len(set(returned_ids)):
        raise ValueError("Review batch returned duplicate prediction_ids")
    if set(returned_ids) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(returned_ids))
        extra = sorted(set(returned_ids) - set(expected_ids))
        raise ValueError(f"Review batch ID mismatch; missing={missing}, extra={extra}")
    by_id = {review.prediction_id: review for review in result.reviews}
    return CandidateReviewBatch(
        reviews=[by_id[prediction_id] for prediction_id in expected_ids]
    )


def review_batch_with_retries(
    task: LoadedTask,
    predictions: list[dict[str, Any]],
    configured_model: str | None,
    max_retries: int,
    retry_delay_seconds: float,
) -> tuple[CandidateReviewBatch, int]:
    """Retry one review batch only; extraction output and other batches are untouched."""
    if max_retries < 0 or retry_delay_seconds < 0:
        raise ValueError("Retry counts and delays must be non-negative")
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 2):
        try:
            return review_batch(task, predictions, configured_model), attempt
        except Exception as exc:
            last_error = exc
            if attempt > max_retries:
                break
            print(
                f"[{predictions[0].get('article_id')}] review batch attempt "
                f"{attempt} failed ({type(exc).__name__}); retrying in "
                f"{retry_delay_seconds:g}s.",
                flush=True,
            )
            time.sleep(retry_delay_seconds)
    assert last_error is not None
    raise last_error


def review_submission(
    task: LoadedTask,
    submission: dict[str, Any],
    *,
    configured_model: str | None = None,
    batch_size: int = 12,
    max_workers: int = 4,
    max_retries: int = 3,
    retry_delay_seconds: float = 5.0,
) -> dict[str, Any]:
    predictions = list(submission.get("predictions", []))
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if max_retries < 0 or retry_delay_seconds < 0:
        raise ValueError("Retry counts and delays must be non-negative")
    batches = _article_batches(predictions, batch_size)
    completed: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    batch_attempts: list[dict[str, Any]] = []
    with ThreadPoolExecutor(
        max_workers=min(max_workers, max(1, len(batches)))
    ) as executor:
        futures = {
            executor.submit(
                review_batch_with_retries,
                task,
                batch,
                configured_model,
                max_retries,
                retry_delay_seconds,
            ): batch
            for batch in batches
        }
        for future in as_completed(futures):
            batch = futures[future]
            prediction_ids = [str(row["prediction_id"]) for row in batch]
            try:
                result, attempts = future.result()
                for review in result.reviews:
                    completed[review.prediction_id] = review.model_dump()
                batch_attempts.append(
                    {
                        "article_id": batch[0].get("article_id"),
                        "prediction_ids": prediction_ids,
                        "attempts": attempts,
                        "status": "completed",
                    }
                )
                print(
                    f"[{batch[0].get('article_id')}] reviewed {len(batch)} candidates "
                    f"in {attempts} attempt(s)",
                    flush=True,
                )
            except Exception as exc:
                failures.append(
                    {
                        "article_id": batch[0].get("article_id"),
                        "prediction_ids": prediction_ids,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "attempts": max_retries + 1,
                    }
                )
                print(
                    f"[{batch[0].get('article_id')}] review failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
    ordered_reviews = [
        {
            "prediction_id": str(prediction["prediction_id"]),
            "article_id": str(prediction["article_id"]),
            "review": completed[str(prediction["prediction_id"])],
        }
        for prediction in predictions
        if str(prediction["prediction_id"]) in completed
    ]
    disposition_counts: dict[str, int] = defaultdict(int)
    for record in ordered_reviews:
        review = record["review"]
        disposition_counts[review["disposition"]] += 1
    return {
        "schema_version": "candidate_evidence_review/structured",
        "workflow_version": "locator_extractor_candidate_review_structured",
        "case_id": task.case_id,
        "task_id": task.task_id,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "boundary": {
            "review_only": True,
            "calculation": False,
            "ranking_or_main_selection": False,
            "clear_mismatch_discard_allowed": True,
            "tool_execution": False,
            "mutates_extraction": False,
        },
        "batching": {
            "strategy": "same article, up to batch_size candidates per LLM call",
            "batch_size": batch_size,
            "batch_count": len(batches),
            "max_workers": min(max_workers, max(1, len(batches))),
        },
        "retry_policy": {
            "max_retries_after_initial_attempt": max_retries,
            "retry_delay_seconds": retry_delay_seconds,
            "batch_attempts": batch_attempts,
        },
        "input_prediction_count": len(predictions),
        "completed_review_count": len(ordered_reviews),
        "failed_review_count": sum(
            len(failure["prediction_ids"]) for failure in failures
        ),
        "disposition_counts": dict(disposition_counts),
        "candidate_reviews": ordered_reviews,
        "failures": failures,
    }


def write_review(path: Path, review: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(review, handle, ensure_ascii=False, indent=2)


def _markdown_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return " ".join(text.replace("|", "/").split())


DISPOSITION_LABELS = {
    "keep_as_valid_candidate": "Keep as valid candidate",
    "discard_clear_mismatch": "Discard (clear mismatch)",
    "needs_further_review": "Needs further review",
}

EVIDENCE_ROLE_LABELS = {
    "main_analysis_candidate": "Main analysis candidate",
    "supplementary_or_sensitivity": "Supplementary / sensitivity evidence",
    "source_only": "Source-only evidence",
    "uncertain": "Insufficient or conflicting evidence",
    "none": "No task-relevant evidence role",
}


def write_review_summary(path: Path, review: dict[str, Any]) -> None:
    """Write a compact, human-readable companion to the full JSON audit trace."""
    counts = review.get("disposition_counts") or {}
    lines = [
        f"# Candidate Evidence Review Summary: {review.get('case_id', '')}",
        "",
        "## Review Boundary",
        "",
        "This stage performs read-only verification without numerical calculation, tool execution, or preferred-model ranking. "
        "Candidates are assigned to main-analysis, supplementary/sensitivity, source-only, or pending-review layers; "
        "only irreconcilable contradictions with non-relaxable task boundaries are marked as discarded.",
        "",
        "## Overview",
        "",
        "| Input candidates | Reviewed | Kept | Discarded | Needs further review | Failed |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        "| {input_count} | {completed} | {kept} | {discarded} | {further} | {failed} |".format(
            input_count=review.get("input_prediction_count", 0),
            completed=review.get("completed_review_count", 0),
            kept=counts.get("keep_as_valid_candidate", 0),
            discarded=counts.get("discard_clear_mismatch", 0),
            further=counts.get("needs_further_review", 0),
            failed=review.get("failed_review_count", 0),
        ),
        "",
    ]
    by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in review.get("candidate_reviews", []):
        by_article[str(record.get("article_id", "unknown"))].append(record)
    for article_id, records in by_article.items():
        article_counts: dict[str, int] = defaultdict(int)
        for record in records:
            article_counts[record["review"]["disposition"]] += 1
        article_overview = (
            "Kept: {kept}; Discarded: {discarded}; Needs further review: {further}."
        ).format(
            kept=article_counts.get("keep_as_valid_candidate", 0),
            discarded=article_counts.get("discard_clear_mismatch", 0),
            further=article_counts.get("needs_further_review", 0),
        )
        lines.extend(
            [
                f"## Article: {article_id}",
                "",
                article_overview,
                "",
                "| Candidate ID | Disposition | Evidence role | Review rationale |",
                "| --- | --- | --- | --- |",
            ]
        )
        for record in records:
            audited = record["review"]
            lines.append(
                "| {id} | {disposition} | {role} | {reason} |".format(
                    id=_markdown_cell(record.get("prediction_id")),
                    disposition=_markdown_cell(
                        DISPOSITION_LABELS.get(
                            audited.get("disposition"), audited.get("disposition")
                        )
                    ),
                    role=_markdown_cell(
                        EVIDENCE_ROLE_LABELS.get(
                            audited.get("evidence_role"),
                            audited.get("evidence_role"),
                        )
                    ),
                    reason=_markdown_cell(audited.get("reason")),
                )
            )
        lines.append("")
    if review.get("failures"):
        lines.extend(["## Failed Review Batches", ""])
        for failure in review["failures"]:
            lines.append(
                f"- `{_markdown_cell(failure.get('article_id'))}`: "
                f"{_markdown_cell(failure.get('error_type'))} - "
                f"{_markdown_cell(failure.get('error'))}"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
