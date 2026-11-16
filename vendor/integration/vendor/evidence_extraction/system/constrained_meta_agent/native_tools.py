from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from llm_endpoint import chat_model
from .models import DocumentInput, LoadedTask
from .tool_models import (
    BinaryArmCounts,
    ContinuousArmSummary,
    ContinuousChangeScoreSummary,
    EffectExtraction,
    DataFamilyExtraction,
    ExtractedBinaryArmCounts,
    ExtractedContinuousArmSummary,
    ExtractedContinuousChangeScoreSummary,
    ExtractedEventCountsWithPersonTime,
    ExtractedEffect,
    ExtractedPercentageOnly,
    EventCountsWithPersonTime,
    ExtractionPlan,
    ExtractionPlanTask,
)


def _llm(task: LoadedTask, role: str) -> ChatOpenAI:
    return chat_model(task.raw.get("models", {}).get(role), timeout=90.0, max_retries=3)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(_as_text(item) for item in value if item is not None)
    return str(value)


def _candidate_text(item: dict[str, Any]) -> str:
    keys = (
        "text",
        "table_caption",
        "table_body",
        "table_footnote",
        "img_caption",
        "img_footnote",
    )
    return "\n".join(part for key in keys if (part := _as_text(item.get(key))))


def _pico_terms(task: LoadedTask) -> set[str]:
    raw = " ".join(
        [task.pico["population"], task.pico["exposure"], task.pico["comparator"], *task.pico["outcomes"]]
    ).lower()
    stop = {"with", "from", "that", "this", "have", "study", "studies", "including", "related", "defined"}
    return {token for token in re.findall(r"[a-z][a-z0-9-]+", raw) if len(token) >= 4 and token not in stop}


def _extraction_contract(task: LoadedTask) -> dict[str, Any]:
    return {
        "pico": task.pico,
        "targets": task.raw.get("extraction", {}).get("targets", []),
        "allowed_effect_types": task.allowed_effect_types,
        "estimands": task.estimands,
        "model_selection_policy": task.model_selection_policy,
        "exclusion_concepts": task.exclusion_concepts,
    }


def _data_families_for_plan(plan_task: ExtractionPlanTask) -> list[str]:
    return list(dict.fromkeys(plan_task.data_families))


def _binary_prompt_instructions() -> str:
    return """This task uses data_family=\"binary_arm_counts\". Extract the source-reported
two-arm event counts and denominators only. Return intervention_events, intervention_total,
control_events, control_total, and outcome_unit inside raw_data. Do not return a precomputed
effect, CI, or percentage; the calculator will derive those deterministically. The two event
counts must be counts of independent units with at least one event, not the number of events
when one unit can contribute multiple events. If a required count or denominator is absent,
return no row and explain it in unresolved_items. Never infer a zero from a missing value."""


_FAMILY_INSTRUCTIONS = {
    "reported_ratio_effect": "Return ExtractedEffect with effect_type/value/CI copied from the source.",
    "adjusted_ratio_effect": "Return ExtractedEffect with the adjusted source-reported effect_type/value/CI; do not recalculate.",
    "time_to_event": "Return ExtractedEffect with the source-reported HR and CI; preserve exposure_unit and adjustment_set.",
    "reported_standardized_mean_difference": "Return ExtractedEffect with effect_type=SMD and reported value/CI copied from the source.",
    "binary_arm_counts": _binary_prompt_instructions(),
    "continuous_arm_summary": "Return ExtractedContinuousArmSummary with intervention/control mean, SD, total and outcome_unit. Do not calculate SMD.",
    "continuous_change_score_summary": "Return ExtractedContinuousChangeScoreSummary with intervention/control change mean, SD, total and outcome_unit. Do not calculate SMD.",
    "event_counts_with_person_time": "Return ExtractedEventCountsWithPersonTime with intervention/control episode totals, participant totals, follow_up_months, and any printed rates.",
    "percentage_only": "Return ExtractedPercentageOnly with printed percentages and denominators only. Never reconstruct event counts from rounded percentages; mark source_only.",
}


def build_source_catalog(document: DocumentInput, task: LoadedTask) -> list[dict[str, Any]]:
    path = document.representations.get("content_list")
    if path is None:
        raise ValueError("native_tools requires a content_list representation")
    items = json.loads(path.read_text(encoding="utf-8"))
    terms = _pico_terms(task)
    catalog: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        content = _candidate_text(item)
        if not content.strip():
            continue
        source_type = str(item.get("type", "unknown")).lower()
        lowered = content.lower()
        relevance = sum(1 for term in terms if term in lowered)
        is_structured_evidence = source_type in {"table", "image", "figure"}
        if not is_structured_evidence and relevance < 2:
            continue
        label = _as_text(
            item.get("table_caption")
            or item.get("img_caption")
            or _as_text(item.get("text"))[:240]
        ).strip()
        page_idx = item.get("page_idx")
        catalog.append(
            {
                "source_id": f"content_{index:04d}",
                "source_type": source_type,
                "page": page_idx + 1 if isinstance(page_idx, int) else None,
                "page_index": page_idx,
                "bbox": item.get("bbox"),
                "label": label,
                "content": content,
                "relevance_score": relevance,
            }
        )
    return catalog


def _catalog_for_prompt(catalog: list[dict[str, Any]], max_chars: int) -> list[dict[str, Any]]:
    return [
        {
            "source_id": item["source_id"],
            "source_type": item["source_type"],
            "page": item["page"],
            "label": item["label"],
            "content_preview": item["content"][:max_chars],
        }
        for item in catalog
    ]


def plan_extraction_targets(
    task: LoadedTask,
    catalog: list[dict[str, Any]],
    feedback: str = "",
) -> ExtractionPlan:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You plan evidence extraction for a meta-analysis case. The supplied PICO, target definitions, and extraction contract are authoritative; do not redefine them. Select only analyses actually reported in the supplied source catalog. Use exact target IDs, data families, effect types, estimands, and candidate source IDs from the contract. Prefer the configured model-selection policy. Create separate tasks when targets, estimands, source objects, or analysis populations differ. Respect each target's quantitative-use policy. Maximize coverage of eligible reported PICO outcomes without inventing missing analyses.""",
            ),
            (
                "user",
                "PICO and extraction contract:\n{contract}\n\nCandidate source catalog:\n{catalog}\n\nRevision feedback:\n{feedback}",
            ),
        ]
    )
    chain = prompt | _llm(task, "planner").with_structured_output(ExtractionPlan)
    max_chars = task.execution_policy.get("catalog_preview_chars", 6000)
    return chain.invoke(
        {
            "contract": json.dumps(_extraction_contract(task), ensure_ascii=False),
            "catalog": json.dumps(_catalog_for_prompt(catalog, max_chars), ensure_ascii=False),
            "feedback": feedback or "No prior feedback.",
        }
    )


def locate_candidate_evidence(
    plan: ExtractionPlan,
    catalog: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {item["source_id"]: item for item in catalog}
    located: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for task_item in plan.tasks:
        sources = [by_id[source_id] for source_id in task_item.candidate_source_ids if source_id in by_id]
        invalid = [source_id for source_id in task_item.candidate_source_ids if source_id not in by_id]
        if invalid or not sources:
            issues.append(
                {
                    "task_id": task_item.task_id,
                    "invalid_source_ids": invalid,
                    "reason": "Planner source IDs were not present in the inspected document catalog.",
                }
            )
        if sources:
            located.append({"task": task_item, "sources": sources})
    return located, issues


def extract_effect_rows(
    task: LoadedTask,
    located_task: dict[str, Any],
    feedback: str = "",
) -> EffectExtraction | DataFamilyExtraction:
    plan_task: ExtractionPlanTask = located_task["task"]
    sources = located_task["sources"]
    families = _data_families_for_plan(plan_task)
    legacy_families = {"reported_ratio_effect", "adjusted_ratio_effect", "time_to_event"}
    is_legacy_effect_task = set(families).issubset(legacy_families)
    family_instructions = "\n".join(
        f"- {family}: {_FAMILY_INSTRUCTIONS.get(family, 'Use the target data contract exactly.') }"
        for family in families
    ) or "Use the target data contract exactly."
    system_prompt = (
        """Extract analysis-ready effect estimates from the supplied evidence only. Follow the task, PICO, and target definitions exactly. Return one row per unique target, outcome, estimand, population/subgroup, and model. Copy target_id, data_family, effect type, exposure unit, and quantitative_usability from the applicable target contract. Preserve the source-reported effect type and scale; never silently convert or rescale an effect. Never infer or manufacture missing numbers. Source IDs must exactly match the supplied evidence, and source_quote must quote the evidence supporting the estimate and confidence interval."""
        if is_legacy_effect_task
        else """Extract analysis-ready evidence rows from the supplied evidence only. Follow the task, PICO, and target definitions exactly. Return one row per unique target, outcome, estimand, population/subgroup, and model. Copy target_id, data_family, exposure unit, and quantitative_usability from the applicable target contract. Never infer or manufacture missing numbers. Source IDs must exactly match the supplied evidence, and source_quote must quote the evidence supporting every extracted value. {family_instructions}"""
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            (
                "user",
                "Extraction contract:\n{contract}\n\nTask:\n{task}\n\nLocated evidence:\n{sources}\n\nRetry feedback:\n{feedback}",
            ),
        ]
    )
    response_model = EffectExtraction if is_legacy_effect_task else DataFamilyExtraction
    chain = prompt | _llm(task, "extractor").with_structured_output(response_model)
    return chain.invoke(
        {
            "contract": json.dumps(_extraction_contract(task), ensure_ascii=False),
            "task": plan_task.model_dump_json(),
            "sources": json.dumps(
                [
                    {
                        "source_id": source["source_id"],
                        "source_type": source["source_type"],
                        "page": source["page"],
                        "label": source["label"],
                        "content": source["content"],
                    }
                    for source in sources
                ],
                ensure_ascii=False,
            ),
            "feedback": feedback or "No prior extraction attempt.",
            "family_instructions": family_instructions,
        }
    )


def _number_present(value: float | None, text: str) -> bool:
    if value is None:
        return True
    for raw in re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", text):
        if abs(float(raw) - value) < 1e-8:
            return True
    return False


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _binary_effect_and_ci(raw: BinaryArmCounts, effect_type: str) -> tuple[float, float, float]:
    """Calculate a ratio effect and Wald CI from source arm counts.

    Zero cells use the standard 0.5 continuity correction. This calculation is
    deliberately kept outside the LLM extraction step.
    """
    a = float(raw.intervention_events)
    b = float(raw.intervention_total - raw.intervention_events)
    c = float(raw.control_events)
    d = float(raw.control_total - raw.control_events)
    if min(a, b, c, d) == 0:
        a, b, c, d = (value + 0.5 for value in (a, b, c, d))
    if effect_type == "OR":
        value = (a * d) / (b * c)
        se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    elif effect_type == "RR":
        value = (a / (a + b)) / (c / (c + d))
        se = math.sqrt(1 / a - 1 / (a + b) + 1 / c - 1 / (c + d))
    else:
        raise ValueError(f"binary arm counts cannot derive {effect_type}")
    delta = 1.96 * se
    return value, math.exp(math.log(value) - delta), math.exp(math.log(value) + delta)


def _binary_effect_type(target: dict[str, Any] | None, allowed_effect_types: list[str]) -> str:
    preferred = (target or {}).get("preferred_effect_measures", [])
    for effect_type in [*preferred, *allowed_effect_types]:
        if effect_type in {"OR", "RR"}:
            return effect_type
    raise ValueError("binary_arm_counts requires OR or RR in the target contract")


def _smd_and_ci(raw: ContinuousArmSummary | ContinuousChangeScoreSummary) -> tuple[float, float, float]:
    if isinstance(raw, ContinuousArmSummary):
        mean_i, sd_i = raw.intervention_mean, raw.intervention_sd
        mean_c, sd_c = raw.control_mean, raw.control_sd
    else:
        mean_i, sd_i = raw.intervention_mean_change, raw.intervention_sd_change
        mean_c, sd_c = raw.control_mean_change, raw.control_sd_change
    pooled = math.sqrt(
        ((raw.intervention_total - 1) * sd_i**2 + (raw.control_total - 1) * sd_c**2)
        / (raw.intervention_total + raw.control_total - 2)
    )
    direction = 1 if raw.effect_orientation == "intervention_minus_control" else -1
    d = direction * (mean_i - mean_c) / pooled
    total = raw.intervention_total + raw.control_total
    correction = 1 - 3 / (4 * (total - 2) - 1)
    value = correction * d
    variance = total / (raw.intervention_total * raw.control_total) + d * d / (2 * total)
    delta = 1.96 * math.sqrt(variance)
    return value, value - delta, value + delta


def _event_rate_ratio_and_ci(raw: EventCountsWithPersonTime) -> tuple[float, float, float]:
    intervention_events = float(raw.intervention_episodes)
    control_events = float(raw.control_episodes)
    if min(intervention_events, control_events) == 0:
        intervention_events += 0.5
        control_events += 0.5
    value = (intervention_events / raw.intervention_total) / (control_events / raw.control_total)
    se = math.sqrt(1 / intervention_events + 1 / control_events)
    delta = 1.96 * se
    return value, math.exp(math.log(value) - delta), math.exp(math.log(value) + delta)


def _binary_raw_data_for_target(raw: BinaryArmCounts, target: dict[str, Any] | None) -> dict[str, Any]:
    """Use the protocol's source-data names while keeping the extractor generic."""
    fields = (target or {}).get("data_contract", {}).get("raw_data_fields", [])
    values = [
        raw.intervention_events,
        raw.intervention_total,
        raw.control_events,
        raw.control_total,
    ]
    if len(fields) >= 4:
        result = dict(zip(fields[:4], values, strict=True))
        if "outcome_unit" in fields:
            result["outcome_unit"] = raw.outcome_unit
        return result
    return raw.model_dump()


def _raw_data_for_target(row: Any, target: dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(row, (ExtractedBinaryArmCounts, ExtractedContinuousArmSummary, ExtractedContinuousChangeScoreSummary, ExtractedEventCountsWithPersonTime, ExtractedPercentageOnly)):
        return _binary_raw_data_for_target(row.raw_data, target) if isinstance(row, ExtractedBinaryArmCounts) else row.raw_data.model_dump()
    if isinstance(row, ExtractedEffect):
        fields = (target or {}).get("data_contract", {}).get("raw_data_fields", [])
        values = {
            "effect_value": row.effect_value,
            "ci_lower": row.ci_lower,
            "ci_upper": row.ci_upper,
            "effect_measure": row.effect_type,
            "exposure_unit": row.exposure_unit,
            "adjustment_set": row.adjustment,
        }
        result = {
            "data_type": f"reported_{row.effect_type.casefold()}_effect",
            "effect_value": row.effect_value,
            "ci_lower": row.ci_lower,
            "ci_upper": row.ci_upper,
            "exposure_unit": row.exposure_unit,
            "adjustment_set": row.adjustment,
        }
        result.update({field: values.get(field) for field in fields if field in values})
        return result
    return {}


def normalize_and_verify_effects(
    task: LoadedTask,
    extractions: list[EffectExtraction],
    catalog: list[dict[str, Any]],
    article_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {item["source_id"]: item for item in catalog}
    predictions: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    configured_targets = task.raw.get("extraction", {}).get("targets", [])
    target_by_id = {
        target["target_id"]: target
        for target in configured_targets
        if target.get("target_id")
    }
    row_number = 0
    for extraction in extractions:
        for row in extraction.rows:
            row_number += 1
            prediction_id = f"native_{article_id}_{row_number:03d}"
            row_issues: list[str] = []
            source = by_id.get(row.source_id)
            if source is None:
                row_issues.append("unknown_source_id")
            target = target_by_id.get(row.target_id or "")
            is_binary = isinstance(row, ExtractedBinaryArmCounts)
            is_continuous = isinstance(row, (ExtractedContinuousArmSummary, ExtractedContinuousChangeScoreSummary))
            is_rate = isinstance(row, ExtractedEventCountsWithPersonTime)
            is_percentage = isinstance(row, ExtractedPercentageOnly)
            if is_binary:
                if row.data_family != "binary_arm_counts":
                    row_issues.append("data_family_not_allowed")
                if row.raw_data.intervention_events > row.raw_data.intervention_total:
                    row_issues.append("intervention_events_exceed_total")
                if row.raw_data.control_events > row.raw_data.control_total:
                    row_issues.append("control_events_exceed_total")
                effect_type = _binary_effect_type(target, task.allowed_effect_types)
                effect_value, ci_lower, ci_upper = _binary_effect_and_ci(row.raw_data, effect_type)
            elif is_continuous:
                if "SMD" not in task.allowed_effect_types:
                    row_issues.append("effect_type_not_allowed")
                effect_type = "SMD"
                effect_value, ci_lower, ci_upper = _smd_and_ci(row.raw_data)
            elif is_rate:
                if "IRR" not in task.allowed_effect_types:
                    row_issues.append("effect_type_not_allowed")
                effect_type = "IRR"
                effect_value, ci_lower, ci_upper = _event_rate_ratio_and_ci(row.raw_data)
            elif is_percentage:
                effect_type = None
                effect_value = None
                ci_lower = None
                ci_upper = None
            else:
                effect_type = row.effect_type
                effect_value = row.effect_value
                ci_lower = row.ci_lower
                ci_upper = row.ci_upper
                if row.effect_type not in task.allowed_effect_types:
                    row_issues.append("effect_type_not_allowed")
            if row.estimand not in task.estimands:
                row_issues.append("estimand_not_allowed")
            if target_by_id and target is None:
                row_issues.append("target_id_not_allowed")
            if target is not None:
                allowed_families = target.get("accepted_data_families", [])
                if row.data_family not in allowed_families:
                    row_issues.append("data_family_not_allowed")
                expected_usability = target.get("quantitative_usability")
                if expected_usability and row.quantitative_usability != expected_usability:
                    row_issues.append("quantitative_usability_mismatch")
            outcome_terms = {
                token
                for token in re.findall(r"[a-z][a-z-]+", row.outcome.casefold())
                if len(token) >= 4
            }
            pico_outcome_text = " ".join(task.pico["outcomes"]).casefold()
            if outcome_terms and not any(term in pico_outcome_text for term in outcome_terms):
                row_issues.append("outcome_outside_pico")
            if effect_type in {"HR", "RR", "OR", "IRR", "RRR"} and effect_value is not None and effect_value <= 0:
                row_issues.append("nonpositive_ratio_effect")
            if (ci_lower is None) != (ci_upper is None):
                row_issues.append("partial_confidence_interval")
            if effect_value is not None and ci_lower is not None and ci_upper is not None:
                if not ci_lower <= effect_value <= ci_upper:
                    row_issues.append("effect_outside_confidence_interval")
            if source is not None:
                if row.source_quote and _normalized_text(row.source_quote) not in _normalized_text(source["content"]):
                    row_issues.append("source_quote_not_exact")
                if is_binary:
                    for name, value in row.raw_data.model_dump().items():
                        if name == "outcome_unit":
                            continue
                        if not _number_present(float(value), source["content"]):
                            row_issues.append(f"{name}_not_found_in_source")
                elif not is_percentage:
                    for name, value in (("effect", effect_value), ("ci_lower", ci_lower), ("ci_upper", ci_upper)):
                        if not _number_present(value, source["content"]):
                            row_issues.append(f"{name}_not_found_in_source")

            duplicate_key = (
                row.target_id,
                row.outcome.casefold(),
                row.population_or_subgroup.casefold(),
                row.estimand,
                effect_type,
                effect_value,
                ci_lower,
                ci_upper,
            )
            if duplicate_key in seen:
                row_issues.append("duplicate_effect")
            seen.add(duplicate_key)

            status = "accepted" if not row_issues else "human_review"
            predictions.append(
                {
                    "prediction_id": prediction_id,
                    "article_id": article_id,
                    "target_id": row.target_id,
                    "outcome": row.outcome,
                    "population_or_subgroup": row.population_or_subgroup,
                    "exposure": row.exposure,
                    "comparator": row.comparator,
                    "estimand": row.estimand,
                    "data_family": row.data_family,
                    "timepoint": row.timepoint,
                    "used_effect": {"effect_type": effect_type, "value": effect_value, "ci_lower": ci_lower, "ci_upper": ci_upper},
                    "raw_data": _raw_data_for_target(row, target),
                    "quantitative_usability": row.quantitative_usability,
                    "model_name": row.model_name,
                    "adjustment": row.adjustment,
                    "p_value": getattr(row, "p_value", None),
                    "sample_size": getattr(row, "sample_size", None),
                    "source_location": {
                        "document_id": task.documents[article_id].document_id,
                        "page": source["page"] if source else None,
                        "page_index": source["page_index"] if source else None,
                        "bbox": source["bbox"] if source else None,
                        "source_type": source["source_type"] if source else "unknown",
                        "source_id": row.source_id,
                        "source_quote": row.source_quote,
                    },
                    "verification_status": status,
                }
            )
            if row_issues:
                issues.append({"prediction_id": prediction_id, "issues": row_issues})
    return predictions, issues
