from __future__ import annotations

import math
import re
from typing import Any

from constrained_meta_agent.models import LoadedTask
from .extraction_models import (
    BinaryArmCounts,
    ContinuousArmSummary,
    ContinuousChangeScoreSummary,
    DataFamilyExtraction,
    EffectExtraction,
    EventCountsWithPersonTime,
    ExtractedBinaryArmCounts,
    ExtractedContinuousArmSummary,
    ExtractedContinuousChangeScoreSummary,
    ExtractedEffect,
    ExtractedEventCountsWithPersonTime,
    ExtractedPercentageOnly,
)


SOURCE_TYPE_MAP = {
    "text": "paragraph",
    "paragraph": "paragraph",
    "table": "table",
    "figure": "figure",
    "image": "figure",
    "img": "figure",
    "supplement": "supplement",
    "appendix": "appendix",
}


def _number_present(value: float | None, text: str) -> bool:
    if value is None:
        return True
    return any(
        abs(float(raw) - value) < 1e-8
        for raw in re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", text)
    )


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _benchmark_source_type(source: dict[str, Any] | None) -> str:
    if source is None:
        return "other"
    return SOURCE_TYPE_MAP.get(str(source.get("source_type", "")).casefold(), "other")


def _content_list_item_index(source_id: str) -> int | None:
    match = re.fullmatch(r"content_(\d+)", source_id)
    return int(match.group(1)) if match else None


def _binary_effect_and_ci(
    raw: BinaryArmCounts, effect_type: str
) -> tuple[float, float, float]:
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
        raise ValueError(f"binary_arm_counts cannot derive {effect_type}")
    delta = 1.96 * se
    return value, math.exp(math.log(value) - delta), math.exp(math.log(value) + delta)


def _binary_effect_type(
    target: dict[str, Any] | None, allowed_effect_types: list[str]
) -> str:
    preferred = (target or {}).get("preferred_effect_measures", [])
    for effect_type in [*preferred, *allowed_effect_types]:
        if effect_type in {"OR", "RR"}:
            return effect_type
    raise ValueError("binary_arm_counts requires OR or RR")


def _smd_and_ci(
    raw: ContinuousArmSummary | ContinuousChangeScoreSummary,
) -> tuple[float, float, float]:
    if isinstance(raw, ContinuousArmSummary):
        mean_i, sd_i = raw.intervention_mean, raw.intervention_sd
        mean_c, sd_c = raw.control_mean, raw.control_sd
    else:
        mean_i, sd_i = raw.intervention_mean_change, raw.intervention_sd_change
        mean_c, sd_c = raw.control_mean_change, raw.control_sd_change
    pooled = math.sqrt(
        (
            (raw.intervention_total - 1) * sd_i**2
            + (raw.control_total - 1) * sd_c**2
        )
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


def _event_rate_ratio_and_ci(
    raw: EventCountsWithPersonTime,
) -> tuple[float, float, float]:
    intervention_events = float(raw.intervention_episodes)
    control_events = float(raw.control_episodes)
    if min(intervention_events, control_events) == 0:
        intervention_events += 0.5
        control_events += 0.5
    value = (intervention_events / raw.intervention_total) / (
        control_events / raw.control_total
    )
    se = math.sqrt(1 / intervention_events + 1 / control_events)
    delta = 1.96 * se
    return value, math.exp(math.log(value) - delta), math.exp(math.log(value) + delta)


def _binary_raw_data_for_target(
    raw: BinaryArmCounts, target: dict[str, Any] | None
) -> dict[str, Any]:
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
    if isinstance(row, ExtractedBinaryArmCounts):
        return _binary_raw_data_for_target(row.raw_data, target)
    if isinstance(
        row,
        (
            ExtractedContinuousArmSummary,
            ExtractedContinuousChangeScoreSummary,
            ExtractedEventCountsWithPersonTime,
            ExtractedPercentageOnly,
        ),
    ):
        return row.raw_data.model_dump()
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
    extractions: list[EffectExtraction | DataFamilyExtraction],
    catalog: list[dict[str, Any]],
    article_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {item["source_id"]: item for item in catalog}
    predictions: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    targets = task.raw.get("extraction", {}).get("targets", [])
    target_by_id = {
        target["target_id"]: target for target in targets if target.get("target_id")
    }
    study_id = task.raw.get("inputs", {}).get("study_ids", {}).get(article_id)
    row_number = 0

    for extraction in extractions:
        for row in extraction.rows:
            row_number += 1
            prediction_id = f"locator_{article_id}_{row_number:03d}"
            row_issues: list[str] = []
            source = by_id.get(row.source_id)
            if source is None:
                row_issues.append("unknown_source_id")
            target = target_by_id.get(row.target_id or "")
            is_binary = isinstance(row, ExtractedBinaryArmCounts)
            is_continuous = isinstance(
                row,
                (ExtractedContinuousArmSummary, ExtractedContinuousChangeScoreSummary),
            )
            is_rate = isinstance(row, ExtractedEventCountsWithPersonTime)
            is_percentage = isinstance(row, ExtractedPercentageOnly)

            if is_binary:
                if row.raw_data.intervention_events > row.raw_data.intervention_total:
                    row_issues.append("intervention_events_exceed_total")
                if row.raw_data.control_events > row.raw_data.control_total:
                    row_issues.append("control_events_exceed_total")
                effect_type = _binary_effect_type(target, task.allowed_effect_types)
                effect_value, ci_lower, ci_upper = _binary_effect_and_ci(
                    row.raw_data, effect_type
                )
            elif is_continuous:
                effect_type = "SMD"
                effect_value, ci_lower, ci_upper = _smd_and_ci(row.raw_data)
            elif is_rate:
                effect_type = "IRR"
                effect_value, ci_lower, ci_upper = _event_rate_ratio_and_ci(row.raw_data)
            elif is_percentage:
                effect_type = effect_value = ci_lower = ci_upper = None
            else:
                effect_type = row.effect_type
                effect_value = row.effect_value
                ci_lower = row.ci_lower
                ci_upper = row.ci_upper

            if effect_type and effect_type not in task.allowed_effect_types:
                row_issues.append("effect_type_not_preferred")
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
            if effect_type in {"HR", "RR", "OR", "IRR", "RRR"}:
                if effect_value is not None and effect_value <= 0:
                    row_issues.append("nonpositive_ratio_effect")
            if (ci_lower is None) != (ci_upper is None):
                row_issues.append("partial_confidence_interval")
            if effect_value is not None and ci_lower is not None and ci_upper is not None:
                if not ci_lower <= effect_value <= ci_upper:
                    row_issues.append("effect_outside_confidence_interval")

            if source is not None:
                if row.source_quote and _normalized_text(row.source_quote) not in _normalized_text(
                    source["content"]
                ):
                    row_issues.append("source_quote_not_exact")
                if is_binary:
                    for name, value in row.raw_data.model_dump().items():
                        if name != "outcome_unit" and not _number_present(
                            float(value), source["content"]
                        ):
                            row_issues.append(f"{name}_not_found_in_source")
                elif not is_percentage:
                    for name, value in (
                        ("effect", effect_value),
                        ("ci_lower", ci_lower),
                        ("ci_upper", ci_upper),
                    ):
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

            predictions.append(
                {
                    "prediction_id": prediction_id,
                    "article_id": article_id,
                    "study_id": study_id,
                    "target_id": row.target_id,
                    "outcome": row.outcome,
                    "population_or_subgroup": row.population_or_subgroup,
                    "exposure": row.exposure,
                    "comparator": row.comparator,
                    "estimand": row.estimand,
                    "data_family": row.data_family,
                    "timepoint": row.timepoint,
                    "used_effect": {
                        "effect_type": effect_type,
                        "value": effect_value,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                    },
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
                        "source_type": _benchmark_source_type(source),
                        "source_id": row.source_id,
                        "content_list_item_index": _content_list_item_index(
                            row.source_id
                        ),
                        "source_quote": row.source_quote,
                    },
                    "verification_status": (
                        "accepted" if not row_issues else "human_review"
                    ),
                }
            )
            if row_issues:
                issues.append({"prediction_id": prediction_id, "issues": row_issues})
    return predictions, issues
