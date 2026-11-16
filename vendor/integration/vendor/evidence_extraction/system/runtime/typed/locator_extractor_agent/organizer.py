from __future__ import annotations

import re
import unicodedata
from typing import Any

from constrained_meta_agent.models import LoadedTask
from .extraction_models import (
    BinaryArmCounts,
    DataFamilyExtraction,
    EffectExtraction,
    ExtractedBinaryArmCounts,
    ExtractedContinuousArmSummary,
    ExtractedContinuousChangeScoreSummary,
    ExtractedEffect,
    ExtractedEventCountsWithPersonTime,
    ExtractedAlleleFrequencySummary,
    ExtractedGenotypeCaseControlCounts,
    ExtractedIndependentContinuousArms,
    ExtractedPercentageOnly,
    ExtractedGenericTextEvidence,
    ExtractedSingleArmContinuousSummary,
    ExtractedDiagnosticFourCellCounts,
    ExtractedSingleArmEventCounts,
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

DATA_FAMILY_ALIASES = {
    "binary_arm_outcome": "binary_arm_counts",
    "binary_arm_event_counts": "binary_arm_counts",
    "two_group_event_counts": "binary_arm_counts",
    "binary_exposure_outcome_counts": "binary_arm_counts",
    "adjusted_time_to_event_effect": "adjusted_ratio_effect",
}


def _number_present(value: float | None, text: str) -> bool:
    if value is None:
        return True
    return any(
        abs(float(raw) - value) < 1e-8
        for raw in re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", text)
    )


def _nested_numeric_values(value: Any, prefix: str = "") -> list[tuple[str, float]]:
    """Flatten source-data template numbers for deterministic evidence checks."""
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [(prefix or "value", float(value))]
    if isinstance(value, dict):
        return [
            item
            for key, child in value.items()
            for item in _nested_numeric_values(child, f"{prefix}.{key}".strip("."))
        ]
    if isinstance(value, list):
        return [
            item
            for index, child in enumerate(value)
            for item in _nested_numeric_values(child, f"{prefix}[{index}]")
        ]
    return []


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _benchmark_source_type(source: dict[str, Any] | None) -> str:
    if source is None:
        return "other"
    return SOURCE_TYPE_MAP.get(str(source.get("source_type", "")).casefold(), "other")


def _content_list_item_index(source_id: str) -> int | None:
    match = re.fullmatch(r"content_(\d+)", source_id)
    return int(match.group(1)) if match else None


SOURCE_LABEL_RE = re.compile(
    r"\b(?P<label>(?:table|figure|fig\.|appendix|supplement)\s+"
    r"(?:[0-9]+|[A-Z][A-Z0-9.-]*))\b",
    re.IGNORECASE,
)


def _lookup_text(value: str) -> str:
    """Normalize common PDF/Markdown typography for deterministic quote lookup."""
    normalized = unicodedata.normalize("NFKC", value or "").casefold()
    normalized = normalized.replace("\u00a0", " ").replace("\u00ad", "")
    normalized = normalized.replace("–", "-").replace("—", "-").replace("−", "-")
    normalized = re.sub(r"\s*[-/]\s*", lambda match: match.group(0).strip(), normalized)
    return " ".join(normalized.split())


def _last_source_label(context: str) -> tuple[str, str] | None:
    labels = list(SOURCE_LABEL_RE.finditer(context or ""))
    if not labels:
        return None
    label = labels[-1].group("label")
    normalized = label.casefold().replace("fig.", "figure")
    kind = normalized.split(None, 1)[0]
    return label.strip(), SOURCE_TYPE_MAP.get(kind, "other")


def _resolve_markdown_source(
    task: LoadedTask,
    article_id: str,
    source: dict[str, Any] | None,
    source_quote: str,
) -> dict[str, Any]:
    """Resolve a chunk quote to a human-readable Markdown source label."""
    fallback_id = source.get("source_id") if source else None
    fallback_type = _benchmark_source_type(source)
    markdown_path = task.documents[article_id].representations.get("markdown")
    markdown = ""
    if markdown_path is not None and markdown_path.is_file():
        markdown = markdown_path.read_text(encoding="utf-8")

    quote_key = _lookup_text(source_quote)
    context = source.get("content", "") if source else ""
    resolved_status = "unresolved"
    if quote_key and quote_key in _lookup_text(context):
        resolved_status = "matched_source_chunk"
    elif quote_key and quote_key in _lookup_text(markdown):
        resolved_status = "matched_markdown"

    label = _last_source_label(context)
    if label is None and markdown and quote_key:
        markdown_key = _lookup_text(markdown)
        quote_position = markdown_key.find(quote_key)
        if quote_position >= 0:
            label = _last_source_label(markdown_key[max(0, quote_position - 4000) : quote_position])
            if label is not None:
                resolved_status = "matched_markdown"

    if label is None:
        return {
            "source_type": fallback_type,
            "source_id": fallback_id,
            "resolution_status": resolved_status,
        }
    source_id, source_type = label
    return {
        "source_type": source_type,
        "source_id": source_id,
        "resolution_status": resolved_status,
    }


def _binary_raw_data_for_target(
    raw: BinaryArmCounts, target: dict[str, Any] | None
) -> dict[str, Any]:
    data_contract = (target or {}).get("data_contract") or {}
    fields = data_contract.get("raw_data_fields", [])
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
        data_contract = (target or {}).get("data_contract") or {}
        fields = data_contract.get("raw_data_fields", [])
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
    if isinstance(row, ExtractedGenericTextEvidence):
        return row.raw_data.model_dump()
    if isinstance(
        row,
        (
            ExtractedGenotypeCaseControlCounts,
            ExtractedAlleleFrequencySummary,
            ExtractedIndependentContinuousArms,
            ExtractedSingleArmContinuousSummary,
            ExtractedDiagnosticFourCellCounts,
            ExtractedSingleArmEventCounts,
        ),
    ):
        return row.raw_data.model_dump()
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
            is_generic = isinstance(row, ExtractedGenericTextEvidence)
            is_source_data = isinstance(
                row,
                (
                    ExtractedGenotypeCaseControlCounts,
                    ExtractedAlleleFrequencySummary,
                    ExtractedIndependentContinuousArms,
                    ExtractedSingleArmContinuousSummary,
                    ExtractedDiagnosticFourCellCounts,
                    ExtractedSingleArmEventCounts,
                ),
            )

            if is_binary:
                if row.raw_data.intervention_events > row.raw_data.intervention_total:
                    row_issues.append("intervention_events_exceed_total")
                if row.raw_data.control_events > row.raw_data.control_total:
                    row_issues.append("control_events_exceed_total")
            if is_binary or is_continuous or is_rate or is_percentage or is_generic or is_source_data:
                # Extraction only transcribes source data. Effect conversion and any
                # continuity correction belong to the downstream calculator.
                effect_type = effect_value = ci_lower = ci_upper = None
            else:
                effect_type = row.effect_type
                effect_value = row.effect_value
                ci_lower = row.ci_lower
                ci_upper = row.ci_upper

            if effect_type and effect_type not in task.allowed_effect_types:
                row_issues.append("effect_type_not_preferred")
            if task.estimands and row.estimand not in task.estimands:
                row_issues.append("estimand_not_allowed")
            if target_by_id and target is None:
                row_issues.append("target_id_not_allowed")
            if target is not None:
                aliases = DATA_FAMILY_ALIASES
                if task.raw.get("system_partition") not in {
                    "locator_extractor_structured_23",
                    "locator_extractor_typed_data_family",
                }:
                    aliases = {"binary_arm_outcome": "binary_arm_counts"}
                allowed_families = {
                    aliases.get(family, family)
                    for family in target.get("accepted_data_families", [])
                }
                if row.data_family not in allowed_families and not is_generic:
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
                elif not is_percentage and not is_generic:
                    for name, value in (
                        ("effect", effect_value),
                        ("ci_lower", ci_lower),
                        ("ci_upper", ci_upper),
                    ):
                        if not _number_present(value, source["content"]):
                            row_issues.append(f"{name}_not_found_in_source")
                if is_source_data:
                    for name, value in _nested_numeric_values(row.raw_data.model_dump()):
                        if not _number_present(value, source["content"]):
                            row_issues.append(f"{name}_not_found_in_source")

            source_data_fingerprint = (
                repr(row.raw_data.model_dump()) if is_source_data else None
            )
            duplicate_key = (
                row.target_id,
                row.outcome.casefold(),
                row.population_or_subgroup.casefold(),
                row.estimand,
                effect_type,
                effect_value,
                ci_lower,
                ci_upper,
                source_data_fingerprint,
            )
            if duplicate_key in seen:
                row_issues.append("duplicate_effect")
            seen.add(duplicate_key)

            resolved_source = _resolve_markdown_source(
                task,
                article_id,
                source,
                row.source_quote,
            )
            if row.source_quote and resolved_source["resolution_status"] == "unresolved":
                row_issues.append("source_quote_not_found_in_markdown")

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
                        "source_type": resolved_source["source_type"],
                        "source_id": resolved_source["source_id"],
                        "content_list_item_index": _content_list_item_index(
                            row.source_id
                        ),
                        "source_quote": row.source_quote,
                        "resolution_status": resolved_source["resolution_status"],
                    },
                    "verification_status": (
                        "accepted" if not row_issues else "human_review"
                    ),
                }
            )
            if row_issues:
                issues.append({"prediction_id": prediction_id, "issues": row_issues})
    return predictions, issues
