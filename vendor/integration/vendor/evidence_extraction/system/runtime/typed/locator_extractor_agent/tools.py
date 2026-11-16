from __future__ import annotations

import json
import os
import re
import threading
from copy import deepcopy
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from constrained_meta_agent.models import DocumentInput, LoadedTask
from llm_endpoint import chat_model

from .extraction_models import (
    DataFamilyExtraction,
    EffectExtraction,
    ExtractedEffect,
    ExtractedBinaryArmCounts,
    ExtractedContinuousArmSummary,
    ExtractedContinuousChangeScoreSummary,
    ExtractedEventCountsWithPersonTime,
    ExtractedGenericTextEvidence,
    ExtractedPercentageOnly,
    ExtractedAlleleFrequencySummary,
    ExtractedGenotypeCaseControlCounts,
    ExtractedIndependentContinuousArms,
    ExtractedSingleArmContinuousSummary,
    ExtractedDiagnosticFourCellCounts,
    ExtractedSingleArmEventCounts,
)
from .organizer import normalize_and_verify_effects
from .tool_models import EvidenceLocations


REFERENCE_HEADING = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:references|bibliography|literature\s+cited)\s*:?[ \t]*$"
)
LEGACY_EFFECT_FAMILIES = {
    "reported_ratio_effect",
    "adjusted_ratio_effect",
    "time_to_event",
    "reported_standardized_mean_difference",
}
DATA_FAMILY_TEMPLATE_MODELS = {
    "binary_arm_counts": ExtractedBinaryArmCounts,
    "continuous_arm_summary": ExtractedContinuousArmSummary,
    "continuous_change_score_summary": ExtractedContinuousChangeScoreSummary,
    "event_counts_with_person_time": ExtractedEventCountsWithPersonTime,
    "percentage_only": ExtractedPercentageOnly,
    "generic_text_evidence": ExtractedGenericTextEvidence,
    "complete_genotype_case_control_counts": ExtractedGenotypeCaseControlCounts,
    "allele_frequency_summary_only": ExtractedAlleleFrequencySummary,
    "independent_continuous_arms": ExtractedIndependentContinuousArms,
    "single_arm_continuous_summary": ExtractedSingleArmContinuousSummary,
    "diagnostic_four_cell_counts": ExtractedDiagnosticFourCellCounts,
    "single_arm_event_counts": ExtractedSingleArmEventCounts,
    # Existing effect rows can also be selected as a schema. This keeps mixed
    # count-plus-adjusted-effect targets out of an endpoint-unfriendly union.
    "adjusted_ratio_effect": ExtractedEffect,
    "reported_ratio_effect": ExtractedEffect,
    "time_to_event": ExtractedEffect,
    "reported_standardized_mean_difference": ExtractedEffect,
}
DATA_FAMILY_ALIASES = {
    # Portable benchmark Gold uses this label, while the extractor's existing
    # Pydantic template is named binary_arm_counts.
    "binary_arm_outcome": "binary_arm_counts",
    "binary_arm_event_counts": "binary_arm_counts",
    "two_group_event_counts": "binary_arm_counts",
    "binary_exposure_outcome_counts": "binary_arm_counts",
    "adjusted_time_to_event_effect": "adjusted_ratio_effect",
}
TEMPLATE_PARTITIONS = {
    "locator_extractor_structured_23",
    "locator_extractor_typed_data_family",
}

_PROVIDER_ENV_LOCK = threading.Lock()


def _llm(task: LoadedTask, role: str, provider: str = "official_openai") -> ChatOpenAI:
    """Construct a provider-bound model without racing on process-wide env vars."""
    with _PROVIDER_ENV_LOCK:
        previous = os.environ.get("META_AGENT_LLM_PROVIDER")
        os.environ["META_AGENT_LLM_PROVIDER"] = provider
        try:
            timeout = float(task.execution_policy.get("llm_timeout_seconds", 110.0))
            return chat_model(
                task.raw.get("models", {}).get(role), timeout=timeout, max_retries=0
            )
        finally:
            if previous is None:
                os.environ.pop("META_AGENT_LLM_PROVIDER", None)
            else:
                os.environ["META_AGENT_LLM_PROVIDER"] = previous


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


def _contract(task: LoadedTask) -> dict[str, Any]:
    if task.raw.get("system_partition") not in TEMPLATE_PARTITIONS:
        # Keep the frozen baseline request contract byte-for-byte equivalent in
        # structure to its original form. Structured-only template metadata must not
        # alter legacy extraction prompts.
        return {
            "pico": task.pico,
            "targets": task.raw.get("extraction", {}).get("targets", []),
            "allowed_effect_types": task.allowed_effect_types,
            "estimands": task.estimands,
            "model_selection_policy": task.model_selection_policy,
            "exclusion_concepts": task.exclusion_concepts,
        }

    targets = deepcopy(task.raw.get("extraction", {}).get("targets", []))
    for target in targets:
        target["accepted_data_families"] = [
            DATA_FAMILY_ALIASES.get(family, family)
            for family in target.get("accepted_data_families", [])
        ]
    return {
        "pico": task.pico,
        "targets": targets,
        "allowed_effect_types": task.allowed_effect_types,
        "estimands": task.estimands,
        "estimand_policy": task.raw.get("extraction", {}).get("estimand_policy"),
        "model_selection_policy": task.model_selection_policy,
        "exclusion_concepts": task.exclusion_concepts,
    }


def static_clean_document(
    document: DocumentInput,
    *,
    remove_reference_items_only: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a source catalog and remove bibliography content without an LLM.

    The original frozen workflow truncates at References. MinerU documents used
    by the structured cases can put appendices and tables after the bibliography, so
    that workflow instead removes only explicit reference entries.
    """
    path = document.representations.get("content_list")
    if path is None:
        raise ValueError("locator_extractor requires a content_list representation")
    items = json.loads(path.read_text(encoding="utf-8"))
    catalog: list[dict[str, Any]] = []
    removed_source_ids: list[str] = []
    reference_marker: dict[str, Any] | None = None
    reference_tail = False

    for index, item in enumerate(items):
        source_id = f"content_{index:04d}"
        content = _candidate_text(item)
        source_type = str(item.get("type", "unknown")).lower()

        if remove_reference_items_only and source_type == "ref_text":
            removed_source_ids.append(source_id)
            continue
        if reference_tail:
            removed_source_ids.append(source_id)
            continue
        marker = REFERENCE_HEADING.search(content)
        if marker:
            reference_marker = {
                "source_id": source_id,
                "heading": marker.group(0).strip(),
            }
            content = content[: marker.start()].rstrip()
            if not remove_reference_items_only:
                reference_tail = True
            if not content:
                removed_source_ids.append(source_id)
                continue
        if not content.strip():
            removed_source_ids.append(source_id)
            continue

        page_idx = item.get("page_idx")
        label = _as_text(
            item.get("table_caption")
            or item.get("img_caption")
            or _as_text(item.get("text"))[:240]
        ).strip()
        catalog.append(
            {
                "source_id": source_id,
                "source_type": source_type,
                "page": page_idx + 1 if isinstance(page_idx, int) else None,
                "page_index": page_idx,
                "bbox": item.get("bbox"),
                "label": label,
                "content": content,
            }
        )

    audit = {
        "input_items": len(items),
        "retained_items": len(catalog),
        "removed_items": len(removed_source_ids),
        "removed_source_ids": removed_source_ids,
        "reference_marker": reference_marker,
        "policy": (
            "remove exact reference headings and ref_text bibliography items; "
            "retain all other source regions"
            if remove_reference_items_only
            else "truncate at an exact References/Bibliography/Literature Cited heading"
        ),
    }
    return catalog, audit


def locate_relevant_sources(
    task: LoadedTask,
    catalog: list[dict[str, Any]],
    provider: str = "official_openai",
) -> tuple[EvidenceLocations, list[dict[str, Any]], list[dict[str, Any]]]:
    """Locate source regions only; do not plan, extract, or decide eligibility."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a high-recall document locator. Find every supplied source region that may contain data relevant to the PICO and extraction targets. Return source IDs only. Include tables, figures, results text, methods needed to interpret a result, and nearby footnotes. Do not create extraction tasks. Do not copy effect values. Do not decide whether a study or result is eligible, adjusted, poolable, source-only, or suitable for meta-analysis. When uncertain, keep the location for the extractor.""",
            ),
            (
                "user",
                "PICO and target contract:\n{contract}\n\nAvailable source regions:\n{catalog}",
            ),
        ]
    )
    preview_chars = int(task.execution_policy.get("locator_preview_chars", 5000))
    locator_catalog = [
        {
            "source_id": source["source_id"],
            "source_type": source["source_type"],
            "page": source["page"],
            "label": source["label"],
            "content_preview": source["content"][:preview_chars],
        }
        for source in catalog
    ]
    chain = prompt | _llm(task, "locator", provider).with_structured_output(
        EvidenceLocations
    )
    locations = chain.invoke(
        {
            "contract": json.dumps(_contract(task), ensure_ascii=False),
            "catalog": json.dumps(locator_catalog, ensure_ascii=False),
        }
    )

    by_id = {source["source_id"]: source for source in catalog}
    valid_ids = list(dict.fromkeys(source_id for source_id in locations.source_ids if source_id in by_id))
    invalid_ids = list(dict.fromkeys(source_id for source_id in locations.source_ids if source_id not in by_id))
    max_sources = int(task.execution_policy.get("max_locator_sources", 16))
    if max_sources > 0:
        valid_ids = valid_ids[:max_sources]
    selected = [by_id[source_id] for source_id in valid_ids]
    issues: list[dict[str, Any]] = []
    if invalid_ids:
        issues.append({"stage": "locate", "issues": ["invalid_source_ids"], "source_ids": invalid_ids})
    if not selected:
        issues.append({"stage": "locate", "issues": ["no_sources_located"]})
    sanitized = EvidenceLocations(
        source_ids=valid_ids,
        unresolved_location_terms=locations.unresolved_location_terms,
    )
    return sanitized, selected, issues


def _family_instructions(families: set[str]) -> str:
    instructions = {
        "reported_ratio_effect": "Copy every source-reported comparative ratio effect and CI. Preserve whether the source calls it OR, RR, HR, rate ratio, or IRR; do not suppress a candidate because adjustment differs from the protocol.",
        "adjusted_ratio_effect": "Copy adjusted comparative effects and their CIs without recalculation.",
        "time_to_event": "Copy source-reported HRs and CIs with model and adjustment details.",
        "reported_standardized_mean_difference": "Copy source-reported SMD and CI.",
        "binary_arm_counts": "Return source-reported two-arm event counts and denominators in raw_data; do not calculate an effect.",
        "continuous_arm_summary": "Return group means, SDs, and totals in raw_data; do not calculate SMD.",
        "continuous_change_score_summary": "Return change means, SDs, and totals in raw_data; do not calculate SMD.",
        "event_counts_with_person_time": "Return episodes, participant totals, follow-up, and printed rates in raw_data.",
        "percentage_only": "Return printed percentages and available denominators only; never reconstruct counts.",
        "complete_genotype_case_control_counts": "Return the source's rsid, all printed genotype labels, and the full case/control count maps. Do not select a genetic contrast or calculate an OR.",
        "allele_frequency_summary_only": "Return printed case/control sample sizes and allele percentages only. Set genotype_counts_status to not_reported; never infer genotype counts.",
        "independent_continuous_arms": "Return the two named source arms with n, mean, and SD, plus outcome axis, measurement semantics, and unit. Do not calculate MD or SMD.",
        "single_arm_continuous_summary": "Return the named single-arm/subgroup n and every printed mean/SD summary, preserving outcome axis, semantics, unit, and any explicit planned target. Do not construct a comparator or calculate an effect.",
        "diagnostic_four_cell_counts": "Return source-reported TP, FP, FN, and TN for the named test set. Do not calculate sensitivity or specificity.",
        "single_arm_event_counts": "Return the source-reported event count and denominator for the named single arm. Do not calculate a proportion or comparative effect.",
    }
    return "\n".join(f"- {family}: {instructions.get(family, 'Follow the target data contract.')}" for family in sorted(families))


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts)
    return str(content)


def _parse_template_json(message: Any) -> DataFamilyExtraction:
    """Validate selected-template output locally instead of registering a union schema.

    Some OpenAI-compatible endpoints reject valid Pydantic union schemas before
    the model sees them. The allowed templates are still sent to the model,
    then this parser applies the same Pydantic constraints locally.
    """
    text = _message_text(message).strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    elif not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Template extractor did not return one valid JSON object") from exc
    return DataFamilyExtraction.model_validate(payload)


def _template_schemas(families: set[str]) -> str:
    selectable = set(families)
    # This is the explicit generic fallback requested for records that fit none
    # of the task's numeric templates.
    selectable.add("generic_text_evidence")
    return json.dumps(
        {
            family: DATA_FAMILY_TEMPLATE_MODELS[family].model_json_schema()
            for family in sorted(selectable)
            if family in DATA_FAMILY_TEMPLATE_MODELS
        },
        ensure_ascii=False,
    )


def _validate_template_choice(
    extraction: DataFamilyExtraction, families: set[str]
) -> DataFamilyExtraction:
    allowed = set(families) | {"generic_text_evidence"}
    unexpected = sorted(
        {
            row.data_family
            for row in extraction.rows
            if row.data_family not in allowed
        }
    )
    if unexpected:
        raise ValueError(
            "Template extractor selected a data family outside this case contract: "
            + ", ".join(unexpected)
        )
    return extraction


def extract_from_locations(
    task: LoadedTask,
    sources: list[dict[str, Any]],
    feedback: str = "",
    provider: str = "official_openai",
) -> EffectExtraction | DataFamilyExtraction:
    """Extract directly from locations with no intermediate extraction plan."""
    targets = task.raw.get("extraction", {}).get("targets", [])
    if task.raw.get("system_partition") not in TEMPLATE_PARTITIONS:
        # Frozen baseline path: use the original structured-output schema directly.
        # It never uses data-family template selection or the generic fallback
        # as a replacement for a reported HR/RR/OR row.
        families = {
            family
            for target in targets
            for family in target.get("accepted_data_families", [])
        }
        response_model = (
            EffectExtraction
            if families and families.issubset(LEGACY_EFFECT_FAMILIES)
            else DataFamilyExtraction
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are the extraction stage for known-included reports. Your job is transcription and structured extraction, not screening, planning, criticism, study selection, or meta-analysis. Inspect every located source and return every source-supported candidate row matching the target exposure, comparator, and outcome concepts.

The PICO tells you what to look for. Do not suppress a reported candidate merely because one population qualifier is not restated in the located excerpt, because the article population is broader, or because a reported model/effect label differs from the preferred model. Preserve those facts in population_or_subgroup, model_name, adjustment, effect_type, and unresolved_items. Downstream analysis will decide final eligibility and poolability.

Use the target's default quantitative_usability. Use source_only only when the target explicitly requires it or when the accepted data family is inherently source-only. Never infer or manufacture values, never convert effect scales silently, and quote the exact contiguous source text supporting each value. Return one row per distinct target, subgroup, timepoint, comparison, and model.

Data-family instructions:\n{family_instructions}""",
                ),
                (
                    "user",
                    "Extraction contract:\n{contract}\n\nLocated source regions:\n{sources}\n\nRetry feedback:\n{feedback}",
                ),
            ]
        )
        chain = prompt | _llm(task, "extractor", provider).with_structured_output(
            response_model
        )
        return chain.invoke(
            {
                "contract": json.dumps(_contract(task), ensure_ascii=False),
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
                "family_instructions": _family_instructions(families),
            }
        )

    families = {
        DATA_FAMILY_ALIASES.get(family, family)
        for target in targets
        for family in target.get("accepted_data_families", [])
    }
    response_model = (
        EffectExtraction
        if families and families.issubset(LEGACY_EFFECT_FAMILIES)
        else DataFamilyExtraction
    )
    template_families = set(families)
    template_selection_mode = bool(families) and families.issubset(template_families)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are the extraction stage for known-included reports. Your job is transcription and structured extraction, not screening, planning, criticism, study selection, or meta-analysis. Inspect every located source and return every source-supported candidate row matching the target exposure, comparator, and outcome concepts.

The PICO tells you what to look for. Do not suppress a reported candidate merely because one population qualifier is not restated in the located excerpt, because the article population is broader, or because a reported model/effect label differs from the preferred model. Preserve those facts in population_or_subgroup, model_name, adjustment, effect_type, and unresolved_items. Downstream analysis will decide final eligibility and poolability.

If no numeric template fits the located evidence, use generic_text_evidence to preserve a plain-language source description without inventing numeric values.

Use the target's default quantitative_usability when applicable. For source-data templates, use calculator_eligible only for complete source data and extraction_only or requires_analysis_profile when the source lacks a calculable contrast or needs later profile selection. Never infer or manufacture values, never convert effect scales silently, and quote the exact contiguous source text supporting each value. Return one row per distinct target, subgroup, timepoint, comparison, and model.

When selectable template schemas are supplied, choose the matching schema for every row and return exactly one JSON object with keys rows and unresolved_items. Do not wrap JSON in prose or Markdown. Choose generic_text_evidence only when none of the supplied numeric templates fits the source; never use it to avoid transcribing available values.

Data-family instructions:\n{family_instructions}""",
            ),
            (
                "user",
                "Extraction contract:\n{contract}\n\nSelectable extraction template schemas:\n{template_schemas}\n\nLocated source regions:\n{sources}\n\nRetry feedback:\n{feedback}",
            ),
        ]
    )
    inputs = {
        "contract": json.dumps(_contract(task), ensure_ascii=False),
        "template_schemas": (
            _template_schemas(template_families)
            if template_selection_mode
            else "Not applicable for this effect-extraction contract."
        ),
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
        "family_instructions": _family_instructions(families),
    }
    if template_selection_mode:
        raw_chain = prompt | _llm(task, "extractor", provider)
        return _validate_template_choice(
            _parse_template_json(raw_chain.invoke(inputs)), families
        )
    chain = prompt | _llm(task, "extractor", provider).with_structured_output(
        response_model
    )
    return chain.invoke(inputs)


def organize_extractions(
    task: LoadedTask,
    extraction: EffectExtraction | DataFamilyExtraction,
    catalog: list[dict[str, Any]],
    article_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Use the existing deterministic calculators and source verification."""
    return normalize_and_verify_effects(task, [extraction], catalog, article_id)
