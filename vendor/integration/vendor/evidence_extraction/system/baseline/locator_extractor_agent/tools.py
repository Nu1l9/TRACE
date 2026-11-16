from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from constrained_meta_agent.models import DocumentInput, LoadedTask
from llm_endpoint import chat_model

from .extraction_models import DataFamilyExtraction, EffectExtraction
from .organizer import normalize_and_verify_effects
from .tool_models import EvidenceLocations


REFERENCE_HEADING = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:references|bibliography|literature\s+cited)\s*:?[ \t]*$"
)
LEGACY_EFFECT_FAMILIES = {
    "reported_ratio_effect",
    "adjusted_ratio_effect",
    "time_to_event",
}


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


def _contract(task: LoadedTask) -> dict[str, Any]:
    return {
        "pico": task.pico,
        "targets": task.raw.get("extraction", {}).get("targets", []),
        "allowed_effect_types": task.allowed_effect_types,
        "estimands": task.estimands,
        "model_selection_policy": task.model_selection_policy,
        "exclusion_concepts": task.exclusion_concepts,
    }


def static_clean_document(
    document: DocumentInput,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a source catalog and remove the reference tail without an LLM."""
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
            reference_tail = True
            if not content:
                removed_source_ids.append(source_id)
                continue
        if not content.strip():
            removed_source_ids.append(source_id)
            continue

        page_idx = item.get("page_idx")
        source_type = str(item.get("type", "unknown")).lower()
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
        "policy": "truncate at an exact References/Bibliography/Literature Cited heading",
    }
    return catalog, audit


def locate_relevant_sources(
    task: LoadedTask,
    catalog: list[dict[str, Any]],
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
    chain = prompt | _llm(task, "locator").with_structured_output(EvidenceLocations)
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
    }
    return "\n".join(f"- {family}: {instructions.get(family, 'Follow the target data contract.')}" for family in sorted(families))


def extract_from_locations(
    task: LoadedTask,
    sources: list[dict[str, Any]],
    feedback: str = "",
) -> EffectExtraction | DataFamilyExtraction:
    """Extract directly from locations with no intermediate extraction plan."""
    targets = task.raw.get("extraction", {}).get("targets", [])
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
    chain = prompt | _llm(task, "extractor").with_structured_output(response_model)
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


def organize_extractions(
    task: LoadedTask,
    extraction: EffectExtraction | DataFamilyExtraction,
    catalog: list[dict[str, Any]],
    article_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Use the existing deterministic calculators and source verification."""
    return normalize_and_verify_effects(task, [extraction], catalog, article_id)
