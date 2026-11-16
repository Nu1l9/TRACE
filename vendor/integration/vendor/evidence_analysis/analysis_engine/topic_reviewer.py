from __future__ import annotations

import json
import os
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from metapipeline.evidence_context import needs_source_resolution, numeric_fields, resolution_problem

from .eligibility import eligibility_reasons, selection_score
from analysis_core.models import AnalysisDefinition, CandidateSet

from .llm_transport import create_chat_completion
from .models import TopicReviewResult



class FieldSourceSupport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str
    source_id: str
    explanation: str


class SourceResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_id: str
    decision: Literal["supported", "unsupported", "unresolved"]
    reason: str
    field_support: list[FieldSourceSupport] = Field(default_factory=list)


def _parse_source_resolutions(values: list[dict]) -> list[dict]:
    """A malformed resolution quarantines its record, not independent evidence."""
    result = []
    for item in values:
        try:
            result.append(SourceResolution.model_validate(item).model_dump())
        except ValueError as exc:
            result.append({"evidence_id": str(item.get("evidence_id") or ""),
                           "decision": "unresolved", "reason": "Invalid source resolution structure.",
                           "field_support": [], "validation_error": str(exc)})
    return result


def _record_candidate(record) -> dict:
    return {
        "raw_data": record.raw_data,
        "effect": {"value": record.reported_effect_value,
                   "ci_lower": record.ci_lower, "ci_upper": record.ci_upper},
        "source_location": record.source_location,
    }


def _needs_resolution(record) -> bool:
    return needs_source_resolution((record.source_location or {}).get("review_context"))


def _record_resolution_problem(record, resolutions: list[dict]) -> str | None:
    matching = [item for item in resolutions if item.get("evidence_id") == record.evidence_id]
    if len(matching) > 1:
        return "duplicate source resolutions require a single unambiguous review"
    return resolution_problem(_record_candidate(record), matching[0] if matching else None)


def _numeric_field_names(record) -> list[str]:
    candidate = _record_candidate(record)
    return sorted({**numeric_fields(candidate["raw_data"]),
                   **numeric_fields(candidate["effect"], "effect")})


class TopicReviewer(Protocol):
    mode: str

    def review(self, analysis: AnalysisDefinition, candidates: CandidateSet) -> TopicReviewResult: ...


def _reviewable_groups(candidates: CandidateSet) -> list:
    """Return all graph-retrieved Evidence without an inclusion hint filter.

    ``quantitative_usability`` is an extraction-layer annotation, not an
    analysis input. The reviewer and pairwise tool must decide usability from
    PICO alignment and the supplied source fields instead of a gold-adjacent
    preclassification.
    """
    return list(candidates.groups)


def build_topic_reviewer_prompt(
    analysis: AnalysisDefinition, groups: list, *, hide_calculation_details: bool = False,
    allow_expansion: bool = False,
) -> str:
    """Build the analysis-only body used by the local reviewer.

    Provenance, extraction diagnostics, source blocks, and review handoff data
    remain in local artifacts.  This stage receives only fields that can affect
    statistical eligibility or the choice between competing estimates.
    """
    def analysis_raw_data(record):
        return {key: value for key, value in record.raw_data.items() if not key.startswith("_")}

    def analysis_candidate(record):
        if "Reviewer input: compact_effects" in analysis.selection_rules:
            # This opt-in view keeps every candidate ID and every field needed
            # to choose a study effect, without repeating the full extraction
            # row (including long covariate lists) hundreds of times.  The
            # unabridged source record remains in the graph for calculation.
            raw = record.raw_data
            def concise(value, limit=120):
                text = str(value or "").strip()
                return text if len(text) <= limit else text[:limit - 1] + "…"
            values = {
                "evidence_id": record.evidence_id,
                "study_id": record.study_id,
                "outcome": record.outcome,
                "exposure": concise(record.intervention_or_exposure),
                "comparison": concise(record.comparator),
                "population": concise(record.population_or_subgroup),
                "timepoint": record.follow_up,
                "effect_type": record.reported_effect_type,
                "effect_value": record.reported_effect_value,
                "ci_lower": record.ci_lower,
                "ci_upper": record.ci_upper,
                "data_type": raw.get("data_type"),
                "exposure_unit": raw.get("exposure_unit"),
                "adjustment_model": concise(record.adjustment_model),
                "adjustment_set": concise(raw.get("adjustment_set")),
            }
            return {key: value for key, value in values.items() if value not in (None, "", [], {})}
        values = {
            "evidence_id": record.evidence_id,
            "study_id": record.study_id,
            "outcome": record.outcome,
            "outcome_code": record.outcome_code,
            "comparison": record.comparator,
            "population": record.population_or_subgroup,
            "timepoint": record.follow_up,
            "estimand": record.estimand,
            "adjustment_model": record.adjustment_model,
            "data_family": record.data_family,
            "effect_type": record.reported_effect_type,
            "reported_effect": record.reported_effect_value,
            "ci_lower": record.ci_lower,
            "ci_upper": record.ci_upper,
            "raw_data": analysis_raw_data(record),
            "analysis_metadata": record.raw_data.get("_analysis_metadata", {}),
        }
        return {key: value for key, value in values.items() if value not in (None, "", [], {})}

    contexts = [
        {
            "context_id": group.group_id,
            "candidates": [analysis_candidate(record) for record in group.records],
        }
        for group in groups
    ]
    reviewer_analysis_payload: dict[str, object] | None = None
    reviewer_analysis = analysis
    if hide_calculation_details:
        # Evidence adjudication does not need the runtime calculator route.
        # Remove it entirely instead of exposing a placeholder that could be
        # mistaken for a method recommendation.
        reviewer_analysis_payload = analysis.model_dump()
        reviewer_analysis_payload.pop("pooling_model", None)
        reviewer_analysis_payload.pop("pooling_scale", None)
    serialized_analysis = (
        json.dumps(reviewer_analysis_payload, ensure_ascii=False, separators=(",", ":"))
        if reviewer_analysis_payload is not None
        else reviewer_analysis.model_dump_json()
    )
    output_instruction = (
        'Return JSON only: {"selected_evidence_ids":[...],"summary":"..."}. Do not invent IDs.\n\n'
    )
    if allow_expansion:
        output_instruction = (
            "Request expansion only when the supplied analysis records indicate that a relevant competing "
            "estimate may be missing from this graph slice. Return JSON only: "
            '{"selected_evidence_ids":[...],"summary":"...","request_expansion":false,"expansion_reason":"..."}. '
            "Use a JSON boolean and supplied IDs only.\n\n"
        )
    return (
        "You are the local Evidence reviewer for one prespecified meta-analysis topic. Select analysis "
        "records only from the supplied local GraphRAG contexts. Respect outcome, comparison, population, "
        "estimand, data-family, effect-type, and any metadata constraints in selection_rules. In a blinded "
        "standardized package, target_id may be unresolved, outcome_code may be null, and estimand may be "
        "the transport label source_reported_association; in that case judge the corresponding semantic "
        "requirements from the PICO, outcome wording, comparison, timepoint, and structured analysis fields. Select a record only when its numeric fields "
        "are sufficient for the declared quantitative data family; exclude incomplete, composite, withdrawal-only, "
        "or non-comparable records. Respect any explicit cohort-overlap metadata and prefer an overall eligible "
        "cohort to its overlapping subgroup. Within a context, select at most one competing model; prefer the "
        "most fully adjusted eligible model. Do not calculate any pooled effect.\n\n"
        + output_instruction
        +
        f"ANALYSIS={serialized_analysis}\n\n"
        f"LOCAL_GRAPH_CONTEXTS={json.dumps(contexts, ensure_ascii=False, separators=(',', ':'))}"
    )


def _make_result(
    analysis: AnalysisDefinition,
    candidates: CandidateSet,
    selected_ids: set[str],
    *,
    summary: str,
    reviewer_mode: str,
    source_resolutions: list[dict] | None = None,
    enforce_source_resolutions: bool = True,
) -> TopicReviewResult:
    resolutions = list(source_resolutions or [])
    by_id = {}
    duplicate_ids = set()
    for item in resolutions:
        evidence_id = item.get("evidence_id")
        if evidence_id in by_id:
            duplicate_ids.add(evidence_id)
        by_id[evidence_id] = item
    selected_ids = set(selected_ids)
    resolution_issues = {}
    if enforce_source_resolutions:
        for group in candidates.groups:
            for record in group.records:
                problem = resolution_problem(_record_candidate(record), by_id.get(record.evidence_id))
                if record.evidence_id in duplicate_ids:
                    problem = "duplicate source resolutions require a single unambiguous review"
                if problem:
                    selected_ids.discard(record.evidence_id)
                    resolution_issues[record.evidence_id] = problem
    assignments = []
    for group in candidates.groups:
        for record in group.records:
            selected = record.evidence_id in selected_ids
            assignments.append({
                "analysis_id": analysis.analysis_id,
                "evidence_id": record.evidence_id,
                "study_id": record.study_id,
                "role": "included" if selected else "excluded",
                "reason": (
                    resolution_issues.get(record.evidence_id)
                    or ("Selected after explicit source resolution: " + str(by_id[record.evidence_id].get("reason"))
                        if selected and enforce_source_resolutions and _needs_resolution(record) else
                        "Selected by the topic reviewer from this local GraphRAG context."
                        if selected else "Not selected from this local GraphRAG context.")
                ),
                "reviewer_mode": reviewer_mode,
                "selected_within_context": selected,
            })
    return TopicReviewResult(
        analysis_id=analysis.analysis_id,
        assignments=assignments,
        selected_evidence_ids=sorted(selected_ids),
        source_resolutions=resolutions,
        source_resolution_issues=resolution_issues,
        summary=summary,
        reviewer_mode=reviewer_mode,
    )


class DeterministicTopicReviewer:
    """Stable local-context baseline for the topic-review tool."""

    mode = "deterministic"

    def review(self, analysis: AnalysisDefinition, candidates: CandidateSet) -> TopicReviewResult:
        selected_ids: set[str] = set()
        for group in _reviewable_groups(candidates):
            eligible = [record for record in group.records if not eligibility_reasons(record, analysis)
                        and not _record_resolution_problem(record, [])]
            if eligible:
                selected_ids.add(max(eligible, key=lambda record: selection_score(record, analysis)).evidence_id)
        return _make_result(
            analysis,
            candidates,
            selected_ids,
            summary=(
                f"Deterministically selected {len(selected_ids)} records across {len(candidates.groups)} local contexts."
            ),
            reviewer_mode=self.mode,
        )


class LLMTopicReviewer:
    """One LLM call per analysis topic, operating only on its retrieved subgraph."""

    mode = "llm"

    def __init__(
        self,
        model: str = "gpt-5.4-mini",
        reasoning_effort: str = "xhigh",
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 3,
        always_review: bool = False,
        hide_calculation_details: bool = False,
    ):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.max_retries = max_retries
        self.always_review = always_review
        self.hide_calculation_details = hide_calculation_details
        self.calls_made = 0

    def review(self, analysis: AnalysisDefinition, candidates: CandidateSet) -> TopicReviewResult:
        from openai import OpenAI

        if not self.api_key:
            raise RuntimeError("LLM topic reviewer needs OPENAI_API_KEY.")

        class RawReview(BaseModel):
            selected_evidence_ids: list[str] = Field(default_factory=list)
            summary: str
            source_resolutions: list[dict] = Field(default_factory=list)

        reviewable_groups = _reviewable_groups(candidates)
        uncontested_ids = {
            group.records[0].evidence_id
            for group in reviewable_groups
            if len(group.records) == 1 and not _needs_resolution(group.records[0])
        }
        contested_groups = [group for group in reviewable_groups if len(group.records) > 1 or any(_needs_resolution(record) for record in group.records)]
        if not contested_groups and not self.always_review:
            return _make_result(
                analysis,
                candidates,
                uncontested_ids,
                summary=(
                    f"All {len(uncontested_ids)} local contexts had one protocol-eligible Evidence; "
                    "no LLM model-choice call was necessary."
                ),
                reviewer_mode="deterministic",
            )

        groups_for_llm = reviewable_groups if self.always_review else contested_groups
        prompt = build_topic_reviewer_prompt(
            analysis, groups_for_llm, hide_calculation_details=self.hide_calculation_details
        )
        # Keep the local-review prompt portable across OpenAI-compatible
        # providers; the benchmark JSON output contract remains unchanged.
        self.calls_made += 1
        request = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            max_retries=0,
            timeout=180.0,
        )
        response = create_chat_completion(
            client,
            request,
            component=f"Reviewer/{analysis.analysis_id}",
            max_retries=self.max_retries,
        )
        content = (response.choices[0].message.content or "").removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            raw = RawReview.model_validate_json(content)
        except ValueError as exc:
            raise RuntimeError(f"Topic reviewer returned invalid JSON for {analysis.analysis_id}.") from exc

        known_ids = {record.evidence_id for group in groups_for_llm for record in group.records}
        requested_ids = set(raw.selected_evidence_ids) & known_ids
        source_resolutions = _parse_source_resolutions(raw.source_resolutions)
        # Enforce one selected model per local graph context even on malformed output.
        selected_ids = set() if self.always_review else set(uncontested_ids)
        for group in groups_for_llm:
            requested = [record for record in group.records if record.evidence_id in requested_ids
                         and not _record_resolution_problem(record, source_resolutions)]
            if requested:
                selected_ids.add(max(requested, key=lambda record: selection_score(record, analysis)).evidence_id)
        return _make_result(
            analysis,
            candidates,
            selected_ids,
            summary=(
                f"{raw.summary} "
                + (
                    f"LLM reviewed all {len(groups_for_llm)} local GraphRAG contexts without a usability hint."
                    if self.always_review
                    else f"Automatically retained {len(uncontested_ids)} uncontested local contexts; LLM reviewed {len(contested_groups)} competing-model contexts."
                )
            ),
            reviewer_mode=self.mode,
            source_resolutions=source_resolutions,
        )
