from __future__ import annotations

import json
import os
import re
from typing import Protocol

from .models import AnalysisDefinition, CandidateGroup, EvidenceRecord


class EvidenceAssignmentReviewer(Protocol):
    mode: str

    def assign(
        self, analysis: AnalysisDefinition, group: CandidateGroup
    ) -> tuple[str | None, str, str]: ...


def _normalise(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _matches_any(value: object, terms: list[str]) -> bool:
    if not terms:
        return True
    text = _normalise(value)
    padded_text = f" {text} "
    text_tokens = set(text.split())
    return any(
        (
            term_text
            and (
                text == term_text
                or f" {term_text} " in padded_text
                or set(term_text.split()).issubset(text_tokens)
            )
        )
        for term in terms
        if (term_text := _normalise(term))
    )


def _field_value(record: EvidenceRecord, field: str | None) -> object:
    if not field:
        return ""
    value = getattr(record, field, None)
    return value if value not in (None, "") else record.raw_data.get(field, "")


def _overall_population(record: EvidenceRecord, analysis: AnalysisDefinition) -> bool:
    terms = analysis.population_terms or [
        "overall",
        "all participants",
        "full cohort",
        "entire cohort",
        "total cohort",
    ]
    value = _normalise(record.population_or_subgroup)
    return not value or _matches_any(value, terms)


def eligibility_reasons(record: EvidenceRecord, analysis: AnalysisDefinition) -> list[str]:
    reasons: list[str] = []
    if record.target_id != analysis.target_id:
        reasons.append("target does not match")
    if analysis.eligible_outcome_codes and record.outcome_code not in analysis.eligible_outcome_codes:
        reasons.append("outcome is outside the analysis scope")
    if record.data_family not in analysis.eligible_data_families:
        reasons.append("data family is not accepted")
    # Raw arm-level records (for example 2x2 event counts) intentionally do
    # not carry a reported OR/RR.  Their requested effect is derived only
    # after selection by the pairwise statistics tool, so an absent source
    # effect type must not exclude an otherwise PICO-compatible record.
    if (
        analysis.accepted_effect_types
        and record.reported_effect_type
        and record.reported_effect_type not in analysis.accepted_effect_types
    ):
        reasons.append("effect type is not accepted")
    if analysis.estimands and record.estimand not in analysis.estimands:
        reasons.append("estimand does not match")
    if analysis.exposure_terms and not _matches_any(record.intervention_or_exposure, analysis.exposure_terms):
        reasons.append("exposure does not match")
    if analysis.comparison_terms and not _matches_any(record.comparator, analysis.comparison_terms):
        reasons.append("comparison does not match")
    if analysis.population_policy == "overall_only" and not _overall_population(record, analysis):
        reasons.append("record is a subgroup rather than the primary overall population")
    if analysis.population_policy == "dimension_level" and not _matches_any(
        record.population_or_subgroup, analysis.population_terms
    ):
        reasons.append("record does not match the declared subgroup level")
    if analysis.kind == "subgroup" and analysis.dimension_field != "population_or_subgroup" and not _matches_any(
        _field_value(record, analysis.dimension_field), analysis.dimension_terms
    ):
        reasons.append("record does not match the declared subgroup level")
    if analysis.kind == "sensitivity" and not _matches_any(
        _field_value(record, analysis.dimension_field), analysis.dimension_terms
    ):
        reasons.append("record does not match the sensitivity-analysis level")
    return reasons


def selection_score(record: EvidenceRecord, analysis: AnalysisDefinition) -> tuple[int, int, str]:
    model = _normalise(record.adjustment_model)
    adjustment_score = sum(
        marker in model
        for marker in ("fully adjusted", "most adjusted", "multivariable", "model 3", "model 4")
    )
    completeness = sum(
        value is not None
        for value in (record.reported_effect_value, record.ci_lower, record.ci_upper)
    )
    completeness += int(bool(record.source_location))
    return adjustment_score, completeness, record.evidence_id


class DeterministicEvidenceAssignmentReviewer:
    """Stable offline baseline for structured evidence-to-analysis assignment."""

    mode = "deterministic"

    def assign(
        self, analysis: AnalysisDefinition, group: CandidateGroup
    ) -> tuple[str | None, str, str]:
        eligible = [
            record for record in group.records
            if not eligibility_reasons(record, analysis)
        ]
        if not eligible:
            sample_reasons = eligibility_reasons(group.records[0], analysis) if group.records else []
            return None, "excluded", "; ".join(sample_reasons) or "No eligible record was supplied."
        selected = max(eligible, key=lambda record: selection_score(record, analysis))
        return (
            selected.evidence_id,
            "included",
            "Matches this analysis definition; selected as the preferred model within its evidence context.",
        )


class LLMEvidenceAssignmentReviewer:
    """Legacy per-context reviewer, retained for experiments outside the structured global path."""

    mode = "llm"

    def __init__(
        self,
        model: str = "gpt-5.4-mini",
        reasoning_effort: str = "xhigh",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    def assign(
        self, analysis: AnalysisDefinition, group: CandidateGroup
    ) -> tuple[str | None, str, str]:
        try:
            from openai import OpenAI
            from pydantic import BaseModel
        except ImportError as exc:
            raise RuntimeError("LLM assignment reviewer requires the openai package.") from exc
        if not self.api_key:
            raise RuntimeError("LLM assignment reviewer needs OPENAI_API_KEY.")

        class Choice(BaseModel):
            role: str
            selected_evidence_id: str | None = None
            reason: str

        candidates = [
            {
                "evidence_id": record.evidence_id,
                "study_id": record.study_id,
                "exposure": record.intervention_or_exposure,
                "outcome": record.outcome,
                "outcome_code": record.outcome_code,
                "comparison": record.comparator,
                "population_or_subgroup": record.population_or_subgroup,
                "follow_up": record.follow_up,
                "estimand": record.estimand,
                "adjustment_model": record.adjustment_model,
                "effect_type": record.reported_effect_type,
                "quantitative_usability": record.quantitative_usability,
                "source_location": record.source_location,
            }
            for record in group.records
        ]
        prompt = (
            "You assign source-grounded meta-analysis evidence to one prespecified analysis. "
            "Decide whether this evidence context belongs in the supplied analysis definition. "
            "If it belongs, select exactly one competing model from this context according to model_policy "
            "and selection_rules. Distinct outcomes, populations and timepoints are handled in separate "
            "contexts. Use only supplied fields and never invent an evidence ID. Return exactly one JSON "
            "object with keys role, selected_evidence_id, and reason. role must be included, excluded, or "
            "needs_review.\n\n"
            f"ANALYSIS_DEFINITION:\n{analysis.model_dump_json()}\n\n"
            f"GRAPH_CONTEXT:\n{json.dumps(group.selection_context, ensure_ascii=False)}\n\n"
            f"CANDIDATES:\n{json.dumps(candidates, ensure_ascii=False)}"
        )
        response = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            max_retries=0,
            timeout=120.0,
        ).responses.create(
            model=self.model,
            input=[{"role": "user", "content": prompt}],
            reasoning={"effort": self.reasoning_effort},
            store=False,
        )
        content = response.output_text or ""
        content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            parsed = Choice.model_validate_json(content)
        except ValueError:
            return None, "needs_review", "LLM response was not a valid assignment JSON object."
        if parsed.role not in {"included", "excluded", "needs_review"}:
            return None, "needs_review", "LLM returned an unsupported assignment role."
        if parsed.role == "included" and parsed.selected_evidence_id not in group.evidence_ids:
            return None, "needs_review", "LLM selected an ID outside the supplied GraphRAG context."
        return parsed.selected_evidence_id, parsed.role, parsed.reason
