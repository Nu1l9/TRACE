from __future__ import annotations

import json
import os
from typing import Protocol

from .models import CandidateGroup, EvidenceRecord, ResolvedAnalysisProfile, SelectionDecision, SelectionSet


class StudyReviewer(Protocol):
    mode: str

    def select(self, profile: ResolvedAnalysisProfile, group: CandidateGroup) -> SelectionDecision: ...


def _completeness_score(profile: ResolvedAnalysisProfile, record: EvidenceRecord) -> int:
    score = sum(record.raw_data.get(field) is not None for field in profile.required_raw_fields)
    score += sum(bool(getattr(record, field, "")) for field in profile.required_record_fields)
    score += 3 if record.quantitative_usability == "poolable" else 0
    score += 1 if record.source_location.get("source_snippet") or record.source_location.get("source_quote") else 0
    score += 1 if record.source_location.get("source_type") == "table" else 0
    return score


class DeterministicStudyReviewer:
    """Offline fallback and stable baseline for the GraphRAG reviewer tool."""

    mode = "deterministic"

    def select(self, profile: ResolvedAnalysisProfile, group: CandidateGroup) -> SelectionDecision:
        candidates = group.records
        if not candidates:
            return SelectionDecision(
                study_id=group.study_id,
                candidate_evidence_ids=[],
                status="excluded",
                reason="No candidate records were supplied for this study group.",
                reviewer_mode=self.mode,
            )
        selected = max(candidates, key=lambda item: (_completeness_score(profile, item), item.evidence_id))
        return SelectionDecision(
            study_id=group.study_id,
            candidate_evidence_ids=group.evidence_ids,
            selected_evidence_id=selected.evidence_id,
            status="selected",
            reason=(
                "Deterministic fallback selected the most complete source-grounded candidate within the "
                "study-exposure-outcome graph subgraph."
            ),
            reviewer_mode=self.mode,
        )


class LLMStudyReviewer:
    """Legacy structured reviewer, using the same official OpenAI configuration as structured."""

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

    def select(self, profile: ResolvedAnalysisProfile, group: CandidateGroup) -> SelectionDecision:
        try:
            from openai import OpenAI
            from pydantic import BaseModel, Field
        except ImportError as exc:
            raise RuntimeError("LLM reviewer requires the openai package.") from exc
        if not self.api_key:
            raise RuntimeError("LLM reviewer needs OPENAI_API_KEY.")

        class LLMChoice(BaseModel):
            selected_evidence_id: str | None = Field(default=None)
            status: str
            reason: str

        candidates = [
            {
                "evidence_id": record.evidence_id,
                "outcome": record.outcome,
                "outcome_code": record.outcome_code,
                "comparison": record.comparator,
                "estimand": record.estimand,
                "adjustment_model": record.adjustment_model,
                "follow_up": record.follow_up,
                "data_family": record.data_family,
                "quantitative_usability": record.quantitative_usability,
                "raw_data": record.raw_data,
                "reported_effect_type": record.reported_effect_type,
                "source_location": record.source_location,
            }
            for record in group.records
        ]
        prompt = (
            "You are reviewing one study-exposure-outcome evidence subgraph for a prespecified main meta-analysis. "
            "Use only the supplied profile and candidates. Prefer the record that satisfies required fields, "
            "has the prespecified data family, estimand and comparison, and is source-grounded. "
            "Apply the profile selection_rules before selecting. "
            "Never invent a record ID. If none is eligible, return status='needs_review' with null selection.\n\n"
            f"PROFILE:\n{profile.model_dump_json()}\n\n"
            f"TRIPLET:\n{json.dumps({'study_id': group.study_id, 'exposure': group.exposure, 'outcome': group.outcome, 'comparator': group.comparator}, ensure_ascii=False)}\n\n"
            f"SELECTION_CONTEXT:\n{json.dumps(group.selection_context, ensure_ascii=False)}\n\n"
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
        content = (response.output_text or "").removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            choice = LLMChoice.model_validate_json(content)
        except ValueError:
            return SelectionDecision(
                study_id=group.study_id,
                candidate_evidence_ids=group.evidence_ids,
                status="needs_review",
                reason="LLM response was not a valid selection JSON object.",
                reviewer_mode=self.mode,
            )
        selected_id = choice.selected_evidence_id
        if selected_id not in group.evidence_ids:
            return SelectionDecision(
                study_id=group.study_id,
                candidate_evidence_ids=group.evidence_ids,
                status="needs_review",
                reason="LLM selected an ID outside the supplied study group; selection was rejected.",
                reviewer_mode=self.mode,
            )
        status = "selected" if selected_id else "needs_review"
        return SelectionDecision(
            study_id=group.study_id,
            candidate_evidence_ids=group.evidence_ids,
            selected_evidence_id=selected_id,
            status=status,
            reason=choice.reason,
            reviewer_mode=self.mode,
        )


def review_candidate_groups(
    profile: ResolvedAnalysisProfile, groups: list[CandidateGroup], reviewer: StudyReviewer
) -> SelectionSet:
    decisions = [reviewer.select(profile, group) for group in groups]
    record_by_id = {
        record.evidence_id: record
        for group in groups
        for record in group.records
    }
    selected = [
        record_by_id[decision.selected_evidence_id]
        for decision in decisions
        if decision.status == "selected" and decision.selected_evidence_id in record_by_id
    ]
    # Each graph group is one study/exposure/outcome/comparison/population/
    # timepoint decision unit. The reviewer selects one model within that unit.
    # Distinct outcomes, populations, timepoints, or prespecified analyses from
    # the same study remain available downstream.
    return SelectionSet(decisions=decisions, selected_records=selected)
