"""Project a completed primary synthesis into an independent diagnostic set.

The projection never changes the primary meta-analysis.  It resolves a
different question: which one already-selected effect can represent each study
in a funnel-asymmetry diagnostic without treating multiple reports from the
same study as independent observations.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Protocol

from pydantic import BaseModel, Field

from analysis_core.assignment_reviewer import selection_score
from analysis_core.models import AnalysisDefinition, EvidenceRecord, StudyEffect

from .llm_transport import create_chat_completion
from .models import ReportingBiasProjection, ReportingBiasProjectionDecision


class ReportingBiasSelector(Protocol):
    mode: str

    def project(
        self,
        analysis: AnalysisDefinition,
        effects: list[StudyEffect],
        records_by_id: dict[str, EvidenceRecord],
    ) -> ReportingBiasProjection: ...


def _effect_type_candidates(
    analysis: AnalysisDefinition,
    effects: list[StudyEffect],
    minimum_studies: int,
) -> tuple[str | None, list[StudyEffect], dict[str, int]]:
    """Use protocol effect-type order, never a model-generated preference."""
    coverage = {
        effect_type: len({effect.study_id for effect in effects if effect.effect_type == effect_type})
        for effect_type in sorted({effect.effect_type for effect in effects})
    }
    ordered_types = list(analysis.accepted_effect_types)
    ordered_types.extend(effect_type for effect_type in coverage if effect_type not in ordered_types)
    for effect_type in ordered_types:
        typed = [effect for effect in effects if effect.effect_type == effect_type]
        if len({effect.study_id for effect in typed}) >= minimum_studies:
            return effect_type, typed, coverage
    return None, [], coverage


def _groups_by_study(effects: list[StudyEffect]) -> dict[str, list[StudyEffect]]:
    groups: dict[str, list[StudyEffect]] = defaultdict(list)
    for effect in effects:
        groups[effect.study_id].append(effect)
    return dict(sorted(groups.items()))


def _deterministic_choice(
    analysis: AnalysisDefinition,
    effects: list[StudyEffect],
    records_by_id: dict[str, EvidenceRecord],
) -> StudyEffect:
    def score(effect: StudyEffect):
        record = records_by_id.get(effect.evidence_id)
        if record is None:
            return (0, 0, 0, effect.evidence_id)
        return selection_score(record, analysis)

    return max(effects, key=score)


def _candidate_payload(
    effects: list[StudyEffect], records_by_id: dict[str, EvidenceRecord]) -> list[dict[str, object]]:
    payload = []
    for effect in effects:
        record = records_by_id.get(effect.evidence_id)
        payload.append({
            "evidence_id": effect.evidence_id,
            "outcome": record.outcome if record else "",
            "outcome_code": record.outcome_code if record else "",
            "comparison": record.comparator if record else "",
            "population_or_subgroup": record.population_or_subgroup if record else "",
            "follow_up": record.follow_up if record else "",
            "estimand": record.estimand if record else "",
            "adjustment_model": record.adjustment_model if record else "",
            "effect_type": effect.effect_type,
            "effect": effect.value,
            "ci_lower": effect.ci_lower,
            "ci_upper": effect.ci_upper,
        })
    return payload


def _make_projection(
    analysis: AnalysisDefinition,
    effect_type: str,
    groups: dict[str, list[StudyEffect]],
    records_by_id: dict[str, EvidenceRecord],
    requested: dict[str, str],
    reasons: dict[str, str],
    *,
    reviewer_mode: str,
    summary: str,
    minimum_studies: int,
) -> ReportingBiasProjection:
    decisions: list[ReportingBiasProjectionDecision] = []
    selected: list[StudyEffect] = []
    fallback_count = 0
    for study_id, candidates in groups.items():
        candidate_ids = [effect.evidence_id for effect in candidates]
        requested_id = requested.get(study_id)
        chosen = next((effect for effect in candidates if effect.evidence_id == requested_id), None)
        if chosen is None:
            chosen = _deterministic_choice(analysis, candidates, records_by_id)
            fallback_count += 1
            reason = "No valid LLM selection for this study; applied the prespecified deterministic tie-breaker."
        else:
            reason = reasons.get(study_id) or "Selected as the representative effect for this study."
        selected.append(chosen)
        decisions.append(ReportingBiasProjectionDecision(
            study_id=study_id,
            candidate_evidence_ids=candidate_ids,
            selected_evidence_id=chosen.evidence_id,
            status="selected",
            reason=reason,
            reviewer_mode=reviewer_mode if requested_id == chosen.evidence_id else "deterministic",
        ))
    return ReportingBiasProjection(
        status="completed",
        effect_type=effect_type,
        source_effect_count=sum(len(group) for group in groups.values()),
        independent_study_count=len(groups),
        minimum_studies=minimum_studies,
        selected_evidence_ids=[effect.evidence_id for effect in selected],
        selected_effects=selected,
        decisions=decisions,
        reviewer_mode=reviewer_mode,
        summary=f"{summary} {fallback_count} study groups used the deterministic tie-breaker.",
        policy=(
            "Protocol effect-type priority; one already-selected effect per study; "
            "LLM choice constrained to supplied candidate IDs."
        ),
    )


def _not_assessable_projection(
    effects: list[StudyEffect],
    minimum_studies: int,
    coverage: dict[str, int],
) -> ReportingBiasProjection:
    return ReportingBiasProjection(
        status="not_assessable",
        source_effect_count=len(effects),
        independent_study_count=len({effect.study_id for effect in effects}),
        minimum_studies=minimum_studies,
        reviewer_mode="deterministic",
        summary="No protocol-allowed single effect type reached the required independent-study threshold.",
        policy="Protocol effect-type priority; one already-selected effect per study.",
        coverage_by_effect_type=coverage,
    )


class DeterministicReportingBiasSelector:
    """API-free baseline that applies the same projection contract as the LLM."""

    mode = "deterministic"

    def __init__(self, *, minimum_studies: int = 10) -> None:
        self.minimum_studies = minimum_studies

    def project(self, analysis, effects, records_by_id) -> ReportingBiasProjection:
        effect_type, typed_effects, coverage = _effect_type_candidates(
            analysis, effects, self.minimum_studies
        )
        if effect_type is None:
            return _not_assessable_projection(effects, self.minimum_studies, coverage)
        groups = _groups_by_study(typed_effects)
        return _make_projection(
            analysis,
            effect_type,
            groups,
            records_by_id,
            requested={},
            reasons={},
            reviewer_mode=self.mode,
            summary="Deterministically projected one protocol-compatible effect per study.",
            minimum_studies=self.minimum_studies,
        )


class LLMReportingBiasSelector(DeterministicReportingBiasSelector):
    """LLM reviewer constrained to a one-effect-per-study diagnostic projection."""

    mode = "llm"

    def __init__(
        self,
        model: str = "gpt-5.4-mini",
        reasoning_effort: str = "xhigh",
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 3,
        minimum_studies: int = 10,
    ) -> None:
        super().__init__(minimum_studies=minimum_studies)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.max_retries = max_retries
        self.calls_made = 0

    def project(self, analysis, effects, records_by_id) -> ReportingBiasProjection:
        effect_type, typed_effects, coverage = _effect_type_candidates(
            analysis, effects, self.minimum_studies
        )
        if effect_type is None:
            return _not_assessable_projection(effects, self.minimum_studies, coverage)
        groups = _groups_by_study(typed_effects)
        if all(len(group) == 1 for group in groups.values()):
            return _make_projection(
                analysis, effect_type, groups, records_by_id, {}, {},
                reviewer_mode="deterministic",
                summary="Each study already had one protocol-compatible diagnostic effect; no LLM call was needed.",
                minimum_studies=self.minimum_studies,
            )
        from openai import OpenAI

        if not self.api_key:
            raise RuntimeError("LLM reporting-bias selector needs an API key.")

        class RawDecision(BaseModel):
            study_id: str
            selected_evidence_id: str
            reason: str

        class RawProjection(BaseModel):
            decisions: list[RawDecision] = Field(default_factory=list)
            summary: str

        payload = {
            "analysis": analysis.model_dump(),
            "required_effect_type": effect_type,
            "selection_policy": (
                "Select exactly one candidate from each study. The candidates are already eligible for the "
                "primary analysis. Prefer the overall PICO-matching population over an overlapping subgroup, "
                "the longest eligible follow-up, and the most fully adjusted model. Do not invent identifiers."
            ),
            "study_groups": [
                {"study_id": study_id, "candidates": _candidate_payload(group, records_by_id)}
                for study_id, group in groups.items()
            ],
        }
        prompt = (
            "You are the publication-bias diagnostic selector for a prespecified meta-analysis. Your task is "
            "not to change the primary analysis and not to calculate a pooled result. Select one representative "
            "already-selected effect per study so that each funnel-plot point comes from a distinct study. "
            "Use only the supplied evidence IDs.\n\n"
            "Return JSON only: {\"decisions\":[{\"study_id\":...,\"selected_evidence_id\":...,\"reason\":...}],"
            "\"summary\":...}.\n\n"
            f"PAYLOAD={json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
        )
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
            component=f"ReportingBias/{analysis.analysis_id}",
            max_retries=self.max_retries,
        )
        content = (response.choices[0].message.content or "").removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            raw = RawProjection.model_validate_json(content)
        except ValueError as exc:
            raise RuntimeError("Reporting-bias selector did not return valid JSON.") from exc
        requested = {item.study_id: item.selected_evidence_id for item in raw.decisions if item.study_id in groups}
        reasons = {item.study_id: item.reason for item in raw.decisions if item.study_id in groups}
        return _make_projection(
            analysis, effect_type, groups, records_by_id, requested, reasons,
            reviewer_mode=self.mode,
            summary=raw.summary,
            minimum_studies=self.minimum_studies,
        )
