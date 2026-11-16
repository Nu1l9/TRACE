from __future__ import annotations

from typing import Any, Protocol

from .evidence_graph import EvidenceGraph, load_evidence_records
from .models import CandidateSet, EvidenceInventory, PoolabilitySet, ResolvedAnalysisProfile, SelectionSet, StudyEffect
from .profile_resolver import resolve_main_analysis_profile
from .reviewer import StudyReviewer, review_candidate_groups
from .statistics import calculate_ratio_meta, qualify_adjusted_ratio_records, qualify_rate_ratio_records


class AnalysisTool(Protocol):
    """Small explicit tool boundary used by the Analysis Agent."""

    name: str


class ResolveAnalysisProfileTool:
    name = "resolve_analysis_profile"

    def invoke(
        self, protocol: dict[str, Any], task: dict[str, Any], analysis_profile: dict[str, Any] | None = None
    ) -> ResolvedAnalysisProfile:
        return resolve_main_analysis_profile(protocol, task, analysis_profile)


class BuildEvidenceGraphTool:
    name = "inspect_evidence_inventory"

    def invoke(
        self, extractor_payloads: list[dict[str, Any]], study_by_article: dict[str, str]
    ) -> tuple[EvidenceGraph, EvidenceInventory]:
        graph = EvidenceGraph(load_evidence_records(extractor_payloads, study_by_article))
        return graph, graph.inventory()


class RetrieveEvidenceCandidatesTool:
    name = "retrieve_evidence_candidates"

    def invoke(self, graph: EvidenceGraph, profile: ResolvedAnalysisProfile) -> CandidateSet:
        return graph.retrieve(profile)


class ReviewStudyLevelEvidenceTool:
    name = "review_study_level_evidence"

    def __init__(self, reviewer: StudyReviewer):
        self.reviewer = reviewer

    def invoke(self, profile: ResolvedAnalysisProfile, candidates: CandidateSet) -> SelectionSet:
        return review_candidate_groups(profile, candidates.groups, self.reviewer)


class AssessPoolabilityTool:
    name = "assess_poolability"

    def invoke(self, profile: ResolvedAnalysisProfile, selections: SelectionSet) -> PoolabilitySet:
        if profile.data_family == "event_counts_with_person_time" and profile.effect_type == "IRR":
            return qualify_rate_ratio_records(profile, selections.selected_records)
        if profile.data_family == "adjusted_ratio_effect":
            return qualify_adjusted_ratio_records(profile, selections.selected_records)
        raise NotImplementedError(
            f"No poolability tool is registered for {profile.data_family}/{profile.effect_type}."
        )


class CalculateMainMetaTool:
    name = "calculate_main_meta"

    def invoke(self, profile: ResolvedAnalysisProfile, effects: list[StudyEffect]):
        return calculate_ratio_meta(profile, effects)


class AnalysisToolRegistry:
    """All internal analysis tools exposed to the analysis sub-agent."""

    def __init__(self, reviewer: StudyReviewer):
        self.resolve_profile = ResolveAnalysisProfileTool()
        self.build_graph = BuildEvidenceGraphTool()
        self.retrieve = RetrieveEvidenceCandidatesTool()
        self.review = ReviewStudyLevelEvidenceTool(reviewer)
        self.poolability = AssessPoolabilityTool()
        self.calculate = CalculateMainMetaTool()
