from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from analysis_core.models import (
    AnalysisDefinition,
    EvidenceAssignmentDecision,
    EvidenceInventory,
    MainMetaStatistics,
    PoolabilitySet,
    ResolvedAnalysisPlan,
    StudyEffect,
    ToolTraceEntry,
)

from .method_selector import MethodSelectionDecision


class TopicCandidateInventory(BaseModel):
    analysis_id: str
    candidate_record_count: int
    candidate_context_count: int
    candidate_study_count: int
    embedding_retrieval_summary: str


class PlannerAction(BaseModel):
    action: Literal["review_topic", "assess_poolability", "calculate_meta", "block_topic"]
    analysis_id: str
    reason: str


class PlannerDecision(BaseModel):
    phase: Literal["retrieval", "feasibility"]
    actions: list[PlannerAction] = Field(default_factory=list)
    summary: str
    planner_mode: Literal["deterministic", "llm"]


class TopicMemory(BaseModel):
    analysis_id: str
    title: str
    kind: str
    minimum_studies: int
    state: Literal["retrieved", "reviewed", "poolability_checked", "ready", "completed", "blocked"]
    candidate_record_count: int = 0
    candidate_context_count: int = 0
    candidate_study_count: int = 0
    selected_record_count: int = 0
    selected_study_count: int = 0
    poolable_effect_count: int = 0
    poolable_study_count: int = 0
    next_allowed_actions: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class AnalysisMemory(BaseModel):
    topics: dict[str, TopicMemory] = Field(default_factory=dict)
    planner_decisions: list[PlannerDecision] = Field(default_factory=list)


class TopicReviewResult(BaseModel):
    analysis_id: str
    assignments: list[EvidenceAssignmentDecision]
    selected_evidence_ids: list[str]
    source_resolutions: list[dict[str, object]] = Field(default_factory=list)
    source_resolution_issues: dict[str, str] = Field(default_factory=dict)
    summary: str
    reviewer_mode: Literal["deterministic", "llm"]


class ReportingBiasAssessment(BaseModel):
    """Post-pooling assessment of small-study effects, not study-level RoB.

    ``funnel_plot_points`` is an analysis artefact rather than an input to the
    decision: it makes the exact study-level coordinates used by the diagnostic
    inspectable and ready for a manuscript funnel plot.
    """

    status: Literal[
        "completed",
        "not_assessable",
        "incompatible_effect_set",
        "unavailable",
        "failed",
    ]
    tool: str = "metafor_reporting_bias"
    k: int
    independent_study_count: int
    minimum_studies: int
    effect_type: str | None = None
    model: str | None = None
    methods_run: list[str] = Field(default_factory=list)
    precision_diagnostics: dict[str, float | int | str | None] = Field(default_factory=dict)
    funnel_asymmetry: dict[str, float | str | None] = Field(default_factory=dict)
    funnel_plot_points: list[dict[str, float | str]] = Field(default_factory=list)
    figure_artifacts: dict[str, str] = Field(default_factory=dict)
    trim_and_fill: dict[str, float | int | str | None] = Field(default_factory=dict)
    interpretation: str
    caveats: list[str] = Field(default_factory=list)
    error: str | None = None


class ReportingBiasProjectionDecision(BaseModel):
    """One constrained study-level choice before funnel diagnostics."""

    study_id: str
    candidate_evidence_ids: list[str] = Field(default_factory=list)
    selected_evidence_id: str | None = None
    status: Literal["selected", "excluded", "needs_review"]
    reason: str
    reviewer_mode: Literal["deterministic", "llm"]


class ReportingBiasProjection(BaseModel):
    """An auditable, one-effect-per-study view of a completed primary analysis."""

    status: Literal["completed", "not_assessable"]
    effect_type: str | None = None
    source_effect_count: int
    independent_study_count: int
    minimum_studies: int
    selected_evidence_ids: list[str] = Field(default_factory=list)
    selected_effects: list[StudyEffect] = Field(default_factory=list)
    decisions: list[ReportingBiasProjectionDecision] = Field(default_factory=list)
    reviewer_mode: Literal["deterministic", "llm"]
    summary: str
    policy: str
    coverage_by_effect_type: dict[str, int] = Field(default_factory=dict)


class RobustnessDiagnostics(BaseModel):
    """Post-pooling influence and model-robustness diagnostics.

    These diagnostics flag records for review. They never remove evidence or
    overwrite the prespecified primary pooled result.
    """

    status: Literal["completed", "not_assessable", "unavailable", "failed"]
    tool: str = "metafor_robustness_diagnostics"
    k: int
    independent_study_count: int
    minimum_studies: int
    effect_type: str | None = None
    primary_model: str | None = None
    methods_run: list[str] = Field(default_factory=list)
    model_comparison: dict[str, float | str | None] = Field(default_factory=dict)
    leave_one_out: list[dict[str, float | str | None]] = Field(default_factory=list)
    influential_studies: list[dict[str, float | str | None]] = Field(default_factory=list)
    heterogeneity_drivers: list[dict[str, float | str | None]] = Field(default_factory=list)
    stability: str
    caveats: list[str] = Field(default_factory=list)
    error: str | None = None


class ForestPlotArtifact(BaseModel):
    """Presentation artifact from completed inputs; never a decision input."""

    status: Literal["completed", "not_assessable", "skipped", "failed"]
    tool: str = "render_forest_plot"
    k: int
    independent_study_count: int
    effect_type: str | None = None
    scale: Literal["ratio", "difference"] | None = None
    displayed_pooled_effect: float | None = None
    displayed_ci_lower: float | None = None
    displayed_ci_upper: float | None = None
    figure_artifacts: dict[str, str] = Field(default_factory=dict)
    reason: str
    notes: list[str] = Field(default_factory=list)
    error: str | None = None


class TopicAnalysisResult(BaseModel):
    interval_sensitivity: dict[str, object] = Field(default_factory=dict)
    analysis: AnalysisDefinition
    status: Literal["completed", "insufficient_data", "blocked"]
    candidate_inventory: TopicCandidateInventory
    assignments: list[EvidenceAssignmentDecision]
    selected_evidence_ids: list[str]
    source_resolutions: list[dict[str, object]] = Field(default_factory=list)
    source_resolution_issues: dict[str, str] = Field(default_factory=dict)
    poolability: PoolabilitySet
    statistics: MainMetaStatistics | None = None
    method_selection: MethodSelectionDecision | None = None
    reporting_bias_projection: ReportingBiasProjection | None = None
    reporting_bias: ReportingBiasAssessment | None = None
    robustness: RobustnessDiagnostics | None = None
    forest_plot: ForestPlotArtifact | None = None
    warnings: list[str] = Field(default_factory=list)


class PipelineAnalysisAgentResult(BaseModel):
    status: Literal["completed", "partial", "blocked"]
    plan: ResolvedAnalysisPlan
    inventory: EvidenceInventory
    retrieval_inventory: list[TopicCandidateInventory]
    memory: AnalysisMemory
    analyses: list[TopicAnalysisResult]
    trace: list[ToolTraceEntry] = Field(default_factory=list)
