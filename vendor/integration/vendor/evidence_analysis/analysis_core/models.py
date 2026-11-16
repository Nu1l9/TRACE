from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ResolvedAnalysisProfile(BaseModel):
    """Runtime profile compiled from allowed benchmark inputs, never from gold."""

    analysis_id: str
    task_id: str
    protocol_id: str
    target_id: str
    outcome: str
    comparison: str
    data_family: str
    effect_type: str
    pooling_model: Literal["fixed_effect", "random_effects", "unspecified"]
    pooling_scale: str = "log_ratio_inverse_variance"
    minimum_studies: int = 2
    required_raw_fields: list[str] = Field(default_factory=list)
    required_record_fields: list[str] = Field(default_factory=list)
    exposure_query_terms: list[str] = Field(default_factory=list)
    outcome_query_terms: list[str] = Field(default_factory=list)
    eligible_estimands: list[str] = Field(default_factory=list)
    eligible_outcome_codes: list[str] = Field(default_factory=list)
    accepted_effect_types: list[str] = Field(default_factory=list)
    one_effect_per_triplet: bool = True
    semantic_match_threshold: float = 0.3
    selection_rules: list[str] = Field(default_factory=list)
    derivation_trace: list[str] = Field(default_factory=list)


class EvidenceRecord(BaseModel):
    evidence_id: str
    article_id: str
    study_id: str
    target_id: str
    intervention_or_exposure: str = ""
    comparator: str = ""
    outcome: str = ""
    outcome_code: str = ""
    estimand: str = ""
    adjustment_model: str = ""
    population_or_subgroup: str = ""
    follow_up: str = ""
    data_family: str = ""
    raw_data: dict[str, Any] = Field(default_factory=dict)
    derivation_status: str = ""
    quantitative_usability: str = ""
    reported_effect_type: str | None = None
    reported_effect_value: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    source_location: dict[str, Any] = Field(default_factory=dict)
    original: dict[str, Any] = Field(default_factory=dict)


class EvidenceInventory(BaseModel):
    record_count: int
    article_count: int
    study_count: int
    target_counts: dict[str, int]
    data_family_counts: dict[str, int]
    graph_node_count: int = 0
    graph_edge_count: int = 0


class CandidateGroup(BaseModel):
    group_id: str
    study_id: str
    article_ids: list[str]
    evidence_ids: list[str]
    records: list[EvidenceRecord]
    exposure: str = ""
    outcome: str = ""
    comparator: str = ""
    triplet: tuple[str, str, str] = ("", "", "")
    selection_context: dict[str, str] = Field(default_factory=dict)
    retrieval_reasons: list[str] = Field(default_factory=list)


class CandidateSet(BaseModel):
    target_id: str
    groups: list[CandidateGroup]
    retrieval_trace: list[str] = Field(default_factory=list)


class SelectionDecision(BaseModel):
    study_id: str
    candidate_evidence_ids: list[str]
    selected_evidence_id: str | None = None
    status: Literal["selected", "excluded", "needs_review"]
    reason: str
    reviewer_mode: Literal["deterministic", "llm"]


class SelectionSet(BaseModel):
    decisions: list[SelectionDecision]
    selected_records: list[EvidenceRecord]


class PoolabilityDecision(BaseModel):
    study_id: str
    evidence_id: str
    status: Literal["poolable", "insufficient_for_analysis", "excluded"]
    reasons: list[str] = Field(default_factory=list)


class StudyEffect(BaseModel):
    study_id: str
    evidence_id: str
    effect_type: str
    value: float
    log_effect: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    source_data: dict[str, float]
    # The legacy field name log_effect is retained for compatibility. For
    # difference-scale effects it stores the untransformed analysis estimate.
    effect_scale: Literal["log_ratio", "difference"] = "log_ratio"


class PoolabilitySet(BaseModel):
    decisions: list[PoolabilityDecision]
    study_effects: list[StudyEffect]


class MainMetaStatistics(BaseModel):
    effect_type: str
    model: str
    k: int
    pooled_effect: float
    ci_lower: float
    ci_upper: float
    i_squared: float | None = None
    q: float | None = None
    q_p_value: float | None = None
    tau_squared: float | None = None
    member_evidence_ids: list[str]


class ToolTraceEntry(BaseModel):
    tool_name: str
    status: Literal["completed", "skipped", "blocked"]
    summary: str


class MainAnalysisResult(BaseModel):
    status: Literal["completed", "needs_extraction", "blocked"]
    profile: ResolvedAnalysisProfile
    inventory: EvidenceInventory
    candidate_set: CandidateSet
    selections: SelectionSet
    poolability: PoolabilitySet
    statistics: MainMetaStatistics | None = None
    evidence_gaps: list[str] = Field(default_factory=list)
    trace: list[ToolTraceEntry] = Field(default_factory=list)


AnalysisKind = Literal["primary", "secondary_outcome", "subgroup", "sensitivity"]


class DimensionLevel(BaseModel):
    value_id: str
    label: str
    terms: list[str] = Field(default_factory=list)
    analysis_overrides: dict[str, Any] = Field(default_factory=dict)


class CandidateAnalysisDimension(BaseModel):
    dimension_id: str
    label: str
    kind: Literal["subgroup", "sensitivity"]
    # Standard Evidence fields and protocol-defined extraction metadata share
    # one dimension interface.  Custom fields are resolved from raw_data.
    field: str
    levels: list[DimensionLevel]
    minimum_studies_per_level: int = 2
    exploratory: bool = False


class AnalysisDefinition(BaseModel):
    analysis_id: str
    pooled_result_id: str
    title: str
    kind: AnalysisKind
    target_id: str
    outcome_group: str
    # Optional protocol vocabulary used only for semantic/graph retrieval.
    # ``outcome_group`` remains the human-facing analysis label.
    outcome_terms: list[str] = Field(default_factory=list)
    eligible_outcome_codes: list[str] = Field(default_factory=list)
    eligible_data_families: list[str]
    accepted_effect_types: list[str]
    estimands: list[str] = Field(default_factory=list)
    exposure_terms: list[str] = Field(default_factory=list)
    comparison_terms: list[str] = Field(default_factory=list)
    population_policy: Literal["overall_only", "any", "dimension_level"] = "overall_only"
    population_terms: list[str] = Field(default_factory=list)
    timepoint_policy: Literal["longest_eligible", "all_eligible", "source_defined"] = "longest_eligible"
    model_policy: str = "prefer the most fully adjusted eligible model"
    pooling_model: Literal["fixed_effect", "random_effects", "unspecified"] = "random_effects"
    pooling_scale: str = "log_ratio_inverse_variance"
    minimum_studies: int = 2
    dependence_policy: Literal[
        "one_effect_per_study",
        "one_effect_per_study_outcome",
        "correlated_effects_not_implemented",
        "clustered_effects_rho_0_6",
    ] = "one_effect_per_study_outcome"
    parent_analysis_id: str | None = None
    dimension_id: str | None = None
    dimension_field: str | None = None
    dimension_value_id: str | None = None
    dimension_terms: list[str] = Field(default_factory=list)
    prespecified: bool = True
    selection_rules: list[str] = Field(default_factory=list)


class ResolvedAnalysisPlan(BaseModel):
    plan_id: str
    profile_version: str = "2.0"
    task_id: str
    protocol_id: str
    primary_analysis_id: str
    analyses: list[AnalysisDefinition]
    candidate_dimensions: list[CandidateAnalysisDimension] = Field(default_factory=list)
    derivation_trace: list[str] = Field(default_factory=list)


class EvidenceAssignmentDecision(BaseModel):
    analysis_id: str
    evidence_id: str
    study_id: str
    role: Literal["included", "excluded", "needs_review"]
    reason: str
    reviewer_mode: Literal["deterministic", "llm"]
    selected_within_context: bool = False


class GlobalEvidenceRoute(BaseModel):
    """One globally judged Evidence node and every analysis it finally feeds."""

    evidence_id: str
    study_id: str
    selected_analysis_ids: list[str] = Field(default_factory=list)
    status: Literal["routed", "excluded", "needs_review"]
    reason: str


class GlobalRoutingResult(BaseModel):
    reviewer_mode: Literal["deterministic", "llm"]
    routes: list[GlobalEvidenceRoute]
    summary: str


class AnalysisExecutionResult(BaseModel):
    analysis: AnalysisDefinition
    status: Literal["completed", "insufficient_data", "blocked"]
    candidate_set: CandidateSet
    assignments: list[EvidenceAssignmentDecision]
    selected_records: list[EvidenceRecord]
    poolability: PoolabilitySet
    statistics: MainMetaStatistics | None = None
    warnings: list[str] = Field(default_factory=list)


class BaseAnalysisAgentResult(BaseModel):
    status: Literal["completed", "partial", "blocked"]
    plan: ResolvedAnalysisPlan
    inventory: EvidenceInventory
    global_routing: GlobalRoutingResult
    analyses: list[AnalysisExecutionResult]
    evidence_assignment_index: dict[str, list[str]] = Field(default_factory=dict)
    trace: list[ToolTraceEntry] = Field(default_factory=list)
