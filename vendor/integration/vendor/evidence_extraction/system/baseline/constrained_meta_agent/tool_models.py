from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class ExtractionPlanTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    candidate_source_ids: list[str]
    target_outcomes: list[str]
    target_ids: list[str] = Field(default_factory=list)
    data_families: list[str] = Field(default_factory=list)
    exposure: str
    comparator: str
    estimand: str
    effect_types: list[str]
    model_selection_rule: str
    rationale: str


class ExtractionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tasks: list[ExtractionPlanTask]
    coverage_notes: str


class ExtractedEffect(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: str
    target_id: Optional[str] = None
    data_family: Optional[str] = None
    exposure: str
    exposure_unit: Optional[str] = None
    comparator: str
    population_or_subgroup: str
    estimand: str
    timepoint: Optional[str] = None
    effect_type: str
    effect_value: float
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    p_value: Optional[str] = None
    model_name: str
    adjustment: str
    sample_size: Optional[str] = None
    quantitative_usability: Literal["poolable", "source_only", "needs_extraction"] = "poolable"
    source_id: str
    source_quote: str


class BinaryArmCounts(BaseModel):
    """Source-reported two-arm binary outcome data."""

    model_config = ConfigDict(extra="forbid")

    intervention_events: int = Field(ge=0)
    intervention_total: int = Field(gt=0)
    control_events: int = Field(ge=0)
    control_total: int = Field(gt=0)
    outcome_unit: str


class ExtractedBinaryArmCounts(BaseModel):
    """An extraction row for a binary_arm_counts target.

    The LLM supplies only source data; effect conversion is deterministic.
    """

    model_config = ConfigDict(extra="forbid")

    outcome: str
    target_id: Optional[str] = None
    data_family: Literal["binary_arm_counts"] = "binary_arm_counts"
    exposure: str
    exposure_unit: Optional[str] = None
    comparator: str
    population_or_subgroup: str
    estimand: str
    timepoint: Optional[str] = None
    model_name: str
    adjustment: str = ""
    p_value: Optional[str] = None
    sample_size: Optional[str] = None
    quantitative_usability: Literal["poolable", "source_only", "needs_extraction"] = "poolable"
    source_id: str
    source_quote: str
    raw_data: BinaryArmCounts


class ContinuousArmSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intervention_mean: float
    intervention_sd: float = Field(gt=0)
    intervention_total: int = Field(gt=1)
    control_mean: float
    control_sd: float = Field(gt=0)
    control_total: int = Field(gt=1)
    outcome_unit: str
    # The extractor states orientation; normalisation performs the arithmetic.
    effect_orientation: Literal["intervention_minus_control", "control_minus_intervention"]


class ExtractedContinuousArmSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: str
    target_id: Optional[str] = None
    data_family: Literal["continuous_arm_summary"] = "continuous_arm_summary"
    exposure: str
    comparator: str
    population_or_subgroup: str
    estimand: str
    timepoint: Optional[str] = None
    model_name: str
    adjustment: str = ""
    p_value: Optional[str] = None
    sample_size: Optional[str] = None
    quantitative_usability: Literal["poolable", "source_only", "needs_extraction"] = "poolable"
    source_id: str
    source_quote: str
    raw_data: ContinuousArmSummary


class ContinuousChangeScoreSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intervention_mean_change: float
    intervention_sd_change: float = Field(gt=0)
    intervention_total: int = Field(gt=1)
    control_mean_change: float
    control_sd_change: float = Field(gt=0)
    control_total: int = Field(gt=1)
    outcome_unit: str
    effect_orientation: Literal["intervention_minus_control", "control_minus_intervention"]


class ExtractedContinuousChangeScoreSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: str
    target_id: Optional[str] = None
    data_family: Literal["continuous_change_score_summary"] = "continuous_change_score_summary"
    exposure: str
    comparator: str
    population_or_subgroup: str
    estimand: str
    timepoint: Optional[str] = None
    model_name: str
    adjustment: str = ""
    p_value: Optional[str] = None
    sample_size: Optional[str] = None
    quantitative_usability: Literal["poolable", "source_only", "needs_extraction"] = "poolable"
    source_id: str
    source_quote: str
    raw_data: ContinuousChangeScoreSummary


class EventCountsWithPersonTime(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intervention_episodes: int = Field(ge=0)
    intervention_total: int = Field(gt=0)
    control_episodes: int = Field(ge=0)
    control_total: int = Field(gt=0)
    follow_up_months: float = Field(gt=0)
    reported_rates_per_person_month: Optional[dict[str, float]] = None


class ExtractedEventCountsWithPersonTime(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: str
    target_id: Optional[str] = None
    data_family: Literal["event_counts_with_person_time"] = "event_counts_with_person_time"
    exposure: str
    comparator: str
    population_or_subgroup: str
    estimand: str
    timepoint: Optional[str] = None
    model_name: str
    adjustment: str = ""
    p_value: Optional[str] = None
    sample_size: Optional[str] = None
    quantitative_usability: Literal["poolable", "source_only", "needs_extraction"] = "poolable"
    source_id: str
    source_quote: str
    raw_data: EventCountsWithPersonTime


class PercentageOnly(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intervention_percent: Optional[float] = Field(default=None, ge=0, le=100)
    control_percent: Optional[float] = Field(default=None, ge=0, le=100)
    intervention_total: Optional[int] = Field(default=None, gt=0)
    control_total: Optional[int] = Field(default=None, gt=0)
    outcome_unit: str


class ExtractedPercentageOnly(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: str
    target_id: Optional[str] = None
    data_family: Literal["percentage_only"] = "percentage_only"
    exposure: str
    comparator: str
    population_or_subgroup: str
    estimand: str
    timepoint: Optional[str] = None
    model_name: str
    adjustment: str = ""
    p_value: Optional[str] = None
    sample_size: Optional[str] = None
    quantitative_usability: Literal["source_only", "needs_extraction"] = "source_only"
    source_id: str
    source_quote: str
    raw_data: PercentageOnly


ExtractedRow = Union[
    ExtractedEffect,
    ExtractedBinaryArmCounts,
    ExtractedContinuousArmSummary,
    ExtractedContinuousChangeScoreSummary,
    ExtractedEventCountsWithPersonTime,
    ExtractedPercentageOnly,
]


class EffectExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # This is intentionally the original reported-effect contract. Existing
    # ratio/time-to-event cases continue to send the exact same schema to LLMs.
    rows: list[ExtractedEffect]
    unresolved_items: list[str] = Field(default_factory=list)


class DataFamilyExtraction(BaseModel):
    """Response contract used only when a task opts into a non-legacy family."""

    model_config = ConfigDict(extra="forbid")

    rows: list[ExtractedRow]
    unresolved_items: list[str] = Field(default_factory=list)
