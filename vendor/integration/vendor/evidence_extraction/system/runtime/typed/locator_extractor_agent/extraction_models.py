from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


QuantitativeUsability = Literal[
    "poolable",
    "quantitative_not_pooled",
    "source_only",
    "needs_extraction",
    "calculator_eligible",
    "extraction_only",
    "requires_analysis_profile",
]


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
    quantitative_usability: QuantitativeUsability = "poolable"
    source_id: str
    source_quote: str


class BinaryArmCounts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intervention_events: int = Field(ge=0)
    intervention_total: int = Field(gt=0)
    control_events: int = Field(ge=0)
    control_total: int = Field(gt=0)
    outcome_unit: str


class ExtractedBinaryArmCounts(BaseModel):
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
    quantitative_usability: QuantitativeUsability = "poolable"
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
    effect_orientation: Literal[
        "intervention_minus_control", "control_minus_intervention"
    ]


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
    quantitative_usability: QuantitativeUsability = "poolable"
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
    effect_orientation: Literal[
        "intervention_minus_control", "control_minus_intervention"
    ]


class ExtractedContinuousChangeScoreSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: str
    target_id: Optional[str] = None
    data_family: Literal["continuous_change_score_summary"] = (
        "continuous_change_score_summary"
    )
    exposure: str
    comparator: str
    population_or_subgroup: str
    estimand: str
    timepoint: Optional[str] = None
    model_name: str
    adjustment: str = ""
    p_value: Optional[str] = None
    sample_size: Optional[str] = None
    quantitative_usability: QuantitativeUsability = "poolable"
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
    data_family: Literal["event_counts_with_person_time"] = (
        "event_counts_with_person_time"
    )
    exposure: str
    comparator: str
    population_or_subgroup: str
    estimand: str
    timepoint: Optional[str] = None
    model_name: str
    adjustment: str = ""
    p_value: Optional[str] = None
    sample_size: Optional[str] = None
    quantitative_usability: QuantitativeUsability = "poolable"
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
    quantitative_usability: Literal["source_only", "needs_extraction"] = (
        "source_only"
    )
    source_id: str
    source_quote: str
    raw_data: PercentageOnly


class GenericTextEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_description: str


class ExtractedGenericTextEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: str
    target_id: Optional[str] = None
    data_family: Literal["generic_text_evidence"] = "generic_text_evidence"
    exposure: str
    comparator: str
    population_or_subgroup: str
    estimand: str
    timepoint: Optional[str] = None
    model_name: str = "Source description"
    adjustment: str = ""
    p_value: Optional[str] = None
    sample_size: Optional[str] = None
    quantitative_usability: Literal["source_only", "needs_extraction"] = (
        "source_only"
    )
    source_id: str
    source_quote: str
    raw_data: GenericTextEvidence


class GenotypeCaseControlCounts(BaseModel):
    """Source-transcribed genotype counts; no genetic contrast is calculated here."""

    model_config = ConfigDict(extra="forbid")

    rsid: str
    genotype_labels: list[str] = Field(min_length=2)
    case_counts: dict[str, int]
    control_counts: dict[str, int]


class ExtractedGenotypeCaseControlCounts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: str
    target_id: Optional[str] = None
    data_family: Literal["complete_genotype_case_control_counts"] = (
        "complete_genotype_case_control_counts"
    )
    exposure: str
    comparator: str
    population_or_subgroup: str
    estimand: str
    timepoint: Optional[str] = None
    model_name: str = "Source-reported genotype distribution"
    adjustment: str = ""
    p_value: Optional[str] = None
    sample_size: Optional[str] = None
    quantitative_usability: Literal["calculator_eligible", "extraction_only"] = (
        "calculator_eligible"
    )
    source_id: str
    source_quote: str
    raw_data: GenotypeCaseControlCounts


class AlleleFrequencySummary(BaseModel):
    """Allele-level source summary kept separate from genotype counts."""

    model_config = ConfigDict(extra="forbid")

    rsid: str
    case_total_n: int = Field(gt=0)
    control_total_n: int = Field(gt=0)
    case_allele_percent: dict[str, float]
    control_allele_percent: dict[str, float]
    genotype_counts_status: Literal["not_reported"] = "not_reported"


class ExtractedAlleleFrequencySummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: str
    target_id: Optional[str] = None
    data_family: Literal["allele_frequency_summary_only"] = (
        "allele_frequency_summary_only"
    )
    exposure: str
    comparator: str
    population_or_subgroup: str
    estimand: str
    timepoint: Optional[str] = None
    model_name: str = "Source-reported allele frequency"
    adjustment: str = ""
    p_value: Optional[str] = None
    sample_size: Optional[str] = None
    quantitative_usability: Literal["extraction_only"] = "extraction_only"
    source_id: str
    source_quote: str
    raw_data: AlleleFrequencySummary


class NamedContinuousArm(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    n: int = Field(gt=1)
    mean: float
    sd: float = Field(ge=0)


class IndependentContinuousArms(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome_axis: str
    measurement_semantics: str
    unit: str
    intervention_arm: NamedContinuousArm
    comparator_arm: NamedContinuousArm


class ExtractedIndependentContinuousArms(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: str
    target_id: Optional[str] = None
    data_family: Literal["independent_continuous_arms"] = (
        "independent_continuous_arms"
    )
    exposure: str
    comparator: str
    population_or_subgroup: str
    subject_type: Optional[str] = None
    arthroplasty_type: Optional[str] = None
    estimand: str
    timepoint: Optional[str] = None
    model_name: str = "Source-reported independent-arm summary"
    adjustment: str = ""
    p_value: Optional[str] = None
    sample_size: Optional[str] = None
    quantitative_usability: QuantitativeUsability = "calculator_eligible"
    source_id: str
    source_quote: str
    raw_data: IndependentContinuousArms


class ContinuousStatistic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mean: float
    sd: float = Field(gt=0)
    min: Optional[float] = None
    max: Optional[float] = None


class SingleArmContinuousSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome_axis: str
    measurement_semantics: str
    unit: str
    n: int = Field(gt=1)
    reported_summary: dict[str, ContinuousStatistic]
    planned_target: Optional[dict[str, float]] = None


class ExtractedSingleArmContinuousSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: str
    target_id: Optional[str] = None
    data_family: Literal["single_arm_continuous_summary"] = (
        "single_arm_continuous_summary"
    )
    exposure: str
    comparator: str
    population_or_subgroup: str
    subject_type: Optional[str] = None
    arthroplasty_type: Optional[str] = None
    estimand: str
    timepoint: Optional[str] = None
    model_name: str = "Source-reported single-arm summary"
    adjustment: str = ""
    p_value: Optional[str] = None
    sample_size: Optional[str] = None
    quantitative_usability: QuantitativeUsability = "requires_analysis_profile"
    source_id: str
    source_quote: str
    raw_data: SingleArmContinuousSummary


class DiagnosticFourCellCounts(BaseModel):
    """Source-transcribed diagnostic-test counts; no accuracy metric is calculated."""

    model_config = ConfigDict(extra="forbid")

    true_positive: int = Field(ge=0)
    false_positive: int = Field(ge=0)
    false_negative: int = Field(ge=0)
    true_negative: int = Field(ge=0)
    test_set_label: str


class ExtractedDiagnosticFourCellCounts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: str
    target_id: Optional[str] = None
    data_family: Literal["diagnostic_four_cell_counts"] = "diagnostic_four_cell_counts"
    exposure: str
    comparator: str
    population_or_subgroup: str
    estimand: str
    timepoint: Optional[str] = None
    model_name: str = "Source-reported diagnostic test set"
    adjustment: str = ""
    p_value: Optional[str] = None
    sample_size: Optional[str] = None
    quantitative_usability: Literal["calculator_eligible", "extraction_only"] = (
        "calculator_eligible"
    )
    source_id: str
    source_quote: str
    raw_data: DiagnosticFourCellCounts


class SingleArmEventCounts(BaseModel):
    """Source-transcribed event count and denominator for a single study arm."""

    model_config = ConfigDict(extra="forbid")

    events: int = Field(ge=0)
    total: int = Field(gt=0)
    outcome_unit: str

    @model_validator(mode="after")
    def events_must_not_exceed_total(self) -> "SingleArmEventCounts":
        if self.events > self.total:
            raise ValueError("events must not exceed total")
        return self


class ExtractedSingleArmEventCounts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: str
    target_id: Optional[str] = None
    data_family: Literal["single_arm_event_counts"] = "single_arm_event_counts"
    exposure: str
    comparator: str
    population_or_subgroup: str
    estimand: str
    timepoint: Optional[str] = None
    model_name: str = "Source-reported single-arm event count"
    adjustment: str = ""
    p_value: Optional[str] = None
    sample_size: Optional[str] = None
    quantitative_usability: Literal["calculator_eligible", "extraction_only"] = (
        "calculator_eligible"
    )
    source_id: str
    source_quote: str
    raw_data: SingleArmEventCounts


ExtractedRow = Union[
    ExtractedEffect,
    ExtractedBinaryArmCounts,
    ExtractedContinuousArmSummary,
    ExtractedContinuousChangeScoreSummary,
    ExtractedEventCountsWithPersonTime,
    ExtractedPercentageOnly,
    ExtractedGenericTextEvidence,
    ExtractedGenotypeCaseControlCounts,
    ExtractedAlleleFrequencySummary,
    ExtractedIndependentContinuousArms,
    ExtractedSingleArmContinuousSummary,
    ExtractedDiagnosticFourCellCounts,
    ExtractedSingleArmEventCounts,
]


class EffectExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[Union[ExtractedEffect, ExtractedGenericTextEvidence]]
    unresolved_items: list[str] = Field(default_factory=list)


class DataFamilyExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[ExtractedRow]
    unresolved_items: list[str] = Field(default_factory=list)
