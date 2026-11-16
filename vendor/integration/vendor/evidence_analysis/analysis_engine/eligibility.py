"""Graph protocol eligibility checks, including portable measurement metadata."""

from __future__ import annotations

import re

from analysis_core.models import AnalysisDefinition, EvidenceRecord


def is_categorical_high_low(record: EvidenceRecord) -> bool:
    """Require an explicit highest-versus-lowest category contrast in source fields."""
    comparator = record.comparator.casefold()
    contrast = " ".join((
        record.intervention_or_exposure,
        record.comparator,
        record.estimand,
        str(record.raw_data.get("exposure_unit") or ""),
    )).casefold()
    noncategorical = bool(re.search(
        r"\b(?:per\s+(?:1|one|each)|continuous|no categorical reference|"
        r"lower four quintiles|q1\s*[–-]\s*q[2345])\b", comparator
    ))
    lowest = bool(re.search(r"\b(?:lowest|least|q1|first quartile|first quintile|low variability)\b", comparator))
    highest = bool(re.search(r"\b(?:highest|upper quartile|top quartile|top quintile|q4|q5|fifth quintile|high variability)\b", contrast))
    return not noncategorical and lowest and highest


def _normalise(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _matches_any(value: object, terms: list[str]) -> bool:
    if not terms:
        return True
    text = _normalise(value)
    padded_text = f" {text} "
    text_tokens = set(text.split())
    return any(
        term_text
        and (
            text == term_text
            or f" {term_text} " in padded_text
            or set(term_text.split()).issubset(text_tokens)
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
        "overall", "all participants", "full cohort", "entire cohort", "total cohort",
    ]
    value = _normalise(record.population_or_subgroup)
    return not value or _matches_any(value, terms)


def _metadata_constraints(analysis: AnalysisDefinition) -> list[tuple[str, list[str]]]:
    """Parse adapter-emitted constraints without expanding the frozen structured model."""
    prefix = "Metadata constraint: "
    marker = " must match one of: "
    constraints: list[tuple[str, list[str]]] = []
    for rule in analysis.selection_rules:
        if not rule.startswith(prefix) or marker not in rule:
            continue
        field, values = rule[len(prefix):].split(marker, 1)
        terms = [item.strip().rstrip(".") for item in values.split(",") if item.strip()]
        if field.strip() and terms:
            constraints.append((field.strip(), terms))
    return constraints


def eligibility_reasons(record: EvidenceRecord, analysis: AnalysisDefinition) -> list[str]:
    reasons: list[str] = []
    if record.target_id != analysis.target_id:
        reasons.append("target does not match")
    if analysis.eligible_outcome_codes and record.outcome_code not in analysis.eligible_outcome_codes:
        reasons.append("outcome is outside the analysis scope")
    if record.data_family not in analysis.eligible_data_families:
        reasons.append("data family is not accepted")
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
    metadata = record.raw_data.get("_analysis_metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    for field, terms in _metadata_constraints(analysis):
        value = metadata.get(field, record.raw_data.get(field, ""))
        if not _matches_any(value, terms):
            reasons.append(f"metadata field '{field}' does not match the analysis constraint")
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
