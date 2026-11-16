from __future__ import annotations

import math
import re

from .models import (
    EvidenceRecord,
    MainMetaStatistics,
    PoolabilityDecision,
    PoolabilitySet,
    ResolvedAnalysisProfile,
    StudyEffect,
)


def _positive_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _follow_up_months(value: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:months?|mos?\.?)\b", value.lower())
    return float(match.group(1)) if match else None


def _is_rate_consistent(observed_rate: float, reported_rate: object) -> bool:
    """Allow ordinary source rounding, but reject a contradictory arm-level count."""
    try:
        reported = float(reported_rate)
    except (TypeError, ValueError):
        return True
    if reported <= 0:
        return True
    return abs(observed_rate - reported) / reported <= 0.25


def qualify_rate_ratio_records(
    profile: ResolvedAnalysisProfile, records: list[EvidenceRecord]
) -> PoolabilitySet:
    """Validate source fields and derive per-study IRRs from exact arm data."""
    decisions: list[PoolabilityDecision] = []
    effects: list[StudyEffect] = []
    for record in records:
        reasons: list[str] = []
        if record.quantitative_usability != "poolable":
            reasons.append("quantitative_usability is not 'poolable'.")
        if record.data_family != profile.data_family:
            reasons.append(f"data_family '{record.data_family}' does not match '{profile.data_family}'.")
        raw = record.raw_data
        values = {field: _positive_number(raw.get(field)) for field in profile.required_raw_fields}
        missing = [field for field, value in values.items() if value is None]
        if missing:
            reasons.append("Missing or non-positive raw fields: " + ", ".join(missing) + ".")
        follow_up = _follow_up_months(record.follow_up)
        if follow_up is None:
            reasons.append("follow_up does not contain a positive duration in months.")
        if reasons:
            decisions.append(
                PoolabilityDecision(
                    study_id=record.study_id,
                    evidence_id=record.evidence_id,
                    status="insufficient_for_analysis",
                    reasons=reasons,
                )
            )
            continue

        intervention_events = values["intervention_episodes"]
        control_events = values["control_episodes"]
        intervention_total = values["intervention_total"]
        control_total = values["control_total"]
        intervention_rate = intervention_events / (intervention_total * follow_up)
        control_rate = control_events / (control_total * follow_up)
        reported_rates = raw.get("reported_rates_per_person_month")
        if isinstance(reported_rates, dict):
            if not _is_rate_consistent(intervention_rate, reported_rates.get("intervention")):
                reasons.append(
                    "intervention episode count is inconsistent with the record's reported intervention rate."
                )
            if not _is_rate_consistent(control_rate, reported_rates.get("control")):
                reasons.append(
                    "control episode count is inconsistent with the record's reported control rate."
                )
        if reasons:
            decisions.append(
                PoolabilityDecision(
                    study_id=record.study_id,
                    evidence_id=record.evidence_id,
                    status="insufficient_for_analysis",
                    reasons=reasons,
                )
            )
            continue
        irr = intervention_rate / control_rate
        log_irr = math.log(irr)
        # Poisson-count approximation; baseline refuses zero counts rather than silently applying a correction.
        standard_error = math.sqrt((1 / intervention_events) + (1 / control_events))
        lower = math.exp(log_irr - 1.96 * standard_error)
        upper = math.exp(log_irr + 1.96 * standard_error)
        effects.append(
            StudyEffect(
                study_id=record.study_id,
                evidence_id=record.evidence_id,
                effect_type="IRR",
                value=irr,
                log_effect=log_irr,
                standard_error=standard_error,
                ci_lower=lower,
                ci_upper=upper,
                source_data={
                    "intervention_episodes": intervention_events,
                    "control_episodes": control_events,
                    "intervention_total": intervention_total,
                    "control_total": control_total,
                    "follow_up_months": follow_up,
                },
            )
        )
        decisions.append(
            PoolabilityDecision(
                study_id=record.study_id,
                evidence_id=record.evidence_id,
                status="poolable",
                reasons=["All baseline IRR source fields are present and positive."],
            )
        )
    return PoolabilitySet(decisions=decisions, study_effects=effects)


def qualify_adjusted_ratio_records(
    profile: ResolvedAnalysisProfile, records: list[EvidenceRecord]
) -> PoolabilitySet:
    """Validate directly reported adjusted HR/RR/OR records on the log-ratio scale."""
    decisions: list[PoolabilityDecision] = []
    effects: list[StudyEffect] = []
    accepted_types = set(profile.accepted_effect_types)
    for record in records:
        reasons: list[str] = []
        if record.quantitative_usability != "poolable":
            reasons.append("quantitative_usability is not 'poolable'.")
        if record.data_family != profile.data_family:
            reasons.append(f"data_family '{record.data_family}' does not match '{profile.data_family}'.")
        effect_type = str(record.reported_effect_type or "")
        if not effect_type:
            reasons.append("reported effect type is missing.")
        elif accepted_types and effect_type not in accepted_types:
            reasons.append(f"effect type '{effect_type}' is not accepted by the analysis profile.")
        value = _positive_number(record.reported_effect_value)
        lower = _positive_number(record.ci_lower)
        upper = _positive_number(record.ci_upper)
        if value is None or lower is None or upper is None:
            reasons.append("reported effect value and both positive confidence limits are required.")
        elif not lower <= value <= upper:
            reasons.append("reported effect must lie within its confidence interval.")
        if reasons:
            decisions.append(
                PoolabilityDecision(
                    study_id=record.study_id,
                    evidence_id=record.evidence_id,
                    status="insufficient_for_analysis",
                    reasons=reasons,
                )
            )
            continue
        log_effect = math.log(value)
        standard_error = (math.log(upper) - math.log(lower)) / (2 * 1.96)
        if not math.isfinite(standard_error) or standard_error <= 0:
            decisions.append(
                PoolabilityDecision(
                    study_id=record.study_id,
                    evidence_id=record.evidence_id,
                    status="insufficient_for_analysis",
                    reasons=["confidence interval does not define a positive finite standard error."],
                )
            )
            continue
        effects.append(
            StudyEffect(
                study_id=record.study_id,
                evidence_id=record.evidence_id,
                effect_type=effect_type,
                value=value,
                log_effect=log_effect,
                standard_error=standard_error,
                ci_lower=lower,
                ci_upper=upper,
                source_data={
                    "reported_effect": value,
                    "ci_lower": lower,
                    "ci_upper": upper,
                },
            )
        )
        decisions.append(
            PoolabilityDecision(
                study_id=record.study_id,
                evidence_id=record.evidence_id,
                status="poolable",
                reasons=["Reported adjusted ratio effect and confidence interval are complete."],
            )
        )
    return PoolabilitySet(decisions=decisions, study_effects=effects)


def calculate_fixed_effect_irr(effects: list[StudyEffect]) -> MainMetaStatistics:
    if len(effects) < 2:
        raise ValueError("Fixed-effect meta-analysis requires at least two study effects.")
    weights = [1 / (effect.standard_error**2) for effect in effects]
    pooled_log = sum(weight * effect.log_effect for weight, effect in zip(weights, effects)) / sum(weights)
    pooled_se = math.sqrt(1 / sum(weights))
    q = sum(weight * ((effect.log_effect - pooled_log) ** 2) for weight, effect in zip(weights, effects))
    df = len(effects) - 1
    i_squared = max(0.0, ((q - df) / q) * 100) if q > 0 else 0.0
    return MainMetaStatistics(
        effect_type="IRR",
        model="fixed_effect",
        k=len(effects),
        pooled_effect=math.exp(pooled_log),
        ci_lower=math.exp(pooled_log - 1.96 * pooled_se),
        ci_upper=math.exp(pooled_log + 1.96 * pooled_se),
        i_squared=i_squared,
        q=q,
        q_p_value=None,
        member_evidence_ids=[effect.evidence_id for effect in effects],
    )


def calculate_ratio_meta(profile: ResolvedAnalysisProfile, effects: list[StudyEffect]) -> MainMetaStatistics:
    """Pool source-reported ratio effects with fixed or DerSimonian-Laird random effects."""
    if len(effects) < 2:
        raise ValueError("Meta-analysis requires at least two independent study effects.")
    fixed_weights = [1 / (effect.standard_error**2) for effect in effects]
    fixed_log = sum(weight * effect.log_effect for weight, effect in zip(fixed_weights, effects)) / sum(fixed_weights)
    q = sum(weight * ((effect.log_effect - fixed_log) ** 2) for weight, effect in zip(fixed_weights, effects))
    df = len(effects) - 1
    i_squared = max(0.0, ((q - df) / q) * 100) if q > 0 else 0.0
    tau_squared = 0.0
    weights = fixed_weights
    if profile.pooling_model == "random_effects":
        denominator = sum(fixed_weights) - (sum(weight**2 for weight in fixed_weights) / sum(fixed_weights))
        tau_squared = max(0.0, (q - df) / denominator) if denominator > 0 else 0.0
        weights = [1 / ((effect.standard_error**2) + tau_squared) for effect in effects]
    pooled_log = sum(weight * effect.log_effect for weight, effect in zip(weights, effects)) / sum(weights)
    pooled_se = math.sqrt(1 / sum(weights))
    return MainMetaStatistics(
        effect_type=profile.effect_type,
        model=profile.pooling_model,
        k=len(effects),
        pooled_effect=math.exp(pooled_log),
        ci_lower=math.exp(pooled_log - 1.96 * pooled_se),
        ci_upper=math.exp(pooled_log + 1.96 * pooled_se),
        i_squared=i_squared,
        q=q,
        q_p_value=None,
        tau_squared=tau_squared if profile.pooling_model == "random_effects" else None,
        member_evidence_ids=[effect.evidence_id for effect in effects],
    )
