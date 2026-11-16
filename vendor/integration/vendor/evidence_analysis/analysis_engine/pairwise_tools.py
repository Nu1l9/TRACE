"""Typed pairwise meta-analysis tools for frozen Evidence records.

Each adapter owns only quantitative validation and effect derivation. Evidence
selection stays in the GraphRAG/Reviewer path, while pooled statistics are
delegated to the installed R ``metafor`` package.
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from analysis_core.models import (
    MainMetaStatistics,
    PoolabilityDecision,
    PoolabilitySet,
    ResolvedAnalysisProfile,
    StudyEffect,
)
REPORTED_RATIO_FAMILIES = frozenset({
    "adjusted_ratio_effect",
    "reported_ratio_effect",
    "time_to_event",
})
BINARY_TWO_GROUP_FAMILIES = frozenset({
    "binary_arm_counts",
    "binary_arm_outcome",
    "binary_arm_event_counts",
    "binary_exposure_outcome_counts",
    "two_group_event_counts",
})


def _finite_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _positive_number(value: object) -> float | None:
    number = _finite_number(value)
    return number if number is not None and number > 0 else None


def _numeric_source_data(raw: dict[str, object]) -> dict[str, float]:
    return {
        str(key): number
        for key, value in raw.items()
        if (number := _finite_number(value)) is not None
    }


def _follow_up_months(value: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:months?|mos?\.?)\b", value.lower())
    return float(match.group(1)) if match else None


def _is_rate_consistent(observed_rate: float, reported_rate: object) -> bool:
    """Allow ordinary source rounding, but reject contradictory arm-level counts."""
    reported = _positive_number(reported_rate)
    return reported is None or abs(observed_rate - reported) / reported <= 0.25


def _binary_effect_value(
    intervention_events: float,
    intervention_total: float,
    control_events: float,
    control_total: float,
    effect_type: str,
) -> float:
    ai = intervention_events
    bi = intervention_total - intervention_events
    ci = control_events
    di = control_total - control_events
    if any(value == 0 for value in (ai, bi, ci, di)):
        ai, bi, ci, di = (value + 0.5 for value in (ai, bi, ci, di))
    if effect_type == "OR":
        return (ai * di) / (bi * ci)
    return (ai / (ai + bi)) / (ci / (ci + di))


def _nested_binary_arms(raw: dict[str, object]) -> tuple[float, float, float, float] | None:
    """Read public portable two-arm records without renaming their data family."""
    for intervention_key, control_key in (
        ("intervention_arm", "comparator_arm"),
        ("exposed_arm", "unexposed_arm"),
        ("pandemic_group", "reference_group"),
    ):
        intervention = raw.get(intervention_key)
        control = raw.get(control_key)
        if not isinstance(intervention, dict) or not isinstance(control, dict):
            continue
        values = (
            _finite_number(intervention.get("events")),
            _positive_number(intervention.get("total") or intervention.get("n")),
            _finite_number(control.get("events")),
            _positive_number(control.get("total") or control.get("n")),
        )
        if all(value is not None for value in values) and values[0] <= values[1] and values[2] <= values[3]:
            return values  # type: ignore[return-value]
    return None


def _binary_arms(record, effect_type: str) -> tuple[float, float, float, float] | None:
    """Read canonical, nested, or prefixed two-arm source records.

    The orientation comes from explicit source arm roles whenever they exist.
    Legacy prefixed fields retain the historical reported-effect orientation
    fallback without consulting Gold.
    """
    raw = record.raw_data
    canonical = (
        _finite_number(raw.get("intervention_events")),
        _positive_number(raw.get("intervention_total")),
        _finite_number(raw.get("control_events")),
        _positive_number(raw.get("control_total")),
    )
    if all(value is not None for value in canonical):
        return canonical  # type: ignore[return-value]
    nested = _nested_binary_arms(raw)
    if nested is not None:
        return nested
    arms: list[tuple[str, float, float]] = []
    event_suffixes = ("_events", "_deaths", "_cases", "_outcomes")
    for key, events in raw.items():
        if not isinstance(key, str):
            continue
        suffix = next((value for value in event_suffixes if key.endswith(value)), None)
        if not suffix:
            continue
        prefix = key.removesuffix(suffix)
        total = raw.get(f"{prefix}_total")
        event_value = _finite_number(events)
        total_value = _positive_number(total)
        if event_value is not None and event_value >= 0 and total_value is not None and event_value <= total_value:
            arms.append((prefix, event_value, total_value))
    if len(arms) != 2:
        return None
    first, second = arms
    orientations = [
        (first[1], first[2], second[1], second[2]),
        (second[1], second[2], first[1], first[2]),
    ]
    reported = _positive_number(record.reported_effect_value)
    if reported is None:
        # Arm prefixes are source metadata.  For raw-count evidence without a
        # reported ratio, use the PICO intervention text to orient obvious
        # labelled arms (for example ``new_ppm`` versus ``no_new_ppm``).
        # Stable source order is the last tie-breaker, never a gold lookup.
        def orientation_score(values: tuple[float, float, float, float]) -> tuple[int, int]:
            intervention_events, intervention_total, _, _ = values
            prefix = first[0] if (intervention_events, intervention_total) == first[1:3] else second[0]
            prefix_tokens = set(prefix.casefold().replace("_", " ").split())
            exposure_tokens = set(
                str(record.intervention_or_exposure or "").casefold().replace("-", " ").split()
            )
            overlap = len(prefix_tokens & exposure_tokens)
            negated = int(any(token in {"no", "non", "without", "control"} for token in prefix_tokens))
            return overlap, -negated
        return max(orientations, key=orientation_score)
    return min(
        orientations,
        key=lambda values: abs(math.log(_binary_effect_value(*values, effect_type)) - math.log(reported)),
    )


class PairwiseEffectTool:
    """Dispatch frozen Evidence records to the appropriate effect adapter."""

    name = "derive_pairwise_effects"

    def qualify(self, profile: ResolvedAnalysisProfile, records: list) -> PoolabilitySet:
        if profile.data_family in REPORTED_RATIO_FAMILIES:
            return self._qualify_reported_ratio(profile, records)
        if profile.data_family in BINARY_TWO_GROUP_FAMILIES:
            return self._qualify_binary_counts(profile, records)
        if profile.data_family == "complete_genotype_case_control_counts":
            return self._qualify_genotype_or(profile, records)
        if profile.data_family == "diagnostic_four_cell_counts":
            return self._qualify_diagnostic_accuracy(profile, records)
        if profile.data_family == "single_arm_event_counts":
            return self._qualify_single_arm_proportion(profile, records)
        if profile.data_family == "independent_continuous_arms":
            return self._qualify_independent_mean_difference(profile, records)
        if profile.data_family == "reported_standardized_mean_difference":
            return self._qualify_reported_smd(profile, records)
        if profile.data_family == "event_counts_with_person_time":
            return self._qualify_rate_ratio(profile, records)
        raise NotImplementedError(
            f"No pairwise effect adapter is registered for data_family={profile.data_family!r}."
        )

    @staticmethod
    def _base_reasons(profile: ResolvedAnalysisProfile, record) -> list[str]:
        reasons: list[str] = []
        if record.data_family != profile.data_family:
            reasons.append(
                f"data_family '{record.data_family}' does not match '{profile.data_family}'."
            )
        return reasons

    def _qualify_reported_ratio(self, profile: ResolvedAnalysisProfile, records: list) -> PoolabilitySet:
        decisions: list[PoolabilityDecision] = []
        effects: list[StudyEffect] = []
        for record in records:
            reasons = self._base_reasons(profile, record)
            effect_type = str(record.reported_effect_type or "")
            if not effect_type:
                reasons.append("reported effect type is missing.")
            elif profile.accepted_effect_types and effect_type not in profile.accepted_effect_types:
                reasons.append(f"effect type '{effect_type}' is not accepted by the analysis profile.")
            value = _positive_number(record.reported_effect_value)
            lower = _positive_number(record.ci_lower)
            upper = _positive_number(record.ci_upper)
            if value is None or lower is None or upper is None:
                reasons.append("reported effect value and both positive confidence limits are required.")
            elif not lower <= value <= upper:
                reasons.append("reported effect must lie within its confidence interval.")
            if reasons:
                decisions.append(PoolabilityDecision(
                    study_id=record.study_id, evidence_id=record.evidence_id,
                    status="insufficient_for_analysis", reasons=reasons,
                ))
                continue
            standard_error = (math.log(upper) - math.log(lower)) / (2 * 1.96)
            if standard_error <= 0 or not math.isfinite(standard_error):
                decisions.append(PoolabilityDecision(
                    study_id=record.study_id, evidence_id=record.evidence_id,
                    status="insufficient_for_analysis",
                    reasons=["confidence interval does not define a positive finite standard error."],
                ))
                continue
            effects.append(StudyEffect(
                study_id=record.study_id,
                evidence_id=record.evidence_id,
                effect_type=effect_type,
                value=value,
                log_effect=math.log(value),
                standard_error=standard_error,
                ci_lower=lower,
                ci_upper=upper,
                source_data={
                    "reported_effect": value,
                    "ci_lower": lower,
                    "ci_upper": upper,
                },
                effect_scale="log_ratio",
            ))
            decisions.append(PoolabilityDecision(
                study_id=record.study_id, evidence_id=record.evidence_id,
                status="poolable",
                reasons=["Reported ratio effect and confidence interval are complete."],
            ))
        return PoolabilitySet(decisions=decisions, study_effects=effects)

    def _qualify_rate_ratio(self, profile: ResolvedAnalysisProfile, records: list) -> PoolabilitySet:
        """Derive IRRs directly from episode counts and source follow-up time."""
        decisions: list[PoolabilityDecision] = []
        effects: list[StudyEffect] = []
        for record in records:
            reasons = self._base_reasons(profile, record)
            raw = record.raw_data
            values = {
                field: _positive_number(raw.get(field))
                for field in profile.required_raw_fields
            }
            missing = [field for field, value in values.items() if value is None]
            if missing:
                reasons.append("Missing or non-positive raw fields: " + ", ".join(missing) + ".")
            follow_up = _follow_up_months(record.follow_up)
            if follow_up is None:
                reasons.append("follow_up does not contain a positive duration in months.")
            if reasons:
                decisions.append(PoolabilityDecision(
                    study_id=record.study_id,
                    evidence_id=record.evidence_id,
                    status="insufficient_for_analysis",
                    reasons=reasons,
                ))
                continue

            intervention_events = values["intervention_episodes"]
            control_events = values["control_episodes"]
            intervention_total = values["intervention_total"]
            control_total = values["control_total"]
            assert all(value is not None for value in (
                intervention_events, control_events, intervention_total, control_total, follow_up,
            ))
            intervention_rate = intervention_events / (intervention_total * follow_up)
            control_rate = control_events / (control_total * follow_up)
            reported_rates = raw.get("reported_rates_per_person_month")
            if isinstance(reported_rates, dict):
                if not _is_rate_consistent(intervention_rate, reported_rates.get("intervention")):
                    reasons.append("intervention episode count is inconsistent with the reported intervention rate.")
                if not _is_rate_consistent(control_rate, reported_rates.get("control")):
                    reasons.append("control episode count is inconsistent with the reported control rate.")
            if reasons:
                decisions.append(PoolabilityDecision(
                    study_id=record.study_id,
                    evidence_id=record.evidence_id,
                    status="insufficient_for_analysis",
                    reasons=reasons,
                ))
                continue

            irr = intervention_rate / control_rate
            log_irr = math.log(irr)
            standard_error = math.sqrt((1 / intervention_events) + (1 / control_events))
            effects.append(StudyEffect(
                study_id=record.study_id,
                evidence_id=record.evidence_id,
                effect_type="IRR",
                value=irr,
                log_effect=log_irr,
                standard_error=standard_error,
                ci_lower=math.exp(log_irr - 1.96 * standard_error),
                ci_upper=math.exp(log_irr + 1.96 * standard_error),
                source_data={
                    "intervention_episodes": intervention_events,
                    "control_episodes": control_events,
                    "intervention_total": intervention_total,
                    "control_total": control_total,
                    "follow_up_months": follow_up,
                },
                effect_scale="log_ratio",
            ))
            decisions.append(PoolabilityDecision(
                study_id=record.study_id,
                evidence_id=record.evidence_id,
                status="poolable",
                reasons=["Required episode-count and follow-up fields are complete."],
            ))
        return PoolabilitySet(decisions=decisions, study_effects=effects)

    def _qualify_binary_counts(self, profile: ResolvedAnalysisProfile, records: list) -> PoolabilitySet:
        """Derive RR, OR, or RD from a complete two-arm binary record."""
        decisions: list[PoolabilityDecision] = []
        effects: list[StudyEffect] = []
        for record in records:
            reasons = self._base_reasons(profile, record)
            raw = record.raw_data
            effect_type = str(record.reported_effect_type or "")
            if not effect_type and len(profile.accepted_effect_types) == 1:
                effect_type = profile.accepted_effect_types[0]
            if effect_type not in {"RR", "OR", "RD"}:
                reasons.append("Two-arm binary data currently supports RR, OR, or RD only.")
            elif profile.accepted_effect_types and effect_type not in profile.accepted_effect_types:
                reasons.append(f"effect type '{effect_type}' is not accepted by the analysis profile.")
            arms = _binary_arms(record, effect_type) if not reasons else None
            if arms is None and not reasons:
                reasons.append(
                    "Could not identify exactly two valid binary arms. Use canonical intervention/control fields "
                    "or two <arm>_events/<arm>_total pairs with a reported effect direction."
                )
            if reasons:
                decisions.append(PoolabilityDecision(
                    study_id=record.study_id, evidence_id=record.evidence_id,
                    status="insufficient_for_analysis", reasons=reasons,
                ))
                continue
            assert arms is not None
            intervention_events, intervention_total, control_events, control_total = arms
            ai = intervention_events
            bi = intervention_total - intervention_events
            ci = control_events
            di = control_total - control_events
            zero_cell_correction = effect_type in {"RR", "OR"} and any(
                value == 0 for value in (ai, bi, ci, di)
            )
            if zero_cell_correction:
                ai, bi, ci, di = (value + 0.5 for value in (ai, bi, ci, di))
            if effect_type == "OR":
                value = (ai * di) / (bi * ci)
                variance = (1 / ai) + (1 / bi) + (1 / ci) + (1 / di)
                effect_scale = "log_ratio"
                log_effect = math.log(value)
                ci_lower = math.exp(log_effect - 1.96 * math.sqrt(variance))
                ci_upper = math.exp(log_effect + 1.96 * math.sqrt(variance))
            elif effect_type == "RR":
                value = (ai / (ai + bi)) / (ci / (ci + di))
                variance = (1 / ai) - (1 / (ai + bi)) + (1 / ci) - (1 / (ci + di))
                effect_scale = "log_ratio"
                log_effect = math.log(value)
                ci_lower = math.exp(log_effect - 1.96 * math.sqrt(variance))
                ci_upper = math.exp(log_effect + 1.96 * math.sqrt(variance))
            else:
                intervention_risk = intervention_events / intervention_total
                control_risk = control_events / control_total
                value = intervention_risk - control_risk
                variance = (
                    intervention_risk * (1 - intervention_risk) / intervention_total
                    + control_risk * (1 - control_risk) / control_total
                )
                effect_scale = "difference"
                log_effect = value
                ci_lower = value - 1.96 * math.sqrt(variance)
                ci_upper = value + 1.96 * math.sqrt(variance)
            standard_error = math.sqrt(variance)
            effects.append(StudyEffect(
                study_id=record.study_id,
                evidence_id=record.evidence_id,
                effect_type=effect_type,
                value=value,
                log_effect=log_effect,
                standard_error=standard_error,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                source_data={
                    "ai": intervention_events,
                    "bi": intervention_total - intervention_events,
                    "ci": control_events,
                    "di": control_total - control_events,
                    "zero_cell_correction": float(zero_cell_correction),
                },
                effect_scale=effect_scale,
            ))
            decisions.append(PoolabilityDecision(
                study_id=record.study_id, evidence_id=record.evidence_id,
                status="poolable",
                reasons=[
                    "Binary 2x2 counts are complete."
                    + (" Applied prespecified 0.5 zero-cell correction." if zero_cell_correction else "")
                ],
            ))
        return PoolabilitySet(decisions=decisions, study_effects=effects)

    def _qualify_genotype_or(self, profile: ResolvedAnalysisProfile, records: list) -> PoolabilitySet:
        """Calculate a prespecified genotype contrast from complete case-control counts."""
        decisions: list[PoolabilityDecision] = []
        effects: list[StudyEffect] = []
        fields = (
            "case_heterozygote",
            "case_common_homozygote",
            "control_heterozygote",
            "control_common_homozygote",
        )
        for record in records:
            reasons = self._base_reasons(profile, record)
            if profile.accepted_effect_types and "OR" not in profile.accepted_effect_types:
                reasons.append("The declared genotype calculator produces an OR, which this profile does not accept.")
            values = {field: _finite_number(record.raw_data.get(field)) for field in fields}
            missing = [field for field, value in values.items() if value is None or value < 0]
            if missing:
                reasons.append("Missing or invalid genotype count(s): " + ", ".join(missing) + ".")
            elif any(value == 0 for value in values.values()):
                reasons.append("A genotype contrast cell is zero and the public profile declares no continuity correction.")
            if reasons:
                decisions.append(PoolabilityDecision(
                    study_id=record.study_id, evidence_id=record.evidence_id,
                    status="insufficient_for_analysis", reasons=reasons,
                ))
                continue
            assert all(value is not None for value in values.values())
            case_heterozygote = values["case_heterozygote"]
            case_common = values["case_common_homozygote"]
            control_heterozygote = values["control_heterozygote"]
            control_common = values["control_common_homozygote"]
            assert None not in (case_heterozygote, case_common, control_heterozygote, control_common)
            value = (case_heterozygote * control_common) / (case_common * control_heterozygote)
            standard_error = math.sqrt(
                (1 / case_heterozygote) + (1 / case_common)
                + (1 / control_heterozygote) + (1 / control_common)
            )
            log_effect = math.log(value)
            effects.append(StudyEffect(
                study_id=record.study_id,
                evidence_id=record.evidence_id,
                effect_type="OR",
                value=value,
                log_effect=log_effect,
                standard_error=standard_error,
                ci_lower=math.exp(log_effect - 1.96 * standard_error),
                ci_upper=math.exp(log_effect + 1.96 * standard_error),
                source_data={
                    "case_heterozygote": case_heterozygote,
                    "case_common_homozygote": case_common,
                    "control_heterozygote": control_heterozygote,
                    "control_common_homozygote": control_common,
                },
                effect_scale="log_ratio",
            ))
            decisions.append(PoolabilityDecision(
                study_id=record.study_id, evidence_id=record.evidence_id,
                status="poolable",
                reasons=["Complete heterozygote-versus-common-homozygote case-control counts."],
            ))
        return PoolabilitySet(decisions=decisions, study_effects=effects)

    def _qualify_diagnostic_accuracy(
        self, profile: ResolvedAnalysisProfile, records: list
    ) -> PoolabilitySet:
        """Pool sensitivity or specificity on the public profile's logit scale."""
        scale = profile.pooling_scale.split("__", 1)[0]
        if scale not in {"logit_sensitivity", "logit_specificity"}:
            raise ValueError(
                "diagnostic_four_cell_counts requires logit_sensitivity or logit_specificity pooling."
            )
        decisions: list[PoolabilityDecision] = []
        effects: list[StudyEffect] = []
        for record in records:
            reasons = self._base_reasons(profile, record)
            raw = record.raw_data
            values = {
                name: _finite_number(raw.get(name))
                for name in ("true_positive", "false_positive", "false_negative", "true_negative")
            }
            invalid = [name for name, value in values.items() if value is None or value < 0]
            if invalid:
                reasons.append("Missing or invalid 2x2 diagnostic cells: " + ", ".join(invalid) + ".")
            if reasons:
                decisions.append(PoolabilityDecision(
                    study_id=record.study_id, evidence_id=record.evidence_id,
                    status="insufficient_for_analysis", reasons=reasons,
                ))
                continue
            assert all(value is not None for value in values.values())
            if scale == "logit_sensitivity":
                success = values["true_positive"]
                failure = values["false_negative"]
                effect_type = "sensitivity"
            else:
                success = values["true_negative"]
                failure = values["false_positive"]
                effect_type = "specificity"
            assert success is not None and failure is not None
            if success + failure <= 0:
                decisions.append(PoolabilityDecision(
                    study_id=record.study_id, evidence_id=record.evidence_id,
                    status="insufficient_for_analysis",
                    reasons=["The selected diagnostic denominator is zero."],
                ))
                continue
            # The public profile explicitly permits a 0.5 boundary correction
            # during logit pooling when a source sensitivity/specificity is 0 or 1.
            adjusted_success = success + 0.5
            adjusted_failure = failure + 0.5
            value = success / (success + failure)
            logit_value = math.log(adjusted_success / adjusted_failure)
            standard_error = math.sqrt((1 / adjusted_success) + (1 / adjusted_failure))
            effects.append(StudyEffect(
                study_id=record.study_id,
                evidence_id=record.evidence_id,
                effect_type=effect_type,
                value=value,
                log_effect=logit_value,
                standard_error=standard_error,
                ci_lower=1 / (1 + math.exp(-(logit_value - 1.96 * standard_error))),
                ci_upper=1 / (1 + math.exp(-(logit_value + 1.96 * standard_error))),
                source_data={
                    "true_positive": values["true_positive"],
                    "false_positive": values["false_positive"],
                    "false_negative": values["false_negative"],
                    "true_negative": values["true_negative"],
                },
                effect_scale="difference",
            ))
            decisions.append(PoolabilityDecision(
                study_id=record.study_id, evidence_id=record.evidence_id,
                status="poolable",
                reasons=[f"Complete diagnostic 2x2 counts for logit {effect_type} pooling."],
            ))
        return PoolabilitySet(decisions=decisions, study_effects=effects)

    def _qualify_single_arm_proportion(
        self, profile: ResolvedAnalysisProfile, records: list
    ) -> PoolabilitySet:
        """Apply the public Freeman-Tukey transform to source event counts."""
        if not profile.pooling_scale.startswith("freeman_tukey_proportion"):
            raise ValueError(
                "single_arm_event_counts requires the declared Freeman-Tukey proportion calculator."
            )
        decisions: list[PoolabilityDecision] = []
        effects: list[StudyEffect] = []
        for record in records:
            reasons = self._base_reasons(profile, record)
            events = _finite_number(record.raw_data.get("events"))
            total = _positive_number(record.raw_data.get("total"))
            if events is None or events < 0 or total is None or events > total:
                reasons.append("Single-arm source record requires 0 <= events <= total and total > 0.")
            if reasons:
                decisions.append(PoolabilityDecision(
                    study_id=record.study_id, evidence_id=record.evidence_id,
                    status="insufficient_for_analysis", reasons=reasons,
                ))
                continue
            assert events is not None and total is not None
            transformed = math.asin(math.sqrt(events / (total + 1))) + math.asin(
                math.sqrt((events + 1) / (total + 1))
            )
            standard_error = math.sqrt(1 / (total + 0.5))
            lower = math.sin((transformed - 1.96 * standard_error) / 2) ** 2
            upper = math.sin((transformed + 1.96 * standard_error) / 2) ** 2
            effects.append(StudyEffect(
                study_id=record.study_id,
                evidence_id=record.evidence_id,
                effect_type="proportion",
                value=events / total,
                log_effect=transformed,
                standard_error=standard_error,
                ci_lower=max(0.0, lower),
                ci_upper=min(1.0, upper),
                source_data={"events": events, "total": total},
                effect_scale="difference",
            ))
            decisions.append(PoolabilityDecision(
                study_id=record.study_id, evidence_id=record.evidence_id,
                status="poolable",
                reasons=["Complete single-arm events and denominator for Freeman-Tukey pooling."],
            ))
        return PoolabilitySet(decisions=decisions, study_effects=effects)

    def _qualify_independent_mean_difference(
        self, profile: ResolvedAnalysisProfile, records: list
    ) -> PoolabilitySet:
        """Derive independent-arm mean differences from n, mean, and SD."""
        decisions: list[PoolabilityDecision] = []
        effects: list[StudyEffect] = []
        for record in records:
            reasons = self._base_reasons(profile, record)
            intervention = record.raw_data.get("intervention_arm")
            comparator = record.raw_data.get("comparator_arm")
            if not isinstance(intervention, dict) or not isinstance(comparator, dict):
                reasons.append("Independent continuous data requires intervention_arm and comparator_arm summaries.")
            if not reasons:
                values = {
                    "intervention_n": _positive_number(intervention.get("n")),
                    "intervention_mean": _finite_number(intervention.get("mean")),
                    "intervention_sd": _finite_number(intervention.get("sd")),
                    "comparator_n": _positive_number(comparator.get("n")),
                    "comparator_mean": _finite_number(comparator.get("mean")),
                    "comparator_sd": _finite_number(comparator.get("sd")),
                }
                invalid = [name for name, value in values.items() if value is None or (name.endswith("_sd") and value < 0)]
                if invalid:
                    reasons.append("Missing or invalid continuous summary fields: " + ", ".join(invalid) + ".")
            if reasons:
                decisions.append(PoolabilityDecision(
                    study_id=record.study_id, evidence_id=record.evidence_id,
                    status="insufficient_for_analysis", reasons=reasons,
                ))
                continue
            assert isinstance(intervention, dict) and isinstance(comparator, dict)
            intervention_n = _positive_number(intervention.get("n"))
            intervention_mean = _finite_number(intervention.get("mean"))
            intervention_sd = _finite_number(intervention.get("sd"))
            comparator_n = _positive_number(comparator.get("n"))
            comparator_mean = _finite_number(comparator.get("mean"))
            comparator_sd = _finite_number(comparator.get("sd"))
            assert None not in (
                intervention_n, intervention_mean, intervention_sd,
                comparator_n, comparator_mean, comparator_sd,
            )
            value = intervention_mean - comparator_mean
            variance = (intervention_sd**2 / intervention_n) + (comparator_sd**2 / comparator_n)
            if variance <= 0:
                decisions.append(PoolabilityDecision(
                    study_id=record.study_id, evidence_id=record.evidence_id,
                    status="insufficient_for_analysis",
                    reasons=["Independent-arm mean difference has non-positive within-study variance."],
                ))
                continue
            standard_error = math.sqrt(variance)
            effects.append(StudyEffect(
                study_id=record.study_id,
                evidence_id=record.evidence_id,
                effect_type="MD",
                value=value,
                log_effect=value,
                standard_error=standard_error,
                ci_lower=value - 1.96 * standard_error,
                ci_upper=value + 1.96 * standard_error,
                source_data={
                    "intervention_n": intervention_n,
                    "intervention_mean": intervention_mean,
                    "intervention_sd": intervention_sd,
                    "comparator_n": comparator_n,
                    "comparator_mean": comparator_mean,
                    "comparator_sd": comparator_sd,
                },
                effect_scale="difference",
            ))
            decisions.append(PoolabilityDecision(
                study_id=record.study_id, evidence_id=record.evidence_id,
                status="poolable",
                reasons=["Complete independent-arm n, mean, and SD summaries."],
            ))
        return PoolabilitySet(decisions=decisions, study_effects=effects)

    def _qualify_reported_smd(self, profile: ResolvedAnalysisProfile, records: list) -> PoolabilitySet:
        decisions: list[PoolabilityDecision] = []
        effects: list[StudyEffect] = []
        for record in records:
            reasons = self._base_reasons(profile, record)
            effect_type = str(record.reported_effect_type or "")
            if effect_type != "SMD":
                reasons.append("reported_standardized_mean_difference requires effect_type='SMD'.")
            value = _finite_number(record.reported_effect_value)
            lower = _finite_number(record.ci_lower)
            upper = _finite_number(record.ci_upper)
            if value is None or lower is None or upper is None:
                reasons.append("SMD value and both confidence limits are required.")
            elif not lower <= value <= upper:
                reasons.append("SMD must lie within its confidence interval.")
            if reasons:
                decisions.append(PoolabilityDecision(
                    study_id=record.study_id, evidence_id=record.evidence_id,
                    status="insufficient_for_analysis", reasons=reasons,
                ))
                continue
            standard_error = (upper - lower) / (2 * 1.96)
            if standard_error <= 0 or not math.isfinite(standard_error):
                decisions.append(PoolabilityDecision(
                    study_id=record.study_id, evidence_id=record.evidence_id,
                    status="insufficient_for_analysis",
                    reasons=["confidence interval does not define a positive finite standard error."],
                ))
                continue
            assert value is not None and lower is not None and upper is not None
            effects.append(StudyEffect(
                study_id=record.study_id,
                evidence_id=record.evidence_id,
                effect_type="SMD",
                value=value,
                log_effect=value,
                standard_error=standard_error,
                ci_lower=lower,
                ci_upper=upper,
                source_data=_numeric_source_data(record.raw_data),
                effect_scale="difference",
            ))
            decisions.append(PoolabilityDecision(
                study_id=record.study_id, evidence_id=record.evidence_id,
                status="poolable",
                reasons=["Reported standardized mean difference and confidence interval are complete."],
            ))
        return PoolabilitySet(decisions=decisions, study_effects=effects)


class MetaforPairwiseMetaTool:
    """Pool standardized effects using the installed R metafor implementation."""

    name = "calculate_pairwise_meta_metafor"

    def __init__(self, *, rscript_binary: str = "Rscript", script_path: Path | None = None) -> None:
        self.rscript_binary = rscript_binary
        self.script_path = script_path or Path(__file__).with_name("metafor_pairwise_meta.R")

    def invoke(self, profile: ResolvedAnalysisProfile, effects: list[StudyEffect]) -> MainMetaStatistics:
        if len(effects) < profile.minimum_studies:
            raise ValueError(
                f"Meta-analysis requires {profile.minimum_studies} effects; received {len(effects)}."
            )
        effect_scales = {effect.effect_scale for effect in effects}
        if len(effect_scales) != 1:
            raise ValueError("A pooled result cannot mix ratio-scale and difference-scale effects.")
        effect_types = {effect.effect_type for effect in effects}
        if len(effect_types) != 1 and profile.effect_type != "generic_ratio":
            raise ValueError("A pooled result cannot mix incompatible reported effect types.")
        base_scale = profile.pooling_scale.split("__", 1)[0]
        effective_effect_type = (
            "sensitivity" if base_scale == "logit_sensitivity"
            else "specificity" if base_scale == "logit_specificity"
            else "proportion" if base_scale == "freeman_tukey_proportion"
            else next(iter(effect_types)) if profile.effect_type == "generic_ratio" and len(effect_types) == 1
            else profile.effect_type
        )
        if shutil.which(self.rscript_binary) is None:
            raise RuntimeError("Rscript is required for the metafor pairwise tool but was not found on PATH.")
        with tempfile.TemporaryDirectory(prefix="metafor_pairwise_") as directory:
            workdir = Path(directory)
            input_path = workdir / "effects.csv"
            output_path = workdir / "result.json"
            with input_path.open("w", newline="", encoding="utf-8") as handle:
                fields = ["study_id", "evidence_id", "yi", "vi", "ai", "bi", "ci", "di"]
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for effect in effects:
                    writer.writerow({
                        "study_id": effect.study_id,
                        "evidence_id": effect.evidence_id,
                        "yi": effect.log_effect,
                        "vi": effect.standard_error**2,
                        "ai": effect.source_data.get("ai", ""),
                        "bi": effect.source_data.get("bi", ""),
                        "ci": effect.source_data.get("ci", ""),
                        "di": effect.source_data.get("di", ""),
                    })
            from extensions.r_runtime import run_rscript
            completed = run_rscript(
                [
                    self.rscript_binary,
                    str(self.script_path),
                    str(input_path),
                    str(output_path),
                    profile.pooling_model,
                    effects[0].effect_scale,
                    effective_effect_type,
                    profile.pooling_scale,
                ],
                purpose="pairwise meta-analysis",
            )
            if completed.returncode != 0 or not output_path.exists():
                detail = (completed.stderr or completed.stdout or "metafor did not produce output").strip()
                raise RuntimeError(f"metafor pairwise calculation failed: {detail[:1000]}")
            result = json.loads(output_path.read_text(encoding="utf-8"))
        return MainMetaStatistics(
            effect_type=str(result.get("effect_type") or effective_effect_type),
            model=str(result.get("model") or profile.pooling_model),
            k=int(result["k"]),
            pooled_effect=float(result["pooled_effect"]),
            ci_lower=float(result["ci_lower"]),
            ci_upper=float(result["ci_upper"]),
            i_squared=_finite_number(result.get("i_squared")),
            q=_finite_number(result.get("q")),
            q_p_value=_finite_number(result.get("q_p_value")),
            tau_squared=_finite_number(result.get("tau_squared")),
            member_evidence_ids=[effect.evidence_id for effect in effects],
        )
