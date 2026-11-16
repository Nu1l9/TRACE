from __future__ import annotations

from copy import deepcopy
from typing import Any

from .models import (
    AnalysisDefinition,
    CandidateAnalysisDimension,
    ResolvedAnalysisPlan,
    ResolvedAnalysisProfile,
)
from .profile_resolver import REQUIRED_FIELDS_BY_FAMILY


def _strings(values: object) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values] if values.strip() else []
    if isinstance(values, (list, tuple)):
        return [str(value) for value in values if str(value).strip()]
    return [str(values)]


def _definition(raw: dict[str, Any], *, kind: str) -> AnalysisDefinition:
    pooling = raw.get("pooling", {})
    population = raw.get("population", {})
    exposure = raw.get("exposure", {})
    comparison = raw.get("comparison", {})
    return AnalysisDefinition(
        analysis_id=str(raw["analysis_id"]),
        pooled_result_id=str(raw.get("pooled_result_id") or raw["analysis_id"]),
        title=str(raw.get("title") or raw["analysis_id"]),
        kind=kind,
        target_id=str(raw["target_id"]),
        outcome_group=str(raw.get("outcome_group") or raw["target_id"]),
        outcome_terms=_strings(raw.get("outcome_terms")),
        eligible_outcome_codes=_strings(raw.get("eligible_outcome_codes")),
        eligible_data_families=_strings(raw.get("eligible_data_families")),
        accepted_effect_types=_strings(raw.get("accepted_effect_types")),
        estimands=_strings(raw.get("estimands")),
        exposure_terms=_strings(exposure.get("terms") if isinstance(exposure, dict) else exposure),
        comparison_terms=_strings(
            comparison.get("terms") if isinstance(comparison, dict) else comparison
        ),
        population_policy=str(population.get("policy", "overall_only")),
        population_terms=_strings(population.get("terms")),
        timepoint_policy=str(raw.get("timepoint_policy", "longest_eligible")),
        model_policy=str(raw.get("model_policy", "prefer the most fully adjusted eligible model")),
        pooling_model=str(pooling.get("model", "random_effects")),
        pooling_scale=str(pooling.get("scale", "log_ratio_inverse_variance")),
        minimum_studies=int(pooling.get("minimum_studies", 2)),
        dependence_policy=str(raw.get("dependence_policy", "one_effect_per_study_outcome")),
        parent_analysis_id=raw.get("parent_analysis_id"),
        dimension_id=raw.get("dimension_id"),
        dimension_field=raw.get("dimension_field"),
        dimension_value_id=raw.get("dimension_value_id"),
        dimension_terms=_strings(raw.get("dimension_terms")),
        prespecified=bool(raw.get("prespecified", True)),
        selection_rules=_strings(raw.get("selection_rules")),
    )


def resolve_analysis_plan_structured(
    protocol: dict[str, Any], task: dict[str, Any], analysis_profile: dict[str, Any]
) -> ResolvedAnalysisPlan:
    """Compile public structured analysis inputs into executable analysis definitions.

    Candidate dimensions declare analyses the agent is allowed to attempt. They
    never identify evidence IDs and therefore do not leak benchmark selection
    labels.
    """
    version = str(analysis_profile.get("profile_version", ""))
    if not version.startswith("2"):
        raise ValueError("Analysis Agent structured requires analysis_profile.profile_version starting with '2'.")

    primary_raw = analysis_profile.get("primary_analysis_spec")
    if not isinstance(primary_raw, dict):
        raise ValueError("analysis_profile.primary_analysis_spec is required.")
    primary = _definition(primary_raw, kind="primary")
    if len(primary.eligible_data_families) != 1:
        raise ValueError("Each structured analysis currently requires exactly one eligible data family.")
    if not primary.accepted_effect_types:
        raise ValueError("The primary analysis must declare accepted_effect_types.")

    analyses: list[AnalysisDefinition] = [primary]
    for raw in analysis_profile.get("secondary_outcomes", []):
        merged = deepcopy(primary_raw)
        merged.update(raw)
        merged["parent_analysis_id"] = primary.analysis_id
        analyses.append(_definition(merged, kind="secondary_outcome"))

    dimensions: list[CandidateAnalysisDimension] = []
    for key, kind in (
        ("candidate_subgroup_dimensions", "subgroup"),
        ("candidate_sensitivity_dimensions", "sensitivity"),
    ):
        for raw_dimension in analysis_profile.get(key, []):
            dimension = CandidateAnalysisDimension(kind=kind, **raw_dimension)
            dimensions.append(dimension)
            for level in dimension.levels:
                derived = deepcopy(primary_raw)
                derived.update(deepcopy(level.analysis_overrides))
                derived.update(
                    {
                        "analysis_id": f"{kind}_{dimension.dimension_id}_{level.value_id}",
                        "pooled_result_id": (
                            f"{primary_raw.get('pooled_result_id') or primary_raw['analysis_id']}"
                            f"__{kind}_{dimension.dimension_id}_{level.value_id}"
                        ),
                        "title": f"{dimension.label}: {level.label}",
                        "parent_analysis_id": primary.analysis_id,
                        "dimension_id": dimension.dimension_id,
                        "dimension_field": dimension.field,
                        "dimension_value_id": level.value_id,
                        "dimension_terms": level.terms,
                        "prespecified": not dimension.exploratory,
                    }
                )
                derived_pooling = dict(primary_raw.get("pooling", {}))
                derived_pooling["minimum_studies"] = dimension.minimum_studies_per_level
                derived["pooling"] = derived_pooling
                if kind == "subgroup" and dimension.field == "population_or_subgroup":
                    derived["population"] = {
                        "policy": "dimension_level",
                        "terms": level.terms,
                    }
                elif kind == "subgroup":
                    # A protocol-defined study/evidence metadata field (for
                    # example, surgical procedure) is adjudicated after graph
                    # retrieval, not mistaken for a population-string filter.
                    derived["population"] = {"policy": "any", "terms": []}
                analyses.append(_definition(derived, kind=kind))

    analysis_ids = [analysis.analysis_id for analysis in analyses]
    if len(set(analysis_ids)) != len(analysis_ids):
        raise ValueError("Every generated analysis_id must be unique.")

    return ResolvedAnalysisPlan(
        plan_id=str(analysis_profile.get("plan_id") or f"{protocol.get('case_id', 'case')}_analysis_structured"),
        profile_version=version,
        task_id=str(task.get("task_id", "")),
        protocol_id=str(protocol.get("protocol_id", "")),
        primary_analysis_id=primary.analysis_id,
        analyses=analyses,
        candidate_dimensions=dimensions,
        derivation_trace=[
            f"Primary analysis '{primary.analysis_id}' was read from primary_analysis_spec.",
            f"Expanded {len(analysis_profile.get('secondary_outcomes', []))} secondary outcomes.",
            f"Expanded {len(dimensions)} candidate dimensions into {len(analyses) - 1 - len(analysis_profile.get('secondary_outcomes', []))} executable analysis levels.",
            "No evidence IDs or benchmark gold labels were read while resolving the plan.",
        ],
    )


def runtime_profile_for_analysis(
    definition: AnalysisDefinition, plan: ResolvedAnalysisPlan
) -> ResolvedAnalysisProfile:
    effect_type = (
        definition.accepted_effect_types[0]
        if len(definition.accepted_effect_types) == 1
        else "generic_ratio"
    )
    return ResolvedAnalysisProfile(
        analysis_id=definition.analysis_id,
        task_id=plan.task_id,
        protocol_id=plan.protocol_id,
        target_id=definition.target_id,
        outcome=definition.outcome_group,
        comparison=" | ".join(definition.comparison_terms),
        data_family=definition.eligible_data_families[0],
        effect_type=effect_type,
        pooling_model=definition.pooling_model,
        pooling_scale=definition.pooling_scale,
        minimum_studies=definition.minimum_studies,
        required_raw_fields=REQUIRED_FIELDS_BY_FAMILY.get(
            definition.eligible_data_families[0], []
        ),
        required_record_fields=["follow_up", "quantitative_usability"],
        exposure_query_terms=definition.exposure_terms,
        outcome_query_terms=(
            definition.outcome_terms
            or [definition.outcome_group, *definition.eligible_outcome_codes]
        ),
        eligible_estimands=definition.estimands,
        eligible_outcome_codes=definition.eligible_outcome_codes,
        accepted_effect_types=definition.accepted_effect_types,
        one_effect_per_triplet=True,
        selection_rules=definition.selection_rules,
        derivation_trace=[
            f"Compiled from Analysis Agent structured definition '{definition.analysis_id}'.",
            f"kind={definition.kind}; dependence_policy={definition.dependence_policy}",
        ],
    )
