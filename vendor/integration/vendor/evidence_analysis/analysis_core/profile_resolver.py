from __future__ import annotations

import re
from typing import Any

from .models import ResolvedAnalysisProfile


REQUIRED_FIELDS_BY_FAMILY = {
    "event_counts_with_person_time": [
        "intervention_episodes",
        "control_episodes",
        "intervention_total",
        "control_total",
    ],
}


def _terms(*values: object) -> list[str]:
    """Deduplicate stable query anchors while preserving their readable form."""
    seen: set[str] = set()
    terms: list[str] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            for term in _terms(*value):
                if term.casefold() not in seen:
                    seen.add(term.casefold())
                    terms.append(term)
            continue
        text = str(value or "").strip()
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            terms.append(text)
    return terms


def _target_query_terms(
    protocol: dict[str, Any], target: dict[str, Any], declared_outcome: str = ""
) -> tuple[list[str], list[str]]:
    pico_exposure = protocol.get("pico", {}).get("intervention_or_exposure", {})
    outcome = target.get("outcome", {})
    return (
        _terms(pico_exposure.get("description"), pico_exposure.get("aliases", [])),
        _terms(declared_outcome, outcome.get("description"), outcome.get("aliases", [])),
    )


def resolve_main_analysis_profile(
    protocol: dict[str, Any], task: dict[str, Any], analysis_profile: dict[str, Any] | None = None
) -> ResolvedAnalysisProfile:
    """Compile a profile from task/protocol inputs without consulting gold data."""
    if analysis_profile is not None:
        analysis_ids = task.get("target_scope", {}).get("analysis_ids", [])
        if len(analysis_ids) != 1:
            raise ValueError("Analysis-profile mode needs exactly one task.target_scope.analysis_ids value.")
        declared = next(
            (item for item in analysis_profile.get("analyses", []) if item.get("analysis_id") == analysis_ids[0]),
            None,
        )
        if declared is None:
            raise ValueError(f"Analysis '{analysis_ids[0]}' is absent from analysis_profile.")
        target_id = str(declared["target_id"])
        target = next((item for item in protocol.get("targets", []) if item.get("target_id") == target_id), None)
        if target is None:
            raise ValueError(f"Analysis target '{target_id}' is absent from extraction protocol.")
        data_families = declared.get("eligible_data_families", [])
        measures = declared.get("accepted_effect_types", [])
        if len(data_families) != 1 or not measures:
            raise ValueError("Analysis-profile needs one data family and at least one accepted effect type per analysis.")
        pooling_model = declared.get("pooling", {}).get("model")
        if pooling_model not in {"fixed_effect", "random_effects", "unspecified"}:
            raise NotImplementedError(f"Analysis agent does not implement '{pooling_model}'.")
        exposure_terms, outcome_terms = _target_query_terms(protocol, target, str(declared["outcome_group"]))
        selection_rules = declared.get("selection_rules", {})
        return ResolvedAnalysisProfile(
            analysis_id=str(declared["analysis_id"]),
            task_id=task.get("task_id", ""),
            protocol_id=protocol.get("protocol_id", ""),
            target_id=target_id,
            outcome=str(declared["outcome_group"]),
            comparison=target.get("comparison", ""),
            data_family=data_families[0],
            effect_type=measures[0] if len(measures) == 1 else "generic_ratio",
            pooling_model=pooling_model,
            pooling_scale=str(declared.get("pooling", {}).get("scale", "log_ratio_inverse_variance")),
            minimum_studies=int(declared.get("pooling", {}).get("minimum_independent_studies", 2)),
            required_raw_fields=REQUIRED_FIELDS_BY_FAMILY.get(data_families[0], []),
            required_record_fields=["follow_up", "quantitative_usability"],
            exposure_query_terms=exposure_terms,
            outcome_query_terms=outcome_terms,
            eligible_estimands=[str(value) for value in declared.get("estimands", [])],
            eligible_outcome_codes=[str(value) for value in declared.get("eligible_outcome_codes", [])],
            accepted_effect_types=[str(value) for value in measures],
            # The benchmark's legacy flag means one selected model within a
            # study/exposure/outcome decision unit, not one effect from the
            # entire study across distinct outcomes or analyses.
            one_effect_per_triplet=bool(selection_rules.get("one_effect_per_study", True)),
            selection_rules=[str(rule) for rule in selection_rules.get("rules", [])],
            derivation_trace=[
                f"analysis_id={declared['analysis_id']} from analysis_profile",
                f"target_id={target_id} and data_family={data_families[0]} from analysis_profile",
                f"effect_type={measures[0]} and pooling_model={pooling_model} from analysis_profile",
            ],
        )
    main_outcomes = task.get("target_scope", {}).get("main_outcomes", [])
    if len(main_outcomes) != 1:
        raise ValueError(
            "Main-analysis baseline needs exactly one task.target_scope.main_outcomes target."
        )

    target_id = main_outcomes[0]
    target = next((item for item in protocol.get("targets", []) if item.get("target_id") == target_id), None)
    if target is None:
        raise ValueError(f"Task target '{target_id}' is absent from extraction protocol.")

    data_families = target.get("accepted_data_families", [])
    measures = target.get("preferred_effect_measures", [])
    if len(data_families) != 1 or not measures:
        raise ValueError(
            "Main-analysis needs one accepted data family and at least one preferred effect measure per target."
        )

    definition = task.get("task_definition", "").lower()
    pooling_model = (
        "fixed_effect" if re.search(r"fixed[ -]?effect", definition)
        else "random_effects" if re.search(r"random[ -]?effects?", definition)
        else "unspecified"
    )
    data_family = data_families[0]
    effect_type = measures[0] if len(measures) == 1 else "generic_ratio"
    outcome = target.get("outcome", {}).get("description", target_id)
    exposure_terms, outcome_terms = _target_query_terms(protocol, target, str(outcome))

    return ResolvedAnalysisProfile(
        analysis_id=f"{protocol.get('case_id', 'case')}_{target_id}_main_baseline",
        task_id=task.get("task_id", ""),
        protocol_id=protocol.get("protocol_id", ""),
        target_id=target_id,
        outcome=outcome,
        comparison=target.get("comparison", ""),
        data_family=data_family,
        effect_type=effect_type,
        pooling_model=pooling_model,
        pooling_scale="log_ratio_inverse_variance",
        required_raw_fields=REQUIRED_FIELDS_BY_FAMILY.get(data_family, []),
        required_record_fields=["follow_up", "quantitative_usability"],
        exposure_query_terms=exposure_terms,
        outcome_query_terms=outcome_terms,
        eligible_estimands=[str(value) for value in target.get("estimands", [])],
        accepted_effect_types=[str(value) for value in measures],
        one_effect_per_triplet=True,
        selection_rules=[str(target.get("effect_selection_rule"))] if target.get("effect_selection_rule") else [],
        derivation_trace=[
            f"target_id={target_id} from task.target_scope.main_outcomes",
            f"data_family={data_family} and effect_type={effect_type} from extraction protocol target",
            f"pooling_model={pooling_model} parsed from task_definition",
        ],
    )
