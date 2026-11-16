"""Adapt a standard frozen analysis benchmark case to the graph tool contract."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


_TIMEPOINT_POLICIES = {"longest_eligible", "all_eligible", "source_defined"}
_PROFILE_METADATA_FIELDS = (
    "outcome_axis",
    "measurement_semantics",
    "unit",
    "intervention_class",
    "comparator_class",
    "rsid",
    "study_design",
    "phenotype",
    "genotype_role",
)


def _candidate_analysis_metadata(candidate: dict[str, Any]) -> dict[str, Any]:
    """Read optional analysis semantics without expanding the public envelope.

    The public Current contract keeps a fixed set of top-level Evidence fields.
    When a continuous-outcome case needs measurement semantics to distinguish
    otherwise generic source summaries, those semantics live inside the
    existing ``outcome`` object as ``analysis_semantics``.  Older packages
    retain their top-level representation, so this is backwards compatible.
    """
    metadata = {
        key: candidate[key]
        for key in _PROFILE_METADATA_FIELDS
        if candidate.get(key) not in (None, "", [], {})
    }
    outcome = candidate.get("outcome")
    if not isinstance(outcome, dict):
        return metadata
    nested = outcome.get("analysis_semantics")
    if not isinstance(nested, dict):
        return metadata
    for key in _PROFILE_METADATA_FIELDS:
        value = nested.get(key)
        if key not in metadata and value not in (None, "", [], {}):
            metadata[key] = value
    return metadata


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _terms(value: object) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return []


def _candidate_outcome(candidate: dict[str, Any]) -> dict[str, str]:
    """Normalise nested and flat frozen-candidate outcome representations."""
    raw = candidate.get("outcome")
    if isinstance(raw, dict):
        return {
            "label": str(raw.get("label") or raw.get("description") or raw.get("code") or ""),
            "code": str(raw.get("code") or candidate.get("outcome_code") or ""),
        }
    return {
        "label": str(raw or candidate.get("outcome_label") or candidate.get("outcome_code") or ""),
        "code": str(candidate.get("outcome_code") or ""),
    }


def _effective_eligible_outcome_codes(
    declared_codes: list[str], *, target_id: str, candidates: list[dict[str, Any]]
) -> list[str]:
    """Keep source codes hard only when they are actual source-level identifiers.

    Some frozen cases put the target-level semantic label in
    ``eligible_outcome_codes`` (for example, ``adverse_cardiac_event``), while
    the evidence ledger correctly preserves several more specific source
    outcome codes.  The target edge already constrains this situation.  An
    exact source-code edge would silently discard eligible source variants.
    """
    codes = [code for code in declared_codes if code]
    source_codes = {
        _candidate_outcome(candidate)["code"]
        for candidate in candidates
        if str(candidate.get("target_id") or "") == target_id
        and _candidate_outcome(candidate)["code"]
    }
    if codes == [target_id] and len(source_codes) > 1:
        return []
    return codes


def _pico_component(protocol: dict[str, Any], *keys: str) -> dict[str, Any]:
    """Read a PICO component while keeping source-schema variation at the adapter edge."""
    pico = protocol.get("pico") or {}
    for key in keys:
        value = pico.get(key)
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value:
            return {"description": value}
    return {}


def _component_terms(component: dict[str, Any]) -> list[str]:
    return [
        term
        for term in [str(component.get("description") or ""), *(_terms(component.get("aliases")))]
        if term
    ]


def _target_outcome_terms(target: dict[str, Any], fallback: str) -> list[str]:
    """Expose protocol outcome wording to graph retrieval without renaming the analysis."""
    outcome = target.get("outcome") or {}
    if isinstance(outcome, dict):
        terms = _component_terms(outcome)
        if terms:
            return terms
    return [fallback] if fallback else []


def _single_protocol_target_id(protocol: dict[str, Any]) -> str:
    """Use a protocol target only when its identity is unambiguous.

    Some portable profiles state their target once in the extraction protocol
    and define analysis topics through measurement metadata.  This is a schema
    variation, not an invitation to infer a target from Gold or source values.
    """
    targets = [item for item in protocol.get("targets", []) if item.get("target_id")]
    return str(targets[0]["target_id"]) if len(targets) == 1 else ""


def _inferred_effect_types(source: dict[str, Any], pooling: dict[str, Any]) -> list[str]:
    """Fill a portable profile's omitted effect label from its declared scale."""
    declared = _terms(source.get("accepted_effect_types"))
    if declared:
        return declared
    scale = str(pooling.get("scale") or "").casefold()
    if "mean_difference" in scale:
        return ["MD"]
    if "risk_difference" in scale:
        return ["RD"]
    if "log_odds" in scale or "log_or" in scale:
        return ["OR"]
    if "log_risk" in scale or "log_rr" in scale:
        return ["RR"]
    if "logit_sensitivity" in scale or "logit_specificity" in scale:
        return ["diagnostic_accuracy_pair"]
    if scale == "proportion":
        return ["proportion"]
    return []


def _inferred_runtime_scale(source: dict[str, Any], effect_types: list[str]) -> str:
    """Route a method-neutral public task to a family-compatible calculator.

    This is a generic capability map, not a review-specific method recipe. The
    public task specifies its data family and admissible effect type; model,
    heterogeneity estimator and interval method are selected later by the
    method-selection tool.
    """
    family = str((_terms(source.get("eligible_data_families")) or [""])[0])
    effect_type = effect_types[0] if len(effect_types) == 1 else ""
    if family == "diagnostic_four_cell_counts":
        analysis_id = str(source.get("analysis_id") or "").casefold()
        outcome = str(source.get("outcome_group") or "").casefold()
        return "logit_specificity" if "specific" in analysis_id or "specific" in outcome else "logit_sensitivity"
    if family == "single_arm_event_counts":
        return "freeman_tukey_proportion"
    if family == "independent_continuous_arms":
        return "mean_difference_inverse_variance"
    if family == "reported_standardized_mean_difference":
        return "inverse_variance_standardized_mean_difference"
    if family == "event_counts_with_person_time":
        return "log_rate_ratio_inverse_variance"
    if effect_type == "RD":
        return "risk_difference_inverse_variance"
    if effect_type == "OR":
        return "log_odds_ratio_inverse_variance"
    if effect_type == "RR":
        return "log_risk_ratio_inverse_variance"
    return "log_ratio_inverse_variance"


def _hydrate_method_neutral_structured_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Add private calculator dispatch keys to a public Structured profile without methods.

    Format deliberately removes every public ``pooling`` object.  The Structured resolver
    still needs an internal family route after it has read the public data
    family and effect representation.  This creates that transient route in
    memory only; it is removed again before the Evidence Reviewer is prompted.
    """
    hydrated = deepcopy(profile)
    primary = hydrated.get("primary_analysis_spec")
    if not isinstance(primary, dict):
        return hydrated

    def add_runtime_route(raw: dict[str, Any], inherited: dict[str, Any] | None = None) -> dict[str, Any]:
        effective = deepcopy(inherited or {})
        effective.update(raw)
        source_pooling = effective.get("pooling") or {}
        effect_types = _inferred_effect_types(effective, source_pooling)
        minimum = effective.get("minimum_independent_studies")
        if minimum is None:
            minimum = source_pooling.get(
                "minimum_independent_studies", source_pooling.get("minimum_studies", 2)
            )
        routed = deepcopy(raw)
        routed["pooling"] = {
            "model": "unspecified",
            "scale": _inferred_runtime_scale(effective, effect_types),
            "minimum_studies": int(minimum),
        }
        return routed

    original_primary = deepcopy(primary)
    hydrated["primary_analysis_spec"] = add_runtime_route(primary)
    hydrated["secondary_outcomes"] = [
        add_runtime_route(item, original_primary)
        for item in hydrated.get("secondary_outcomes", [])
        if isinstance(item, dict)
    ]
    return hydrated


def _selection_metadata_rules(source: dict[str, Any]) -> list[str]:
    """Expose declared measurement constraints to the local reviewer.

    The extraction candidates preserve these fields at top level rather than
    inside raw numeric data.  Graph carries them into the reviewer payload so a
    continuous-outcome topic can distinguish, for example, version from tilt
    without inventing a case-specific filter.
    """
    mappings = {
        "outcome_axis": "outcome_axis",
        "accepted_measurement_semantics": "measurement_semantics",
        "unit": "unit",
        "intervention_class": "intervention_class",
        "comparator_class": "comparator_class",
        "rsid": "rsid",
        "study_design": "study_design",
        "phenotype": "phenotype",
        "genotype_role": "genotype_role",
    }
    rules: list[str] = []
    for profile_key, candidate_key in mappings.items():
        values = _terms(source.get(profile_key))
        if values:
            rules.append(
                f"Metadata constraint: {candidate_key} must match one of: {', '.join(values)}."
            )
    return rules


def _runtime_pooling_scale(pooling: dict[str, Any]) -> str:
    """Preserve public synthesis rules in an internal calculator dispatch key."""
    scale = str(pooling.get("scale") or "log_ratio_inverse_variance")
    specification = pooling.get("calculator_specification") or {}
    synthesis_policy = pooling.get("source_synthesis_policy") or {}
    calculator_id = str(
        specification.get("calculator_id") or synthesis_policy.get("calculator_spec") or ""
    )
    if calculator_id == "single_arm_proportion_freeman_tukey_dl_baseline":
        scale = "freeman_tukey_proportion"
    heterogeneity_estimator = str(pooling.get("heterogeneity_estimator") or "").casefold()
    if heterogeneity_estimator == "reml":
        scale += "__reml"
    elif heterogeneity_estimator.replace("-", "_").replace(" ", "_") in {
        "paule_mandel",
        "pm",
    }:
        scale += "__paule_mandel"
    confidence_interval = str(pooling.get("confidence_interval") or "").casefold()
    if confidence_interval.replace("-", "_").replace(" ", "_") in {
        "hartung_knapp_t",
        "hartung_knapp",
        "knapp_hartung",
        "knha",
    }:
        scale += "__hartung_knapp_t"
    model_selection = str(
        (specification.get("pooling") or {}).get("model_selection")
        or (synthesis_policy.get("pooling") or {}).get("model_selection")
        or ""
    ).casefold()
    if "q-test p<0.10" in model_selection and "i2>50" in model_selection:
        scale += "__heterogeneity_switch"
    return scale


def _extension_subgroup_dimensions(case_root: Path) -> list[dict[str, Any]]:
    """Compile public, protocol-declared extension topics without reading Gold.

    Dimensions listed under ``subgroup_analyses`` reproduce a target review's
    stated analysis plan.  ``candidate_subgroup_analyses`` model prospective
    user requests.  Both are only questions: eligibility remains dependent on
    source fields emitted by the extractor at runtime.
    """
    path = case_root / "inputs" / "analysis_extensions.json"
    if not path.exists():
        return []
    extension = _load(path)
    dimensions: list[dict[str, Any]] = []
    for key, exploratory in (
        ("subgroup_analyses", False),
        ("candidate_subgroup_analyses", True),
    ):
        for source in extension.get(key) or []:
            levels = []
            for level in source.get("levels") or []:
                # Include a compact canonical ID alongside human wording so
                # both structured extractors and narrative source records can
                # satisfy the same predeclared level definition.
                levels.append({
                    "value_id": str(level["level_id"]),
                    "label": str(level["label"]),
                    "terms": [
                        str(level["level_id"]),
                        str(level["label"]),
                        str(level["source_definition"]),
                    ],
                })
            dimensions.append({
                "dimension_id": str(source["analysis_id"]),
                "label": str(source["label"]),
                "field": str(source["grouping_variable"]),
                "levels": levels,
                "minimum_studies_per_level": int(source["minimum_independent_studies"]),
                "exploratory": exploratory,
            })
    return dimensions


def merge_extension_subgroup_dimensions(
    profile: dict[str, Any], dimensions: list[dict[str, Any]]
) -> dict[str, Any]:
    """Add extension topics without replacing a case's existing frozen plan.

    A specialised case may already expose part of an extension plan.  The
    dimension ID is the stable protocol key: an existing ID wins so a later
    declaration cannot silently change the original level rules.
    """
    merged = deepcopy(profile)
    existing = list(merged.get("candidate_subgroup_dimensions") or [])
    existing_ids = {str(item.get("dimension_id") or "") for item in existing}
    existing.extend(item for item in dimensions if item["dimension_id"] not in existing_ids)
    merged["candidate_subgroup_dimensions"] = existing
    return merged


def _compile_analysis_spec(
    protocol: dict[str, Any],
    source: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Convert one public benchmark analysis declaration to the structured runtime form."""
    source = dict(source)
    target_id = str(source.get("target_id") or _single_protocol_target_id(protocol))
    if not target_id:
        raise ValueError(
            f"Analysis '{source.get('analysis_id', 'unknown')}' has no target_id and the protocol "
            "does not declare exactly one target."
        )
    target = next((item for item in protocol.get("targets", []) if item.get("target_id") == target_id), {})
    exposure = _pico_component(protocol, "intervention_or_exposure", "intervention", "exposure")
    comparator = _pico_component(protocol, "comparator", "comparison", "control")
    selection = source.get("selection_rules") or {}
    pooling = source.get("pooling") or {}
    eligible_outcome_codes = _effective_eligible_outcome_codes(
        _terms(source.get("eligible_outcome_codes")), target_id=target_id, candidates=candidates
    )
    available_outcome_codes = {
        _candidate_outcome(candidate)["code"]
        for candidate in candidates
        if str(candidate.get("target_id") or "") == target_id
    }
    # Some frozen baseline cases use a target-level outcome code in the profile but
    # preserve source-level outcome codes in the evidence ledger.  Only keep
    # an outcome-code restriction when it has an actual candidate-side match;
    # target ID and estimand remain hard constraints either way.
    if eligible_outcome_codes and not (set(eligible_outcome_codes) & available_outcome_codes):
        eligible_outcome_codes = []
    rules = _terms(selection.get("rules") if isinstance(selection, dict) else selection)
    rules.extend(_selection_metadata_rules(source))
    timepoint_policy = str(source.get("timepoint_policy") or "source_defined")
    if timepoint_policy not in _TIMEPOINT_POLICIES:
        rules.append(f"Timepoint selection rule: {timepoint_policy}")
        timepoint_policy = "source_defined"
    accepted_effect_types = _inferred_effect_types(source, pooling)
    public_scale = pooling.get("scale")
    return {
        "analysis_id": source["analysis_id"],
        "pooled_result_id": source.get("pooled_result_id", source["analysis_id"]),
        "title": source.get("title", source["analysis_id"]),
        "target_id": target_id,
        "outcome_group": source.get("outcome_group") or source.get("title") or target_id,
        "outcome_terms": _target_outcome_terms(
            target, str(source.get("outcome_group") or target_id)
        ),
        "eligible_outcome_codes": eligible_outcome_codes,
        "eligible_data_families": _terms(source.get("eligible_data_families")),
        "accepted_effect_types": accepted_effect_types,
        "estimands": _terms(source.get("estimands")),
        "exposure": {"terms": _component_terms(exposure)},
        # Comparison wording is retained on every Evidence node for the
        # Reviewer. It is not a hard generic adapter filter, because sources
        # can express the same PICO contrast with different arm labels.
        "comparison": {"terms": []},
        "population": {
            "policy": source.get("population_policy", "any"),
            "terms": _terms(source.get("population_terms")),
        },
        "timepoint_policy": timepoint_policy,
        "model_policy": source.get(
            "model_policy",
            "Prefer the most fully adjusted eligible model within each source context.",
        ),
        "dependence_policy": (
            "one_effect_per_study" if isinstance(selection, dict) and selection.get("one_effect_per_study", False)
            else source.get("dependence_policy", "one_effect_per_study_outcome")
        ),
        "selection_rules": rules,
        "pooling": {
            "model": pooling.get("model", "unspecified"),
            "scale": (
                _runtime_pooling_scale(pooling)
                if public_scale else _inferred_runtime_scale(source, accepted_effect_types)
            ),
            "minimum_studies": int(source.get(
                "minimum_independent_studies",
                pooling.get("minimum_independent_studies", pooling.get("minimum_studies", 2)),
            )),
        },
    }


def _compile_analysis_profile(
    protocol: dict[str, Any],
    profile: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    extension_dimensions: list[dict[str, Any]] | None = None,
    sensitivity_dimensions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compile all declared primary/secondary analysis topics for the graph agent."""
    analyses = profile.get("analyses") or []
    if not analyses:
        raise ValueError("The public analysis profile does not declare any analyses.")
    compiled = [_compile_analysis_spec(protocol, item, candidates) for item in analyses]
    return {
        "profile_version": "2.0.0",
        "plan_id": f"{profile.get('case_id', 'case')}_declared_analysis_plan_source",
        "primary_analysis_spec": compiled[0],
        # Every remaining declared analysis remains an executable, scored
        # secondary outcome rather than being discarded by a single-analysis
        # adapter assumption.
        "secondary_outcomes": compiled[1:],
        "candidate_subgroup_dimensions": extension_dimensions or [],
        "candidate_sensitivity_dimensions": sensitivity_dimensions or [],
    }


def load_public_case_graph(
    case_root: Path,
    *,
    prefer_frozen_structured_profile: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    """Load public case inputs only; no gold or review material is read."""
    protocol = _load(case_root / "inputs" / "extraction_protocol.json")
    task_path = case_root / "tasks" / "analysis_main_outcomes.json"
    if not task_path.exists():
        task_path = case_root / "inputs" / "analysis_task.json"
    task = _load(task_path)
    # graph analysis cases may expose an unlabeled projection so older frozen
    # extraction/analysis versions can retain their historical contract.
    candidate_path = case_root / "inputs" / "analysis_evidence_candidates_source_unlabelled.json"
    if not candidate_path.exists():
        candidate_path = case_root / "inputs" / "analysis_evidence_candidates.json"
    candidates = _load(candidate_path)["candidates"]
    extension_dimensions = _extension_subgroup_dimensions(case_root)
    structured_profile_path = case_root / "inputs" / "analysis_profile_structured.json"
    if prefer_frozen_structured_profile and structured_profile_path.exists():
        # Some cases already have a frozen multi-topic structured profile. Treat it as
        # the runtime contract verbatim: it may contain predeclared secondary,
        # subgroup, and sensitivity topics beyond the score-only baseline profile.
        # Reading it does not alter benchmark inputs or create new tasks.
        structured_profile = _load(structured_profile_path)
        if structured_profile.get("method_selection_policy"):
            structured_profile = _hydrate_method_neutral_structured_profile(structured_profile)
        profile = merge_extension_subgroup_dimensions(structured_profile, extension_dimensions)
    else:
        profile = _compile_analysis_profile(
            protocol,
            _load(case_root / "inputs" / "analysis_profile.json"),
            candidates,
            extension_dimensions=extension_dimensions,
        )
    exposure = _pico_component(protocol, "intervention_or_exposure", "intervention", "exposure")
    exposure_fallback = str(exposure.get("description") or "")
    by_article: dict[str, list[dict[str, Any]]] = {}
    study_by_article: dict[str, str] = {}
    for candidate in candidates:
        outcome = _candidate_outcome(candidate)
        effect = dict(candidate.get("effect") or {})
        article_id = str(candidate["article_id"])
        study_id = str(candidate["study_id"])
        metadata = _candidate_analysis_metadata(candidate)
        raw_data = dict(candidate.get("raw_data") or {})
        if metadata:
            raw_data["_analysis_metadata"] = metadata
        by_article.setdefault(article_id, []).append({
            "evidence_id": candidate["candidate_id"],
            "article_id": article_id,
            "study_id": study_id,
            "target_id": candidate.get("target_id"),
            "intervention_or_exposure": (
                candidate.get("exposure")
                or candidate.get("intervention_or_exposure")
                or exposure_fallback
            ),
            "comparator": candidate.get("comparison") or "",
            "outcome": outcome["label"],
            "outcome_code": outcome["code"],
            "population_or_subgroup": candidate.get("population_or_subgroup") or "",
            "follow_up": candidate.get("timepoint") or "",
            "data_family": candidate.get("data_family") or "",
            "estimand": candidate.get("estimand") or "",
            "raw_data": raw_data,
            "reported_effect_type": effect.get("effect_type"),
            "reported_effect_value": effect.get("value"),
            "ci_lower": effect.get("ci_lower"),
            "ci_upper": effect.get("ci_upper"),
            "source_location": candidate.get("source_location") or {},
        })
        study_by_article[article_id] = study_id
    payloads = [
        {"article_id": article_id, "study_id": study_by_article[article_id], "evidence": evidence}
        for article_id, evidence in sorted(by_article.items())
    ]
    return protocol, task, profile, payloads, study_by_article
