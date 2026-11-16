"""Compile study requests into normalized, stage-specific pipeline inputs.

This module reads only the request. It never reads benchmark labels or outcomes.
Schema belongs in structured output; task semantics belong in these inputs.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from .statistical_methods import protocol_method_policy

CANONICAL_CASES = tuple('ma_' + n for n in (
    '0001', '0005', '0010', '0011', '0012', '0017', '0024', '0025',
    '0026', '0028', '0032', '0034', '0037', '0041', '0044', '0046',
    '0054', '0055', '0056', '0064', '0076', '0086'))
FAMILY_ALIASES = {name: 'binary_arm_counts' for name in (
    'binary_arm_outcome', 'binary_arm_event_counts', 'two_group_event_counts',
    'binary_exposure_outcome_counts')}
FAMILY_ALIASES['adjusted_time_to_event_effect'] = 'adjusted_ratio_effect'
SUPPORTED_ANALYSIS_FAMILIES = {
    'binary_arm_counts', 'adjusted_ratio_effect', 'reported_ratio_effect',
    'time_to_event', 'reported_standardized_mean_difference',
    'complete_genotype_case_control_counts', 'diagnostic_four_cell_counts',
    'single_arm_event_counts', 'independent_continuous_arms',
    'event_counts_with_person_time',
}


class InputContractError(ValueError):
    """An explicit, actionable input error before model work starts."""


def _unique(values):
    return list(dict.fromkeys(value for value in values if value))


def description(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return description(value.get('description') or value.get('label') or value.get('rule'))
    if isinstance(value, (list, tuple)):
        return '; '.join(filter(None, (description(v) for v in value)))
    raise InputContractError('Semantic descriptions must be strings, objects, or lists')


def _terms(value: Any) -> list[str]:
    if isinstance(value, dict):
        value = value.get('terms') if 'terms' in value else description(value)
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if value is None:
        return []
    if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
        raise InputContractError('Semantic term lists must contain strings')
    return _unique(v.strip() for v in value)


def selection_rules(value: Any) -> list[str]:
    if isinstance(value, dict):
        unknown = set(value) - {'rules', 'one_effect_per_study', 'description'}
        if unknown:
            raise InputContractError('Unsupported selection_rules keys: ' + ', '.join(sorted(unknown)))
        rules = _terms(value.get('rules'))
        if value.get('one_effect_per_study') is True:
            rules.append('Select at most one eligible effect per independent study for this analysis.')
        if value.get('description'):
            rules.extend(_terms(value['description']))
        return _unique(rules)
    return _terms(value)


def _timepoint(value: Any) -> str:
    if not isinstance(value, dict):
        return description(value)
    parts = [description(value.get('rule'))]
    if description(value.get('preferred')):
        parts.append('Preferred timepoint: ' + description(value['preferred']))
    if description(value.get('allowed')):
        parts.append('Allowed timepoints: ' + description(value['allowed']))
    return '; '.join(filter(None, parts))


def _compile_spec(raw: dict, target: dict, pico: dict) -> dict:
    spec = deepcopy(raw)
    spec.setdefault('outcome_group', description(target.get('outcome')))
    # Extraction targets can deliberately cover several clinical endpoints.
    # The analysis question is narrower (e.g. stroke within cardiovascular
    # events), so use its declared group before falling back to that target.
    outcome_group = description(spec.get('outcome_group')).replace('_', ' ')
    spec['outcome_terms'] = (_terms(spec.get('outcome_terms'))
        or _terms(outcome_group) or _terms(target.get('outcome')))
    for name, fallback in (
        ('exposure', pico.get('intervention_or_exposure') or pico.get('exposure') or pico.get('intervention')),
        ('comparison', target.get('comparison') or pico.get('comparison') or pico.get('comparator')),
        ('population', target.get('population_or_subgroup') or pico.get('population')),
    ):
        current = spec.get(name)
        entry = deepcopy(current) if isinstance(current, dict) else {'terms': _terms(current)}
        entry['terms'] = _terms(entry.get('terms')) or _terms(fallback)
        if name == 'population':
            # The legacy overall_only path performs lexical phrase matching.
            # These descriptions are semantic constraints, not exact-match keys.
            entry.setdefault('policy', 'any')
        spec[name] = entry
    raw_timepoint = spec.get('timepoint_policy')
    controls = {'longest_eligible', 'all_eligible', 'source_defined'}
    is_control = isinstance(raw_timepoint, str) and raw_timepoint in controls
    spec['timepoint_policy'] = raw_timepoint if is_control else 'source_defined'
    spec['timepoint_constraints'] = (_timepoint(spec.get('timepoint_constraints'))
        or ('' if is_control else _timepoint(raw_timepoint))
        or _timepoint(target.get('timepoint_policy'))
        or 'Preserve the source-reported timepoint; select only a target-compatible observation window.')
    rules = selection_rules(spec.get('selection_rules'))
    rules.append('Timepoint eligibility: ' + spec['timepoint_constraints'])
    if description(target.get('effect_selection_rule')):
        rules.append('Source selection: ' + description(target['effect_selection_rule']))
    rules.extend(_terms(target.get('source_constraints')))
    # Extraction-only instructions such as "retain every model" contradict a
    # prespecified synthesis contrast. Keep the existing inheritance for all
    # other requests; this analysis-only opt-in keeps its own explicit rules.
    if 'Analysis eligibility: categorical_highest_vs_lowest' not in rules:
        rules.extend(_terms(target.get('extraction_notes')))
    spec['selection_rules'] = _unique(rules)
    spec.setdefault('model_policy', 'Prefer a source-reported model compatible with the declared estimand; retain its adjustment and exposure units.')
    families = spec.get('eligible_data_families') or target.get('accepted_data_families') or []
    spec['eligible_data_families'] = _unique(FAMILY_ALIASES.get(f, f) for f in families)
    spec.setdefault('accepted_effect_types', deepcopy(target.get('preferred_effect_measures', [])))
    spec.setdefault('estimands', deepcopy(target.get('estimands', [])))
    pooling = deepcopy(spec.get('pooling') or {})
    minimum = pooling.get('minimum_studies', pooling.get('minimum_independent_studies', spec.get('minimum_independent_studies', 2)))
    if type(minimum) is not int or minimum < 2:
        raise InputContractError('pooling.minimum_studies must be an integer >= 2')
    if 'minimum_studies' in pooling and 'minimum_independent_studies' in pooling and pooling['minimum_studies'] != pooling['minimum_independent_studies']:
        raise InputContractError('Conflicting pooling minimum study counts')
    pooling['minimum_studies'] = minimum
    # Keep the declared estimator/model; downstream public method-neutral
    # projections must not alter the canonical request.
    spec['pooling'] = pooling
    return spec


def normalize_request(request: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with explicit analysis semantics and a Typed extractor route.

    Call before the existing trace_runtime path/credential boundary validation.
    This function intentionally does not broaden or narrow a declared target.
    """
    result = deepcopy(request)
    protocol = result.get('protocol')
    if not isinstance(protocol, dict):
        raise InputContractError('protocol must be an object')
    try:
        protocol['statistical_method_policy'] = protocol_method_policy(protocol)
    except ValueError as exc:
        raise InputContractError(str(exc)) from exc
    targets = protocol.get('targets')
    if not isinstance(targets, list) or not targets:
        raise InputContractError('protocol.targets must be a nonempty array')
    pico = protocol.setdefault('pico', {})
    if not description(pico.get('outcome')):
        pico['outcome'] = description(pico.get('outcomes')) or '; '.join(_unique(description(t.get('outcome')) for t in targets))
    by_id = {}
    for target in targets:
        tid = target.get('target_id')
        if not tid or tid in by_id:
            raise InputContractError('Every extraction target must have a unique target_id')
        target['accepted_data_families'] = _unique(FAMILY_ALIASES.get(f, f) for f in target.get('accepted_data_families', []))
        by_id[tid] = target
    profile = result.get('analysis_profile')
    if not isinstance(profile, dict):
        raise InputContractError('analysis_profile must be an object')
    if not profile.get('primary_analysis_spec') and isinstance(profile.get('analyses'), list) and profile['analyses']:
        analyses = profile.pop('analyses')
        profile['primary_analysis_spec'] = analyses[0]
        profile['secondary_outcomes'] = analyses[1:]
    profile.setdefault('profile_version', '2.0.0')
    primary = profile.get('primary_analysis_spec')
    if not isinstance(primary, dict):
        raise InputContractError('analysis_profile.primary_analysis_spec is required')
    output = []
    for raw in [primary, *profile.get('secondary_outcomes', [])]:
        if not isinstance(raw, dict):
            raise InputContractError('Each analysis must be an object')
        # Keep inherited execution settings, but never borrow a different
        # target's PICO terms or selection rules for a secondary outcome.
        effective = deepcopy(primary)
        if raw is not primary and raw.get('target_id', primary.get('target_id')) != primary.get('target_id'):
            for key in ('outcome_group', 'outcome_terms', 'eligible_outcome_codes', 'exposure', 'comparison', 'population', 'timepoint_policy', 'timepoint_constraints', 'selection_rules', 'eligible_data_families', 'accepted_effect_types', 'estimands'):
                effective.pop(key, None)
        if (raw is not primary and 'outcome_terms' not in raw
                and raw.get('outcome_group', primary.get('outcome_group')) != primary.get('outcome_group')):
            effective.pop('outcome_terms', None)
        effective.update(raw)
        tid = effective.get('target_id')
        if tid not in by_id:
            raise InputContractError(f'Unknown analysis target_id: {tid}')
        output.append(_compile_spec(effective, by_id[tid], protocol.get('pico') or {}))
    profile['primary_analysis_spec'], profile['secondary_outcomes'] = output[0], output[1:]
    for key in ('candidate_subgroup_dimensions', 'candidate_sensitivity_dimensions'):
        for dimension in profile.get(key, []):
            for level in dimension.get('levels', []):
                overrides = level.get('analysis_overrides') or {}
                if 'selection_rules' in overrides:
                    overrides['selection_rules'] = selection_rules(overrides['selection_rules'])
                if 'timepoint_policy' in overrides and overrides['timepoint_policy'] not in ('longest_eligible', 'all_eligible', 'source_defined'):
                    overrides['timepoint_constraints'] = _timepoint(overrides['timepoint_policy'])
                    overrides['timepoint_policy'] = 'source_defined'
                    overrides['selection_rules'] = _unique([*selection_rules(overrides.get('selection_rules') or output[0]['selection_rules']), 'Timepoint eligibility: ' + overrides['timepoint_constraints']])
    result.setdefault('options', {})['extraction_runtime'] = 'typed'
    validate_request(result)
    return result


def validate_request(request: dict[str, Any]) -> None:
    """Reject unresolved semantics or unsupported quantitative routes early."""
    try:
        protocol_method_policy(request['protocol'])
    except ValueError as exc:
        raise InputContractError(str(exc)) from exc
    targets = {target['target_id']: target for target in request['protocol']['targets']}
    profile = request['analysis_profile']
    ids = set()
    for spec in [profile['primary_analysis_spec'], *profile.get('secondary_outcomes', [])]:
        aid = spec.get('analysis_id')
        if not aid or aid in ids:
            raise InputContractError('Every analysis must have a unique analysis_id')
        ids.add(aid)
        tid = spec.get('target_id')
        if tid not in targets:
            raise InputContractError(f'{aid}: unknown target_id {tid}')
        families = spec.get('eligible_data_families', [])
        if len(families) != 1 or families[0] not in SUPPORTED_ANALYSIS_FAMILIES:
            raise InputContractError(f'{aid}: declare exactly one supported quantitative family; received {families}. Source-only families may remain in extraction targets.')
        if not set(families) <= set(targets[tid]['accepted_data_families']):
            raise InputContractError(f'{aid}: analysis family is not captured by its extraction target')
        if not spec.get('accepted_effect_types'):
            raise InputContractError(f'{aid}: accepted_effect_types is required')
        if not _terms(spec.get('outcome_terms')) or spec.get('timepoint_policy') not in ('longest_eligible', 'all_eligible', 'source_defined'):
            raise InputContractError(f'{aid}: outcome and timepoint semantics are required')
        if not isinstance(spec.get('selection_rules'), list):
            raise InputContractError(f'{aid}: selection_rules must compile to a string array')
