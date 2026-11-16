"""External, callback-driven planning and bounded query feedback for frozen baseline.

No model SDK, credentials, network calls, or upstream module changes are needed.
The callback is an application-owned JSON provider; all its output is untrusted.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import re
from typing import Callable

from .common import ContractError, digest, require
from metapipeline.contracts import SUPPORTED, VERSION, identity, no_labels, normalize_request, text

JsonCallback = Callable[[dict], dict]
PICO = ('population', 'intervention_or_exposure', 'comparison', 'outcome')
SECRET_KEY = re.compile(r'api[_-]?key|secret|password|authorization|access[_-]?token|refresh[_-]?token', re.I)
SOURCE_KEYS = {'documents', 'article_id', 'document_id', 'study_id', 'candidate_id',
               'raw_data', 'source_location', 'source_evidence', 'effect', 'effect_size',
               'p_value', 'standard_error', 'selected', 'selection_status', 'selected_status'}
PROTOCOL_FIELDS = {'pico', 'targets', 'inclusion_criteria', 'exclusion_criteria',
                   'study_designs', 'followup_window'}
TARGET_FIELDS = {'outcome', 'accepted_data_families', 'preferred_effect_measures',
                 'estimands', 'description', 'measurement_semantics'}
PROFILE_FIELDS = {'primary_analysis_spec', 'secondary_outcomes',
                  'candidate_subgroup_dimensions', 'candidate_sensitivity_dimensions'}
SPEC_FIELDS = {'title', 'target_index', 'outcome_group', 'outcome_terms',
               'eligible_data_families', 'accepted_effect_types', 'estimands',
               'exposure', 'comparison', 'population', 'timepoint_policy',
               'model_policy', 'dependence_policy', 'study_design', 'study_designs',
               'allowed_model_types', 'genotype_model', 'rsid'}


def _safe_json(value):
    """Reject secret-bearing input before it can reach a provider or an artifact."""
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ContractError('Brain input and output must be finite JSON') from exc
    require(len(encoded) <= 2_000_000, 'Brain JSON exceeds the 2 MB boundary')
    def visit(item):
        if isinstance(item, dict):
            for key, child in item.items():
                require(isinstance(key, str), 'Brain JSON keys must be strings')
                require(not SECRET_KEY.search(key), 'Credentials must not enter Brain JSON; use provider environment configuration')
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
        elif isinstance(item, str):
            require(not re.search(r'\bsk-[A-Za-z0-9_-]{20,}', item), 'Credential-like value must not enter Brain JSON')
    visit(value)


def _leaves(value, path=''):
    if isinstance(value, dict) and value:
        for key, child in value.items():
            yield from _leaves(child, f'{path}.{key}' if path else key)
    elif isinstance(value, list) and value:
        for i, child in enumerate(value):
            yield from _leaves(child, f'{path}[{i}]')
    else:
        yield path, value


def _overlay(suggestions, known):
    """Known scalar values and entire arrays win; dictionaries allow missing keys."""
    if not isinstance(suggestions, dict) or not isinstance(known, dict):
        return deepcopy(known)
    merged = deepcopy(suggestions)
    for key, value in known.items():
        merged[key] = _overlay(merged[key], value) if key in merged else deepcopy(value)
    return merged


def _validate_known(known):
    require(isinstance(known, dict), 'known_fields must be a pipeline-shaped object')
    _safe_json(known)
    require(known.get('schema_version', VERSION) == VERSION, f'known_fields schema_version must be {VERSION}')
    for key in ('protocol', 'analysis_profile', 'options'):
        require(key not in known or isinstance(known[key], dict), f'known_fields.{key} must be an object')
    require('documents' not in known or isinstance(known['documents'], list), 'known_fields.documents must be an array')
    options = known.get('options', {})
    require(options.get('provider', 'official_openai') in ('official_openai', 'openai_compatible'), 'unknown provider')
    if 'extraction_runtime' in options:
        require(options['extraction_runtime'] in ('baseline', 'typed'), 'extraction_runtime must be baseline or typed')
    for key in ('protocol', 'analysis_profile'):
        no_labels(known.get(key, {}), key)


def _validate_completion(value):
    require(isinstance(value, dict), 'Brain callback must return an object')
    _safe_json(value)
    require(value.get('schema_version') == 'brain_completion/2.0', 'Brain callback must return brain_completion/2.0')
    require(set(value) <= {'schema_version', 'fields', 'uncertainties', 'field_reasons'}, 'Unexpected Brain completion fields')
    fields = value.get('fields')
    require(isinstance(fields, dict) and set(fields) <= {'protocol', 'analysis_profile'}, 'Brain may propose only protocol and analysis_profile fields')
    require(isinstance(value.get('uncertainties'), list) and all(isinstance(x, str) for x in value['uncertainties']), 'uncertainties must be a string array')
    reasons = value.get('field_reasons', {})
    require(isinstance(reasons, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in reasons.items()), 'field_reasons must map paths to reasons')
    def visit(item):
        if isinstance(item, dict):
            for key, child in item.items():
                require(key not in SOURCE_KEYS and not key.endswith('_id'), 'Brain cannot invent source data or identifiers')
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
    visit(fields)
    no_labels(fields, 'brain.fields')
    if 'protocol' in fields:
        p = fields['protocol']
        require(isinstance(p, dict) and set(p) <= PROTOCOL_FIELDS, 'Unsupported proposed protocol fields')
        if 'pico' in p:
            require(isinstance(p['pico'], dict) and set(p['pico']) <= set(PICO), 'Use canonical PICO field names')
            for field, item in p['pico'].items():
                require(isinstance(item, str) or (isinstance(item, dict) and set(item) <= {'description', 'terms'}), f'Invalid proposed PICO {field}')
        if 'targets' in p:
            require(isinstance(p['targets'], list), 'Proposed targets must be an array')
            for target in p['targets']:
                require(isinstance(target, dict) and set(target) <= TARGET_FIELDS, 'Unsupported proposed target fields')
    if 'analysis_profile' in fields:
        profile = fields['analysis_profile']
        require(isinstance(profile, dict) and set(profile) <= PROFILE_FIELDS, 'Unsupported proposed analysis profile fields')
        require(isinstance(profile.get('secondary_outcomes', []), list), 'secondary_outcomes must be an array')
        specs = ([profile['primary_analysis_spec']] if 'primary_analysis_spec' in profile else []) + profile.get('secondary_outcomes', [])
        for spec in specs:
            require(isinstance(spec, dict) and set(spec) <= SPEC_FIELDS, 'Unsupported proposed analysis fields')
            require('target_index' not in spec or (type(spec['target_index']) is int and spec['target_index'] >= 0), 'target_index must be a nonnegative integer')
    return deepcopy(value)


def _missing(request):
    missing = []
    p = request.get('protocol', {})
    pico = p.get('pico', {})
    require(isinstance(pico, dict), 'protocol.pico must be an object')
    for key in PICO:
        if not text(pico.get(key)):
            missing.append('protocol.pico.' + key)
    targets = p.get('targets', [])
    require(isinstance(targets, list), 'protocol.targets must be an array')
    if not targets:
        missing.append('protocol.targets')
    for i, target in enumerate(targets):
        require(isinstance(target, dict), 'Each target must be an object')
        for key in ('outcome', 'accepted_data_families'):
            if not target.get(key):
                missing.append(f'protocol.targets[{i}].{key}')
        if 'preferred_effect_measures' not in target:
            missing.append(f'protocol.targets[{i}].preferred_effect_measures')
    profile = request.get('analysis_profile', {})
    primary = profile.get('primary_analysis_spec')
    if not isinstance(primary, dict) or not primary:
        missing.append('analysis_profile.primary_analysis_spec')
    else:
        for key in ('target_id', 'outcome_group', 'eligible_data_families', 'accepted_effect_types'):
            if not primary.get(key):
                missing.append('analysis_profile.primary_analysis_spec.' + key)
    if not request.get('documents'):
        missing.append('documents')
    return missing


def fill_topic(topic: str, known_fields: dict | None = None, callback: JsonCallback | None = None,
               *, base: str | Path = '.', check_paths: bool = True) -> dict:
    """Propose fields; never execute a pipeline or overwrite an explicitly supplied field.

    Ready means the frozen validator passed and there are no missing fields or
    reported uncertainties. Model provenance remains visible, but does not itself
    require an approval. This function creates an artifact and never runs a study.
    """
    require(isinstance(topic, str) and bool(topic.strip()), 'topic must be nonempty text')
    _safe_json(topic)
    known = deepcopy(known_fields if known_fields is not None else {})
    _validate_known(known)
    require('topic' not in known or known['topic'] == topic, 'topic conflicts with known_fields.topic')
    completion = {'fields': {}, 'uncertainties': [], 'field_reasons': {}}
    if callback is not None:
        require(callable(callback), 'callback must be callable')
        # Source paths, document identity and runtime configuration are not provider input.
        context = {key: deepcopy(known[key]) for key in ('protocol', 'analysis_profile') if key in known}
        context = _without_identifiers(context)
        completion = _validate_completion(callback({
            'schema_version': 'brain_fill_request/2.0', 'topic': topic, 'known_fields': context,
            'response_schema': 'brain_completion/2.0',
            'response_example': {'schema_version': 'brain_completion/2.0', 'fields': {},
                                 'uncertainties': [], 'field_reasons': {}},
            'constraints': ['Preserve the user research scope. Leave unspecified scientific choices missing or uncertain.',
                            'Return only protocol and analysis_profile additions. Do not invent documents, IDs, source evidence, or numerical results.',
                            'Use target_index (zero based) to link proposed analyses to proposed extraction targets.',
                            'Report unresolved scientific choices in uncertainties; do not hide uncertainty behind a filled value.'],
            'field_contract': {
                'protocol_fields': sorted(PROTOCOL_FIELDS), 'pico_fields': list(PICO),
                'target_fields': sorted(TARGET_FIELDS), 'analysis_profile_fields': sorted(PROFILE_FIELDS),
                'analysis_spec_fields': sorted(SPEC_FIELDS), 'supported_data_families': sorted(SUPPORTED),
                'target_shape': {'outcome': {'description': 'outcome stated by the user'},
                                 'accepted_data_families': ['one supported family'],
                                 'preferred_effect_measures': ['compatible effect type'], 'estimands': []},
                'analysis_rule': 'Exactly one supported eligible_data_families value per analysis; compatible extraction target and accepted_effect_types are required.',
            },
        }))
    request = _overlay(completion['fields'], known)
    request.setdefault('schema_version', VERSION)
    request.setdefault('topic', topic)
    seed = digest({'topic': topic, 'known_fields': known})[:12]
    request.setdefault('case_id', 'brain_' + seed)
    identity(request['case_id'], 'case_id')
    protocol = request.setdefault('protocol', {})
    protocol.setdefault('case_id', request['case_id'])
    protocol.setdefault('protocol_id', 'protocol_' + digest({'case_id': request['case_id']})[:12])
    require(protocol.get('case_id') == request['case_id'], 'protocol.case_id does not match request')
    identity(protocol['protocol_id'], 'protocol.protocol_id')
    targets = protocol.get('targets', [])
    require(isinstance(targets, list), 'protocol.targets must be an array')
    for i, target in enumerate(targets):
        require(isinstance(target, dict), 'Each target must be an object')
        target.setdefault('target_id', 'target_' + digest({'case_id': request['case_id'], 'index': i})[:12])
        identity(target['target_id'], 'target_id')
    profile = request.setdefault('analysis_profile', {})
    profile.setdefault('profile_version', '2.0.0')
    profile.setdefault('plan_id', 'plan_' + digest({'case_id': request['case_id']})[:12])
    secondary = profile.get('secondary_outcomes', [])
    require(isinstance(secondary, list), 'secondary_outcomes must be an array')
    specs = ([profile['primary_analysis_spec']] if 'primary_analysis_spec' in profile else []) + secondary
    for i, spec in enumerate(specs):
        require(isinstance(spec, dict), 'Each analysis spec must be an object')
        spec.setdefault('analysis_id', 'analysis_' + digest({'case_id': request['case_id'], 'index': i})[:12])
        identity(spec['analysis_id'], 'analysis_id')
        spec.setdefault('pooled_result_id', spec['analysis_id'])
        index = spec.get('target_index', 0 if i == 0 else i)
        require(type(index) is int and index >= 0, 'target_index must be a nonnegative integer')
        if 'target_id' not in spec and index < len(targets):
            spec['target_id'] = targets[index]['target_id']
        # target_index is intentionally retained as harmless planning metadata;
        # this also preserves any explicitly supplied user field exactly.
    request.setdefault('documents', [])
    _safe_json(request)
    user_paths = dict(_leaves(known))
    proposed_paths = dict(_leaves(completion['fields']))
    provenance = {}
    pending = []
    for path, value in _leaves(request):
        # A supplied empty dictionary contains no user values; new children still
        # belong to the model/program, while an explicitly supplied empty array wins.
        is_user = path in user_paths
        if is_user:
            source, reason, confirm = 'user', 'Explicitly supplied field preserved', False
        elif path == 'topic':
            source, reason, confirm = 'user', 'User topic', False
        elif path in proposed_paths:
            source, reason, confirm = 'model_suggestion', completion.get('field_reasons', {}).get(path, 'Proposed from topic'), False
            pending.append(path)
        else:
            source, reason, confirm = 'program', 'Deterministic identifier or structural default', False
        provenance[path] = {'source': source, 'reason': reason, 'requires_confirmation': confirm}
    missing = _missing(request)
    errors = []
    try:
        normalize_request(request, base=Path(base), check_paths=check_paths)
    except (ContractError, TypeError, AttributeError, KeyError) as exc:
        errors.append(str(exc))
    uncertainties = list(dict.fromkeys(completion['uncertainties']))
    ready = not missing and not errors and not uncertainties
    return {'schema_version': 'brain_proposal/2.0', 'status': 'ready' if ready else 'draft',
            'pipeline_request': request, 'missing_fields': missing, 'uncertainties': uncertainties,
            'field_provenance': provenance, 'suggested_fields': pending,
            'validation': {'valid': not errors, 'validator': 'frozen_meta_pipeline/1.0',
                           'paths_checked': check_paths, 'errors': errors},
            'proposal_fingerprint': digest(request)}


def _without_identifiers(value):
    if isinstance(value, dict):
        return {key: _without_identifiers(child) for key, child in value.items()
                if key not in SOURCE_KEYS and not key.endswith('_id')}
    if isinstance(value, list):
        return [_without_identifiers(child) for child in value]
    return deepcopy(value)


FEEDBACK_COUNTS = ('source_count', 'before_candidate_count', 'after_candidate_count',
                   'before_study_count', 'after_study_count', 'added_candidate_count',
                   'source_contradiction_count')
GRAPH_FIELDS = ('outcome', 'exposure', 'comparison', 'population')


def sanitize_graph_feedback(feedback: dict) -> dict:
    """Allowlist metadata: effect values and selection decisions never reach Brain."""
    require(isinstance(feedback, dict) and feedback.get('schema_version') == 'brain_graph_feedback/1.0', 'Expected brain_graph_feedback/1.0')
    _safe_json(feedback)
    clean = {'schema_version': feedback['schema_version'], 'topics': [],
             'stop_scope': 'retrieval_only_no_effect_values'}
    for key in ('base_protocol_fingerprint', 'base_ledger_fingerprint'):
        require(isinstance(feedback.get(key), str) and re.fullmatch(r'[0-9a-f]{64}', feedback[key]), f'Invalid {key}')
        clean[key] = feedback[key]
    require(isinstance(feedback.get('topics'), list), 'Graph topics must be an array')
    for topic in feedback['topics']:
        require(isinstance(topic, dict), 'Graph feedback topic must be an object')
        row = {'analysis_id': identity(topic.get('analysis_id'), 'feedback.analysis_id'), 'fields': {}}
        for key in FEEDBACK_COUNTS:
            if key in topic:
                require(type(topic[key]) is int and topic[key] >= 0, f'Invalid feedback count {key}')
                row[key] = topic[key]
        fields = topic.get('fields', {})
        require(isinstance(fields, dict), 'Graph feedback fields must be an object')
        for field in GRAPH_FIELDS:
            if field in fields:
                require(isinstance(fields[field], dict), 'Graph field counts must be an object')
                row['fields'][field] = {}
                for key in ('known_count', 'unknown_count', 'alias_match_count'):
                    if key in fields[field]:
                        count = fields[field][key]
                        require(type(count) is int and count >= 0, f'Invalid graph field count {key}')
                        row['fields'][field][key] = count
        clean['topics'].append(row)
    return clean


def suggest_graph_patch(request: dict, graph_feedback: dict, callback: JsonCallback | None) -> dict | None:
    """Ask for additive query aliases only, then use the graph extension validator."""
    _validate_known(request)
    feedback = sanitize_graph_feedback(graph_feedback)
    require(digest(request.get('protocol', {})) == feedback['base_protocol_fingerprint'], 'Graph feedback belongs to a different protocol')
    if callback is None:
        return None
    require(callable(callback), 'callback must be callable')
    from .graph_feedback import validate_graph_patch
    profile = request.get('analysis_profile', {})
    specs = [profile.get('primary_analysis_spec', {}), *profile.get('secondary_outcomes', [])]
    queries = []
    for spec in specs:
        if spec.get('analysis_id'):
            queries.append({key: deepcopy(spec[key]) for key in ('analysis_id', 'outcome_group', 'outcome_terms', 'exposure', 'comparison', 'population') if key in spec})
    result = callback({'schema_version': 'brain_graph_request/1.0', 'queries': queries,
                       'feedback': feedback, 'response_schema': 'brain_graph_patch/1.0',
                       'response_example': {'schema_version': 'brain_graph_patch/1.0',
                           'analysis_id': queries[0]['analysis_id'] if queries else '',
                           'base_protocol_fingerprint': feedback['base_protocol_fingerprint'],
                           'base_ledger_fingerprint': feedback['base_ledger_fingerprint'],
                           'operations': []},
                       'operation_shape': {'operation': 'add_query_aliases',
                           'field': 'outcome|exposure|comparison|population',
                           'terms': ['an equivalent expression for this query concept'],
                           'basis': 'why these expressions preserve the research scope', 'scope_change': False},
                       'constraints': ['Add retrieval aliases only; do not alter scientific eligibility or evidence attributes.',
                                       'Stop when no useful equivalent expression remains; return operations: [].']})
    _safe_json(result)
    if isinstance(result, dict) and result.get('operations') == []:
        require(set(result) <= {'schema_version', 'analysis_id', 'base_protocol_fingerprint', 'base_ledger_fingerprint', 'operations', 'patch_id'}, 'Unexpected no-op graph patch fields')
        require(result.get('schema_version') == 'brain_graph_patch/1.0', 'Expected brain_graph_patch/1.0')
        require(result.get('analysis_id') in [row['analysis_id'] for row in queries], 'Unknown analysis_id in no-op graph patch')
        for key in ('base_protocol_fingerprint', 'base_ledger_fingerprint'):
            require(result.get(key) == feedback[key], 'Stale no-op graph patch')
        return None
    return validate_graph_patch(result, protocol_fingerprint=feedback['base_protocol_fingerprint'],
                                ledger_fingerprint=feedback['base_ledger_fingerprint'],
                                analysis_ids=[row['analysis_id'] for row in queries])


def run_graph_feedback_loop(request: dict, initial_feedback: dict, callback: JsonCallback | None,
                            executor: Callable[[dict], dict], *, max_rounds: int = 2) -> dict:
    """Run at most two external patch/executor cycles; stop on no gain or repeats.

    executor accepts one validated patch and returns fresh graph feedback. It owns
    filesystem writes and Analysis execution; this controller never changes evidence.
    """
    require(type(max_rounds) is int and 0 <= max_rounds <= 2, 'max_rounds must be an integer from 0 to 2')
    feedback = sanitize_graph_feedback(initial_feedback)
    history, patches, seen = [feedback], [], set()
    reason = 'round_limit'
    for _ in range(max_rounds):
        patch = suggest_graph_patch(request, feedback, callback)
        if patch is None or not patch.get('operations'):
            reason = 'no_patch'
            break
        # Different justifications or operation order do not make the same aliases new.
        aliases = sorted({(operation['field'], ' '.join(term.casefold().split()))
                          for operation in patch['operations'] for term in operation['terms']})
        fingerprint = digest({'analysis_id': patch['analysis_id'], 'aliases': aliases})
        if fingerprint in seen:
            reason = 'duplicate_patch'
            break
        seen.add(fingerprint)
        fresh = sanitize_graph_feedback(executor(deepcopy(patch)))
        require(fresh['base_protocol_fingerprint'] == feedback['base_protocol_fingerprint'], 'Executor changed the scientific protocol')
        require(fresh['base_ledger_fingerprint'] == feedback['base_ledger_fingerprint'], 'Executor changed source evidence during retrieval expansion')
        topic = next((row for row in fresh['topics'] if row['analysis_id'] == patch['analysis_id']), None)
        require(topic is not None and 'added_candidate_count' in topic, 'Executor must report added_candidate_count for the patched analysis')
        patches.append(patch)
        history.append(fresh)
        feedback = fresh
        if topic['added_candidate_count'] == 0:
            reason = 'no_new_candidates'
            break
    return {'schema_version': 'brain_graph_loop/1.0', 'patches': patches,
            'feedback_history': history, 'stop_reason': reason, 'rounds': len(patches)}
