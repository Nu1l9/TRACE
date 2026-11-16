"""Public statistical-method policy, independent of providers and run results."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

DEFAULT_METHOD_POLICY = 'fixed'
METHOD_POLICIES = ('fixed', 'model_selected')


def validate_method_policy(value):
    if not isinstance(value, str) or value not in METHOD_POLICIES:
        raise ValueError('protocol.statistical_method_policy must be fixed or model_selected')
    return value


def protocol_method_policy(protocol):
    return validate_method_policy(protocol.get('statistical_method_policy', DEFAULT_METHOD_POLICY))


def assert_method_policy_output_compatible(output, mode):
    """Keep method modes separate even when legacy update hashes are bypassed."""
    validate_method_policy(mode)
    output = Path(output)
    marker = output / 'statistical_method_policy.json'
    if marker.is_file():
        previous = json.loads(marker.read_text())
        if not isinstance(previous, dict) or previous.get('mode') != mode:
            raise ValueError('Statistical method policy changed; use a new analysis output directory')
    elif mode != 'fixed' and any((output / name).exists() for name in (
            'analysis_result.json', 'analysis_inputs.lock.json', 'checkpoints')):
        raise ValueError('Legacy analysis cannot be reused in model_selected mode; use a new output directory')


def method_policy_manifest(mode):
    """Describe the selected interface without claiming a model decision occurred."""
    validate_method_policy(mode)
    if mode == 'fixed':
        from analysis_engine.fixed_policy import POLICY
        return deepcopy(POLICY)
    return {
        'policy_id': 'source_aware_model_selection',
        'version': 1,
        'mode': 'model_selected',
        'selector': 'ToolAwareMethodSelector',
        'allowed_models': ['fixed_effect', 'random_effects'],
        'allowed_random_effects_estimators': ['dersimonian_laird', 'reml', 'paule_mandel'],
        'allowed_random_effects_intervals': ['normal_approximation', 'hartung_knapp_t'],
        'fixed_effect_interval': 'normal_approximation',
        'confidence_level': 0.95,
        'declared_clustered_route': 'preserve existing dedicated tool',
        'failure_policy': 'block affected analysis; never fall back to fixed policy',
        'selection': 'source context and tool capabilities; no trial pooling or reference answers',
        'secondary_ci': 'not automatically calculated; selected interval is primary',
    }
