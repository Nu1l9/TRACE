"""Prespecified statistical synthesis policy applied prior to model fitting."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
import math

from .method_selector import MethodSelectionDecision
from .pairwise_tools import MetaforPairwiseMetaTool

POLICY_PATH = Path(__file__).resolve().parents[1] / 'statistical_policy.json'
POLICY_BYTES = POLICY_PATH.read_bytes()
POLICY = json.loads(POLICY_BYTES)
POLICY_HASH = sha256(POLICY_BYTES).hexdigest()
COMPARATIVE_TYPES = {'RR', 'OR', 'HR', 'generic_ratio', 'MD', 'SMD', 'RD'}


def _save(folder, analysis_id, value):
    if folder is not None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / (sha256(analysis_id.encode()).hexdigest()[:24] + '.json')).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def validate_effects(effects):
    if len(effects) < 2:
        raise ValueError('Fixed policy requires at least two effects.')
    if len({e.evidence_id for e in effects}) != len(effects):
        raise ValueError('Duplicate evidence IDs cannot be synthesized.')
    if len({e.effect_scale for e in effects}) != 1:
        raise ValueError('Mixed analysis scales are unsupported.')
    if any(not math.isfinite(e.log_effect) or not math.isfinite(e.standard_error)
           or e.standard_error <= 0 for e in effects):
        raise ValueError('Effects must be finite with positive finite standard errors.')


def validate_interval(statistics, effects):
    values = (statistics.ci_lower, statistics.pooled_effect, statistics.ci_upper)
    if not all(math.isfinite(x) for x in values):
        raise ValueError('non_finite_interval')
    if statistics.ci_lower >= statistics.ci_upper:
        raise ValueError('non_positive_interval_width')
    if not statistics.ci_lower <= statistics.pooled_effect <= statistics.ci_upper:
        raise ValueError('estimate_outside_interval')
    if effects[0].effect_scale == 'log_ratio' and statistics.ci_lower <= 0:
        raise ValueError('non_positive_ratio_interval')
    if statistics.k != len(effects) or statistics.member_evidence_ids != [e.evidence_id for e in effects]:
        raise ValueError('calculator_membership_mismatch')


def interval_class(statistics, effects):
    null = 1 if effects[0].effect_scale == 'log_ratio' else 0
    return ('above_null' if statistics.ci_lower > null else
            'below_null' if statistics.ci_upper < null else 'includes_null')


class FixedMethodSelector:
    mode = 'deterministic'
    calls_made = 0

    def __init__(self, *, artifact_dir=None):
        self.artifact_dir = artifact_dir
        # Fail closed if the executable rule and the frozen declaration diverge.
        expected = ('random_effects', 'reml', 'normal_approximation', .95)
        actual = tuple(POLICY[k] for k in ['default_model', 'heterogeneity_estimator', 'primary_ci', 'confidence_level'])
        if actual != expected:
            raise ValueError('Policy JSON differs from the implemented fixed statistical policy contract.')

    def select(self, analysis, effects):
        validate_effects(effects)
        if '__clustered' in analysis.pooling_scale or getattr(analysis, 'dependence_policy', '') == 'clustered_effects_rho_0_6':
            raise ValueError('Declared clustered analysis must use its dedicated calculator, not this selector.')
        model = 'fixed_effect' if analysis.pooling_model == 'fixed_effect' else 'random_effects'
        decision = MethodSelectionDecision(
            pooling_model=model,
            heterogeneity_estimator='not_applicable' if model == 'fixed_effect' else 'reml',
            confidence_interval='normal_approximation', selector_mode='deterministic',
            rationale=(f"Fixed statistical policy ({POLICY['policy_id']}): "
                       + ('honor the explicit public fixed-effect model.' if model == 'fixed_effect'
                          else 'apply random-effects REML with a 95% normal interval.')
                      ),
        )
        _save(self.artifact_dir, analysis.analysis_id, {
            'policy_id': POLICY['policy_id'], 'policy_sha256': POLICY_HASH,
            'analysis_id': analysis.analysis_id, 'public_pooling_model': analysis.pooling_model,
            'decision': decision.model_dump(), 'gold_accessed': False,
            'trial_pooling_used_for_selection': False, 'method_selector_llm_calls': 0,
        })
        return decision

    def select_with_context(self, analysis, effects, **context):
        # Narrative source text cannot change a frozen statistical contract.
        return self.select(analysis, effects)


class FixedIntervalCalculator(MetaforPairwiseMetaTool):
    """Validate the primary; record HK separately with no promotion or silent repair."""
    def __init__(self, *, artifact_dir=None, **kwargs):
        super().__init__(**kwargs)
        self.artifact_dir = artifact_dir
        self.audits = {}

    def invoke(self, profile, effects):
        validate_effects(effects)
        clustered = '__clustered' in profile.pooling_scale
        if profile.pooling_model not in {'fixed_effect', 'random_effects'}:
            raise ValueError('Fixed statistical policy calculator requires a resolved pooling model.')
        if not clustered:
            if '__hartung_knapp_t' in profile.pooling_scale or '__heterogeneity_switch' in profile.pooling_scale:
                raise ValueError('Fixed statistical primary policy does not allow HK or a data-dependent model switch.')
            if profile.pooling_model == 'random_effects' and '__reml' not in profile.pooling_scale:
                raise ValueError('Fixed statistical random-effects primary must explicitly use REML.')
        counts = Counter(e.study_id for e in effects)
        audit = {'policy_id': POLICY['policy_id'], 'policy_sha256': POLICY_HASH,
                 'analysis_id': profile.analysis_id, 'primary_profile': profile.model_dump(),
                 'effect_count': len(effects), 'distinct_study_ids': len(counts),
                 'repeated_study_ids': {k:v for k,v in counts.items() if v>1},
                 'small_k_caution': len(counts) <= 3, 'gold_accessed': False,
                 'dependence_note': 'Study-level dependence is evaluated according to analysis_profile rules.'}
        try:
            primary = super().invoke(profile, effects)
            validate_interval(primary, effects)
            audit.update(primary_status='completed', primary=primary.model_dump())
            comparative = (profile.effect_type in COMPARATIVE_TYPES
                           and all(e.effect_type in COMPARATIVE_TYPES for e in effects))
            if profile.pooling_model != 'random_effects' or clustered or not comparative:
                audit.update(hk_status='not_applicable', interval_method_sensitive=None)
                return primary
            audit['primary_class'] = interval_class(primary, effects)
            hk_profile = profile.model_copy(update={'pooling_scale': profile.pooling_scale + '__hartung_knapp_t'})
            try:
                hk = super().invoke(hk_profile, effects)
                audit['hk_raw'] = hk.model_dump()
                validate_interval(hk, effects)
                audit.update(hk_status='completed', hk=hk.model_dump(), hk_class=interval_class(hk, effects),
                             interval_method_sensitive=interval_class(hk, effects) != audit['primary_class'])
            except (ValueError, RuntimeError, TypeError, OverflowError) as exc:
                audit.update(hk_status='failed', hk_error=str(exc), interval_method_sensitive=None)
            return primary
        except (TypeError, OverflowError) as exc:
            audit.update(primary_status='failed', primary_error=str(exc))
            raise ValueError(f'Invalid primary numeric output: {exc}') from exc
        except (ValueError, RuntimeError) as exc:
            audit.update(primary_status='failed', primary_error=str(exc))
            raise
        finally:
            self.audits[profile.analysis_id] = audit
            _save(self.artifact_dir, profile.analysis_id, audit)
