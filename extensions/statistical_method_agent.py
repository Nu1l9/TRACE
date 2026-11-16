"""Release adapter: fixed statistical defaults or explicit source-aware selection."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .common import write
from .statistical_methods import validate_method_policy
from analysis_engine.fixed_policy import validate_effects, validate_interval
from analysis_engine.fixed_policy_agent import FixedPolicyAnalysisAgent
from analysis_engine.source_aware_method_selector import ToolAwareMethodSelector
from analysis_engine.pairwise_tools import MetaforPairwiseMetaTool


class ValidatedSelectedMethodCalculator(MetaforPairwiseMetaTool):
    """Validate the chosen primary without forcing REML or replacing its interval."""

    def __init__(self, *, artifact_dir, **kwargs):
        super().__init__(**kwargs)
        self.artifact_dir = Path(artifact_dir)
        self.audits = {}

    def invoke(self, profile, effects):
        from hashlib import sha256
        audit = {
            'policy_id': 'source_aware_model_selection',
            'mode': 'model_selected',
            'analysis_id': profile.analysis_id,
            'primary_profile': profile.model_dump(),
            'effect_count': len(effects),
            'distinct_study_ids': len(Counter(e.study_id for e in effects)),
            'hk_status': 'not_run_as_separate_sensitivity',
            'interval_method_sensitive': None,
        }
        try:
            validate_effects(effects)
            if profile.pooling_model not in {'fixed_effect', 'random_effects'}:
                raise ValueError('Method selection must resolve the pooling model before calculation.')
            if '__heterogeneity_switch' in profile.pooling_scale:
                raise ValueError('Automatic heterogeneity-based model switches are unsupported.')
            primary = super().invoke(profile, effects)
            validate_interval(primary, effects)
            audit.update(primary_status='completed', primary=primary.model_dump())
            return primary
        except (ValueError, RuntimeError, TypeError, OverflowError) as exc:
            audit.update(primary_status='failed', primary_error=str(exc))
            raise ValueError(f'Selected-method calculation failed: {exc}') from exc
        finally:
            self.audits[profile.analysis_id] = audit
            name = sha256(profile.analysis_id.encode()).hexdigest()[:24] + '.json'
            write(self.artifact_dir / name, audit)


class ReleaseAnalysisAgent(FixedPolicyAnalysisAgent):
    """Release extension over the fixed-policy analysis agent."""

    def __init__(self, *, statistical_method_policy='fixed', method_selector=None, **kwargs):
        self.statistical_method_policy = validate_method_policy(statistical_method_policy)
        if self.statistical_method_policy == 'fixed' and method_selector is not None:
            raise ValueError('fixed policy does not accept an injected method selector')
        if self.statistical_method_policy == 'model_selected' and not isinstance(
                method_selector, ToolAwareMethodSelector):
            raise ValueError('model_selected requires the source-aware method selector')
        super().__init__(**kwargs)
        if self.statistical_method_policy == 'model_selected':
            self.method_selector = method_selector
            self.calculator = ValidatedSelectedMethodCalculator(
                artifact_dir=Path(kwargs['audit_dir']) / 'calculation_audit')
