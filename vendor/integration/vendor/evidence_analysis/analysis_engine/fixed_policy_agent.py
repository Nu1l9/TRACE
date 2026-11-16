"""Graph-based analysis with a fixed statistical policy."""
from pathlib import Path
from .graph_agent import GraphAnalysisAgent
from .fixed_policy import FixedMethodSelector, FixedIntervalCalculator


class FixedPolicyAnalysisAgent(GraphAnalysisAgent):
    def __init__(self, *, audit_dir, **kwargs):
        audit_dir = Path(audit_dir)
        if 'method_selector' in kwargs or 'calculator' in kwargs:
            raise ValueError('The fixed-policy agent installs its selector and guarded calculator.')
        super().__init__(audit_dir=audit_dir, method_selector=FixedMethodSelector(
            artifact_dir=audit_dir/'method_selection'), **kwargs)
        self.calculator = FixedIntervalCalculator(artifact_dir=audit_dir/'interval_sensitivity')

    def run(self, **kwargs):
        result = super().run(**kwargs)
        for topic in result.analyses:
            audit = self.calculator.audits.get(topic.analysis.analysis_id, {})
            topic.interval_sensitivity = audit
            if audit.get('interval_method_sensitive'):
                topic.warnings.append('Fixed policy: normal and HK intervals yield different null-crossing conclusions. The frozen normal primary remains unchanged.')
            if audit.get('small_k_caution'):
                topic.warnings.append('Fixed policy: only two or three distinct study IDs; the fixed policy does not remove small-sample uncertainty.')
            if audit.get('hk_status') == 'failed':
                topic.warnings.append('Fixed policy: HK sensitivity could not be validated: ' + audit.get('hk_error', 'unknown'))
        return result
