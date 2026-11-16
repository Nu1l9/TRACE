"""Regression: a valid tiny HK ratio interval must survive the R/JSON boundary."""
import math
import shutil
import unittest
from extensions.common import activate
activate()
from analysis_core.models import ResolvedAnalysisProfile, StudyEffect
from analysis_engine.fixed_policy import FixedIntervalCalculator


@unittest.skipUnless(shutil.which('Rscript'), 'Rscript is unavailable')
class StatisticalPrecisionTests(unittest.TestCase):
    def test_two_study_hk_lower_bound_survives_serialization(self):
        effects = []
        for i, (a, n1, c, n0) in enumerate([(1,49,5,49), (6,156,5,152)]):
            yi = math.log((a/n1)/(c/n0))
            se = math.sqrt(1/a-1/n1+1/c-1/n0)
            effects.append(StudyEffect(study_id=f's{i}', evidence_id=f'e{i}', effect_type='RR',
                value=math.exp(yi), log_effect=yi, standard_error=se,
                ci_lower=math.exp(yi-1.96*se), ci_upper=math.exp(yi+1.96*se), source_data={}))
        profile = ResolvedAnalysisProfile(analysis_id='precision', task_id='test', protocol_id='test',
            target_id='mortality', outcome='mortality', comparison='intervention vs control',
            data_family='binary_arm_counts', effect_type='RR', pooling_model='random_effects',
            pooling_scale='log_ratio_inverse_variance__reml')
        calculator = FixedIntervalCalculator()
        result = calculator.invoke(profile, effects)
        audit = calculator.audits['precision']
        self.assertEqual(audit['hk_status'], 'completed')
        hk = audit['hk']
        self.assertGreater(hk['ci_lower'], 0)
        self.assertLess(hk['ci_lower'], 0.00005)
        self.assertAlmostEqual(hk['ci_lower']/1.193722845605154e-5, 1, places=7)
        self.assertAlmostEqual(hk['ci_upper']/30912.98462026805, 1, places=7)
        self.assertAlmostEqual(result.pooled_effect, 0.607466344475599, places=10)
        self.assertAlmostEqual(result.ci_lower, 0.114160133208504, places=10)
        self.assertAlmostEqual(result.ci_upper, 3.232436309412598, places=10)


if __name__ == '__main__':
    unittest.main()
