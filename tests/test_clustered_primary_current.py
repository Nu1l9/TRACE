"""Opt-in dependent-effect synthesis must preserve study clustering."""

import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "vendor/integration"),
    str(ROOT / "vendor/integration/vendor/evidence_analysis"),
]

from analysis_core.models import AnalysisDefinition, EvidenceRecord, ResolvedAnalysisProfile, StudyEffect
from analysis_engine.eligibility import is_categorical_high_low
from analysis_engine.pairwise_tools import MetaforPairwiseMetaTool
from analysis_engine.pipeline_agent import _dependent_study_ids


def record(exposure, comparison, unit=""):
    return EvidenceRecord(
        evidence_id="e", article_id="a", study_id="s", target_id="t",
        intervention_or_exposure=exposure, comparator=comparison,
        raw_data={"exposure_unit": unit},
    )


class ClusteredPrimaryTests(unittest.TestCase):
    def test_contrast_gate_rejects_continuous_and_lower_four_reference(self):
        self.assertTrue(is_categorical_high_low(record(
            "Upper quartile of weight variability", "Lowest quartile (Q1) reference")))
        self.assertTrue(is_categorical_high_low(record(
            "Weight variability", "Lowest quintile reference", "Highest quintile")))
        self.assertFalse(is_categorical_high_low(record(
            "Weight variability", "Per 1-SD increase; no categorical reference")))
        self.assertFalse(is_categorical_high_low(record(
            "Highest quintile", "Lower four quintiles (Q1–Q4) reference")))

    def test_repeated_studies_require_explicit_clustered_route(self):
        definition = AnalysisDefinition(
            analysis_id="primary", pooled_result_id="primary", title="Primary",
            kind="primary", target_id="t", outcome_group="o",
            eligible_data_families=["adjusted_ratio_effect"],
            accepted_effect_types=["HR", "RR"],
            dependence_policy="clustered_effects_rho_0_6",
            pooling_model="random_effects",
            pooling_scale="log_ratio_inverse_variance__clustered_rho_0_6",
        )
        effects = [StudyEffect(study_id="s", evidence_id=eid, effect_type="HR",
            value=1.2, log_effect=.182, standard_error=.1,
            ci_lower=1.0, ci_upper=1.5, source_data={}) for eid in ("e1", "e2")]
        self.assertEqual(_dependent_study_ids(definition, effects), [])
        with self.assertRaises(ValueError):
            _dependent_study_ids(definition.model_copy(update={
                "pooling_scale": "log_ratio_inverse_variance"}), effects)

    @unittest.skipUnless(shutil.which("Rscript"), "Rscript is unavailable")
    def test_clustered_calculator_handles_repeated_effects(self):
        profile = ResolvedAnalysisProfile(
            analysis_id="primary", task_id="task", protocol_id="protocol",
            target_id="t", outcome="o", comparison="high vs low",
            data_family="adjusted_ratio_effect", effect_type="generic_ratio",
            pooling_model="random_effects",
            pooling_scale="log_ratio_inverse_variance__clustered_rho_0_6",
            minimum_studies=2,
        )
        effects = [StudyEffect(
            study_id=study, evidence_id=f"e{index}", effect_type="HR",
            value=1.2, log_effect=yi, standard_error=se,
            ci_lower=1.0, ci_upper=1.6, source_data={},
        ) for index, (study, yi, se) in enumerate([
            ("s1", .1, .14), ("s1", .2, .17),
            ("s2", .3, .16), ("s2", .4, .14),
            ("s3", .2, .2), ("s4", .5, .17),
        ])]
        result = MetaforPairwiseMetaTool().invoke(profile, effects)
        self.assertEqual(result.k, 6)
        self.assertEqual(result.model, "multilevel_random_effects_cluster_robust_rho_0_6")
        self.assertLess(result.ci_lower, result.pooled_effect)
        self.assertLess(result.pooled_effect, result.ci_upper)
        self.assertIsNone(result.i_squared)


if __name__ == "__main__":
    unittest.main()
