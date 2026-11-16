import unittest
import json
from types import SimpleNamespace

from extensions.common import ROOT  # noqa: F401 - activate frozen analysis imports
from contract_adapter import _analysis_effect_type
from analysis_engine.pairwise_tools import PairwiseEffectTool


class AnalysisEffectTypeAdapterTests(unittest.TestCase):
    def test_adjusted_odds_ratio_uses_or_scale_without_losing_source_label(self):
        raw_data = {"adjustment_set": "age and sex"}
        effect_type = _analysis_effect_type("ma_0012", {"effect_type": "aOR"}, raw_data)

        self.assertEqual(effect_type, "OR")
        self.assertEqual(raw_data["_source_effect_type"], "aOR")
        record = SimpleNamespace(
            study_id="study_1", evidence_id="evidence_1",
            data_family="reported_ratio_effect", reported_effect_type=effect_type,
            reported_effect_value=2.08, ci_lower=1.25, ci_upper=3.45,
        )
        profile = SimpleNamespace(data_family="reported_ratio_effect", accepted_effect_types=["OR", "RR"])
        result = PairwiseEffectTool().qualify(profile, [record])

        self.assertEqual(result.decisions[0].status, "poolable")
        self.assertEqual(result.study_effects[0].effect_type, "OR")

    def test_incidence_rate_ratio_is_not_relabelled_as_or_or_risk_ratio(self):
        raw_data = {}

        self.assertEqual(_analysis_effect_type("ma_0012", {"effect_type": "rate ratio"}, raw_data), "rate ratio")
        self.assertEqual(raw_data, {})

    def test_other_cases_keep_their_original_effect_type_and_payload(self):
        raw_data = {"adjustment_set": "age and sex"}

        self.assertEqual(_analysis_effect_type("ma_0041", {"effect_type": "aOR"}, raw_data), "aOR")
        self.assertEqual(raw_data, {"adjustment_set": "age and sex"})



if __name__ == "__main__":
    unittest.main()
