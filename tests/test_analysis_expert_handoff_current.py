import unittest
from types import SimpleNamespace

from extensions.analysis_worker import retain_excluded_candidates_for_expert_review


class AnalysisExpertHandoffTests(unittest.TestCase):
    def test_underpowered_topic_retains_aligned_exclusion_without_selecting_it(self):
        candidate = SimpleNamespace(
            evidence_id='candidate_a', target_id='episode_rate',
            data_family='event_counts_with_person_time', outcome_code='',
            reported_effect_type=None, estimand='surgery_vs_control',
        )
        wrong_target = SimpleNamespace(
            evidence_id='candidate_b', target_id='symptom_resolution',
            data_family='event_counts_with_person_time', outcome_code='',
            reported_effect_type=None, estimand='surgery_vs_control',
        )
        aligned_assignment = SimpleNamespace(
            evidence_id='candidate_a', role='excluded', reason='Reviewer exclusion',
        )
        unrelated_assignment = SimpleNamespace(
            evidence_id='candidate_b', role='excluded', reason='Reviewer exclusion',
        )
        topic = SimpleNamespace(
            status='insufficient_data',
            analysis=SimpleNamespace(
                target_id='episode_rate',
                eligible_data_families=['event_counts_with_person_time'],
                eligible_outcome_codes=['episode_rate'],
                accepted_effect_types=['IRR'],
                estimands=['surgery_vs_control'],
                minimum_studies=2,
            ),
            poolability=SimpleNamespace(study_effects=[SimpleNamespace(study_id='study_1')]),
            source_resolutions=[],
            assignments=[aligned_assignment, unrelated_assignment],
            warnings=[],
        )
        result = SimpleNamespace(analyses=[topic])

        promoted = retain_excluded_candidates_for_expert_review(result, [candidate, wrong_target])

        self.assertEqual(promoted, 1)
        self.assertEqual(aligned_assignment.role, 'needs_review')
        self.assertIn('not included in automatic pooling', aligned_assignment.reason)
        self.assertEqual(unrelated_assignment.role, 'excluded')
        self.assertEqual(topic.warnings, [
            '1 target/data-family-aligned reviewer-excluded candidate(s) retained for expert review; '
            'they remain excluded from automatic pooling.'
        ])


if __name__ == '__main__':
    unittest.main()
