import copy
import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vendor/integration'))

from extensions.evidence_context import build_review_context, needs_source_resolution, resolution_problem


class EvidenceContextTests(unittest.TestCase):
    def candidate(self):
        prediction = {
            'raw_data': {'intervention_events': 11, 'intervention_total': 39,
                         'control_events': 31, 'control_total': 178},
            'quantitative_usability': 'poolable', 'verification_status': 'human_review',
        }
        decision = {'evidence_role': 'source_only', 'disposition': 'keep_as_valid_candidate',
                    'reason': 'Only survival percentages support this record; counts need verification.'}
        return {**prediction, 'effect': {}, 'source_location': {
            'source_id': 'table_2', 'source_quote': 'Test source table with independently reviewed numerical cells.', 'review_context': build_review_context(prediction, decision)}}

    def resolution(self):
        return {'decision': 'supported', 'reason': 'The table directly reports each arm count.',
                'field_support': [{'field': 'raw_data.' + name, 'source_id': 'table_2',
                                  'explanation': 'Direct table cell, with the named arm and timepoint.'}
                                  for name in self.candidate()['raw_data']]}

    def test_source_only_cannot_silently_become_poolable(self):
        candidate = self.candidate()
        self.assertTrue(resolution_problem(candidate, None))
        self.assertEqual(candidate['source_location']['review_context']['evidence_role'], 'source_only')
        self.assertEqual(candidate['raw_data']['intervention_events'], 11)

    def test_complete_explicit_source_resolution_can_override_role(self):
        self.assertIsNone(resolution_problem(self.candidate(), self.resolution()))

    def test_partial_or_invented_source_resolution_cannot_override_role(self):
        resolution = self.resolution()
        resolution['field_support'].pop()
        self.assertTrue(resolution_problem(self.candidate(), resolution))
        resolution = self.resolution()
        resolution['field_support'][0]['source_id'] = 'not_supplied'
        self.assertTrue(resolution_problem(self.candidate(), resolution))

    def test_cannot_resolve_stays_unresolved(self):
        resolution = self.resolution()
        resolution['decision'] = 'unresolved'
        self.assertTrue(resolution_problem(self.candidate(), resolution))

    def test_legacy_locator_status_alone_does_not_exclude(self):
        candidate = self.candidate()
        candidate['source_location']['review_context']['evidence_role'] = 'main_analysis_candidate'
        self.assertIsNone(resolution_problem(candidate, None))

    def test_new_unsupported_finding_overrides_old_main_candidate(self):
        candidate = self.candidate()
        candidate['source_location']['review_context']['evidence_role'] = 'main_analysis_candidate'
        self.assertTrue(resolution_problem(candidate, {'decision': 'unsupported'}))

    def test_optional_positive_resolution_does_not_recheck_primary_candidate(self):
        candidate = self.candidate()
        candidate['source_location']['review_context']['evidence_role'] = 'main_analysis_candidate'
        # Topic reviewers sometimes volunteer this record even though it was
        # not flagged for resolution.  Empty field_support must not erase the
        # existing source-backed primary-candidate decision.
        self.assertIsNone(resolution_problem(candidate, {
            'decision': 'supported', 'reason': 'The supplied table reports both arms.',
            'field_support': [],
        }))

    def test_empty_quote_and_unknown_block_do_not_establish_support(self):
        candidate = self.candidate()
        candidate['source_location']['source_quote'] = ''
        self.assertTrue(resolution_problem(candidate, self.resolution()))

    def test_full_source_block_can_support_resolution_when_quote_is_short(self):
        candidate = self.candidate()
        candidate['source_location']['source_quote'] = ''
        candidate['source_location']['source_evidence'] = [
            {'source_id': 'table_2', 'text': 'Complete source table retained from parsing.'}]
        self.assertIsNone(resolution_problem(candidate, self.resolution()))

    def test_context_does_not_mutate_original_data(self):
        prediction = {'quantitative_usability': 'human_review', 'issues': [{'code': 'numeric_conflict'}]}
        original = copy.deepcopy(prediction)
        context = build_review_context(prediction, {'evidence_role': 'uncertain'})
        context['extraction_diagnostics']['issues'][0]['code'] = 'changed'
        self.assertEqual(prediction, original)
        self.assertTrue(needs_source_resolution(context))


if __name__ == '__main__':
    unittest.main()
