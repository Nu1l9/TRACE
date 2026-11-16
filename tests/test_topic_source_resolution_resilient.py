import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'vendor/integration'),
               str(ROOT / 'vendor/integration/vendor/evidence_analysis'),
               str(ROOT / 'vendor/integration/vendor/evidence_extraction/system')]
from analysis_core.models import AnalysisDefinition, CandidateSet, CandidateGroup, EvidenceRecord
from analysis_engine.topic_reviewer import _make_result, build_topic_reviewer_prompt, DeterministicTopicReviewer
from analysis_engine.topic_reviewer_graph import GraphTopicReviewer
from evidence_reviewer.reviewer import _compact_batch_payload, _review_contract


class TopicSourceResolutionTests(unittest.TestCase):
    def fixture(self):
        analysis = AnalysisDefinition(analysis_id='mortality', pooled_result_id='pooled', title='Mortality',
            kind='primary', target_id='death', outcome_group='death', eligible_data_families=['binary_arm_counts'],
            accepted_effect_types=['RR'], population_policy='any')
        direct = EvidenceRecord(evidence_id='direct_counts', article_id='article_3', study_id='study_3',
            target_id='death', data_family='binary_arm_counts',
            raw_data={'intervention_events': 5, 'intervention_total': 19, 'control_events': 9, 'control_total': 37},
            source_location={'source_id': 'table_1', 'source_quote': 'Five of 19 and nine of 37 died at one year.',
                             'review_context': {'evidence_role': 'main_analysis_candidate'}})
        disputed = EvidenceRecord(evidence_id='unsupported_counts', article_id='article_17', study_id='study_17',
            target_id='death', data_family='binary_arm_counts',
            raw_data={'intervention_events': 11, 'intervention_total': 39, 'control_events': 31, 'control_total': 178},
            source_location={'source_id': 'figure_2', 'source_quote': 'Survival percentages at 36–42 months.',
                'review_context': {'evidence_role': 'source_only',
                    'reason': 'Source provides survival percentages; integer death counts are unsupported.'}})
        candidates = CandidateSet(target_id='death', groups=[CandidateGroup(group_id=r.study_id, study_id=r.study_id,
            article_ids=[r.article_id], evidence_ids=[r.evidence_id], records=[r]) for r in (direct, disputed)])
        return analysis, candidates

    def test_semantic_selects_from_analysis_fields_without_source_resolution(self):
        analysis, candidates = self.fixture()
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=
            '{"selected_evidence_ids":["direct_counts","unsupported_counts"],"summary":"select both"}'))])
        with patch('analysis_engine.topic_reviewer_graph.create_chat_completion', return_value=response):
            result = GraphTopicReviewer(api_key='offline-test', always_review=True).review(analysis, candidates)
        self.assertEqual(result.selected_evidence_ids, ['direct_counts', 'unsupported_counts'])
        self.assertEqual(result.source_resolution_issues, {})

    def test_semantic_ignores_legacy_source_resolution_output(self):
        analysis, candidates = self.fixture()
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=
            '{"selected_evidence_ids":["direct_counts","unsupported_counts"],"summary":"reviewed",'
            '"source_resolutions":[{"evidence_id":"unsupported_counts","decision":"supported",'
            '"field_support":[{"field_name":"unknown shape"}]}]}'))])
        with patch('analysis_engine.topic_reviewer_graph.create_chat_completion', return_value=response):
            result = GraphTopicReviewer(api_key='offline-test', always_review=True).review(analysis, candidates)
        self.assertEqual(result.selected_evidence_ids, ['direct_counts', 'unsupported_counts'])
        self.assertEqual(result.source_resolutions, [])

    def test_deterministic_reviewer_obeys_same_numeric_boundary(self):
        analysis, candidates = self.fixture()
        result = DeterministicTopicReviewer().review(analysis, candidates)
        self.assertEqual(result.selected_evidence_ids, ['direct_counts'])

    def test_semantic_keeps_one_requested_record_per_competing_context(self):
        analysis, candidates = self.fixture()
        good = candidates.groups[0].records[0]
        bad = candidates.groups[1].records[0].model_copy(update={"study_id": good.study_id})
        group = candidates.groups[0].model_copy(update={"records": [good, bad],
                                                       "evidence_ids": [good.evidence_id, bad.evidence_id]})
        candidates = candidates.model_copy(update={"groups": [group]})
        # The old lexical tie-breaker preferred unsupported_counts; gate before ranking.
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=
            '{"selected_evidence_ids":["direct_counts","unsupported_counts"],"summary":"select both"}'))])
        with patch('analysis_engine.topic_reviewer_graph.create_chat_completion', return_value=response):
            result = GraphTopicReviewer(api_key='offline-test', always_review=True).review(analysis, candidates)
        self.assertEqual(len(result.selected_evidence_ids), 1)
        self.assertTrue(set(result.selected_evidence_ids) <= {'direct_counts', 'unsupported_counts'})
        self.assertEqual(DeterministicTopicReviewer().review(analysis, candidates).selected_evidence_ids, ['direct_counts'])

    def test_actual_source_resolution_can_restore_a_misclassified_record(self):
        analysis, candidates = self.fixture()
        record = candidates.groups[1].records[0]
        record.source_location['source_evidence'] = [{'source_id': 'observed_table',
            'text': 'Observed death counts in the named arms at the protocol timepoint.'}]
        resolution = {'evidence_id': record.evidence_id, 'decision': 'supported',
            'reason': 'The retained table supports the arm-level facts omitted from the quote.',
            'field_support': [{'field': 'raw_data.' + field, 'source_id': 'observed_table',
                               'explanation': 'Source table cell for the corresponding arm.'}
                              for field in record.raw_data]}
        result = _make_result(analysis, candidates, {record.evidence_id}, summary='reviewed',
            reviewer_mode='llm', source_resolutions=[resolution])
        self.assertEqual(result.selected_evidence_ids, [record.evidence_id])
        self.assertEqual(result.source_resolutions, [resolution])
        self.assertIn('explicit source resolution', result.assignments[1].reason)

    def test_prompt_contains_analysis_data_without_review_or_provenance_payload(self):
        analysis, candidates = self.fixture()
        prompt = build_topic_reviewer_prompt(analysis, candidates.groups, allow_expansion=True)
        self.assertNotIn('source_resolutions', prompt)
        self.assertNotIn('source_location', prompt)
        self.assertNotIn('review_context', prompt)
        self.assertNotIn('integer death counts are unsupported', prompt)
        self.assertIn('"intervention_events":5', prompt)

    def test_analysis_prompt_does_not_send_full_source_blocks(self):
        analysis, candidates = self.fixture()
        first = candidates.groups[0].records[0]
        second = first.model_copy(deep=True, update={"evidence_id": "same_table_second_row"})
        source = {"source_id": "table_1", "text": "Unique complete source table text."}
        first.source_location['source_evidence'] = [source]
        second.source_location['source_evidence'] = [copy.deepcopy(source)]
        group = candidates.groups[0].model_copy(update={"records": [first, second]})
        prompt = build_topic_reviewer_prompt(analysis, [group])
        self.assertNotIn('Unique complete source table text.', prompt)
        self.assertNotIn('SOURCE_EVIDENCE_BLOCKS', prompt)
        self.assertNotIn('source_evidence_refs', prompt)
        self.assertNotIn('source_quote', prompt)

    def test_evidence_reviewer_receives_numeric_fields_and_deduplicated_full_sources(self):
        raw = {'intervention_events': 5, 'intervention_total': 19, 'control_events': 9, 'control_total': 37}
        rows = [{'prediction_id': 'a', 'raw_data': raw, 'used_effect': {}, 'source_location': {
            'source_id': 'table_1', 'source_quote': 'Mortality row.',
            'source_evidence': [{'source_id': 'table_1', 'text': 'Complete table with denominators.', 'source_type': 'table'}]}}]
        second = copy.deepcopy(rows[0]);second.update(prediction_id='b')
        second['source_location']['source_quote'] = 'Another row from the same table.'
        sources, compact = _compact_batch_payload(rows + [second])
        self.assertEqual(compact[0]['raw_data'], raw)
        self.assertEqual(compact[0]['supporting_source_evidence_ids'], compact[1]['supporting_source_evidence_ids'])
        self.assertEqual(sum('source_text' in block for block in sources), 1)
        compact[0]['raw_data']['intervention_events'] = 99
        self.assertEqual(rows[0]['raw_data']['intervention_events'], 5)


if __name__ == '__main__':
    unittest.main()
