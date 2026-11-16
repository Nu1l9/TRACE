"""Offline interface regressions from the raw-count/source-only failure pattern."""
import copy
import sys
import unittest
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'vendor/integration'),
               str(ROOT / 'vendor/integration/vendor/evidence_analysis')]
from trace_runtime.contracts import evidence_handoff, write_analysis_input
from contract_adapter import load_analysis_contract
from analysis_engine.evidence_graph import load_evidence_records


class ReviewHandoffTests(unittest.TestCase):
    def test_batch_caveats_reach_review_and_analysis_without_becoming_verdicts(self):
        from types import SimpleNamespace
        from extensions.resilient_extraction import _attach_extraction_diagnostics
        from trace_runtime.worker import load_extractor
        load_extractor({'options': {'extraction_runtime': 'typed'}})
        from evidence_reviewer.reviewer import _compact_candidate
        args = self.fixture()
        row = args[2]['predictions'][0]
        warning = 'Source figure_2: integer deaths were not explicitly reported.'
        _attach_extraction_diagnostics([row], [
            {'batch_id': 'batch_1', 'source_ids': ['figure_2']}],
            [SimpleNamespace(unresolved_items=[warning])])
        compact = _compact_candidate(row, 'source_1')
        self.assertEqual(compact['extraction_issues'][0]['notes'], [warning])
        self.assertEqual(compact['extraction_issues'][0]['scope'], 'batch_context_not_row_verdict')
        ledger, _ = evidence_handoff(*args)
        context = ledger['candidates'][0]['source_location']['review_context']
        self.assertEqual(context['extraction_diagnostics']['issues'][0]['notes'], [warning])
        self.assertEqual(row['raw_data']['intervention_events'], 11)

    def fixture(self):
        request = {'case_id': 'example', 'documents': [
            {'article_id': 'article_1', 'study_id': 'study_1', 'document_id': 'document_1'}],
            'protocol': {'targets': [{'target_id': 'mortality'}]}}
        screening = {'case_id': 'example', 'decisions': [
            {'candidate_id': 'article_1', 'final_status': 'Include'}]}
        row = {'prediction_id': 'evidence_1', 'article_id': 'article_1', 'target_id': 'mortality',
               'outcome': 'All-cause mortality', 'data_family': 'binary_arm_counts',
               'raw_data': {'intervention_events': 11, 'intervention_total': 39,
                            'control_events': 31, 'control_total': 178},
               'quantitative_usability': 'poolable', 'verification_status': 'human_review',
               'used_effect': {'effect_type': None, 'value': None, 'ci_lower': None, 'ci_upper': None},
               'source_location': {'document_id': 'document_1', 'source_id': 'figure_2',
                                   'source_quote': 'Survival percentages at the specified follow-up.'}}
        decision = {'prediction_id': 'evidence_1', 'evidence_role': 'source_only',
                    'disposition': 'keep_as_valid_candidate',
                    'reason': 'The supplied source reports survival percentages, not observed integer death counts.'}
        submission = {'case_id': 'example', 'predictions': [row]}
        review = {'case_id': 'example', 'candidate_reviews': [
            {'prediction_id': 'evidence_1', 'article_id': 'article_1', 'review': decision}]}
        return request, screening, submission, review

    def test_raw_counts_and_review_concern_survive_together(self):
        args = self.fixture()
        original = copy.deepcopy(args)
        ledger, audit = evidence_handoff(*args)
        candidate = ledger['candidates'][0]
        context = candidate['source_location']['review_context']
        self.assertEqual(candidate['raw_data'], args[2]['predictions'][0]['raw_data'])
        self.assertEqual(context['evidence_role'], 'source_only')
        self.assertEqual(context['reason'], args[3]['candidate_reviews'][0]['review']['reason'])
        self.assertEqual(audit['records'][0]['review_context'], context)
        self.assertEqual(args, original)
        self.assertNotIn('quantitative_usability', candidate)
        self.assertEqual(context['extraction_diagnostics']['numeric_assessment'], 'poolable')

    def test_review_context_survives_real_contract_adapter_and_graph_record_loading(self):
        args = self.fixture()
        request = args[0]
        request['topic'] = 'All-cause mortality'
        request['protocol']['pico'] = {'intervention_or_exposure': {'description': 'Intervention'}}
        request['analysis_profile'] = {'profile_version': '2.0', 'primary_analysis_spec': {
            'analysis_id': 'death', 'eligible_data_families': ['binary_arm_counts'], 'accepted_effect_types': ['RR']}}
        ledger, _ = evidence_handoff(*args)
        with tempfile.TemporaryDirectory() as directory:
            write_analysis_input(request, ledger, Path(directory))
            _, _, _, payloads, studies = load_analysis_contract(Path(directory))
            records = load_evidence_records(payloads, studies)
        self.assertEqual(records[0].source_location['review_context'], ledger['candidates'][0]['source_location']['review_context'])
        self.assertEqual(records[0].raw_data['intervention_events'], 11)

    def test_clear_semantic_mismatch_still_excluded_but_audited(self):
        args = self.fixture()
        args[3]['candidate_reviews'][0]['review'].update(
            evidence_role='none', disposition='discard_clear_mismatch')
        ledger, audit = evidence_handoff(*args)
        self.assertEqual(ledger['candidates'], [])
        self.assertEqual(audit['discarded_count'], 1)
        self.assertEqual(audit['records'][0]['review_context']['evidence_role'], 'none')

    def test_raw_arms_are_not_replaced_with_upstream_calculated_effect(self):
        args = self.fixture()
        args[2]['predictions'][0]['used_effect'] = {'effect_type': 'RR', 'value': 42}
        ledger, audit = evidence_handoff(*args)
        self.assertEqual(ledger['candidates'][0]['effect'], {})
        self.assertIn('upstream_derived_effect_omitted; analysis recalculates from raw_data',
                      audit['records'][0]['warnings'])


if __name__ == '__main__':
    unittest.main()
