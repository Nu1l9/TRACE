import copy
import http.client
import json
import math
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from expert_qa.agent import Agent, AnswerError, validate_arguments, tools_schema
from expert_qa.store import Snapshot, FILES, digest
from expert_qa.server import Application, make_server
from expert_qa.provider import Transport

def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False))

def fixture(root):
    run = root / 'run'; run.mkdir()
    (run / 'source.md').write_text('Title\nOriginal result RR 1.2 (95% CI 1.1-1.3).\n' + 'Context\n' * 70)
    write(run / 'content.json', [{'text': 'Title'}, {'table_body': '<table>RR 1.2</table>'}])
    write(run / FILES['request'], {'case_id': 'test_case', 'documents': [{'article_id': 'paper1',
          'title': 'Study one', 'paths': {'markdown': 'source.md', 'content_list': 'content.json'}}],
          'protocol': {'population': 'adults'}, 'analysis_profile': {}})
    analysis = {'analysis': {'analysis_id': 'a1', 'target_id': 't1'}, 'status': 'completed',
                'selected_evidence_ids': ['e1', 'e2'], 'poolability': {'study_effects': [{'evidence_id': 'e1', 'value': 1.2}]},
                'assignments': [{'evidence_id': 'e1', 'reason': 'Prespecified adjusted model'}]}
    write(run / FILES['analysis'], {'analyses': [analysis, {'analysis': {'analysis_id': 'a2', 'target_id': 't2'},
                                                            'status': 'blocked', 'selected_evidence_ids': []}]})
    write(run / FILES['extraction'], {'predictions': [{'prediction_id': 'e1', 'article_id': 'paper1',
          'raw_data': {'effect': 1.2}, 'source_location': {'content_list_item_index': 1}},
          {'prediction_id': 'e2', 'article_id': 'paper1', 'raw_data': {'effect': 1.5}}]})
    write(run / FILES['review'], {'candidate_reviews': [{'prediction_id': 'e1', 'disposition': 'retain'}]})
    write(run / FILES['rob'], {'assessments': [{'article_id': 'paper1', 'scope': {'target_id': 't2'}, 'status': 'unresolved'}]})
    write(run / FILES['screening_original'], {'decisions': [{'article_id': 'paper1', 'decision': 'include'}]})
    return run

def call(name, args):
    return {'message': {'content': None, 'tool_calls': [{'id': 'call_' + name, 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(args)}}]}, 'usage': {'total_tokens': 12}}

class Fake:
    def __init__(self, bad_citation=False):
        self.requests = []; self.bad_citation = bad_citation
    def metadata(self):
        return {'model': 'offline-test', 'provider_host': 'mock'}
    def complete(self, messages, tools):
        self.requests.append(copy.deepcopy(messages))
        n = len(self.requests)
        if n == 1: return call('get_analysis_selection', {})
        if n == 2: return call('get_extraction', {'evidence_id': 'e1'})
        if n == 3: return call('get_source', {'article_id': 'paper1', 'content_item': 1})
        citations = [json.loads(m['content'])['citation_id'] for m in messages if m['role'] == 'tool' and 'citation_id' in json.loads(m['content'])]
        if n == 4 and self.bad_citation:
            citations = ['invented']
        return call('finish_answer', {'claims': [{'kind': 'recorded', 'text': 'Both the saved extraction record and source table report RR 1.2.', 'citation_ids': citations}]})

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.run = fixture(self.root); self.snapshot = Snapshot(self.run)
    def tearDown(self):
        self.tmp.cleanup()
    def test_english_suggestions_keep_research_modes(self):
        import re
        from expert_qa.research import is_research_question, wants_hypotheses
        from expert_qa.server import ASSETS
        questions = re.findall(r'data-question="([^"]+)"', (ASSETS/'index.html').read_text())
        self.assertEqual(len(questions), 5)
        self.assertTrue(is_research_question(questions[3]))
        self.assertTrue(wants_hypotheses(questions[3]))
        self.assertTrue(is_research_question(questions[4]))
        self.assertFalse(wants_hypotheses(questions[4]))

    def test_sources_and_chain(self):
        s = self.snapshot
        selection = s.query('get_analysis_selection', {}, 'a1')
        self.assertEqual(selection['selected_evidence_ids'], ['e1', 'e2'])
        self.assertEqual(len(selection['poolability']['study_effects']), 1)
        row = s.query('get_extraction', {'evidence_id': 'e1'}, 'a1')
        self.assertEqual(row['review']['disposition'], 'retain')
        self.assertIsNone(row['analysis_input'])
        self.assertFalse(s.query('get_extraction', {'evidence_id': 'e1'}, 'a2')['selected_in_current_analysis'])
        self.assertEqual(s.query('get_source', {'article_id': 'paper1', 'content_item': 1}, 'a1')['text'], '<table>RR 1.2</table>')
        self.assertIn('2: Original', s.query('get_source', {'article_id': 'paper1', 'query': 'RR'}, 'a1')['text'])
    def test_missing_not_invented_and_scoped_rob(self):
        rob = self.snapshot.query('get_rob', {'article_id': 'paper1'}, 'a1')
        self.assertEqual(rob['current_target_id'], 't1')
        self.assertEqual(rob['assessments'][0]['scope']['target_id'], 't2')
        self.assertIsNone(self.snapshot.query('get_analysis_selection', {}, 'a1')['review'])
        self.assertEqual(self.snapshot.query('get_screening', {'article_id': 'paper1'}, 'a1')['decision']['decision'], 'include')
    def test_frozen_snapshot_and_fingerprint(self):
        old = self.snapshot.fingerprint
        (self.run / 'source.md').write_text('Changed')
        self.assertIn('Original result', self.snapshot.query('get_source', {'article_id': 'paper1'}, 'a1')['text'])
        self.assertNotEqual(old, Snapshot(self.run).fingerprint)
    def test_table_search_prefers_caption_to_prose_mention(self):
        self.snapshot.sources['paper1']['text'] = '\n'.join(
            ['Results are shown in', 'Table 2 and Supplementary Figure 1.', *(['Context'] * 12),
             'Table 2: Entire and matched cohorts', 'Entire cohort: 4/39 vs 17/256'])
        result = self.snapshot.query('get_source', {'article_id': 'paper1', 'query': 'Table 2',
                                                    'line_count': 12}, 'a1')
        self.assertIn('Entire cohort: 4/39 vs 17/256', result['text'])
        self.assertEqual(result['matching_lines'], [2, 15])
        self.assertEqual(result['matching_previews'][0]['line'], 2)
        # An explicit line read must still return the original beginning.
        direct = self.snapshot.query('get_source', {'article_id': 'paper1', 'line': 1,
                                                   'line_count': 3}, 'a1')
        self.assertNotIn('Entire cohort', direct['text'])
    def test_model_comparison_does_not_substitute_single_study_effect(self):
        self.snapshot.analyses['a1']['robustness'] = {
            'model_comparison': {'fixed_effect_pooled_effect': 1.1, 'random_effects_pooled_effect': 1.09}}
        result = self.snapshot.query('get_model_comparison', {}, 'a1')
        self.assertEqual(result['saved_pooled_model_comparison']['random_effects_pooled_effect'], 1.09)
        self.assertNotIn('ci_lower', result['saved_pooled_model_comparison'])
        self.assertNotIn('poolability', result)
    def test_study_effect_reads_table_independently_of_unresolved_saved_quote(self):
        self.snapshot.sources['paper1']['text'] = (
            'Original title\nTable 3: Mortality\nGroup A (n=40); Group B (n=50)\nDeaths: 3; 7')
        location = self.snapshot.evidence['e1']['source_location']
        location.update(source_quote='Table 3: copied extraction incorrectly says 9 vs 8',
                        resolution_status='unresolved')
        before = copy.deepcopy(self.snapshot.evidence)
        result = self.snapshot.query('get_study_effect', {'position': 1}, 'a1')['matches'][0]
        candidate = result['original_table_candidate']
        self.assertIn('Deaths: 3; 7', candidate['source']['text'])
        self.assertNotIn('9 vs 8', candidate['source']['text'])
        self.assertEqual(candidate['status'], 'candidate_original_passage_not_a_verdict')
        self.assertNotIn('source_quote', result['extraction']['source_location'])
        self.assertEqual(result['extraction']['source_location']['resolution_status'], 'unresolved')
        self.assertEqual(self.snapshot.evidence, before)
    def test_pagination_and_pointer(self):
        self.snapshot.analyses['a1']['long'] = 'x' * 18000
        p = self.snapshot.page('get_analysis', {'field': '/record/long'}, 'a1')
        self.assertEqual(p['next_offset'], 8000)
        end = self.snapshot.page('get_analysis', {'field': '/record/long', 'offset': 16000}, 'a1')
        self.assertEqual(len(end['text']), 2000); self.assertIsNone(end['next_offset'])
    def test_default_projections_preserve_explicit_raw_access(self):
        self.snapshot.analyses['a1']['statistics'] = {'k': 1, 'pooled_effect': 1.2}
        self.snapshot.analyses['a1']['candidate_inventory'] = {
            'candidate_record_count': 2, 'candidate_study_count': 1, 'large_detail': 'x' * 18000}
        self.snapshot.evidence['e1']['source_location']['source_evidence'] = ['embedded copy' * 2000]
        before = copy.deepcopy(self.snapshot.raw)
        overview = json.loads(self.snapshot.page('get_analysis', {}, 'a1')['text'])
        self.assertEqual(overview['record']['statistics']['pooled_effect'], 1.2)
        self.assertIn('projection', overview)
        selection = json.loads(self.snapshot.page('get_analysis_selection', {}, 'a1')['text'])
        self.assertEqual(selection['pooled_effect_count'], 1)
        self.assertEqual(selection['candidate_inventory']['candidate_record_count'], 2)
        extraction = json.loads(self.snapshot.page('get_extraction', {'evidence_id': 'e1'}, 'a1')['text'])
        self.assertNotIn('source_evidence', extraction['extraction']['source_location'])
        raw = self.snapshot.page('get_extraction', {'evidence_id': 'e1', 'field': '/extraction/source_location/source_evidence'}, 'a1')
        self.assertIn('embedded copy', raw['text'])
        self.assertEqual(raw['next_offset'], 8000)
        self.assertEqual(before, self.snapshot.raw)
    def saved_omission(self):
        self.snapshot.analyses['a1']['robustness'] = {
            'status': 'completed', 'k': 13, 'effect_type': 'RR', 'primary_model': 'fixed_effect',
            'leave_one_out': [{'evidence_id': 'e1', 'study_id': 's1', 'pooled_log_effect': .083,
                              'ci_lower_log': .0016, 'ci_upper_log': .1644, 'i_squared': .2625,
                              'tau_squared': None}]}
        return self.snapshot.analyses['a1']['robustness']

    def test_saved_omission_lookup_and_display_without_mutation(self):
        robustness = self.saved_omission()
        before = copy.deepcopy(self.snapshot.raw)
        for args in ({'evidence_id': 'e1'}, {'study_id': 's1'},
                     {'evidence_id': 'e1', 'study_id': 's1'}):
            result = self.snapshot.query('get_leave_one_out', args, 'a1')
            self.assertEqual(result['availability'], 'found')
            match = result['matches'][0]
            self.assertEqual(match['record'], robustness['leave_one_out'][0])
            self.assertAlmostEqual(match['display']['pooled_effect'], math.exp(.083))
            self.assertAlmostEqual(match['display']['ci_lower'], math.exp(.0016))
            self.assertAlmostEqual(match['display']['ci_upper'], math.exp(.1644))
            self.assertEqual(match['display']['i_squared'], .2625)
            self.assertIsNone(match['display']['tau_squared'])
            self.assertEqual(match['remaining_effect_count'], 12)
            self.assertEqual(match['source_pointer'], '/robustness/leave_one_out/0')
        self.assertEqual(before, self.snapshot.raw)

    def test_omission_missing_scoped_and_conflicting_identifiers(self):
        self.saved_omission()
        for args in ({'evidence_id': 'e2'}, {'evidence_id': 'e1', 'study_id': 'other'}):
            result = self.snapshot.query('get_leave_one_out', args, 'a1')
            self.assertEqual(result['availability'], 'not_found')
            self.assertEqual(result['matches'], [])
        self.assertEqual(self.snapshot.query('get_leave_one_out', {'evidence_id': 'e1'}, 'a2')['availability'], 'unavailable')
        for rows in (None, [], {'e1': {}}, ['malformed']):
            self.snapshot.analyses['a1']['robustness']['leave_one_out'] = rows
            result = self.snapshot.query('get_leave_one_out', {'evidence_id': 'e1'}, 'a1')
            self.assertEqual(result['availability'], 'unavailable')

    def test_omission_identifier_field_mismatch_requires_explicit_retry(self):
        self.saved_omission()
        before = copy.deepcopy(self.snapshot.analyses)
        for args, expected in (({'evidence_id': 's1'}, 'study_id'),
                               ({'study_id': 'e1'}, 'evidence_id')):
            with self.assertRaisesRegex(ValueError, 'saved ' + expected):
                self.snapshot.query('get_leave_one_out', args, 'a1')
        self.assertEqual(self.snapshot.query('get_leave_one_out', {'evidence_id': 'unknown'}, 'a1')['availability'], 'not_found')
        self.assertEqual(self.snapshot.query('get_leave_one_out', {'study_id': 's1'}, 'a1')['availability'], 'found')
        self.assertEqual(before, self.snapshot.analyses)

    def test_saved_ci_null_comparison_uses_inclusive_bounds_and_effect_scale(self):
        robustness = self.saved_omission()
        row = robustness['leave_one_out'][0]
        def interpretation():
            return self.snapshot.query('get_leave_one_out', {'study_id': 's1'}, 'a1')['matches'][0]['ci_interpretation']
        self.assertFalse(interpretation()['contains_null'])
        for lo, hi, expected in ((0, .2, True), (-.2, 0, True), (-.2, -.01, False)):
            row.update(ci_lower_log=lo, ci_upper_log=hi)
            self.assertEqual(interpretation()['contains_null'], expected)
        row.update(ci_lower_log=None)
        self.assertIsNone(interpretation()['contains_null'])
        robustness['effect_type'] = 'MD'
        row.update(ci_lower=-.2, ci_upper=.1)
        self.assertEqual(interpretation()['null_value'], 0)
        self.assertTrue(interpretation()['contains_null'])
        row.update(ci_lower=.2, ci_upper=.1)
        self.assertIsNone(interpretation()['contains_null'])

    def test_omission_scale_and_missing_values(self):
        robustness = self.saved_omission()
        row = robustness['leave_one_out'][0]
        for effect_type in ('OR', 'HR', 'IRR'):
            robustness['effect_type'] = effect_type
            self.assertAlmostEqual(self.snapshot.query('get_leave_one_out', {'study_id': 's1'}, 'a1')
                                   ['matches'][0]['display']['pooled_effect'], math.exp(.083))
        robustness['effect_type'] = 'MD'
        match = self.snapshot.query('get_leave_one_out', {'study_id': 's1'}, 'a1')['matches'][0]
        self.assertIsNone(match['display']['pooled_effect'])
        row.update(pooled_effect=0, ci_lower=-.2, ci_upper=.2)
        match = self.snapshot.query('get_leave_one_out', {'study_id': 's1'}, 'a1')['matches'][0]
        self.assertEqual(match['display']['pooled_effect'], 0)
        self.assertEqual(match['display']['ci_lower'], -.2)
        robustness['effect_type'] = 'RR'
        row.update(pooled_effect=None, pooled_log_effect=1000, ci_lower=None, ci_lower_log=None)
        match = self.snapshot.query('get_leave_one_out', {'study_id': 's1'}, 'a1')['matches'][0]
        self.assertIsNone(match['display']['pooled_effect'])
        self.assertIsNone(match['display']['ci_lower'])

    def test_omission_arguments(self):
        for args in ({}, {'field': '/matches'}, {'evidence_id': ''}, {'study_id': 12}):
            with self.assertRaises(ValueError):
                validate_arguments('get_leave_one_out', args)
        validate_arguments('get_leave_one_out', {'evidence_id': 'e1'})
        validate_arguments('get_leave_one_out', {'study_id': 's1'})
    def test_tool_schemas_are_official_api_compatible(self):
        for tool in tools_schema():
            schema = tool['function']['parameters']
            self.assertEqual(schema['type'], 'object')
            self.assertFalse(set(schema) & {'oneOf', 'anyOf', 'allOf', 'enum', 'const', 'not'})

    def test_author_lookup_in_source_opening_with_placeholder_title(self):
        self.snapshot.sources['paper1']['text'] = 'Clinical outcomes\nAlice Exampleton et al. 2018\nAbstract\n'
        self.snapshot.documents['paper1']['title'] = 'Candidate 001'
        result = self.snapshot.query('search_case_evidence', {'query': 'Exampleton et al 2018'}, 'a1')
        hit = next(row for row in result['matches'] if row['kind'] == 'source_opening')
        self.assertEqual(hit['article_id'], 'paper1')
        self.assertEqual(hit['selected_evidence_ids'], ['e1', 'e2'])
        self.assertIn('Exampleton', hit['excerpt'])
        self.snapshot.sources['paper1']['text'] = 'x' * 6100 + '\nExampleton 2018'
        result = self.snapshot.query('search_case_evidence', {'query': 'Exampleton'}, 'a1')
        self.assertFalse(any(row['kind'] == 'source_opening' for row in result['matches']))

    def test_single_effect_provenance_and_bounds(self):
        # Selection includes e2, but only e1 is actually pooled.
        self.snapshot.evidence['e1']['source_location']['source_evidence'] = ['x' * 20000]
        result = self.snapshot.query('get_study_effect', {'position': 1}, 'a1')
        match = result['matches'][0]
        self.assertEqual(match['effect']['evidence_id'], 'e1')
        self.assertEqual(match['article']['title'], 'Study one')
        self.assertEqual(match['extraction']['source_location']['content_list_item_index'], 1)
        self.assertNotIn('source_evidence', match['extraction']['source_location'])
        self.assertIn('source_evidence', self.snapshot.evidence['e1']['source_location'])
        self.assertEqual(self.snapshot.query('get_study_effect', {'evidence_id': 'e1'}, 'a1')['matches'], [match])
        for args in ({'evidence_id': 'e2'}, {'position': 1, 'evidence_id': 'e2'}):
            self.assertEqual(self.snapshot.query('get_study_effect', args, 'a1')['availability'], 'not_found')
        self.assertEqual(self.snapshot.query('get_study_effect', {'evidence_id': 'e1'}, 'a2')['matches'], [])
        for args in ({}, {'position': 0}, {'position': 2}, {'position': True}, {'evidence_id': ''}):
            with self.assertRaises(ValueError):
                self.snapshot.query('get_study_effect', args, 'a1')
        with self.assertRaises(ValueError):
            validate_arguments('get_study_effect', {})

    def test_focused_source_read(self):
        result = self.snapshot.query('get_source', {'article_id': 'paper1', 'query': 'Original', 'line_count': 4}, 'a1')
        self.assertEqual(result['line_end'], 4)
        self.assertEqual(result['next_line'], 5)
        self.assertIn('Original result', result['text'])
        for count in (0, 41, True):
            with self.assertRaises(ValueError):
                self.snapshot.query('get_source', {'article_id': 'paper1', 'line_count': count}, 'a1')

    def test_omission_agent_trace_citations_and_history(self):
        self.saved_omission()
        class Omission(Fake):
            def complete(self, messages, tools):
                self.requests.append(copy.deepcopy(messages))
                if len(self.requests) == 1:
                    return call('get_leave_one_out', {'evidence_id': 'e1'})
                page = json.loads(messages[-1]['content'])
                result = json.loads(page['text'])
                display = result['matches'][0]['display']
                return call('finish_answer', {'claims': [{'kind': 'recorded',
                    'text': f"Saved omit-one RR {display['pooled_effect']:.3f}; no synthesis rerun.",
                    'citation_ids': [page['citation_id']]}]})
        answer = Agent(self.snapshot, Omission()).answer('Omit e1', 'a1', history=[
            {'question': 'Omit e1', 'claims': [{'text': 'No saved result exists.'}]}])
        self.assertIn('1.087', answer['claims'][0]['text'])
        page = next(iter(answer['citations'].values()))
        self.assertEqual(page['tool'], 'get_leave_one_out')
        self.assertIsNone(page['next_offset'])
        from expert_qa.presentation import citation_html
        self.assertIn('Saved leave-one-out results', citation_html(page, self.snapshot))
        self.assertEqual(answer['process_trace'][-1]['type'], 'answer_ready')
    def test_unknown_ids_and_tools_rejected(self):
        for name, args, aid in [('read_file', {'path': '/etc/passwd'}, 'a1'), ('get_analysis', {}, 'other'),
                                ('get_extraction', {'evidence_id': 'elsewhere'}, 'a1')]:
            with self.assertRaises(ValueError): self.snapshot.query(name, args, aid)
        with self.assertRaises(ValueError): validate_arguments('get_source', {'article_id':'paper1', 'path':'/etc/passwd'})
        with self.assertRaises(ValueError): validate_arguments('get_source', {'article_id':'paper1', 'line':True})
    def test_gold_not_read(self):
        write(self.run / 'gold.json', {'answer': 'GOLD_SECRET'})
        self.assertNotIn('GOLD_SECRET', str(Snapshot(self.run).raw))
    def test_agent_citations_and_no_writes(self):
        before = {p.name: digest(p.read_bytes()) for p in self.run.rglob('*') if p.is_file()}
        answer = Agent(self.snapshot, Fake(bad_citation=True)).answer('Verify original source', 'a1')
        self.assertEqual(len(answer['citations']), 3)
        self.assertTrue(all(cid in answer['citations'] for cid in answer['claims'][0]['citation_ids']))
        self.assertTrue(any('Unknown citation' in str(r) for r in answer['tool_audit']))
        after = {p.name: digest(p.read_bytes()) for p in self.run.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
    def test_short_citation_labels_resolve_and_unknown_labels_rejected(self):
        class Aliases(Fake):
            def complete(self, messages, tools):
                self.requests.append(copy.deepcopy(messages))
                n = len(self.requests)
                if n == 1:
                    return call('get_analysis', {})
                if n == 2:
                    return call('finish_answer', {'claims': [
                        {'kind': 'recorded', 'text': 'Unsupported alias', 'citation_ids': ['E99']}]})
                return call('finish_answer', {'claims': [
                    {'kind': 'recorded', 'text': 'Saved record', 'citation_ids': ['E1']}]})
        answer = Agent(self.snapshot, Aliases()).answer('Explain', 'a1')
        canonical = next(iter(answer['citations']))
        self.assertEqual(answer['claims'][0]['citation_ids'], [canonical])
        self.assertEqual(answer['citations'][canonical]['citation_label'], 'E1')
        failure = next(row['result'] for row in answer['tool_audit'] if 'error' in row['result'])
        self.assertIn('Unknown citation', failure['error'])
        self.assertEqual(failure['available_citation_labels'], ['E1'])
    def test_bounded_invalid_model(self):
        class Bad(Fake):
            def complete(self, messages, tools): return call('execute_python', {'code': 'write file'})
        with self.assertRaises(AnswerError): Agent(self.snapshot, Bad(), max_rounds=2).answer('hi', 'a1')
    def test_final_round_has_one_bounded_citation_repair(self):
        class Repair(Fake):
            def complete(self, messages, tools):
                self.requests.append(copy.deepcopy(messages))
                if len(self.requests) == 1:
                    return call('get_analysis', {})
                self.asserted_tools = [tool['function']['name'] for tool in tools]
                return call('finish_answer', {'claims': [
                    {'kind': 'recorded', 'text': 'The saved analysis is completed.',
                     'citation_ids': [] if len(self.requests) == 2 else ['E1']}]})
        model = Repair()
        result = Agent(self.snapshot, model, max_rounds=2).answer('Status?', 'a1')
        self.assertEqual(len(model.requests), 3)
        self.assertEqual(model.asserted_tools, ['finish_answer'])
        self.assertEqual(len(result['citations']), 1)
    def test_repeated_frozen_read_reserves_evidence_bounded_final_answer(self):
        class Repeats(Fake):
            def complete(self, messages, tools):
                self.requests.append(copy.deepcopy(messages))
                names = [tool['function']['name'] for tool in tools]
                if names == ['finish_answer']:
                    return call('finish_answer', {'claims': [
                        {'kind': 'limitation', 'text': 'The requested new result is not established by these records.',
                         'citation_ids': ['E1']}]})
                return call('get_analysis', {})
        answer = Agent(self.snapshot, Repeats()).answer('Explain', 'a1')
        self.assertEqual(len(answer['usage']), 3)
        self.assertEqual(answer['claims'][0]['kind'], 'limitation')
    def test_process_events_match_actual_reads_and_are_saved(self):
        events=[]
        answer=Agent(self.snapshot, Fake()).answer('Explain', 'a1', on_event=events.append)
        self.assertEqual(events,answer['process_trace'])
        self.assertEqual(events[0]['type'],'model_started')
        self.assertEqual(events[-1]['type'],'answer_ready')
        reads=[e for e in events if e['type']=='tool_completed']
        self.assertEqual([e['tool'] for e in reads],['get_analysis_selection','get_extraction','get_source'])
        self.assertEqual([e['sequence'] for e in events],list(range(1,len(events)+1)))
        for e in reads:self.assertEqual(e['result'],answer['citations'][e['result']['citation_id']])
    def test_failed_question_keeps_retrieved_citations(self):
        class Fails(Fake):
            def complete(self,messages,tools):
                if self.requests:raise AnswerError('Mock timeout')
                return super().complete(messages,tools)
        app=Application(self.snapshot,Fails(),self.root/'logs'); events=[]
        with self.assertRaises(AnswerError):app.ask({'question':'Explain','analysis_id':'a1'},on_event=events.append)
        self.assertEqual(events[-1]['type'],'model_failed')
        self.assertEqual(len(app.citations),1)
        self.assertTrue(any(e['type']=='tool_completed' for e in events))
    def test_http_events_arrive_before_model_finishes(self):
        release=threading.Event()
        class Waiting(Fake):
            def complete(self,messages,tools):
                if not self.requests:
                    if not release.wait(5):raise AnswerError('Test gate timed out')
                return super().complete(messages,tools)
        app=Application(self.snapshot,Waiting(),self.root/'logs');server=make_server(app,0)
        t=threading.Thread(target=server.serve_forever,daemon=True);t.start()
        conn=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=3)
        try:
            conn.request('POST','/api/ask',json.dumps({'question':'Explain','analysis_id':'a1'}),
                         {'Content-Type':'application/json','Accept':'application/x-ndjson','X-QA-Token':app.token})
            response=conn.getresponse();self.assertEqual(response.status,200)
            first=json.loads(response.readline());self.assertEqual(first['type'],'model_started')
            self.assertFalse(release.is_set())
            release.set();events=[json.loads(line) for line in response.read().splitlines()]
            self.assertEqual(events[-1]['type'],'answer')
            self.assertEqual(len(events[-1]['answer']['citations']),3)
        finally:
            release.set();conn.close();server.shutdown();server.server_close();t.join()
    def test_session_cannot_switch_analysis_and_logs_separate(self):
        with self.assertRaises(ValueError): Application(self.snapshot, Fake(), self.run / 'logs')
        app = Application(self.snapshot, Fake(), self.root / 'qa_logs')
        answer = app.ask({'question':'Explain', 'analysis_id':'a1'})
        self.assertTrue(answer['log_saved']); self.assertEqual(len(list((self.root/'qa_logs').glob('*.json'))), 1)
        with self.assertRaises(AnswerError): app.ask({'question':'Explain', 'analysis_id':'a2', 'session_id':answer['session_id']})
    def test_restart_restores_only_exact_snapshot_records(self):
        app=Application(self.snapshot,Fake(),self.root/'logs')
        answer=app.ask({'question':'Explain','analysis_id':'a1'})
        path=self.root/'logs'/(answer['answer_id']+'.json')
        old=json.loads(path.read_text());old.pop('process_trace');write(path,old)
        restored=Application(self.snapshot,Fake(),self.root/'logs')
        self.assertEqual(len(restored.saved_answers),1)
        self.assertEqual(restored.citations,answer['citations'])
        self.assertEqual(restored.sessions[answer['session_id']]['history'][0]['question'],'Explain')
        (self.run/'source.md').write_text('Changed source')
        changed=Application(Snapshot(self.run),Fake(),self.root/'logs')
        self.assertEqual(changed.saved_answers,[]);self.assertEqual(changed.citations,{})
    def test_provider_is_lazy_and_no_credential_fallback_on_custom_host(self):
        p = Transport(self.root / 'nonexistent_config', self.root / 'nonexistent_env')
        self.assertIsNone(p.client)
        write(self.root / 'provider.json', {'text_provider': {'base_url':'https://relay.example/v1','model':'model'}})
        with patch.dict('os.environ', {'EXPERT_QA_BASE_URL':'https://other.example/v1','TEXT_PROVIDER_API_KEY':'secret'}, clear=True):
            with self.assertRaises(AnswerError): Transport(self.root / 'provider.json')._initialize()
    def test_actual_sdk_serializes_native_tools_without_network(self):
        import httpx
        from openai import OpenAI
        captured = []
        def respond(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200, json={'id':'mock_completion', 'object':'chat.completion', 'created':0,
                'model':'mock-model', 'choices':[{'index':0, 'finish_reason':'tool_calls',
                'message':{'role':'assistant', **call('get_analysis', {})['message']}}],
                'usage':{'prompt_tokens':10,'completion_tokens':5,'total_tokens':15}})
        client = OpenAI(api_key='mock-not-a-key', base_url='https://offline.invalid/v1',
                        http_client=httpx.Client(transport=httpx.MockTransport(respond)))
        transport = Transport(self.root/'unused')
        transport.client=client; transport.model='mock-model'; transport.base_url='https://offline.invalid/v1'
        try:
            response=transport.complete([{'role':'user','content':'Explain'}], tools_schema())
            self.assertEqual(response['message']['tool_calls'][0]['function']['name'], 'get_analysis')
            self.assertEqual(response['usage']['total_tokens'],15)
            self.assertEqual(len(captured[0]['tools']),12)
            self.assertEqual(captured[0]['tool_choice'], 'required')
            self.assertFalse(captured[0]['parallel_tool_calls'])
            self.assertNotIn('temperature', captured[0])
            self.assertNotIn('response_format',captured[0])
        finally: client.close()
    def test_official_provider_does_not_reuse_relay_credential(self):
        config = self.root/'provider.json'
        write(config, {'text_provider': {'name': 'openai', 'base_url': 'https://api.openai.com/v1',
                                       'model': 'gpt-5.6-luna'}})
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'official-test-only',
                                     'TEXT_PROVIDER_API_KEY': 'relay-test-only'}, clear=True):
            transport = Transport(config)
            transport._initialize()
            try:
                self.assertEqual(transport.client.api_key, 'official-test-only')
                self.assertEqual(transport.model, 'gpt-5.6-luna')
            finally:
                transport.client.close()
    def test_luna_responses_replays_tool_and_opaque_context(self):
        import httpx
        from openai import OpenAI
        captured = []
        def respond(request):
            self.assertEqual(request.url.path, '/v1/responses')
            captured.append(json.loads(request.content))
            return httpx.Response(200, json={
                'id': 'resp_mock', 'object': 'response', 'created_at': 0,
                'model': 'gpt-5.6-luna', 'status': 'completed',
                'output': [
                    {'id': 'rs_mock', 'type': 'reasoning', 'summary': [], 'encrypted_content': 'opaque-test-only'},
                    {'id': 'fc_mock', 'type': 'function_call', 'call_id': 'call_mock',
                     'name': 'get_analysis', 'arguments': '{}', 'status': 'completed'}],
                'usage': {'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15,
                          'input_tokens_details': {'cached_tokens': 0},
                          'output_tokens_details': {'reasoning_tokens': 3}}})
        with OpenAI(api_key='mock', base_url='https://offline.invalid/v1',
                    http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
            transport = Transport(self.root/'unused')
            transport.client = client; transport.model = 'gpt-5.6-luna'
            messages = [{'role': 'user', 'content': 'Explain'}]
            result = transport.complete(messages, tools_schema(research=True))
            self.assertEqual(result['usage']['total_tokens'], 15)
            self.assertNotIn('opaque-test-only', json.dumps(result))
            messages += [{'role': 'assistant', **result['message']},
                         {'role': 'tool', 'tool_call_id': 'call_mock', 'content': '{"saved": true}'}]
            transport.complete(messages, tools_schema(research=True))
            payload = captured[1]
            self.assertFalse(payload['store'])
            self.assertEqual(payload['reasoning'], {'effort': 'medium'})
            self.assertNotIn('temperature', payload)
            self.assertEqual(payload['input'][1]['encrypted_content'], 'opaque-test-only')
            self.assertEqual(payload['input'][2]['call_id'], 'call_mock')
            self.assertEqual(payload['input'][3]['type'], 'function_call_output')
            self.assertTrue(payload['tools'][-1]['strict'])
            self.assertFalse(payload['tools'][0]['strict'])
    def test_http_and_citation_escaping(self):
        app = Application(self.snapshot, Fake(), self.root/'qa_logs'); server = make_server(app, 0)
        t = threading.Thread(target=server.serve_forever, daemon=True); t.start()
        def request(method, path, payload=None, headers=None):
            conn = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=5)
            conn.request(method, path, payload, headers or {}); r = conn.getresponse(); value=r.read(); conn.close(); return r.status,value
        try:
            status, english = request('GET','/')
            self.assertEqual(status, 200)
            self.assertIn(b'<html lang="en">', english)
            self.assertIn(b'Understand results. Trace the evidence.', english)
            self.assertIn(b'Recorded evidence', request('GET', '/app.js')[1])
            status, data = request('GET','/api/snapshot'); token=json.loads(data)['csrf_token']
            self.assertEqual(status,200)
            self.assertEqual(request('GET','/api/snapshot',headers={'Host':'evil.example'})[0],403)
            self.assertEqual(request('GET','/../../request.normalized.json')[0],404)
            body=json.dumps({'question':'Explain','analysis_id':'a1'})
            self.assertEqual(request('POST','/api/ask',body,{'Content-Type':'application/json'})[0],403)
            headers={'Content-Type':'application/json','X-QA-Token':token,'Origin':'https://evil.example'}
            self.assertEqual(request('POST','/api/ask',body,headers)[0],403)
            del headers['Origin']; status, value = request('POST','/api/ask',body,headers)
            self.assertEqual(status,200); answer=json.loads(value)
            cid=next(c for c,p in answer['citations'].items() if p['tool']=='get_source')
            status, page = request('GET','/citation/'+cid)
            self.assertEqual(status,200); self.assertIn(b'&lt;table&gt;', page); self.assertNotIn(b'<table>',page)
            self.assertIn(b'Back to results', page)
        finally:
            server.shutdown(); server.server_close(); t.join()

if __name__ == '__main__': unittest.main()
