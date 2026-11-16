import copy
import json
import unittest
import test_expert_qa as base
from test_expert_qa import call
from expert_qa.agent import Agent, tools_schema
from expert_qa.research import flatten_research, is_research_question


class ResearchTests(unittest.TestCase):
    setUp = base.Tests.setUp
    tearDown = base.Tests.tearDown
    saved_omission = base.Tests.saved_omission
    def test_research_context_is_scoped_and_does_not_mutate(self):
        self.saved_omission()
        before = copy.deepcopy(self.snapshot.raw)
        result = self.snapshot.query('get_research_context', {}, 'a1')
        self.assertEqual(result['pooled_effects_total'], 1)
        self.assertEqual(result['robustness']['saved_leave_one_out_count'], 1)
        self.assertEqual(result['robustness']['leave_one_out_intervals_containing_null'], [])
        self.assertEqual(self.snapshot.raw, before)
        self.assertEqual(self.snapshot.query('get_research_context', {}, 'a2')['pooled_effects_total'], 0)

    def test_research_schema_and_routing(self):
        for text in ['Discuss potential explanations and future research', 'Please summarize the results', 'What are next steps?']:
            self.assertTrue(is_research_question(text))
        self.assertFalse(is_research_question('What is the RR of study_016?'))
        function = tools_schema(research=True)[-1]['function']
        self.assertEqual(set(function['parameters']['required']), {'overview', 'hypotheses', 'next_steps', 'limitations'})

    def test_hypothesis_retrieval_moves_from_extractions_to_source(self):
        class ReachedSource(Exception):
            pass
        class Model:
            def __init__(self): self.n = 0
            def complete(self, messages, tools):
                self.n += 1
                if self.n == 1:
                    return call('get_research_context', {})
                if self.n <= 3:
                    return call('get_extraction', {'evidence_id': 'e1'})
                assert [t['function']['name'] for t in tools] == ['get_source']
                raise ReachedSource()
        with self.assertRaises(ReachedSource):
            Agent(self.snapshot, Model()).answer('Discuss potential explanations', 'a1')

    def test_research_references_structure_and_server_numeric_facts(self):
        payload = {
            'overview': [{'kind': 'inference', 'text': 'Saved records are insufficient to establish causation.', 'citation_ids': ['E1']}],
            'hypotheses': [{'hypothesis': 'Possible baseline selection difference', 'supporting_observation': 'Recorded subgroup difference', 'supporting_quote': 'Original result RR 1.2',
                            'counter_evidence': 'Not identified in retrieved passages', 'uncertainty': 'Incomplete baseline covariates',
                            'verification': 'Inspect primary baseline table', 'citation_ids': ['E2']}],
            'next_steps': [{'priority': 1, 'action': 'Check baseline covariates', 'rationale': 'Missing covariate details',
                            'possible_implications': 'May attenuate or preserve the current interpretation', 'citation_ids': ['E1']}],
            'limitations': 'Proposed checks have not been executed.'}
        citations = {'c1': {'tool': 'get_research_context', 'arguments': {}, 'offset': 0},
                     'c2': {'tool': 'get_source', 'arguments': {}, 'offset': 0,
                            'text': json.dumps({'text': '2: Original result RR 1.2 (95% CI 1.1-1.3).'})}}
        packet = {'statistics': {'effect_type': 'RR', 'pooled_effect': 1.1, 'ci_lower': 1.0016,
                                'ci_upper': 1.2, 'k': 12, 'i_squared': .2625}}
        result = flatten_research(copy.deepcopy(payload), {'E1': 'c1', 'E2': 'c2'}, citations, packet)
        self.assertIn('I² 0.2625%', result[0]['text'])
        self.assertEqual(result[0]['citation_ids'], ['c1'])
        bad = copy.deepcopy(payload); del bad['hypotheses'][0]['counter_evidence']
        with self.assertRaises(ValueError): flatten_research(bad, {'E1': 'c1', 'E2': 'c2'}, citations, packet)
        bad = copy.deepcopy(payload); bad['next_steps'][0]['citation_ids'] = ['made_up']
        with self.assertRaises(ValueError): flatten_research(bad, {'E1': 'c1', 'E2': 'c2'}, citations, packet)
        bad = copy.deepcopy(payload); bad['hypotheses'][0]['supporting_quote'] = 'This fabricated passage is not present'
        with self.assertRaises(ValueError): flatten_research(bad, {'E1': 'c1', 'E2': 'c2'}, citations, packet)
        snapshot = self.snapshot
        class Model:
            def __init__(self): self.n = 0
            def metadata(self): return {'provider_host': 'mock'}
            def complete(self, messages, tools):
                self.n += 1
                if self.n == 1:
                    assert [t['function']['name'] for t in tools] == ['get_research_context']
                    return call('get_research_context', {})
                if self.n == 2:
                    return call('get_source', {'article_id': 'paper1', 'line': 1, 'line_count': 3})
                return call('finish_answer', copy.deepcopy(payload))
        answer = Agent(snapshot, Model()).answer('Discuss potential explanations and future research', 'a1')
        self.assertEqual(answer['mode'], 'research')
        self.assertEqual(len(answer['research']['hypotheses']), 1)
        self.assertIn('Verification step', answer['claims'][1]['text'])
