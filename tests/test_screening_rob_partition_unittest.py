import tempfile
import unittest
import copy
import json
import hashlib
from pathlib import Path
from unittest.mock import patch

from extensions import screening_bias
from extensions.bias import SCHEMA, ROUTED_TOOL_SPECS, create_sensitivity_inputs, routed_tool, unassessed_record
from extensions.rob_resilience import assess_nonblocking
from extensions.runner import ExtensionRunner
from extensions.common import read, write


class ScreeningRoBRoutePartitionTests(unittest.TestCase):
    def setUp(self):
        # Routing is tested against a synthetic resource; no third-party guidance
        # or scientific interpretation is asserted by this unit test.
        resource = tempfile.TemporaryDirectory()
        self.addCleanup(resource.cleanup)
        root = Path(resource.name)
        guide = root / screening_bias.GUIDES['ROBINS-E'][0]
        guide.parent.mkdir(parents=True)
        guide.write_text('Synthetic routing fixture only. ' * 10)
        prompt = root / 'prompts/robins-e.txt'
        prompt.parent.mkdir()
        prompt.write_text('Synthetic model-input fixture only. ' * 10)
        def metadata(path):
            data = path.read_bytes()
            return {'path': path.relative_to(root).as_posix(), 'bytes': len(data),
                    'sha256': hashlib.sha256(data).hexdigest()}
        (root / 'catalog.json').write_text(json.dumps({'tools':[{
            'id':'robins-e', 'tool':'ROBINS-E', 'version':'2024-03-24',
            'guide':metadata(guide), 'prompt':metadata(prompt)}]}))
        patcher = patch.object(screening_bias, 'RESOURCE_ROOT', root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _request(self, directory):
        documents=[]
        for article_id in ('candidate_002', 'candidate_006', 'candidate_008', 'candidate_012',
                           'candidate_013', 'candidate_014', 'candidate_017'):
            source=Path(directory)/(article_id+'.md')
            source.write_text('# Methods\nA retrospective cohort report.\n', encoding='utf-8')
            documents.append({'article_id':article_id, 'study_id':'study_'+article_id.split('_')[1],
                              'paths':{'markdown':str(source)}})
        return {
            'case_id':'route_partition_case', 'documents':documents,
            'options':{'max_workers':2},
            'protocol':{'pico':{'intervention_or_exposure':'pandemic exposure', 'comparison':'pre-pandemic'},
                        'targets':[{'target_id':'mortality', 'outcome':'all-cause mortality'}]},
            'analysis_profile':{'primary_analysis_spec':{'target_id':'mortality'}},
        }

    @staticmethod
    def _screening():
        return {'decisions':[{'candidate_id':article_id, 'final_status':'Include'} for article_id in (
            'candidate_002', 'candidate_006', 'candidate_008', 'candidate_012',
            'candidate_013', 'candidate_014', 'candidate_017')]}

    @staticmethod
    def _routes():
        valid=lambda article_id: {'article_id':article_id, 'recommended_tool':'ROBINS-E',
                                  'tool_version':'2024-03-24', 'tool_variant':'',
                                  'question_type':'exposure_effect'}
        return {'recommendations':[
            valid('candidate_002'),
            {'article_id':'candidate_006', 'recommended_tool':None, 'question_type':'cross_sectional_period_comparison'},
            {'article_id':'candidate_008', 'recommended_tool':None, 'question_type':'unclear'},
            {'article_id':'candidate_012', 'recommended_tool':None, 'question_type':'cross_sectional_period_comparison'},
            valid('candidate_013'), valid('candidate_014'), valid('candidate_017'),
        ]}

    def test_partition_keeps_routable_articles_when_other_routes_are_uncovered(self):
        partition=screening_bias.partition_effective_routes({}, self._routes(),
            [row['candidate_id'] for row in self._screening()['decisions']])
        self.assertEqual(['candidate_002', 'candidate_013', 'candidate_014', 'candidate_017'],
                         [row['article_id'] for row in partition['routable']['recommendations']])
        self.assertEqual(['candidate_006', 'candidate_008', 'candidate_012'],
                         [row['article_id'] for row in partition['uncovered_routes']])
        self.assertTrue(all(row['reason']=='no_defensible_rob_tool_route'
                            for row in partition['uncovered_routes']))

    def test_assessment_submits_only_routable_articles_and_preserves_unknown_manual_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            request=self._request(temp)
            submitted=[]

            def fake_assess(req, routes, ledger, **_kwargs):
                submitted.extend(candidate['article_id'] for candidate in ledger['candidates'])
                by_article={row['article_id']:row for row in routes['recommendations']}
                return {'schema_version':SCHEMA, 'case_id':req['case_id'],
                        'assessments':[unassessed_record(req, candidate, routed_tool(by_article[candidate['article_id']]))
                                       for candidate in ledger['candidates']], 'failures':[]}

            with patch.object(screening_bias, 'assess_request', side_effect=fake_assess):
                result=screening_bias.assess_screened(request, self._screening(), self._routes(),
                                                       Path(temp)/'out', lambda _payload: self.fail('no provider call in this test'),
                                                       allow_partial=True)
            self.assertEqual(['candidate_002', 'candidate_013', 'candidate_014', 'candidate_017'], submitted)
            self.assertEqual(['candidate_006', 'candidate_008', 'candidate_012'],
                             [row['article_id'] for row in result['uncovered_routes']])
            manual=[row for row in result['assessments'] if row.get('routing_status')=='uncovered']
            self.assertEqual(['candidate_006', 'candidate_008', 'candidate_012'],
                             [row['article_id'] for row in manual])
            self.assertTrue(all(row['status']=='needs_assessment' and row['overall_judgment']=='unknown'
                                for row in manual))

    def test_nonblocking_completion_is_partial_not_stage_failed_for_uncovered_routes(self):
        with tempfile.TemporaryDirectory() as temp:
            request=self._request(temp)

            def fake_assess(req, routes, ledger, **_kwargs):
                by_article={row['article_id']:row for row in routes['recommendations']}
                return {'schema_version':SCHEMA, 'case_id':req['case_id'],
                        'assessments':[unassessed_record(req, candidate, routed_tool(by_article[candidate['article_id']]))
                                       for candidate in ledger['candidates']], 'failures':[]}

            with patch.object(screening_bias, 'assess_request', side_effect=fake_assess):
                result=assess_nonblocking(request, self._screening(), self._routes(), Path(temp)/'out', lambda _payload: None)
            self.assertEqual('partial', result['completion']['status'])
            self.assertEqual(3, result['completion']['uncovered'])
            self.assertEqual([], result['failures'])
            self.assertTrue(all(row['overall_judgment']=='unknown' for row in result['assessments']))

    def test_missing_prompt_stops_before_provider_and_records_failed_unknown(self):
        (screening_bias.RESOURCE_ROOT / 'prompts/robins-e.txt').unlink()
        with tempfile.TemporaryDirectory() as temp:
            request = self._request(temp)
            result = assess_nonblocking(request, self._screening(), self._routes(), Path(temp)/'out',
                lambda _payload: self.fail('Missing resources must not reach the provider'))
            self.assertEqual('failed', result['completion']['status'])
            self.assertTrue(all(a['overall_judgment']=='unknown' for a in result['assessments']))
            self.assertIn('resources/rob/README.md', result['failures'][0]['message'])

    def test_high_risk_sensitivity_excludes_only_completed_applicable_assessment(self):
        with tempfile.TemporaryDirectory() as temp:
            request=self._request(temp)
            request=copy.deepcopy(request)
            request['analysis_profile']['primary_analysis_spec'].update({
                'analysis_id':'primary', 'eligible_data_families':['effect'],
            })
            ledger={'candidates':[
                {'candidate_id':'evidence_high','article_id':'candidate_002','study_id':'study_002',
                 'target_id':'mortality','data_family':'effect'},
                {'candidate_id':'evidence_unknown','article_id':'candidate_006','study_id':'study_006',
                 'target_id':'mortality','data_family':'effect'},
            ]}
            tool={'name':'ROBINS-E','version':'2024-03-24','variant':''}
            domains=[{'domain_id':domain_id,'judgment':'high','rationale':'source-supported concern','evidence':[]}
                     for domain_id in ROUTED_TOOL_SPECS[('ROBINS-E','2024-03-24')]['domains']]
            high={'assessment_id':'draft_high','article_id':'candidate_002','study_id':'study_002',
                  'scope':{'candidate_id':'evidence_high','target_id':'mortality'},'tool':tool,'status':'draft',
                  'domains':domains,'overall_judgment':'high','overall_rationale':'At least one high-risk domain.',
                  'provenance':{'mode':'automated_draft'}}
            unknown=unassessed_record(request,ledger['candidates'][1])
            assessments={'schema_version':SCHEMA,'case_id':request['case_id'],'assessments':[high,unknown]}
            policy={'sensitivity_id':'exclude_high','analysis_id':'primary',
                    'tools':{'ROBINS-E':{'version':'2024-03-24','exclude_judgments':['high','very_high']}},
                    'unknown_policy':'keep','allow_automated_drafts':True}
            run=create_sensitivity_inputs(request,ledger,assessments,policy)['runs'][0]
            audit={row['candidate_id']:row for row in run['audit']}
            self.assertTrue(audit['evidence_high']['excluded'])
            self.assertFalse(audit['evidence_unknown']['excluded'])
            self.assertEqual('unknown_policy_keep',audit['evidence_unknown']['reason'])

    def test_resume_recovers_only_the_obsolete_stagewide_unknown_placeholder(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            robdir=out/'01_screening'; robdir.mkdir()
            write(robdir/'risk_of_bias_assessments.json',{
                'assessments':[{'status':'needs_assessment','overall_judgment':'unknown'}],
            })
            write(robdir/'risk_of_bias_failures.json',{
                'failures':[{'scope':'stage','message':'No defensible RoB tool route for included article: candidate_006'}],
            })
            write(robdir/'risk_of_bias_complete.json',{'status':'failed'})
            runner=object.__new__(ExtensionRunner)
            runner._resume=True
            runner.out=out
            runner.state={'stages':{'screening_rob':{'status':'completed'}}}
            runner.save=lambda _status: None
            self.assertTrue(runner._recover_pre_partition_screening_rob())
            self.assertNotIn('screening_rob', runner.state['stages'])
            self.assertTrue((robdir/'risk_of_bias_route_partition_recovery.json').is_file())
            record=read(robdir/'risk_of_bias_route_partition_recovery.json')
            archive=robdir/record['archive']
            self.assertTrue((archive/'risk_of_bias_assessments.json').is_file())
            self.assertFalse((robdir/'risk_of_bias_assessments.json').exists())


if __name__ == '__main__':
    unittest.main()
