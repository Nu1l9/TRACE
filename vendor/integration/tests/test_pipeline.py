import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from trace_runtime.contracts import *
from trace_runtime.pipeline import Runner,runtime_environment,replay
ROOT=Path(__file__).resolve().parents[1];EX=ROOT/'examples'

class PipelineTests(unittest.TestCase):
    def setUp(self):self.r=normalize_request(read(EX/'request.json'),EX)
    def fixture_stage(self,name,command,outputs):
        values={'screening':[read(EX/'screening.json'),{'recommendations':[]}], 'extraction':[read(EX/'submission.json'),{'article_runs':[],'failed_article_ids':[]}], 'review':[read(EX/'review.json')], 'analysis':[{'status':'completed','analyses':[]},{'status':'completed'}]}
        for p,v in zip(outputs,values.get(name,[])):write(p,v)
    def test_orchestrates_brain_output(self):
        with tempfile.TemporaryDirectory() as d:
            runner=Runner(self.r,Path(d)/'run',os.environ.copy())
            with patch.object(runner,'stage',side_effect=self.fixture_stage):state=runner.run()
            self.assertEqual(state['status'],'completed')
            result=read(Path(d)/'run/brain_result.json');self.assertEqual(result['analysis_result_path'],'05_analysis/analysis_result.json')
            self.assertEqual(read(Path(d)/'run/04_analysis_input/analysis_evidence_candidates.json')['candidates'][0]['candidate_id'],'ev_1')
    def test_unsure_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            runner=Runner(self.r,Path(d)/'run',os.environ.copy());calls=[]
            def stage(name,command,outputs):
                calls.append(name);self.fixture_stage(name,command,outputs)
                if name=='screening':
                    s=read(outputs[0]);s['decisions'][0]['final_status']='Unsure';write(outputs[0],s)
            with patch.object(runner,'stage',side_effect=stage):state=runner.run()
            self.assertEqual(state['status'],'needs_screening_review');self.assertEqual(calls,['screening'])
    def test_empty_extraction_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            runner=Runner(self.r,Path(d)/'run',os.environ.copy());calls=[]
            def stage(name,command,outputs):
                calls.append(name);self.fixture_stage(name,command,outputs)
                if name=='extraction':write(outputs[0],{'case_id':'demo_case','predictions':[]})
            with patch.object(runner,'stage',side_effect=stage):state=runner.run()
            self.assertEqual(state['status'],'no_extracted_evidence');self.assertEqual(calls,['screening','extraction'])
    def test_failure_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            runner=Runner(self.r,Path(d)/'run',os.environ.copy())
            with patch.object(runner,'stage',side_effect=ContractError('intentional offline failure')):
                with self.assertRaises(ContractError):runner.run()
            self.assertEqual(read(runner.manifest)['status'],'failed')
    def test_checkpoint_resume_and_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            runner=Runner(self.r,Path(d)/'run',os.environ.copy());output=runner.out/'artifact.json'
            cmd=[sys.executable,'-c',"import pathlib;pathlib.Path(__import__('sys').argv[1]).write_text('{}')",str(output)]
            runner.stage('test',cmd,[output]);r2=Runner(self.r,runner.out,os.environ.copy(),resume=True)
            r2.stage('test',[sys.executable,'-c','raise Exception()'],[output]);output.write_text('changed')
            with self.assertRaisesRegex(ContractError,'checkpoint changed'):r2.stage('test',cmd,[output])
    def test_changed_request_resume_refused(self):
        with tempfile.TemporaryDirectory() as d:
            runner=Runner(self.r,Path(d)/'run',os.environ.copy());self.r['topic']='different'
            with self.assertRaisesRegex(ContractError,'changed'):Runner(self.r,runner.out,os.environ.copy(),resume=True)
    def test_provider_isolation(self):
        self.r['options']['provider']='openai_compatible'
        with patch.dict(os.environ,{'PIPELINE_LLM_API_KEY':'fixture','PIPELINE_LLM_BASE_URL':'https://provider.example/v1'},clear=True):env=runtime_environment(self.r)
        self.assertEqual(env['COMPATIBLE_API_KEY'],'fixture');self.assertEqual(env['ANALYSIS_LLM_API_KEY'],'fixture');self.assertNotIn('ANALYSIS_EMBEDDING_API_KEY',env)
    def test_actual_frozen_loaders_baseline_and_typed(self):
        for version in ('baseline','typed'):
            with self.subTest(version=version),tempfile.TemporaryDirectory() as d:
                self.r['options']['extraction_runtime']=version
                result=replay(self.r,read(EX/'screening.json'),read(EX/'submission.json'),read(EX/'review.json'),Path(d)/'replay',runtime_environment(self.r))
                self.assertEqual(result['status'],'contracts_valid');self.assertFalse(result['statistical_analysis_executed'])
    def test_actual_miners_fulltext_screening(self):
        sys.path.insert(0,str(ROOT/'vendor/miners'))
        from miners.case_screening_rob import FullTextProtocolScreener
        prompts=[];responses=iter(['{"status":"Unsure","reason":"need full text"}','{"status":"Include","reason":"eligible"}'])
        def callback(prompt):prompts.append(prompt);return next(responses)
        s=FullTextProtocolScreener(callback)
        result=s.screen('article_a','Example',{'page_1':'Example\nAbstract\nTest abstract.\nMethods\nIntervention versus comparator.'},screening_protocol(self.r))
        self.assertEqual(result['final_status'],'Include');self.assertEqual(len(prompts),2);self.assertIn('comparator',prompts[0])
    def test_partial_extraction_is_not_silently_pooled(self):
        with tempfile.TemporaryDirectory() as d:
            runner=Runner(self.r,Path(d)/'run',os.environ.copy());calls=[]
            def stage(name,command,outputs):
                calls.append(name);self.fixture_stage(name,command,outputs)
                if name=='extraction':
                    data=read(outputs[0]);data['predictions'].pop();write(outputs[0],data)
            with patch.object(runner,'stage',side_effect=stage):state=runner.run()
            self.assertEqual(state['status'],'needs_extraction_review');self.assertNotIn('analysis',calls)
