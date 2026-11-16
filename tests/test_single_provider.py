"""Retries never select another endpoint or borrow embedding credentials."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from extensions import provider, resilient_extraction, transport
ROOT = Path(__file__).resolve().parents[1]

class SingleProviderTests(unittest.TestCase):
    def settings(self, name='openai_compatible'):
        return {'TEXT_PROVIDER_NAME':name,'TEXT_PROVIDER_API_KEY':'text-secret',
                'TEXT_PROVIDER_BASE_URL':'https://text.example.invalid/v1','TEXT_PROVIDER_MODEL':'text-model',
                'EMBEDDING_PROVIDER_NAME':'official_openai','EMBEDDING_PROVIDER_API_KEY':'embedding-secret',
                'EMBEDDING_PROVIDER_BASE_URL':'https://api.openai.com/v1','EMBEDDING_PROVIDER_MODEL':'embedding-model'}

    def test_example_config_has_only_explicit_routes(self):
        cfg=provider.selection_values(ROOT/'config/provider.example.json')
        self.assertEqual(set(k.split('_')[0] for k in cfg),{'TEXT','EMBEDDING'})
        self.assertEqual(cfg['TEXT_PROVIDER_NAME'],'official_openai')
        self.assertEqual(cfg['EMBEDDING_PROVIDER_NAME'],'official_openai')
        self.assertEqual(cfg['TEXT_PROVIDER_BASE_URL'],'https://api.openai.com/v1')
        self.assertEqual(cfg['EMBEDDING_PROVIDER_BASE_URL'],'https://api.openai.com/v1')
        with patch.dict(os.environ,{},clear=True), patch.object(provider,'_load',return_value={**cfg,'OPENAI_API_KEY':'official-key'}), patch.object(provider,'require_local_documents'):
            _,env,prof=provider.runtime({'options':{'model':'unused','provider':'official_openai'},'documents':[]})
        self.assertEqual(env['OPENAI_API_KEY'],'official-key')
        self.assertEqual(env['TEXT_PROVIDER_API_KEY'],'official-key')
        self.assertEqual(env['ANALYSIS_EMBEDDING_API_KEY'],'official-key')
        self.assertNotIn('official-key',json.dumps(provider.audit(prof)))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'provider.json'
            raw=json.loads((ROOT/'config/provider.example.json').read_text())
            raw['unexpected_alternate']={};p.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError,'obsolete alternate routes'):provider.selection_values(p)

    def test_runtime_isolates_text_and_embedding_credentials(self):
        for name in ('official_openai','openai_compatible'):
            with self.subTest(name=name), patch.dict(os.environ,{},clear=True), patch.object(provider,'_load',return_value=self.settings(name)), patch.object(provider,'require_local_documents'):
                derived,env,profile=provider.runtime({'options':{'model':'unused','provider':'official_openai'},'documents':[]})
            self.assertEqual(derived['options']['provider'],name)
            for k in ('PIPELINE_LLM_API_KEY','ANALYSIS_LLM_API_KEY','BIAS_API_KEY','BRAIN_API_KEY'):self.assertEqual(env[k],'text-secret')
            self.assertEqual(env['ANALYSIS_EMBEDDING_API_KEY'],'embedding-secret')
            self.assertEqual(env.get('OPENAI_API_KEY'), 'text-secret' if name=='official_openai' else None)
            self.assertNotIn('secret',json.dumps(provider.audit(profile)))

    def test_compatible_route_requires_its_own_url(self):
        env=self.settings();env.pop('TEXT_PROVIDER_BASE_URL')
        with self.assertRaisesRegex(ValueError,'TEXT_PROVIDER_BASE_URL'):provider._profile(env)
        env=self.settings();env['EMBEDDING_PROVIDER_NAME']='openai_compatible';env.pop('EMBEDDING_PROVIDER_BASE_URL')
        with self.assertRaisesRegex(ValueError,'EMBEDDING_PROVIDER_BASE_URL'):provider._profile(env)

    def test_text_retry_exhaustion_and_auth_keep_same_callable(self):
        for status,expected in ((503,3),(401,1),(524,3)):
            error=type('ProviderError',(Exception,),{'status_code':status})();call=Mock(side_effect=error)
            with self.assertRaises(type(error)):transport.retry_text_call(call,attempts=3,base_delay=0,label='test',sleep=lambda _:None)
            self.assertEqual(call.call_count,expected)

    def test_extraction_outage_saves_failure(self):
        error=type('ProviderError',(Exception,),{'status_code':524})();call=Mock(side_effect=error);failures=[]
        with self.assertRaises(type(error)):
            resilient_extraction._retry_call(call,article_id='article',stage='extract',request={'options':{'max_retries':2}},on_failure=lambda n,e:failures.append((n,e)))
        self.assertEqual(call.call_count,1);self.assertEqual(len(failures),1)

    def test_endpoint_does_not_borrow_other_route_credentials(self):
        for path in (ROOT/'vendor').rglob('llm_endpoint.py'):
            spec=importlib.util.spec_from_file_location('single_endpoint',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            with patch.dict(os.environ,{'OPENAI_API_KEY':'wrong-key'},clear=True):
                with self.assertRaises(RuntimeError):module._get_api_key('openai_compatible')
                with self.assertRaises(RuntimeError):module._get_base_url('openai_compatible')

    def test_typed_standalone_retries_one_provider(self):
        path=ROOT/'vendor/integration/vendor/evidence_extraction/system/runtime/typed/run.py'
        import ast
        node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_run_article_with_retries')
        context={'Any':object,'Path':Path,'os':os,'time':Mock(),'traceback':__import__('traceback'),
                 'datetime':__import__('datetime').datetime,'timezone':__import__('datetime').timezone,
                 'shutil':__import__('shutil'),'_write_json':Mock(),'run_article':Mock(side_effect=RuntimeError('upstream failed'))}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),context)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{'META_AGENT_LLM_PROVIDER':'openai_compatible'},clear=True):
            context['_run_article_with_retries'](object(),'a',Path(tmp),False,2,0)
        self.assertEqual(context['run_article'].call_count,3)
        self.assertEqual({c.kwargs['provider'] for c in context['run_article'].call_args_list},{'openai_compatible'})

if __name__=='__main__':unittest.main()
