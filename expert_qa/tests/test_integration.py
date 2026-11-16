import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from expert_qa.integration import register_run, install_entry, after_pipeline, ensure_service, project_root
from expert_qa.library import Library
from expert_qa.server import make_server
from expert_qa.store import FILES, digest
from expert_qa.tests.test_expert_qa import fixture, write, Fake

class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.run=fixture(self.root);self.catalog=self.root/'library'
        write(self.run/FILES['manifest'],{'status':'completed','stages':{'analysis':{'status':'completed'}}})
        (self.run/'dashboard.html').write_text('<html><body>Existing dashboard</body></html>')
    def tearDown(self):self.temp.cleanup()
    def test_postrun_registration_is_idempotent_and_science_unchanged(self):
        before={p:digest((self.run/p).read_bytes()) for p in FILES.values() if (self.run/p).exists()}
        record=register_run(self.run,env_file=self.root/'private.env',data_dir=self.catalog)
        install_entry(self.run,record,data_dir=self.catalog)
        install_entry(self.run,record,data_dir=self.catalog)
        page=(self.run/'dashboard.html').read_text()
        self.assertEqual(page.count('id="result-guide-entry"'),1)
        self.assertEqual((self.run/'dashboard.pre_result_guide.html').read_text(),'<html><body>Existing dashboard</body></html>')
        self.assertTrue((self.run/'Open_results.command').is_file())
        after={p:digest((self.run/p).read_bytes()) for p in before}
        self.assertEqual(before,after)
        self.assertEqual(len(list((self.catalog/'runs').glob('*.json'))),1)
    def test_running_result_not_published(self):
        write(self.run/FILES['manifest'],{'status':'running'})
        self.assertIsNone(register_run(self.run,data_dir=self.catalog))
        self.assertFalse(self.catalog.exists())
    def test_registration_preserves_existing_bookmark_after_path_move(self):
        record = register_run(self.run, data_dir=self.catalog)
        moved = self.root/'moved_run'
        self.run.rename(moved)
        self.run.symlink_to(moved, target_is_directory=True)
        updated = register_run(moved, data_dir=self.catalog)
        self.assertEqual(updated['id'], record['id'])
        self.assertEqual(len(list((self.catalog/'runs').glob('*.json'))), 1)
    def test_package_keeps_its_own_config_when_other_project_has_a_venv(self):
        home = self.root/'home'
        write(home/'config/provider.json', {})
        project = self.root/'actual_project'
        write(project/'config/provider.json', {})
        executable = project/'.venv/bin/python'
        executable.parent.mkdir(parents=True)
        executable.touch()
        self.assertEqual(project_root(home, project/'.venv'), home.resolve())
    def test_library_uses_persistent_qa_only_provider(self):
        record = register_run(self.run, env_file=self.root/'local.env', data_dir=self.catalog)
        config = self.catalog/'provider.json'
        write(config, {'text_provider': {'name': 'openai', 'model': 'gpt-5.6-luna',
                                       'base_url': 'https://api.openai.com/v1'}})
        lib = Library(self.catalog, log_dir=self.root/'logs')
        app = lib.get(record['id'])
        self.assertEqual(app.agent.transport.config, config.resolve())
        self.assertEqual(str(app.agent.transport.env_file), str((self.root/'local.env').resolve()))
    def test_service_reuses_healthy_process(self):
        with patch('expert_qa.integration.healthy',return_value=True),patch('expert_qa.integration.subprocess.Popen') as start:
            self.assertTrue(ensure_service(self.catalog))
            start.assert_not_called()
    def test_service_waits_for_startup_and_reports_child_failure(self):
        for stopped, expected in [(None,True),(1,False)]:
            with self.subTest(stopped=stopped),patch('expert_qa.integration.healthy',side_effect=[False,False,False,True]),patch('socket.socket'),patch('expert_qa.integration.subprocess.Popen') as start,patch('time.sleep'):
                start.return_value.poll.return_value=stopped
                self.assertEqual(ensure_service(self.catalog),expected)
    def test_hook_nonfatal_and_no_analysis_no_service(self):
        with patch('expert_qa.integration.register_run',return_value=None),patch('expert_qa.integration.ensure_service') as service:
            after_pipeline(['run','--output',str(self.run)])
            service.assert_not_called()
        with patch('expert_qa.integration.register_run',side_effect=ValueError('bad result')):
            after_pipeline(['retry-analysis','--output',str(self.run)])
    def test_hook_wires_final_output_and_environment(self):
        with patch('expert_qa.integration.register_run',return_value={'id':'test'}) as register,patch('expert_qa.integration.install_entry',return_value='local') as install,patch('expert_qa.integration.ensure_service',return_value=True):
            after_pipeline(['run','--output',str(self.run),'--env-file','private.env'])
            register.assert_called_once_with(self.run.resolve(),env_file='private.env')
            install.assert_called_once()
    def test_multi_run_routes_and_tokens_are_isolated(self):
        first=register_run(self.run,data_dir=self.catalog)
        other=self.root/'second';other.mkdir();second_run=fixture(other)
        write(second_run/FILES['manifest'],{'status':'completed'})
        second=register_run(second_run,data_dir=self.catalog)
        lib=Library(self.catalog,transport_factory=lambda row:Fake(),log_dir=self.root/'qa_logs')
        self.assertEqual(len(lib.listing()),2)
        self.assertNotIn('run',lib.listing()[0]);self.assertNotIn('env_file',lib.listing()[0])
        server=make_server(port=0,library=lib)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        def req(method,path,data=None,token=None):
            conn=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=5)
            headers={'Content-Type':'application/json'}
            if token:headers['X-QA-Token']=token
            conn.request(method,path,json.dumps(data) if data else None,headers)
            response=conn.getresponse();body=response.read();conn.close();return response.status,body
        try:
            self.assertEqual(req('GET','/')[0],200)
            self.assertEqual(len(json.loads(req('GET','/api/catalog')[1])['runs']),2)
            base='/r/'+first['id']+'/'
            self.assertEqual(req('GET',base)[0],200)
            a=json.loads(req('GET',base+'api/snapshot')[1])
            b=json.loads(req('GET','/r/'+second['id']+'/api/snapshot')[1])
            self.assertNotEqual(a['csrf_token'],b['csrf_token'])
            self.assertEqual(req('POST',base+'api/ask',{'question':'Explain','analysis_id':'a1'},b['csrf_token'])[0],403)
            status,data=req('POST',base+'api/ask',{'question':'Explain','analysis_id':'a1'},a['csrf_token'])
            self.assertEqual(status,200)
            answer=json.loads(data);cid=next(iter(answer['citations']))
            self.assertEqual(req('GET',base+'citation/'+cid)[0],200)
            self.assertEqual(req('GET','/r/'+second['id']+'/citation/'+cid)[0],404)
            self.assertEqual(req('GET','/r/../../etc/passwd')[0],409)
            write(self.run/FILES['manifest'],{'status':'running'})
            self.assertEqual(req('GET',base+'api/snapshot')[0],409)
        finally:server.shutdown();server.server_close();thread.join()

if __name__=='__main__':unittest.main()
