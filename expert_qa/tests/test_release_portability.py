"""Release integration must keep endpoint selection and checkout paths explicit."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from expert_qa.integration import default_data_dir, install_entry, project_root, register_run
from expert_qa.provider import Transport
from expert_qa.tests.test_expert_qa import fixture, write


class ReleasePortabilityTests(unittest.TestCase):
    def test_text_provider_overrides_and_explicit_env_file_take_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'provider.json'
            write(config, {'text_provider': {'base_url': 'https://config.invalid/v1', 'model': 'config'}})
            env = root / 'keys.env'
            env.write_text('TEXT_PROVIDER_API_KEY=fixture\nTEXT_PROVIDER_BASE_URL=https://selected.invalid/v1\nTEXT_PROVIDER_MODEL=selected\n')
            with patch.dict(os.environ, {'TEXT_PROVIDER_MODEL': 'stale'}, clear=True), patch('openai.OpenAI') as client:
                transport = Transport(config, env)
                transport._initialize()
                self.assertEqual(transport.base_url, 'https://selected.invalid/v1')
                self.assertEqual(transport.model, 'selected')
                self.assertEqual(client.call_args.kwargs['api_key'], 'fixture')

    def test_launcher_uses_current_python_and_does_not_require_a_checkout_venv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = fixture(root)
            write(run / 'pipeline_manifest.json', {'status': 'completed'})
            record = register_run(run, data_dir=root / 'library')
            with patch.object(sys, 'executable', '/custom/env/bin/python3'):
                install_entry(run, record, data_dir=root / 'library')
            command = (run / 'Open_results.command').read_text()
            self.assertIn('/custom/env/bin/python3', command)
            self.assertNotIn('.venv/bin/python', command)
            self.assertNotRegex(command, r'trace_' + r'v[0-9]+')

    def test_data_directory_can_be_explicitly_selected(self):
        with patch.dict(os.environ, {'TRACE_QA_DATA_DIR': '~/qa-fixture'}, clear=True):
            self.assertEqual(default_data_dir(), Path('~/qa-fixture').expanduser())

    def test_checkout_configuration_never_comes_from_an_unrelated_environment(self):
        self.assertEqual(project_root('/tmp/published-checkout', '/tmp/unrelated/.venv'), Path('/tmp/published-checkout').resolve())


if __name__ == '__main__':
    unittest.main()
