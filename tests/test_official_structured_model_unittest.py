import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = (ROOT/'vendor/integration/vendor/evidence_extraction/'
            'system/llm_endpoint.py')


def load_endpoint():
    spec = importlib.util.spec_from_file_location('current_test_llm_endpoint', ENDPOINT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OfficialStructuredModelTests(unittest.TestCase):
    def test_official_gpt5_tool_model_disables_reasoning(self):
        endpoint = load_endpoint()
        captured = {}
        endpoint.ChatOpenAI = lambda **kwargs: captured.update(kwargs) or kwargs
        with patch.dict(os.environ, {
            'META_AGENT_LLM_PROVIDER': 'official_openai',
            'OPENAI_API_KEY': 'test-key',
            'OFFICIAL_OPENAI_MODEL': 'gpt-5.6-luna',
            'OFFICIAL_OPENAI_BASE_URL': 'https://api.openai.com/v1',
        }, clear=False):
            endpoint.chat_model('ignored', max_retries=0)
        self.assertEqual('none', captured['reasoning_effort'])
        self.assertNotIn('temperature', captured)

    def test_compatible_primary_keeps_existing_reasoning_parameters(self):
        endpoint = load_endpoint()
        captured = {}
        endpoint.ChatOpenAI = lambda **kwargs: captured.update(kwargs) or kwargs
        endpoint.httpx.Client = lambda **kwargs: object()
        with patch.dict(os.environ, {
            'META_AGENT_LLM_PROVIDER': 'openai_compatible',
            'META_AGENT_COMPATIBLE_STRICT': '1',
            'COMPATIBLE_API_KEY': 'test-key',
            'META_AGENT_COMPATIBLE_MODEL_OVERRIDE': 'gpt-5.6-luna',
            'META_AGENT_COMPATIBLE_BASE_URL_OVERRIDE': 'https://example.invalid/v1',
        }, clear=False):
            endpoint.chat_model('ignored', max_retries=0)
        self.assertNotIn('reasoning_effort', captured)


if __name__ == '__main__':
    unittest.main()
