import unittest
from types import SimpleNamespace
from unittest.mock import patch, Mock

from extensions.transport import json_callback
from extensions.settings import load_settings


class CallbackTransportTests(unittest.TestCase):
    def test_brain_alias_loop_is_opt_in(self):
        self.assertFalse(load_settings()['graph']['brain_loop'])

    def test_brain_and_bias_errors_stay_on_selected_provider(self):
        class GatewayTimeout(Exception):
            status_code = 524
        for purpose in ('brain', 'bias'):
            with self.subTest(purpose=purpose):
                client = Mock()
                client.chat.completions.create.side_effect = GatewayTimeout()
                env = dict(TEXT_PROVIDER_API_KEY='text-key', TEXT_PROVIDER_BASE_URL='https://example.invalid/v1',
                    TRACE_TEXT_PRIMARY_ATTEMPTS='3')
                with patch('openai.OpenAI', return_value=client) as factory, \
                     patch('extensions.transport.time.sleep'), \
                     patch('extensions.rob_schema.provider_format', return_value={'type':'json_schema'}):
                    with self.assertRaises(GatewayTimeout):
                        json_callback(env, purpose=purpose)({'queries':[]})
                factory.assert_called_once()
                self.assertEqual(factory.call_args.kwargs['base_url'], 'https://example.invalid/v1')
                self.assertEqual(factory.call_args.kwargs['api_key'], 'text-key')
                self.assertEqual(3, client.chat.completions.create.call_count)

    def test_official_primary_and_explicit_timeout(self):
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content='{}'))])
        with patch('openai.OpenAI', return_value=client) as factory:
            json_callback({'TEXT_PROVIDER_API_KEY':'test', 'TEXT_PROVIDER_NAME':'official_openai',
                'TEXT_PROVIDER_BASE_URL':'https://api.openai.com/v1', 'BRAIN_TIMEOUT_SECONDS':'240'})({})
        self.assertNotIn('temperature', client.chat.completions.create.call_args.kwargs)
        self.assertEqual(240, factory.call_args.kwargs['timeout'])
