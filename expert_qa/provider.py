"""Lazy text-only provider adapter. Never imports pipeline environment loaders."""
import os
import json
from pathlib import Path
from urllib.parse import urlparse
from .agent import AnswerError

class Transport:
    def __init__(self, config, env_file=None):
        self.config, self.env_file, self.client = Path(config), env_file, None
        self.model = None; self.base_url = None
        self._response_items = {}

    def _initialize(self):
        if self.client is not None:
            return
        # Credentials are read only when an actual question is submitted, never at preview.
        env = dict(os.environ)
        if self.env_file:
            from dotenv import dotenv_values
            env.update(dotenv_values(self.env_file, interpolate=False))
        profile = json.loads(self.config.read_text())['text_provider']
        base = env.get('EXPERT_QA_BASE_URL') or env.get('TEXT_PROVIDER_BASE_URL') or profile['base_url']
        model = env.get('EXPERT_QA_MODEL') or env.get('TEXT_PROVIDER_MODEL') or profile['model']
        parsed = urlparse(base)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.query:
            raise AnswerError('The Q&A provider must use an HTTPS URL without credential parameters.')
        key = env.get('EXPERT_QA_API_KEY')
        if not key and not env.get('EXPERT_QA_BASE_URL'):
            # Explicit credentials are used only with the selected endpoint.
            if parsed.hostname == 'api.openai.com':
                key = env.get('OPENAI_API_KEY') or env.get('TEXT_PROVIDER_API_KEY')
            else:
                key = env.get('TEXT_PROVIDER_API_KEY')
        if not key:
            raise AnswerError('No Q&A API key found. Set EXPERT_QA_API_KEY or configure a text provider key.')
        from openai import OpenAI
        self.client = OpenAI(api_key=key, base_url=base, timeout=60, max_retries=0)
        self.model, self.base_url = model, base

    def metadata(self):
        return {'model': self.model, 'provider_host': urlparse(self.base_url or '').hostname,
                'automatic_fallback': False,
                **({'endpoint': 'responses', 'reasoning_effort': 'medium'}
                   if self.model == 'gpt-5.6-luna' else {})}

    def _complete_responses(self, messages, tools):
        """Replay opaque reasoning only in memory, never in answer/audit logs."""
        if not any(m.get('tool_calls') for m in messages):
            self._response_items.clear()
        items = []
        for message in messages:
            calls = message.get('tool_calls') or []
            if calls:
                saved = self._response_items.get(calls[0]['id'])
                if saved is not None:
                    items.extend(saved)
                else:
                    if message.get('content'):
                        items.append({'role': 'assistant', 'content': message['content']})
                    items.extend({'type': 'function_call', 'call_id': c['id'],
                                  'name': c['function']['name'], 'arguments': c['function']['arguments']}
                                 for c in calls)
            elif message['role'] == 'tool':
                items.append({'type': 'function_call_output', 'call_id': message['tool_call_id'],
                              'output': message['content']})
            else:
                items.append({'role': message['role'], 'content': message.get('content') or ''})
        native_tools = [{'type': 'function', **t['function'],
                         'strict': t['function'].get('strict', False)} for t in tools]
        response = self.client.responses.create(
            model=self.model, input=items, tools=native_tools, tool_choice='required',
            parallel_tool_calls=False, reasoning={'effort': 'medium'}, max_output_tokens=12000,
            store=False)
        if response.status != 'completed':
            raise AnswerError('The model did not complete this tool response. Try again; no research conclusion was generated.')
        calls = [r for r in response.output if r.type == 'function_call']
        if calls:
            self._response_items[calls[0].call_id] = [r.model_dump(exclude_none=True) for r in response.output]
        u = response.usage
        return {'message': {'content': response.output_text or None, 'tool_calls': [
                    {'id': r.call_id, 'type': 'function',
                     'function': {'name': r.name, 'arguments': r.arguments}} for r in calls]},
                'usage': {'prompt_tokens': u.input_tokens, 'completion_tokens': u.output_tokens,
                          'total_tokens': u.total_tokens,
                          'completion_tokens_details': u.output_tokens_details.model_dump()}
                         if u else {}}

    def complete(self, messages, tools):
        try:
            self._initialize()
            if self.model == 'gpt-5.6-luna':
                return self._complete_responses(messages, tools)
            response = self.client.chat.completions.create(model=self.model, messages=messages,
                tools=tools, tool_choice='required', parallel_tool_calls=False,
                max_completion_tokens=3000)
            m = response.choices[0].message
            return {'message': {'content': m.content, 'tool_calls': [
                {'id': c.id, 'type': 'function', 'function': {'name': c.function.name, 'arguments': c.function.arguments}}
                for c in (m.tool_calls or [])]},
                'usage': response.usage.model_dump() if response.usage else {}}
        except AnswerError:
            raise
        except Exception as exc:
            code = getattr(exc, 'status_code', None)
            # Do not expose response bodies, keys, request headers or provider echo text.
            if isinstance(code, int) and 500 <= code <= 599:
                raise AnswerError(f"The model service is temporarily unavailable (HTTP {code}). This answer is incomplete; try again later. Saved research results are still available.") from None
            raise AnswerError(f'Model request failed ({type(exc).__name__}' + (f', HTTP {code}' if code else '') +
                              '). Check quota, network connectivity and provider tool calling support. The original run is unchanged.') from None
