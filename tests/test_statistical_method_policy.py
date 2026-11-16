"""Release protocol and routing checks; no external model calls."""
import copy
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from extensions.input_contract import normalize_request, InputContractError
from extensions.statistical_methods import protocol_method_policy, assert_method_policy_output_compatible


def request_fixture():
    return {
        'protocol': {
            'pico': {'population': 'adults', 'exposure': 'intervention',
                     'comparison': 'control', 'outcome': 'event'},
            'targets': [{'target_id': 'event', 'outcome': 'event',
                         'accepted_data_families': ['binary_arm_counts'],
                         'preferred_effect_measures': ['RR']}],
        },
        'analysis_profile': {'primary_analysis_spec': {
            'analysis_id': 'primary', 'target_id': 'event'}},
    }


class MethodPolicyProtocolTests(unittest.TestCase):
    def test_omitted_policy_is_explicit_fixed_without_mutating_input(self):
        request = request_fixture()
        original = copy.deepcopy(request)
        normalized = normalize_request(request)
        self.assertEqual(normalized['protocol']['statistical_method_policy'], 'fixed')
        self.assertEqual(request, original)

    def test_opt_in_survives_normalization_and_is_idempotent(self):
        request = request_fixture()
        request['protocol']['statistical_method_policy'] = 'model_selected'
        normalized = normalize_request(request)
        self.assertEqual(normalize_request(normalized), normalized)
        self.assertEqual(protocol_method_policy(normalized['protocol']), 'model_selected')

    def test_malformed_policy_fails_before_model_work(self):
        for value in [None, True, {}, [], 'automatic', 'FIXED']:
            with self.subTest(value=value):
                request = request_fixture()
                request['protocol']['statistical_method_policy'] = value
                with self.assertRaises(InputContractError):
                    normalize_request(request)

    def test_supplied_test_parameters_explicitly_select_fixed(self):
        root = Path(__file__).resolve().parents[1]
        fragment = json.loads((root/'config/test_protocol_parameters.json').read_text())
        self.assertEqual(protocol_method_policy(fragment['protocol']), 'fixed')

    def test_modes_cannot_share_output_even_when_legacy_hashes_are_bypassed(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory)/'statistical_method_policy.json'
            marker.write_text('{"mode":"fixed"}')
            assert_method_policy_output_compatible(directory, 'fixed')
            with self.assertRaisesRegex(ValueError, 'new analysis output directory'):
                assert_method_policy_output_compatible(directory, 'model_selected')

    def test_legacy_outputs_cannot_be_promoted_to_model_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'analysis_inputs.lock.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'new output directory'):
                assert_method_policy_output_compatible(directory, 'model_selected')


class MethodPolicyRuntimeTests(unittest.TestCase):
    def test_default_components_do_not_inject_selector(self):
        from extensions.analysis_worker import agent_components
        components = agent_components(Path('/private/tmp'), offline=True)
        self.assertNotIn('method_selector', components)

    def test_model_selected_offline_is_rejected(self):
        from extensions.analysis_worker import agent_components
        with self.assertRaisesRegex(ValueError, 'requires a live method selector'):
            agent_components(Path('/private/tmp'), offline=True,
                             statistical_method_policy='model_selected')

    def test_model_selected_components_use_configured_provider_without_calling_it(self):
        from extensions.analysis_worker import agent_components
        from analysis_engine.source_aware_method_selector import ToolAwareMethodSelector
        env = {'TEXT_PROVIDER_API_KEY': 'test-placeholder',
               'EMBEDDING_PROVIDER_API_KEY': 'test-placeholder',
               'TEXT_PROVIDER_MODEL': 'test-model',
               'TEXT_PROVIDER_BASE_URL': 'https://example.invalid/v1'}
        # Isolate selector construction from the unrelated reviewer checkpoint wrapper.
        checkpoints = SimpleNamespace(CheckpointedTopicReviewer=lambda delegate, path: delegate)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, env), \
                patch.dict(sys.modules, {'extensions.analysis_checkpoints': checkpoints}), \
                patch('openai.OpenAI', side_effect=AssertionError('No remote calls')):
            components = agent_components(Path(directory),
                statistical_method_policy='model_selected')
        selector = components['method_selector']
        self.assertIsInstance(selector, ToolAwareMethodSelector)
        self.assertEqual(selector.base_url, env['TEXT_PROVIDER_BASE_URL'])
        self.assertEqual(selector.calls_made, 0)

    def test_metadata_reports_selected_policy_and_actual_selector_calls(self):
        from extensions.analysis_worker import write_run_metadata
        with tempfile.TemporaryDirectory() as directory:
            write_run_metadata(directory, {'model_calls': {'method_selector': 3}}, 'model_selected')
            metadata = json.loads((Path(directory)/'run_metadata.json').read_text())
        self.assertEqual(metadata['statistical_method_policy'], 'model_selected')
        self.assertEqual(metadata['method_selector_llm_calls'], 3)
        self.assertEqual(metadata['statistical_policy']['selector'], 'ToolAwareMethodSelector')

    def test_default_agent_retains_fixed_selector_and_calculator(self):
        from extensions.statistical_method_agent import ReleaseAnalysisAgent
        from analysis_engine.fixed_policy import FixedMethodSelector, FixedIntervalCalculator
        with tempfile.TemporaryDirectory() as directory:
            agent = ReleaseAnalysisAgent(audit_dir=Path(directory))
        self.assertIsInstance(agent.method_selector, FixedMethodSelector)
        self.assertIsInstance(agent.calculator, FixedIntervalCalculator)

    def test_model_selected_agent_installs_selector_and_validated_calculator(self):
        from extensions.statistical_method_agent import ReleaseAnalysisAgent, ValidatedSelectedMethodCalculator
        from analysis_engine.source_aware_method_selector import ToolAwareMethodSelector
        with tempfile.TemporaryDirectory() as directory:
            selector = ToolAwareMethodSelector(api_key='test-placeholder')
            agent = ReleaseAnalysisAgent(audit_dir=Path(directory),
                statistical_method_policy='model_selected', method_selector=selector)
        self.assertIs(agent.method_selector, selector)
        self.assertIsInstance(agent.calculator, ValidatedSelectedMethodCalculator)


@unittest.skipUnless(shutil.which('Rscript'), 'Rscript is unavailable')
class MethodPolicyCalculationTests(unittest.TestCase):
    def _run(self, directory, mode, response=None):
        from extensions.analysis_worker import agent_components
        from extensions.graph_feedback import ExtendedAnalysisAgent
        from analysis_engine.source_aware_method_selector import ToolAwareMethodSelector
        from contract_adapter import load_analysis_contract
        fixture = Path(__file__).parent/'fixtures/statistical_methods'
        protocol, task, profile, payloads, studies = load_analysis_contract(fixture)
        components = agent_components(Path(directory), offline=True)
        if mode == 'model_selected':
            selector = ToolAwareMethodSelector(api_key='test-placeholder',
                artifact_dir=Path(directory)/'method_selection')
            def reply(*args):
                selector.calls_made += 1
                return json.dumps(response)
            selector._request_content = reply
            components['method_selector'] = selector
        agent = ExtendedAnalysisAgent(audit_dir=Path(directory),
            statistical_method_policy=mode, **components)
        with patch('openai.OpenAI', side_effect=AssertionError('No external model calls')):
            result = agent.run(protocol=protocol, task=task, analysis_profile=profile,
                extractor_payloads=payloads, study_by_article=studies)
        return result.analyses[0]

    def test_default_runs_reml_normal_with_separate_hk(self):
        with tempfile.TemporaryDirectory() as directory:
            topic = self._run(directory, 'fixed')
        self.assertEqual(topic.status, 'completed')
        self.assertEqual(topic.method_selection.heterogeneity_estimator, 'reml')
        self.assertEqual(topic.method_selection.confidence_interval, 'normal_approximation')
        self.assertEqual(topic.interval_sensitivity['hk_status'], 'completed')

    def test_model_selected_executes_pm_hk_without_replacing_it_with_reml(self):
        response = dict(pooling_model='random_effects', heterogeneity_estimator='paule_mandel',
            confidence_interval='hartung_knapp_t', rationale='Synthetic transport fixture for route verification.')
        with tempfile.TemporaryDirectory() as directory:
            topic = self._run(directory, 'model_selected', response)
            decisions = list((Path(directory)/'method_selection').glob('*.json'))
            self.assertEqual(len(decisions), 1)
        self.assertEqual(topic.status, 'completed')
        self.assertEqual(topic.method_selection.heterogeneity_estimator, 'paule_mandel')
        scale = topic.interval_sensitivity['primary_profile']['pooling_scale']
        self.assertIn('__paule_mandel', scale)
        self.assertIn('__hartung_knapp_t', scale)
        self.assertNotIn('__reml', scale)
        self.assertIsNotNone(topic.statistics)

    def test_invalid_model_choice_blocks_instead_of_falling_back(self):
        response = dict(pooling_model='random_effects', heterogeneity_estimator='unsupported',
            confidence_interval='normal_approximation', rationale='Invalid test transport response.')
        with tempfile.TemporaryDirectory() as directory:
            topic = self._run(directory, 'model_selected', response)
        self.assertEqual(topic.status, 'blocked')
        self.assertIsNone(topic.statistics)

    def test_worker_saves_selected_mode_and_keeps_modes_separate_on_resume(self):
        from extensions.analysis_worker import execute_analysis, agent_components
        from analysis_engine.source_aware_method_selector import ToolAwareMethodSelector
        fixture = Path(__file__).parent/'fixtures/statistical_methods'
        response = dict(pooling_model='random_effects', heterogeneity_estimator='paule_mandel',
            confidence_interval='hartung_knapp_t', rationale='Synthetic worker transport fixture.')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root/'input'
            shutil.copytree(fixture, inputs)
            protocol_path = inputs/'extraction_protocol.json'
            protocol = json.loads(protocol_path.read_text())
            protocol['statistical_method_policy'] = 'model_selected'
            protocol_path.write_text(json.dumps(protocol))
            output = root/'analysis'
            components = agent_components(output, offline=True)
            selector = ToolAwareMethodSelector(api_key='test-placeholder',
                artifact_dir=output/'method_selection')
            def reply(*args):
                selector.calls_made += 1
                return json.dumps(response)
            selector._request_content = reply
            components['method_selector'] = selector
            with patch('extensions.analysis_worker.agent_components', return_value=components), \
                    patch('extensions.analysis_cache.analysis_implementation_fingerprint', return_value='test-source-id'), \
                    patch('openai.OpenAI', side_effect=AssertionError('No external model calls')):
                summary = execute_analysis(inputs, output)
                cached = execute_analysis(inputs, output)
                self.assertEqual(cached, summary)
                self.assertEqual(selector.calls_made, 1)
                metadata = json.loads((output/'run_metadata.json').read_text())
                self.assertEqual(metadata['statistical_method_policy'], 'model_selected')
                self.assertEqual(metadata['method_selector_llm_calls'], 1)
                self.assertEqual(summary['statistical_method_policy'], 'model_selected')
                protocol['statistical_method_policy'] = 'fixed'
                protocol_path.write_text(json.dumps(protocol))
                with patch('extensions.analysis_cache.bind_analysis_inputs'), \
                        self.assertRaisesRegex(ValueError, 'new analysis output directory'):
                    execute_analysis(inputs, output, offline=True)


if __name__ == '__main__':
    unittest.main()
