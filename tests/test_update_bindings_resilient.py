"""Incremental reuse remains safe and executable through real Resilient worker gates."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from extensions import update
from extensions.common import read, write, digest
from extensions.input_contract import normalize_request as compile_request
from extensions.stage_inputs import bind_stage_inputs, stage_signature
from trace_runtime.contracts import normalize_request, prepare_extraction, ContractError
from trace_runtime.pipeline import fingerprint

VERSIONS = {stage: 'test-implementation-' + stage for stage in update.STAGES}


def case(root):
    md = root / 'article.md'
    md.write_text('# Results\nObserved mortality was 5 of 19 versus 9 of 37 participants at one year.\n')
    content = root / 'article_content.json'
    write(content, [{'type': 'text', 'text': 'Results', 'text_level': 1, 'page_idx': 0},
                    {'type': 'text', 'text': 'Observed mortality was 5 of 19 versus 9 of 37 participants at one year.', 'page_idx': 0}])
    return normalize_request(compile_request({
        'schema_version': 'trace/1.0', 'case_id': 'update_example', 'topic': 'Treatment and mortality',
        'protocol': {'protocol_id': 'protocol', 'case_id': 'update_example', 'pico': {
            'population': 'Adults', 'intervention_or_exposure': 'Treatment', 'comparison': 'Control', 'outcome': 'All-cause mortality'},
            'targets': [{'target_id': 'death', 'outcome': 'All-cause mortality',
                         'accepted_data_families': ['binary_arm_counts'], 'preferred_effect_measures': ['RR'],
                         'timepoint_policy': 'At least 10 months'}]},
        'analysis_profile': {'profile_version': '2.0', 'primary_analysis_spec': {
            'analysis_id': 'mortality', 'target_id': 'death', 'eligible_data_families': ['binary_arm_counts'],
            'accepted_effect_types': ['RR'], 'selection_rules': {'rules': ['Use independent studies']}}},
        'documents': [{'article_id': 'a1', 'study_id': 's1', 'document_id': 'd1',
                       'paths': {'markdown': str(md), 'content_list': str(content)}}],
        'options': {'extraction_runtime': 'typed', 'provider': 'openai_compatible', 'model': 'offline-test',
                    'route_rob': False, 'max_workers': 1, 'max_retries': 0},
    }), root)


class UpdateBindingsTests(unittest.TestCase):
    def baseline(self, root):
        request = case(root)
        base = root / 'baseline'; base.mkdir()
        write(base / 'request.normalized.json', request)
        write(base / 'update_dependencies.json', VERSIONS)
        profile = {'text': {'name': 'openai_compatible', 'model': 'offline-test', 'base_url': 'https://example.invalid'}, 'embedding': {}}
        settings = {'bias': {'mode': 'draft'}}
        write(base / 'provider_routes.json', profile)
        write(base / 'extensions.normalized.json', settings)
        screening = {'case_id': request['case_id'], 'decisions': [{'candidate_id': 'a1', 'final_status': 'Include'}]}
        bind_stage_inputs(request, base / '01_screening', 'screening')
        write(base / '01_screening/screening_results.json', screening)
        write(base / '01_screening/effective_screening_results.json', screening)
        write(base / '01_screening/risk_of_bias_recommendations.json', {'recommendations': []})
        config = prepare_extraction(request, ['a1'], base / 'extraction_config')
        source = base / '02_extraction/articles/a1/attempt_01'; source.mkdir(parents=True)
        signatures = update._extraction_reuse_fingerprints(request, config, 'a1')
        from extensions.resilient_extraction import STATE_SCHEMA
        write(source / 'run_manifest.json', {'case_id': request['case_id'], 'article_id': 'a1'})
        write(source / 'resilient_article_state.json', {'schema_version': STATE_SCHEMA, 'article_id': 'a1', 'status': 'completed', **signatures})
        write(source / 'extraction_submission.json', {'case_id': request['case_id'], 'predictions': []})
        rows = [{'article_id': 'a1', 'status': 'completed', 'run_dir': str(source), 'issues': []}]
        bind_stage_inputs(request, base / '02_extraction', 'extraction')
        write(base / '02_extraction/case_run_manifest.json', {'article_runs': rows})
        write(base / '02_extraction/case_extraction_submission.json', {'case_id': request['case_id'], 'predictions': []})
        outputs = {'screening': ['01_screening/screening_results.json', '01_screening/risk_of_bias_recommendations.json'],
                   'extraction': ['02_extraction/case_run_manifest.json', '02_extraction/case_extraction_submission.json']}
        state = {'status': 'completed', 'input_fingerprint': fingerprint(request),
                 'effective_screening_hash': digest(screening),
                 'stages': {stage: {'status': 'completed', 'outputs': {name: update.sha(base / name) for name in files}}
                            for stage, files in outputs.items()}}
        write(base / 'extensions.lock.json', {'baseline_lock_hash': update.sha(update.ROOT / 'component_provenance.json')})
        write(base / 'pipeline_manifest.json', state)
        return base, request, state, profile, settings

    def runner(self, base, new, plan, out):
        runner = object.__new__(update.UpdateRunner)
        runner.base_run, runner.out, runner.r, runner.update_plan = base, out, new, plan
        out.mkdir(); write(out / 'update_reuse.json', {})
        return runner

    @patch.object(update, 'implementation_versions', return_value=VERSIONS)
    def test_seeded_stages_pass_real_worker_boundaries_with_an_added_excluded_paper(self, _):
        from extensions.parallel_worker import screen, extract, MINERS
        sys.path.insert(0, str(MINERS))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            base, old, state, profile, settings = self.baseline(root)
            new = copy.deepcopy(old)
            second = root / 'other.md'; second.write_text('A different topic, excluded by screening.')
            new['documents'].append({'article_id': 'a2', 'document_id': 'd2', 'study_id': 's2', 'paths': {'markdown': str(second)}})
            plan = update.make_plan(base, old, new, state, settings, profile)
            out = root / 'updated'; runner = self.runner(base, new, plan, out)
            runner._seed('screening')
            self.assertEqual(read(out / '01_screening/screening_inputs.lock.json')['fingerprint'], stage_signature(new, 'screening'))
            with patch('miners.paper_node_adapter.build_llm_callback', return_value=lambda _: 'unused'), \
                 patch('miners.case_screening_rob.FullTextProtocolScreener.screen', return_value={'candidate_id': 'a2', 'final_status': 'Exclude'}) as llm_screen:
                screen(new, out / '01_screening')
            self.assertEqual(llm_screen.call_count, 1)
            effective = read(out / '01_screening/screening_results.json')
            write(out / '01_screening/effective_screening_results.json', effective)
            config = prepare_extraction(new, ['a1'], out / 'extraction_config')
            runner._seed('extraction')
            self.assertEqual(read(out / '02_extraction/extraction_inputs.lock.json')['fingerprint'], stage_signature(new, 'extraction'))
            with patch('extensions.resilient_extraction.run_article', side_effect=AssertionError('A completed inherited article must not call a model')):
                extract(new, config, out / '02_extraction')
            self.assertEqual(read(out / '02_extraction/extraction_complete.json')['article_count'], 1)
            inherited = read(out / '02_extraction/extraction_progress.json')['article_runs'][0]
            self.assertIn('source_fingerprint', inherited['update_source_provenance'])
            self.assertTrue(Path(inherited['run_dir']).is_relative_to(out))

    @patch.object(update, 'implementation_versions', return_value=VERSIONS)
    def test_modified_article_semantic_state_is_rejected_before_new_checkpoint_binding(self, _):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); base, old, state, profile, settings = self.baseline(root)
            plan = update.make_plan(base, old, old, state, settings, profile)
            out = root / 'updated'; runner = self.runner(base, old, plan, out)
            write(out / '01_screening/effective_screening_results.json', read(base / '01_screening/effective_screening_results.json'))
            prepare_extraction(old, ['a1'], out / 'extraction_config')
            file = base / '02_extraction/articles/a1/attempt_01/resilient_article_state.json'
            payload = read(file); payload['semantic_fingerprint'] = 'stale-schema'; write(file, payload)
            with self.assertRaisesRegex(ContractError, 'source/task/schema changed'):
                runner._seed('extraction')
            self.assertFalse((out / '02_extraction/extraction_progress.json').exists())
            self.assertFalse((out / '02_extraction/extraction_inputs.lock.json').exists())

    @patch.object(update, 'implementation_versions', return_value=VERSIONS)
    def test_missing_dependency_provenance_disables_reuse(self, _):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); base, old, state, profile, settings = self.baseline(root)
            (base / 'update_dependencies.json').unlink()
            plan = update.make_plan(base, old, old, state, settings, profile)
            self.assertFalse(any(plan['compatible'].values()))
            self.assertTrue(all(not ids for ids in plan['reusable_articles'].values()))

    def test_protocol_delta_requires_explicit_profile_and_recompiles_when_supplied(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); old = case(root)
            protocol = copy.deepcopy(old['protocol'])
            protocol['targets'][0]['timepoint_policy'] = 'At least 24 months'
            delta = {'schema_version': update.SCHEMA, 'protocol': protocol}
            file = root / 'delta.json'; write(file, delta)
            with self.assertRaisesRegex(ContractError, 'explicit analysis_profile'):
                update.updated_request(old, delta_file=file)
            delta['analysis_profile'] = {'profile_version': '2.0', 'primary_analysis_spec': {
                'analysis_id': 'mortality', 'target_id': 'death', 'eligible_data_families': ['binary_arm_counts'],
                'accepted_effect_types': ['RR']}}
            write(file, delta)
            compiled = update.updated_request(old, delta_file=file)
            rules = ' '.join(compiled['analysis_profile']['primary_analysis_spec']['selection_rules'])
            self.assertIn('24 months', rules); self.assertNotIn('10 months', rules)

    def test_full_request_and_document_only_delta_use_same_resilient_compiler(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); old = case(root)
            file = root / 'request.json'; write(file, old)
            delta = root / 'delta.json'; write(delta, {'schema_version': update.SCHEMA, 'documents': []})
            self.assertEqual(update.updated_request(old, request_file=file), update.updated_request(old, delta_file=delta))

    def test_method_interface_changes_invalidate_analysis_only(self):
        original = update.implementation_versions()
        real_sha = update.sha
        for filename in ('statistical_methods.py', 'statistical_method_agent.py'):
            with self.subTest(module=filename), patch.object(
                    update, 'sha', side_effect=lambda path: 'changed-interface'
                    if Path(path).name == filename else real_sha(path)):
                changed = update.implementation_versions()
            self.assertNotEqual(original['analysis'], changed['analysis'])
            for stage in update.STAGES:
                if stage != 'analysis':
                    self.assertEqual(original[stage], changed[stage])

    def test_vendor_schema_change_invalidates_extraction_dependencies(self):
        original = update.implementation_versions()
        real_sha = update.sha
        with patch.object(update, 'sha', side_effect=lambda path: 'changed-schema' if str(path).endswith('runtime/typed/locator_extractor_agent/extraction_models.py') else real_sha(path)):
            changed = update.implementation_versions()
        self.assertNotEqual(original['extraction'], changed['extraction'])
        self.assertEqual(original['screening'], changed['screening'])


if __name__ == '__main__':
    unittest.main()
