import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from extensions.common import BASELINE
from extensions.stage_inputs import bind_stage_inputs, validate_screening_reuse
from extensions.runner import (archive_terminal_analysis_for_retry, frozen_analysis_request_for_retry,
                               rebind_runner_input_for_analysis_retry)
from extensions.parallel_worker import _workers
from trace_runtime.contracts import screening_protocol

sys.path.insert(0, str(BASELINE / 'vendor' / 'miners'))
from miners.case_screening_rob import FullTextProtocolScreener
from miners.screening import build_screening_criteria


class StageInputsTests(unittest.TestCase):
    def request(self, directory):
        path = Path(directory) / 'article.md'
        path.write_text('Observed 5 deaths among 19 participants.')
        return {'case_id': 'test', 'topic': 'Mortality',
                'protocol': {'pico': {'population': 'Adults', 'intervention_or_exposure': 'Exposure',
                                     'comparison': 'No exposure', 'outcome': 'All-cause death'},
                             'eligible_study_designs': ['cohort'],
                             'targets': [{'outcome': 'Explicit binary arm counts with complete denominators'}]},
                'analysis_profile': {}, 'options': {'max_workers': 4, 'extraction_runtime': 'typed'},
                'documents': [{'article_id': 'a', 'paths': {'markdown': str(path)}}]}

    def test_screening_receives_scope_and_design_without_numeric_availability_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            request = self.request(directory)
            task = FullTextProtocolScreener._protocol_task(screening_protocol(request))
            criteria = build_screening_criteria(task)
            self.assertIn('Outcome: All-cause death', criteria['include'])
            self.assertIn('Eligible study designs: cohort', criteria['include'])
            self.assertNotIn('complete denominators', str(criteria))

    def test_worker_setting_can_change_without_reusing_changed_scientific_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            request = self.request(directory)
            output = Path(directory) / 'out'
            signature = bind_stage_inputs(request, output, 'extraction')
            request['options']['max_workers'] = 2
            self.assertEqual(signature, bind_stage_inputs(request, output, 'extraction'))
            with patch.dict('os.environ', {'TRACE_ACTIVE_WORKERS': '1'}):
                self.assertEqual(_workers(request, 10), 1)
            Path(request['documents'][0]['paths']['markdown']).write_text('Changed source counts.')
            with self.assertRaisesRegex(ValueError, 'inputs changed'):
                bind_stage_inputs(request, output, 'extraction')

    def test_old_unbound_article_progress_cannot_bypass_input_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            request = self.request(directory)
            output = Path(directory) / 'out';output.mkdir()
            (output / 'extraction_progress.json').write_text('{"article_runs":[]}')
            with self.assertRaisesRegex(ValueError, 'no input binding'):
                bind_stage_inputs(request, output, 'extraction')

    def test_reused_screening_rejects_changed_protocol_and_modified_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            request = self.request(directory)
            source = Path(directory) / 'old';screen = source / '01_screening'
            bind_stage_inputs(request, screen, 'screening')
            (source / 'request.normalized.json').write_text(json.dumps(request))
            artifact = screen / 'results.json';artifact.write_text('{}')
            entry = {'status': 'completed', 'outputs': {'01_screening/results.json': hashlib.sha256(b'{}').hexdigest()}}
            (source / 'pipeline_manifest.json').write_text(json.dumps({'stages': {'screening': entry, 'screening_rob': entry}}))
            validate_screening_reuse(request, source)
            changed = copy.deepcopy(request);changed['protocol']['pico']['population'] = 'Children'
            with self.assertRaisesRegex(ValueError, 'eligibility conditions differ'):
                validate_screening_reuse(changed, source)
            artifact.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                validate_screening_reuse(request, source)

    def test_explicit_analysis_retry_archives_only_terminal_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for relative in ('02_extraction/extraction_complete.json', '03_review/review_complete.json',
                             '05_analysis/analysis_summary.json', 'brain_result.json'):
                path = output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{}')
            state = {'stages': {
                'screening': {'status': 'completed'}, 'screening_rob': {'status': 'completed'},
                'extraction': {'status': 'completed'}, 'review': {'status': 'completed'},
                'analysis_check': {'status': 'completed'}, 'analysis': {'status': 'completed'},
            }, 'analysis_status': 'blocked', 'error': {'message': 'old failure'}}
            audit = archive_terminal_analysis_for_retry(output, state)
            self.assertFalse((output / '05_analysis').exists())
            self.assertFalse((output / 'brain_result.json').exists())
            self.assertTrue((output / audit['archive'] / '05_analysis' / 'analysis_summary.json').is_file())
            self.assertNotIn('analysis_check', state['stages'])
            self.assertNotIn('analysis', state['stages'])
            self.assertNotIn('analysis_status', state)
            self.assertEqual(audit['prior_error'], {'message': 'old failure'})

    def test_explicit_analysis_retry_never_moves_a_running_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for relative in ('02_extraction/extraction_complete.json', '03_review/review_complete.json'):
                path = output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{}')
            state = {'stages': {
                'screening': {'status': 'completed'}, 'screening_rob': {'status': 'completed'},
                'extraction': {'status': 'completed'}, 'review': {'status': 'completed'},
                'analysis_check': {'status': 'completed'}, 'analysis': {'status': 'running'},
            }}
            with self.assertRaisesRegex(ValueError, 'cannot replace a running analysis'):
                archive_terminal_analysis_for_retry(output, state)

    def test_rebuild_handoff_archives_derived_evidence_but_keeps_upstream(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for relative in ('02_extraction/extraction_complete.json',
                             '03_review/review_complete.json',
                             '04_analysis_input/analysis_evidence_candidates.json',
                             'handoff_audit.json', '05_analysis/analysis_summary.json'):
                path = output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{}')
            state = {'stages': {
                'screening': {'status': 'completed'}, 'screening_rob': {'status': 'completed'},
                'extraction': {'status': 'completed'}, 'review': {'status': 'completed'},
                'analysis_check': {'status': 'completed'}, 'analysis': {'status': 'completed'},
            }}
            audit = archive_terminal_analysis_for_retry(
                output, state, preserve_analysis_check=True, rebuild_handoff=True)
            archived = output / audit['archive']
            self.assertTrue((archived / '04_analysis_input/analysis_evidence_candidates.json').is_file())
            self.assertTrue((archived / 'handoff_audit.json').is_file())
            self.assertTrue((archived / '05_analysis/analysis_summary.json').is_file())
            self.assertFalse((output / '04_analysis_input').exists())
            self.assertTrue((output / '02_extraction/extraction_complete.json').is_file())
            self.assertTrue((output / '03_review/review_complete.json').is_file())
            self.assertNotIn('analysis_check', state['stages'])
            self.assertNotIn('analysis', state['stages'])
            self.assertTrue(audit['rebuilt_handoff'])

    def test_analysis_only_retry_can_keep_completed_contract_check(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for relative in ('02_extraction/extraction_complete.json', '03_review/review_complete.json',
                             '05_analysis/analysis_summary.json'):
                path = output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{}')
            state = {'stages': {
                'screening': {'status': 'completed'}, 'screening_rob': {'status': 'completed'},
                'extraction': {'status': 'completed'}, 'review': {'status': 'completed'},
                'analysis_check': {'status': 'completed', 'outputs': {}}, 'analysis': {'status': 'failed'},
            }}
            audit = archive_terminal_analysis_for_retry(output, state, preserve_analysis_check=True)
            self.assertTrue(audit['preserved_analysis_check'])
            self.assertEqual(state['stages']['analysis_check']['status'], 'completed')
            self.assertNotIn('analysis', state['stages'])

    def test_analysis_retry_uses_frozen_request_not_new_pdf_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            frozen = {'case_id': 'ma_0041', 'documents': [{'article_id': 'candidate_001',
                      'paths': {'markdown': '/cached/full.md'}}], 'options': {'max_workers': 6}}
            (output / 'request.normalized.json').write_text(json.dumps(frozen))
            launch = {'case_id': 'ma_0041', 'documents': [{'article_id': 'candidate_001',
                      'paths': {'pdf': '/original/file.pdf'}}], 'options': {'max_workers': 2}}
            self.assertEqual(frozen, frozen_analysis_request_for_retry(launch, output))
            with self.assertRaisesRegex(ValueError, 'case_id does not match'):
                frozen_analysis_request_for_retry({**launch, 'case_id': 'ma_0010'}, output)

    def test_analysis_retry_rebinds_only_runner_code_fingerprint_after_stage_lock_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            request = self.request(directory)
            (output / 'request.normalized.json').write_text(json.dumps(request))
            bind_stage_inputs(request, output / '01_screening', 'screening')
            bind_stage_inputs(request, output / '02_extraction', 'extraction')
            stages = {}
            for stage, relative in {
                    'screening': ('01_screening/screening_results.json',),
                    'screening_rob': ('01_screening/risk_of_bias_assessments.json',),
                    'extraction': ('02_extraction/extraction_complete.json',),
                    'review': ('03_review/review_complete.json',),
            }.items():
                outputs = {}
                for item in relative:
                    path = output / item; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('{}')
                    outputs[item] = hashlib.sha256(path.read_bytes()).hexdigest()
                stages[stage] = {'status': 'completed', 'outputs': outputs}
            (output / 'pipeline_manifest.json').write_text(json.dumps({
                'input_fingerprint': 'old-runner-code-fingerprint', 'stages': stages,
            }))
            rebind_runner_input_for_analysis_retry(request, output)
            manifest = json.loads((output / 'pipeline_manifest.json').read_text())
            self.assertNotEqual('old-runner-code-fingerprint', manifest['input_fingerprint'])
            self.assertTrue((output / 'analysis_retry_input_binding_audit.json').is_file())


if __name__ == '__main__':
    unittest.main()
