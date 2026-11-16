import json
from pathlib import Path
import tempfile
import unittest

from extensions.analysis_cache import bind_analysis_inputs


class AnalysisCacheTests(unittest.TestCase):
    def test_identical_inputs_resume_but_changed_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            data = {'protocol': {'target': 'mortality'}, 'evidence': [{'events': 5}], 'mode': 'live'}
            first = bind_analysis_inputs(output, data)
            (output / 'analysis_result.json').write_text('{"status":"completed"}')
            self.assertEqual(bind_analysis_inputs(output, data), first)
            changed = {**data, 'evidence': [{'events': 14}]}
            with self.assertRaisesRegex(ValueError, 'changed'):
                bind_analysis_inputs(output, changed)
            self.assertEqual(json.loads((output / 'analysis_result.json').read_text())['status'], 'completed')

    def test_completed_unbound_legacy_result_is_not_trusted(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / 'analysis_summary.json').write_text('{"status":"completed"}')
            with self.assertRaisesRegex(ValueError, 'no input binding'):
                bind_analysis_inputs(output, {'new': 'inputs'})

    def test_offline_result_cannot_be_reused_for_live_run(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            bind_analysis_inputs(output, {'offline': True})
            with self.assertRaises(ValueError):
                bind_analysis_inputs(output, {'offline': False})

    def test_incomplete_legacy_lock_is_archived_and_upgraded(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            output.mkdir(exist_ok=True)
            (output / 'analysis_inputs.lock.json').write_text(json.dumps({
                'schema_version': 'analysis_inputs/8.0', 'fingerprint': 'old-combined-hash',
            }))
            (output / 'runtime_facets.json').write_text('{}')
            bind_analysis_inputs(output, {'protocol': {'outcome': 'mortality'}, 'implementation': 'new-code'})
            lock = json.loads((output / 'analysis_inputs.lock.json').read_text())
            self.assertEqual(lock['schema_version'], 'analysis_inputs/8.1')
            self.assertTrue(lock['partial_restart_archive'])
            archived = output / lock['partial_restart_archive']
            self.assertTrue((archived / 'analysis_inputs.lock.json').is_file())
            self.assertTrue((archived / 'runtime_facets.json').is_file())

    def test_code_change_restarts_only_incomplete_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            first = {'protocol': {'outcome': 'mortality'}, 'implementation': 'code-a'}
            bind_analysis_inputs(output, first)
            (output / 'checkpoints').mkdir()
            (output / 'checkpoints' / 'partial.json').write_text('{}')
            bind_analysis_inputs(output, {**first, 'implementation': 'code-b'})
            self.assertFalse((output / 'checkpoints').exists())
            self.assertTrue(next((output / 'partial_analysis_history').iterdir()).is_dir())
            with self.assertRaisesRegex(ValueError, 'changed'):
                bind_analysis_inputs(output, {'protocol': {'outcome': 'readmission'}, 'implementation': 'code-b'})


if __name__ == '__main__':
    unittest.main()
