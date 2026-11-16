import json
import tempfile
import unittest
from pathlib import Path

from extensions.runner import pristine_extraction_rebind


class ExtractionRebindTests(unittest.TestCase):
    def locks(self):
        previous = {'schema_version': 'extensions_lock/8.0', 'baseline_lock_hash': 'base',
                    'settings_fingerprint': 'old', 'implementation_fingerprint': 'old-impl'}
        current = {**previous, 'settings_fingerprint': 'new',
                   'implementation_fingerprint': 'new-impl'}
        return previous, current

    def manifest(self):
        return {'stages': {'screening': {'status': 'completed'},
                           'screening_rob': {'status': 'completed'},
                           'extraction': {'status': 'paused'}}}

    def test_allows_rebind_before_any_completed_extraction(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'02_extraction/articles').mkdir(parents=True)
            settings = {'graph': {'brain_loop': False}, 'bias': {'mode': 'draft'}}
            (root/'extensions.normalized.json').write_text(json.dumps(
                {'graph': {'brain_loop': True}, 'bias': settings['bias']}))
            result = pristine_extraction_rebind(
                *self.locks(), self.manifest(), root, settings)
            self.assertEqual(0, result['completed_extraction_articles'])

    def test_rejects_rebind_after_a_completed_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            batch = root/'02_extraction/articles/a/attempt_01/batches'
            batch.mkdir(parents=True)
            (batch/'batch_001.json').write_text('{}')
            settings = {'graph': {}, 'bias': {'mode': 'draft'}}
            (root/'extensions.normalized.json').write_text(json.dumps(settings))
            self.assertIsNone(pristine_extraction_rebind(
                *self.locks(), self.manifest(), root, settings))

    def test_rejects_bias_setting_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'02_extraction/articles').mkdir(parents=True)
            (root/'extensions.normalized.json').write_text(json.dumps(
                {'graph': {}, 'bias': {'mode': 'draft'}}))
            self.assertIsNone(pristine_extraction_rebind(
                *self.locks(), self.manifest(), root,
                {'graph': {}, 'bias': {'mode': 'final'}}))


if __name__ == '__main__':
    unittest.main()
