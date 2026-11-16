"""Distribution boundaries; no remote services or private corpus required."""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class DistributionTests(unittest.TestCase):
    def test_public_figures_are_valid_and_match_manuscript_bytes(self):
        import hashlib
        from scripts.check_repository import findings
        manifest = json.loads((ROOT/'docs/assets/provenance.json').read_text())
        for row in manifest['figures']:
            path = ROOT/'docs/assets'/row['file']
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row['sha256'])
            self.assertEqual(findings([path]), [])

    def test_arbitrary_binary_is_still_rejected(self):
        import tempfile
        from unittest.mock import patch
        from scripts import check_repository
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'unexpected.bin'
            path.write_bytes(bytes([255, 0, 128]))
            with patch.object(check_repository, 'ROOT', root):
                self.assertIn(('unexpected.bin', 'unexpected binary'), check_repository.findings([path]))

    def test_internal_development_names_are_rejected(self):
        import tempfile
        from unittest.mock import patch
        from scripts import check_repository
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'module.py'
            path.write_text('class FixedMethodSelector' + 'V' + '22: pass')
            with patch.object(check_repository, 'ROOT', root):
                self.assertIn(('module.py', 'internal development label'), check_repository.findings([path]))

    def test_external_api_and_guidance_versions_remain_allowed(self):
        import tempfile
        from unittest.mock import patch
        from scripts import check_repository
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'reference.txt'
            path.write_text('https://api.openai.com/v1\nROBINS-I V2\nactions/checkout@v4\n*_content_list_v2.json')
            with patch.object(check_repository, 'ROOT', root):
                self.assertEqual(check_repository.findings([path]), [])

    def test_document_client_is_independent(self):
        from document_io import mineru
        self.assertEqual(Path(mineru.__file__).resolve().parent, ROOT / 'document_io')
        self.assertEqual(mineru.PROJECT_ROOT, ROOT)
        self.assertNotIn('bench.manalyzer', (ROOT / 'extensions/document_preparation.py').read_text())

    def test_external_api_versions_are_preserved(self):
        config = json.loads((ROOT / 'config/provider.example.json').read_text())
        for key in ('text_provider', 'embedding_provider'):
            self.assertEqual(config[key]['base_url'], 'https://api.openai.com/v1')

    def test_separate_distribution(self):
        self.assertTrue((ROOT / 'run_pipeline.py').is_file())
        for name in ('cases', 'evaluation', 'gold', 'requests/benchmark_22_cases'):
            self.assertFalse((ROOT / name).exists())

    def test_external_release_and_mineru_identifiers(self):
        workflow = (ROOT / '.github/workflows/ci.yml').read_text()
        self.assertIn('actions/checkout@v4', workflow)
        self.assertIn('actions/setup-python@v5', workflow)
        adapters = [p for p in (ROOT / 'vendor').rglob('benchmark_adapter.py')
                    if '/runtime/typed/' in p.as_posix()]
        self.assertEqual(len(adapters), 2)
        for adapter in adapters:
            self.assertIn('*_content_list_v2.json', adapter.read_text())


if __name__ == '__main__':
    unittest.main()
