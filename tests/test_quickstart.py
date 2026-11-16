"""Portable quickstart tests: synthetic JATS, no model or network calls."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_quickstart', ROOT / 'scripts/prepare_quickstart.py')
quickstart = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quickstart)


class QuickstartTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.example = self.root / 'example'
        self.cache = self.root / 'cache'
        self.example.mkdir()
        self.cache.mkdir()
        self.manifest = json.loads((quickstart.EXAMPLE / 'sources.json').read_text())
        self.template = json.loads((quickstart.EXAMPLE / 'request.template.json').read_text())
        for source in self.manifest['sources']:
            xml = (f'<article><front><article-meta><article-id pub-id-type="pmcid">{source["pmcid"]}</article-id>'
                   f'<article-id pub-id-type="doi">{source["doi"]}</article-id>'
                   '<abstract><p>Randomized trial abstract.</p></abstract></article-meta></front>'
                   '<body><sec><title>Methods</title><p>Random allocation; standard care control.</p></sec>'
                   '<sec><title>Results</title><p>Mortality was assessed at the scheduled visit.</p>'
                   '<table-wrap><label>Table 1</label><caption><p>Mortality by randomized arm</p></caption>'
                   '<table><thead><tr><th>Outcome</th><th colspan="2">Groups</th></tr></thead>'
                   '<tbody><tr><td>Mortality</td><td>2/10</td><td>3/10</td></tr></tbody></table>'
                   '<table-wrap-foot><p>Participants were followed for 28 days.</p></table-wrap-foot></table-wrap>'
                   '</sec></body><back><ref-list><ref>Do not mix this cited review into the main text.</ref>'
                   '</ref-list></back></article>').encode()
            source['xml_sha256'] = quickstart.digest(xml)
            (self.cache / (source['pmcid'] + '.xml')).write_bytes(xml)
        (self.example / 'sources.json').write_bytes(quickstart.json_bytes(self.manifest))
        (self.example / 'request.template.json').write_bytes(quickstart.json_bytes(self.template))
        self.patch = patch.object(quickstart, 'EXAMPLE', self.example)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def prepare(self, name='inputs', **kwargs):
        out = self.root / name
        with patch.object(quickstart, 'urlopen', side_effect=AssertionError('unexpected network')):
            summary = quickstart.prepare(out, cache=self.cache, offline=True, **kwargs)
        return out, summary

    def test_default_two_independent_oa_studies_and_fixed_protocol(self):
        out, summary = self.prepare()
        request = json.loads((out / 'request.json').read_text())
        self.assertEqual(summary['articles'], 2)
        self.assertEqual({d['study_id'] for d in request['documents']}, {'NCT01052480', 'NCT02287467'})
        self.assertEqual(request['protocol']['statistical_method_policy'], 'fixed')
        self.assertEqual(request['analysis_profile']['primary_analysis_spec']['pooling']['heterogeneity_estimator'], 'reml')
        self.assertIn('Standard care alone', request['protocol']['pico']['comparison'])
        self.assertTrue(request['options']['route_rob'])
        for document in request['documents']:
            for value in document['paths'].values():
                self.assertFalse(Path(value).is_absolute())
                self.assertTrue((out / value).is_file())

    def test_optional_manuscript_keeps_access_classification(self):
        out, summary = self.prepare(include_author_manuscript=True)
        self.assertEqual(summary['articles'], 3)
        sources = json.loads((out / 'SOURCE_MANIFEST.json').read_text())['sources']
        third = next(s for s in sources if s['pmcid'] == 'PMC6941345')
        self.assertFalse(third['is_pmc_openaccess'])
        self.assertEqual(third['license_code'], 'TDM')

    def test_update_delta_preserves_registered_baseline_source_paths(self):
        from extensions.input_contract import normalize_request as compile_request
        from extensions.update import updated_request
        from metapipeline.contracts import normalize_request
        base, _ = self.prepare('base')
        extra, _ = self.prepare('extra', include_author_manuscript=True)
        old = normalize_request(compile_request(json.loads((base / 'request.json').read_text())), base)
        new = updated_request(old, delta_file=extra / 'update.delta.json')
        self.assertEqual(new['documents'][:2], old['documents'])
        self.assertEqual(len(new['documents']), 3)
        self.assertEqual(new['documents'][2]['article_id'], 'PMC6941345')
        self.assertEqual(Path(new['documents'][2]['paths']['markdown']).parent, (extra / 'documents').resolve())

    def test_preserves_tables_footnotes_and_avoids_fake_pages(self):
        out, _ = self.prepare()
        blocks = json.loads((out / 'documents/PMC5828518.content_list.json').read_text())
        tables = [b for b in blocks if b['type'] == 'table']
        self.assertEqual(len(tables), 1)
        self.assertIn('colspan="2"', tables[0]['table_body'])
        self.assertIn('28 days', tables[0]['table_footnote'][0])
        self.assertTrue(all('page_idx' not in b for b in blocks))
        self.assertNotIn('cited review', (out / 'documents/PMC5828518.md').read_text())

    def test_corrupt_cache_fails_before_any_destination_write(self):
        (self.cache / 'PMC6868512.xml').write_text('corrupt')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.prepare()
        self.assertFalse((self.root / 'inputs').exists())

    def test_identity_is_checked_even_when_hash_matches(self):
        source = copy.deepcopy(self.manifest['sources'][0])
        data = (self.cache / 'PMC5828518.xml').read_bytes().replace(b'PMC5828518', b'PMC9999999')
        source['xml_sha256'] = quickstart.digest(data)
        with self.assertRaisesRegex(ValueError, 'identity'):
            quickstart.verify_xml(data, source)

    def test_rerun_is_identical_and_never_overwrites_user_changes(self):
        out, _ = self.prepare()
        before = {p.relative_to(out): p.read_bytes() for p in out.rglob('*') if p.is_file()}
        self.prepare()
        self.assertEqual(before, {p.relative_to(out): p.read_bytes() for p in out.rglob('*') if p.is_file()})
        request = out / 'request.json'
        request.write_text('user edited protocol')
        with self.assertRaisesRegex(ValueError, 'Existing file differs'):
            self.prepare()
        self.assertEqual(request.read_text(), 'user edited protocol')

    def test_missing_offline_source_does_not_download(self):
        (self.cache / 'PMC5828518.xml').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing cached full text'):
            self.prepare()

    def test_manifest_binds_all_generated_scientific_inputs(self):
        out, _ = self.prepare()
        manifest = json.loads((out / 'SOURCE_MANIFEST.json').read_text())
        self.assertEqual(len(manifest['generated_files']), 7)
        for name, expected in manifest['generated_files'].items():
            self.assertEqual(quickstart.digest((out / name).read_bytes()), expected)


if __name__ == '__main__':
    unittest.main()
