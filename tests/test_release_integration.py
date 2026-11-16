"""Release behavior with arbitrary case identities, no private research data."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from expert_qa.provider import Transport
from scripts import render_run_dashboard

ROOT = Path(__file__).resolve().parents[1]


class ReleaseIntegrationTests(unittest.TestCase):
    def test_dashboard_renders_standalone_and_preserves_saved_i_squared(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / 'paper.md').write_text('# Study\nFull source with n = 30 and RR = 2.0.\n')
            (run / 'blocks.json').write_text(json.dumps([{'text': 'Exact block zero'}]))
            files = {
                'pipeline_manifest.json': {'status': 'completed', 'case_id': 'trial_x'},
                'request.normalized.json': {
                    'case_id': 'trial_x',
                    'documents': [{'article_id': 'article_1', 'title': 'Paper',
                                   'paths': {'markdown': 'paper.md', 'content_list': 'blocks.json'}}],
                },
                '01_screening/effective_screening_results.json': {
                    'decisions': [{'candidate_id': 'article_1', 'decision': 'include'}],
                },
                '01_screening/risk_of_bias_assessments.json': {'assessments': []},
                '02_extraction/case_extraction_submission.json': {
                    'predictions': [{
                        'prediction_id': 'e1', 'article_id': 'article_1', 'target_id': 't1',
                        'used_effect': {'value': 2.0},
                        'source_location': {'source_quote': '</script><img onerror=evil()>'},
                    }],
                },
                '03_review/candidate_evidence_review_structured.json': {},
                '04_analysis_input/analysis_evidence_candidates.json': {},
                '05_analysis/analysis_result.json': {
                    'analyses': [{
                        'analysis': {'analysis_id': 'a1', 'minimum_studies': 2},
                        'status': 'completed',
                        'selected_evidence_ids': ['e1'],
                        'poolability': {'study_effects': [{'evidence_id': 'e1', 'value': 3.0}]},
                        'statistics': {'k': 6, 'q': 14.0691, 'i_squared': 0.2269},
                    }],
                },
            }
            for name, value in files.items():
                p = run / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(value))
            dashboard = render_run_dashboard.data_for(run)
            self.assertEqual(dashboard['analyses'][0]['statistics']['i_squared'], 0.2269)
            self.assertEqual(dashboard['alerts'], [])
            page = render_run_dashboard.render(dashboard)
            self.assertIn('<title>TRACE · End-to-end run</title>', page)
            self.assertNotIn('expert_review/', page)
            self.assertNotIn('</script><img', page)

    def test_qa_uses_same_explicit_endpoint_and_env_file_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'provider.json'
            p.write_text(json.dumps({'text_provider': {'name': 'openai_compatible',
                                                       'base_url': 'https://config.invalid/v1', 'model': 'config'}}))
            env = Path(tmp) / 'keys.env'
            env.write_text('TEXT_PROVIDER_API_KEY=fixture\n'
                           'TEXT_PROVIDER_BASE_URL=https://selected.invalid/v1\n'
                           'TEXT_PROVIDER_MODEL=selected\n')
            with patch.dict(os.environ, {'TEXT_PROVIDER_MODEL': 'stale'}, clear=True), patch('openai.OpenAI') as client:
                t = Transport(p, env)
                t._initialize()
                self.assertEqual(t.base_url, 'https://selected.invalid/v1')
                self.assertEqual(t.model, 'selected')
                self.assertEqual(client.call_args.kwargs['api_key'], 'fixture')

    def test_migrated_request_keeps_source_paths_when_moved(self):
        from extensions.typed_migration import write_migration
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'migration'
            source = Path(tmp) / 'source'
            shutil.copytree(ROOT / 'vendor/integration/examples', source)
            req = source / 'request.json'
            raw = json.loads(req.read_text())
            raw.setdefault('options', {})['extraction_runtime'] = 'baseline'
            req.write_text(json.dumps(raw))
            write_migration(req, output)
            result = json.loads((output / 'request.typed.json').read_text())
            self.assertNotIn('provider', result['options'])
            self.assertNotIn('model', result['options'])
            self.assertTrue(all(Path(d['paths']['markdown']).is_absolute() for d in result['documents']))
            self.assertTrue(all(Path(d['paths']['markdown']).is_file() for d in result['documents']))


if __name__ == '__main__':
    unittest.main()
