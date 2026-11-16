"""Cross-version updates rerun incompatible work without trusting stale sources."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from extensions import update
from trace_runtime import pipeline


class CrossVersionUpdateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root/'baseline'
        self.base.mkdir()
        self.checkout = self.root/'current'
        self.integration = self.checkout/'vendor/integration'
        (self.integration/'trace_runtime').mkdir(parents=True)
        self.code = self.integration/'trace_runtime/pipeline.py'
        self.code.write_text('# original implementation\n')
        update.write(self.integration/'source_archives.json', {'source': 'original'})
        self.components = self.checkout/'component_provenance.json'
        update.write(self.components, {'components': 'original'})
        self.source = self.root/'article.md'
        self.source.write_text('events: 2/10\n')
        self.request = {'case_id': 'quickstart', 'topic': 'Example',
                        'documents': [{'article_id': 'article_a', 'study_id': 'study_a',
                                       'paths': {'markdown': str(self.source)}}],
                        'protocol': {'targets': [{'target_id': 'mortality'}]},
                        'analysis_profile': {'primary_analysis_spec': {'target_id': 'mortality'}},
                        'options': {'max_workers': 1}}
        self.profile = {'text': {'model': 'test'}, 'embedding': {'model': 'test'}}
        self.settings = {'bias': {'mode': 'draft'}}
        self.versions = {stage: 'original-'+stage for stage in update.STAGES}
        for target, value in [('extensions.update.ROOT', self.checkout),
                              ('trace_runtime.pipeline.ROOT', self.integration),
                              ('extensions.update.implementation_versions', lambda: self.versions.copy())]:
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.state = {'mode': 'live', 'status': 'completed', 'case_id': 'quickstart',
                      'input_fingerprint': pipeline.fingerprint(self.request), 'stages': {}}
        for stage in update.STAGES:
            path = self.base/(stage+'.json')
            update.write(path, {'case_id': 'quickstart', 'stage': stage})
            self.state['stages'][stage] = {'status': 'completed', 'outputs': {path.name: update.sha(path)}}
        update.write(self.base/'pipeline_manifest.json', self.state)
        update.write(self.base/'request.normalized.json', self.request)
        update.write(self.base/'extensions.lock.json', {'baseline_lock_hash': update.sha(self.components)})
        update.write(self.base/'provider_routes.json', self.profile)
        update.write(self.base/'extensions.normalized.json', self.settings)
        update.write(self.base/'update_dependencies.json', self.versions)

    def plan(self, request=None):
        base, old, state = update.load_baseline(self.base)
        return update.make_plan(base, old, request or copy.deepcopy(old), state, self.settings, self.profile)

    def assert_no_reuse(self, plan):
        self.assertFalse(any(plan['compatible'].values()))
        self.assertFalse(any(plan['reusable_articles'].values()))

    def test_same_release_and_sources_can_reuse(self):
        plan = self.plan()
        self.assertTrue(all(plan['compatible'].values()))
        self.assertEqual(plan['reusable_articles']['screening'], ['article_a'])
        self.assertEqual(plan['documents']['unverified'], [])

    def test_new_components_allow_update_and_invalidate_reuse(self):
        update.write(self.components, {'components': 'changed'})
        plan = self.plan()
        self.assert_no_reuse(plan)
        self.assertFalse(plan['baseline_compatibility']['components_match'])

    def test_changed_pipeline_code_allows_update_and_invalidates_reuse(self):
        self.code.write_text('# newer implementation\n')
        plan = self.plan()
        self.assert_no_reuse(plan)
        self.assertEqual(plan['documents']['unverified'], ['article_a'])

    def test_stage_change_invalidates_only_dependent_stages(self):
        self.versions['extraction'] = 'new-extraction'
        plan = self.plan()
        self.assertTrue(plan['compatible']['screening'])
        self.assertFalse(plan['compatible']['extraction'])
        self.assertFalse(plan['compatible']['review'])
        self.assertEqual(plan['reusable_articles']['screening'], ['article_a'])

    def test_same_path_revised_source_cannot_reuse_saved_decisions(self):
        self.source.write_text('events: 9/10\n')
        plan = self.plan()
        self.assert_no_reuse(plan)
        self.assertEqual(plan['documents']['unchanged'], [])
        self.assertEqual(plan['documents']['unverified'], ['article_a'])

    def test_relocated_source_can_replace_missing_old_source(self):
        revised = copy.deepcopy(self.request)
        new_source = self.root/'revised.md'
        new_source.write_text(self.source.read_text())
        revised['documents'][0]['paths']['markdown'] = str(new_source)
        self.source.unlink()
        plan = self.plan(revised)
        self.assert_no_reuse(plan)
        self.assertEqual(plan['documents']['changed'], ['article_a'])

    def test_legacy_component_binding_absent_allows_rerun(self):
        (self.base/'extensions.lock.json').unlink()
        self.assert_no_reuse(self.plan())

    def test_changed_completed_output_is_still_rejected(self):
        update.write(self.base/'screening.json', {'tampered': True})
        with self.assertRaisesRegex(ValueError, 'baseline checkpoint changed'):
            self.plan()

    def test_source_change_after_planning_is_rejected_before_seeding(self):
        plan = self.plan()
        self.source.write_text('events: 9/10\n')
        with self.assertRaisesRegex(ValueError, 'changed before seeding'):
            update._verify_reuse_stage(self.base, self.request, plan, 'review', ['article_a'])

    def test_update_planning_keeps_parent_files_immutable(self):
        before = {str(p.relative_to(self.base)): p.read_bytes() for p in self.base.rglob('*') if p.is_file()}
        self.code.write_text('# newer implementation\n')
        self.plan()
        after = {str(p.relative_to(self.base)): p.read_bytes() for p in self.base.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_no_reuse_skips_old_stage_payload_parsing(self):
        runner = object.__new__(update.UpdateRunner)
        runner.base_run = self.base
        runner.out = self.root/'updated'
        runner.update_plan = {'reusable_articles': {stage: [] for stage in update.STAGES}}
        with patch.object(runner, '_record') as record, patch('extensions.update.read') as read:
            for stage in update.STAGES[:-1]:
                runner._seed(stage)
            self.assertEqual(record.call_count, 4)
            read.assert_not_called()


if __name__ == '__main__':
    unittest.main()
