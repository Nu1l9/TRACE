import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from extensions.parallel_worker import EXTRACT
sys.path.insert(0, str(EXTRACT))
from evidence_reviewer.models import CandidateReviewBatch
from extensions.resumable_review import run_review


class ResumableReviewTests(unittest.TestCase):
    def test_partial_batch_survives_failure_and_only_failed_rows_resume(self):
        task = SimpleNamespace(raw={'execution_policy': {'llm_timeout_seconds': 110}},
            execution_policy={'llm_timeout_seconds': 110}, case_id='test', task_id='test_review')
        payload = {'predictions': [{'prediction_id': str(i), 'article_id': 'a'} for i in range(10)]}
        calls = []

        def call(task, rows, *args):
            calls.append([r['prediction_id'] for r in rows])
            self.assertEqual(task.execution_policy['llm_timeout_seconds'], 300)
            if rows[0]['prediction_id'] == '5' and len(calls) == 2:
                raise TimeoutError('simulated')
            return CandidateReviewBatch(reviews=[dict(prediction_id=r['prediction_id'],
                disposition='keep_as_valid_candidate', evidence_role='main_analysis_candidate', reason='Supported')
                for r in rows]), 1

        with tempfile.TemporaryDirectory() as td, patch(
                'evidence_reviewer.reviewer.review_batch_with_retries', side_effect=call):
            args = dict(model='test', workers=1, max_retries=0, retry_delay=0)
            first = run_review(task, payload, Path(td), **args)
            self.assertEqual(first['completed_review_count'], 5)
            self.assertEqual(first['failed_review_count'], 5)
            second = run_review(task, payload, Path(td), **args)
            self.assertEqual(second['completed_review_count'], 10)
            self.assertEqual(second['failed_review_count'], 0)
            self.assertEqual(calls, [['0','1','2','3','4'], ['5','6','7','8','9'], ['5','6','7','8','9']])
            run_review(task, payload, Path(td), **args)
            self.assertEqual(len(calls), 3)
            changed = copy.deepcopy(payload)
            changed['predictions'][0]['raw_data'] = {'n': 10}
            with self.assertRaisesRegex(ValueError, 'review inputs or contract changed'):
                run_review(task, changed, Path(td), **args)
        self.assertEqual(task.execution_policy['llm_timeout_seconds'], 110)

    def test_unbound_legacy_results_require_provenance_validation(self):
        task = SimpleNamespace(raw={}, execution_policy={}, case_id='test', task_id='review')
        with tempfile.TemporaryDirectory() as td:
            (Path(td)/'candidate_evidence_review_structured.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'verified extraction provenance'):
                run_review(task, {'predictions': []}, Path(td), model='test', workers=1,
                           max_retries=0, retry_delay=0)
