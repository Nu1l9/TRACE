"""Public TRACE CLI routing, checkpoint isolation and read-only status contracts."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import trace_cli
import run_pipeline


class TraceCliTests(unittest.TestCase):
    def test_resume_forwards_request_without_changing_output(self):
        args = ['--request', 'request.json', '--output', 'runs/exact', '--env-file', '.env']
        with patch('run_pipeline.main', return_value=2) as launch:
            self.assertEqual(trace_cli.main(['resume', *args]), 2)
        launch.assert_called_once_with(['run', *args, '--resume'])
        self.assertEqual(run_pipeline.timestamped_output(launch.call_args.args[0]),
                         launch.call_args.args[0])

    def test_prepare_and_update_forward_all_options(self):
        for command, target, args in [('prepare', 'brain-fill', ['--topic', 'question', '--output', 'draft.json']),
                                      ('update', 'update', ['--base-run', 'old', '--delta', 'delta.json', '--output', 'new', '--plan-only'])]:
            with self.subTest(command=command), patch('run_pipeline.main', return_value=0) as launch:
                self.assertEqual(trace_cli.main([command, *args]), 0)
                launch.assert_called_once_with([target, *args])

    def test_help_has_no_output_or_qa_handoff_side_effects(self):
        with patch('extensions.__main__.main', return_value=0), \
                patch('run_pipeline.timestamped_output') as timestamp, \
                patch('expert_qa.integration.after_pipeline') as handoff:
            self.assertEqual(run_pipeline.main(['run', '--output', 'existing', '--help']), 0)
        timestamp.assert_not_called()
        handoff.assert_not_called()

    def test_qa_delegates_arguments_and_exit_status(self):
        with patch('expert_qa.__main__.main', return_value=1) as launch:
            self.assertEqual(trace_cli.main(['qa', 'inspect', '--run', 'saved']), 1)
        launch.assert_called_once_with(['inspect', '--run', 'saved'])

    def test_status_is_read_only_and_does_not_claim_process_liveness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'pipeline_manifest.json').write_text(json.dumps({'status':'running', 'stages':{'analysis':{'status':'running'}}}))
            (root/'dashboard.html').write_text('saved dashboard')
            before = {p.name:p.read_bytes() for p in root.iterdir()}
            with patch('run_pipeline.main', side_effect=AssertionError('must not launch')), contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(trace_cli.main(['status', '--run', directory]), 0)
            result = json.loads(output.getvalue())
            self.assertEqual(result['status'], 'running')
            self.assertIn('does not prove a live process', result['state_source'])
            self.assertEqual(len(result['dashboards']), 1)
            self.assertEqual(before, {p.name:p.read_bytes() for p in root.iterdir()})

    def test_invalid_manifest_fails_without_synthesizing_success(self):
        with tempfile.TemporaryDirectory() as directory:
            for data in ('{', '[]', '{"stages": {"analysis": "completed"}}'):
                (Path(directory)/'pipeline_manifest.json').write_text(data)
                with self.subTest(data=data), self.assertRaises(ValueError):
                    trace_cli.saved_status(directory)

    def test_equals_output_is_timestamped_for_new_runs_only(self):
        for args, changed in [(['run','--output=runs/new'], True),
                              (['run','--output=runs/exact','--resume'], False),
                              (['update','--output=runs/plan','--plan-only'], False)]:
            with patch('run_pipeline.require_local_runtime_sources'), patch('run_pipeline.dashboard_link'), \
                    patch('extensions.__main__.main', return_value=0) as run, \
                    patch('expert_qa.integration.after_pipeline'), contextlib.redirect_stdout(io.StringIO()):
                run_pipeline.main(args)
            forwarded = run.call_args.args[0]
            output = forwarded[forwarded.index('--output')+1]
            original = args[1].split('=',1)[1]
            self.assertEqual(output != original, changed)


if __name__ == '__main__':
    unittest.main()
