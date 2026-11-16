"""Public TRACE protocol-guided orchestrator CLI: route commands to stage runners."""
import argparse
import json
import sys
from pathlib import Path

COMMANDS = {
    'prepare': 'Prepare a protocol/request draft (brain-fill; inspect unresolved fields)',
    'check': 'Validate a prepared request and component configuration',
    'run': 'Run supplied candidates through A1/A2/A3',
    'review': 'Retrieve corpus candidates, then run the pipeline',
    'resume': 'Resume run with the same request, settings and exact output directory',
    'update': 'Plan or execute an update from a previous run',
    'status': 'Read saved stage states and dashboard locations; no model calls',
    'retry-analysis': 'Recover terminal analysis from saved reviewed evidence',
    'qa': 'Inspect, serve or open the saved-results assistant',
    'provider-check': 'Check configured provider connectivity',
    'rob-retry': 'Retry unresolved risk-of-bias assessments',
    'expert': 'Manage expert protocol/session review',
    'analysis': 'Run the analysis stage from prepared evidence',
    'prepare-corpus': 'Prepare a local literature corpus',
    'build-bm25': 'Build a local BM25 index',
    'build-dense': 'Build a dense retrieval index',
}


def saved_status(run):
    """Read persisted state only; it is not a process-liveness assertion."""
    run = Path(run).expanduser().resolve()
    path = run / 'pipeline_manifest.json'
    if getattr(path.stat(), 'st_flags', 0) & 0x40000000:
        raise ValueError('Download the run manifest from cloud storage first.')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(manifest, dict) or not isinstance(manifest.get('stages'), dict):
        raise ValueError('Invalid pipeline manifest: expected a stages object.')
    stages = manifest['stages']
    if any(not isinstance(v, dict) for v in stages.values()):
        raise ValueError('Invalid pipeline manifest: expected stage objects.')
    return {
        'run': str(run), 'case_id': manifest.get('case_id'),
        'status': manifest.get('status'), 'updated_at': manifest.get('updated_at'),
        'analysis_status': manifest.get('analysis_status'),
        'stages': {name: {key: row[key] for key in ('status', 'completed_at') if key in row}
                   for name, row in stages.items()},
        'dashboards': [str(run / name) for name in ('dashboard.html', 'dashboard_en.html')
                       if (run / name).is_file()],
        'state_source': 'saved_manifest; running status does not prove a live process',
    }


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description='TRACE protocol-guided orchestrator\n\n' + '\n'.join(f'  {k:18} {v}' for k, v in COMMANDS.items()),
        epilog='Use python trace_cli.py COMMAND --help for command-specific options. See docs/CLI.md.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=COMMANDS, help='command to execute')
    if not argv or argv[0] in ('-h', '--help'):
        parser.print_help()
        return 0
    command = parser.parse_args(argv[:1]).command
    args = argv[1:]
    if command == 'status':
        status_parser = argparse.ArgumentParser(prog='trace_cli.py status', description=COMMANDS['status'])
        status_parser.add_argument('--run', required=True, type=Path)
        parsed = status_parser.parse_args(args)
        try:
            print(json.dumps(saved_status(parsed.run), ensure_ascii=False, indent=2))
            return 0
        except (OSError, ValueError) as exc:
            status_parser.exit(1, f'Cannot read saved run status: {exc}\n')
    if command == 'qa':
        from expert_qa.__main__ import main as qa_main
        return qa_main(args)
    if command == 'resume':
        command = 'run'
        if '--resume' not in args:
            args.append('--resume')
    elif command == 'prepare':
        command = 'brain-fill'
    from run_pipeline import main as pipeline_main
    return pipeline_main([command, *args])


if __name__ == '__main__':
    raise SystemExit(main())
