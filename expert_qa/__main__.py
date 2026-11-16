import argparse
import json
from pathlib import Path
from .store import Snapshot

def main(argv=None):
    from .integration import ROOT
    root = ROOT
    parser = argparse.ArgumentParser(description='Independent read-only expert Q&A for saved TRACE runs')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('inspect', 'serve'):
        p = sub.add_parser(name)
        p.add_argument('--run', required=True, type=Path)
        if name == 'serve':
            p.add_argument('--port', type=int, default=8765)
            p.add_argument('--env-file', type=Path)
            p.add_argument('--provider-config', type=Path, default=root / 'config/provider.json')
            p.add_argument('--log-dir', type=Path, default=root / 'expert_qa_sessions')
    from .integration import DATA, PORT
    p = sub.add_parser('library', help='Serve the integrated results workspace')
    p.add_argument('--data-dir', type=Path, default=DATA)
    p.add_argument('--port', type=int, default=PORT)
    for name in ('register', 'open'):
        p = sub.add_parser(name)
        group = p.add_mutually_exclusive_group(required=True)
        group.add_argument('--run', type=Path)
        if name == 'register':
            group.add_argument('--registry', type=Path)
            p.add_argument('--install-entries', action='store_true')
        p.add_argument('--env-file', type=Path)
        p.add_argument('--data-dir', type=Path, default=DATA)
    args = parser.parse_args(argv)
    try:
        if args.command == 'library':
            from .library import Library
            from .server import make_server
            server = make_server(port=args.port, library=Library(args.data_dir))
            print(f'Results workspace: http://127.0.0.1:{server.server_port}', flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
            return
        if args.command in ('register', 'open'):
            from .integration import register_run, install_entry, ensure_service, healthy
            runs = [args.run] if args.run else [Path(row['run']) for row in json.loads(args.registry.read_text())['runs']]
            for run in runs:
                record = register_run(run, env_file=args.env_file, data_dir=args.data_dir)
                if not record:
                    print(f'No final results available: {run.name}'); continue
                if args.command == 'open' or args.install_entries:
                    install_entry(run, record, data_dir=args.data_dir)
                print(f"Registered {record['case_id']} · {len(record['analyses'])} analyses")
                if args.command == 'open':
                    if not ensure_service(args.data_dir):
                        raise ValueError('The local workspace has not started. Check whether port 8770 is in use.')
                    import time, webbrowser
                    for _ in range(20):
                        if healthy(args.data_dir):
                            break
                        time.sleep(0.2)
                    if not healthy(args.data_dir):
                        raise ValueError(f'Could not start the local workspace. See {args.data_dir / "service.log"}')
                    webbrowser.open(f"http://127.0.0.1:{PORT}/r/{record['id']}/")
            return
        snapshot = Snapshot(args.run)
        if args.command == 'inspect':
            print(json.dumps(snapshot.info(), ensure_ascii=False, indent=2)); return
        from .provider import Transport
        from .server import Application, make_server
        app = Application(snapshot, Transport(args.provider_config, args.env_file), args.log_dir)
        server = make_server(app, args.port)
        print(f'Expert Q&A: http://127.0.0.1:{server.server_port}', flush=True)
        print(f'Case: {snapshot.case_id}; analyses: {len(snapshot.analyses)}; snapshot: {snapshot.fingerprint[:12]}', flush=True)
        print('Read-only snapshot loaded. Model calls begin only when a question is submitted. Ctrl-C stops this service.', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Cannot start expert Q&A ({type(exc).__name__}): {exc}\n')

if __name__ == '__main__':
    main()
