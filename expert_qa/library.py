"""Lazy, per-run applications behind one local research library."""
import hashlib
import threading
import json
from pathlib import Path
from .integration import DATA, ROOT, read
from .store import Snapshot, FILES
from .provider import Transport

class Library:
    def __init__(self, data_dir=DATA, transport_factory=None, log_dir=None):
        self.data_dir = Path(data_dir).resolve()
        self.identity = hashlib.sha256(str(self.data_dir).encode()).hexdigest()
        self.apps = {}; self.lock = threading.Lock()
        self.transport_factory = transport_factory
        self.log_dir = Path(log_dir) if log_dir else self.data_dir/'sessions'
        if log_dir is None:
            # Preserve the original independent-module conversations; copy, never move.
            for path in (ROOT/'expert_qa_sessions').glob('*.json'):
                if len(path.stem) != 32 or any(c not in '0123456789abcdef' for c in path.stem):
                    continue
                target = self.log_dir/path.name
                if target.exists():
                    continue
                try:
                    value = read(path)
                    if not value.get('snapshot') or not value.get('answer_id'):
                        continue
                    from .integration import atomic
                    atomic(target, json.dumps(value, ensure_ascii=False, indent=2))
                except (OSError, ValueError):
                    continue

    def records(self):
        rows = []
        for path in (self.data_dir / 'runs').glob('*.json'):
            try:
                row = read(path)
                if Path(row['run']).is_dir():
                    rows.append(row)
            except (OSError, ValueError, KeyError):
                continue
        return sorted(rows, key=lambda r: (r['case_id'], r.get('completed_at', ''), r.get('run_name', '')))

    def listing(self):
        return [{k: row[k] for k in ('id', 'case_id', 'topic', 'documents', 'analyses', 'status', 'completed_at', 'run_name')}
                for row in self.records()]

    def get(self, key):
        if len(key) != 24 or any(c not in '0123456789abcdef' for c in key):
            raise ValueError('Unknown research result')
        row = read(self.data_dir / 'runs' / (key + '.json'))
        run = Path(row['run'])
        # A rerun of the same directory replaces the cached app only after it is terminal.
        manifest = read(run / FILES['manifest'])
        if manifest.get('status') == 'running' or (manifest.get('stages', {}).get('analysis') or {}).get('status') == 'running':
            raise ValueError('This result is currently being updated; wait for analysis to finish')
        stamps = tuple((name, (run / path).stat().st_mtime_ns, (run / path).stat().st_size)
                       for name, path in FILES.items() if (run / path).is_file())
        with self.lock:
            cached = self.apps.get(key)
            if cached and cached[0] == stamps:
                return cached[1]
            from .server import Application
            snapshot = Snapshot(run)
            local_config = self.data_dir / 'provider.json'
            transport = self.transport_factory(row) if self.transport_factory else Transport(
                local_config if local_config.is_file() else ROOT/'config/provider.json', row.get('env_file'))
            app = Application(snapshot, transport, self.log_dir)
            self.apps[key] = (stamps, app)
            return app
