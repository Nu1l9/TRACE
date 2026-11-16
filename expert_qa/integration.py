"""Post-run presentation integration. Never changes scientific artifacts or run locks."""
from __future__ import annotations
import hashlib
import html
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import build_opener, ProxyHandler, Request
from .store import load_file
from .presentation import analysis_title

PACKAGE_PARENT = Path(__file__).resolve().parent.parent

def project_root(package_parent=PACKAGE_PARENT, prefix=None):
    """Use this checkout's configuration; never discover another local project."""
    return Path(package_parent).resolve()

ROOT = project_root()
def default_data_dir():
    override = os.environ.get('TRACE_QA_DATA_DIR')
    if override:
        return Path(override).expanduser()
    if sys.platform == 'darwin':
        return Path.home() / 'Library/Application Support/trace/result_guide'
    if os.name == 'nt':
        return Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData/Local')) / 'trace/result_guide'
    return Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local/share')) / 'trace/result_guide'

DATA = default_data_dir()
PORT = 8770
MARKER = 'trace-result-guide-v1'

def read(path):
    return json.loads(load_file(path))

def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.guide-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            handle.write(value)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)

def register_run(run, *, env_file=None, data_dir=DATA):
    run = Path(run).expanduser().resolve()
    if not all((run / p).is_file() for p in ('pipeline_manifest.json', 'request.normalized.json', '05_analysis/analysis_result.json')):
        return None
    manifest = read(run / 'pipeline_manifest.json')
    stage = (manifest.get('stages') or {}).get('analysis', {})
    # A failed/blocked final analysis can still have useful, explicitly incomplete results.
    if manifest.get('status') == 'running' or stage.get('status') == 'running':
        return None
    request = read(run / 'request.normalized.json')
    result = read(run / '05_analysis/analysis_result.json')
    rows = result.get('analyses') or []
    if not rows:
        return None
    key = hashlib.sha256(str(run).encode()).hexdigest()[:24]
    # Preserve bookmarks when an existing run was moved behind a compatibility
    # symlink; registering it again must not create a second library entry.
    for candidate in (Path(data_dir) / 'runs').glob('*.json'):
        try:
            existing = read(candidate)
            if (existing.get('id') == candidate.stem and len(candidate.stem) == 24
                    and all(c in '0123456789abcdef' for c in candidate.stem)
                    and Path(existing['run']).expanduser().resolve() == run):
                key = candidate.stem
                break
        except (OSError, ValueError, KeyError):
            continue
    record_path = Path(data_dir) / 'runs' / (key + '.json')
    previous = read(record_path) if record_path.exists() else {}
    record = {'id': key, 'run': str(run), 'case_id': request.get('case_id', run.name),
              'topic': request.get('topic') or 'Research results', 'documents': len(request.get('documents', [])),
              'run_name': run.name, 'status': manifest.get('status', 'unknown'),
              'completed_at': stage.get('completed_at') or manifest.get('updated_at') or '',
              'env_file': str(Path(env_file).expanduser().resolve()) if env_file else previous.get('env_file'),
              'analyses': [{'id': r['analysis']['analysis_id'],
                            'title': analysis_title(r['analysis']),
                            'status': r.get('status', 'unknown')} for r in rows if r.get('analysis', {}).get('analysis_id')]}
    if record != previous:
        atomic(record_path, json.dumps(record, ensure_ascii=False, indent=2))
    return record

def install_entry(run, record, *, data_dir=DATA):
    """Only add presentation files and a marked navigation link; leave stage data unchanged."""
    run = Path(run)
    url = f'http://127.0.0.1:{PORT}/r/{record["id"]}/'
    # Native launcher is the explicit fallback after reboot, not browser script execution.
    command = '#!/bin/sh\nset -e\nexport PYTHONPATH=' + shlex.quote(str(PACKAGE_PARENT)) + '\ncd ' + shlex.quote(str(ROOT)) + '\nexec ' + shlex.join([
        sys.executable, '-m', 'expert_qa', 'open', '--run', str(run.resolve()),
        '--data-dir', str(Path(data_dir).resolve())]) + '\n'
    atomic(run / 'Open_results.command', command)
    (run / 'Open_results.command').chmod(0o700)
    title = html.escape(record['topic'])
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Explore results · TRACE</title><style>body{{font:17px/1.8 system-ui;color:#213d40;background:#f5f7f3;margin:0}}main{{max-width:760px;margin:12vh auto;padding:40px}}small{{color:#517477}}h1{{font-size:38px}}a{{display:inline-block;background:#23675d;color:white;text-decoration:none;padding:12px 24px;border-radius:10px;margin:16px 12px 16px 0}}.sub{{background:white;color:#23675d;border:1px solid #ccdcd6}}aside{{color:#657b77;font-size:14px;margin-top:32px}}</style>
<main><small>TRACE · RESEARCH WORKSPACE</small><h1>Trace research results to source evidence</h1><p>{title}</p>
<p>Explore results, ask about study selection and check the source evidence for each conclusion.</p><a id="enter" href="{url}">Explore results →</a><a class="sub" href="dashboard.html">Back to full report</a>
<aside>If the local service is not running, double-click Open_results.command in this run folder. Opening the page does not call the model.</aside></main>
<script>const q=new URLSearchParams(location.search);if(q.has('analysis'))document.getElementById('enter').href+=('?analysis='+encodeURIComponent(q.get('analysis')));</script></html>'''
    atomic(run / 'result_guide.html', page)
    for name, label in [('dashboard.html', 'Explore results & evidence')]:
        path = run / name
        if not path.exists():
            continue
        content = path.read_text(encoding='utf-8')
        if 'id="result-guide-entry"' not in content:
            backup = path.with_name(path.stem + '.pre_result_guide.html')
            if not backup.exists():
                atomic(backup, content)
            nav = ('<nav id="result-guide-entry" style="padding:14px 24px;background:#f0f5ee;border-bottom:1px solid #d8e4dc">'
                   '<a href="result_guide.html" target="_blank" rel="noopener" style="color:#245d53;font-weight:650;text-decoration:none">'
                   + label + ' →</a></nav>')
            atomic(path, content.replace('<body>', '<body>' + nav, 1))
    return url

def healthy(data_dir=DATA):
    try:
        opener = build_opener(ProxyHandler({}))
        with opener.open(Request(f'http://127.0.0.1:{PORT}/api/health'), timeout=0.5) as response:
            result = json.load(response)
        return result.get('service') == MARKER and result.get('library') == hashlib.sha256(str(Path(data_dir).resolve()).encode()).hexdigest()
    except (OSError, ValueError):
        return False

def ensure_service(data_dir=DATA):
    if healthy(data_dir):
        return True
    try:
        import fcntl
    except ImportError:
        # Direct `python -m expert_qa library` remains available on Windows.
        return False
    data_dir = Path(data_dir); data_dir.mkdir(parents=True, exist_ok=True)
    # Concurrent pipelines can complete together; only one should start the local library.
    with (data_dir / 'service.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        if healthy(data_dir):
            return True
        # Do not start duplicates or collide with an unrelated process on this port.
        import socket
        with socket.socket() as probe:
            try:
                probe.bind(('127.0.0.1', PORT))
            except OSError:
                return False
        with (data_dir / 'service.log').open('ab') as log:
            env = os.environ.copy()
            env['PYTHONPATH'] = str(PACKAGE_PARENT)
            process = subprocess.Popen([sys.executable, '-m', 'expert_qa', 'library',
                              '--data-dir', str(data_dir.resolve()), '--port', str(PORT)],
                             cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                             start_new_session=True, close_fds=True)
        # Keep the startup lock until the child binds, so simultaneous completions
        # cannot launch a second copy during the process-start window.
        import time
        for _ in range(20):
            if healthy(data_dir):
                return True
            if process.poll() is not None:
                return False
            time.sleep(0.1)
        return False

def after_pipeline(arguments):
    """Called only after the scientific CLI returns. Failure must not change its exit status."""
    if not arguments or arguments[0] not in ('run', 'retry-analysis', 'update') or '--output' not in arguments:
        return
    try:
        run = Path(arguments[arguments.index('--output') + 1]).expanduser().resolve()
        env = arguments[arguments.index('--env-file') + 1] if '--env-file' in arguments else None
        record = register_run(run, env_file=env)
        if record:
            url = install_entry(run, record)
            started = ensure_service()
            print('[results] ' + (url if started else str(run / 'result_guide.html')), flush=True)
    except (OSError, ValueError, KeyError, IndexError) as exc:
        print(f'[results] Entry could not be refreshed ({type(exc).__name__}); saved analysis results are unaffected.', flush=True)
