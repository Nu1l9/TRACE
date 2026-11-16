"""Loopback-only UI. All run files are read-only; Q&A records live separately."""
from __future__ import annotations
import html
import json
import secrets
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from .agent import Agent, AnswerError

ASSETS = Path(__file__).parent / 'assets'

class Application:
    def __init__(self, snapshot, transport, log_dir):
        self.snapshot = snapshot
        self.agent = Agent(snapshot, transport)
        self.log_dir = Path(log_dir).expanduser().resolve()
        if self.log_dir == snapshot.run or snapshot.run in self.log_dir.parents:
            raise ValueError('Q&A log directory must be outside the original run')
        self.token = secrets.token_urlsafe(32)
        self.sessions, self.citations = {}, {}
        self.saved_answers = []
        self.busy = threading.Lock()
        # Restore only this module's completed records for the exact same snapshot.
        # This keeps existing Q&A and citation links usable when the UI is upgraded.
        if self.log_dir.is_dir():
            for path in sorted(self.log_dir.glob('*.json'), key=lambda p: p.stat().st_mtime)[-100:]:
                if len(path.stem) != 32 or any(c not in '0123456789abcdef' for c in path.stem):
                    continue
                try:
                    if path.stat().st_size > 5_000_000:
                        continue
                    answer = json.loads(path.read_text())
                    if answer.get('snapshot') != snapshot.fingerprint:
                        continue
                    aid, sid = answer['analysis_id'], answer['session_id']
                    if aid not in snapshot.analyses or not isinstance(sid, str):
                        continue
                    if not isinstance(answer['claims'], list) or not isinstance(answer['citations'], dict):
                        continue
                    session = self.sessions.setdefault(sid, {'analysis_id': aid, 'history': []})
                    if session['analysis_id'] != aid:
                        continue
                    session['history'] = [*session['history'][-2:], {'question': answer['question'], 'claims': answer['claims']}]
                    self.citations.update(answer['citations'])
                    answer['log_saved'] = True
                    self.saved_answers.append(answer)
                except (OSError, ValueError, KeyError, TypeError):
                    continue

    def ask(self, data, on_event=None):
        if not isinstance(data, dict) or set(data) - {'question', 'analysis_id', 'session_id'}:
            raise AnswerError('Invalid request format.')
        aid, question = data.get('analysis_id'), data.get('question')
        if not isinstance(aid, str) or aid not in self.snapshot.analyses:
            raise AnswerError('Select an analysis from this run.')
        sid = data.get('session_id')
        if sid is not None and (not isinstance(sid, str) or sid not in self.sessions):
            raise AnswerError('This session has expired. Start a new conversation.')
        if sid and self.sessions[sid]['analysis_id'] != aid:
            raise AnswerError('Start a new conversation when switching analyses to keep evidence separate.')
        if not sid:
            if len(self.sessions) >= 100:
                raise AnswerError('Session limit reached. Restart the Q&A service.')
            sid = secrets.token_urlsafe(24)
            self.sessions[sid] = {'analysis_id': aid, 'history': []}
        def progress(event):
            # Make newly retrieved pages available immediately, even if a later call fails.
            page = event.get('result', {})
            if page.get('citation_id'):
                self.citations[page['citation_id']] = page
            if on_event:
                on_event(event)
        answer = self.agent.answer(question, aid, self.sessions[sid]['history'], on_event=progress)
        answer['session_id'] = sid
        answer['created_at'] = datetime.now(timezone.utc).isoformat()
        answer['answer_id'] = secrets.token_hex(16)
        self.citations.update(answer['citations'])
        self.sessions[sid]['history'] = [*self.sessions[sid]['history'][-2:],
                                       {'question': question, 'claims': answer['claims']}]
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            path = self.log_dir / (answer['answer_id'] + '.json')
            # New file only, created after completion; never overwrite previous records.
            with path.open('x', encoding='utf-8') as handle:
                json.dump(answer, handle, ensure_ascii=False, indent=2)
            path.chmod(0o600)
            answer['log_saved'] = True
        except OSError:
            answer['log_saved'] = False
        self.saved_answers = [*self.saved_answers[-99:], answer]
        return answer

def make_server(app=None, port=8765, library=None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, status, body, content_type='application/json; charset=utf-8'):
            body = body if isinstance(body, bytes) else body.encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(body)

        def json(self, status, value):
            self.send(status, json.dumps(value, ensure_ascii=False))

        def valid_origin(self):
            allowed = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
            host = self.headers.get('Host', '')
            origin = self.headers.get('Origin')
            return host in allowed and (origin is None or origin == 'http://' + host)

        def context(self, path):
            if library is None:
                return app, path, '/'
            parts = path.split('/')
            if len(parts) < 4 or parts[1] != 'r':
                self.json(404, {'error': 'Study not found.'}); return None
            try:
                current = library.get(parts[2])
            except (OSError, ValueError, KeyError):
                self.json(409, {'error': 'These results are temporarily unavailable. They may be updating or not yet downloaded.'}); return None
            return current, '/' + '/'.join(parts[3:]), f'/r/{parts[2]}/'

        def do_GET(self):
            if not self.valid_origin():
                self.json(403, {'error': 'Local origin required'}); return
            path = urlparse(self.path).path
            if library is not None:
                if path == '/api/health':
                    from .integration import MARKER
                    self.json(200, {'service': MARKER, 'library': library.identity}); return
                if path == '/api/catalog':
                    self.json(200, {'runs': library.listing()}); return
                if path in ('/', '/library.js', '/style.css'):
                    name, typ = {'/': ('library.html', 'text/html'), '/library.js': ('library.js', 'text/javascript'),
                                 '/style.css': ('style.css', 'text/css')}[path]
                    self.send(200, (ASSETS / name).read_bytes(), typ + '; charset=utf-8'); return
            bound = self.context(path)
            if bound is None:
                return
            app, path, prefix = bound
            if path in ('/', '/app.js', '/style.css'):
                name = {'/': 'index.html', '/app.js': 'app.js', '/style.css': 'style.css'}[path]
                typ = {'/': 'text/html', '/app.js': 'text/javascript', '/style.css': 'text/css'}[path]
                self.send(200, (ASSETS / name).read_bytes(), typ + '; charset=utf-8'); return
            if path == '/api/snapshot':
                info = app.snapshot.info()
                info['csrf_token'] = app.token
                info['library_url'] = '/' if library is not None else None
                info['articles'] = {aid: doc.get('title') or aid for aid, doc in app.snapshot.documents.items()}
                info['evidence_articles'] = {eid: row.get('article_id') for eid, row in
                                            {**app.snapshot.evidence, **app.snapshot.candidates}.items()}
                info['analyses'] = [{**row, 'statistics': app.snapshot.analyses[row['id']].get('statistics'),
                                     'warnings': app.snapshot.analyses[row['id']].get('warnings', [])}
                                    for row in info['analyses']]
                self.json(200, info); return
            if path == '/api/history':
                aid = parse_qs(urlparse(self.path).query).get('analysis_id', [''])[0]
                if aid not in app.snapshot.analyses:
                    self.json(400, {'error': 'Unknown analysis'}); return
                answers = [a for a in app.saved_answers if a['analysis_id'] == aid]
                sid = answers[-1]['session_id'] if answers else None
                self.json(200, {'answers': [a for a in answers if a['session_id'] == sid][-20:]}); return
            if path.startswith('/citation/'):
                cid = path.removeprefix('/citation/')
                page = app.citations.get(cid)
                if page:
                    from .presentation import citation_html
                    body = citation_html(page, app.snapshot, prefix)
                    self.send(200, body, 'text/html; charset=utf-8'); return
            self.json(404, {'error': 'Not found'})

        def do_POST(self):
            if not self.valid_origin():
                self.json(403, {'error': 'Invalid request origin or session token.'}); return
            bound = self.context(urlparse(self.path).path)
            if bound is None:
                return
            app, path, prefix = bound
            if self.headers.get('X-QA-Token') != app.token:
                self.json(403, {'error': 'Invalid request origin or session token.'}); return
            if path != '/api/ask':
                self.json(404, {'error': 'Not found'}); return
            if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                self.json(415, {'error': 'JSON required'}); return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 24000:
                    raise ValueError()
                self.connection.settimeout(10)
                data = json.loads(self.rfile.read(length))
            except (ValueError, OSError):
                self.json(400, {'error': 'Invalid request format or size.'}); return
            if not app.busy.acquire(blocking=False):
                self.json(409, {'error': 'A question is already being processed. Please wait.'}); return
            streaming = self.headers.get('Accept') == 'application/x-ndjson'
            connected = True
            def stream(event):
                nonlocal connected
                if connected:
                    try:
                        self.wfile.write((json.dumps(event, ensure_ascii=False) + '\n').encode('utf-8'))
                        self.wfile.flush()
                    except OSError:
                        connected = False
            try:
                if streaming:
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/x-ndjson; charset=utf-8')
                    self.send_header('Cache-Control', 'no-store')
                    self.send_header('X-Content-Type-Options', 'nosniff')
                    self.send_header('Connection', 'close')
                    self.end_headers()
                    self.close_connection = True
                result = app.ask(data, on_event=stream if streaming else None)
                if streaming:
                    stream({'type': 'answer', 'answer': result})
                else:
                    self.json(200, result)
            except AnswerError as exc:
                if streaming:
                    stream({'type': 'error', 'error': str(exc)})
                else:
                    self.json(422, {'error': str(exc)})
            except Exception:
                error = 'The Q&A service could not complete this request. The original run is unchanged.'
                if streaming:
                    stream({'type': 'error', 'error': error})
                else:
                    self.json(500, {'error': error})
            finally:
                app.busy.release()

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    return server
