"""Retrieve once, freeze candidates, then reuse the existing screening-to-analysis path."""
from __future__ import annotations

from copy import deepcopy
from contextlib import closing
import hashlib
import json
from pathlib import Path
import os
import tempfile

from .dataset import DocumentStore


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix='.retrieval-')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
        os.replace(temp, path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def stamp(path):
    path = Path(path).resolve(); s = path.stat()
    return {'path': str(path), 'size': s.st_size, 'mtime_ns': s.st_mtime_ns}


def store_binding(store):
    """Reject moved/changed source files before using stored byte offsets."""
    path = store._corpus_path()
    with closing(store.connect()) as c:
        row = c.execute("SELECT value FROM metadata WHERE key='retrieval_binding'").fetchone()
    if row is None:
        raise ValueError('Store has no corpus binding; run prepare-corpus --rebuild.')
    binding = json.loads(row[0])
    if stamp(path) != binding['corpus_stat']:
        raise ValueError('Corpus moved or changed; rebuild the store and its indexes.')
    return {'store_sha256': digest(store.db_path), **binding}


def prepare_store(corpus, store_path, rebuild=False):
    corpus = Path(corpus).resolve(); store_path = Path(store_path).resolve()
    if store_path.exists() and not rebuild:
        store = DocumentStore(store_path); binding = store_binding(store)
        if binding['corpus_stat']['path'] != str(corpus):
            raise ValueError('Store belongs to another corpus; use --rebuild.')
        return {'documents': store.count(), 'store': str(store_path), 'reused': True}
    before = stamp(corpus); sha = digest(corpus)
    store = DocumentStore.build(corpus, store_path, rebuild=rebuild)
    if stamp(corpus) != before:
        raise ValueError('Corpus changed during preparation; rebuild before use.')
    with closing(store.connect()) as c:
        c.execute('INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)',
                  ('retrieval_binding', json.dumps({'corpus_stat': before, 'corpus_sha256': sha})))
        c.commit()
    return {'documents': store.count(), 'store': str(store_path), 'reused': False}


def index_files(directory):
    return {p.name: stamp(p) for p in sorted(Path(directory).iterdir())
            if p.is_file() and p.name != 'input_binding.json'}


def check_index(directory, binding, expected=None):
    path = Path(directory) / 'input_binding.json'
    if not path.is_file():
        raise ValueError('Index has no input binding; rebuild it with this launcher.')
    saved = read(path)
    if saved['store'] != binding or saved['files'] != index_files(directory):
        raise ValueError('Index or corpus changed; rebuild the index before retrieval.')
    if expected is not None and saved['settings'] != expected:
        raise ValueError('Index settings/model changed; use --rebuild.')
    return saved


def build_index(kind, store_path, directory, *, rebuild=False, model='BAAI/bge-small-en-v1.5', batch_size=32, min_df=2):
    from .indexes import BM25Index, DenseIndex
    store = DocumentStore(store_path); binding = store_binding(store)
    directory = Path(directory).resolve()
    settings = {'kind': kind, **({'model': model} if kind == 'dense' else {'min_df': min_df})}
    if directory.exists() and any(directory.iterdir()) and not rebuild:
        check_index(directory, binding, settings)
        return {'index': str(directory), 'reused': True}
    if kind == 'dense':
        DenseIndex.build(store, directory, model_name=model, batch_size=batch_size, rebuild=True)
    else:
        BM25Index.build(store, directory, min_df=min_df, rebuild=True)
    if store_binding(store) != binding:
        raise ValueError('Store changed during indexing; rebuild the index.')
    atomic(directory / 'input_binding.json', {'store': binding, 'settings': settings, 'files': index_files(directory)})
    return {'index': str(directory), 'reused': False}


def field_text(value):
    if isinstance(value, dict): return str(value.get('description') or value.get('label') or '')
    if isinstance(value, list): return '; '.join(field_text(v) for v in value)
    return str(value or '')


def query_task(request):
    pico = request['protocol']['pico']
    return {'Research_Question': request['topic'],
            'Population': field_text(pico.get('population')),
            'Intervention': field_text(pico.get('intervention_or_exposure') or pico.get('intervention') or pico.get('exposure')),
            'Comparison': field_text(pico.get('comparison') or pico.get('comparator')),
            'Outcome': field_text(pico.get('outcome'))}


def validate_template(raw, base):
    # Use the same actual contract as run; the temporary document is never written.
    if raw.get('documents'):
        raise ValueError('review expects documents omitted or empty; use run for supplied candidates.')
    if any(k in raw for k in ('matched_corpus_ids', 'gold', 'expected_label', 'positive_ids', 'negative_ids')):
        raise ValueError('Evaluation labels are not allowed in a review request.')
    from extensions.input_contract import normalize_request
    from trace_runtime.contracts import normalize_request as validate
    probe = deepcopy(raw)
    probe['documents'] = [{'article_id': 'retrieval_probe', 'document_id': 'retrieval_probe',
                           'study_id': 'retrieval_probe', 'paths': {'markdown': 'probe.md'}}]
    validate(normalize_request(probe), base, check_paths=False)


def source_document(record, corpus_base, output_sources, override=None, map_base=None):
    """No abstract-only fallback, guessed study grouping, or sample labels."""
    cid = int(record['ID']); aid = f'corpus_{cid}'
    override = override or {}
    study_id = override.get('study_id') or record.get('study_id')
    if not study_id:
        raise ValueError('study_id required: provide independent-study grouping in corpus or document map')
    paths = {}
    for source, base in [(record.get('paths') or {}, corpus_base),
                         (override.get('paths') or {}, map_base or corpus_base)]:
        for key in ('markdown', 'pdf', 'content_list'):
            if source.get(key):
                p = Path(source[key]).expanduser()
                paths[key] = str((base / p).resolve() if not p.is_absolute() else p.resolve())
    for key, path in paths.items():
        if not Path(path).is_file(): raise ValueError(f'missing {key} source: {path}')
    if not paths.get('markdown') and not paths.get('pdf'):
        sections = {}
        for key in ('introduction', 'methods', 'results', 'discussion', 'conclusion'):
            if isinstance(record.get(key), str) and record[key].strip(): sections[key] = record[key]
        for key, value in (record.get('sections') or {}).items():
            if isinstance(value, str) and value.strip(): sections[key] = value
        if not sections: raise ValueError('full text required: provide sections, Markdown or PDF')
        output_sources.mkdir(parents=True, exist_ok=True)
        p = output_sources / (aid + '.md')
        p.write_text('\n\n'.join(f'## {k}\n\n{v}' for k, v in sections.items()), encoding='utf-8')
        paths['markdown'] = str(p.resolve())
    return {'article_id': aid, 'document_id': aid, 'study_id': str(study_id),
            'title': str(record.get('title') or aid), 'abstract': str(record.get('abstract') or ''), 'paths': paths}


def prepare_review(args):
    request_path = args.request.resolve(); raw = read(request_path)
    validate_template(raw, request_path.parent)
    if not (1 <= args.final_k <= args.fusion_k and args.per_query_k >= 1):
        raise ValueError('Require 1 <= final-k <= fusion-k and per-query-k >= 1.')
    config = {k: getattr(args, k) for k in ('mode','per_query_k','fusion_k','final_k','max_year','reranker_model')}
    signature = {'template': raw, 'settings': config,
                 'document_map': read(args.document_map) if args.document_map else None,
                 'document_map_base': str(args.document_map.resolve().parent) if args.document_map else None}
    out = args.output.resolve() / '00_retrieval'
    final = out / 'request.json'; audit = out / 'retrieval.json'
    if audit.exists():
        if not args.resume: raise ValueError('Retrieval output exists; use --resume or a new directory.')
        saved = read(audit)
        if saved['input'] != signature: raise ValueError('Retrieval task changed; use a new run.')
        if not saved['ready']: raise ValueError('Retrieval awaits sources/study identities; complete inputs and use a new run.')
        if digest(final) != saved['request_sha256']:
            raise ValueError('Frozen retrieval request changed; use a new run.')
        for path, sha in saved['source_hashes'].items():
            if digest(path) != sha: raise ValueError('Frozen candidate source changed; use a new run.')
        print('[retrieval] Reusing frozen candidate set; no retrieval repeated.', flush=True)
        return final
    if args.resume: raise ValueError('No completed retrieval snapshot to resume.')
    from .indexes import BM25Index, DenseIndex, CrossEncoderReranker
    from .retrieval import HybridRetriever, StructuredQueryPlanner
    store = DocumentStore(args.store.resolve()); binding = store_binding(store)
    indexes = {}; bindings = {}
    for kind, cls in [('bm25', BM25Index), ('dense', DenseIndex)]:
        if args.mode in (kind, 'hybrid'):
            path = getattr(args, kind + '_index').resolve()
            bindings[kind] = check_index(path, binding)
            indexes[kind] = cls(path)
    retriever = HybridRetriever(store, **indexes, reranker=CrossEncoderReranker(args.reranker_model) if args.reranker_model else None)
    task = query_task(raw); variants = StructuredQueryPlanner().plan(task)
    if not variants: raise ValueError('No usable retrieval query.')
    candidates = retriever.retrieve(task, variants=variants, per_query_k=args.per_query_k,
                                    fusion_k=args.fusion_k, final_k=args.final_k, max_year=args.max_year)
    documents = []; missing = []
    mapping = signature['document_map'] or {}
    if not isinstance(mapping, dict): raise ValueError('Document map must be an object keyed by corpus ID.')
    for candidate in candidates:
        record = store.get_source_record(candidate.corpus_id)
        if not record or int(record['ID']) != candidate.corpus_id:
            raise ValueError('Corpus byte-offset identity mismatch; rebuild the store.')
        try:
            documents.append(source_document(record, store._corpus_path().parent, out / 'sources',
                             mapping.get(str(candidate.corpus_id)), args.document_map.resolve().parent if args.document_map else None))
        except ValueError as exc:
            missing.append({'corpus_id': candidate.corpus_id, 'reason': str(exc)})
    if store_binding(store) != binding: raise ValueError('Corpus changed during retrieval.')
    ready = bool(candidates) and not missing
    report = {'input': signature, 'store_binding': binding, 'index_bindings': bindings,
              'queries': [v.__dict__ for v in variants], 'candidates': [c.to_dict() for c in candidates],
              'candidate_count': len(candidates), 'missing_inputs': missing, 'ready': ready,
              'status': 'ready' if ready else ('needs_sources_or_study_ids' if candidates else 'no_candidates'),
              'source_hashes': {}}
    if ready:
        request = deepcopy(raw); request['documents'] = documents
        # Resolve original request semantics before handing off to another directory.
        from extensions.__main__ import normalize_request
        request = normalize_request(request, request_path.parent)
        atomic(final, request)
        report['request_sha256'] = digest(final)
        report['source_hashes'] = {p: digest(p) for d in documents for p in d['paths'].values()}
    atomic(audit, report)
    print(f'[retrieval] candidates={len(candidates)}; ready={ready}; audit={audit}', flush=True)
    if not ready and not args.prepare_only:
        raise ValueError('Retrieval saved but downstream not started; inspect missing_inputs in retrieval.json.')
    return final if ready else None


def register_commands(sub):
    p = sub.add_parser('prepare-corpus', help='Prepare a local literature store for retrieval')
    p.add_argument('--corpus', type=Path, required=True); p.add_argument('--store', type=Path, required=True)
    p.add_argument('--rebuild', action='store_true')
    for kind in ('bm25', 'dense'):
        p = sub.add_parser('build-' + kind, help='Build a corpus-bound retrieval index')
        p.add_argument('--store', type=Path, required=True); p.add_argument('--index-dir', type=Path, required=True)
        p.add_argument('--rebuild', action='store_true')
        if kind == 'dense':
            p.add_argument('--model', default='BAAI/bge-small-en-v1.5'); p.add_argument('--batch-size', type=int, default=32)
        else: p.add_argument('--min-df', type=int, default=2)
    p = sub.add_parser('review', help='Retrieve candidates, then use the existing screening-to-analysis pipeline')
    p.add_argument('--request', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--store', type=Path, default=Path('cache/literature.sqlite'))
    p.add_argument('--bm25-index', type=Path, default=Path('cache/bm25'))
    p.add_argument('--dense-index', type=Path, default=Path('cache/dense'))
    p.add_argument('--mode', choices=['bm25','dense','hybrid'], default='hybrid')
    p.add_argument('--per-query-k', type=int, default=1000); p.add_argument('--fusion-k', type=int, default=500)
    p.add_argument('--final-k', type=int, default=200); p.add_argument('--max-year', type=int)
    p.add_argument('--reranker-model'); p.add_argument('--document-map', type=Path)
    p.add_argument('--prepare-only', action='store_true'); p.add_argument('--resume', action='store_true')
    p.add_argument('--env-file', type=Path); p.add_argument('--extensions', type=Path)
    p.add_argument('--workers', type=int, default=4); p.add_argument('--retry-deferred', action='store_true')


def validate_seed(output, request):
    """Allow only a verified retrieval seed in an otherwise new run directory."""
    out = Path(output) / '00_retrieval'
    saved = read(out / 'retrieval.json')
    if not saved.get('ready') or digest(out / 'request.json') != saved.get('request_sha256'):
        raise ValueError('Invalid retrieval seed.')
    frozen = read(out / 'request.json')
    for key in ('case_id', 'protocol', 'analysis_profile'):
        if frozen[key] != request[key]: raise ValueError('Retrieval seed does not match the scientific task.')
    def ids(r):
        return [(d['article_id'], d['study_id'], d['document_id']) for d in r['documents']]
    if ids(frozen) != ids(request): raise ValueError('Retrieval candidate identities changed.')
    for original, current in zip(frozen['documents'], request['documents']):
        for key, path in original['paths'].items():
            # Preparation deliberately drops the archival PDF when usable text exists.
            # Still verify its frozen bytes below; never allow replacement by another PDF.
            if (key == 'pdf' and key not in current['paths'] and
                    (current['paths'].get('markdown') or current['paths'].get('content_list'))):
                continue
            if current['paths'].get(key) != path:
                raise ValueError('Retrieval candidate source path changed.')
    for path, sha in saved['source_hashes'].items():
        if digest(path) != sha: raise ValueError('Retrieval source changed.')


def execute(args, scientific_main):
    if args.command == 'prepare-corpus':
        result = prepare_store(args.corpus, args.store, args.rebuild)
    elif args.command in ('build-bm25', 'build-dense'):
        result = build_index(args.command[6:], args.store, args.index_dir, rebuild=args.rebuild,
                             model=getattr(args,'model',''), batch_size=getattr(args,'batch_size',32), min_df=getattr(args,'min_df',2))
    else:
        request = prepare_review(args)
        if args.prepare_only: return 0
        argv = ['run', '--request', str(request), '--output', str(args.output), '--workers', str(args.workers)]
        for name in ('env_file','extensions'):
            if getattr(args,name): argv += ['--' + name.replace('_','-'), str(getattr(args,name))]
        for name in ('resume','retry_deferred'):
            if getattr(args,name): argv += ['--' + name.replace('_','-')]
        return scientific_main(argv)
    print(json.dumps(result, indent=2)); return 0
