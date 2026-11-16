"""Incremental evidence updates over immutable baseline run directories.

Updates can start from earlier implementations. Only dependency-compatible,
hash-verified article artefacts are seeded; other stages run again in the new
output directory. The parent run is never rewritten.
"""
from __future__ import annotations
import copy
import hashlib
import os
import shutil
from collections import Counter
from pathlib import Path

from .common import ROOT, read, write, require, digest
from .runner import ExtensionRunner
from trace_runtime.contracts import normalize_request
from trace_runtime.pipeline import fingerprint, now

SCHEMA = 'trace_update/1.0'
STAGES = ('screening', 'screening_rob', 'extraction', 'review', 'analysis')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def implementation_versions():
    """Hash executable stage dependencies, including schemas and vendor prompts."""
    shared = ['extensions/input_contract.py', 'extensions/stage_inputs.py',
              'extensions/runner.py', 'extensions/common.py', 'extensions/settings.py',
              'vendor/integration/trace_runtime/pipeline.py',
              'extensions/update.py', 'extensions/provider.py', 'extensions/transport.py',
              'vendor/integration/trace_runtime/contracts.py']
    groups = {
        'screening': ['extensions/parallel_worker.py'],
        'screening_rob': ['extensions/screening_bias.py', 'extensions/rob_resilience.py',
                          'extensions/rob_schema.py', 'extensions/bias.py'],
        'extraction': ['extensions/parallel_worker.py', 'extensions/resilient_extraction.py'],
        'review': ['extensions/parallel_worker.py', 'vendor/integration/trace_runtime/evidence_context.py'],
        'analysis': ['extensions/analysis_worker.py', 'extensions/analysis_cache.py',
                     'extensions/statistical_methods.py', 'extensions/statistical_method_agent.py',
                     'extensions/analysis_checkpoints.py',
                     'extensions/r_runtime.py', 'extensions/rob_resilience.py', 'extensions/bias.py',
                     'extensions/graph_feedback.py', 'extensions/brain.py',
                     'vendor/integration/trace_runtime/evidence_context.py'],
    }
    vendor = ROOT / 'vendor/integration/vendor'
    stage_roots = {
        'screening': [vendor / 'miners'],
        'screening_rob': [vendor / 'miners', ROOT / 'resources/rob'],
        'extraction': [vendor / 'evidence_extraction/system'],
        'review': [vendor / 'evidence_extraction/system'],
        'analysis': [vendor / 'evidence_analysis'],
    }
    versions = {}
    for stage, names in groups.items():
        files = {ROOT / name for name in [*shared, *names]}
        for directory in stage_roots[stage]:
            require(directory.is_dir(), 'missing update stage dependency: ' + str(directory))
            files.update(file for file in directory.rglob('*') if file.is_file()
                         and file.suffix in ('.py', '.R', '.txt', '.json')
                         and '__pycache__' not in file.parts and not file.name.startswith('test_'))
        require(all(file.is_file() for file in files), 'missing executable update dependency for ' + stage)
        versions[stage] = digest({str(file.relative_to(ROOT)): sha(file) for file in sorted(files)})
    return versions


def load_baseline(path):
    base = Path(path).expanduser().resolve()
    state = read(base/'pipeline_manifest.json')
    require(state.get('mode') == 'live' and state.get('status') != 'running',
            'update requires a stopped live baseline run')
    request = read(base/'request.normalized.json')
    require(state.get('case_id', request['case_id']) == request['case_id'],
            'baseline request and manifest case identities differ')
    # Component and input fingerprints also include implementation code. A new
    # release must be allowed to update an old run. Those comparisons determine
    # reuse eligibility in make_plan, not whether an update may be started.
    for stage in state.get('stages', {}).values():
        if stage.get('status') != 'completed':
            continue
        for relative, expected in stage.get('outputs', {}).items():
            file = (base/relative).resolve()
            require(file.is_relative_to(base) and file.is_file() and sha(file) == expected,
                    'baseline checkpoint changed: '+relative)
    if state.get('stages', {}).get('analysis', {}).get('status') == 'completed':
        index = base/'05_analysis/extension_files.json'
        if index.is_file():
            for item in read(index)['files']:
                file = (base/'05_analysis'/item['path']).resolve()
                require(file.is_relative_to(base/'05_analysis') and file.is_file() and sha(file) == item['sha256'],
                        'baseline analysis artefact changed: '+item['path'])
    return base, request, state


def updated_request(base_request, *, request_file=None, delta_file=None):
    require(bool(request_file) != bool(delta_file), 'supply exactly one of --request or --delta')
    if request_file:
        path = Path(request_file).resolve()
        from .input_contract import normalize_request as compile_request
        result = normalize_request(compile_request(read(path)), path.parent)
    else:
        path = Path(delta_file).resolve()
        delta = read(path)
        require(delta.get('schema_version') == SCHEMA, 'delta schema_version must be '+SCHEMA)
        require(set(delta) <= {'schema_version', 'documents', 'remove_article_ids', 'topic', 'protocol', 'analysis_profile'},
                'unknown update delta field')
        if 'protocol' in delta and delta['protocol'] != base_request['protocol']:
            require('analysis_profile' in delta,
                    'Changing protocol in an update delta requires an explicit analysis_profile; inherited compiled conditions may be stale')
        result = copy.deepcopy(base_request)
        additions = delta.get('documents', [])
        removed = delta.get('remove_article_ids', [])
        require(isinstance(additions, list) and all(isinstance(d, dict) for d in additions), 'delta documents must be objects')
        ids = [d.get('article_id') for d in additions]
        require(all(isinstance(i, str) for i in ids) and len(ids) == len(set(ids)), 'duplicate/missing delta article IDs')
        require(isinstance(removed, list) and all(isinstance(i, str) for i in removed)
                and len(removed) == len(set(removed)), 'invalid remove_article_ids')
        original = {d['article_id']: d for d in result['documents']}
        require(set(removed) <= set(original), 'cannot remove an unknown baseline article')
        require(not set(ids) & set(removed), 'cannot add and remove the same article')
        original.update({d['article_id']: copy.deepcopy(d) for d in additions})
        result['documents'] = [d for i, d in original.items() if i not in removed]
        for key in ('topic', 'protocol', 'analysis_profile'):
            if key in delta:
                result[key] = copy.deepcopy(delta[key])
        from .input_contract import normalize_request as compile_request
        result = normalize_request(compile_request(result), path.parent)
    require(result['case_id'] == base_request['case_id'], 'update must retain the baseline case_id')
    return result


def document_signature(document):
    value = copy.deepcopy(document)
    # Paths remain significant: extraction and RoB citations must still resolve.
    value['source_sha256'] = {kind: sha(path) for kind, path in document['paths'].items()
                              if kind in ('markdown', 'content_list', 'pdf')}
    return digest(value)


def baseline_reuse_compatibility(base, request, state):
    """Conservatively invalidate reuse without blocking a new-version update."""
    lock_path = base/'extensions.lock.json'
    lock = read(lock_path) if lock_path.is_file() else {}
    components_match = lock.get('baseline_lock_hash') == sha(ROOT/'component_provenance.json')
    try:
        inputs_match = fingerprint(request) == state.get('input_fingerprint')
    except OSError:
        # An old source may have moved or been removed. The updated request
        # still has to resolve all its sources before it can be executed.
        inputs_match = False
    reasons = []
    if not components_match:
        reasons.append('frozen components differ or the baseline component binding is absent')
    if not inputs_match:
        reasons.append('baseline source/request/implementation fingerprint cannot be verified with this checkout')
    return {'components_match': components_match, 'input_binding_matches': inputs_match,
            'reuse_allowed': components_match and inputs_match, 'rerun_reasons': reasons}


def primary_target(request):
    tid = request['analysis_profile']['primary_analysis_spec']['target_id']
    return next(t for t in request['protocol']['targets'] if t['target_id'] == tid)


def make_plan(base, old, new, state, settings, profile):
    old_docs = {d['article_id']: d for d in old['documents']}
    new_docs = {d['article_id']: d for d in new['documents']}
    added = sorted(new_docs.keys()-old_docs.keys())
    removed = sorted(old_docs.keys()-new_docs.keys())
    def same_document(aid):
        try:
            return document_signature(new_docs[aid]) == document_signature(old_docs[aid])
        except OSError:
            return False
    changed = sorted(i for i in new_docs.keys() & old_docs.keys() if not same_document(i))
    stable = sorted((new_docs.keys() & old_docs.keys()) - set(changed))
    baseline_compatibility = baseline_reuse_compatibility(base, old, state)
    # Both request objects may point at the same revised file. Rehashing them
    # against each other cannot prove it still matches the saved run. Without
    # the historical binding, classify these documents as unverified and rerun.
    unverified = stable if not baseline_compatibility['input_binding_matches'] else []
    stable = stable if not unverified else []
    contents = {}
    for aid, document in new_docs.items():
        paths=document['paths']; signature=sha(paths.get('markdown') or paths.get('pdf') or paths['content_list'])
        if signature in contents:
            require(aid not in added and contents[signature] not in added,
                    'duplicate source under different article IDs: '+contents[signature]+', '+aid+
                    '; retain the original article_id for an existing paper')
        contents[signature]=aid
    old_profile = read(base/'provider_routes.json') if (base/'provider_routes.json').is_file() else {}
    old_settings = read(base/'extensions.normalized.json') if (base/'extensions.normalized.json').is_file() else {}
    previous_versions = read(base/'update_dependencies.json') if (base/'update_dependencies.json').is_file() else {}
    versions = implementation_versions()
    text_same = old_profile.get('text') == profile.get('text')
    protocol_same = old['protocol'] == new['protocol'] and old['topic'] == new['topic']
    options_same = {k:v for k,v in old['options'].items() if k not in ('max_workers','max_retries','provider','model')} == {
        k:v for k,v in new['options'].items() if k not in ('max_workers','max_retries','provider','model')}
    compatible = {
        'screening': protocol_same and text_same,
        'screening_rob': protocol_same and text_same and primary_target(old) == primary_target(new)
                         and old_settings.get('bias') == settings.get('bias'),
        'extraction': protocol_same and text_same and options_same and old['analysis_profile'] == new['analysis_profile'],
        'review': protocol_same and text_same and options_same,
        'analysis': protocol_same and text_same and options_same and old['analysis_profile'] == new['analysis_profile']
                    and old_settings == settings and old_profile.get('embedding') == profile.get('embedding'),
    }
    for stage in compatible:
        if not baseline_compatibility['reuse_allowed'] or previous_versions.get(stage) != versions[stage]:
            compatible[stage] = False
    # An implementation change in extraction also invalidates review reuse.
    compatible['review'] = compatible['review'] and compatible['extraction']
    reuse = {stage: stable if compatible[stage] and state.get('stages', {}).get(stage, {}).get('status') == 'completed' else []
             for stage in STAGES if stage != 'analysis'}
    ancestry = read(base/'update_manifest.json') if (base/'update_manifest.json').is_file() else {}
    return {'schema_version': SCHEMA, 'parent_run': str(base), 'generation': ancestry.get('generation', 0)+1,
            'root_run': ancestry.get('root_run', str(base)), 'case_id': new['case_id'],
            'base_input_fingerprint': state['input_fingerprint'], 'request_fingerprint': fingerprint(new),
            'base_status': state['status'], 'documents': {'added': added, 'changed': changed, 'removed': removed,
                                                        'unchanged': stable, 'unverified': unverified},
            'changed_fields': [key for key in ('topic','protocol','analysis_profile') if old[key] != new[key]],
            'compatible': compatible, 'reusable_articles': reuse,
            'implementation_versions': versions,
            'baseline_compatibility': baseline_compatibility,
            'legacy_baseline': not bool(previous_versions),
            'legacy_reuse_policy': 'no article reuse without matching implementation and input bindings',
            'analysis_policy': 'reuse only if all analysis inputs, providers and settings match; otherwise recompute merged evidence'}


def _relocate(value, old, new):
    if isinstance(value, dict):
        return {k: _relocate(v, old, new) for k,v in value.items()}
    if isinstance(value, list):
        return [_relocate(v, old, new) for v in value]
    return value.replace(str(old), str(new)) if isinstance(value, str) else value


def _extraction_reuse_fingerprints(request, config, article_id):
    """Rebuild deterministic source/task signatures without any model calls."""
    from .resilient_extraction import (_load_runtime, _call_static_clean,
                                       _source_fingerprint, _semantic_fingerprint)
    load_task, _, tools, _ = _load_runtime(request)
    task = load_task(config)
    is_typed = request['options'].get('extraction_runtime') == 'typed'
    catalog, _ = _call_static_clean(tools, task, task.documents[article_id], is_typed)
    return {'source_fingerprint': _source_fingerprint(catalog),
            'semantic_fingerprint': _semantic_fingerprint(task, tools, is_typed)}


def _verify_reuse_stage(base, request, plan, stage, article_ids):
    """Recheck provenance at seeding time, before trusting checkpoint records."""
    if not article_ids:
        return
    require(plan.get('compatible', {}).get(stage), 'update stage is not compatible for reuse: ' + stage)
    current = implementation_versions().get(stage)
    previous = read(base / 'update_dependencies.json').get(stage)
    require(previous == current == plan['implementation_versions'].get(stage),
            'update implementation changed before seeding: ' + stage)
    state = read(base / 'pipeline_manifest.json').get('stages', {}).get(stage, {})
    require(state.get('status') == 'completed' and state.get('outputs'),
            'update reuse requires hash-verified completed stage: ' + stage)
    for relative, expected in state['outputs'].items():
        file = (base / relative).resolve()
        require(file.is_relative_to(base.resolve()) and file.is_file() and sha(file) == expected,
                'baseline checkpoint changed before seeding: ' + relative)
    old = read(base / 'request.normalized.json')
    require(baseline_reuse_compatibility(base, old, read(base/'pipeline_manifest.json'))['reuse_allowed'],
            'baseline source/request/implementation changed before seeding; create a new update plan')
    old_docs = {item['article_id']: item for item in old['documents']}
    new_docs = {item['article_id']: item for item in request['documents']}
    for aid in article_ids:
        require(aid in old_docs and aid in new_docs and document_signature(old_docs[aid]) == document_signature(new_docs[aid]),
                'update reuse article source changed: ' + aid)
    if stage in ('screening', 'extraction'):
        from .stage_inputs import stage_signature
        directory = '01_screening' if stage == 'screening' else '02_extraction'
        lock = base / directory / (stage + '_inputs.lock.json')
        require(lock.is_file() and read(lock).get('fingerprint') == stage_signature(old, stage),
                'baseline has no matching ' + stage + ' input provenance; rerun this stage')


def _validate_article_reuse(source, article_id, request, config):
    from .resilient_extraction import STATE_SCHEMA
    manifest = read(source / 'run_manifest.json')
    state = read(source / 'resilient_article_state.json')
    require(manifest.get('case_id') == request['case_id'] and manifest.get('article_id') == article_id,
            'article extraction manifest identity mismatch: ' + article_id)
    require(state.get('schema_version') == STATE_SCHEMA and state.get('article_id') == article_id
            and state.get('status') == 'completed', 'article extraction checkpoint is not completed output: ' + article_id)
    expected = _extraction_reuse_fingerprints(request, config, article_id)
    require(all(state.get(key) == value for key, value in expected.items()),
            'article extraction source/task/schema changed: ' + article_id)
    files = {}
    for file in source.rglob('*'):
        if file.is_file():
            require(file.resolve().is_relative_to(source.resolve()), 'article artifact escapes its run directory')
            files[str(file.relative_to(source))] = sha(file)
    return {'article_id': article_id, **expected, 'files': files}


class UpdateRunner(ExtensionRunner):
    def __init__(self, request, output, environment, settings, *, plan, resume=False, provider_profile=None):
        self.update_plan = plan
        self.base_run = Path(plan['parent_run'])
        output = Path(output).resolve()
        require(output != self.base_run and not output.is_relative_to(self.base_run)
                and not self.base_run.is_relative_to(output), 'update output must be separate from baseline')
        if resume:
            saved = read(output/'update_manifest.json')
            operational = {'implementation_versions', 'compatible', 'reusable_articles'}
            require({k:v for k,v in saved['plan'].items() if k not in operational} ==
                    {k:v for k,v in plan.items() if k not in operational},
                    'update inputs/baseline changed; use a new output')
            # ExtensionRunner separately enforces the implementation/config
            # resume guard. Preserve the original reuse plan on continuation.
            self.update_plan = saved['plan']
        super().__init__(request, output, environment, settings, resume, provider_profile)
        if not resume:
            write(output/'update_manifest.json', {'schema_version': SCHEMA, 'generation': plan['generation'],
                'root_run': plan['root_run'], 'parent_run': str(self.base_run), 'created_at': now(), 'plan': plan})
            write(output/'update_dependencies.json', plan['implementation_versions'])
            write(output/'update_reuse.json', {})
        # Always snapshot comparison inputs before executing, never edit parent.
        self.old_summary = read(self.base_run/'05_analysis/analysis_summary.json') if (self.base_run/'05_analysis/analysis_summary.json').is_file() else {}

    def _record(self, stage, items):
        audit = read(self.out/'update_reuse.json')
        audit[stage] = {'reused_ids': items, 'count': len(items), 'parent_run': str(self.base_run)}
        write(self.out/'update_reuse.json', audit)
        print(f'[update] {stage}: reusing {len(items)} saved records', flush=True)

    def _seed(self, name):
        base, out = self.base_run, self.out
        ids = set(self.update_plan['reusable_articles'].get(name, []))
        if not ids:
            # A cross-version update may intentionally reuse nothing. Let the
            # normal runner initialise the stage rather than parsing old,
            # potentially incompatible per-stage output schemas.
            self._record(name, [])
            return
        _verify_reuse_stage(base, self.r, self.update_plan, name, ids)
        if name == 'screening':
            src = base/'01_screening/screening_results.json'
            rows = [r for r in read(src)['decisions'] if r['candidate_id'] in ids] if src.is_file() else []
            routes_file = base/'01_screening/risk_of_bias_recommendations.json'
            routes = [r for r in read(routes_file)['recommendations'] if r['article_id'] in ids] if routes_file.is_file() else []
            from .stage_inputs import bind_stage_inputs
            bind_stage_inputs(self.r, out/'01_screening', 'screening')
            write(out/'01_screening/screening_results.json', {'case_id': self.r['case_id'], 'decisions': rows})
            write(out/'01_screening/risk_of_bias_recommendations.json', {'recommendations': routes})
            if (base/'screening_resolutions.json').is_file() and ids:
                effective = read(base/'01_screening/effective_screening_results.json')
                require(digest(effective) == read(base/'pipeline_manifest.json').get('effective_screening_hash'),
                        'baseline resolved screening changed before update reuse')
                effective_by_id = {row['candidate_id']: row for row in effective['decisions']}
                resolutions = [r for r in read(base/'screening_resolutions.json') if r['article_id'] in ids]
                require(all(effective_by_id.get(r['article_id'], {}).get('human_resolution') == r
                            and effective_by_id[r['article_id']]['final_status'] == r['decision'] for r in resolutions),
                        'baseline human screening resolution changed before update reuse')
                write(out/'screening_resolutions.json', resolutions)
            self._record(name, [r['candidate_id'] for r in rows])
        elif name == 'screening_rob':
            from .screening_bias import assessment_units, partition_effective_routes
            from .bias import routed_tool, validate_assessments, SCHEMA as BIAS_SCHEMA
            included = [r['candidate_id'] for r in read(out/'01_screening/effective_screening_results.json')['decisions'] if r['final_status']=='Include']
            partition = partition_effective_routes(self.r, read(out/'01_screening/risk_of_bias_recommendations.json'), included)
            tools = {r['article_id']: routed_tool(r) for r in partition['routable']['recommendations']}
            src = base/'01_screening/risk_of_bias_assessments.json'
            rows = [a for a in read(src)['assessments'] if a['article_id'] in ids & set(included)
                    and a['article_id'] in tools and a['tool'] == tools[a['article_id']] and a['scope']['target_id'] == primary_target(self.r)['target_id']
                    and a['status'] == 'draft'] if src.is_file() else []
            payload = {'schema_version': BIAS_SCHEMA, 'case_id': self.r['case_id'], 'assessments': rows}
            checked = validate_assessments(payload, self.r, assessment_units(self.r, included))
            write(out/'01_screening/risk_of_bias.progress.json', checked)
            self._record(name, [a['assessment_id'] for a in rows])
        elif name == 'extraction':
            included = {r['candidate_id'] for r in read(out/'01_screening/effective_screening_results.json')['decisions'] if r['final_status']=='Include'}
            src = base/'02_extraction/case_run_manifest.json'
            rows = []
            planned_copies = []
            aggregate_path = base/'02_extraction/case_extraction_submission.json'
            aggregate = read(aggregate_path)['predictions'] if ids else []
            for row in read(src).get('article_runs', []) if src.is_file() else []:
                aid = row['article_id']
                if aid not in ids & included or row.get('status') != 'completed':
                    continue
                source = Path(row['run_dir']).resolve()
                require(source.is_relative_to(base/'02_extraction'), 'extraction reuse source outside parent run')
                require(read(source/'extraction_submission.json')['predictions'] ==
                        [p for p in aggregate if p['article_id'] == aid],
                        'article extraction differs from verified aggregate: '+aid)
                dest = out/'02_extraction/articles'/aid/'inherited'
                provenance = _validate_article_reuse(source, aid, self.r, out/'extraction_config/case_config.json')
                planned_copies.append((source, dest, provenance))
                inherited = copy.deepcopy(row); inherited['run_dir'] = str(dest)
                inherited['update_source_provenance'] = provenance
                rows.append(inherited)
            from .stage_inputs import bind_stage_inputs
            bind_stage_inputs(self.r, out/'02_extraction', 'extraction')
            for source, dest, provenance in planned_copies:
                shutil.copytree(source, dest, dirs_exist_ok=True)
                require(all((dest/relative).is_file() and sha(dest/relative) == expected
                            for relative, expected in provenance['files'].items()),
                        'article extraction copy verification failed: ' + provenance['article_id'])
            write(out/'02_extraction/extraction_progress.json', {'case_id': self.r['case_id'], 'article_runs': rows})
            self._record(name, [r['article_id'] for r in rows])
        elif name == 'review':
            old_predictions = read(base/'02_extraction/case_extraction_submission.json').get('predictions', []) if (base/'02_extraction/case_extraction_submission.json').is_file() else []
            current = read(out/'02_extraction/case_extraction_submission.json')['predictions']
            old_by_article = {}; new_by_article = {}
            for row in old_predictions: old_by_article.setdefault(row['article_id'], []).append(row)
            for row in current: new_by_article.setdefault(row['article_id'], []).append(row)
            reusable = {aid for aid in ids if old_by_article.get(aid) == new_by_article.get(aid)}
            src = base/'03_review/candidate_evidence_review_structured.json'
            rows = [r for r in read(src)['candidate_reviews'] if r['article_id'] in reusable] if src.is_file() else []
            write(out/'03_review/update_review_seed.json', {'candidate_reviews': rows, 'prediction_hashes': {
                r['prediction_id']: digest(r) for r in current if r['article_id'] in reusable}})
            self._record(name, [r['prediction_id'] for r in rows])

    def stage(self, name, command, outputs):
        # Seed once before the usual stage executes. Failed/resumed work owns
        # its checkpoints thereafter; never overwrite it with parent records.
        if name not in self.state['stages']:
            if name in self.update_plan['reusable_articles']:
                self._seed(name)
            if name == 'analysis' and self._reuse_analysis(outputs):
                return
        return super().stage(name, command, outputs)

    def _reuse_analysis(self, outputs):
        old = self.base_run/'05_analysis'
        if not self.update_plan['compatible']['analysis'] or self.old_summary.get('status') != 'completed':
            return False
        state = read(self.base_run/'pipeline_manifest.json')
        if state['stages'].get('analysis', {}).get('status') != 'completed': return False
        for name in ('extraction_protocol.json','analysis_task.json','analysis_profile_structured.json','analysis_evidence_candidates.json'):
            if read(self.base_run/'04_analysis_input'/name) != read(self.out/'04_analysis_input'/name): return False
        if read(self.base_run/'01_screening/risk_of_bias_assessments.json')['assessments'] != read(self.out/'01_screening/risk_of_bias_assessments.json')['assessments']: return False
        dest = self.out/'05_analysis'
        shutil.copytree(old, dest, dirs_exist_ok=True)
        for file in dest.rglob('*.json'):
            write(file, _relocate(read(file), self.base_run, self.out))
        for pattern in ('*.md','*.html','*.svg'):
            for file in dest.rglob(pattern):
                file.write_text(file.read_text().replace(str(self.base_run),str(self.out)), encoding='utf-8')
        write(dest/'extension_files.json', {'files': [{'path':str(f.relative_to(dest)), 'sha256':sha(f)}
              for f in sorted(dest.rglob('*')) if f.is_file() and f.name != 'extension_files.json']})
        required = [*outputs, dest/'extension_summary.json', dest/'extension_files.json']
        self.state['stages']['analysis'] = {'status':'completed','completed_at':now(),
            'outputs':{str(p.relative_to(self.out)):sha(p) for p in required}, 'reused_from':str(self.base_run)}
        self._record('analysis', ['complete_analysis']); self.save('running')
        return True

    def run(self):
        try:
            return super().run()
        finally:
            write_comparison(self.base_run, self.out)
            self._render_dashboard()


def incremental_review(request, config, submission, output):
    """Review only new/changed article batches and merge verified old reviews."""
    import sys
    from .parallel_worker import EXTRACT
    sys.path.insert(0, str(EXTRACT))
    from constrained_meta_agent.loader import load_case_config
    from evidence_reviewer.reviewer import review_submission, write_review_summary
    seed = read(output/'update_review_seed.json')
    data = read(submission); predictions = data['predictions']
    by_id = {p['prediction_id']:p for p in predictions}
    require(len(by_id) == len(predictions), 'duplicate update prediction IDs')
    prior = {}
    for row in seed['candidate_reviews']:
        pid = row['prediction_id']
        require(pid in by_id and digest(by_id[pid]) == seed['prediction_hashes'][pid], 'update review seed changed')
        require(row['article_id'] == by_id[pid]['article_id'], 'update review article mismatch')
        prior[pid] = row
    # Saved successful delta batches survive a later batch failure.
    progress = output/'update_review_progress.json'
    if progress.is_file():
        cached = read(progress)
        require(cached['submission_hash'] == digest(data), 'review resume submission changed')
        prior.update({r['prediction_id']:r for r in cached['candidate_reviews']})
    pending = [p for p in predictions if p['prediction_id'] not in prior]
    from .parallel_worker import _review_workers, _env_float
    result = review_submission(load_case_config(config), {**data, 'predictions':pending},
        configured_model=request['options']['model'], batch_size=12,
        max_workers=_review_workers(request, len(pending)), max_retries=request['options']['max_retries'],
        retry_delay_seconds=_env_float('TRACE_REVIEW_RETRY_DELAY_SECONDS', 60.0))
    prior.update({r['prediction_id']:r for r in result['candidate_reviews']})
    write(progress, {'submission_hash':digest(data), 'candidate_reviews':list(prior.values())})
    result.update(candidate_reviews=[prior[p['prediction_id']] for p in predictions if p['prediction_id'] in prior],
                  input_prediction_count=len(predictions), completed_review_count=len(prior),
                  reused_review_count=len(predictions)-len(pending))
    result['disposition_counts'] = dict(Counter(r['review']['disposition'] for r in result['candidate_reviews']))
    write(output/'candidate_evidence_review_structured.json', result)
    write_review_summary(output/'review_summary.md', result)
    require(not result['failed_review_count'], 'update candidate review failed; completed batches saved; use --resume')
    write(output/'review_complete.json', {'case_id':request['case_id'], 'reviewed_count':len(prior)})


def selected_population(run, analysis):
    """Sum only explicit sample counts, one selected estimate per study."""
    path=run/'04_analysis_input/analysis_evidence_candidates.json'
    candidates=read(path)['candidates'] if path.is_file() else []
    selected=set(analysis.get('selected_candidate_ids', []))
    rows=[c for c in candidates if c['candidate_id'] in selected]
    studies={c['study_id'] for c in rows}
    samples=[]
    for c in rows:
        raw=c.get('raw_data') or {}
        values=None
        if c.get('data_family') == 'binary_arm_counts':
            values=[raw.get('intervention_total'),raw.get('control_total')]
        elif c.get('data_family') == 'single_arm_event_counts':
            values=[raw.get('total')]
        if values and all(type(v) in (int,float) and v >= 0 and float(v).is_integer() for v in values):
            samples.append(int(sum(values)))
    known=bool(rows) and len(rows)==len(studies)==len(selected)==len(samples)
    return {'selected_studies':len(studies), 'selected_evidence':len(selected),
            'sample_size':sum(samples) if known else None,
            'sample_size_basis':'explicit counts; one estimate per supplied independent study ID' if known else 'unknown_or_overlapping'}


def interval_position(result):
    if result.get('effect_type') in ('RR','OR','HR','IRR'): null=1
    elif result.get('effect_type') in ('MD','SMD','RD'): null=0
    else: return 'not_applicable'
    low,high=result.get('ci_lower'),result.get('ci_upper')
    if not isinstance(low,(int,float)) or not isinstance(high,(int,float)): return 'unavailable'
    return 'below_null' if high < null else 'above_null' if low > null else 'includes_null'


def write_comparison(base, output):
    def summaries(run):
        p = run/'05_analysis/analysis_summary.json'
        return {a['analysis_id']:a for a in read(p).get('analyses', [])} if p.is_file() else {}
    before, after = summaries(base), summaries(output)
    rows = []
    for aid in sorted(before.keys() | after.keys()):
        b, a = before.get(aid, {}), after.get(aid, {})
        bp, ap = b.get('pooled_result') or {}, a.get('pooled_result') or {}
        comparable = bool(bp and ap and bp.get('effect_type') == ap.get('effect_type'))
        def difference(key):
            x,y=bp.get(key),ap.get(key)
            return y-x if comparable and isinstance(x,(int,float)) and isinstance(y,(int,float)) else None
        rows.append({'analysis_id':aid,'before':b,'after':a,'same_effect_scale':comparable,
                     'before_population':selected_population(base,b), 'after_population':selected_population(output,a),
                     'before_ci_relation':interval_position(bp), 'after_ci_relation':interval_position(ap),
                     'effect_change':difference('pooled_effect'), 'i2_change':difference('heterogeneity_i2'),
                     'direction_changed':bp.get('direction') != ap.get('direction') if comparable else None,
                     'interpretation':'Descriptive comparison only; changes in criteria or methods also affect results.'})
    def counts(run):
        p=run/'01_screening/effective_screening_results.json'
        dec=read(p)['decisions'] if p.is_file() else []
        p=run/'04_analysis_input/analysis_evidence_candidates.json'
        candidates=read(p)['candidates'] if p.is_file() else []
        return {'screened_articles':len(dec), 'included_articles':sum(r['final_status']=='Include' for r in dec),
                'evidence_records':len(candidates), 'evidence_studies':len({c['study_id'] for c in candidates})}
    payload={'schema_version':'trace_update_comparison/1.0','parent_run':str(base),'updated_run':str(output),
             'status':read(output/'pipeline_manifest.json')['status'], 'before_counts':counts(base),
             'after_counts':counts(output),'analyses':rows,
             'sample_size_note':'Participant totals are not summed across overlapping results; inspect selected study data.',
             'plan':read(output/'update_manifest.json')['plan']}
    write(output/'update_comparison.json',payload)
    lines=['# Evidence update comparison', '', f'Parent: {base}', f'Update: {output}', f'Status: {payload["status"]}', '',
           '| Metric | Before | After |','|---|---:|---:|']
    for k,v in payload['before_counts'].items(): lines.append(f'| {k} | {v} | {payload["after_counts"][k]} |')
    lines += ['', '| Analysis | Studies (before → after) | N (before → after) | Effect (before → after) | 95% CI (before → after) | I² (before → after) |', '|---|---|---|---|---|---|']
    for row in rows:
        b=row['before'].get('pooled_result') or {}; a=row['after'].get('pooled_result') or {}
        lines.append(f'| {row["analysis_id"]} | {row["before_population"]["selected_studies"]} → {row["after_population"]["selected_studies"]} | '
                     f'{row["before_population"]["sample_size"]} → {row["after_population"]["sample_size"]} | '
                     f'{b.get("pooled_effect")} → {a.get("pooled_effect")} | '
                     f'[{b.get("ci_lower")}, {b.get("ci_upper")}] → [{a.get("ci_lower")}, {a.get("ci_upper")}] | '
                     f'{b.get("heterogeneity_i2")} → {a.get("heterogeneity_i2")} |')
    lines += ['', 'Descriptive comparison: criteria and method changes can also change results. Missing estimates are not zero. Participant counts are not summed across overlapping results.']
    (output/'update_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return payload


def run_update(base_run, output, *, request_file=None, delta_file=None, settings_file=None,
               env_file=None, workers=4, resume=False, plan_only=False):
    from .provider import runtime, selection_values, file_values
    from .settings import load_settings
    from .document_preparation import prepare_request
    from trace_runtime.pipeline import preflight_runtime
    base, old, state = load_baseline(base_run)
    output=Path(output).expanduser().resolve()
    require(output != base and not output.is_relative_to(base) and not base.is_relative_to(output),
            'update output must be separate from baseline')
    if not plan_only:
        require(resume or not output.exists() or not any(output.iterdir()),
                'update output already exists; use --resume or a new directory')
    require(1 <= workers <= 8, '--workers must be between 1 and 8')
    new = updated_request(old, request_file=request_file, delta_file=delta_file)
    settings = load_settings(settings_file) if settings_file else read(base/'extensions.normalized.json')
    require(settings['bias']['mode']=='draft' and settings['bias'].get('auto_sensitivity'), 'update requires mandatory integrated RoB policy')
    environment = dict(os.environ)
    if env_file and not plan_only: environment.update(file_values(env_file))
    environment['TRACE_WORKERS'] = str(workers)
    new, preparation = prepare_request(new, environment, allow_conversion=not plan_only)
    new['options'].update(route_rob=True, max_workers=workers)
    if plan_only:
        # A plan never reads private .env or calls a model/conversion service.
        values=selection_values()
        profile={key:{field:values[prefix+field.upper()] for field in ('name','base_url','model')}
                 for key,prefix in [('text','TEXT_PROVIDER_'),('embedding','EMBEDDING_PROVIDER_')]}
        new['options'].update(provider=profile['text']['name'],model=profile['text']['model'])
        for document in new['documents']:
            if document['paths'].get('markdown') or document['paths'].get('content_list'): document['paths'].pop('pdf',None)
    else:
        new,environment,profile = runtime(new,env_file)
        environment['TRACE_ROB_WORKERS']=str(min(workers,2))
        environment['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
    plan=make_plan(base,old,new,state,settings,profile)
    if plan_only:
        return {'status':'update_plan','remote_calls':0,'plan':plan,'document_preparation':preparation,
                'note':'Recomputed at execution after PDF preparation and private environment provider overrides.'}
    if resume:
        saved=read(Path(output)/'request.normalized.json')
        new['options']['max_workers']=saved['options']['max_workers']
        plan=make_plan(base,old,new,state,settings,profile)
    preflight_runtime(new,environment)
    runner=UpdateRunner(new,output,environment,settings,plan=plan,resume=resume,provider_profile=profile)
    write(runner.out/'document_preparation.json',preparation)
    return runner.run()
