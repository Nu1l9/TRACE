"""Explicit, source-bound import of completed legacy run artifacts into a fresh run.

Only final article submissions are imported. Graph locator/batch state is deliberately
not copied because its extraction contract cannot be resumed in place.
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from .common import digest, read, require, write
from .stage_inputs import stage_signature


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_legacy_run(request: dict, source: Path) -> dict:
    source = Path(source).resolve()
    manifest = read(source / 'pipeline_manifest.json')
    previous = read(source / 'request.normalized.json')
    require(manifest.get('case_id') == request['case_id'] == previous.get('case_id'),
            'legacy run case_id differs from the current request')
    require(previous.get('protocol') == request.get('protocol'),
            'legacy protocol differs from the current request; screening/RoB cannot be reused')
    old_docs = {row['article_id']: row for row in previous['documents']}
    new_docs = {row['article_id']: row for row in request['documents']}
    require(len(old_docs) == len(previous['documents']) == len(new_docs) == len(request['documents'])
            and old_docs.keys() == new_docs.keys(), 'legacy document registry differs from the current request')
    document_hashes = {}
    for aid, old in old_docs.items():
        new = new_docs[aid]
        require(all(old.get(key) == new.get(key) for key in
                    ('document_id', 'study_id', 'title', 'abstract')),
                f'{aid}: legacy document identity differs')
        hashes = {}
        for kind in ('markdown', 'content_list'):
            old_path = Path(old.get('paths', {}).get(kind, ''))
            new_path = Path(new.get('paths', {}).get(kind, ''))
            require(old_path.is_file() and new_path.is_file(),
                    f'{aid}: prepared {kind} missing in legacy or the current request')
            require(_sha(old_path) == _sha(new_path),
                    f'{aid}: prepared {kind} changed since legacy run')
            hashes[kind] = _sha(old_path)
        document_hashes[aid] = hashes
    stages = manifest.get('stages', {})
    for stage in ('screening', 'screening_rob'):
        entry = stages.get(stage, {})
        require(entry.get('status') == 'completed' and entry.get('outputs'),
                f'legacy {stage} is not complete')
        for relative, expected in entry['outputs'].items():
            path = (source / relative).resolve()
            require(path.is_relative_to(source) and path.is_file() and _sha(path) == expected,
                    f'legacy {stage} artifact missing or changed: {relative}')
    screening = read(source / '01_screening/effective_screening_results.json')
    decisions = screening.get('decisions', [])
    require(len(decisions) == len(new_docs) and
            {row.get('candidate_id') for row in decisions} == set(new_docs) and
            all(row.get('final_status') in ('Include', 'Exclude') for row in decisions),
            'legacy screening is incomplete or has unresolved decisions')
    included = {row['candidate_id'] for row in decisions if row['final_status'] == 'Include'}
    rob = read(source / '01_screening/risk_of_bias_complete.json')
    require(rob.get('status') == 'complete' and rob.get('failed') == 0 and
            rob.get('uncovered') == 0 and rob.get('article_count') == len(included),
            'legacy screening RoB is incomplete')
    progress = read(source / '02_extraction/extraction_progress.json')
    require(progress.get('case_id') == request['case_id'], 'legacy extraction case_id differs')
    target_ids = {row['target_id'] for row in request['protocol']['targets']}
    article_runs = []
    prediction_ids = set()
    for row in progress.get('article_runs', []):
        aid = row.get('article_id')
        require(aid in included and aid not in {item['article_id'] for item in article_runs}
                and row.get('status') == 'completed', 'invalid legacy completed article')
        run_dir = (source / '02_extraction/articles' / aid / 'attempt_01').resolve()
        require(run_dir.is_relative_to(source) and Path(row.get('run_dir', '')).resolve() == run_dir,
                f'{aid}: legacy extraction run path differs')
        state = read(run_dir / 'resilient_article_state.json')
        submission_path = run_dir / 'extraction_submission.json'
        submission = read(submission_path)
        require(state.get('schema_version') == 'trace_legacy/article_extraction_state/1.0'
                and state.get('article_id') == aid and state.get('status') == 'completed',
                f'{aid}: legacy article is not complete')
        predictions = submission.get('predictions')
        require(submission.get('case_id') == request['case_id'] and isinstance(predictions, list),
                f'{aid}: invalid legacy final submission')
        require(state.get('prediction_count') == len(predictions),
                f'{aid}: legacy state/submission count differs')
        for prediction in predictions:
            pid = prediction.get('prediction_id')
            require(isinstance(pid, str) and pid and pid not in prediction_ids and
                    prediction.get('article_id') == aid and
                    prediction.get('study_id') == new_docs[aid]['study_id'] and
                    prediction.get('target_id') in target_ids and
                    isinstance(prediction.get('raw_data'), dict),
                    f'{aid}: invalid or duplicate legacy prediction')
            prediction_ids.add(pid)
        article_runs.append({'article_id': aid, 'source_dir': str(run_dir),
                             'submission_sha256': _sha(submission_path),
                             'prediction_count': len(predictions), 'issues': row.get('issues', [])})
    require(progress.get('prediction_count') == len(prediction_ids),
            'legacy extraction progress count differs from article submissions')
    return {'source_run': str(source), 'case_id': request['case_id'],
            'included_article_ids': sorted(included), 'document_hashes': document_hashes,
            'article_runs': article_runs,
            'pending_article_ids': sorted(included - {row['article_id'] for row in article_runs})}


def import_legacy_run(request: dict, runner, inspection: dict) -> None:
    source = Path(inspection['source_run'])
    destination = runner.out
    screening_dir = destination / '01_screening'
    extraction_dir = destination / '02_extraction'
    require(not screening_dir.exists() and not extraction_dir.exists(),
            'legacy import requires untouched screening and extraction directories')
    shutil.copytree(source / '01_screening', screening_dir)
    for stage, directory in (('screening', screening_dir), ('extraction', extraction_dir)):
        write(directory / f'{stage}_inputs.lock.json',
              {'schema_version': 'stage_inputs/8.0', 'fingerprint': stage_signature(request, stage)})
    resolution = source / 'screening_resolutions.json'
    if resolution.is_file():
        shutil.copy2(resolution, destination / 'screening_resolutions.json')
    old_manifest = read(source / 'pipeline_manifest.json')
    for stage in ('screening', 'screening_rob'):
        outputs = {}
        for relative in old_manifest['stages'][stage]['outputs']:
            outputs[relative] = _sha(destination / relative)
        runner.state['stages'][stage] = {
            'status': 'completed',
            'completed_at': old_manifest['stages'][stage].get('completed_at'),
            'outputs': outputs, 'reused_from': str(source),
        }
    runs = []
    for item in inspection['article_runs']:
        aid = item['article_id']
        new_dir = extraction_dir / 'articles' / aid / 'attempt_01'
        new_dir.mkdir(parents=True)
        shutil.copy2(Path(item['source_dir']) / 'extraction_submission.json',
                     new_dir / 'extraction_submission.json')
        require(_sha(new_dir / 'extraction_submission.json') == item['submission_sha256'],
                f'{aid}: final submission changed during migration')
        runs.append({'article_id': aid, 'status': 'completed', 'run_dir': str(new_dir),
                     'issues': item['issues']})
    write(extraction_dir / 'extraction_progress.json', {
        'case_id': request['case_id'], 'article_runs': runs,
        'prediction_count': sum(row['prediction_count'] for row in inspection['article_runs']),
    })
    write(destination / 'legacy_run_migration_audit.json', {
        'schema_version': 'legacy_run_migration/1.0', **inspection,
        'policy': 'Graph screening/RoB and final per-article extraction submissions imported after source and identity checks; Graph locator/batch checkpoints are not reused; pending articles, review and analysis run in Current.',
    })
    runner.state['effective_screening_hash'] = digest(read(screening_dir / 'effective_screening_results.json'))
    runner.save('paused')
    print(f"[pipeline] imported legacy screening/RoB and {len(runs)} final extraction submissions; pending={inspection['pending_article_ids']}", flush=True)
