"""Input-bound, atomic evidence-review checkpoints for the outer scheduler."""
from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .common import digest, read, require


def atomic_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def validate_legacy_review(request, submission_path, output):
    """Only import legacy successes from the same frozen, completed extraction."""
    from .stage_inputs import stage_signature
    root = Path(output).parent
    require(read(root / 'request.normalized.json') == request,
            'review recovery requires the frozen run request')
    manifest = read(root / 'pipeline_manifest.json')
    for stage, directory in (('screening', '01_screening'), ('extraction', '02_extraction')):
        lock = read(root / directory / f'{stage}_inputs.lock.json')
        require(lock['fingerprint'] == stage_signature(request, stage),
                f'review recovery: {stage} inputs changed')
    entry = manifest['stages']['extraction']
    require(entry['status'] == 'completed', 'review recovery requires completed extraction')
    relative = str(Path(submission_path).resolve().relative_to(root.resolve()))
    require(relative in entry['outputs'], 'review submission is not bound to completed extraction')
    for relative, expected in entry['outputs'].items():
        path = (root / relative).resolve()
        require(path.is_relative_to(root.resolve()) and path.is_file()
                and hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                f'review recovery: extraction artifact changed: {relative}')


def run_review(task, payload, output, *, model, workers, max_retries, retry_delay,
               timeout=300.0, batch_size=5, legacy_validator=None):
    from evidence_reviewer import reviewer
    from evidence_reviewer.models import CandidateReview
    output = Path(output)
    checkpoint = output / 'review_progress.json'
    result_path = output / 'candidate_evidence_review_structured.json'
    raw = copy.deepcopy(task.raw)
    raw.get('execution_policy', {}).pop('llm_timeout_seconds', None)
    implementation = {}
    for name in ('reviewer.py', 'models.py'):
        path = Path(reviewer.__file__).with_name(name)
        implementation[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    signature = digest({'task': raw, 'submission': payload, 'model': model,
                        'review_contract': implementation})
    predictions = payload['predictions']
    by_id = {str(row['prediction_id']): row for row in predictions}
    require(len(by_id) == len(predictions), 'duplicate review prediction ID')
    completed, attempts, history = {}, [], []

    def load_rows(rows):
        for row in rows:
            pid = row['prediction_id']
            require(pid in by_id and pid not in completed, 'unknown/duplicate saved review ID')
            require(row['article_id'] == by_id[pid]['article_id'], 'saved review article mismatch')
            value = CandidateReview.model_validate(row['review']).model_dump()
            require(value['prediction_id'] == pid, 'saved review identity mismatch')
            completed[pid] = {**row, 'review': value}

    if checkpoint.exists():
        saved = read(checkpoint)
        require(saved['fingerprint'] == signature, 'review inputs or contract changed; choose a new review output')
        load_rows(saved['candidate_reviews'])
        attempts = saved.get('batch_attempts', [])
        history = saved.get('failure_history', [])
    elif result_path.exists():
        require(legacy_validator is not None, 'legacy review requires verified extraction provenance')
        legacy_validator()
        saved = read(result_path)
        require(saved['case_id'] == task.case_id and saved['task_id'] == task.task_id
                and saved['input_prediction_count'] == len(predictions), 'legacy review belongs to different inputs')
        load_rows(saved['candidate_reviews'])
        attempts = saved.get('retry_policy', {}).get('batch_attempts', [])
        history = saved.get('failures', [])
        atomic_write(output / 'review_legacy_import.json', {
            'fingerprint': signature, 'imported_review_count': len(completed),
            'validation': 'frozen request, stage source locks and completed extraction artifact hashes',
        })

    pending = [row for row in predictions if row['prediction_id'] not in completed]
    batches = reviewer._article_batches(pending, batch_size)
    # Modify only the in-memory operational timeout, not any frozen input file.
    task = copy.deepcopy(task)
    task.execution_policy['llm_timeout_seconds'] = timeout
    result = reviewer.review_submission(task, {'predictions': []}, configured_model=model,
        batch_size=batch_size, max_workers=workers, max_retries=max_retries,
        retry_delay_seconds=retry_delay)
    failures = []

    def save():
        rows = [completed[pid] for pid in by_id if pid in completed]
        result.update(input_prediction_count=len(predictions), completed_review_count=len(rows),
            failed_review_count=sum(len(f['prediction_ids']) for f in failures),
            pending_review_count=len(predictions) - len(rows), candidate_reviews=rows,
            failures=list(failures), disposition_counts=dict(Counter(r['review']['disposition'] for r in rows)))
        result['retry_policy']['batch_attempts'] = list(attempts)
        result['batching'].update(batch_count=len(batches), max_workers=workers, timeout_seconds=timeout)
        atomic_write(checkpoint, {'schema_version': 'review_checkpoint/1.0', 'fingerprint': signature,
            'candidate_reviews': rows, 'batch_attempts': attempts, 'failure_history': history})
        atomic_write(result_path, result)

    save()  # Persist imported successes before any model request.
    print(f'[review] resumed={len(completed)} pending={len(pending)}; '
          f'batch_size={batch_size}; client timeout={timeout:g}s', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(reviewer.review_batch_with_retries, task, batch, model,
                              max_retries, retry_delay): batch for batch in batches}
        for future in as_completed(futures):
            batch = futures[future]
            ids = [row['prediction_id'] for row in batch]
            aid = batch[0]['article_id']
            try:
                batch_result, count = future.result()
                returned = [row.prediction_id for row in batch_result.reviews]
                require(len(returned) == len(ids) and set(returned) == set(ids), 'review batch ID mismatch')
                for row in batch_result.reviews:
                    completed[row.prediction_id] = {'prediction_id': row.prediction_id,
                        'article_id': aid, 'review': row.model_dump()}
                attempts.append({'article_id': aid, 'prediction_ids': ids, 'attempts': count, 'status': 'completed'})
                print(f'[review] {aid}: saved {len(ids)}; completed={len(completed)}/{len(predictions)}', flush=True)
            except Exception as exc:
                failure = {'article_id': aid, 'prediction_ids': ids, 'error_type': type(exc).__name__,
                           'error': str(exc)}
                failures.append(failure)
                history.append(failure)
                print(f'[review] {aid}: {type(exc).__name__}: {exc}', flush=True)
            save()
    return result
