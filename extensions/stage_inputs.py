"""Input bindings for reusable article-level stage results."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json


def stage_signature(request, stage):
    from metapipeline.contracts import screening_protocol
    documents = []
    for item in request['documents']:
        paths = item.get('paths', {})
        files = {}
        for kind in ('markdown', 'content_list', 'pdf'):
            if paths.get(kind):
                path = Path(paths[kind])
                files[kind] = hashlib.sha256(path.read_bytes()).hexdigest()
        documents.append({key: item.get(key) for key in ('article_id', 'document_id', 'study_id', 'title', 'abstract')}
                         | {'files': files})
    semantic = screening_protocol(request) if stage == 'screening' else {
        'protocol': request['protocol'], 'analysis_profile': request['analysis_profile'],
        'extraction_runtime': request.get('options', {}).get('extraction_runtime'),
    }
    data = {'version': 'resilient/' + stage, 'semantics': semantic, 'documents': documents}
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def bind_stage_inputs(request, output, stage, *, signature=None):
    output = Path(output)
    signature = signature or stage_signature(request, stage)
    lock = output / (stage + '_inputs.lock.json')
    if lock.exists():
        if json.loads(lock.read_text()).get('fingerprint') != signature:
            raise ValueError(stage + ' inputs changed; choose a new run output directory')
    else:
        marker = 'screening_results.json' if stage == 'screening' else 'extraction_progress.json'
        if (output / marker).exists():
            raise ValueError(stage + ' checkpoint has no input binding; choose a new run output directory')
        output.mkdir(parents=True, exist_ok=True)
        temporary = lock.with_suffix('.json.tmp')
        temporary.write_text(json.dumps({'schema_version': 'stage_inputs/8.0', 'fingerprint': signature}, indent=2) + '\n')
        temporary.replace(lock)
    return signature


def validate_screening_reuse(request, source):
    """Reusing another run requires matching source and screening semantics."""
    source = Path(source)
    lock = source / '01_screening' / 'screening_inputs.lock.json'
    if not lock.is_file():
        raise ValueError('Screening reuse requires source/protocol provenance; rerun screening for this migration.')
    if json.loads(lock.read_text()).get('fingerprint') != stage_signature(request, 'screening'):
        raise ValueError('Screening reuse source documents or eligibility conditions differ; rerun screening.')
    previous_request = json.loads((source / 'request.normalized.json').read_text())
    if previous_request.get('protocol') != request.get('protocol'):
        raise ValueError('RoB assessment targets or protocol conditions differ; rerun screening/RoB.')
    manifest = json.loads((source / 'pipeline_manifest.json').read_text())
    for stage in ('screening', 'screening_rob'):
        saved = manifest.get('stages', {}).get(stage, {})
        if saved.get('status') != 'completed' or not saved.get('outputs'):
            raise ValueError('Screening reuse requires completed screening and RoB artifacts')
        for relative, sha in saved['outputs'].items():
            file = (source / relative).resolve()
            if not file.is_relative_to(source.resolve()) or not file.is_file():
                raise ValueError('Invalid source screening artifact path')
            if hashlib.sha256(file.read_bytes()).hexdigest() != sha:
                raise ValueError('Source screening artifact changed: ' + relative)
