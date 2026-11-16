"""Nonblocking RoB stage and isolated retries; never invent risk judgments."""
from __future__ import annotations
import argparse
import copy
import shutil
from datetime import datetime, timezone
from pathlib import Path
from .common import read, write, require
from .bias import SCHEMA, validate_assessments, unassessed_record


def warning_report(payload):
    completion = payload['completion']
    lines = ['## Risk-of-bias assessment status', '',
             f"Status: {completion['status']}; assessed {completion['completed']}/{completion['total']}.",
             'Primary analysis continues. Unassessed studies are unknown, never low risk.',
             'High-risk exclusion uses only completed supported assessments; unassessed studies are retained.',
             'Retry only this stage: `./.venv/bin/python run_pipeline.py rob-retry --run-dir RUN_DIRECTORY --env-file .env`.',
             'Retry creates a separate result directory and preserves the original run. Updated sensitivity analysis requires a subsequent analysis run.', '',
             '| Article / assessment | Error |', '| --- | --- |']
    for failure in payload.get('failures', []):
        label = failure.get('article_id', failure.get('candidate_id', 'stage'))
        message = (failure.get('error_type', '') + ': ' + failure.get('message', '')).replace('|', '/').replace('\n', ' ')
        lines.append(f'| {label} | {message} |')
    for route in payload.get('uncovered_routes', []):
        label = route.get('article_id', 'article')
        message = ('manual methodological assessment required: ' + route.get('reason', 'no registered RoB route')).replace('|', '/')
        lines.append(f'| {label} | {message} |')
    return '\n'.join(lines)+'\n'


def assess_nonblocking(request, screening, recommendations, output, callback):
    from .screening_bias import assess_screened, assessment_units
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    included = [r['candidate_id'] for r in screening['decisions'] if r['final_status']=='Include']
    units = assessment_units(request, included)
    try:
        result = assess_screened(request, screening, recommendations, output, callback, allow_partial=True)
    except Exception as exc:
        # Stage-wide setup/contract failures cannot establish checkpoint validity.
        # Keep original checkpoint files untouched for inspection/retry; publish
        # unknown records rather than reuse potentially incompatible assessments.
        result = validate_assessments({'schema_version':SCHEMA,'case_id':request['case_id'],
                    'assessments':[unassessed_record(request,c) for c in units['candidates']]},request,units)
        result['failures'] = [{'error_type':type(exc).__name__,'message':str(exc),'scope':'stage'}]
        print(f'[screening-rob] warning: {type(exc).__name__}: {exc}; continuing primary pipeline',flush=True)
    result.update(assessment_scope='article_target',assessment_units=units,source_stage='screening_rob')
    lookup = {c['candidate_id']:c for c in units['candidates']}
    for failure in result.get('failures',[]):
        c = lookup.get(failure.get('candidate_id'),{})
        failure.update({k:c[k] for k in ('article_id','study_id','target_id') if k in c})
    for route in result.get('uncovered_routes', []):
        article_id=route.get('article_id')
        candidate=next((item for item in units['candidates'] if item['article_id']==article_id), {})
        route.update({k:candidate[k] for k in ('candidate_id','study_id','target_id') if k in candidate})
    completed = sum(a['status'] in ('draft','confirmed') for a in result['assessments'])
    total = len(units['candidates'])
    operational_failures=bool(result.get('failures'))
    uncovered_count=len(result.get('uncovered_routes', []))
    status = ('complete' if completed==total and not operational_failures and not uncovered_count
              else 'partial' if completed or uncovered_count else 'failed')
    result['completion'] = {'status':status,'completed':completed,'total':total,'failed':total-completed,
                            'uncovered':uncovered_count,'provider_failures':len(result.get('failures',[])),
                            'recorded_at':datetime.now(timezone.utc).isoformat(),
                            'primary_analysis_blocked':False,
                            'sensitivity_uses_completed_assessments_only':True}
    write(output/'risk_of_bias_assessments.json',result)
    write(output/'risk_of_bias_failures.json',{'completion':result['completion'],'failures':result.get('failures',[]),
                                                'uncovered_routes':result.get('uncovered_routes',[])})
    write(output/'risk_of_bias_complete.json',{'case_id':request['case_id'],'article_count':len(included),
          'assessment_count':total,**result['completion']})
    if status!='complete':
        (output/'risk_of_bias_retry.md').write_text(warning_report(result),encoding='utf-8')
        print(f'[screening-rob] WARNING assessed={completed}/{total}; provider failures and uncovered routes recorded; primary pipeline continues',flush=True)
    return result


def archive_failed_responses(directory):
    """Explicit retry refreshes failed responses while retaining their audit trail."""
    directory=Path(directory)
    failures=read(directory/'risk_of_bias_failures.json').get('failures',[]) if (directory/'risk_of_bias_failures.json').exists() else []
    ids={f.get('candidate_id') for f in failures if f.get('candidate_id')}
    raw=directory/'risk_of_bias_raw_responses'
    if raw.is_dir():
        ids.update(p.name.removesuffix('.validation_error.json') for p in raw.glob('*.validation_error.json'))
    checkpoint=directory/'risk_of_bias.progress.json'
    completed={a.get('scope',{}).get('candidate_id') for a in read(checkpoint).get('assessments',[]) if a.get('status') in ('draft','confirmed')} if checkpoint.exists() else set()
    archive=directory/'failed_response_history'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    for candidate_id in ids-completed:
        for path in raw.glob(candidate_id+'.*.json'):
            archive.mkdir(parents=True,exist_ok=True)
            shutil.move(str(path),archive/path.name)


def main(argv=None):
    p=argparse.ArgumentParser(description='Retry RoB in a separate directory without rerunning screening, extraction or primary analysis.')
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--env-file',type=Path)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args(argv)
    source=args.run_dir.resolve()
    output=(args.output or source/'rob_retries'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    require(output!=source,'RoB retry must not overwrite the original run')
    source_request=read(source/'request.normalized.json')
    if args.resume:
        require(args.output is not None and output.is_dir(),'--resume requires existing --output')
        require(read(output/'retry_source.json')['run_dir']==str(source),'retry baseline changed')
        require(read(output/'request.normalized.json')==source_request,'retry request changed')
    else:
        require(not output.exists(),'retry output already exists; use --resume or a new output')
        output.mkdir(parents=True)
        write(output/'request.normalized.json',source_request)
        write(output/'retry_source.json',{'run_dir':str(source)})
        shutil.copytree(source/'01_screening',output/'01_screening')
    from .provider import runtime
    from .transport import json_callback
    _,environment,_=runtime(source_request,args.env_file)
    dest=output/'01_screening'
    archive_failed_responses(dest)
    result=assess_nonblocking(source_request,read(dest/'effective_screening_results.json'),
                             read(dest/'risk_of_bias_recommendations.json'),dest,lambda payload: json_callback(environment,purpose='bias')(payload))
    print(f"[rob-retry] {result['completion']['status']}: {output}",flush=True)
    return 0
