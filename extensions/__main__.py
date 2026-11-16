from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from .common import ROOT, BASELINE, read, write, require, ContractError
from .settings import load_settings
from metapipeline.contracts import normalize_request as normalize_pipeline_request
from .input_contract import normalize_request as compile_input_contract


def normalize_request(request, base):
    return normalize_pipeline_request(compile_input_contract(request), base)


def _request(path):
    return normalize_request(read(path),path.resolve().parent)


def _env(path=None):
    environment=dict(os.environ)
    if path:
        require(path.is_file(),'missing env file')
        from .provider import file_values
        environment.update(file_values(path))
    environment['PYTHONDONTWRITEBYTECODE']='1'
    environment['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
    from .provider import DEFAULT_PROVIDER_CONFIG
    configured = bool(environment.get('TEXT_PROVIDER_API_KEY') or (environment.get('OPENAI_API_KEY') and DEFAULT_PROVIDER_CONFIG.is_file()))
    if configured:
        from .provider import environment as routed_environment
        environment,_=routed_environment(path)
        environment['PYTHONDONTWRITEBYTECODE']='1';environment['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
    return environment


def _callback(args):
    if getattr(args,'completion',None):
        payload=read(args.completion)
        return lambda _:payload
    if args.live:
        from .transport import json_callback
        return json_callback(_env(args.env_file))
    return None


def _preview(args,patch=None,output=None):
    from contract_adapter import load_analysis_contract
    from .graph_feedback import preview_graph_feedback
    protocol,task,profile,payloads,studies=load_analysis_contract(args.input_dir)
    facets=None
    if args.runtime_facets:
        from analysis_engine.runtime_facets import RuntimeFacetResult
        facets=RuntimeFacetResult.model_validate(read(args.runtime_facets)).by_evidence_id()
    return preview_graph_feedback(protocol=protocol,task=task,analysis_profile=profile,
        extractor_payloads=payloads,study_by_article=studies,graph_patch=patch,
        runtime_facets=facets,output_dir=output)


def main(argv=None):
    parser=argparse.ArgumentParser(description='Meta Pipeline evidence synthesis')
    sub=parser.add_subparsers(dest='command',required=True)
    from literature_retrieval.pipeline import register_commands
    register_commands(sub)
    p=sub.add_parser('update', help='Incrementally update an existing run into a new version')
    p.add_argument('--base-run', type=Path, required=True)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--request', type=Path, help='Complete updated candidate registry and protocol')
    source.add_argument('--delta', type=Path, help='New/revised articles and optional protocol changes')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--extensions', type=Path);p.add_argument('--env-file', type=Path)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--plan-only', action='store_true', help='Inspect dependencies without API calls or reading .env')
    p=sub.add_parser('provider-check');p.add_argument('--env-file',type=Path,required=True);p.add_argument('--request',type=Path)
    p=sub.add_parser('retry-analysis', help='Rerun only analysis from a completed run\'s frozen reviewed evidence')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--extensions', type=Path);p.add_argument('--env-file', type=Path)
    p.add_argument('--workers', type=int, default=4, help='parallel workers for analysis-side tasks (default: 4)')
    p.add_argument('--rebuild-handoff', action='store_true',
                   help='Rebuild evidence handoff and analysis from completed screening, extraction, and review artifacts')
    p=sub.add_parser('migrate-typed', help='Create separate Typed migration requests for frozen Baseline benchmark cases')
    migration_source=p.add_mutually_exclusive_group(required=True)
    migration_source.add_argument('--request',type=Path)
    migration_source.add_argument('--all-benchmark',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    for name in ('check','run'):
        p=sub.add_parser(name);p.add_argument('--request',type=Path,required=True)
        p.add_argument('--extensions',type=Path);p.add_argument('--env-file',type=Path)
        if name=='run':
            p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true')
            p.add_argument('--reuse-screening-from',type=Path,
                           help='Reuse completed screening and screening-RoB artifacts from another run; rerun extraction downstream')
            p.add_argument('--reuse-legacy-from',type=Path,
                           help='Import source-matched Graph screening/RoB and completed final article submissions into a fresh pipeline run')
            p.add_argument('--retry-deferred',action='store_true',
                           help='On a resume, replay articles previously deferred after a 524 timeout')
            p.add_argument('--retry-analysis',action='store_true',
                           help='On a resume, archive only the prior terminal analysis and rerun analysis from completed review evidence')
            p.add_argument('--workers',type=int,default=4,help='parallel workers for document, article, review, and RoB tasks (default: 4)')
    p=sub.add_parser('analysis');p.add_argument('--screening-rob',type=Path);p.add_argument('--request',type=Path,required=True)
    p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--extensions',type=Path);p.add_argument('--recommendations',type=Path)
    p.add_argument('--offline',action='store_true');p.add_argument('--check',action='store_true');p.add_argument('--resume',action='store_true',help='Reuse input-bound analysis checkpoints in an existing analysis output directory');p.add_argument('--env-file',type=Path)
    p=sub.add_parser('brain-fill');p.add_argument('--topic',required=True);p.add_argument('--known',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--request-output',type=Path)
    group=p.add_mutually_exclusive_group();group.add_argument('--live',action='store_true');group.add_argument('--completion',type=Path)
    p.add_argument('--env-file',type=Path)
    for name in ('graph-preview','brain-graph-loop'):
        p=sub.add_parser(name);p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
        p.add_argument('--runtime-facets',type=Path)
        if name=='graph-preview':p.add_argument('--patch',type=Path)
        else:
            p.add_argument('--request',type=Path,required=True);p.add_argument('--live',action='store_true')
            p.add_argument('--env-file',type=Path);p.add_argument('--max-rounds',type=int,default=2)
    a=parser.parse_args(argv)
    try:
        if a.command in ('review', 'prepare-corpus', 'build-bm25', 'build-dense'):
            from literature_retrieval.pipeline import execute
            return execute(a, main)
        elif a.command=='update':
            from .update import run_update
            result=run_update(a.base_run,a.output,request_file=a.request,delta_file=a.delta,
                              settings_file=a.extensions,env_file=a.env_file,workers=a.workers,
                              resume=a.resume,plan_only=a.plan_only)
        elif a.command=='provider-check':
            from .provider import profile,audit
            result=audit(profile(a.env_file,_request(a.request) if a.request else None))
        elif a.command=='retry-analysis':
            from .runner import run
            require(1 <= a.workers <= 8, '--workers must be between 1 and 8')
            frozen_request = a.output.resolve()/'request.normalized.json'
            require(frozen_request.is_file(), '--retry-analysis output has no frozen request.normalized.json')
            r=read(frozen_request)
            result=run(r,a.output,load_settings(a.extensions),a.env_file,True,
                       {'META_PIPELINE_WORKERS':str(a.workers)},workers=a.workers,
                       retry_analysis=True,rebuild_handoff=a.rebuild_handoff)
        elif a.command=='migrate-typed':
            from .typed_migration import migrate_all_benchmark, write_migration
            result=migrate_all_benchmark(a.output) if a.all_benchmark else write_migration(a.request,a.output)
        elif a.command in ('check','run'):
            from metapipeline.pipeline import runtime_environment,validate_components,preflight_runtime
            r=_request(a.request);settings=load_settings(a.extensions)
            from .document_preparation import prepare_request
            preparation_env = dict(os.environ)
            if a.env_file:
                from .provider import file_values
                preparation_env.update(file_values(a.env_file))
            env=runtime_environment(r,a.env_file);env.update({k:v for k,v in _env().items() if k in ('PYTHONDONTWRITEBYTECODE','MPLCONFIGDIR')})
            if a.command=='check':
                prepared, preparation = prepare_request(r, preparation_env, allow_conversion=False)
                pending = [x for x in preparation['documents'] if x['status'] == 'needs_mineru_conversion']
                if pending:
                    result={'status':'needs_document_preparation','case_id':r['case_id'],'document_count':len(r['documents']),'remote_calls':0,
                            'document_preparation': preparation}
                else:
                    # Provider selection is automatic and project-level.  The
                    # raw launch request deliberately has no provider/model.
                    from .provider import runtime as provider_runtime
                    prepared, env, _ = provider_runtime(prepared, a.env_file)
                    validate_components(prepared,env)
                    result={'status':'request_valid','case_id':r['case_id'],'document_count':len(r['documents']),'extensions_enabled':True,'remote_calls':0,
                            'document_preparation': preparation,
                            'deferred_validation':'Graph fingerprints and bias source links are checked when the evidence ledger exists.' if a.extensions else None}
            else:
                # source always composes the frozen stages with the external
                # Graph/RoB coordinator.  --extensions only overrides the
                # default mandatory policy; it is not a feature toggle.
                from .runner import run
                require(1 <= a.workers <= 8, '--workers must be between 1 and 8')
                require(not a.retry_deferred or a.resume, '--retry-deferred requires --resume')
                require(not a.retry_analysis or a.resume, '--retry-analysis requires --resume')
                preparation_env['META_PIPELINE_WORKERS']=str(a.workers)
                result=run(r,a.output,settings,a.env_file,a.resume,preparation_env,
                           workers=a.workers,retry_deferred=a.retry_deferred,
                           retry_analysis=a.retry_analysis,
                           reuse_screening_from=a.reuse_screening_from,
                           reuse_legacy_from=a.reuse_legacy_from)
        elif a.command=='analysis':
            from .analysis_worker import run_analysis
            r=_request(a.request)
            if a.offline:
                from metapipeline.pipeline import runtime_environment
                environment=runtime_environment(r,a.env_file)
                environment['PYTHONDONTWRITEBYTECODE']='1';environment['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
            else:
                from .provider import runtime
                r,environment,_=runtime(r,a.env_file)
                environment['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
            os.environ.update(environment)
            if not a.check:
                nonempty=a.output.exists() and any(a.output.iterdir())
                require(a.resume or not nonempty,'analysis output must be empty; choose a new output directory or use --resume')
                require(not a.resume or nonempty,'--resume requires an existing non-empty analysis output directory')
            result=run_analysis(r,a.input_dir,a.output,load_settings(a.extensions),read(a.recommendations) if a.recommendations else None,offline=a.offline,check=a.check,screening_rob=read(a.screening_rob) if a.screening_rob else None)
        elif a.command=='brain-fill':
            from .brain import fill_topic
            result=fill_topic(a.topic,read(a.known) if a.known else None,_callback(a),base=a.known.resolve().parent if a.known else Path.cwd())
            write(a.output,result)
            if a.request_output:
                require(result['status']=='ready','Brain proposal has missing fields or unresolved uncertainties; request not exported')
                # Export relative document paths against the known-fields file,
                # before moving the request to a different output directory.
                write(a.request_output,normalize_request(result['pipeline_request'],
                    a.known.resolve().parent if a.known else Path.cwd()))
        elif a.command=='graph-preview':
            result=_preview(a,read(a.patch) if a.patch else None,a.output)
        else:
            from .brain import run_graph_feedback_loop
            from contract_adapter import load_analysis_contract
            r=_request(a.request);protocol,_,_,_,_=load_analysis_contract(a.input_dir)
            require(protocol==r['protocol'],'Graph input protocol differs from request')
            require(read(a.input_dir/'analysis_profile_structured.json')==r['analysis_profile'],'Graph input profile differs from request')
            initial=_preview(a,output=a.output/'round_00')
            patches=[]
            def execute(patch):
                patches.append(patch)
                return _preview(a,patches,a.output/f'round_{len(patches):02d}')
            result=run_graph_feedback_loop(r,initial,_callback(a),execute,max_rounds=a.max_rounds)
            write(a.output/'brain_graph_loop.json',result);write(a.output/'graph_patches.json',result['patches'])
        print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
        return 2 if result.get('status','').startswith(('needs_','no_','draft','paused')) else 0
    except (ContractError,FileNotFoundError,ValueError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr);return 1


if __name__=='__main__':
    sys.dont_write_bytecode=True
    raise SystemExit(main())
