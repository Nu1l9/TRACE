"""Release composition: pipeline extensions around the stage-A3 synthesis core."""
from __future__ import annotations
import argparse
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from extensions.common import ROOT, ANALYSIS, read, write, require, digest
from extensions.settings import load_settings
from metapipeline.contracts import normalize_request, write_analysis_input
from contract_adapter import load_analysis_contract
from analysis_core.plan_resolver import resolve_analysis_plan_structured
from analysis_engine.fixed_policy import POLICY, POLICY_HASH
from extensions.statistical_methods import (protocol_method_policy, method_policy_manifest,
    validate_method_policy, assert_method_policy_output_compatible)


def saved_provider_routes(output):
    """Use the provider profile frozen at run start; never re-read ambient env."""
    path=Path(output).resolve()
    for directory in (path,*path.parents):
        candidate=directory/'provider_routes.json'
        if candidate.is_file():
            return read(candidate)
    return None


def completed_analysis_summary(output):
    """Return an auditable completed analysis cache, if this exact output has one."""
    output=Path(output)
    required=('analysis_result.json','analysis_summary.json','analysis_report.md','graph.json')
    if not all((output/name).is_file() for name in required):
        return None
    result,summary=read(output/'analysis_result.json'),read(output/'analysis_summary.json')
    if result.get('status')!='completed' or summary.get('status')!='completed':
        return None
    return summary


def write_run_metadata(output, metadata, statistical_method_policy='fixed'):
    routes=saved_provider_routes(output)
    if routes is not None:
        metadata['provider_routes']=routes
    policy=method_policy_manifest(statistical_method_policy)
    metadata.update(
        statistical_method_policy=statistical_method_policy,
        statistical_policy=policy,
        statistical_policy_sha256=POLICY_HASH if statistical_method_policy=='fixed' else digest(policy),
        method_selector_llm_calls=metadata.get('method_selector_llm_calls',
            metadata.get('model_calls',{}).get('method_selector',0)),
        text_fallback_enabled=False,
    )
    write(Path(output)/'run_metadata.json',metadata)


def agent_components(output,offline=False,statistical_method_policy='fixed'):
    validate_method_policy(statistical_method_policy)
    require(not (offline and statistical_method_policy=='model_selected'),
            'model_selected requires a live method selector; offline mode cannot silently use fixed methods')
    from analysis_engine.retrieval import HashEmbeddingBackend,OpenAIEmbeddingBackend
    from analysis_engine.planner import DeterministicAnalysisPlanner,LLMAnalysisPlanner
    from analysis_engine.topic_reviewer import DeterministicTopicReviewer
    from analysis_engine.topic_reviewer_graph import GraphTopicReviewer
    from analysis_engine.reporting_bias_selector import DeterministicReportingBiasSelector,LLMReportingBiasSelector
    if offline:
        return dict(embeddings=HashEmbeddingBackend(),planner=DeterministicAnalysisPlanner(),reviewer=DeterministicTopicReviewer(),reporting_bias_selector=DeterministicReportingBiasSelector(),runtime_facet_annotator=None)
    from urllib.parse import urlparse
    base=os.getenv('TEXT_PROVIDER_BASE_URL') or os.getenv('ANALYSIS_LLM_BASE_URL','https://api.openai.com/v1')
    emb_base=os.getenv('EMBEDDING_PROVIDER_BASE_URL') or os.getenv('ANALYSIS_EMBEDDING_BASE_URL','https://api.openai.com/v1')
    key=os.getenv('TEXT_PROVIDER_API_KEY') or os.getenv('ANALYSIS_LLM_API_KEY') or (os.getenv('OPENAI_API_KEY') if urlparse(base).hostname=='api.openai.com' else None)
    embedding_key=os.getenv('EMBEDDING_PROVIDER_API_KEY') or os.getenv('ANALYSIS_EMBEDDING_API_KEY') or (os.getenv('OPENAI_API_KEY') if urlparse(emb_base).hostname=='api.openai.com' else None)
    require(bool(key and embedding_key),'analysis and embedding API credentials required')
    model=os.getenv('TEXT_PROVIDER_MODEL') or os.getenv('ANALYSIS_LLM_MODEL','gpt-5.6-luna')
    from extensions.transport import primary_attempts
    kwargs=dict(model=model,api_key=key,base_url=base,max_retries=int(os.getenv('ANALYSIS_API_MAX_RETRIES', str(primary_attempts() - 1))))
    from extensions.analysis_checkpoints import CheckpointedTopicReviewer
    reviewer=CheckpointedTopicReviewer(
        GraphTopicReviewer(always_review=True,hide_calculation_details=True,**kwargs),
        output/'checkpoints'/'topic_reviews',
    )
    runtime_facets=None
    components=dict(embeddings=OpenAIEmbeddingBackend(model=os.getenv('EMBEDDING_PROVIDER_MODEL') or os.getenv('ANALYSIS_EMBEDDING_MODEL','text-embedding-3-small'),api_key=embedding_key,base_url=os.getenv('EMBEDDING_PROVIDER_BASE_URL') or os.getenv('ANALYSIS_EMBEDDING_BASE_URL','https://api.openai.com/v1'),cache_path=output/'embedding_cache.json',max_retries=kwargs['max_retries']),planner=LLMAnalysisPlanner(reasoning_effort=os.getenv('ANALYSIS_LLM_REASONING_EFFORT','high'),**kwargs),reviewer=reviewer,runtime_facet_annotator=runtime_facets,reporting_bias_selector=LLMReportingBiasSelector(**kwargs))
    if statistical_method_policy=='model_selected':
        from analysis_engine.source_aware_method_selector import ToolAwareMethodSelector
        components['method_selector']=ToolAwareMethodSelector(
            artifact_dir=output/'method_selection',**kwargs)
    return components


def retain_excluded_candidates_for_expert_review(result, graph_records):
    """Keep plausible reviewer exclusions visible when a topic cannot be pooled.

    This is a handoff only: these records remain outside selected_evidence_ids and
    therefore cannot enter effect derivation or pooled statistics automatically.
    """
    records = {record.evidence_id: record for record in graph_records or []}
    promoted = 0
    for topic in result.analyses:
        definition = topic.analysis
        if topic.status not in {'insufficient_data', 'blocked'}:
            continue
        effects = topic.poolability.study_effects
        poolable_studies = len({effect.study_id for effect in effects})
        if poolable_studies >= definition.minimum_studies:
            continue

        unsupported = {
            item.get('evidence_id') for item in topic.source_resolutions
            if item.get('decision') == 'unsupported'
        }
        topic_promoted = []
        for assignment in topic.assignments:
            if assignment.role != 'excluded' or assignment.evidence_id in unsupported:
                continue
            record = records.get(assignment.evidence_id)
            if record is None or record.target_id != definition.target_id:
                continue
            if record.data_family not in definition.eligible_data_families:
                continue
            # Missing/variant outcome labels, exposure aliases, and comparison
            # wording are common reasons a reviewer can reject a real candidate.
            # Keep those semantic disagreements visible for a human when the
            # topic otherwise falls below its minimum study count. An explicit
            # conflicting outcome code or effect type remains out of scope.
            if (definition.eligible_outcome_codes and record.outcome_code
                    and record.outcome_code not in definition.eligible_outcome_codes):
                continue
            if (definition.accepted_effect_types and record.reported_effect_type
                    and record.reported_effect_type not in definition.accepted_effect_types):
                continue
            if definition.estimands and record.estimand and record.estimand not in definition.estimands:
                continue
            assignment.role = 'needs_review'
            assignment.reason = (
                'Reviewer excluded this target/data-family-aligned candidate, but the topic has fewer than '
                f'{definition.minimum_studies} poolable independent studies. Retained for expert '
                'adjudication; not included in automatic pooling.'
            )
            topic_promoted.append(assignment.evidence_id)
        if topic_promoted:
            promoted += len(topic_promoted)
            topic.warnings.append(
                f'{len(topic_promoted)} target/data-family-aligned reviewer-excluded candidate(s) retained for '
                'expert review; they remain excluded from automatic pooling.'
            )
    return promoted


def execute_analysis(input_dir,output,*,patch=None,assessments=None,ledger=None,offline=False):
    from extensions.graph_feedback import ExtendedAnalysisAgent
    from analysis_engine.retrieval_artifacts import graph_artifact,embedding_retrieval_artifact
    from analysis_engine.reporting import render_graph_report
    from run_analysis import _summary
    protocol,task,profile,payloads,studies=load_analysis_contract(input_dir)
    method_policy=protocol_method_policy(protocol)
    require(not (offline and method_policy=='model_selected'),
            'model_selected requires a live method selector; offline mode cannot silently use fixed methods')
    class ComposedAgent(ExtendedAnalysisAgent):
        def _build_graph(self,records,plan,runtime_facets):
            graph=super()._build_graph(records,plan,runtime_facets)
            if assessments is not None:
                from extensions.bias import build_bias_annotations,attach_bias_graph
                annotations=build_bias_annotations(assessments,ledger)
                write(output/'bias_annotations.json',annotations)
                # A separate view links assessments to real Evidence graph nodes.
                graph=attach_bias_graph(graph,annotations)
                write(output/'bias_graph_assessments.json',assessments)
            return graph
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    # Enforce the mode boundary even for update runs that bypass legacy hashes.
    policy_path=output/'statistical_method_policy.json'
    assert_method_policy_output_compatible(output,method_policy)
    from extensions.analysis_cache import bind_analysis_inputs, analysis_implementation_fingerprint
    bind_analysis_inputs(output, {
        'protocol': protocol, 'task': task, 'profile': profile,
        'extractor_payloads': payloads, 'studies': studies,
        'patch': patch, 'assessments': assessments, 'ledger': ledger,
        'offline': offline,
        'statistical_method_policy': method_policy,
        'implementation': analysis_implementation_fingerprint(ROOT),
    })
    write(policy_path,{'schema_version':'statistical_method_policy/1.0','mode':method_policy})
    write(output/'statistical_policy.json',method_policy_manifest(method_policy))
    cached=completed_analysis_summary(output)
    if cached is not None:
        print('[analysis] reusing completed analysis artefacts; model calls=0',flush=True)
        from extensions.statistics_audit import write_statistics_quality_audit
        write_statistics_quality_audit(output, read(output/'analysis_result.json'))
        original_metadata=read(output/'run_metadata.json') if (output/'run_metadata.json').is_file() else {}
        write_run_metadata(output,{'release':'meta_pipeline_release','frozen_analysis':'fixed_policy' if method_policy=='fixed' else 'source_aware_selection',
                                   'mode':'synthetic_offline' if offline else 'live','model_calls':{'reused_completed_analysis':0},
                                   'method_selector_llm_calls':original_metadata.get('method_selector_llm_calls',0),
                                   'protocol_fingerprint':digest(protocol),'primary_evidence_mutated':False,
                                   'graph_feedback_enabled':bool(patch),'reused_completed_analysis':True},method_policy)
        return cached
    components=agent_components(output,offline,method_policy)
    agent=ComposedAgent(statistical_method_policy=method_policy,graph_patch=patch,graph_feedback_dir=output/'brain_graph',audit_dir=output,pre_review_filter_mode='semantic_semantic',progress_callback=lambda x:print(x,flush=True),**components)
    result=agent.run(protocol=protocol,task=task,analysis_profile=profile,extractor_payloads=payloads,study_by_article=studies,reporting_bias_artifact_dir=output/'reporting_bias',forest_plot_artifact_dir=output/'forest_plots')
    expert_review_count=retain_excluded_candidates_for_expert_review(
        result, getattr(agent.last_graph, 'records', []),
    )
    if expert_review_count:
        print(f'[analysis] retained {expert_review_count} reviewer-excluded, PICO-aligned candidate(s) for expert review; pooling unchanged',flush=True)
    write(output/'analysis_result.json',result.model_dump(mode='json'))
    from extensions.statistics_audit import write_statistics_quality_audit
    write_statistics_quality_audit(output, read(output/'analysis_result.json'))
    directions={c['candidate_id']:(c.get('effect') or {}).get('direction','') for c in (ledger or read(Path(input_dir)/'analysis_evidence_candidates.json'))['candidates']}
    summary=_summary(result,directions)
    summary['statistical_method_policy']=method_policy
    write(output/'analysis_summary.json',summary)
    (output/'analysis_report.md').write_text(render_graph_report(result),encoding='utf-8')
    if agent.last_graph is not None:write(output/'graph.json',graph_artifact(agent.last_graph))
    if agent.last_candidate_sets is not None:write(output/'embedding_retrieval.json',embedding_retrieval_artifact(agent.last_candidate_sets))
    counts={name:getattr(component,'calls_made',None) for name,component in components.items()}
    counts['method_selector']=agent.method_selector.calls_made
    metadata={'release':'meta_pipeline_release','frozen_analysis':'fixed_policy' if method_policy=='fixed' else 'source_aware_selection','mode':'synthetic_offline' if offline else 'live','model_calls':counts,'embedding':components['embeddings'].diagnostics() if hasattr(components['embeddings'],'diagnostics') else {'remote_calls':0},'protocol_fingerprint':digest(protocol),'primary_evidence_mutated':False,'graph_feedback_enabled':bool(patch)}
    write_run_metadata(output,metadata,method_policy)
    return summary


def run_analysis(request,input_dir,output,settings,recommendations=None,*,offline=False,check=False,screening_rob=None):
    protocol,task,profile,payloads,studies=load_analysis_contract(input_dir)
    plan=resolve_analysis_plan_structured(protocol,task,profile)
    ledger=read(Path(input_dir)/'analysis_evidence_candidates.json')
    patch=read(settings['graph']['patch_file']) if settings['graph'].get('patch_file') else None
    if patch:
        from extensions.graph_feedback import validate_graph_patches,ledger_fingerprint
        # Public helper from the external adapter; validation precedes any model call.
        patch=validate_graph_patches(patch,protocol_fingerprint=digest(protocol),ledger_fingerprint=ledger_fingerprint(payloads,studies),analysis_ids={x.analysis_id for x in plan.analyses})
    assessments=None;bias=settings['bias']
    if bias['mode']=='import':
        from extensions.bias import validate_assessments
        assessments=validate_assessments(read(bias['assessments_file']),request,ledger)
    require(protocol==request['protocol'],'analysis input protocol differs from request')
    require(read(Path(input_dir)/'analysis_profile_structured.json')==request['analysis_profile'],'analysis input profile differs from request')
    documents={d['article_id']:d['study_id'] for d in request['documents']}
    require(all(documents.get(aid)==sid for aid,sid in studies.items()),'analysis article/study mapping differs from request')
    require(task.get('case_id')==request['case_id'] and ledger.get('case_id',request['case_id'])==request['case_id'],'analysis input belongs to another case')
    if bias['mode']=='draft':
        from extensions.screening_bias import map_to_evidence
        require(screening_rob is not None,'Analysis requires completed screening RoB; no downstream reassessment')
        assessments=map_to_evidence(request,ledger,screening_rob)
        print(f"[risk-of-bias] reused screening assessments for {len(assessments['assessments'])} evidence candidates; model calls=0",flush=True)
    rob_incomplete = bool(screening_rob and screening_rob.get("completion", {}).get("status") in ("partial", "failed"))
    if check:
        return {'status':'valid','extensions':'2.0','analysis_count':len(plan.analyses),'statistical_method_policy':protocol_method_policy(protocol),'remote_calls':0}
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    if settings['graph'].get('brain_loop'):
        from extensions.brain import run_graph_feedback_loop
        from extensions.graph_feedback import preview_graph_feedback
        from extensions.transport import json_callback
        accumulated=list(patch or [])
        def preview(patches,directory):
            return preview_graph_feedback(protocol=protocol,task=task,analysis_profile=profile,
                extractor_payloads=payloads,study_by_article=studies,graph_patch=patches,output_dir=directory)
        initial=preview(accumulated,output/'brain_planning/round_00')
        def apply_suggestion(suggestion):
            accumulated.append(suggestion)
            return preview(accumulated,output/'brain_planning'/f'round_{len(accumulated):02d}')
        loop=run_graph_feedback_loop(request,initial,None if offline else json_callback(),apply_suggestion,
            max_rounds=settings['graph'].get('max_rounds',2))
        write(output/'brain_planning/loop.json',loop)
        write(output/'brain_planning/graph_patches.json',accumulated)
        patch=accumulated or None
    if assessments is not None:
        write(output/'risk_of_bias_assessments.json',assessments)
        print(f"[risk-of-bias] screening assessment mappings={len(assessments.get('assessments', []))}", flush=True)
    prepared=[]
    sensitivity_setup_error=None
    try:
        policies=list(bias['sensitivity_policies'])
        if bias.get('auto_sensitivity') and assessments is not None:
            from extensions.bias import automatic_sensitivity_policies
            policies.extend(automatic_sensitivity_policies(request, assessments))
        if rob_incomplete:
            # A partial RoB batch is not a reason to discard its completed,
            # applicable assessments. `create_sensitivity_inputs` explicitly
            # keeps unmatched (therefore unknown) evidence under
            # unknown_policy=keep, so no unassessed article is relabelled as
            # low risk or excluded as high risk.
            print("[risk-of-bias] incomplete: high-risk exclusion uses completed supported assessments; unassessed evidence is retained",flush=True)
        if policies:
            from extensions.bias import create_sensitivity_inputs
            require(assessments is not None,'bias sensitivity requires assessments')
            ids=set()
            for policy in policies:
                runs=create_sensitivity_inputs(request,ledger,assessments,policy)
                for run in runs['runs']:
                    sid=run['sensitivity_id'];require(sid not in ids,'duplicate sensitivity_id');ids.add(sid);prepared.append(run)
    except Exception as exc:
        prepared=[]
        sensitivity_setup_error={'status':'failed','error_type':type(exc).__name__,'message':str(exc)}
        write(output/'bias_sensitivity'/'preparation_failure.json',sensitivity_setup_error)
        print('[risk-of-bias] sensitivity preparation failed; primary analysis continues',flush=True)
    primary=execute_analysis(input_dir,output,patch=patch,assessments=assessments,ledger=ledger,offline=offline)
    def _execute_sensitivity(run):
        dest=output/'bias_sensitivity'/run['sensitivity_id'];dest.mkdir(parents=True,exist_ok=True)
        write(dest/'audit.json',{'source_analysis_id':run['source_analysis_id'],'policy':run['policy'],'records':run['audit']})
        if not any(record['excluded'] for record in run['audit']):
            summary={'status':'not_applicable_no_high_risk_exclusions','analyses':[]}
            write(dest/'analysis_summary.json',summary)
        elif not run['ledger']['candidates']:
            summary={'status':'no_retained_evidence','analyses':[]};write(dest/'analysis_summary.json',summary)
        else:
            write_analysis_input(run['request'],run['ledger'],dest/'input')
            summary=execute_analysis(dest/'input',dest/'analysis',ledger=run['ledger'],offline=offline)
        return {'sensitivity_id':run['sensitivity_id'],'status':summary['status'],'summary':summary,'audit_path':str((dest/'audit.json').relative_to(output))}
    def execute_sensitivity(run):
        try:
            return _execute_sensitivity(run)
        except Exception as exc:
            failure={'sensitivity_id':run['sensitivity_id'],'status':'failed','error_type':type(exc).__name__,'message':str(exc)}
            write(output/'bias_sensitivity'/run['sensitivity_id']/'failure.json',failure)
            print(f"[risk-of-bias] sensitivity failed: {run['sensitivity_id']}; primary result preserved",flush=True)
            return failure
    sensitivity_workers=min(max(1,request['options'].get('max_workers',4)),max(1,len(prepared)))
    if prepared:
        print(f'[analysis] RoB sensitivities parallel workers={sensitivity_workers}',flush=True)
        with ThreadPoolExecutor(max_workers=sensitivity_workers,thread_name_prefix='sensitivity') as executor:
            sensitivities=list(executor.map(execute_sensitivity,prepared))
    else:
        sensitivities=[]
    summary={'schema_version':'extension_summary/2.0','mode':'synthetic_offline' if offline else 'live','graph_patch_applied':bool(patch),'bias_mode':bias['mode'],'bias_assessments_path':'risk_of_bias_assessments.json' if assessments is not None else None,'bias_sensitivities':sensitivities,'primary_status':primary['status'],'original_primary_input_unchanged':True}
    summary['statistical_method_policy']=protocol_method_policy(protocol)
    summary['rob_completion'] = (screening_rob or {}).get('completion', {})
    summary['rob_failures'] = (screening_rob or {}).get('failures', [])
    summary['rob_sensitivity_status'] = ('processed_with_unassessed_retained' if rob_incomplete else 'processed')
    sensitivity_failures=[s for s in sensitivities if s.get('status')=='failed']
    if sensitivity_setup_error: sensitivity_failures.append(sensitivity_setup_error)
    summary['rob_sensitivity_failures']=sensitivity_failures
    if sensitivity_failures:
        summary['rob_sensitivity_status']='failed'
        report=output/'analysis_report.md'
        if report.exists():
            with report.open('a',encoding='utf-8') as handle:
                handle.write('\n## RoB sensitivity analysis warning\n\nPrimary analysis completed; RoB exclusion analysis failed. See extension_summary.json and bias_sensitivity failure artifacts; retry this analysis separately.\n')
    if rob_incomplete:
        from extensions.rob_resilience import warning_report
        warning = warning_report(screening_rob)
        write(output/'rob_warnings.json', {'completion':summary['rob_completion'], 'failures':summary['rob_failures']})
        report = output/'analysis_report.md'
        if report.exists():
            with report.open('a', encoding='utf-8') as handle: handle.write('\n'+warning)
    write(output/'extension_summary.json',summary)
    files=[]
    for path in sorted(output.rglob('*')):
        if path.is_file() and path.name != 'extension_files.json':
            files.append({'path':str(path.relative_to(output)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    write(output/'extension_files.json',{'files':files})
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--extensions',type=Path,required=True);p.add_argument('--recommendations',type=Path);p.add_argument('--screening-rob',type=Path);p.add_argument('--check',action='store_true');p.add_argument('--offline',action='store_true')
    a=p.parse_args();r=normalize_request(read(a.request),a.request.resolve().parent)
    if not a.offline:
        has_text=bool(os.getenv('TEXT_PROVIDER_API_KEY') or os.getenv('OPENAI_API_KEY'))
        has_emb=bool(os.getenv('EMBEDDING_PROVIDER_API_KEY') or os.getenv('OPENAI_API_KEY'))
        require(has_text and has_emb,'OPENAI_API_KEY (or TEXT_PROVIDER_API_KEY and EMBEDDING_PROVIDER_API_KEY) required for live analysis')
    result=run_analysis(r,a.input_dir,a.output_dir,load_settings(a.extensions),read(a.recommendations) if a.recommendations else None,offline=a.offline,check=a.check,screening_rob=read(a.screening_rob) if a.screening_rob else None)
    print(__import__('json').dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
