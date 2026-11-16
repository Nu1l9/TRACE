"""Subprocess entrypoint: never mix baseline/typed or local historical analysis imports."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from trace_runtime.contracts import read, write, screening_protocol
EXTRACT=ROOT/'vendor/evidence_extraction/system'
MINERS=ROOT/'vendor/miners'


def screen(r,out):
    sys.path.insert(0,str(MINERS))
    from miners.paper_node_adapter import build_llm_callback
    from miners.case_screening_rob import FullTextProtocolScreener
    from miners.case_rob import pdf_sections
    from miners.rob_routing import RiskOfBiasRecommendationAgent
    o=r['options']
    callback=build_llm_callback(o['model'], 'PIPELINE_LLM_API_KEY',os.environ.get('PIPELINE_LLM_BASE_URL'))
    screener=FullTextProtocolScreener(callback)
    router=RiskOfBiasRecommendationAgent(callback) if o['route_rob'] else None
    out.mkdir(parents=True,exist_ok=True)
    decisions_by_article={}; routes_by_article={}
    previous=out/'screening_results.json'
    if previous.is_file():
        payload=read(previous)
        if payload.get('case_id')!=r['case_id']:
            raise ValueError('screening checkpoint belongs to a different case')
        for item in payload.get('decisions',[]):
            aid=item.get('candidate_id')
            if not isinstance(aid,str) or aid in decisions_by_article:
                raise ValueError('invalid or duplicate screening checkpoint decision')
            decisions_by_article[aid]=item
    previous=out/'risk_of_bias_recommendations.json'
    if previous.is_file():
        for item in read(previous).get('recommendations',[]):
            aid=item.get('article_id')
            if not isinstance(aid,str) or aid in routes_by_article:
                raise ValueError('invalid or duplicate RoB-routing checkpoint')
            routes_by_article[aid]=item
    protocol=screening_protocol(r)
    for d in r['documents']:
        aid=d['article_id']; print(f'[screening] {aid}',flush=True)
        paths=d['paths']
        if paths.get('pdf'):
            sections=pdf_sections(Path(paths['pdf']),max_pages=1000,max_chars_per_page=20000)
        else:
            sections={'page_1':Path(paths['markdown']).read_text(encoding='utf-8')}
        decision=decisions_by_article.get(aid)
        if decision is None:
            decision=screener.screen(aid,d.get('title',aid),sections,protocol,
                abstract_source_text=d.get('abstract') or (Path(paths['markdown']).read_text(encoding='utf-8') if paths.get('markdown') else None))
            decisions_by_article[aid]=decision
        if router and decision['final_status']=='Include' and aid not in routes_by_article:
            recommendation=router.recommend({'study_id':d['study_id'],'title':decision.get('title',aid),'abstract':d.get('abstract','')},sections,review_question=r['topic'])
            routes_by_article[aid]={'article_id':aid,**recommendation}
        decisions=[decisions_by_article[x['article_id']] for x in r['documents'] if x['article_id'] in decisions_by_article]
        routes=[routes_by_article[x['article_id']] for x in r['documents'] if x['article_id'] in routes_by_article]
        write(out/'screening_results.json',{'case_id':r['case_id'],'decisions':decisions})
        write(out/'risk_of_bias_recommendations.json',{'recommendations':routes})
    write(out/'screening_complete.json',{'case_id':r['case_id'],'article_count':len(decisions_by_article)})


def load_extractor(r):
    if r['options']['extraction_runtime']=='baseline':
        sys.path.insert(0,str(EXTRACT/'baseline'))
    else:
        sys.path.insert(0,str(EXTRACT))
        sys.path.insert(0,str(EXTRACT/'runtime/typed'))
    from constrained_meta_agent.loader import load_case_config
    from locator_extractor_agent.controller import run_article
    return load_case_config,run_article


def extract(r,config,out,check=False):
    load_case_config,run_article=load_extractor(r)
    task=load_case_config(config)
    out.mkdir(parents=True,exist_ok=True)
    if check:
        write(out/'extractor_check.json',{'status':'valid','article_count':len(task.included_article_ids),'remote_calls':0,'runtime':r['options']['extraction_runtime']})
        return
    predictions=[]; articles=[]
    progress_path=out/'extraction_progress.json'
    prior_by_article={}
    if progress_path.is_file():
        progress=read(progress_path)
        if progress.get('case_id')!=r['case_id']:
            raise ValueError('extraction checkpoint belongs to a different case')
        for item in progress.get('article_runs',[]):
            aid=item.get('article_id')
            run_dir=Path(item.get('run_dir','')).resolve()
            if not isinstance(aid,str) or aid in prior_by_article or not run_dir.is_relative_to(out.resolve()):
                raise ValueError('invalid extraction checkpoint')
            submission_path=run_dir/'extraction_submission.json'
            if item.get('status')!='completed' or not submission_path.is_file():
                raise ValueError('incomplete extraction checkpoint')
            payload=read(submission_path)
            if any(row.get('article_id')!=aid for row in payload.get('predictions',[])):
                raise ValueError('extraction checkpoint article mismatch')
            prior_by_article[aid]=(item,payload.get('predictions',[]))
    provider=r['options']['provider']
    for aid in task.included_article_ids:
        if aid in prior_by_article:
            item,stored_predictions=prior_by_article[aid]
            articles.append(item);predictions.extend(stored_predictions)
            print(f'[extraction] resume {aid}',flush=True)
            continue
        for attempt in range(r['options']['max_retries']+1):
            article_dir=out/'articles'/aid/f'attempt_{attempt+1:02d}'
            # Distinct attempt paths keep an interrupted previous attempt auditable.
            if article_dir.exists():
                article_dir=article_dir.with_name(article_dir.name+'_'+str(time.time_ns()))
            try:
                kwargs={'provider':provider} if r['options']['extraction_runtime']=='typed' else {}
                result=run_article(task,aid,article_dir,False,**kwargs)
                for name,key in [('run_manifest','manifest'),('raw_system_output','raw_system_output'),('extraction_submission','extraction_submission'),('issues','issues'),('controller_state','controller_state')]:
                    write(article_dir/(name+'.json'),result[key])
                predictions.extend(result['extraction_submission']['predictions'])
                articles.append({'article_id':aid,'status':'completed','run_dir':str(article_dir),'issues':result['issues']})
                write(progress_path,{'case_id':r['case_id'],'article_runs':articles,'prediction_count':len(predictions)})
                break
            except Exception as exc:
                write(article_dir/'failure.json',{'error_type':type(exc).__name__,'status':'failed'})
                if attempt>=r['options']['max_retries']: raise
                time.sleep(min(5,attempt+1))
    write(out/'case_extraction_submission.json',{'case_id':r['case_id'],'task_id':task.task_id,'predictions':predictions})
    write(out/'case_run_manifest.json',{'case_id':r['case_id'],'config_path':str(config),'article_runs':articles,'failed_article_ids':[],'prediction_count':len(predictions)})
    write(out/'extraction_complete.json',{'case_id':r['case_id'],'article_count':len(articles),'prediction_count':len(predictions)})


def review(r,config,submission,out):
    sys.path.insert(0,str(EXTRACT))
    from constrained_meta_agent.loader import load_case_config
    from evidence_reviewer.reviewer import review_submission,write_review_summary
    task=load_case_config(config)
    result=review_submission(task,read(submission),configured_model=r['options']['model'],batch_size=12,max_workers=r['options']['max_workers'],max_retries=r['options']['max_retries'],retry_delay_seconds=5)
    write(out/'candidate_evidence_review_structured.json',result)
    write_review_summary(out/'review_summary.md',result)
    if result['failed_review_count']: raise RuntimeError('candidate review failed; see review artifacts')
    write(out/'review_complete.json',{'case_id':r['case_id'],'reviewed_count':result['completed_review_count']})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['screen','extract','review','extract-check','plan-check'])
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--submission',type=Path)
    a=parser.parse_args();r=read(a.request)
    if a.stage=='plan-check':
        sys.path.insert(0,str(ROOT/'vendor/evidence_analysis'))
        from analysis_engine.benchmark_case_adapter import _hydrate_method_neutral_structured_profile
        from analysis_core.plan_resolver import resolve_analysis_plan_structured
        plan=resolve_analysis_plan_structured(r['protocol'],{'task_id':r['case_id']+'_analysis'},_hydrate_method_neutral_structured_profile(r['analysis_profile']))
        write(a.output/'plan_check.json',{'status':'valid','analyses':len(plan.analyses),'remote_calls':0})
    elif a.stage=='screen': screen(r,a.output)
    elif a.stage in ('extract','extract-check'):extract(r,a.config,a.output,a.stage=='extract-check')
    else:review(r,a.config,a.submission,a.output)

if __name__=='__main__':main()
