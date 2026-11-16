from __future__ import annotations
import copy
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import shutil
import tempfile
from datetime import datetime,timezone
from .contracts import (read,write,digest,normalize_request,accepted_articles,prepare_extraction,evidence_handoff,write_analysis_input,require,ContractError)

ROOT=Path(__file__).resolve().parents[1]
ANALYSIS=ROOT/'vendor/evidence_analysis'
WORKER=ROOT/'metapipeline/worker.py'

def now():return datetime.now(timezone.utc).isoformat()

def runtime_environment(r,env_file=None):
    env=os.environ.copy()
    if env_file:
        from dotenv import dotenv_values
        env.update({k:v for k,v in dotenv_values(env_file).items() if v is not None})
    # Explicit credentials are confined to the child environment, never request snapshots.
    provider=r['options']['provider']; model=r['options']['model']
    keyname={'official_openai':'OPENAI_API_KEY','openai_compatible':'COMPATIBLE_API_KEY'}[provider]
    key=env.get('PIPELINE_LLM_API_KEY') or env.get(keyname)
    if key:
        env['PIPELINE_LLM_API_KEY']=key;env[keyname]=key
    default_url={'official_openai':'https://api.openai.com/v1','openai_compatible':''}[provider]
    url=env.get('PIPELINE_LLM_BASE_URL') or env.get({'official_openai':'OFFICIAL_OPENAI_BASE_URL','openai_compatible':'COMPATIBLE_BASE_URL'}[provider]) or default_url
    if url:env['PIPELINE_LLM_BASE_URL']=url
    env['META_AGENT_LLM_PROVIDER']=provider
    env['META_AGENT_COMPATIBLE_STRICT']='1'
    env['META_AGENT_COMPATIBLE_MODEL_OVERRIDE']=model
    env['META_AGENT_COMPATIBLE_BASE_URL_OVERRIDE']=url
    env['OFFICIAL_OPENAI_MODEL']=model
    env['OFFICIAL_OPENAI_BASE_URL']=url
    if key:
        if not env.get('ANALYSIS_LLM_API_KEY'):env['ANALYSIS_LLM_API_KEY']=key
    if not env.get('ANALYSIS_LLM_BASE_URL'):env['ANALYSIS_LLM_BASE_URL']=url
    if not env.get('ANALYSIS_LLM_MODEL'):env['ANALYSIS_LLM_MODEL']=model
    # Embeddings may use a different provider; no automatic copying of third-party credentials.
    env['PYTHONUNBUFFERED']='1'
    # Child analysis modules import shared evidence contracts from this package.
    env['PYTHONPATH']=str(ROOT)
    env['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
    return env

def fingerprint(r):
    files={}
    for d in r['documents']:
        for k,p in d['paths'].items():
            if k in ('markdown','content_list','pdf'):
                files[f"{d['article_id']}:{k}"]=hashlib.sha256(Path(p).read_bytes()).hexdigest()
    implementation={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'metapipeline').glob('*.py'))}
    return digest({'request':r,'document_sha256':files,'implementation':implementation,'sources':read(ROOT/'source_archives.json')})

def preflight_runtime(r,env):
    validate_components(r,env)
    require(bool(env.get('PIPELINE_LLM_API_KEY')), 'Set PIPELINE_LLM_API_KEY or the selected provider key in --env-file')
    require(bool(env.get('PIPELINE_LLM_BASE_URL')), 'Set PIPELINE_LLM_BASE_URL for the selected provider')
    require(bool(env.get('ANALYSIS_EMBEDDING_API_KEY') or env.get('OPENAI_API_KEY')), 'Set ANALYSIS_EMBEDDING_API_KEY before starting the full pipeline')
    rscript=shutil.which('Rscript',path=env.get('PATH'))
    require(bool(rscript),'Rscript is required by the statistical analysis component')
    result=subprocess.run([rscript,'--vanilla','-e',"quit(status=if(requireNamespace('metafor', quietly=TRUE)) 0 else 1)"],env=env,capture_output=True,text=True,timeout=60)
    require(result.returncode==0,'Install the R package metafor before starting the full pipeline')

def validate_components(r,env):
    with tempfile.TemporaryDirectory(prefix='meta_pipeline_check_') as temp:
        temp=Path(temp);write(temp/'request.json',r)
        config=prepare_extraction(r,[d['article_id'] for d in r['documents']],temp/'config')
        for stage in ('extract-check','plan-check'):
            result=subprocess.run([sys.executable,str(WORKER),stage,'--request',str(temp/'request.json'),'--config',str(config),'--output',str(temp)],env=env,cwd=ROOT,capture_output=True,text=True)
            require(result.returncode==0,f'{stage} failed: '+result.stderr[-2000:])

class Runner:
    def __init__(self,r,out,env,resume=False,initial_artifacts=()):
        self.r=r; self.out=Path(out).resolve();self.env=env
        self.manifest=self.out/'pipeline_manifest.json'
        fp=fingerprint(r)
        if self.manifest.exists():
            require(resume,'output already contains a run; use --resume or a new output directory')
            self.state=read(self.manifest)
            require(self.state['input_fingerprint']==fp,'request/documents changed; use a new output directory')
            require(self.state.get('mode')=='live','replay output cannot be resumed as a live run')
        else:
            require(not self.out.exists() or all(p.name in initial_artifacts for p in self.out.iterdir()),'output directory must be empty')
            self.state={'schema_version':'pipeline_run/1.0','mode':'live','case_id':r['case_id'],'input_fingerprint':fp,'started_at':now(),'status':'running','stages':{}}
        self.save('running')
        write(self.out/'request.normalized.json',r)
    def save(self,status):
        self.state['status']=status;self.state['updated_at']=now();write(self.manifest,self.state)
    def stage(self,name,command,outputs):
        previous=self.state['stages'].get(name,{})
        def checkpoint_outputs():
            checks={}
            for path in outputs:
                path=Path(path).resolve()
                if not path.is_file(): return None
                checks[str(path.relative_to(self.out))]=hashlib.sha256(path.read_bytes()).hexdigest()
            return checks
        if previous.get('status')=='completed':
            for p,sha in previous['outputs'].items():
                path=self.out/p
                require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==sha,f'checkpoint changed: {p}; start a new run')
            return
        recovered=checkpoint_outputs() if outputs else None
        if previous.get('status') in ('paused','running') and recovered is not None:
            self.state['stages'][name]={'status':'completed','completed_at':now(),'outputs':recovered,'recovered_after_pause':True}
            self.save('running');return
        print(f'[pipeline] {name}',flush=True)
        self.state['stages'][name]={'status':'running','started_at':now()};self.save('running')
        log=self.out/'logs'/f'{name}.log';log.parent.mkdir(parents=True,exist_ok=True)
        try:
            with log.open('a',encoding='utf-8') as handle:
                result=subprocess.run(command,cwd=ROOT,env=self.env,stdout=handle,stderr=subprocess.STDOUT)
        except KeyboardInterrupt:
            partial={}
            for path in outputs:
                path=Path(path).resolve()
                if path.is_file():partial[str(path.relative_to(self.out))]=hashlib.sha256(path.read_bytes()).hexdigest()
            self.state['stages'][name]={'status':'paused','paused_at':now(),'partial_outputs':partial}
            self.save('paused')
            raise
        require(result.returncode==0,f'{name} failed (exit {result.returncode}); see {log}')
        checks={}
        for path in outputs:
            path=Path(path).resolve()
            require(path.is_file(),f'{name}: missing expected output {path.name}')
            checks[str(path.relative_to(self.out))]=hashlib.sha256(path.read_bytes()).hexdigest()
        self.state['stages'][name]={'status':'completed','completed_at':now(),'outputs':checks};self.save('running')
    def worker(self,stage,out,*extra):
        return [sys.executable,str(WORKER),stage,'--request',str(self.out/'request.normalized.json'),'--output',str(out),*map(str,extra)]
    def run(self):
        try:
            screening_dir=self.out/'01_screening';extract_dir=self.out/'02_extraction';review_dir=self.out/'03_review';inputs=self.out/'04_analysis_input';analysis=self.out/'05_analysis'
            self.stage('screening',self.worker('screen',screening_dir),[screening_dir/'screening_results.json',screening_dir/'risk_of_bias_recommendations.json',screening_dir/'screening_complete.json'])
            screening=read(screening_dir/'screening_results.json');accepted_articles(self.r,screening)
            resolution_path=self.out/'screening_resolutions.json'
            if resolution_path.exists():
                resolutions=read(resolution_path)
                require(isinstance(resolutions,list),'screening_resolutions must be an array')
                by_id={x['candidate_id']:x for x in screening['decisions']}; seen=set()
                for res in resolutions:
                    aid=res.get('article_id')
                    require(aid not in seen and aid in by_id,'unknown/duplicate screening resolution')
                    seen.add(aid)
                    require(by_id[aid]['final_status']=='Unsure','resolutions only apply to Unsure articles')
                    require(res.get('decision') in ('Include','Exclude') and res.get('reason') and res.get('reviewer'),'resolution needs decision, reason, reviewer')
                    by_id[aid]['final_status']=res['decision'];by_id[aid]['human_resolution']=res
            queue=[x for x in screening['decisions'] if x['final_status']=='Unsure']
            write(self.out/'manual_review_queue.json',queue)
            if queue:
                self.save('needs_screening_review');return self.state
            included=accepted_articles(self.r,screening)
            write(screening_dir/'effective_screening_results.json',screening)
            selection_hash=digest(screening)
            if self.state.get('effective_screening_hash'):
                require(self.state['effective_screening_hash']==selection_hash,'screening resolutions changed after downstream work started; use a new output directory')
            self.state['effective_screening_hash']=selection_hash
            if not included:self.save('no_included_articles');return self.state
            config=prepare_extraction(self.r,included,self.out/'extraction_config')
            self.stage('extraction',self.worker('extract',extract_dir,'--config',config),[extract_dir/'case_extraction_submission.json',extract_dir/'case_run_manifest.json',extract_dir/'extraction_complete.json'])
            submission=read(extract_dir/'case_extraction_submission.json')
            if not submission['predictions']:self.save('no_extracted_evidence');return self.state
            extraction_manifest=read(extract_dir/'case_run_manifest.json')
            require(not extraction_manifest.get('failed_article_ids'),'one or more article extractions failed')
            missing=set(included)-{x.get('article_id') for x in submission['predictions']}
            write(self.out/'extraction_review_queue.json',{'articles_without_predictions':sorted(missing),'article_runs':extraction_manifest.get('article_runs',[])})
            if missing:
                self.save('needs_extraction_review');return self.state
            self.stage('review',self.worker('review',review_dir,'--config',config,'--submission',extract_dir/'case_extraction_submission.json'),[review_dir/'candidate_evidence_review_structured.json',review_dir/'review_complete.json'])
            review=read(review_dir/'candidate_evidence_review_structured.json')
            candidates,audit=evidence_handoff(self.r,screening,submission,review)
            write(self.out/'handoff_audit.json',audit)
            if not candidates['candidates']:self.save('no_retained_evidence');return self.state
            write_analysis_input(self.r,candidates,inputs)
            self.stage('analysis_check',[sys.executable,str(ANALYSIS/'run_analysis.py'),'--input-dir',str(inputs),'--output-dir',str(analysis),'--check'],[])
            self.stage('analysis',[sys.executable,str(ANALYSIS/'run_analysis.py'),'--input-dir',str(inputs),'--output-dir',str(analysis)],[analysis/'analysis_summary.json',analysis/'analysis_result.json'])
            summary=read(analysis/'analysis_summary.json')
            warnings=[x for x in audit['records'] if x['warnings'] or x['disposition']=='needs_further_review' or x['verification_status'] not in ('accepted',None)]
            write(self.out/'brain_result.json',{'schema_version':'brain_result/1.0','case_id':self.r['case_id'],'analysis_summary':summary,'analysis_result_path':'05_analysis/analysis_result.json','handoff_audit_path':'handoff_audit.json','upstream_warnings':warnings,'extraction_issues':[x for x in extraction_manifest.get('article_runs',[]) if x.get('issues')]})
            self.state['analysis_status']=summary.get('status')
            self.save('completed')
            return self.state
        except KeyboardInterrupt:
            for stage in self.state['stages'].values():
                if stage['status']=='running':stage['status']='paused';stage['paused_at']=now()
            self.save('paused');return self.state
        except Exception as exc:
            for stage in self.state['stages'].values():
                if stage['status']=='running':stage['status']='failed'
            self.state['error']={'type':type(exc).__name__,'message':str(exc)};self.save('failed');raise


def replay(r,screening,submission,review,out,env):
    out=Path(out).resolve()
    require(not out.exists() or not any(out.iterdir()),'replay output directory must be empty')
    candidates,audit=evidence_handoff(r,screening,submission,review)
    write(out/'request.normalized.json',r)
    config=prepare_extraction(r,accepted_articles(r,screening),out/'extraction_config')
    write_analysis_input(r,candidates,out/'analysis_input')
    write(out/'handoff_audit.json',audit)
    # Actual frozen loader and Analysis adapter, without a model call or pooled result.
    for command in [
        [sys.executable,str(WORKER),'extract-check','--request',str(out/'request.normalized.json'),'--config',str(config),'--output',str(out)],
        [sys.executable,str(ANALYSIS/'run_analysis.py'),'--input-dir',str(out/'analysis_input'),'--output-dir',str(out/'analysis'),'--check']]:
        result=subprocess.run(command,cwd=ROOT,env=env,check=False,capture_output=True,text=True)
        require(result.returncode==0,'frozen component validation failed: '+result.stderr[-3000:])
    result={'mode':'replay','status':'contracts_valid','case_id':r['case_id'],'candidate_count':len(candidates['candidates']),'remote_calls':0,'statistical_analysis_executed':False}
    write(out/'pipeline_manifest.json',result)
    return result
