"""Synthetic integration test: real components + deterministic LLM test doubles + real R."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
ROOT=PROJECT/'vendor/integration'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'vendor/miners'))
EXTRACT=ROOT/'vendor/evidence_extraction/system'
sys.path.insert(0,str(EXTRACT));sys.path.insert(0,str(EXTRACT/'runtime/typed'))
sys.path.insert(0,str(ROOT/'vendor/evidence_analysis'))
from metapipeline.contracts import *
from miners.case_screening_rob import FullTextProtocolScreener
from constrained_meta_agent.loader import load_case_config
from locator_extractor_agent import controller as extraction_controller, extraction_models
from extensions.resilient_extraction import run_article as run_resilient_article
from extensions.input_contract import normalize_request as compile_request
from locator_extractor_agent import tools as extraction_tools
from evidence_reviewer import reviewer as review_module
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage
from contract_adapter import load_analysis_contract
from analysis_engine.method_selector import DeterministicMethodSelector
from analysis_engine.planner import DeterministicAnalysisPlanner
from analysis_engine.reporting_bias_selector import DeterministicReportingBiasSelector
from analysis_engine.retrieval import HashEmbeddingBackend
from analysis_engine.runtime_facets import RuntimeFacetResult
from analysis_engine.topic_reviewer import DeterministicTopicReviewer
from extensions.statistical_method_agent import ReleaseAnalysisAgent
from run_analysis import _summary

class StructuredFixture:
    def __init__(self,payload):self.payload=payload
    def with_structured_output(self,schema,**kwargs):
        return RunnableLambda(lambda _:schema.model_validate(self.payload))
class EmptyFacets:
    mode='offline_fixture';calls_made=0
    def annotate(self,plan,records):return RuntimeFacetResult(mode=self.mode,note='Synthetic test only')

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
    require(not out.exists() or not any(out.iterdir()),'offline output must be empty')
    r=normalize_request(compile_request(read(ROOT/'examples/request.json')),ROOT/'examples')
    decisions=[]
    for i,d in enumerate(r['documents']):
        response={'status':'Include' if i<2 else 'Exclude','reason':'Synthetic deterministic screening fixture'}
        s=FullTextProtocolScreener(lambda _,response=response:json.dumps(response))
        decisions.append(s.screen(d['article_id'],d['title'],{'page_1':Path(d['paths']['markdown']).read_text()},screening_protocol(r)))
    screening={'case_id':r['case_id'],'decisions':decisions};write(out/'screening_results.json',screening)
    included=accepted_articles(r,screening);config=prepare_extraction(r,included,out/'extraction_config');task=load_case_config(config)
    predictions=[]
    for aid in included:
        catalog,_=extraction_tools.static_clean_document(task.documents[aid],remove_reference_items_only=True)
        source=next(x for x in catalog if '10 events' in x['content'])
        row={'outcome':'prespecified outcome','target_id':'demo_target','data_family':'binary_arm_counts','exposure':'intervention','comparator':'comparator','population_or_subgroup':'eligible population','estimand':'risk_ratio','timepoint':'source-defined','model_name':'source-reported arm counts','source_id':source['source_id'],'source_quote':'Intervention: 10 events among 100 participants. Comparator: 20 events among 100 participants.','raw_data':{'intervention_events':10,'intervention_total':100,'control_events':20,'control_total':100,'outcome_unit':'participants'}}
        def model(task,role,provider,source=source,row=row):
            if role=='locator':return StructuredFixture({'source_ids':[source['source_id']]})
            return StructuredFixture({'rows':[row],'unresolved_items':[]})
        with patch.object(extraction_tools,'_llm',side_effect=model):
            result=run_resilient_article(r,task,aid,out/'articles'/aid, runtime=(load_case_config, extraction_controller, extraction_tools, extraction_models))
        write(out/'articles'/aid/'extraction_submission.json',result['extraction_submission']);write(out/'articles'/aid/'issues.json',result['issues'])
        predictions.extend(result['extraction_submission']['predictions'])
    submission={'case_id':r['case_id'],'predictions':predictions};write(out/'case_extraction_submission.json',submission)
    # The real reviewer still builds its prompt, validates each batch and joins IDs.
    def review_model(task,configured):
        class Model:
            def with_structured_output(self,schema,**kwargs):
                def answer(prompt):
                    content=prompt.to_messages()[-1].content
                    items=json.loads(content.split('their evidence (including supporting_source_evidence_ids):\n',1)[1].split('\n\nOnly return',1)[0])
                    return schema.model_validate({'reviews':[{'prediction_id':x['prediction_id'],'disposition':'keep_as_valid_candidate','evidence_role':'main_analysis_candidate','reason':'Synthetic test fixture'} for x in items]})
                return RunnableLambda(answer)
        return Model()
    with patch.object(review_module,'_model',side_effect=review_model):
        review=review_module.review_submission(task,submission,max_workers=1,max_retries=0)
    write(out/'candidate_evidence_review_structured.json',review)
    ledger,audit=evidence_handoff(r,screening,submission,review);write_analysis_input(r,ledger,out/'analysis_input');write(out/'handoff_audit.json',audit)
    protocol,task_,profile,payloads,studies=load_analysis_contract(out/'analysis_input')
    analysis=out/'analysis';analysis.mkdir()
    agent=ReleaseAnalysisAgent(statistical_method_policy='fixed',embeddings=HashEmbeddingBackend(),planner=DeterministicAnalysisPlanner(),reviewer=DeterministicTopicReviewer(),reporting_bias_selector=DeterministicReportingBiasSelector(),runtime_facet_annotator=None,pre_review_filter_mode='graph_outcome',audit_dir=analysis)
    result=agent.run(protocol=protocol,task=task_,analysis_profile=profile,extractor_payloads=payloads,study_by_article=studies,reporting_bias_artifact_dir=analysis/'reporting_bias',forest_plot_artifact_dir=analysis/'forest_plots')
    require(result.status=='completed',f'offline analysis status {result.status}')
    require(result.analyses[0].statistics is not None,'no pooled test result')
    require(abs(result.analyses[0].statistics.pooled_effect-0.5)<1e-6,'unexpected pooled synthetic RR')
    write(analysis/'analysis_result.json',result.model_dump(mode='json'));write(analysis/'analysis_summary.json',_summary(result,{}))
    report={'status':'passed','mode':'synthetic_offline','remote_calls':0,'real_components':['MINERS FullTextProtocolScreener','Typed source cleaner + locator + per-family Pydantic extraction + organizer + checkpoints','candidate reviewer structured prompt + batch validation','Analysis graph/selection/statistics + metafor'],'llm_responses':'deterministic test doubles','input_articles':3,'included_articles':len(included),'analysis_candidates':len(ledger['candidates']),'pooled_effect':result.analyses[0].statistics.pooled_effect}
    write(out/'test_report.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':
    import socket
    def deny_network(*args, **kwargs):
        raise RuntimeError('Network is disabled in the offline regression')
    with patch.object(socket.socket, 'connect', deny_network):
        main()
