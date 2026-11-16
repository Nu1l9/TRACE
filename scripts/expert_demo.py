"""Reproduce expert alignment without any model API call."""
import argparse
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from extensions import expert
from extensions.common import BASELINE,read,write,require
from trace_runtime.contracts import normalize_request


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();require(not out.exists(),'Demo output already exists')
    draft=expert.start(out/'topic_only','Compare intervention versus control on the primary clinical outcome')
    require(draft['status']=='draft','Expected incomplete topic-only draft')
    original=read(BASELINE/'examples/request.json');session=out/'expert_session'
    expert.start(session,original['topic'],original,BASELINE/'examples')
    response={'schema_version':'expert_dialogue_response/1.0','reply':'Updated the primary analysis title as requested while keeping the study scope unchanged.',
        'changes':[{'op':'set','path':'analysis_profile.primary_analysis_spec.title','value':'Expert aligned synthetic outcome','reason':'explicit expert request'}],
        'uncertainties':[],'questions':[]}
    changed=expert.reply(session,'Rename the primary analysis title to Expert aligned synthetic outcome',lambda _:response)
    require(changed['version']==2,'Revision not created')
    expert.approve(session,2,note='Synthetic expert alignment only')
    expert.export(session,out/'approved_request.json')
    normalize_request(read(out/'approved_request.json'),out)
    expert.excerpt(session,'article_a',1,1)
    restored=expert.restore(session,1,'Synthetic restoration test')
    require(restored['version']==3 and not restored['approved_stages'],'Restore must create an unapproved revision')
    require(restored['impact']['classification']=='post_results_amendment','Result exposure marker lost')
    report={'status':'passed','remote_model_calls':0,'topic_only_draft':True,
        'dialogue_response':'fixed synthetic response','expert_revision_created':True,
        'real_frozen_loader_and_plan_validation':True,'approved_request_export_valid':True,
        'restoration_preserves_history':True,'post_results_amendment_marked':True,
        'pipeline_execution_in_this_demo':False}
    write(out/'verification_report.json',report);print(__import__('json').dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
