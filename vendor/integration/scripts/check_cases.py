"""Validate all imported cases against both frozen loaders without reading Gold or calling models."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from trace_runtime.contracts import read,write,normalize_request,prepare_extraction
from trace_runtime.pipeline import runtime_environment,WORKER,ANALYSIS

def main():
    results=[]
    for request in sorted((ROOT/'examples/cases').glob('ma_*.json')):
        if request.name.endswith('.import_audit.json'):continue
        r=normalize_request(read(request),request.parent)
        with tempfile.TemporaryDirectory() as temp:
            temp=Path(temp);write(temp/'request.json',r)
            config=prepare_extraction(r,[d['article_id'] for d in r['documents']],temp/'config')
            cmd=[sys.executable,str(WORKER),'extract-check','--request',str(temp/'request.json'),'--config',str(config),'--output',str(temp)]
            result=subprocess.run(cmd,env=runtime_environment(r),capture_output=True,text=True)
            if result.returncode:raise RuntimeError(request.name+result.stderr)
        # The public Analysis check loads candidates but does not compile the plan. Compile explicitly too.
        sys.path.insert(0,str(ANALYSIS))
        from analysis_engine.benchmark_case_adapter import _hydrate_method_neutral_structured_profile
        from analysis_core.plan_resolver import resolve_analysis_plan_structured
        plan=resolve_analysis_plan_structured(r['protocol'],{'task_id':r['case_id']+'_analysis'},_hydrate_method_neutral_structured_profile(r['analysis_profile']))
        results.append({'case_id':r['case_id'],'status':'valid','document_count':len(r['documents']),'runtime':r['options']['extraction_runtime'],'analysis_count':len(plan.analyses),'remote_calls':0,'scope':'configuration/loader/analysis-plan; no screening or extraction accuracy evaluation'})
        print(r['case_id'],'valid',flush=True)
    write(ROOT/'docs/component_case_validation.json',results)
    assert len(results)==22,len(results)
if __name__=='__main__':main()
