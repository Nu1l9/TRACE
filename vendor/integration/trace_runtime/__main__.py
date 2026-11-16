from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from .contracts import read,write,normalize_request,ContractError
from .pipeline import ROOT,Runner,runtime_environment,replay,preflight_runtime,validate_components
from .brain import draft,propose

def main():
    parser=argparse.ArgumentParser(description='MINERS → extraction/review → Analysis analysis')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('check','run','replay'):
        p=sub.add_parser(name);p.add_argument('--request',type=Path,required=True);p.add_argument('--env-file',type=Path)
        if name!='check':p.add_argument('--output',type=Path,required=True)
        if name=='run':p.add_argument('--resume',action='store_true')
        if name=='replay':
            for flag in ('screening','submission','review'):p.add_argument('--'+flag,type=Path,required=True)
    b=sub.add_parser('brain-draft');b.add_argument('--topic',required=True);b.add_argument('--output',type=Path,required=True)
    b=sub.add_parser('brain-fill');b.add_argument('--input',type=Path,required=True);b.add_argument('--plugin',required=True);b.add_argument('--output',type=Path,required=True)
    b=sub.add_parser('import-case');b.add_argument('--case-id',required=True);b.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    try:
        if a.command=='import-case':
            from .import_case import import_case
            print(json.dumps(import_case(a.case_id,a.output),ensure_ascii=False,indent=2));return 0
        if a.command=='brain-draft':write(a.output,draft(a.topic));return 0
        if a.command=='brain-fill':write(a.output,propose(read(a.input),a.plugin));return 0
        r=normalize_request(read(a.request),a.request.resolve().parent)
        if a.command=='check':
            validate_components(r,runtime_environment(r,a.env_file))
            result={'status':'request_valid','case_id':r['case_id'],'document_count':len(r['documents']),'extraction_runtime':r['options']['extraction_runtime'],'remote_calls':0}
        else:
            env=runtime_environment(r,a.env_file)
            if a.command=='run':
                preflight_runtime(r,env)
                result=Runner(r,a.output,env,a.resume).run()
            else:result=replay(r,read(a.screening),read(a.submission),read(a.review),a.output,env)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 2 if result['status'].startswith(('needs_','no_')) else 0
    except (ContractError,FileNotFoundError,ValueError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr);return 1
if __name__=='__main__':raise SystemExit(main())
