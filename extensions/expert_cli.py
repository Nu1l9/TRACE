"""CLI and terminal conversation for the external expert mode."""
import argparse
import json
from pathlib import Path
from . import expert
from .common import read,ContractError
from .settings import load_settings
from .__main__ import _env


def callback(args):
    if getattr(args,'completion',None):
        response=read(args.completion);return lambda _:response
    if getattr(args,'live',False):
        from .transport import json_callback
        return json_callback(_env(args.env_file))
    return None


def main(argv=None):
    parser=argparse.ArgumentParser(description='Expert-guided mode: plan → discuss → approve → run')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('start','show','chat','approve','export','run','source','observe','history','context','restore'):
        p=sub.add_parser(name);p.add_argument('--session',type=Path,required=True)
        if name in ('start','chat'):
            g=p.add_mutually_exclusive_group();g.add_argument('--live',action='store_true');g.add_argument('--completion',type=Path)
            p.add_argument('--env-file',type=Path)
        if name=='start':p.add_argument('--topic',required=True);p.add_argument('--known',type=Path)
        if name=='chat':
            p.add_argument('--message');p.add_argument('--changes',type=Path)
            p.add_argument('--resolve-uncertainties',action='store_true')
        if name=='approve':
            p.add_argument('--version',type=int,required=True);p.add_argument('--stage',choices=['all',*expert.STAGES],default='all');p.add_argument('--note',default='')
        if name=='export':p.add_argument('--output',type=Path,required=True)
        if name=='run':p.add_argument('--extensions',type=Path);p.add_argument('--env-file',type=Path)
        if name=='source':
            p.add_argument('--article-id',required=True);p.add_argument('--start-line',type=int,required=True);p.add_argument('--end-line',type=int,required=True)
        if name=='observe':p.add_argument('--include-results',action='store_true')
        if name=='restore':p.add_argument('--version',type=int,required=True);p.add_argument('--note',required=True)
    a=parser.parse_args(argv)
    try:
        if a.command=='start':
            result=expert.start(a.session,a.topic,read(a.known) if a.known else None,a.known.resolve().parent if a.known else Path.cwd(),callback(a))
        elif a.command=='chat':
            cb=callback(a)
            if not a.message:
                if a.changes:raise ContractError('--changes requires --message')
                print('Expert discussion. Enter /show to view the plan or /quit to exit. Use the separate approve/run commands to approve and execute.')
                while True:
                    try:message=input('Expert> ').strip()
                    except (EOFError,KeyboardInterrupt):break
                    if message=='/quit':break
                    if not message:continue
                    if message=='/show':print(json.dumps(expert.view(expert.load(a.session)),ensure_ascii=False,indent=2));continue
                    result=expert.reply(a.session,message,cb)
                    print(f"Brain> {result['reply']}\nPlan revision {result['version']}: {result['status']}")
                    for question in result['questions']:print('Pending question: '+question)
                    print('Changed fields: '+', '.join(x['path'] for x in result['changes']))
                return 0
            result=expert.reply(a.session,a.message,cb,read(a.changes) if a.changes else None,
                a.changes.resolve().parent if a.changes else Path.cwd(),a.resolve_uncertainties)
        elif a.command=='approve':result=expert.approve(a.session,a.version,a.stage,a.note)
        elif a.command=='export':result=expert.export(a.session,a.output)
        elif a.command=='run':result=expert.execute(a.session,load_settings(a.extensions),a.env_file)
        elif a.command=='source':result=expert.excerpt(a.session,a.article_id,a.start_line,a.end_line)
        elif a.command=='observe':result=expert.observe(a.session,a.include_results)
        elif a.command=='restore':result=expert.restore(a.session,a.version,a.note)
        elif a.command=='history':result=expert.load(a.session)['revisions']
        elif a.command=='context':result=expert.context(expert.load(a.session))
        else:result=expert.view(expert.load(a.session))
        print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False));return 0
    except (ContractError,ValueError,FileNotFoundError,KeyError,TypeError) as exc:
        print('ERROR: '+str(exc));return 1
