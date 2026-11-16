"""Run source-only test files in separate processes to isolate legacy module stubs."""
import json
import os
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1')
 results=[]
 for path in sorted((ROOT/'tests').glob('test_*.py')):
  print('\nTesting '+path.name,flush=True)
  p=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p',path.name,'-v'],cwd=ROOT,env=env)
  results.append({'suite':path.name,'exit_code':p.returncode})
 p=subprocess.run([sys.executable,'-m','unittest','discover','-s','expert_qa/tests','-v'],cwd=ROOT,env=env)
 results.append({'suite':'expert_qa','exit_code':p.returncode})
 print(json.dumps({'suites':len(results),'failures':[r for r in results if r['exit_code']]},indent=2))
 return any(r['exit_code'] for r in results)
if __name__=='__main__':raise SystemExit(main())
