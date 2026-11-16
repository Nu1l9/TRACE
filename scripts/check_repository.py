"""Check tracked (or candidate source) files without printing secret values."""
import argparse
import json
import re
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FORBIDDEN={'data','runs','outputs','audit','gold','benchmark_runs','benchmark_packages','expert_qa_sessions','expert_qa_library','.venv','__pycache__'}
INTERNAL_LABELS=re.compile(r'(?<![A-Za-z0-9])(?:[Vv](?:1[0-9]|2[0-9]))(?![A-Za-z0-9])|(?:fixed_policy|method_selector)_v[0-9]+|\bv[0-9]+_agent\b|(?:FixedMethodSelector|FixedIntervalCalculator|ToolAwareMethodSelector)V[0-9]+|\btrace_v[0-9]+\b')
RETIRED_NAMES=re.compile(r'meta[-_ ]?pipeline', re.IGNORECASE)
PUBLIC_IMAGES={'docs/assets/fig1-system.png', 'docs/assets/fig5-evidence-qa.png'}
SECRETS=[re.compile(r'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}'),re.compile(r'\bgh[pousr]_[A-Za-z0-9]{24,}'),re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')]
def findings(paths):
 errors=[]
 for path in paths:
  rel=path.relative_to(ROOT);name=path.name
  if rel.as_posix().startswith('resources/rob/') and rel.as_posix() not in {'resources/rob/README.md', 'resources/rob/catalog.json', 'resources/rob/required_files.json', 'resources/rob/text/.gitkeep'}:
   errors.append((str(rel),'external RoB text must not enter the source distribution'))
  if INTERNAL_LABELS.search(rel.as_posix()):errors.append((str(rel),'internal development label in path'))
  if RETIRED_NAMES.search(rel.as_posix()):errors.append((str(rel),'retired project name in path'))
  if path.is_symlink():errors.append((str(rel),'symlink'));continue
  if not path.is_file():continue
  if any(part in FORBIDDEN for part in rel.parts) or (name.startswith('.env') and name!='.env.example') or name.endswith('.env') or str(rel)=='config/provider.json':
   errors.append((str(rel),'private/runtime file'))
  if path.stat().st_size>10_000_000:errors.append((str(rel),'large source file'))
  if rel.as_posix() in PUBLIC_IMAGES:
   try:
    from PIL import Image
    with Image.open(path) as im:
     if im.format != 'PNG':raise ValueError('Expected PNG')
     im.verify()
   except Exception:errors.append((str(rel),'invalid public figure'))
   continue
  try:text=path.read_text()
  except UnicodeError:errors.append((str(rel),'unexpected binary'));continue
  if INTERNAL_LABELS.search(text):errors.append((str(rel),'internal development label'))
  if RETIRED_NAMES.search(text):errors.append((str(rel),'retired project name'))
  if any(p.search(text) for p in SECRETS):errors.append((str(rel),'possible credential'))
  # The scanner's own pattern is generic, never embeds a developer identity.
  if rel.as_posix() != 'scripts/check_repository.py' and re.search(r'/Users/[^/\s]+/|[A-Za-z]:\\Users\\[^\\\s]+\\',text):errors.append((str(rel),'workstation path'))
 return errors
def main():
 p=argparse.ArgumentParser();p.add_argument('--tracked',action='store_true');a=p.parse_args()
 if a.tracked:
  names=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
  paths=[ROOT/n for n in names if n]
 else:
  # Respect Git ignore rules, including locally installed research resources.
  names=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')
  paths=sorted({ROOT/n for n in names if n})
 issues=findings(paths)
 print(json.dumps({'files_checked':len(paths),'issues':[{'path':p,'reason':r} for p,r in issues]},indent=2))
 return bool(issues)
if __name__=='__main__':raise SystemExit(main())
