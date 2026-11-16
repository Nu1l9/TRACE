"""Load external RoB text without embedding instrument wording or downloading it."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from extensions.common import ROOT, ContractError

RESOURCE_ROOT = ROOT / 'resources/rob'


def catalog(root=RESOURCE_ROOT):
    return json.loads((Path(root) / 'catalog.json').read_text(encoding='utf-8'))


def safe_path(root, relative):
    root = Path(root).resolve()
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts:
        raise ValueError('Unsafe RoB resource path: ' + str(relative))
    path = root / rel
    if not path.resolve().is_relative_to(root):
        raise ValueError('RoB resource path escapes its directory: ' + str(relative))
    return path


def verified_bytes(root, row):
    path = safe_path(root, row['path'])
    message = (f"RoB resource missing or different: {row['path']}. "
               "See resources/rob/README.md for the exact version and source. "
               "Install authorized files with python scripts/install_rob_resources.py --source <resource-directory>. "
               "RoB assessment cannot proceed; missing guidance is not low risk.")
    if not path.is_file():
        raise ContractError(message)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != row['sha256'] or len(data) != row['bytes']:
        raise ContractError(message)
    return data


def load_resource_text(root, name):
    rows = [entry[k] for entry in catalog(root)['tools'] for k in ('guide', 'prompt')]
    # Old source representations are used only when recovering historical runs.
    rows += json.loads((RESOURCE_ROOT / 'required_files.json').read_text(encoding='utf-8'))['files']
    row = next((r for r in rows if r['path'] == name), None)
    if row is None:
        raise ContractError('Unregistered RoB resource: ' + name)
    data = verified_bytes(root, row)
    # Match historical Path.read_text universal-newline semantics.
    return data.decode('utf-8').replace('\r\n', '\n').replace('\r', '\n')


def load_compact_guidance(root, tool, version=None):
    entries = [e for e in catalog(root)['tools'] if e['tool'] == tool]
    if version is not None:
        entries = [e for e in entries if e['version'] == version]
    elif tool == 'ROBINS-I':
        entries = [e for e in entries if e['id'] == 'robins-i']
    if len(entries) != 1:
        raise ContractError(f'No unique registered RoB resource for {tool} ({version}).')
    return load_resource_text(root, entries[0]['prompt']['path'])
