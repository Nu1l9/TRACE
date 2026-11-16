"""Verify the portable integration release file manifest without remote calls."""
from pathlib import Path
import hashlib
import json
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'PACKAGE_MANIFEST.json').read_text())
for entry in manifest['files']:
    path=ROOT/entry['path']
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
        raise SystemExit('Integrity mismatch: '+entry['path'])
print(f"package_verified=true; files={len(manifest['files'])}")
