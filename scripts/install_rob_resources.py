"""Validate/install authorized local RoB resources; never fetch or substitute text."""
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extensions.rob_resources import catalog, safe_path, verified_bytes
from extensions.common import ContractError


def install(source, target, entries, *, check_only=False):
    rows = {entry[k]['path']: entry[k] for entry in entries for k in ('guide', 'prompt')}
    # Validate all inputs and destinations before copying anything.
    for row in rows.values():
        verified_bytes(source, row)
        safe_path(target, row['path'])
    if not check_only:
        for row in rows.values():
            origin = safe_path(source, row['path'])
            destination = safe_path(target, row['path'])
            if origin.resolve() == destination.resolve():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin, destination)
    return len(rows)


def main(argv=None):
    target = ROOT / 'resources/rob'
    entries = catalog(target)['tools']
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, help='Authorized local resource directory (not a ZIP or PDF).')
    p.add_argument('--tool', action='append', choices=[e['id'] for e in entries],
                   help='Install/check one route; repeat for more. Default: all seven routes.')
    p.add_argument('--check', action='store_true', help='Validate only; without --source, check installed files.')
    p.add_argument('--list', action='store_true', help='List versions, required paths and source URLs.')
    a = p.parse_args(argv)
    selected = [e for e in entries if not a.tool or e['id'] in a.tool]
    if a.list:
        for entry in selected:
            print(f"{entry['id']}: {entry['tool']} / {entry['version']}\n  Source: {entry['source_url']}")
            for kind in ('guide', 'prompt'):
                print(f"  {kind}: resources/rob/{entry[kind]['path']}")
        return 0
    if a.source is None and not a.check:
        p.error('--source is required for installation; use --check to validate installed files.')
    try:
        count = install(a.source or target, target, selected, check_only=a.check)
    except (ContractError, ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"{'Verified' if a.check else 'Installed'} {count} RoB resource files for {len(selected)} routes. External texts remain Git-ignored.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
