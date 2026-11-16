"""Recover exact compact inputs from an authorized historical source copy, without executing it."""
import argparse
import ast
import hashlib
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extensions.rob_resources import catalog, safe_path


def prepare(legacy_code, output, entries):
    values = {}
    for node in ast.parse(Path(legacy_code).read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    'COMPACT_TOOL_GUIDANCE', 'COMPACT_ROBINS_I_2016_GUIDANCE'
                }:
                    values[target.id] = ast.literal_eval(node.value)
    staged = []
    for entry in entries:
        try:
            value = (values['COMPACT_ROBINS_I_2016_GUIDANCE'] if entry['id'] == 'robins-i-2016'
                     else values['COMPACT_TOOL_GUIDANCE'][entry['tool']])
        except (KeyError, TypeError) as exc:
            raise ValueError('Historical source is missing the registered prompt for ' + entry['id']) from exc
        data = value.encode('utf-8')
        row = entry['prompt']
        if hashlib.sha256(data).hexdigest() != row['sha256'] or len(data) != row['bytes']:
            raise ValueError('Historical prompt differs from the frozen inventory: ' + entry['id'])
        staged.append((safe_path(output, row['path']), data))
    for path, data in staged:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return len(staged)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy-code', type=Path, required=True, help='Authorized historical screening_bias.py containing the frozen prompts.')
    p.add_argument('--output', type=Path, required=True, help='Local external resource directory; not a public distribution directory.')
    a = p.parse_args()
    try:
        count = prepare(a.legacy_code, a.output, catalog()['tools'])
    except (OSError, ValueError, SyntaxError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f'Recovered {count} exact local prompts. This grants no redistribution rights. Run install_rob_resources.py to validate/install the complete resources.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
