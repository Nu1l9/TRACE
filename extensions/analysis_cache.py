"""Bind analysis checkpoints to the inputs that produced them."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path


def _fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(',', ':'),
        allow_nan=False,
    ).encode('utf-8')).hexdigest()


def _has_completed_result(output: Path) -> bool:
    """A finished result must never be discarded implicitly on resume."""
    return any((output / name).is_file() for name in (
        'analysis_result.json', 'analysis_summary.json',
    ))


def _archive_incomplete_attempt(output: Path, lock: Path, reason: str) -> str:
    """Preserve a partial analysis before rebuilding it under new code.

    ``05_analysis`` contains only derived analysis artefacts.  Upstream source,
    extraction, and review checkpoints live in earlier stage directories and
    are never moved here.  This makes an interrupted stage-seven repair
    resumable while retaining an auditable copy of the failed attempt.
    """
    history = output / 'partial_analysis_history' / str(time.time_ns())
    history.mkdir(parents=True, exist_ok=False)
    moved = []
    for child in list(output.iterdir()):
        if child == history or child.name == 'partial_analysis_history':
            continue
        shutil.move(str(child), str(history / child.name))
        moved.append(child.name)
    (history / 'restart_reason.json').write_text(json.dumps({
        'schema_version': 'analysis_partial_restart/8.1',
        'reason': reason,
        'archived_items': moved,
    }, indent=2) + '\n', encoding='utf-8')
    return str(history.relative_to(output))


def bind_analysis_inputs(output: Path, inputs: dict) -> str:
    """Bind analysis evidence separately from the implementation version.

    An implementation edit must not force a new seven-stage run when the
    screening, extraction, review, and analysis-input locks still match.  It
    does invalidate *partial* analysis checkpoints, which are archived and
    rebuilt.  Completed results remain immutable until the user explicitly
    invokes ``retry-analysis``.
    """
    output = Path(output)
    implementation = inputs.get('implementation')
    scientific_inputs = {key: value for key, value in inputs.items()
                         if key != 'implementation'}
    scientific_fingerprint = _fingerprint(scientific_inputs)
    signature = _fingerprint(inputs)
    lock = output / 'analysis_inputs.lock.json'
    restart = None
    if lock.exists():
        previous = json.loads(lock.read_text(encoding='utf-8'))
        previous_scientific = previous.get('scientific_fingerprint')
        if previous_scientific is None:
            # Legacy analysis locks stored only a combined hash.  A failed/incomplete attempt
            # cannot be reused, but its upstream stages are separately locked
            # by runner/stage_inputs.  Migrate this one narrow legacy state.
            if _has_completed_result(output):
                raise ValueError('Existing completed analysis has no separable scientific input binding; use retry-analysis.')
            restart = _archive_incomplete_attempt(
                output, lock,
                'legacy combined analysis lock cannot distinguish implementation from scientific inputs',
            )
        elif previous_scientific != scientific_fingerprint:
            raise ValueError('Analysis inputs or implementation changed; use a new analysis output directory.')
        elif previous.get('implementation_fingerprint') != implementation:
            if _has_completed_result(output):
                raise ValueError('Analysis implementation changed after a completed result; use retry-analysis.')
            restart = _archive_incomplete_attempt(
                output, lock,
                'analysis implementation changed after an incomplete attempt',
            )
    if not lock.exists() or restart is not None:
        if any((output / name).exists() for name in (
            'analysis_result.json', 'analysis_summary.json', 'checkpoints',
        )):
            raise ValueError('Existing analysis has no input binding; use a new analysis output directory.')
        output.mkdir(parents=True, exist_ok=True)
        temporary = lock.with_suffix('.json.tmp')
        temporary.write_text(json.dumps({
            'schema_version': 'analysis_inputs/8.1',
            'fingerprint': signature,
            'scientific_fingerprint': scientific_fingerprint,
            'implementation_fingerprint': implementation,
            'partial_restart_archive': restart,
        }, indent=2) + '\n', encoding='utf-8')
        temporary.replace(lock)
    return signature


def analysis_implementation_fingerprint(root: Path) -> str:
    paths = [*root.joinpath('extensions').glob('*.py')]
    vendor = root / 'vendor' / 'integration'
    paths.extend(vendor.rglob('*.py'))
    paths.extend(vendor.rglob('*.R'))
    paths.extend(vendor.rglob('statistical_policy.json'))
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(str(path.relative_to(root)).encode('utf-8'))
        digest.update(b'\0')
        digest.update(path.read_bytes())
    return digest.hexdigest()
