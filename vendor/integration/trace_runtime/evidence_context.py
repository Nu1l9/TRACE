"""Source-grounded review context carried across the Resilient evidence boundary.

These are runtime review observations, never benchmark labels. An upstream role
is not a final analysis decision. Conflicts require a recorded source resolution
before the candidate can contribute to a quantitative pool.
"""
from __future__ import annotations

import copy
from typing import Any

CONTEXT_VERSION = 'evidence_review_context/baseline'


def numeric_fields(value: Any, prefix: str = 'raw_data') -> dict[str, int | float]:
    """Identify supplied numeric facts; metadata strings and booleans are not facts."""
    result: dict[str, int | float] = {}
    if isinstance(value, dict):
        for name, child in value.items():
            if str(name).startswith('_'):
                continue
            result.update(numeric_fields(child, f'{prefix}.{name}'))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.update(numeric_fields(child, f'{prefix}[{index}]'))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        result[prefix] = value
    return result


def build_review_context(prediction: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    """Keep audit observations separate from the numeric evidence itself."""
    diagnostics = {
        'verification_status': prediction.get('verification_status'),
        'numeric_assessment': copy.deepcopy(prediction.get('quantitative_usability')),
        'issues': copy.deepcopy(prediction.get('issues') or prediction.get('organizer_issues') or []),
    }
    return {
        'schema_version': CONTEXT_VERSION,
        'disposition': decision.get('disposition'),
        'evidence_role': decision.get('evidence_role'),
        'reason': decision.get('reason') or '',
        'extraction_diagnostics': diagnostics,
    }


def needs_source_resolution(context: dict[str, Any] | None) -> bool:
    """A non-primary role is a question for analysis, not an automatic exclusion.

The legacy locator's human_review/unresolved status does not by itself create a
numeric conflict, and a 'poolable' extraction hint never establishes validity.
"""
    context = context or {}
    return (context.get('evidence_role') in {'source_only', 'uncertain'}
            or context.get('disposition') == 'needs_further_review')


def supplied_source_ids(location: dict[str, Any]) -> set[str]:
    ids = {str(location['source_id'])} if location.get('source_id') and str(location.get('source_quote') or '').strip() else set()
    for block in location.get('source_evidence', []) or []:
        if isinstance(block, dict) and block.get('source_id') and str(block.get('source_quote') or block.get('text') or '').strip():
            ids.add(str(block['source_id']))
    return ids


def resolution_problem(candidate: dict[str, Any], resolution: dict[str, Any] | None) -> str | None:
    """Return an audit reason when a flagged record has no defensible resolution.

This validates a semantic review's provenance and completeness; it deliberately
never performs literal quotation matching or manufactures replacement numbers.
"""
    location = candidate.get('source_location') or {}
    context = location.get('review_context') or {}
    if not needs_source_resolution(context):
        # A primary candidate already carries its source location.  A reviewer
        # may voluntarily emit a positive or unresolved resolution for it, but
        # that optional observation must not turn into a second mandatory
        # field-by-field transcription task.  Only an explicit unsupported
        # finding can override the existing primary-candidate status.
        if not isinstance(resolution, dict) or resolution.get('decision') != 'unsupported':
            return None
        if not str(resolution.get('reason') or '').strip():
            return 'source review marked the primary candidate unsupported without a rationale'
        return 'source review explicitly marked the primary candidate unsupported'
    if not isinstance(resolution, dict):
        return 'upstream evidence concern has no source-grounded resolution'
    if resolution.get('decision') != 'supported':
        return 'source review did not resolve the upstream evidence concern'
    if not str(resolution.get('reason') or '').strip():
        return 'source resolution is missing its rationale'
    source_ids = supplied_source_ids(location)
    supports = resolution.get('field_support') or []
    if not isinstance(supports, list):
        return 'source resolution field_support must be an array'
    covered: set[str] = set()
    for item in supports:
        if not isinstance(item, dict):
            return 'source resolution contains invalid field support'
        field = item.get('field')
        if not field or field in covered:
            return 'source resolution contains missing or duplicate numeric fields'
        if item.get('source_id') not in source_ids or not str(item.get('explanation') or '').strip():
            return 'source resolution must reference supplied source evidence'
        covered.add(field)
    expected = set(numeric_fields(candidate.get('raw_data') or {}))
    expected.update(numeric_fields(candidate.get('effect') or {}, 'effect'))
    if not expected or covered != expected:
        return 'source resolution must account for every supplied numeric field'
    return None
