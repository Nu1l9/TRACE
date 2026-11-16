"""Read-only research interpretation: evidence coverage and structured proposals."""
import json
import math
import re
from collections import Counter


def is_research_question(question):
    return bool(re.search(
        r'hypothes|summari[sz]|synthesize|future research|next steps|potential explanation|research priorities|follow-up study',
        question, re.I))

def wants_hypotheses(question):
    return bool(re.search(r'hypothes|potential explanation|underlying mechanism', question, re.I))


INSTRUCTIONS = """
Structured research-interpretation mode is active and supersedes the default 2-4 claim brevity rule.
First call get_research_context to inspect summary statistics, leave-one-out diagnostics, model comparison, data family, source resolution, and risk-of-bias coverage.
Then inspect specific primary passages or extraction records relevant to the question: any mechanistic or confounding hypothesis must check primary source text via get_source, looking for adjusted estimates, counter-evidence, and stated limitations in the same article rather than cherry-picking supportive sentences.
Low I² does not imply high robustness; fixed-effect pooling does not validate a common-effect assumption; heterogeneity contribution is distinct from study weight or pooled-effect influence.
When stability is flagged as potentially_fragile_under_leave_one_out, explain the sensitivity limitation rather than claiming high robustness.
Adjusted HRs and crude RRs estimate different quantities; do not treat their numerical difference alone as proof of confounding.
Distinct study IDs do not prove cohort independence; draft RoB assessments do not establish human confirmation; unresolved source locators do not prove absence from the primary article.
Hypotheses must be framed as testable explanations rather than causal claims, and may address methodological, confounding, or follow-up-window differences.
For each hypothesis provide: supporting_observation, counter_evidence, uncertainty, verification, and a verbatim supporting_quote (8-350 characters) from a get_source page retrieved in this turn (cited in citation_ids). Never substitute extractor summaries or analysis overviews for verbatim primary quotes; if no verbatim primary passage supports a hypothesis, leave hypotheses empty and note that limitation.
For each next_steps item provide: distinct priority (1-3), action, rationale, and possible_implications without claiming new analyses were executed.
When no counter-evidence was found in retrieved passages, write "Not identified in retrieved passages" rather than claiming none exists.
Call structured finish_answer with: overview (1-2 synthesis claims without repeating pooled RR/CI/I², which the server renders automatically), hypotheses (0-3 items; empty when only next steps are requested), next_steps (1-3 items), and limitations.
"""


def research_context(snapshot, analysis_id):
    a = snapshot.analyses[analysis_id]
    stats = a.get('statistics') or {}
    robust = a.get('robustness') or {}
    effects = (a.get('poolability') or {}).get('study_effects') or []
    records, articles = [], []
    for effect in effects:
        eid = effect.get('evidence_id')
        row = snapshot.evidence.get(eid) or snapshot.candidates.get(eid) or {}
        aid = row.get('article_id')
        if aid and aid not in articles:
            articles.append(aid)
        records.append({
            'evidence_id': eid, 'study_id': effect.get('study_id'), 'article_id': aid,
            'effect_type': effect.get('effect_type'), 'timepoint': row.get('timepoint'),
            'source_resolution': (row.get('source_location') or {}).get('resolution_status'),
        })
    assessments = []
    for aid in articles:
        if aid in snapshot.documents:
            assessments.extend(snapshot.query('get_rob', {'article_id': aid}, analysis_id)['assessments'])
    status = Counter(r.get('status', 'unknown') for r in assessments)
    loo = []
    for row in robust.get('leave_one_out') or []:
        if not isinstance(row, dict) or not row.get('evidence_id'):
            continue
        result = snapshot.leave_one_out(a, {'evidence_id': row['evidence_id']}, analysis_id)
        for match in result['matches']:
            loo.append({'evidence_id': row['evidence_id'],
                        'ci_contains_null': match['ci_interpretation']['contains_null']})
    return {
        'origin': '05_analysis/analysis_result.json + saved extraction and bias records',
        'analysis_id': analysis_id, 'status': a.get('status'),
        'definition': {k: (a.get('analysis') or {}).get(k) for k in
                       ('outcome_group', 'population_terms', 'comparison_terms', 'timepoint_policy')},
        'statistics': {k: stats.get(k) for k in
                       ('effect_type', 'model', 'k', 'pooled_effect', 'ci_lower', 'ci_upper', 'i_squared')},
        'numeric_note': 'i_squared is already a percentage: append %, never multiply by 100. k counts pooled effects.',
        'robustness': {
            'saved_stability': robust.get('stability'),
            'saved_model_comparison': robust.get('model_comparison'),
            'saved_leave_one_out_count': len(loo),
            'leave_one_out_intervals_containing_null': [r['evidence_id'] for r in loo if r['ci_contains_null'] is True],
            'leave_one_out_intervals_unknown': [r['evidence_id'] for r in loo if r['ci_contains_null'] is None],
            'heterogeneity_drivers': (robust.get('heterogeneity_drivers') or [])[:3],
            'caveats': robust.get('caveats'),
            'interpretation_note': 'Heterogeneity contribution is not pooled-effect influence or weight. Low I² is not proof of robust inference.',
        },
        'evidence_inventory': (a.get('candidate_inventory') or {}).get('candidate_record_count'),
        'pooled_effects': records[:20],
        'pooled_effects_total': len(records),
        'pooled_effects_shown': min(len(records), 20),
        'bias_coverage': {
            'articles_with_pooled_effects': len(articles),
            'saved_assessment_count': len(assessments),
            'status_counts': dict(status),
            'note': 'Assessments can have different scopes and multiple rows per article. Counts do not establish human confirmation. Query get_rob for scope and details.',
        },
        'source_coverage': {
            'articles_with_readable_source': sum(bool(snapshot.sources.get(aid, {}).get('text') or any(snapshot.sources.get(aid, {}).get('blocks', []))) for aid in articles),
            'note': 'Available source text is not the same as resolved extraction location or validated extraction.',
        },
        'warnings': (a.get('warnings') or [])[:8],
        'next_reads': 'Use get_extraction/get_source/get_rob for specific articles; get_leave_one_out for exact saved diagnostics. This packet is not a completed appraisal or a causal mechanism test.',
    }


def schema(allow_hypotheses=True):
    def obj(fields):
        return {'type': 'object', 'additionalProperties': False,
                'properties': fields, 'required': list(fields)}
    text = {'type': 'string'}
    ids = {'type': 'array', 'items': text, 'minItems': 1}
    claim = obj({'kind': {'type': 'string', 'enum': ['recorded', 'inference', 'limitation']},
                 'text': text, 'citation_ids': ids})
    hypothesis = obj({**{k: text for k in (
        'hypothesis', 'supporting_observation', 'supporting_quote', 'counter_evidence', 'uncertainty', 'verification')},
        'citation_ids': ids})
    step = obj({'priority': {'type': 'integer', 'minimum': 1, 'maximum': 3},
                **{k: text for k in ('action', 'rationale', 'possible_implications')}, 'citation_ids': ids})
    return obj({
        'overview': {'type': 'array', 'items': claim, 'minItems': 1, 'maxItems': 2},
        'hypotheses': {'type': 'array', 'items': hypothesis, 'maxItems': 3 if allow_hypotheses else 0},
        'next_steps': {'type': 'array', 'items': step, 'minItems': 1, 'maxItems': 3},
        'limitations': text,
    })


def flatten_research(value, aliases, citations, packet, allow_hypotheses=True):
    """Validate structure and issued references; no claim of semantic verification."""
    if not isinstance(value, dict) or set(value) != {'overview', 'hypotheses', 'next_steps', 'limitations'}:
        raise ValueError('Research answer needs overview, hypotheses, next_steps and limitations')
    for key, low, high in [('overview', 1, 2), ('hypotheses', 0, 3), ('next_steps', 1, 3)]:
        if not isinstance(value[key], list) or not low <= len(value[key]) <= high:
            raise ValueError('Invalid research item count: ' + key)
    if not isinstance(value['limitations'], str) or not 1 <= len(value['limitations']) <= 1500:
        raise ValueError('State the limitations of this interpretation')
    if not allow_hypotheses and value['hypotheses']:
        raise ValueError('This question requests a research plan; leave hypotheses empty')
    def refs(row):
        ids = row.get('citation_ids')
        if not isinstance(ids, list) or not ids:
            raise ValueError('Each research item requires supporting citations')
        mapped = [aliases.get(cid, cid) if isinstance(cid, str) else None for cid in ids]
        if any(cid not in citations for cid in mapped):
            raise ValueError('Unknown research citation')
        row['citation_ids'] = mapped
        return mapped
    def fields(row, keys):
        if not isinstance(row, dict) or set(row) != set(keys) | {'citation_ids'}:
            raise ValueError('Missing research fields')
        for key in keys:
            if key == 'priority':
                if type(row[key]) is not int or not 1 <= row[key] <= 3:
                    raise ValueError('Invalid priority')
            elif not isinstance(row[key], str) or not 1 <= len(row[key]) <= 700:
                raise ValueError('Empty or excessive research field: ' + key)
    claims = []
    packet_cids = [cid for cid, page in citations.items()
                   if page['tool'] == 'get_research_context' and not page['arguments'].get('field')
                   and page.get('offset') == 0]
    if not packet_cids:
        raise ValueError('Read the research context before interpreting it')
    stats = packet['statistics']
    def fmt(v):
        return str(v) if type(v) in (int, float) and math.isfinite(v) else 'not saved'
    if stats.get('pooled_effect') is not None:
        claims.append({'kind': 'recorded', 'text':
                       f"Saved result: {stats.get('effect_type') or 'Effect'} {fmt(stats.get('pooled_effect'))}; "
                       f"95% CI {fmt(stats.get('ci_lower'))}–{fmt(stats.get('ci_upper'))}; "
                       f"{fmt(stats.get('k'))} pooled effects; I² {fmt(stats.get('i_squared'))}"
                       + ('%.' if stats.get('i_squared') is not None else '.'),
                       'citation_ids': [packet_cids[0]]})
    for row in value['overview']:
        fields(row, ('kind', 'text'))
        refs(row)
        claims.append(row.copy())
    for row in value['hypotheses']:
        fields(row, ('hypothesis', 'supporting_observation', 'supporting_quote', 'counter_evidence', 'uncertainty', 'verification'))
        ids = refs(row)
        quote = row['supporting_quote']
        def normalize(text):
            return ' '.join(re.sub(r'(?m)^\d+:\s*', '', text).split())
        sources = []
        for cid in ids:
            if citations[cid]['tool'] != 'get_source':
                continue
            try:
                body = json.loads(citations[cid]['text'])
                if isinstance(body, dict) and isinstance(body.get('text'), str):
                    sources.append(body['text'])
            except (ValueError, TypeError):
                pass
        if not 8 <= len(quote) <= 350 or not any(normalize(quote) in normalize(body) for body in sources):
            raise ValueError('supporting_quote must match a short verbatim passage in a cited get_source page. If unavailable, remove the unsupported hypothesis and explain the limitation.')
        claims.append({'kind': 'inference', 'text':
                       f"Testable hypothesis: {row['hypothesis']}\nSupporting observation: {row['supporting_observation']}\nVerbatim source quote: {quote}\n"
                       f"Counter-evidence / alternative explanation: {row['counter_evidence']}\nMissing information: {row['uncertainty']}\n"
                       f"Verification step: {row['verification']}", 'citation_ids': ids})
    priorities = []
    for row in value['next_steps']:
        fields(row, ('priority', 'action', 'rationale', 'possible_implications'))
        priorities.append(row['priority'])
        claims.append({'kind': 'inference', 'text':
                       f"Follow-up priority {row['priority']}: {row['action']}\nRationale: {row['rationale']}\n"
                       f"Potential implications: {row['possible_implications']}", 'citation_ids': refs(row)})
    if len(set(priorities)) != len(priorities):
        raise ValueError('Research priorities must be distinct')
    claims.append({'kind': 'limitation', 'text': value['limitations'], 'citation_ids': []})
    return claims
