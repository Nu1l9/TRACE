"""Allowlisted saved-run adapter. No pipeline, provider, or evaluation imports."""
from __future__ import annotations
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

FILES = {
    'request': 'request.normalized.json', 'manifest': 'pipeline_manifest.json',
    'preparation': 'document_preparation.json',
    'screening': '01_screening/effective_screening_results.json',
    'screening_original': '01_screening/screening_results.json',
    'rob': '01_screening/risk_of_bias_assessments.json',
    'extraction': '02_extraction/case_extraction_submission.json',
    'review': '03_review/candidate_evidence_review_structured.json',
    'candidates': '04_analysis_input/analysis_evidence_candidates.json',
    'analysis': '05_analysis/analysis_result.json',
}

def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)

def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else encoded(value).encode()).hexdigest()

def load_file(path):
    # Avoid silently hydrating an offline cloud file while the UI appears frozen.
    if getattr(path.stat(), 'st_flags', 0) & 0x40000000:
        raise ValueError('Source is not downloaded locally')
    if path.stat().st_size > 30_000_000:
        raise ValueError('Source exceeds 30 MB reader limit')
    return path.read_bytes()

def pointer(value, path):
    if not path:
        return value
    if not isinstance(path, str) or not path.startswith('/'):
        raise ValueError('field must be a JSON pointer, for example /statistics')
    for part in path[1:].split('/'):
        part = part.replace('~1', '/').replace('~0', '~')
        if isinstance(value, dict) and part not in value:
            raise ValueError('Unknown field ' + repr(part) + '; available keys here: ' +
                             ', '.join(list(value)[:30]) + '. Use / between JSON pointer segments, not dots.')
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value

def text_block(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return '\n'.join(text_block(v) for v in value)
    if isinstance(value, dict):
        return '\n'.join(text_block(value[k]) for k in
                         ('text', 'table_body', 'table_caption', 'table_footnote', 'content', 'caption') if k in value)
    return ''

class Snapshot:
    def __init__(self, run):
        self.run = Path(run).expanduser().resolve()
        self.raw, self.hashes, self.warnings = {}, {}, []
        for key, relative in FILES.items():
            path = self.run / relative
            try:
                blob = load_file(path)
                self.raw[key] = json.loads(blob)
                self.hashes[relative] = digest(blob)
            except FileNotFoundError:
                self.raw[key] = {}
                self.warnings.append(relative + ': not available')
            except (OSError, ValueError) as exc:
                raise ValueError(f'Cannot load {relative}: {type(exc).__name__}') from None
        if not self.raw['request']:
            raise ValueError('A saved request.normalized.json is required')
        self.case_id = self.raw['request'].get('case_id', self.run.name)
        self.analyses = {}
        for row in self.raw['analysis'].get('analyses', []):
            aid = row.get('analysis', {}).get('analysis_id')
            if aid:
                if aid in self.analyses:
                    raise ValueError('Duplicate analysis ID: ' + aid)
                self.analyses[aid] = row
        profile = self.raw['request'].get('analysis_profile', {})
        for definition in [profile.get('primary_analysis_spec', {}), *profile.get('secondary_outcomes', [])]:
            aid = definition.get('analysis_id')
            if aid and aid not in self.analyses:
                self.analyses[aid] = {'analysis': definition, 'status': 'not_available',
                                      'warnings': ['No saved analysis result; do not infer a result.']}
        self.documents = self.index(self.raw['request'].get('documents', []), 'article_id')
        self.evidence = self.index(self.raw['extraction'].get('predictions', []), 'prediction_id')
        self.candidates = self.index(self.raw['candidates'].get('candidates', []), 'candidate_id')
        self.reviews = self.index(self.raw['review'].get('candidate_reviews', []), 'prediction_id')
        screening = self.raw['screening'] or self.raw['screening_original']
        self.screening = {r.get('candidate_id') or r.get('article_id'): r for r in screening.get('decisions', [])}
        self.sources = {}
        prepared = {r['article_id']: r for r in self.raw['preparation'].get('documents', [])}
        for aid, doc in self.documents.items():
            paths = doc.get('paths', {}); prep = prepared.get(aid, {})
            source = {'article_id': aid, 'title': doc.get('title', aid), 'text': '', 'blocks': [], 'warnings': []}
            for kind, suffix in [('markdown', '.md'), ('content_list', '.json')]:
                value = paths.get(kind) or prep.get(kind)
                if not value:
                    continue
                path = Path(value).expanduser()
                if not path.is_absolute():
                    path = self.run / path
                # Only explicitly declared scholarly-source files, never arbitrary tool paths.
                if path.suffix.lower() not in ({'.md', '.txt'} if kind == 'markdown' else {suffix}):
                    source['warnings'].append(kind + ': unsupported file type'); continue
                try:
                    blob = load_file(path)
                    self.hashes[f'source:{aid}:{kind}'] = digest(blob)
                    if kind == 'markdown':
                        source['text'] = blob.decode('utf-8', errors='replace')
                    else:
                        content = json.loads(blob)
                        if isinstance(content, dict):
                            content = content.get('content_list', content.get('items', []))
                        if not isinstance(content, list):
                            raise ValueError('Content list must be an array')
                        source['blocks'] = [text_block(v) for v in content]
                except (OSError, ValueError):
                    source['warnings'].append(kind + ': unavailable; evidence must not be invented')
            if not source['text'] and not any(source['blocks']):
                source['warnings'].append('No original source text available in this snapshot')
            self.sources[aid] = source
        self.fingerprint = digest({'run': str(self.run), 'hashes': self.hashes})

    @staticmethod
    def index(rows, key):
        result = {}
        for row in rows:
            if row.get(key):
                if row[key] in result:
                    raise ValueError('Duplicate record ID: ' + row[key])
                result[row[key]] = row
        return result

    def info(self):
        from .presentation import analysis_title
        missing_sources = [aid for aid, src in self.sources.items() if not src['text'] and not any(src['blocks'])]
        return {'case_id': self.case_id, 'run_name': self.run.name, 'fingerprint': self.fingerprint,
                'topic': self.raw['request'].get('topic'),
                'warnings': [*self.warnings, *[f'{aid}: original source unavailable' for aid in missing_sources]],
                'documents': len(self.documents), 'extractions': len(self.evidence),
                'sources_available': len(self.sources) - len(missing_sources),
                'analyses': [{'id': aid, 'title': analysis_title(a['analysis']),
                              'status': a.get('status', 'unknown')} for aid, a in self.analyses.items()]}

    def query(self, name, args, analysis_id):
        if analysis_id not in self.analyses:
            raise ValueError('Unknown analysis in this saved run')
        analysis = self.analyses[analysis_id]
        if name == 'get_research_context':
            from .research import research_context
            return research_context(self, analysis_id)
        if name == 'get_analysis':
            return {'origin': FILES['analysis'], 'analysis_id': analysis_id, 'record': analysis}
        if name == 'get_model_comparison':
            robustness = analysis.get('robustness') or {}
            return {'origin': FILES['analysis'], 'analysis_id': analysis_id,
                    'primary_statistics': analysis.get('statistics'),
                    'recorded_method_selection': analysis.get('method_selection'),
                    'saved_pooled_model_comparison': robustness.get('model_comparison'),
                    'caveats': robustness.get('caveats'),
                    'note': 'These are saved pooled-model comparisons, not individual study effects. '
                            'Missing confidence intervals remain unavailable. The saved rationale is a recorded '
                            'model-choice explanation, not independent proof that this model is correct.'}
        if name == 'get_leave_one_out':
            return self.leave_one_out(analysis, args, analysis_id)
        if name == 'get_study_effect':
            effects = (analysis.get('poolability') or {}).get('study_effects') or []
            position, eid = args.get('position'), args.get('evidence_id')
            if position is None and eid is None:
                raise ValueError('Provide evidence_id or one-based position')
            if position is not None and (type(position) is not int or not 1 <= position <= len(effects)):
                raise ValueError(f'position must be between 1 and {len(effects)} in saved pooled-effect order')
            if eid is not None and (not isinstance(eid, str) or not eid.strip()):
                raise ValueError('evidence_id must be non-empty')
            matches = []
            for i, effect in enumerate(effects):
                if (position is not None and position != i + 1) or (eid is not None and effect.get('evidence_id') != eid):
                    continue
                evidence_id = effect.get('evidence_id')
                original = self.evidence.get(evidence_id) or {}
                candidate = self.candidates.get(evidence_id) or {}
                article_id = original.get('article_id') or candidate.get('article_id')
                fields = ('article_id', 'study_id', 'outcome', 'timepoint', 'exposure', 'comparator',
                          'comparison', 'raw_data', 'used_effect', 'effect', 'model_name', 'adjustment')
                def compact(row):
                    location = row.get('source_location') or {}
                    return {**{k: row[k] for k in fields if k in row},
                            'source_location': {k: v for k, v in location.items()
                                                if k not in ('source_evidence', 'source_evidence_file', 'source_quote')}}
                doc = self.documents.get(article_id) or {}
                location = original.get('source_location') or candidate.get('source_location') or {}
                table_hint = re.search(r'\bTable\s+[A-Za-z]?\d+\b', str(location.get('source_quote', '')),
                                       re.IGNORECASE)
                original_passage = None
                if table_hint and article_id in self.sources:
                    passage = self.query('get_source', {'article_id': article_id,
                                                       'query': table_hint.group(), 'line_count': 12}, analysis_id)
                    if passage['query_found']:
                        original_passage = {
                            'status': 'candidate_original_passage_not_a_verdict',
                            'lookup_basis': 'Table label in saved extraction; text read independently from original source.',
                            'note': 'Compare the row, outcome, groups and denominators yourself. '
                                    'The saved source_location may still be unresolved; this lookup does not repair it.',
                            'source': passage}
                matches.append({
                    'position': i + 1, 'effect': effect,
                    'original_table_candidate': original_passage,
                    'source_pointer': f'/poolability/study_effects/{i}',
                    'article': {k: doc[k] for k in ('article_id', 'title', 'authors', 'year', 'doi') if k in doc},
                    'extraction': compact(original), 'analysis_input': compact(candidate),
                    'review': self.reviews.get(evidence_id),
                })
            return {'analysis_id': analysis_id, 'availability': 'found' if matches else 'not_found',
                    'pooled_effect_count': len(effects), 'matches': matches,
                    'origins': [FILES[k] for k in ('analysis', 'extraction', 'candidates', 'request', 'review')],
                    'note': 'Position is one-based saved poolability.study_effects order, not pooled estimate or rank. '
                            'Compact projection omits embedded source_evidence copies; use get_source to verify original text. '
                            'Copied source quotes and artifact paths are omitted. An original_table_candidate, when present, '
                            'is independently read source text with its own line range and hash, not a verified-count verdict. '
                            'Full extraction records remain available via get_extraction.'}
        if name == 'get_analysis_selection':
            return {'origin': FILES['analysis'], 'analysis_id': analysis_id,
                    **{k: analysis.get(k) for k in ('selected_evidence_ids', 'assignments', 'candidate_inventory',
                       'poolability', 'source_resolutions', 'source_resolution_issues', 'review', 'topic_review')},
                    'note': 'Missing reason means not recorded. Selection is not identical to statistical membership.'}
        if name == 'get_extraction':
            eid = args['evidence_id']
            if eid not in self.evidence and eid not in self.candidates:
                raise ValueError('Unknown evidence ID in this run')
            return {'evidence_id': eid, 'analysis_id': analysis_id,
                    'selected_in_current_analysis': eid in analysis.get('selected_evidence_ids', []),
                    'extraction': self.evidence.get(eid), 'analysis_input': self.candidates.get(eid),
                    'review': self.reviews.get(eid), 'origins': [FILES['extraction'], FILES['candidates'], FILES['review']]}
        if name in ('get_screening', 'get_rob', 'get_source'):
            aid = args['article_id']
            if aid not in self.documents:
                raise ValueError('Unknown article in this run')
            if name == 'get_screening':
                return {'article_id': aid, 'decision': self.screening.get(aid),
                        'protocol': self.raw['request'].get('protocol'), 'origin': FILES['screening'] if self.raw['screening'] else FILES['screening_original']}
            if name == 'get_rob':
                target = analysis['analysis'].get('target_id')
                rows = [r for r in self.raw['rob'].get('assessments', []) if r.get('article_id') == aid]
                return {'article_id': aid, 'current_target_id': target, 'assessments': rows,
                        'note': 'Check scope.target_id before applying any assessment to this analysis.', 'origin': FILES['rob']}
            src = self.sources[aid]
            item = args.get('content_item')
            if item is not None:
                if type(item) is not int or item < 0 or item >= len(src['blocks']):
                    raise ValueError('Unknown content-list item; use text search instead')
                return {'article_id': aid, 'content_item': item, 'text': src['blocks'][item],
                        'source_sha256': self.hashes.get(f'source:{aid}:content_list'), 'warnings': src['warnings']}
            lines = src['text'].splitlines(); start = args.get('line', 1)
            if type(start) is not int or start < 1:
                raise ValueError('line must be a positive integer')
            term = args.get('query', '').strip()
            hits = [i + 1 for i, line in enumerate(lines) if term.casefold() in line.casefold()] if term else []
            if term and hits:
                # A table-label search should locate the caption before incidental
                # mentions in prose. Explicit line requests remain unchanged.
                captions = [h for h in hits if re.match(
                    r'^\s*(?:#{1,6}\s*)?(?:\*\*)?' + re.escape(term) + r'(?:\b|[:.])',
                    lines[h - 1], re.IGNORECASE)]
                labelled = [h for h in captions if re.match(
                    r'^\s*(?:#{1,6}\s*)?(?:\*\*)?' + re.escape(term) + r'\s*[:.]\s+\S',
                    lines[h - 1], re.IGNORECASE)]
                ranked = (labelled or captions) if re.fullmatch(r'table\s+\w+', term, re.IGNORECASE) and captions else hits
                start = max(1, next((h for h in ranked if h >= start), ranked[-1]) - 3)
            count = args.get('line_count', 40)
            if type(count) is not int or not 1 <= count <= 40:
                raise ValueError('line_count must be between 1 and 40')
            end = min(len(lines), start + count - 1)
            return {'article_id': aid, 'title': src['title'], 'line_start': start, 'line_end': end,
                    'total_lines': len(lines), 'next_line': end + 1 if end < len(lines) else None,
                    'matching_lines': hits[:50], 'matching_lines_total': len(hits),
                    'matching_previews': [{'line': h, 'text': lines[h - 1][:240]} for h in hits[:12]],
                    'query_found': bool(hits) if term else None,
                    'text': '\n'.join(f'{i + 1}: {lines[i]}' for i in range(start - 1, end)),
                    'source_sha256': self.hashes.get(f'source:{aid}:markdown'), 'warnings': src['warnings']}
        if name == 'search_case_evidence':
            query = args['query'].strip().casefold()
            if not query or len(query) > 200:
                raise ValueError('Use a nonempty query of at most 200 characters')
            terms = re.findall(r'[\w]+', query)
            hits = []
            for eid in dict.fromkeys([*self.evidence, *self.candidates]):
                row = self.evidence.get(eid) or self.candidates[eid]
                value = encoded(row).casefold(); score = sum(t in value for t in terms)
                if score:
                    hits.append({'kind': 'evidence', 'id': eid, 'article_id': row.get('article_id'),
                                 'outcome': row.get('outcome'), 'score': score,
                                 'selected_in_current_analysis': eid in analysis.get('selected_evidence_ids', [])})
            for aid, doc in self.documents.items():
                value = encoded(doc).casefold(); score = sum(t in value for t in terms)
                if score:
                    hits.append({'kind': 'article', 'id': aid, 'title': doc.get('title'), 'score': score})
            # Some frozen records have placeholder titles and no author metadata.
            # Search only bounded source openings, and return provenance rather
            # than guessing that a surname mention establishes article identity.
            name_terms = [t for t in terms if len(t) > 2 and t not in {'the', 'and', 'with', 'from'}]
            if name_terms:
                for aid, source in self.sources.items():
                    opening = (source['text'][:6000] if source['text'] else
                               '\n'.join(source['blocks'][:10])[:6000])
                    folded = opening.casefold()
                    if not all(term in folded for term in name_terms):
                        continue
                    evidence_ids = [eid for eid in dict.fromkeys([*self.evidence, *self.candidates])
                                    if (self.evidence.get(eid) or self.candidates[eid]).get('article_id') == aid]
                    needle = next((term for term in name_terms if not term.isdigit()), name_terms[0])
                    index = folded.find(needle)
                    hits.append({'kind': 'source_opening', 'id': aid, 'article_id': aid,
                                 'title': source['title'], 'score': 5 * len(name_terms),
                                 'excerpt': opening[max(0, index-100):index+400],
                                 'evidence_ids': evidence_ids,
                                 'selected_evidence_ids': [eid for eid in evidence_ids
                                                          if eid in analysis.get('selected_evidence_ids', [])],
                                 'note': 'Opening-text match only; verify author/title in get_source, not a reference-list mention.'})
            hits.sort(key=lambda r: -r['score'])
            return {'query': query, 'scope': 'current case metadata, extraction and bounded source openings; selection tags refer to current analysis',
                    'total': len(hits), 'matches': hits}
        raise ValueError('Tool is not allowlisted')

    def leave_one_out(self, analysis, args, analysis_id):
        identifiers = {key: args[key] for key in ('evidence_id', 'study_id') if key in args}
        if not identifiers or any(not isinstance(v, str) or not v.strip() for v in identifiers.values()):
            raise ValueError('Provide a non-empty evidence_id or study_id')
        robustness = analysis.get('robustness') or {}
        rows = robustness.get('leave_one_out')
        valid_rows = isinstance(rows, list) and all(isinstance(row, dict) for row in rows)
        # Reject a known ID placed in the wrong field instead of presenting
        # a misleading absence. Never silently substitute a different record.
        if valid_rows and len(identifiers) == 1:
            supplied_key, supplied_value = next(iter(identifiers.items()))
            other_key = 'study_id' if supplied_key == 'evidence_id' else 'evidence_id'
            if (not any(row.get(supplied_key) == supplied_value for row in rows)
                    and any(row.get(other_key) == supplied_value for row in rows)):
                raise ValueError(f'Identifier is a saved {other_key}, not {supplied_key}. '
                                 f'Retry get_leave_one_out with {other_key}={supplied_value!r}. '
                                 'Do not use the individual study effect as an omission result.')
        context = {key: robustness.get(key) for key in (
            'status', 'tool', 'k', 'effect_type', 'primary_model', 'methods_run', 'caveats', 'error')}
        matches = []
        for index, row in enumerate(rows if valid_rows else []):
            if not all(row.get(key) == value for key, value in identifiers.items()):
                continue
            display = {key: row.get(key) for key in ('pooled_effect', 'ci_lower', 'ci_upper',
                                                    'i_squared', 'tau_squared')}
            converted = []
            if robustness.get('effect_type') in ('RR', 'OR', 'HR', 'IRR'):
                for target, source in (('pooled_effect', 'pooled_log_effect'),
                                       ('ci_lower', 'ci_lower_log'), ('ci_upper', 'ci_upper_log')):
                    value = row.get(source)
                    if display[target] is None and type(value) in (int, float) and math.isfinite(value):
                        try:
                            display[target] = math.exp(value)
                            converted.append(source)
                        except OverflowError:
                            pass
            effect_type = robustness.get('effect_type')
            null_value = (1 if effect_type in ('RR', 'OR', 'HR', 'IRR')
                          else 0 if effect_type in ('MD', 'SMD', 'RD') else None)
            lower, upper = display['ci_lower'], display['ci_upper']
            valid_bounds = (null_value is not None
                            and all(type(v) in (int, float) and math.isfinite(v) for v in (lower, upper))
                            and lower <= upper)
            ci_interpretation = {
                'null_value': null_value,
                'contains_null': lower <= null_value <= upper if valid_bounds else None,
                'position_relative_to_null': ('includes_null' if lower <= null_value <= upper
                                              else 'above_null' if lower > null_value else 'below_null')
                                             if valid_bounds else 'unknown',
                'basis': 'Deterministic inclusive comparison of displayed saved bounds; limited by saved precision. No synthesis rerun.',
            }
            k = robustness.get('k')
            eid = row.get('evidence_id')
            source_record = self.evidence.get(eid) or self.candidates.get(eid) or {}
            article_id = source_record.get('article_id')
            article = self.documents.get(article_id) or {}
            matches.append({
                'source_pointer': f'/robustness/leave_one_out/{index}',
                'record': row,
                'omitted_article': {key: article[key] for key in ('article_id', 'title', 'authors', 'year')
                                    if key in article},
                'display': {'effect_type': robustness.get('effect_type'), **display},
                'ci_interpretation': ci_interpretation,
                'i_squared_unit': 'percent',
                'display_transform': ('exp of saved log fields: ' + ', '.join(converted) +
                                      '; approximate to saved precision; no synthesis rerun.')
                                     if converted else 'Saved original-scale fields only; missing values remain null.',
                'remaining_effect_count': k - 1 if type(k) is int and k > 0 else None,
                'count_basis': 'Derived from saved robustness.k minus one omitted effect; not a count of independent cohorts.',
            })
        return {
            'origin': FILES['analysis'], 'analysis_id': analysis_id, 'requested': identifiers,
            'availability': 'found' if matches else ('not_found' if valid_rows and rows else 'unavailable'),
            'saved_record_count': len(rows) if valid_rows else None,
            'diagnostic_context': context, 'matches': matches,
            'note': 'Exact identifier match in this analysis only; both IDs, if supplied, must match the same record. '
                    'The primary_model label does not establish the diagnostic estimator or CI method. '
                    'These are saved diagnostics, not a newly executed exclusion or a replacement primary analysis.',
        }

    def page(self, name, args, analysis_id):
        value = self.query(name, args, analysis_id)
        if args.get('field'):
            value = pointer(value, args['field'])
        else:
            value = self.compact(name, value)
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
        offset = args.get('offset', 0)
        if type(offset) is not int or offset < 0 or offset > len(text):
            raise ValueError('Invalid page offset')
        end = min(offset + 8000, len(text))
        return {'tool': name, 'arguments': args, 'analysis_id': analysis_id,
                'snapshot': self.fingerprint, 'text': text[offset:end], 'offset': offset,
                'total_chars': len(text), 'next_offset': end if end < len(text) else None,
                'note': 'Partial page; call again with next_offset before claiming completeness.' if end < len(text) else ''}

    @staticmethod
    def compact(name, value):
        """Default views expose decision-relevant fields without copied full texts.

        Explicit JSON pointers always address the untouched raw query result.
        """
        if name == 'get_analysis':
            row = value['record']
            definition = row.get('analysis') or {}
            robustness = row.get('robustness') or {}
            return {
                **value,
                'record': {
                    'analysis': {k: definition[k] for k in (
                        'analysis_id', 'title', 'target_id', 'outcome_group', 'exposure_terms',
                        'comparison_terms', 'population_terms', 'timepoint_policy', 'selection_rules') if k in definition},
                    'status': row.get('status'), 'statistics': row.get('statistics'),
                    'method_selection': row.get('method_selection'), 'warnings': row.get('warnings'),
                    'robustness': {k: robustness.get(k) for k in (
                        'status', 'primary_model', 'model_comparison', 'heterogeneity_drivers',
                        'influential_studies', 'stability', 'caveats')},
                },
                'projection': 'Compact overview. Full definitions, assignments, candidate inventory, source resolutions, reporting bias and leave-one-out rows omitted.',
                'diagnostic_units_note': 'i_squared is a percentage. heterogeneity_contribution is the saved diagnostic contribution score, not a percentage or a significance test.',
                'full_record_fields': ['/record/analysis', '/record/candidate_inventory', '/record/assignments',
                                       '/record/source_resolutions', '/record/reporting_bias', '/record/robustness'],
            }
        if name == 'get_analysis_selection':
            assignments = value.get('assignments') or []
            counts = Counter((row.get('role'), row.get('reason') or row.get('selection_reason'))
                             for row in assignments)
            inventory = value.get('candidate_inventory') or {}
            effects = (value.get('poolability') or {}).get('study_effects') or []
            return {
                'origin': value['origin'], 'analysis_id': value['analysis_id'],
                'candidate_inventory': {k: inventory.get(k) for k in ('candidate_record_count', 'candidate_study_count')},
                'selected_evidence_ids': value.get('selected_evidence_ids'),
                'pooled_effect_count': len(effects),
                'pooled_effect_index': [{k: row.get(k) for k in ('study_id', 'evidence_id', 'effect_type', 'value')} for row in effects],
                'assignment_count': len(assignments),
                'recorded_reason_groups': [{'role': role, 'reason': reason, 'count': count}
                                           for (role, reason), count in counts.items()],
                'projection': 'Compact selection summary. Full assignments, candidate inventory, poolability details and review records omitted. Generic reasons do not explain every exclusion.',
                'full_record_fields': ['/assignments', '/candidate_inventory', '/poolability', '/review',
                                       '/topic_review', '/source_resolutions', '/source_resolution_issues'],
            }
        if name == 'get_extraction':
            def strip_copies(item):
                if isinstance(item, dict):
                    return {k: strip_copies(v) for k, v in item.items() if k != 'source_evidence'}
                if isinstance(item, list):
                    return [strip_copies(v) for v in item]
                return item
            return {**strip_copies(value), 'projection': 'Embedded source_evidence copies omitted. Use get_source for original text.',
                    'full_record_fields': ['/extraction', '/analysis_input', '/review']}
        if name == 'get_rob':
            assessments = []
            for row in value['assessments']:
                compact = {k: row.get(k) for k in ('article_id', 'study_id', 'scope', 'tool', 'status',
                                                  'provenance', 'overall_judgment', 'overall_rationale', 'validation')}
                compact['review_state_note'] = (
                    'This record is a draft; it does not establish human confirmation. '
                    'Do not reinterpret an external-review requirement as completed review.'
                    if row.get('status') == 'draft' else
                    'Report the saved status; do not infer human confirmation from a risk judgment alone.')
                compact['domains'] = [{k: domain.get(k) for k in ('domain_id', 'judgment', 'rationale')}
                                      for domain in row.get('domains', [])]
                assessments.append(compact)
            return {**value, 'assessments': assessments,
                    'projection': 'Compact assessment. Full domain evidence, signalling questions and routing details omitted.',
                    'full_record_fields': ['/assessments']}
        return value
