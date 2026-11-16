"""Human-readable views of retrieved records; raw data remains available on demand."""
import html
import json
import re

def analysis_title(definition):
    title = str(definition.get('title') or '').strip()
    aid = str(definition.get('analysis_id') or '')
    if title and title != aid and not re.fullmatch(r'[a-z0-9]+(?:_[a-z0-9]+)+', title):
        return title
    terms = definition.get('outcome_terms') or []
    outcome = definition.get('outcome_group') or (terms[0] if terms else '')
    label = str(outcome or title or 'Analysis results').replace('_', ' ').strip()
    return label[:1].upper() + label[1:]

NAMES = {'get_analysis':'Analysis results', 'get_study_effect':'Study effect and source', 'get_leave_one_out':'Saved leave-one-out results', 'get_analysis_selection':'Study selection rationale', 'get_extraction':'Extraction and review records',
         'get_screening':'Screening rationale', 'get_rob':'Risk of bias', 'get_source':'Source evidence', 'search_case_evidence':'Related evidence'}
LABELS = {'outcome':'Outcome', 'exposure':'Exposure / intervention', 'comparison':'Comparator', 'comparator':'Comparator',
          'population_or_subgroup':'Population', 'timepoint':'Timepoint', 'estimand':'Estimand', 'raw_data':'Extracted values',
          'source_quote':'Source quotation', 'reason':'Reason', 'rationale':'Assessment rationale', 'effect_type':'Effect measure',
          'model':'Statistical model', 'k':'Study count', 'pooled_effect':'Pooled effect', 'ci_lower':'Lower confidence limit',
          'ci_upper':'Upper confidence limit', 'i_squared':'I²', 'status':'Status', 'overall_judgment':'Overall risk of bias',
          'overall_rationale':'Overall rationale', 'final_status':'Eligibility decision', 'decision':'Decision',
          'true_positive':'True positive', 'false_positive':'False positive', 'true_negative':'True negative', 'false_negative':'False negative',
          'value':'Value', 'effect':'Effect', 'n':'Sample size', 'mean':'Mean', 'sd':'Standard deviation', 'label':'Label'}

def value_html(value):
    if isinstance(value, dict):
        return '<dl class="record-fields">' + ''.join('<dt>'+html.escape(LABELS.get(k, k.replace('_',' ')))+'</dt><dd>'+value_html(v)+'</dd>' for k,v in value.items()) + '</dl>'
    if isinstance(value, list):
        return '<ul>' + ''.join('<li>'+value_html(v)+'</li>' for v in value) + '</ul>'
    return html.escape('Not recorded' if value is None else str(value))

def citation_html(page, snapshot, prefix='/'):
    tool, args = page['tool'], page.get('arguments', {})
    article = args.get('article_id') or (snapshot.evidence.get(args.get('evidence_id'), {}) or
                                       snapshot.candidates.get(args.get('evidence_id'), {})).get('article_id')
    title = (snapshot.documents.get(article) or {}).get('title') or 'Current analysis'
    name = NAMES.get(tool, 'Evidence record')
    try:
        data = json.loads(page['text'])
    except (ValueError, TypeError):
        data = None
    content = ''
    if tool == 'get_source' and isinstance(data, dict):
        location = ('Source lines '+str(data['line_start'])+'–'+str(data.get('line_end'))+'') if 'line_start' in data else 'Source item '+str(data.get('content_item', 0) + 1)
        content = '<p class="source-location">'+html.escape(location)+'</p><blockquote>'+html.escape(data.get('text',''))+'</blockquote>'
    elif tool == 'get_extraction' and isinstance(data, dict):
        row = data.get('extraction') or data.get('analysis_input') or {}
        fields = {k:row[k] for k in ('outcome','exposure','comparison','comparator','population_or_subgroup','timepoint','estimand','raw_data') if k in row}
        content = value_html(fields)
        quote = (row.get('source_location') or {}).get('source_quote')
        if quote:
            content += '<h2>Source quotation saved with the extraction</h2><blockquote>'+html.escape(quote)+'</blockquote><p class="muted">This quotation was saved during extraction. Check the source evidence to verify its accuracy.</p>'
    elif tool == 'get_study_effect' and isinstance(data, dict):
        content = value_html({k: data.get(k) for k in ('availability', 'matches', 'note')})
    elif tool == 'get_leave_one_out' and isinstance(data, dict):
        content = value_html({k: data.get(k) for k in ('availability', 'diagnostic_context', 'matches', 'note')})
    elif tool == 'get_analysis' and isinstance(data, dict):
        row = data.get('record') or data
        content = value_html(row.get('statistics') or {'status': row.get('status', 'Not recorded')})
    elif isinstance(data, dict):
        content = '<p>This record contains '+html.escape(name)+'. Expand the record below to inspect the content used in this answer.</p>'
    if not content:
        content = '<p>This retrieval contains a record excerpt. See the full retrieved content below.</p>'
    partial = '<p class="notice">Only part of this record was retrieved. This does not establish that all material was reviewed.</p>' if page.get('next_offset') is not None or page.get('offset',0)>0 else ''
    metadata = {k:v for k,v in page.items() if k!='text'}
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>'+html.escape(name)+' · TRACE</title><link rel="stylesheet" href="'+prefix+'style.css">'
            '<main class="source-page"><a class="back-link" href="'+prefix+'">← Back to results</a><p class="eyebrow">TRACE / EVIDENCE</p>'
            '<h1>'+html.escape(name)+'</h1><p class="source-title">'+html.escape(title)+'</p>'+partial+content+
            '<details><summary>Full retrieved content</summary><pre>'+html.escape(page['text'])+'</pre></details>'
            '<details class="technical"><summary>Record identifiers and technical details</summary><pre>'+html.escape(json.dumps(metadata,ensure_ascii=False,indent=2))+'</pre></details></main></html>')
