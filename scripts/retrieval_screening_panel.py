"""Optional retrieval-to-screening provenance panel for native run dashboards."""
from collections import Counter
from html import escape
import json
from pathlib import Path
import re


def read(path, default):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def context_for(run):
    run = Path(run)
    context = read(run / 'retrieval_screening_context.json', {})
    if not context:
        return {}
    summary = read(run / '05_analysis/analysis_summary.json', {})
    selected = {pid for a in summary.get('analyses', []) if a.get('kind') == 'primary'
                for pid in a.get('selected_candidate_ids', [])}
    candidates = read(run / '04_analysis_input/analysis_evidence_candidates.json', {}).get('candidates', [])
    selected_articles = {x.get('article_id') for x in candidates if x.get('candidate_id') in selected}
    gate = read(run / 'quantitative_gate_audit.json', {})
    extraction = read(run / '02_extraction/case_run_manifest.json', {})
    article_runs = {x['article_id']: x for x in extraction.get('article_runs', [])}
    predictions = read(run / '02_extraction/case_extraction_submission.json', {}).get('predictions', [])
    n_pred = Counter(x['article_id'] for x in predictions)
    cohort = []
    finished = summary.get('status') == 'completed'
    for row in context['current_cohort']:
        aid = row['native_article_id']
        held_reason = gate.get('article_gates', {}).get(aid)
        status = ('pooled' if aid in selected_articles else 'held' if held_reason else
                  'not_selected' if finished else 'processing')
        cohort.append({'corpus_id': row['corpus_id'], 'article_id': aid, 'title': row['title'], 'doi': row['doi'],
                       'baseline_match': row['baseline_registry_match'], 'refresh_reason': row.get('refresh_reason'),
                       'extraction_status': article_runs.get(aid, {}).get('status', 'pending'),
                       'extracted_records': n_pred[aid], 'quantitative_status': status,
                       'reason': held_reason or row.get('quantitative_gate') or (row.get('source_audit') or {}).get('next_action', '')})
    follows = {x['corpus_id']: x for x in context['followup_rows']}
    rows = []
    for row in context['pool_rows']:
        follow = follows.get(row['corpus_id'])
        final = row['final_status']
        availability = ('available_pending' if follow and final == 'Unsure' and follow.get('fulltext_status') == 'screened'
                        else 'not_retrieved' if follow and final == 'Unsure'
                        else 'available' if follow and follow.get('fulltext_status') == 'screened'
                        else 'not_in_followup')
        reason = (follow or row).get('fulltext_reason') or row.get('reason') or row.get('abstract_reason') or ''
        if follow and follow.get('source_audit'):
            reason = follow['source_audit'].get('finding', '') + ' ' + reason
        rows.append({'corpus_id': row['corpus_id'], 'title': row.get('title'), 'doi': row.get('doi'),
                     'status': final, 'availability': availability, 'reason': reason,
                     'review_disposition': (follow or row).get('review_disposition', '')})
    keys = ('case_id', 'candidate_count', 'pool_counts', 'title_abstract_counts', 'fulltext_followup_counts',
            'fulltext_followup_total', 'fulltext_matched', 'not_retrieved', 'available_pending', 'time_window', 'scope_note')
    return {**{key: context[key] for key in keys}, 'cohort': cohort, 'rows': rows,
            'comparability_audit': read(run / 'comparability_audit.json', {}),
            'cost': read(run / 'update_cost_summary.json', {}),
            'quantitative_counts': dict(Counter(x['quantitative_status'] for x in cohort))}


def panel(data, language='en'):
    if not data:
        return ''
    def e(value):
        return escape(str(value if value is not None else '—'))
    def counts(values):
        return ' · '.join(f'{e(k)} {e(v)}' for k, v in values.items())
    availability = {'available_pending': 'Full text available · held',
                    'not_retrieved': 'No valid full text',
                    'available': 'Full text paired',
                    'not_in_followup': 'Outside this fulltext follow-up'}
    qlabels = {'pooled': 'Pooled', 'held': 'Quantitative hold',
               'not_selected': 'Not selected', 'processing': 'Processing'}
    head = '<div class="panel" id="retrieval-screening-panel"><h2>Retrieval → abstract → fulltext screening</h2>'
    cards = [
        ('Deduplicated candidate pool', data['candidate_count'], 'Europe PMC + PubMed; reports'),
        ('Title/abstract screening', counts(data['title_abstract_counts']), e(data.get('abstract_screening_label') or 'Protocol title/abstract screening')),
        ('Unsure fulltext follow-up', f"{data['fulltext_matched']} / {data['fulltext_followup_total']}", 'Processed full texts / original Unsure'),
        ('Current working decisions', counts(data['pool_counts']), 'Includes user review and report deduplication'),
    ]
    head += '<div class="grid">' + ''.join(f'<div class="card"><div class="label">{a}</div><div class="value">{b}</div><div>{c}</div></div>' for a, b, c in cards) + '</div>'
    win = data['time_window']
    head += f'<p>Publication window: {e(win["start"])} → {e(win["end"])}.</p>'
    head += f'<div class="note">Current Unsure: {data["not_retrieved"]} without valid full text + {data["available_pending"]} with full text held for verification. Include denotes report-level screening eligibility.</div>'
    head += f'<p>Working fulltext follow-up decisions: {counts(data["fulltext_followup_counts"])}</p>'
    overview = head
    head = f'<details id="reviewed-update-cohort"><summary>Reviewed fulltext Includes and update disposition ({len(data["cohort"])})</summary>'
    head += '<p>This update processes the retained fulltext follow-ups and preserves historical update evidence. Other abstract Includes have not all undergone this fulltext audit or extraction.</p>'
    headers = ['ID / DOI', 'Title', 'Historical / new', 'A2 extraction', 'Primary analysis', 'Verification note']
    head += '<div class="table-wrap"><table><thead><tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr></thead><tbody>'
    for row in data['cohort']:
        basis = 'Previously excluded; newly extracted' if row['refresh_reason'] else 'Historical reuse' if row['baseline_match'] else 'New report'
        head += '<tr>' + ''.join(f'<td>{x}</td>' for x in [f'#{row["corpus_id"]}<br>{e(row["doi"])}', e(row['title']), basis,
                   f'{e(row["extraction_status"])} · {row["extracted_records"]}', qlabels[row['quantitative_status']], e(row['reason'])]) + '</tr>'
    cohort = head + '</tbody></table></div></details>'
    head = overview + f'<div id="retrieval-pool-section"><h3>Full retrieval pool and screening reasons · {len(data["rows"])} reports</h3>'
    head += '<p>All retrieved records are listed below. Scroll within the table or filter by title, DOI, ID, reason and current decision.</p>'
    search_label = 'Search title, DOI, ID or reason'
    head += f'<p style="display:flex;gap:10px;align-items:center;flex-wrap:wrap"><input id="retrieval-search" class="filter" style="flex:1;min-width:240px;margin:0" aria-label="{search_label}" placeholder="{search_label}"><select id="retrieval-decision" aria-label="Current screening decision"><option value="">All decisions</option>'
    head += ''.join(f'<option value="{x}">{x}</option>' for x in ('Include', 'Exclude', 'Unsure', 'Duplicate'))
    head += f'</select><button id="retrieval-clear" type="button">Clear filters</button><span id="retrieval-visible-count" aria-live="polite">Showing {len(data["rows"])} / {len(data["rows"])} reports</span></p><div class="table-wrap"><table id="retrieval-pool-table"><thead><tr>'
    head += ''.join(f'<th>{h}</th>' for h in ['ID', 'Title / DOI', 'Working decision', 'Fulltext status', 'Reason']) + '</tr></thead><tbody>'
    for row in data['rows']:
        head += f'<tr data-corpus-id="{row["corpus_id"]}" data-decision="{e(row["status"])}"><td>{row["corpus_id"]}</td><td>{e(row["title"])}<br><code>{e(row["doi"])}</code></td><td>{e(row["status"])}</td><td>{availability[row["availability"]]}</td><td>{e(row["reason"])}</td></tr>'
    head += '</tbody></table></div><p id="retrieval-empty" hidden>No records match the current filters.</p></div>' + cohort
    head += '<p><a href="retrieval_screening_context.json">Full screening snapshot</a> · <a href="reviewed_cohort.json">Update cohort</a></p></div>'
    return head


def inject(html, data, language='en'):
    body = panel(data, language)
    if not body:
        return html
    def remove_previous(match):
        attrs, script = match.group(1), match.group(2).lstrip()
        if any(marker in attrs for marker in ('retrieval-screening-script', 'retrieval-overview-script')):
            return ''
        if script.startswith('document.getElementById("screening").insertAdjacentHTML("afterbegin",') and 'retrieval-screening-panel' in script:
            return ''
        if script.startswith('document.getElementById("overview").insertAdjacentHTML("afterbegin",') and 'Retrieval pool:' in script:
            return ''
        return match.group(0)
    html = re.sub(r'<script([^>]*)>(.*?)</script>', remove_previous, html, flags=re.S)
    encoded = json.dumps(body, ensure_ascii=False).replace('</', '<\\/')
    script = '<script id="retrieval-screening-script">document.getElementById("screening").insertAdjacentHTML("afterbegin",' + encoded + ');'
    labels = {'showing': 'Showing', 'reports': 'reports',
              'pool': 'Retrieved reports (deduplicated)',
              'decisions': 'Pool: Included / Excluded',
              'update': 'Documents processed by update',
              'sources': 'Sources of update documents'}
    script += 'const retrievalLabels=' + json.dumps(labels, ensure_ascii=False) + ';'
    script += '''
const retrievalRows=Array.from(document.querySelectorAll('#retrieval-pool-table tbody tr'));
const retrievalSearch=document.getElementById('retrieval-search'),retrievalDecision=document.getElementById('retrieval-decision');
const retrievalFilter=()=>{const q=retrievalSearch.value.trim().toLowerCase(),status=retrievalDecision.value;let shown=0;retrievalRows.forEach(row=>{row.hidden=!!((status&&row.dataset.decision!==status)||!row.textContent.toLowerCase().includes(q));if(!row.hidden)shown++;});document.getElementById('retrieval-visible-count').textContent=`${retrievalLabels.showing} ${shown} / ${retrievalRows.length} ${retrievalLabels.reports}`;document.getElementById('retrieval-empty').hidden=shown!==0;};
retrievalSearch.addEventListener('input',retrievalFilter);retrievalDecision.addEventListener('change',retrievalFilter);
document.getElementById('retrieval-clear').addEventListener('click',()=>{retrievalSearch.value='';retrievalDecision.value='';retrievalFilter();});
retrievalFilter();
const retrievalCards=document.querySelectorAll('#app > .grid > .card');
if(retrievalCards.length>=2){retrievalCards[0].querySelector('.label').textContent=retrievalLabels.pool;retrievalCards[0].querySelector('.value').textContent=retrievalRows.length.toLocaleString('en-US');retrievalCards[1].querySelector('.label').textContent=retrievalLabels.decisions;retrievalCards[1].querySelector('.value').textContent=['Include','Exclude'].map(status=>retrievalRows.filter(row=>row.dataset.decision===status).length.toLocaleString('en-US')).join(' / ');}
const updateHeading=document.getElementById('screen-table')?.closest('.panel')?.querySelector('h2');if(updateHeading)updateHeading.textContent=`${retrievalLabels.update} · ${d.screening.total}`;
const updateSourceHeading=document.getElementById('source-table')?.closest('.panel')?.querySelector('h2');if(updateSourceHeading)updateSourceHeading.textContent=retrievalLabels.sources;
</script>'''
    intro = f'<div class="note">This case has {data["candidate_count"]} retrieved reports. The full list is shown first under "Screening &amp; sources"; documents processed by update are listed separately below. <a href="#screening" data-open-retrieval-pool>View all {data["candidate_count"]} screening records →</a></div>'
    audit = data.get('comparability_audit') or {}
    if audit:
        fixed = audit['statistics']
        corrected = audit.get('source_reviewed_statistics')
        intro += '<div class="note"><strong>Analysis comparability check</strong><p>'
        english_note = f'Historical scope recalculated: RR {fixed["pooled_effect"]:.3f} [{fixed["ci_lower"]:.3f}, {fixed["ci_upper"]:.3f}], k={fixed["k"]}. The reviewed update retains verified historical selections and method alongside newly verified studies.'
        if corrected:
            english_note += f' Final RR {corrected["pooled_effect"]:.3f} [{corrected["ci_lower"]:.3f}, {corrected["ci_upper"]:.3f}], k={corrected["k"]}.'
        intro += english_note + '</p><a href="comparability_audit.json">Native recalculation audit</a></div>'
    cost = data.get('cost') or {}
    if cost:
        intro += f'<p>Estimated update API cost: ${cost["totals"]["estimated_usd"]:.4f} · <a href="update_cost_summary.json">Usage and pricing</a></p>'
    encoded_intro = json.dumps(intro, ensure_ascii=False).replace('</', '<\\/')
    script += '<script id="retrieval-overview-script">document.getElementById("overview").insertAdjacentHTML("afterbegin",' + encoded_intro + ');'
    script += '''const openRetrievalPool=()=>{document.querySelector('[role="tab"][aria-controls="screening"]')?.click();document.getElementById('retrieval-pool-section')?.scrollIntoView({block:'start'});};document.querySelectorAll('[data-open-retrieval-pool]').forEach(link=>link.addEventListener('click',()=>{openRetrievalPool();}));if(['#screening','#retrieval-pool-section'].includes(location.hash))openRetrievalPool();</script>'''
    return html.replace('</body>', script + '</body>')
