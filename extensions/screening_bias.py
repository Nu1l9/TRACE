"""Screening-time article/target RoB, with deterministic downstream mapping."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extensions.common import ROOT, read, write, require, digest
from extensions.bias import SCHEMA, ROBINS_I_V2_VERSION, ROUTE_DEFAULT_VERSIONS, QUIPS_ITEM_COUNTS, assess_request, validate_assessments, routed_tool, unassessed_record

RESOURCE_ROOT = ROOT / 'resources/rob'
FULL_GUIDES = {
    # The 2013 development paper remains in resources/rob/text for provenance,
    # but it is not an assessment instrument.  Sending its 974-line full text
    # with every article wastes context without adding a scoring criterion.
    'QUIPS': ['quips_rubric.txt'],
    'RoB 2': ['text/RoB 2/03_RoB2_cribsheet_2019.txt'],
    'ROBINS-I': ['text/ROBINS-I V2/ROBINS-I_V2_cribsheet_2025-11-20.txt'],
    'ROBINS-E': ['text/ROBINS-E/03_ROBINS-E_score_sheet_template_24March2024.txt'],
    'QUADAS-3': ['text/QUADAS 3/02_QUADAS3_latest_version_1.2_score_sheet.txt'],
    'PROBAST+AI': ['text/PROBAST/05_PROBAST_AI_latest_score_sheet_table3.txt'],
}
ROBINS_I_2016_GUIDE = ['text/ROBINS-I/04_ROBINS-I_score_sheet_template_2016.txt']
# A distinct mapping preserves the pre-compaction input fingerprint for safe
# recovery of an interrupted run; only `GUIDES` opts into external compact checklists.
ARCHIVED_FULL_GUIDES = dict(FULL_GUIDES)
GUIDES = FULL_GUIDES
LEGACY_QUIPS_GUIDES = {
    **FULL_GUIDES,
    'QUIPS': ['text/QUIPS/01_QUIPS_development_paper_2013_source.txt', 'quips_rubric.txt'],
}
# The layout used immediately before the ROBINS-I V2 upgrade.  It is retained
# solely to recover interrupted 2016 assessments whose actual model payload
# remains unchanged; it never permits an old assessment to masquerade as Structured.
PRE_V2_GUIDES = {
    **FULL_GUIDES,
    'ROBINS-I': ['text/ROBINS-I/04_ROBINS-I_score_sheet_template_2016.txt'],
}
# Tool wording is loaded from separately installed, hash-verified resources.
# Keeping the exact prompt bytes preserves historical model input fingerprints.
from extensions.rob_resources import load_resource_text, load_compact_guidance


def normalize_assessment_envelope(response):
    """Accept one unambiguous assessment without changing its judgments."""
    require(isinstance(response, dict), 'RoB response must be an object')
    candidates=[]
    if isinstance(response.get('domains'), list) and response['domains']:
        candidates.append(('root',response))
    for key in ('assessment_template','assessment'):
        nested=response.get(key)
        if isinstance(nested,dict) and isinstance(nested.get('domains'),list) and nested['domains']:
            candidates.append((key,nested))
    require(len(candidates)<=1,'RoB response contains ambiguous assessment envelopes')
    if not candidates: return copy.deepcopy(response)
    key,assessment=candidates[0]
    result=copy.deepcopy(assessment)
    if key!='root': result['provider_response_envelope']=key
    return result


def deduplicate_identical_domains(response):
    """Remove only byte-for-byte repeated model domain objects before validation.

    A repeated identical object is a serialisation artefact and carries no
    additional assessment information.  Conflicting duplicate domain IDs are
    deliberately retained so the contract validator stops the run for review.
    The original provider response remains separately preserved on disk.
    """
    domains=response.get('domains') if isinstance(response,dict) else None
    if not isinstance(domains,list):
        return response
    kept=[]; canonical_by_id={}; removed=[]
    for index,domain in enumerate(domains):
        if not isinstance(domain,dict) or not isinstance(domain.get('domain_id'),str):
            kept.append(domain)
            continue
        domain_id=domain['domain_id']
        canonical=json.dumps(domain,ensure_ascii=False,sort_keys=True,separators=(',',':'))
        previous=canonical_by_id.get(domain_id)
        if previous is None:
            canonical_by_id[domain_id]=canonical;kept.append(domain)
        elif previous == canonical:
            removed.append({'domain_id':domain_id,'provider_domain_index':index})
        else:
            kept.append(domain)
    if removed:
        response=copy.deepcopy(response)
        response['domains']=kept
        response['provider_duplicate_domains_removed']=removed
    return response


def _robins_i_v2_scope_supported(row):
    """Require source-supported follow-up/cohort scope before upgrading the tool."""
    evidence = row.get('evidence') or []
    if not isinstance(evidence, list):
        return False
    for item in evidence:
        if not isinstance(item, dict) or item.get('claim') != 'study_design':
            continue
        quote = str(item.get('quote') or '').casefold()
        if 'cohort' in quote or 'follow-up' in quote or 'follow up' in quote:
            return True
    return False


def _resolve_effective_route(row):
    """Resolve one router row without making unrelated articles fail.

    The frozen router may deliberately return no tool for study designs outside
    the registered instruments.  That is an evidence-routing outcome, not a
    batch-level contract failure: callers that can operate partially must be
    able to assess the remaining articles.
    """
    row = copy.deepcopy(row)
    if not row.get('recommended_tool'):
        agent_tool=row.get('agent_recommended_tool')
        # MINERS may suppress an otherwise explicit mapping when its page
        # citation fails internal validation.  Retain only the one mapping for
        # which the classified question type itself establishes compatibility.
        compatible=(agent_tool=='QUIPS' and row.get('question_type')=='prognostic_factor')
        if not compatible:
            return None, {
                'reason': 'no_defensible_rob_tool_route',
                'message': 'the router did not select a supported RoB instrument for this study design',
                'routing_recommendation': row,
            }
        row['recommended_tool']=agent_tool
        row['tool_version']=ROUTE_DEFAULT_VERSIONS[agent_tool]
        row['tool_variant']=''
        row['route_resolution']='accepted_agent_tool_for_matching_question_type'
        row['integrated_route_warning_flags']=copy.deepcopy(row.get('validation_flags',[]))
    if row.get('recommended_tool') not in GUIDES:
        return None, {
            'reason': 'no_registered_rob_guide_for_route',
            'message': 'the routed instrument has no registered assessment guide in this release',
            'routing_recommendation': row,
        }
    row.setdefault('tool_version',ROUTE_DEFAULT_VERSIONS.get(row['recommended_tool']))
    if row['recommended_tool'] == 'ROBINS-I':
        # The official Structured document presently covers follow-up/cohort studies.
        # The frozen router's `nonrandomized_intervention` label is broader, so
        # only upgrade when its source-quoted design evidence establishes Structured.
        if _robins_i_v2_scope_supported(row):
            row['tool_version'] = ROBINS_I_V2_VERSION
            row['tool_variant'] = 'follow_up_cohort'
            row['route_resolution'] = 'upgraded_to_robins_i_v2_from_source_supported_cohort_design'
        else:
            row['tool_version'] = '2016'
            row['tool_variant'] = ''
            row['route_resolution'] = 'retained_robins_i_2016_no_source_supported_follow_up_cohort_scope'
    return row, None


def partition_effective_routes(request, recommendations, included):
    """Split included articles into routable and manual-methodology groups.

    Each group remains attributable to the original router row.  In particular,
    an uncovered cross-sectional comparison must not prevent a separate cohort
    article from receiving its applicable ROBINS-E assessment.
    """
    require(isinstance(recommendations,dict) and isinstance(recommendations.get('recommendations'),list),
            'RoB recommendations must contain an array')
    by_article={row.get('article_id'):copy.deepcopy(row) for row in recommendations['recommendations']}
    require(None not in by_article and len(by_article)==len(recommendations['recommendations']),
            'RoB recommendations contain missing or duplicate article IDs')
    rows=[]; uncovered=[]
    for article_id in included:
        if article_id not in by_article:
            uncovered.append({'article_id':article_id, 'reason':'missing_rob_routing_recommendation',
                              'message':'the screening router produced no RoB recommendation for this included article',
                              'routing_recommendation':{}})
            continue
        row, issue = _resolve_effective_route(by_article[article_id])
        if issue is None:
            rows.append(row)
        else:
            uncovered.append({'article_id':article_id, **issue})
    return {'routable':{'recommendations':rows}, 'uncovered_routes':uncovered}


def effective_routes(request, recommendations, included):
    """Strict compatibility wrapper for callers that require all routes."""
    partition=partition_effective_routes(request,recommendations,included)
    require(not partition['uncovered_routes'],
            'No defensible RoB tool route for included article: '+partition['uncovered_routes'][0]['article_id'])
    return partition['routable']


def pre_v2_effective_routes(request, recommendations, included):
    """Freeze an interrupted pre-V2 run on its original ROBINS-I 2016 route."""
    routes=effective_routes(request,recommendations,included)
    for row in routes['recommendations']:
        if row.get('recommended_tool') == 'ROBINS-I':
            row['tool_version']='2016'
            row['tool_variant']=''
            row['route_resolution']='preserved_robins_i_2016_for_pre_v2_interrupted_run'
    return routes

def _guide_paths(guides, tool, version=None):
    if tool == 'ROBINS-I' and version == '2016':
        return ROBINS_I_2016_GUIDE
    return guides.get(tool)


def reference_material(tool, version=None):
    return reference_material_from(GUIDES, tool, version)


def reference_material_from(guides, tool, version=None):
    paths = _guide_paths(guides, tool, version)
    require(paths, 'No supplied assessment guide for routed tool: '+tool)
    result=[]
    for name in paths:
        path=RESOURCE_ROOT/name
        value=load_resource_text(RESOURCE_ROOT, name)
        require(len(value.strip())>100,'Missing or unreadable assessment guide: '+name)
        compact = load_compact_guidance(RESOURCE_ROOT, tool, version) if guides is GUIDES else None
        if compact:
            result.append({'file':'integrated_compact_'+tool.lower().replace(' ','_')+'_checklist.txt',
                           'sha256':digest(compact),'rubric':compact,
                           'source_file':name,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                           'source_representation':'compact_domain_checklist_from_registered_tool'})
        else:
            result.append({'file':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rubric':value})
    return result


def _pre_v2_reference_material(tool):
    """Rebuild the compact material emitted before the Structured source upgrade."""
    paths=PRE_V2_GUIDES.get(tool)
    require(paths, 'No supplied assessment guide for routed tool: '+tool)
    compact = (None if tool == 'QUIPS' else load_compact_guidance(
        RESOURCE_ROOT, tool, '2016' if tool == 'ROBINS-I' else None))
    result=[]
    for name in paths:
        path=RESOURCE_ROOT/name
        value=load_resource_text(RESOURCE_ROOT, name)
        require(len(value.strip())>100,'Missing or unreadable assessment guide: '+name)
        if compact:
            result.append({'file':'integrated_compact_'+tool.lower().replace(' ','_')+'_checklist.txt',
                           'sha256':digest(compact),'rubric':compact,
                           'source_file':name,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                           'source_representation':'compact_domain_checklist_from_registered_tool'})
        else:
            result.append({'file':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rubric':value})
    return result


def input_signature(request, screening, recommendations, included, guides, *, route_partition=None):
    partition=route_partition or partition_effective_routes(request,recommendations,included)
    routes=partition['routable']['recommendations']
    # Preserve the exact pre-partition fingerprint for runs in which every
    # article was routable. This lets completed historical batches resume
    # without relabelling or re-requesting valid RoB drafts.
    if not partition['uncovered_routes']:
        return digest({'request':request,'screening':screening,'routes':recommendations,
                       'guides':{row['article_id']:reference_material_from(
                           guides, row['recommended_tool'], row.get('tool_version')) for row in routes},
                       'sources':{d['article_id']:{k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in d['paths'].items() if k in ('markdown','content_list')}
                                  for d in request['documents'] if d['article_id'] in included}})
    route_ids={row['article_id'] for row in routes}
    return digest({'route_partition_schema':'screening_rob_routes/2.0',
                   'request':request,'screening':screening,'router_output':recommendations,
                   'routable_routes':routes,'uncovered_routes':partition['uncovered_routes'],
                   'guides':{row['article_id']:reference_material_from(
                       guides, row['recommended_tool'], row.get('tool_version')) for row in routes},
                   'sources':{d['article_id']:{k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in d['paths'].items() if k in ('markdown','content_list')}
                              for d in request['documents'] if d['article_id'] in route_ids}})


def pre_v2_input_signature(request, screening, recommendations, included):
    """Fingerprint from the compact-integrated format immediately before Structured support."""
    return digest({'request':request,'screening':screening,'routes':recommendations,
                   'guides':{name:_pre_v2_reference_material(name) for name in PRE_V2_GUIDES},
                   'sources':{d['article_id']:{k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in d['paths'].items() if k in ('markdown','content_list')}
                              for d in request['documents'] if d['article_id'] in included}})

def assessment_units(request, included):
    """Plan RoB only for the protocol's primary (core) analysis outcome."""
    primary_target=request['analysis_profile']['primary_analysis_spec'].get('target_id')
    targets=[target for target in request['protocol']['targets'] if target['target_id']==primary_target]
    require(len(targets)==1,'RoB requires exactly one declared primary-analysis target')
    registered={document['article_id'] for document in request['documents']}
    require(set(included)<=registered,'Screening included an article absent from the request registry')
    target=targets[0]
    return {'candidates':[
        {'candidate_id':'screen_rob_'+digest([d['article_id'],target['target_id']])[:20],
         'article_id':d['article_id'],'study_id':d['study_id'],'target_id':target['target_id'],
         'assessment_scope':'article_target','outcome':t.get('outcome'),
         'exposure':request['protocol'].get('pico',{}).get('intervention_or_exposure'),
         'comparison':request['protocol'].get('pico',{}).get('comparison')}
        for d in request['documents'] if d['article_id'] in included
        for t in [target]]}

def assess_screened(request, screening, recommendations, output, callback, *, allow_partial=False):
    included=[r['candidate_id'] for r in screening['decisions'] if r['final_status']=='Include']
    require(len(included)==len(set(included)),'Duplicate included article in screening results')
    units=assessment_units(request,included)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    route_partition=partition_effective_routes(request,recommendations,included)
    routes=route_partition['routable']
    uncovered_routes=route_partition['uncovered_routes']
    if uncovered_routes and not allow_partial:
        require(False,'screening RoB has uncovered routes; use the nonblocking stage or resolve manually: '
                +uncovered_routes[0]['article_id'])
    route_ids={row['article_id'] for row in routes['recommendations']}
    routed_units={'candidates':[candidate for candidate in units['candidates']
                                if candidate['article_id'] in route_ids]}
    signature=input_signature(request,screening,recommendations,included,GUIDES,route_partition=route_partition)
    lock=output/'risk_of_bias.input.json'
    compact_guide_migration=False
    recover_pre_v2=False
    if lock.exists() and read(lock)['fingerprint']!=signature:
        # Permit a failed checkpoint to move from either legacy full-guide
        # payload to the compact domain checklist. Source documents, routes,
        # and tool/domain requirements remain unchanged.
        legacy_signature=input_signature(request,screening,recommendations,included,LEGACY_QUIPS_GUIDES,
                                         route_partition=route_partition)
        full_signature=input_signature(request,screening,recommendations,included,ARCHIVED_FULL_GUIDES,
                                       route_partition=route_partition)
        pre_v2_signature=(pre_v2_input_signature(request,screening,recommendations,included)
                          if not uncovered_routes else None)
        completed=(output/'risk_of_bias_assessments.json').exists()
        old_fingerprint=read(lock)['fingerprint']
        recover_pre_v2=old_fingerprint==pre_v2_signature
        require(recover_pre_v2 or (old_fingerprint in {legacy_signature,full_signature} and not completed),
                'screening RoB inputs changed; use a new run')
        if recover_pre_v2:
            # The former run's responses were produced under the 2016 contract.
            # Preserve that contract for this run rather than re-labelling a
            # saved assessment as Structured or spending another provider request.
            routes=pre_v2_effective_routes(request,recommendations,included)
            signature=pre_v2_signature
        write(output/'risk_of_bias_input_legacy.json',read(lock))
        compact_guide_migration=old_fingerprint==full_signature
        write(output/'risk_of_bias_input_migration.json',{
            'reason':('reindexed unchanged compact 2016 RoB input after adding scoped ROBINS-I V2 support'
                      if recover_pre_v2 else 'replaced complete routed-tool score sheets with compact domain checklists'
                      if compact_guide_migration else 'removed QUIPS development-paper full text from model input'),
            'old_fingerprint':old_fingerprint,'new_fingerprint':signature,
        })
    write(output/'risk_of_bias_effective_routes.json',{
        'schema_version':'screening_rob_routes/2.0',
        'routable_recommendations':routes['recommendations'],
        'uncovered_routes':uncovered_routes,
    })
    write(lock,{'fingerprint':signature,'scope':'article_target','assessment_count':len(units['candidates']),
                'routable_assessment_count':len(routed_units['candidates']),
                'uncovered_assessment_count':len(units['candidates'])-len(routed_units['candidates'])})
    # integrated.0 initially scheduled every target.  On a resumed legacy run, retain
    # those completed records as an audit artefact and leave only primary-target
    # records in the resumable checkpoint.
    checkpoint=output/'risk_of_bias.progress.json'
    if checkpoint.is_file():
        prior=read(checkpoint)
        if compact_guide_migration:
            # Do not mix one completed full-score-sheet draft with new compact
            # checklist drafts in a single reproducible RoB assessment set.
            write(output/'risk_of_bias_superseded_checkpoint.json',{
                'reason':'reassess every draft with the same compact routed-tool checklist',
                'assessments':prior.get('assessments',[]),
            })
            write(checkpoint,{'schema_version':prior.get('schema_version',SCHEMA),'case_id':request['case_id'],'assessments':[]})
            prior=read(checkpoint)
        allowed={item['candidate_id'] for item in routed_units['candidates']}
        route_by_article={row['article_id']:row for row in routes['recommendations']}
        def reusable(item):
            candidate_id=(item.get('scope') or {}).get('candidate_id')
            route=route_by_article.get(item.get('article_id'),{})
            tool=item.get('tool') or {}
            return (candidate_id in allowed and item.get('status')=='draft'
                    and tool==routed_tool(route))
        retained=[item for item in prior.get('assessments',[]) if reusable(item)]
        removed=[item for item in prior.get('assessments',[]) if not reusable(item)]
        if removed:
            write(output/'risk_of_bias_superseded_checkpoint.json',{
                'reason':'non-primary, unresolved, or route-mismatched legacy assessment',
                'assessments':removed,
            })
            write(checkpoint,{'schema_version':prior.get('schema_version',SCHEMA),'case_id':request['case_id'],'assessments':retained})
    targets={t['target_id']:t for t in request['protocol']['targets']}
    def guided(payload):
        tool=payload['assessment_template']['tool']['name']
        version=payload['assessment_template']['tool'].get('version')
        refs=reference_material(tool,version)
        payload['method_references']=refs
        for region in payload['source_regions']:region['article_id']=payload['candidate']['article_id']
        payload['pico']=request['protocol'].get('pico',{})
        payload['target']=targets[payload['candidate']['target_id']]
        payload['instructions'] += (
            '\nAssess this article for the named target outcome and exposure. Read the supplied rubric as methodological reference only; '
            'use source_regions exclusively for study evidence. Preserve outcome specificity. '
            'Return overall_rationale and domains as an array of {domain_id, judgment, rationale, evidence}. '
            'For each domain also return assessment_items listing the rubric/signalling items, responses and reasoning, using the vocabulary of that tool. '
            'Each evidence item must copy article_id, source_type, locator, and quote from source_regions. '
            'Never cite the methodological references as study evidence. For absent evidence use judgment unknown and evidence []. '
            'For multiple materially different analyses within this target, use unknown for a domain that cannot have a shared assessment and explain why. '
            'Do not fabricate a pooled effect or choose the best numerical result. '
            'For QUIPS, domain high risk drives the pipeline high-risk exclusion policy; do not sum domain scores. '
            'Keep the JSON compact: one evidence item per domain at most; each rationale and item reasoning at most two short sentences; '
            'each quote at most 300 characters. Do not repeat the source text, rubric, question, or item labels in prose.')
        # Record request/response scale without saving content or provider
        # credentials.  This distinguishes a context-size failure from a
        # provider queue or slow generation in the run log.
        request_size=len(json.dumps(payload,ensure_ascii=False,separators=(',',':')))
        unit_id=payload['candidate']['candidate_id']
        started=time.monotonic()
        source_chars=sum(len(str(region.get('text',''))) for region in payload['source_regions'])
        print(f'[screening-rob] request candidate={unit_id} input_chars={request_size} source_chars={source_chars} source_regions={len(payload["source_regions"])}',flush=True)
        # Preserve the provider response locally before applying validation.
        # It is required to diagnose an incompatible JSON shape without another
        # paid model call, and stays inside the run's auditable output directory.
        raw_dir=output/'risk_of_bias_raw_responses'
        raw_dir.mkdir(parents=True,exist_ok=True)
        payload_fingerprint=digest(payload)
        raw_path=raw_dir/(unit_id+'.'+payload_fingerprint[:16]+'.json')
        legacy_raw=raw_dir/(unit_id+'.json')
        if not raw_path.is_file() and legacy_raw.is_file():
            legacy=read(legacy_raw)
            if legacy.get('candidate_id')==unit_id and legacy.get('input_chars')==request_size:
                write(raw_path,{**legacy,'input_fingerprint':payload_fingerprint,
                                'migrated_from':legacy_raw.name})
        # A run that began before the ROBINS-I V2 routing change keeps its
        # original 2016 route.  Its saved provider output is still valid for
        # that route, even though the later compact prompt has a different
        # payload digest.  Recover precisely one response belonging to this
        # assessment; never use this compatibility path for a new run.
        if not raw_path.is_file() and recover_pre_v2:
            legacy_matches=[]
            for legacy_path in sorted(raw_dir.glob(unit_id+'.*.json')):
                legacy=read(legacy_path)
                if legacy.get('candidate_id')==unit_id and isinstance(legacy.get('response'),dict):
                    legacy_matches.append((legacy_path,legacy))
            require(len(legacy_matches)<=1,
                    'multiple saved legacy RoB responses; use a new run rather than choosing one')
            if legacy_matches:
                legacy_path,legacy=legacy_matches[0]
                write(raw_path,{**legacy,'input_fingerprint':payload_fingerprint,
                                'migrated_from':legacy_path.name,
                                'migration_reason':'pre_v2_interrupted_run_preserved_robins_i_2016'})
        if raw_path.is_file():
            saved=read(raw_path)
            require(saved.get('input_fingerprint')==payload_fingerprint,
                    'saved RoB response belongs to different model input')
            response=saved['response']
            print(f'[screening-rob] reusing saved response candidate={unit_id}; no provider call',flush=True)
        else:
            try:
                response=callback(payload)
            except Exception as exc:
                if getattr(exc,'rob_raw_response',None) is not None:
                    write(raw_dir/(unit_id+'.strict_rejection.json'),{
                        'candidate_id':unit_id,'response_contract':'pydantic_strict_rob/1.0',
                        'raw_response':exc.rob_raw_response,'error':str(exc)})
                print(f'[screening-rob] failed candidate={unit_id} input_chars={request_size} elapsed={time.monotonic()-started:.1f}s error={type(exc).__name__}',flush=True)
                raise
            print(f'[screening-rob] response candidate={unit_id} input_chars={request_size} output_chars={len(json.dumps(response,ensure_ascii=False,separators=(",",":")))} elapsed={time.monotonic()-started:.1f}s',flush=True)
            write(raw_path,{'candidate_id':unit_id,'input_chars':request_size,
                            'input_fingerprint':payload_fingerprint,'response':response})
        # Validate each completed unit before checkpointing, including quotes.
        base=payload['assessment_template'];response=copy.deepcopy(response)
        require(isinstance(response,dict),'RoB response must be an object')
        response=deduplicate_identical_domains(normalize_assessment_envelope(response))
        for key in ('assessment_id','article_id','study_id','scope','tool'):response[key]=copy.deepcopy(base[key])
        response['status']='draft';response['provenance']={'mode':'automated_draft'}
        response['effect_of_interest']=payload['effect_of_interest']
        if tool=='QUIPS':
            domains=response.get('domains')
            require(isinstance(domains,list),'QUIPS response domains must be an array')
            # Some providers append a seventh `overall` object even though the
            # overall judgment is already present at the top level.  Preserve
            # it for audit but keep the QUIPS domain contract at Q1-Q6.
            extras=[domain for domain in domains if isinstance(domain,dict)
                    and str(domain.get('domain_id','')).lower() in ('overall','summary')]
            if extras:
                response['provider_extra_summary_domains']=copy.deepcopy(extras)
                domains=[domain for domain in domains if domain not in extras]
                response['domains']=domains
            for domain in domains:
                require(isinstance(domain,dict),'QUIPS response domains must contain objects')
                items=domain.get('assessment_items')
                expected=QUIPS_ITEM_COUNTS.get(domain.get('domain_id'))
                require(expected is not None and isinstance(items,list)
                        and len(items) in (expected-1,expected),
                        'QUIPS response must cover every rubric item in each domain')
                labels=[]
                for item in items:
                    require(isinstance(item,dict) and isinstance(item.get('item'),str) and item['item'].strip()
                            and isinstance(item.get('response'),str) and item['response'].strip()
                            and isinstance(item.get('reasoning'),str) and item['reasoning'].strip(),
                            'QUIPS assessment item requires item, response, and reasoning')
                    labels.append(item['item'].strip().lower())
                require(len(labels)==len(set(labels)),'QUIPS response contains duplicate rubric items')
                domain['assessment_item_coverage']={
                    'expected_rubric_rows':expected,'provider_rows':len(items),
                    'summary_represented_by_domain_judgment':len(items)==expected-1,
                }
            values=[d.get('judgment') for d in response.get('domains',[])]
            response['overall_judgment']=('high' if 'high' in values else 'unknown' if any(v in ('unknown','unclear') for v in values)
                                          else 'moderate' if 'moderate' in values else 'low')
            response['aggregation_policy']='integrated_any_high_domain; otherwise unknown, moderate, low; not an official QUIPS overall score'
        response['method_references']=[{k:v for k,v in r.items() if k!='rubric'} for r in refs]
        try:
            validate_assessments({'schema_version':SCHEMA,'case_id':request['case_id'],'assessments':[response]},request,units)
        except Exception as exc:
            write(raw_dir/(unit_id+'.validation_error.json'),{
                'candidate_id':unit_id,'error_type':type(exc).__name__,'error':str(exc),
            })
            print(f'[screening-rob] validation failed candidate={unit_id} error={type(exc).__name__}: {exc}',flush=True)
            raise
        write(raw_dir/(unit_id+'.validation_status.json'),{
            'candidate_id':unit_id,'status':'valid','input_fingerprint':payload_fingerprint,
            'response_fingerprint':digest(response),
        })
        return response
    print(f'[screening-rob] {len(included)} articles, {len(units["candidates"])} article/target assessments; '
          f'routable={len(routed_units["candidates"])} uncovered={len(uncovered_routes)}',flush=True)
    workers=max(1,min(request['options']['max_workers'],int(os.environ.get('TRACE_ROB_WORKERS', request['options']['max_workers']))))
    print(f'[screening-rob] parallel workers={workers}',flush=True)
    source_budget=int(os.environ.get('TRACE_ROB_SOURCE_CHARS','26000'))
    print(f'[screening-rob] source evidence budget={source_budget} chars per assessment',flush=True)
    routed_result=assess_request(request,routes,routed_units,callback=guided,workers=workers,
                                 checkpoint_path=checkpoint,source_region_max_chars=source_budget,allow_partial=allow_partial) \
        if routed_units['candidates'] else {'schema_version':SCHEMA,'case_id':request['case_id'],
                                             'assessments':[],'failures':[]}
    completed_by_id={a['scope']['candidate_id']:a for a in routed_result['assessments']}
    uncovered_by_article={row['article_id']:row for row in uncovered_routes}
    assessments=[]
    for candidate in units['candidates']:
        assessment=completed_by_id.get(candidate['candidate_id'])
        if assessment is None:
            issue=uncovered_by_article.get(candidate['article_id'])
            if issue is None:
                # A routed provider failure is still unknown. It is distinct
                # from an uncovered instrument and remains eligible for a
                # checkpointed retry on the next run.
                assessment=unassessed_record(request,candidate)
            else:
                route=copy.deepcopy(issue.get('routing_recommendation') or {})
                assessment=unassessed_record(request,candidate,routed_tool(route))
                assessment['reason']=issue['reason']+'; manual_methodological_assessment_required'
                assessment['routing_recommendation']=route
                assessment['routing_status']='uncovered'
        assessments.append(assessment)
    result=validate_assessments({'schema_version':SCHEMA,'case_id':request['case_id'],'assessments':assessments},request,units)
    result.update({'failures':routed_result.get('failures',[]),'uncovered_routes':copy.deepcopy(uncovered_routes),
                   'assessment_scope':'article_target','input_fingerprint':signature,'assessment_units':units,
                   'source_stage':'screening_rob'})
    write(output/'risk_of_bias_assessments.json',result)
    write(output/'risk_of_bias_complete.json',{'case_id':request['case_id'],'article_count':len(included),'assessment_count':len(result['assessments'])})
    return result

def map_to_evidence(request, ledger, payload):
    require(payload.get('assessment_scope')=='article_target','Expected screening article/target RoB')
    checked=validate_assessments(payload,request,payload['assessment_units'])
    lookup={(a['article_id'],a['scope']['target_id']):a for a in checked['assessments']}
    require(len(lookup)==len(checked['assessments']),'Duplicate screening RoB article/target')
    primary_target=request['analysis_profile']['primary_analysis_spec'].get('target_id')
    core_candidates=[candidate for candidate in ledger['candidates'] if candidate['target_id']==primary_target]
    rows=[]
    for candidate in core_candidates:
        a=lookup.get((candidate['article_id'],candidate['target_id']))
        require(a is not None,'Missing screening RoB for '+candidate['candidate_id'])
        mapped=copy.deepcopy(a)
        mapped['screening_assessment_id']=a['assessment_id']
        mapped['assessment_id']='mapped_'+digest([a['assessment_id'],candidate['candidate_id']])[:20]
        mapped['scope']={'candidate_id':candidate['candidate_id'],'target_id':candidate['target_id']}
        mapped['mapping_scope']='article_target_shared_assessment; not result_specific_reassessment'
        rows.append(mapped)
    return validate_assessments({'schema_version':SCHEMA,'case_id':request['case_id'],'assessments':rows},request,{'candidates':core_candidates})

def main():
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);p.add_argument('--screening-dir',type=Path,required=True)
    a=p.parse_args()
    from extensions.transport import json_callback
    from extensions.rob_resilience import assess_nonblocking
    assess_nonblocking(read(a.request),read(a.screening_dir/'effective_screening_results.json'),
                    read(a.screening_dir/'risk_of_bias_recommendations.json'),a.screening_dir,lambda payload: json_callback(purpose='bias')(payload))
if __name__=='__main__':main()
