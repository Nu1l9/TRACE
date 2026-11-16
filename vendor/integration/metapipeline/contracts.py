"""Versioned, standard-library-only boundaries between the frozen components."""
from __future__ import annotations
import copy
import hashlib
import json
import re
from .evidence_context import build_review_context
from pathlib import Path

VERSION = 'meta_pipeline/1.0'
EARLY_BASELINE_CASES = {'ma_0001','ma_0011','ma_0012','ma_0044','ma_0046','ma_0076','ma_0086'}
ALIASES = {k: 'binary_arm_counts' for k in ('binary_arm_outcome','binary_arm_event_counts','two_group_event_counts','binary_exposure_outcome_counts')}
ALIASES['adjusted_time_to_event_effect'] = 'adjusted_ratio_effect'
REPORTED = {'adjusted_ratio_effect','reported_ratio_effect','time_to_event','reported_standardized_mean_difference'}
SUPPORTED = REPORTED | {'binary_arm_counts','complete_genotype_case_control_counts','diagnostic_four_cell_counts','single_arm_event_counts','independent_continuous_arms','event_counts_with_person_time'}
ENVELOPE = ('candidate_id','article_id','study_id','target_id','exposure','comparison','outcome','population_or_subgroup','timepoint','data_family','estimand','raw_data','effect','source_location')
FORBIDDEN = {'gold','gold_labels','expected_label','quantitative_usability','selected_candidate_ids','selected_evidence_ids','candidate_ids','evidence_ids','best_candidate','gold_only','poolable','quantitative_use_policy'}

class ContractError(ValueError):
    pass

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=lambda x: (_ for _ in ()).throw(ContractError('Non-finite JSON number: '+x)))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def text(value):
    if isinstance(value, dict):
        return str(value.get('description') or value.get('label') or '')
    if isinstance(value, list):
        return '; '.join(text(x) for x in value)
    return str(value or '')

def require(condition, message):
    if not condition:
        raise ContractError(message)

def identity(value, label):
    require(isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,159}',value), f'{label}: expected a nonempty safe identifier')
    return value

def no_labels(value, path='request'):
    if isinstance(value, dict):
        for k, v in value.items():
            require(k not in FORBIDDEN, f'{path}.{k}: evaluation/selection labels cannot cross this boundary')
            no_labels(v, f'{path}.{k}')
    elif isinstance(value, list):
        for i, v in enumerate(value): no_labels(v, f'{path}[{i}]')

def normalize_request(raw, base=Path('.'), check_paths=True):
    require(isinstance(raw, dict), 'request must be an object')
    r = copy.deepcopy(raw)
    def reject_credentials(value):
        if isinstance(value, dict):
            for key, child in value.items():
                require(not any(word in key.casefold() for word in ('api_key','api-key','secret','password')), 'Credentials belong in an environment file, not request JSON')
                reject_credentials(child)
        elif isinstance(value, list):
            for child in value: reject_credentials(child)
    reject_credentials(r)
    require(r.get('schema_version') == VERSION, f'schema_version must be {VERSION}')
    identity(r.get('case_id'), 'case_id')
    require(bool(text(r.get('topic'))), 'topic is required')
    p = r.get('protocol')
    require(isinstance(p,dict), 'protocol is required; fill the Brain proposal first')
    no_labels(p, 'protocol')
    identity(p.get('protocol_id'), 'protocol.protocol_id')
    require(p.get('case_id') == r['case_id'], 'protocol.case_id does not match request')
    pico = p.get('pico', {})
    require(isinstance(pico, dict), 'protocol.pico must be an object')
    for canonical, aliases in [('comparison',('comparator',)), ('intervention_or_exposure',('exposure','intervention'))]:
        values = [text(pico[k]) for k in (canonical,*aliases) if text(pico.get(k))]
        require(len(set(values))<=1, f'Conflicting PICO aliases for {canonical}')
        if values: pico[canonical] = {'description':values[0]}
        for alias in aliases: pico.pop(alias,None)
    for k in ('population','intervention_or_exposure','comparison','outcome'):
        require(bool(text(pico.get(k))), f'protocol.pico.{k} is required')
    targets = p.get('targets',[])
    require(isinstance(targets,list) and targets, 'protocol.targets must be nonempty')
    target_ids = []
    for target in targets:
        tid = identity(target.get('target_id'), 'target_id'); target_ids.append(tid)
        require(bool(text(target.get('outcome'))), f'{tid}.outcome is required')
        for key in ('accepted_data_families','preferred_effect_measures'):
            require(isinstance(target.get(key),list) and all(isinstance(x,str) and x for x in target[key]), f'{tid}.{key} must be a string array')
        require(bool(target['accepted_data_families']),f'{tid}.accepted_data_families must be nonempty')
        target['accepted_data_families'] = list(dict.fromkeys(ALIASES.get(x,x) for x in target['accepted_data_families']))
    require(len(set(target_ids))==len(target_ids), 'duplicate target_id')
    profile = r.get('analysis_profile')
    require(isinstance(profile,dict), 'analysis_profile is required')
    no_labels(profile,'analysis_profile')
    require(str(profile.get('profile_version','')).startswith('2'), 'analysis_profile requires version 2')
    primary = profile.get('primary_analysis_spec')
    require(isinstance(primary,dict), 'primary_analysis_spec is required')
    specs = [primary,*profile.get('secondary_outcomes',[])]
    analysis_ids=[]
    for spec in specs:
        effective = {**primary,**spec}
        analysis_ids.append(identity(spec.get('analysis_id'),'analysis_id'))
        require(effective.get('target_id') in target_ids, 'analysis target_id not in protocol')
        require(bool(text(effective.get('outcome_group'))),'analysis outcome_group is required')
        families = effective.get('eligible_data_families',[])
        require(isinstance(families,list) and families,'eligible_data_families is required')
        canonical = list(dict.fromkeys(ALIASES.get(x,x) for x in families))
        require(len(canonical)==1, 'Analysis requires exactly one data family per analysis; declare separate analyses for distinct families')
        require(set(canonical)<=SUPPORTED, f'Analysis has no calculator for {sorted(set(canonical)-SUPPORTED)}; keep source-only evidence outside declared quantitative analyses or extend the analysis component')
        spec['eligible_data_families'] = canonical
        effects = effective.get('accepted_effect_types',[])
        require(isinstance(effects,list) and effects and all(isinstance(x,str) and x for x in effects),'accepted_effect_types is required')
        target = next(t for t in targets if t['target_id']==effective['target_id'])
        require(set(canonical)<=set(target['accepted_data_families']), 'analysis families must be captured by the matching extraction target')
        preferred=set(target['preferred_effect_measures'])
        if 'diagnostic_four_cell_counts' in canonical and {'sensitivity','specificity'} <= preferred:
            preferred.add('diagnostic_accuracy_pair')
        if preferred:
            require(set(effects)<=preferred, 'analysis effects must be captured by the matching extraction target')
    require(len(set(analysis_ids))==len(analysis_ids),'duplicate analysis_id')
    documents = r.get('documents')
    require(isinstance(documents,list) and documents, 'documents must contain supplied candidate articles')
    article_ids=[]; document_ids=[]
    for d in documents:
        no_labels(d,'documents')
        aid=identity(d.get('article_id'),'article_id'); article_ids.append(aid)
        identity(d.get('study_id'),'study_id (must identify an independent study)')
        d.setdefault('document_id',aid); document_ids.append(identity(d['document_id'],'document_id'))
        require(d.get('role','candidate_article')=='candidate_article',f'{aid}: only primary candidate reports are allowed')
        paths=d.get('paths',{})
        require(isinstance(paths,dict) and any(paths.get(k) for k in ('markdown','pdf')), f'{aid}: markdown or PDF is required')
        for key in ('markdown','content_list','pdf'):
            if paths.get(key):
                path=Path(paths[key]).expanduser()
                path=(base/path).resolve() if not path.is_absolute() else path.resolve()
                if check_paths: require(path.is_file(),f'{aid}: missing {key}: {path}')
                paths[key]=str(path)
    require(len(set(article_ids))==len(article_ids),'duplicate article_id')
    require(len(set(document_ids))==len(document_ids),'duplicate document_id')
    options=r.setdefault('options',{})
    # Typed is the canonical extractor for every benchmark case.  EARLY_BASELINE_CASES is
    # retained only as historical metadata/document-discovery support; it must
    # never silently route a current request back to the legacy extractor.
    runtime=options.setdefault('extraction_runtime','typed')
    require(runtime in ('baseline','typed'), 'extraction_runtime must be baseline or typed')
    options.setdefault('provider','official_openai')
    require(options['provider'] in ('official_openai','openai_compatible'),'unknown provider')
    options.setdefault('model','gpt-5.6-luna')
    options.setdefault('route_rob',True)
    for key,default,minimum in [('max_retries',3,0),('max_workers',2,1)]:
        options.setdefault(key,default)
        require(type(options[key]) is int and options[key]>=minimum, f'{key} must be an integer >= {minimum}')
    return r

def screening_protocol(r):
    p=r['protocol']['pico']
    # Study eligibility uses the clinical outcome, not the extraction target's
    # requirements for a particular numeric representation or calculator.
    outcome=text(p.get('outcome'))
    return {'case_id':r['case_id'],'review_question':r['topic'],'pico':{
        'population':text(p['population']), 'intervention_or_exposure':text(p['intervention_or_exposure']),
        'comparator':text(p['comparison']),
        'outcomes':[outcome] if outcome else [text(t['outcome']) for t in r['protocol']['targets']],
        'study_designs':list(r['protocol'].get('eligible_study_designs') or [])}}

def normalize_screening(screening):
    result=copy.deepcopy(screening)
    result.setdefault('case_id',(result.get('protocol') or {}).get('case_id'))
    if 'decisions' not in result and 'screening' in result:
        result['decisions']=result['screening']
    return result

def accepted_articles(r, screening):
    screening=normalize_screening(screening)
    require(screening.get('case_id')==r['case_id'],'screening case_id mismatch')
    rows=screening.get('decisions',[])
    expected={d['article_id'] for d in r['documents']}
    ids=[row.get('candidate_id') for row in rows]
    require(len(ids)==len(set(ids)) and set(ids)==expected,'screening decisions must cover each candidate article exactly once')
    require(all(x.get('final_status') in ('Include','Exclude','Unsure') for x in rows),'invalid screening status')
    return [x['candidate_id'] for x in rows if x['final_status']=='Include']

def prepare_extraction(r, included, directory):
    directory=Path(directory)
    docs={d['article_id']:copy.deepcopy(d) for d in r['documents'] if d['article_id'] in included}
    require(set(docs)==set(included), 'unknown included article')
    require(bool(included),'no included articles')
    for aid,d in docs.items():
        require(d['paths'].get('markdown') or d['paths'].get('content_list'), f'{aid}: extraction requires prepared Markdown or MinerU content_list; PDF alone cannot feed this extractor')
        if (r['options']['extraction_runtime']=='baseline' or not d['paths'].get('content_list')) and d['paths'].get('markdown'):
            # Reuse the frozen baseline text-to-content-list converter (including PDF page markers).
            import importlib.util
            source=Path(__file__).resolve().parents[1]/'vendor/evidence_extraction/system/baseline/run_benchmark_extraction.py'
            spec=importlib.util.spec_from_file_location('_frozen_markdown_adapter',source)
            adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
            content_path=directory/'inputs'/(aid+'_content_list.json')
            write(content_path,adapter.markdown_to_content_list(Path(d['paths']['markdown']).read_text(encoding='utf-8'),12000))
            d['paths']['content_list']=str(content_path.resolve())
        items=read(d['paths']['content_list'])
        require(isinstance(items,list) and items and all(isinstance(x,dict) for x in items),f'{aid}: content_list must contain source-region objects')
    write(directory/'inputs/extraction_documents.json',{'article_ids':included,'manifest_file':'manifest.json','paths_file':'paths.json'})
    write(directory/'inputs/manifest.json',{'documents':{aid:{'document_id':d['document_id']} for aid,d in docs.items()}})
    write(directory/'inputs/paths.json',{'documents':{aid:{k:v for k,v in d['paths'].items() if k in ('markdown','content_list')} for aid,d in docs.items()}})
    pico=r['protocol']['pico']; targets=copy.deepcopy(r['protocol']['targets'])
    config={
      'schema_version':'meta_analysis_case/baseline','case_id':r['case_id'],'task_id':r['case_id']+'_extraction','topic':r['topic'],
      'pico':{'population':text(pico['population']),'exposure':text(pico['intervention_or_exposure']),'comparator':text(pico['comparison']),'outcomes':[text(t['outcome']) for t in targets]},
      'inputs':{'document_index':'inputs/extraction_documents.json','included_article_ids':included,'study_ids':{aid:d['study_id'] for aid,d in docs.items()},'representations':['markdown','content_list'],'forbidden_document_ids':[]},
      'extraction':{'targets':targets,'allowed_effect_types':list(dict.fromkeys(e for t in targets for e in t['preferred_effect_measures'])),'estimands':list(dict.fromkeys(e for t in targets for e in t.get('estimands',[]))), 'model_selection_policy':'Preserve every source-reported candidate; no final effect selection in extraction.', 'exclusion_concepts':'', 'estimand_policy':'preserve_source_reported'},
      'workflow':{'extractor':'locator_extractor_tools','planner':'disabled','critic':'disabled'},
      'system_partition':'locator_extractor_typed_data_family' if r['options']['extraction_runtime']=='typed' else 'locator_extractor_baseline',
      'models':{role:r['options']['model'] for role in ('locator','extractor')},
      'execution_policy':{'max_article_retries':2,'llm_timeout_seconds':110,'locator_preview_chars':5000,'max_locator_sources':0 if r['options']['extraction_runtime']=='typed' else 16}}
    write(directory/'case_config.json',config)
    return directory/'case_config.json'

def evidence_handoff(r, screening, submission, review):
    require(not any(x.get('final_status')=='Unsure' for x in normalize_screening(screening).get('decisions',[])), 'unresolved screening must be resolved before evidence handoff')
    included=set(accepted_articles(r,screening))
    require(submission.get('case_id')==r['case_id'],'extraction case_id mismatch')
    require(review.get('case_id')==r['case_id'],'review case_id mismatch')
    rows=submission.get('predictions',[])
    require(isinstance(rows,list),'predictions must be an array')
    ids=[x.get('prediction_id') for x in rows]
    require(all(isinstance(x,str) and x for x in ids) and len(set(ids))==len(ids),'prediction IDs must be nonempty and unique')
    require(not review.get('failed_review_count') and not review.get('failures'),'candidate review failed; retry before analysis')
    records=review.get('candidate_reviews',[])
    rid=[x.get('prediction_id') for x in records]
    require(len(set(rid))==len(rid) and set(rid)==set(ids),'review must cover each prediction exactly once')
    by_review={x['prediction_id']:x for x in records}
    docs={x['article_id']:x for x in r['documents']}
    targets={x['target_id'] for x in r['protocol']['targets']}
    candidates=[]; audit=[]
    for row in rows:
        pid=row['prediction_id']; aid=row.get('article_id'); rec=by_review[pid]; decision=rec.get('review',{})
        require(aid in included,f'{pid}: evidence from a non-included article')
        require(rec.get('article_id')==aid and decision.get('prediction_id')==pid,f'{pid}: review identity mismatch')
        disposition=decision.get('disposition')
        require(disposition in ('keep_as_valid_candidate','discard_clear_mismatch','needs_further_review'),f'{pid}: invalid review disposition')
        review_context=build_review_context(row,decision)
        entry={'prediction_id':pid,'article_id':aid,'disposition':disposition,'reason':decision.get('reason',''),'verification_status':row.get('verification_status'),'review_context':review_context,'warnings':[]}
        audit.append(entry)
        if disposition=='discard_clear_mismatch': continue
        study=docs[aid]['study_id']
        require(not row.get('study_id') or row['study_id']==study, f'{pid}: study_id conflicts with document registry')
        # Missing target is an unresolved extraction field. Never guess from the primary analysis.
        tid=row.get('target_id')
        require(tid in targets, f'{pid}: target_id missing/unknown; repair extraction before analysis')
        family=ALIASES.get(row.get('data_family'),row.get('data_family'))
        require(isinstance(family,str) and family, f'{pid}: missing data_family')
        raw=copy.deepcopy(row.get('raw_data') or {})
        require(isinstance(raw,dict),f'{pid}: raw_data must be an object')
        no_labels(raw,f'{pid}.raw_data')
        if family=='complete_genotype_case_control_counts' and 'case_counts' in raw:
            mapping=docs[aid].get('genotype_contrast',{})
            require(mapping.get('source_note') and mapping.get('rsid')==raw.get('rsid'), f'{pid}: provide documents[].genotype_contrast with rsid, source-supported heterozygote_label/common_homozygote_label and source_note; allele roles cannot be inferred from counts')
            require(mapping.get('heterozygote_label')!=mapping.get('common_homozygote_label'),f'{pid}: genotype contrast roles must have different source labels')
            for group in ('case','control'):
                for role in ('heterozygote','common_homozygote'):
                    label=mapping.get(role+'_label')
                    require(label in raw.get(group+'_counts',{}),f'{pid}: genotype label {label!r} missing in {group}_counts')
                    raw[group+'_'+role]=raw[group+'_counts'][label]
            entry['raw_field_mapping']=mapping
        # Baseline computes used_effect for arm summaries. Only source-reported effect families pass it on.
        effect=copy.deepcopy(row.get('used_effect') or {}) if family in REPORTED else {}
        if family not in REPORTED and any(v is not None for v in (row.get('used_effect') or {}).values()):
            entry['warnings'].append('upstream_derived_effect_omitted; analysis recalculates from raw_data')
        if family not in SUPPORTED: entry['warnings'].append('no_analysis_calculator_for_data_family')
        source=copy.deepcopy(row.get('source_location') or {})
        require(isinstance(source,dict),f'{pid}: source_location must be an object')
        require(not source.get('document_id') or source['document_id']==docs[aid]['document_id'],f'{pid}: source document mismatch')
        source['document_id']=docs[aid]['document_id']
        source['review_context']=copy.deepcopy(review_context)
        # Retain source model/adjustment context inside an existing envelope field, never a selection label.
        source['extraction_context']={k:row[k] for k in ('model_name','adjustment','p_value','sample_size') if row.get(k) not in (None,'')}
        outcome=copy.deepcopy(row.get('outcome'))
        require(isinstance(outcome,(str,dict)) and bool(text(outcome)), f'{pid}: missing outcome')
        if isinstance(outcome,str): outcome={'label':outcome}
        metadata={k:raw[k] for k in ('outcome_axis','measurement_semantics','unit','rsid','study_design','phenotype','genotype_role') if raw.get(k) not in (None,'')}
        if metadata:outcome['analysis_semantics']={**outcome.get('analysis_semantics',{}),**metadata}
        candidate=dict(zip(ENVELOPE,[pid,aid,study,tid,row.get('exposure',''),row.get('comparator',''),outcome,row.get('population_or_subgroup',''),row.get('timepoint',''),family,row.get('estimand',''),raw,effect,source]))
        no_labels(candidate,f'candidate.{pid}')
        candidates.append(candidate)
    return {'candidates':candidates},{'records':audit,'candidate_count':len(candidates),'discarded_count':sum(x['disposition']=='discard_clear_mismatch' for x in audit)}

def write_analysis_input(r, candidates, directory):
    directory=Path(directory)
    write(directory/'extraction_protocol.json',r['protocol'])
    write(directory/'analysis_task.json',{'task_id':r['case_id']+'_analysis','case_id':r['case_id'],'task_type':'analysis','title':r['topic']})
    write(directory/'analysis_profile_structured.json',r['analysis_profile'])
    write(directory/'analysis_evidence_candidates.json',candidates)
