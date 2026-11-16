"""Import only declared protocol/profile/primary-document inputs from the supplied bundle."""
from __future__ import annotations
import copy
from pathlib import Path
from .contracts import read,write,text,normalize_request,require,VERSION,EARLY_BASELINE_CASES,ALIASES
ROOT=Path(__file__).resolve().parents[1]
BENCH=ROOT/'vendor/evidence_extraction/benchmark'

def import_case(case_id,output):
    cases=list(BENCH.glob('*/cases/'+case_id));require(len(cases)==1,'unknown or ambiguous case_id')
    case=cases[0]
    source=read(case/'inputs/extraction_protocol.json')
    # Explicit projection: never copy the benchmark task, gold, or analysis evidence pool.
    pico=copy.deepcopy(source.get('pico',{}));targets=[]
    for target in source['targets']:
        target={k:copy.deepcopy(v) for k,v in target.items() if k in ('target_id','outcome','comparison','population_or_subgroup','timepoint_policy','accepted_data_families','preferred_effect_measures','estimands','data_contract','extraction_notes','candidate_capture_policy')}
        if isinstance(target.get('outcome'),str):target['outcome']={'description':target['outcome']}
        if not text(target.get('outcome')):target['outcome']=copy.deepcopy(pico.get('outcome',{}))
        targets.append(target)
    if not text(pico.get('outcome')):pico['outcome']={'description':'; '.join(text(t['outcome']) for t in targets)}
    protocol={'protocol_id':source['protocol_id'],'case_id':case_id,'pico':pico,'targets':targets,'eligible_study_designs':source.get('eligible_study_designs',[])}
    raw_profile=read(case/'inputs/analysis_profile.json')
    if str(raw_profile.get('profile_version','')).startswith('2'):
        profile=raw_profile
    else:
        analyses=raw_profile.get('analyses',[])
        if not analyses and (raw_profile.get('selection') or raw_profile.get('eligibility')):
            selection=raw_profile.get('selection') or raw_profile['eligibility']
            family=selection.get('data_family')
            matched=[t for t in targets if family in t['accepted_data_families']]
            require(len(matched)==1,'profile needs an explicit target_id')
            target=matched[0]
            effect=(raw_profile.get('synthesis') or raw_profile.get('pooling') or {}).get('effect_measure')
            effect={'odds_ratio':'OR'}.get(effect,effect)
            require(bool(effect),'profile needs an explicit effect measure')
            rules=[f'{k}: {text(v)}' for k,v in selection.items() if k!='data_family']
            rules += [f'{k}: {text(v)}' for k,v in raw_profile.get('candidate_resolution',{}).items()]
            analyses=[{'analysis_id':case_id+'_primary','target_id':target['target_id'],'outcome_group':text(target['outcome']),'eligible_data_families':[family],'accepted_effect_types':[effect],'selection_rules':rules}]
            if not text(pico.get('comparator')) and selection.get('genotype_role'):
                pico['comparator']={'description':selection['genotype_role']}
        require(bool(analyses),'source case has no declared analyses')
        specs=[]
        for a in analyses:
            s={k:copy.deepcopy(v) for k,v in a.items() if k in ('analysis_id','pooled_result_id','target_id','title','outcome_group','outcome_terms','eligible_outcome_codes','eligible_data_families','accepted_effect_types','estimands','population','exposure','comparison','timepoint_policy','model_policy','dependence_policy')}
            rules=a.get('selection_rules',[])
            s['selection_rules']=rules.get('rules',[]) if isinstance(rules,dict) else rules
            if not s.get('target_id'):
                require(len(targets)==1,'profile target_id is ambiguous')
                s['target_id']=targets[0]['target_id']
            s.setdefault('outcome_group',text(a.get('outcome_axis')) or text(next(t for t in targets if t['target_id']==s['target_id'])['outcome']))
            if not s.get('accepted_effect_types') and s.get('eligible_data_families')==['independent_continuous_arms']:
                s['accepted_effect_types']=['MD']
            for key in ('outcome_axis','accepted_measurement_semantics','unit','intervention_class','comparator_class','rsid','study_design','phenotype','genotype_role'):
                if a.get(key):s['selection_rules'].append(f'Metadata constraint: {key} must match: {text(a[key])}')
            if s.get('timepoint_policy') not in (None,'longest_eligible','all_eligible','source_defined'):
                s['selection_rules'].append('Source timepoint rule: '+text(s['timepoint_policy']))
                s['timepoint_policy']='source_defined'
            s['minimum_independent_studies']=(a.get('pooling') or {}).get('minimum_independent_studies',2)
            specs.append(s)
        profile={'profile_version':'2.0.0','primary_analysis_spec':specs[0],'secondary_outcomes':specs[1:],'candidate_subgroup_dimensions':[],'candidate_sensitivity_dimensions':[]}
    paths=read(case/'inputs/extraction_document_paths.json')['documents']
    if isinstance(paths,list):paths={x.get('article_id') or x['document_id']:x for x in paths}
    manifest=read(case/'inputs/extraction_document_manifest.json')['documents']
    if isinstance(manifest,dict):manifest=list(manifest.values())
    docs=[];warnings=[]
    for d in manifest:
        if d.get('role')!='candidate_article':continue
        aid=d.get('article_id') or d['document_id'];entry=paths.get(aid,{})
        representations=entry.get('paths',entry);resolved={}
        for kind in ('markdown','content_list','pdf'):
            if not representations.get(kind):continue
            path=(case/representations[kind]).resolve()
            require(path.is_relative_to(case.resolve()),'case path escapes package')
            if path.is_file():resolved[kind]=str(path)
            else:warnings.append({'article_id':aid,'missing_original_representation':kind})
        if case_id in EARLY_BASELINE_CASES or not resolved.get('markdown'):
            for folder in ('prepared_documents/local_pdf_text','prepared_documents/mineru'):
                p=case/folder/aid/'full.md'
                if p.is_file():resolved['markdown']=str(p.resolve());break
        docs.append({'article_id':aid,'document_id':d['document_id'],'study_id':d.get('study_id'),'title':d.get('citation') or aid,'paths':resolved})
    r={'schema_version':VERSION,'case_id':case_id,'topic':source.get('review_question') or source['protocol_id'],'protocol':protocol,'analysis_profile':profile,'documents':docs,'options':{'extraction_runtime':'typed','provider':'official_openai','model':'gpt-5.6-luna','route_rob':True}}
    r=normalize_request(r)
    output=Path(output).resolve()
    # Portable to the integration package location; all document paths are relative to this request.
    import os
    for doc in r['documents']:
        doc['paths']={k:os.path.relpath(v,output.parent) for k,v in doc['paths'].items()}
    write(output,r)
    audit={'case_id':case_id,'document_count':len(docs),'gold_accessed':False,'screening_executed':False,'notes':['Inputs are the bundle-declared extraction document set; they are not a new retrieval universe or a screening benchmark.','The first source-declared analysis becomes primary; remaining declared analyses become secondary. Source statistical method recipes are omitted for Analysis internal method selection.'],'warnings':warnings}
    write(output.with_suffix('.import_audit.json'),audit)
    return audit
